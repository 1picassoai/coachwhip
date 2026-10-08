//! Qwen3 mixture-of-experts, read from a GGUF file. Everything except the experts is loaded onto
//! the GPU at start-up; the experts stay in the file and stream through the ExpertStore.
//!
//! Derived from Candle's `quantized_qwen3_moe.rs` (Apache-2.0 / MIT). Changes: the experts are
//! streamed; the KV cache is one preallocated buffer per layer; grouped-query attention runs
//! without copying K and V per head; the token embedding table lives on the CPU.

use crate::experts::{ExpertStore, Settings, StreamedMoe};
use candle::quantized::gguf_file;
use candle::{DType, Device, Module, Result, Tensor};
use candle_nn::kv_cache::KvCache;
use candle_nn::{Embedding, Linear};
use candle_transformers::models::quantized_qwen3::{Gguf, RotaryEmbedding};
use candle_transformers::models::with_tracing::QMatMul;
use candle_transformers::quantized_nn::RmsNorm;
use std::path::Path;
use std::sync::Arc;

struct Attention {
    wq: QMatMul,
    wk: QMatMul,
    wv: QMatMul,
    wo: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    dtype: DType,
    kv_cache: KvCache,
}

impl Attention {
    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, input_pos: usize) -> Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;
        let in_dtype = x.dtype();
        let heads = |t: Tensor, n: usize| -> Result<Tensor> { t.reshape((1, seq_len, n, self.head_dim))?.transpose(1, 2)?.contiguous() };
        let q = heads(self.wq.forward(x)?, self.n_head)?;
        let k = heads(self.wk.forward(x)?, self.n_kv_head)?;
        let v = heads(self.wv.forward(x)?, self.n_kv_head)?;

        // Per-head RMSNorm, as Qwen3 does it.
        let q = self.q_norm.forward(&q.flatten(0, 2)?)?.reshape((1, self.n_head, seq_len, self.head_dim))?;
        let k = self.k_norm.forward(&k.flatten(0, 2)?)?.reshape((1, self.n_kv_head, seq_len, self.head_dim))?;

        let (q, k, v) = (q.to_dtype(self.dtype)?, k.to_dtype(self.dtype)?, v.to_dtype(self.dtype)?);
        let (q, k) = self.rotary_emb.apply(&q, &k, input_pos)?;

        // One preallocated buffer per layer: append writes the new rows in place and hands back
        // views, so nothing is copied per word.
        let (k, v) = self.kv_cache.append(&k, &v)?;
        let s_len = k.dim(2)?;

        // Grouped-query attention without repeating K and V: the G query heads that share one kv
        // head are stacked as G*L rows of one batched matmul.
        let g = self.n_head / self.n_kv_head;
        let q = q.contiguous()?.reshape((1, self.n_kv_head, g * seq_len, self.head_dim))?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?; // (1, KVH, G*L, S)
        if let Some(m) = mask {
            let mask = m.to_dtype(scores.dtype())?.unsqueeze(1)?; // (1, 1, 1, L, S)
            scores = scores
                .reshape((1, self.n_kv_head, g, seq_len, s_len))?
                .broadcast_add(&mask)?
                .reshape((1, self.n_kv_head, g * seq_len, s_len))?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?.reshape((1, self.n_head, seq_len, self.head_dim))?;
        let ctx = ctx.transpose(1, 2)?.reshape((b, seq_len, self.n_head * self.head_dim))?;
        self.wo.forward(&ctx.to_dtype(in_dtype)?)
    }
}

struct Layer {
    attn: Attention,
    attn_norm: RmsNorm,
    moe: StreamedMoe,
    ffn_norm: RmsNorm,
}

/// What the chat template puts right after `<|im_start|>assistant\n`. A thinking model's template
/// opens the turn with an empty think block when thinking is off, so the model answers straight
/// away; Qwen3.5 and later open a think block for it when thinking is on. Both empty for a model
/// with no thinking mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ThinkOpen {
    pub off: &'static str,
    pub on: &'static str,
}

impl ThinkOpen {
    pub fn from_template(template: Option<&str>) -> Self {
        let t = template.unwrap_or("");
        if !t.contains("enable_thinking") {
            return Self { off: "", on: "" };
        }
        Self {
            off: if t.contains(r"'<think>\n\n</think>\n\n'") { "<think>\n\n</think>\n\n" } else { "" },
            on: if t.contains(r"{{- '<think>\n' }}") { "<think>\n" } else { "" },
        }
    }

    pub fn thinks(&self) -> bool {
        !self.off.is_empty()
    }
}

pub struct Qwen3Moe {
    embeddings: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    output: QMatMul,
    store: Arc<ExpertStore>,
    dtype: DType,
    device: Device,
    pub think: ThinkOpen,
    pub progress: Option<Progress>,
}

/// Told what the model is doing while the page would otherwise be silent.
pub type Progress = Box<dyn FnMut(Step)>;

pub enum Step {
    /// (layers done, layers in all) while a prompt is being read.
    Layer(usize, usize),
    /// Words written so far of a summary that makes room in a long chat.
    Summary(usize),
}

impl Qwen3Moe {
    pub fn load(path: &Path, device: &Device, dtype: DType, settings: &Settings) -> Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let ct = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(path))?;
        let md = |s: &str| match ct.metadata.get(s) {
            None => candle::bail!("cannot find {s} in the GGUF metadata"),
            Some(v) => Ok(v.clone()),
        };
        let arch = md("general.architecture")?.to_string()?.clone();
        let md_u = |s: &str| -> Result<usize> { Ok(md(&format!("{arch}.{s}"))?.to_u32()? as usize) };
        let head_count = md_u("attention.head_count")?;
        let head_count_kv = md_u("attention.head_count_kv")?;
        let embedding_length = md_u("embedding_length")?;
        let head_dim = md_u("attention.key_length").unwrap_or(embedding_length / head_count);
        let context_length = md_u("context_length")?;
        let block_count = md_u("block_count")?;
        let top_k = md_u("expert_used_count")?;
        let rms_norm_eps = md(&format!("{arch}.attention.layer_norm_rms_epsilon"))?.to_f32()? as f64;
        let rope_freq_base = md(&format!("{arch}.rope.freq_base")).and_then(|m| m.to_f32()).unwrap_or(10000f32);
        let shared = md_u("expert_shared_feed_forward_length").unwrap_or(0);
        let norm_topk_prob = shared == 0;
        let template = md("tokenizer.chat_template").ok().and_then(|t| t.to_string().ok().cloned());
        let think = ThinkOpen::from_template(template.as_deref());

        let store = ExpertStore::new(&ct, device, block_count, path, settings)?;
        let mut gg = Gguf::new(ct, &mut file, device.clone());

        let embeddings = gg.tensor("token_embd.weight")?.dequantize(device)?.to_dtype(DType::F16)?.to_device(&Device::Cpu)?;
        let norm = gg.rms_norm("output_norm.weight", rms_norm_eps)?;
        let output = match gg.qmatmul("output.weight") {
            Ok(v) => v,
            _ => gg.qmatmul("token_embd.weight")?,
        };
        let rotary_emb = Arc::new(RotaryEmbedding::new(dtype, head_dim, context_length, rope_freq_base as f64, device)?);

        let mut layers = Vec::with_capacity(block_count);
        for layer in 0..block_count {
            let p = format!("blk.{layer}");
            let gate = gg.tensor(&format!("{p}.ffn_gate_inp.weight"))?.dequantize(device)?.to_dtype(DType::F32)?;
            let moe = StreamedMoe {
                layer,
                gate: Linear::new(gate, None),
                next_gate: None,
                gate1: None,
                ahead: settings.ahead.max(1),
                store: store.clone(),
                top_k,
                norm_topk_prob,
                prefetch_k: settings.prefetch,
            };
            let attn = Attention {
                wq: gg.qmatmul(&format!("{p}.attn_q.weight"))?,
                wk: gg.qmatmul(&format!("{p}.attn_k.weight"))?,
                wv: gg.qmatmul(&format!("{p}.attn_v.weight"))?,
                wo: gg.qmatmul(&format!("{p}.attn_output.weight"))?,
                q_norm: gg.rms_norm(&format!("{p}.attn_q_norm.weight"), rms_norm_eps)?,
                k_norm: gg.rms_norm(&format!("{p}.attn_k_norm.weight"), rms_norm_eps)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim,
                rotary_emb: rotary_emb.clone(),
                dtype,
                kv_cache: KvCache::new(2, 4096),
            };
            layers.push(Layer {
                attn,
                attn_norm: gg.rms_norm(&format!("{p}.attn_norm.weight"), rms_norm_eps)?,
                moe,
                ffn_norm: gg.rms_norm(&format!("{p}.ffn_norm.weight"), rms_norm_eps)?,
            });
        }
        // Each layer gets the router of the layer `ahead` on, to guess its experts early; the last
        // `ahead` layers have no later router and use the learned route table.
        let ahead = settings.ahead.max(1);
        for i in ahead..layers.len() {
            layers[i - ahead].moe.next_gate = Some(layers[i].moe.gate.clone());
        }
        Ok(Self { embeddings: Embedding::new(embeddings, embedding_length), layers, norm, output, store, dtype, device: device.clone(), think, progress: None })
    }

    /// Print where each written word's time went since the last call, then reset the counters.
    pub fn report(&self) {
        self.store.report_and_reset();
    }

    /// Forget the conversation so the next prompt starts at position 0.
    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.attn.kv_cache.reset();
        }
    }

    pub fn forward(&mut self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let mut xs = self.embeddings.forward(&x.to_device(&Device::Cpu)?)?.to_device(&self.device)?.to_dtype(DType::F32)?;
        let (_b, l) = x.dims2()?;
        let mask = if l == 1 {
            None
        } else {
            Some(candle_transformers::utils::build_additive_causal_mask(l, offset, None, &self.device, self.dtype)?)
        };
        let n = self.layers.len();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let residual = &xs;
            let h = (layer.attn.forward(&layer.attn_norm.forward(residual)?, mask.as_ref(), offset)? + residual)?;
            let out = (layer.moe.forward(&layer.ffn_norm.forward(&h)?)? + &h)?;
            xs = out;
            if l > 1 {
                if let Some(p) = &mut self.progress {
                    p(Step::Layer(i + 1, n));
                }
            }
        }
        let xs = self.norm.forward(&xs.narrow(1, l - 1, 1)?)?;
        self.output.forward(&xs)?.to_dtype(DType::F32)?.squeeze(1)
    }
}

/// Every model family Coachwhip runs, picked from the GGUF file's architecture.
pub enum Model {
    Qwen3Moe(Qwen3Moe),
    Qwen3Next(crate::model_next::Qwen3Next),
}

impl Model {
    pub fn load(path: &Path, device: &Device, settings: &Settings) -> Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let ct = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(path))?;
        let arch = match ct.metadata.get("general.architecture") {
            Some(v) => v.to_string()?.clone(),
            None => candle::bail!("cannot find general.architecture in the GGUF metadata"),
        };
        drop(ct);
        match arch.as_str() {
            // f16 for attention: bf16 garbles long answers in this model.
            "qwen3moe" => Ok(Self::Qwen3Moe(Qwen3Moe::load(path, device, DType::F16, settings)?)),
            "qwen3next" | "qwen35moe" => Ok(Self::Qwen3Next(crate::model_next::Qwen3Next::load(path, device, settings, arch.as_str())?)),
            other => candle::bail!(
                "this model's architecture is \"{other}\", which Coachwhip does not support yet (today: qwen3moe, qwen3next). \
                 Open an issue at https://github.com/1picassoai/coachwhip/issues so it moves up the list."
            ),
        }
    }

    pub fn forward(&mut self, x: &Tensor, offset: usize) -> Result<Tensor> {
        match self {
            Self::Qwen3Moe(m) => m.forward(x, offset),
            Self::Qwen3Next(m) => m.forward(x, offset),
        }
    }

    pub fn clear_kv_cache(&mut self) {
        match self {
            Self::Qwen3Moe(m) => m.clear_kv_cache(),
            Self::Qwen3Next(m) => m.clear_kv_cache(),
        }
    }

    pub fn report(&self) {
        match self {
            Self::Qwen3Moe(m) => m.report(),
            Self::Qwen3Next(m) => m.report(),
        }
    }

    pub fn thinks(&self) -> bool {
        self.think().thinks()
    }

    /// S6 measurement (Qwen3-Next family only): see `Qwen3Next::speculate_measure`.
    pub fn speculate_measure(&mut self, last: u32, offset: usize, k: usize) -> Result<Option<(Vec<u32>, usize)>> {
        match self {
            Self::Qwen3Moe(_) => Ok(None),
            Self::Qwen3Next(m) => m.speculate_measure(last, offset, k).map(Some),
        }
    }

    /// Foresight (Qwen3-Next family only): see `Qwen3Next::foresee`.
    pub fn foresee(&mut self, last: u32, offset: usize, steps: usize) -> Result<bool> {
        match self {
            Self::Qwen3Moe(_) => Ok(false),
            Self::Qwen3Next(m) => m.foresee(last, offset, steps),
        }
    }

    /// S6 (Qwen3-Next family only): see `Qwen3Next::speculate`.
    pub fn speculate(&mut self, last: u32, offset: usize, k: usize) -> Result<Option<Vec<u32>>> {
        match self {
            Self::Qwen3Moe(_) => Ok(None),
            Self::Qwen3Next(m) => m.speculate(last, offset, k),
        }
    }

    /// The text that opens the assistant's turn, with thinking on or off.
    pub fn assistant_open(&self, think: bool) -> &'static str {
        let t = self.think();
        if think { t.on } else { t.off }
    }

    fn think(&self) -> ThinkOpen {
        match self {
            Self::Qwen3Moe(m) => m.think,
            Self::Qwen3Next(m) => m.think,
        }
    }

    /// Who to tell, layer by layer, while a prompt is being read; `None` to stop telling.
    pub fn set_progress(&mut self, progress: Option<Progress>) {
        match self {
            Self::Qwen3Moe(m) => m.progress = progress,
            Self::Qwen3Next(m) => m.progress = progress,
        }
    }
}
