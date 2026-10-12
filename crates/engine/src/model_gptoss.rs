//! gpt-oss (OpenAI's mixture of experts). The family owns attention (sliding windows and sinks),
//! the biases and YaRN; the experts stream through the core like every other family, with the
//! family's clamped activation plugged into the core's hook. Experts are MXFP4.
use crate::experts::{ExpertAct, ExpertStore, Settings, StreamedMoe};
use crate::model::{Progress, ThinkOpen};
use candle::quantized::{ggml_file::qtensor_from_ggml, gguf_file, GgmlDType};
use candle::{DType, Device, Module, Result, Tensor};
use candle_nn::Linear;
use candle_transformers::models::quantized_qwen3::Gguf;
use candle_transformers::models::with_tracing::QMatMul;
use candle_transformers::quantized_nn::RmsNorm;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::Arc;

const SWIGLU_LIMIT: f32 = 7.0;
const SWIGLU_ALPHA: f64 = 1.702;

struct Layer {
    attn_norm: RmsNorm,
    post_norm: RmsNorm,
    wq: QMatMul,
    bq: Tensor,
    wk: QMatMul,
    bk: Tensor,
    wv: QMatMul,
    bv: Tensor,
    wo: QMatMul,
    bo: Tensor,
    sinks: Tensor,
    moe: StreamedMoe,
    sliding: bool,
    k_cache: Option<Tensor>,
    v_cache: Option<Tensor>,
    /// The position of the first key the cache still holds (a sliding layer drops old ones).
    kv_start: usize,
}

/// The word table, packed as the file stores it (Q8_0), on the CPU; rows are unpacked when asked for.
struct Table {
    bytes: Vec<u8>,
    dtype: GgmlDType,
    row_bytes: usize,
    hidden: usize,
}

impl Table {
    fn rows(&self, ids: &[u32]) -> Result<Tensor> {
        let mut buf = Vec::with_capacity(ids.len() * self.row_bytes);
        for &id in ids {
            let at = id as usize * self.row_bytes;
            buf.extend_from_slice(&self.bytes[at..at + self.row_bytes]);
        }
        qtensor_from_ggml(self.dtype, &buf, vec![ids.len(), self.hidden], &Device::Cpu)?.dequantize(&Device::Cpu)
    }
}

pub struct GptOss {
    table: Table,
    layers: Vec<Layer>,
    norm: RmsNorm,
    output: QMatMul,
    store: Arc<ExpertStore>,
    device: Device,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    window: usize,
    /// The experts' input width in the file (wider than the model when squeezed and padded).
    xpad: usize,
    cos: Tensor,
    sin: Tensor,
    pub think: ThinkOpen,
    pub progress: Option<Progress>,
}

/// YaRN rotary frequencies, as the reference computes them (truncate = false), with the
/// attention factor folded into cos and sin.
fn yarn_tables(dim: usize, base: f64, factor: f64, orig: f64, beta_fast: f64, beta_slow: f64, max_pos: usize, device: &Device) -> Result<(Tensor, Tensor)> {
    let half = dim / 2;
    let corr = |rot: f64| (dim as f64 * (orig / (rot * 2.0 * std::f64::consts::PI)).ln()) / (2.0 * base.ln());
    let low = corr(beta_fast).max(0.0);
    let mut high = corr(beta_slow).min(dim as f64 - 1.0);
    if (high - low).abs() < 1e-9 {
        high += 0.001;
    }
    let mscale = 0.1 * factor.ln() + 1.0;
    let mut inv = Vec::with_capacity(half);
    for i in 0..half {
        let pos_freq = base.powf((2 * i) as f64 / dim as f64);
        let extrap = 1.0 / pos_freq;
        let interp = 1.0 / (factor * pos_freq);
        let ramp = ((i as f64 - low) / (high - low)).clamp(0.0, 1.0);
        let extrap_factor = 1.0 - ramp;
        inv.push(interp * (1.0 - extrap_factor) + extrap * extrap_factor);
    }
    let mut c = Vec::with_capacity(max_pos * half);
    let mut s = Vec::with_capacity(max_pos * half);
    for p in 0..max_pos {
        for f in &inv {
            let a = p as f64 * f;
            c.push((a.cos() * mscale) as f32);
            s.push((a.sin() * mscale) as f32);
        }
    }
    Ok((Tensor::from_vec(c, (max_pos, half), device)?, Tensor::from_vec(s, (max_pos, half), device)?))
}


impl GptOss {
    pub fn load(path: &Path, device: &Device, settings: &Settings) -> Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let ct = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(path))?;
        let md_u = |k: &str| -> Result<usize> { ct.metadata.get(k).ok_or_else(|| candle::Error::Msg(format!("missing {k}")))?.to_u32().map(|v| v as usize) };
        let md_f = |k: &str| -> Result<f64> { ct.metadata.get(k).ok_or_else(|| candle::Error::Msg(format!("missing {k}")))?.to_f32().map(|v| v as f64) };
        let block_count = md_u("gpt-oss.block_count")?;
        let heads = md_u("gpt-oss.attention.head_count")?;
        let kv_heads = md_u("gpt-oss.attention.head_count_kv")?;
        let head_dim = md_u("gpt-oss.attention.key_length")?;
        let window = md_u("gpt-oss.attention.sliding_window")?;
        // Fast mode (`--experts n`, `COACHWHIP_TOPK=n`): fewer experts per word than the model was trained with.
        let trained_k = md_u("gpt-oss.expert_used_count")?;
        let top_k = match std::env::var("COACHWHIP_TOPK").ok().and_then(|v| v.parse::<usize>().ok()) {
            Some(k) if k >= 1 && k < trained_k => k,
            _ => trained_k,
        };
        let eps = md_f("gpt-oss.attention.layer_norm_rms_epsilon")?;
        let (cos, sin) = yarn_tables(
            head_dim,
            md_f("gpt-oss.rope.freq_base")?,
            md_f("gpt-oss.rope.scaling.factor")?,
            md_u("gpt-oss.rope.scaling.original_context_length")? as f64,
            md_f("gpt-oss.rope.scaling.yarn_beta_fast")?,
            md_f("gpt-oss.rope.scaling.yarn_beta_slow")?,
            16384,
            device,
        )?;
        let template = ct.metadata.get("tokenizer.chat_template").and_then(|t| t.to_string().ok().cloned());
        let think = ThinkOpen::from_template(template.as_deref());

        // A squeezed file pads expert rows to whole 256-weight blocks; the input to the experts
        // (and to the router, which shares it) gets the same zeros.
        let cols = |m: &str| ct.tensor_infos.get(&format!("blk.0.ffn_{m}_exps.weight")).map(|i| i.shape.dims()[2]).unwrap_or(0);
        let (xpad, hpad) = (cols("gate"), cols("down"));
        let store = ExpertStore::new(&ct, device, block_count, path, settings)?;
        let table = {
            let q = ct.tensor(&mut file, "token_embd.weight", &Device::Cpu)?;
            let hidden = *q.shape().dims().last().unwrap();
            let dtype = q.dtype();
            let row_bytes = hidden / dtype.block_size() * dtype.type_size();
            Table { bytes: q.data()?.into_owned(), dtype, row_bytes, hidden }
        };
        let mut gg = Gguf::new(ct, &mut file, device.clone());
        let dense = |gg: &mut Gguf<&mut std::fs::File>, name: &str| -> Result<Tensor> { gg.tensor(name)?.dequantize(device)?.to_dtype(DType::F32) };
        let mut layers = Vec::with_capacity(block_count);
        for i in 0..block_count {
            let p = format!("blk.{i}");
            let moe = StreamedMoe {
                act: ExpertAct::ClampedSwiglu {
                    alpha: SWIGLU_ALPHA,
                    limit: SWIGLU_LIMIT,
                    gate_b: dense(&mut gg, &format!("{p}.ffn_gate_exps.bias"))?,
                    up_b: dense(&mut gg, &format!("{p}.ffn_up_exps.bias"))?,
                    down_b: dense(&mut gg, &format!("{p}.ffn_down_exps.bias"))?,
                    pad: hpad,
                },
                layer: i,
                gate: Linear::new(
                    {
                        let w = dense(&mut gg, &format!("{p}.ffn_gate_inp.weight"))?;
                        let width = w.dim(1)?;
                        if xpad > width { w.pad_with_zeros(1, 0, xpad - width)? } else { w }
                    },
                    Some(dense(&mut gg, &format!("{p}.ffn_gate_inp.bias"))?),
                ),
                next_gate: None,
                gate1: None,
                ahead: settings.ahead.max(1),
                store: store.clone(),
                top_k,
                norm_topk_prob: true,
                prefetch_k: settings.prefetch,
            };
            layers.push(Layer {
                attn_norm: gg.rms_norm(&format!("{p}.attn_norm.weight"), eps)?,
                post_norm: gg.rms_norm(&format!("{p}.post_attention_norm.weight"), eps)?,
                wq: gg.qmatmul(&format!("{p}.attn_q.weight"))?,
                bq: dense(&mut gg, &format!("{p}.attn_q.bias"))?,
                wk: gg.qmatmul(&format!("{p}.attn_k.weight"))?,
                bk: dense(&mut gg, &format!("{p}.attn_k.bias"))?,
                wv: gg.qmatmul(&format!("{p}.attn_v.weight"))?,
                bv: dense(&mut gg, &format!("{p}.attn_v.bias"))?,
                wo: gg.qmatmul(&format!("{p}.attn_output.weight"))?,
                bo: dense(&mut gg, &format!("{p}.attn_output.bias"))?,
                sinks: dense(&mut gg, &format!("{p}.attn_sinks.weight"))?,
                moe,
                // The reference's layer_types start with sliding_attention and alternate.
                sliding: i % 2 == 0,
                k_cache: None,
                v_cache: None,
                kv_start: 0,
            });
        }
        // The guess for a later layer comes from that layer's own router, run early.
        let ahead = settings.ahead.max(1);
        for i in ahead..layers.len() {
            layers[i - ahead].moe.next_gate = Some(layers[i].moe.gate.clone());
            if ahead >= 2 {
                layers[i - ahead].moe.gate1 = Some(layers[i - ahead + 1].moe.gate.clone());
            }
        }
        let norm = gg.rms_norm("output_norm.weight", eps)?;
        let output = gg.qmatmul("output.weight")?;
        eprintln!("coachwhip: gpt-oss, {block_count} layers, {heads} heads over {kv_heads}, sliding window {window}, {top_k} experts per word");
        let m = Self { table, layers, norm, output, store, device: device.clone(), heads, kv_heads, head_dim, window, xpad, cos, sin, think, progress: None };
        if std::env::var("COACHWHIP_GPTOSS_CHECK").is_ok() {
            check_experts(path, 0, device)?;
        }
        Ok(m)
    }

    pub fn forward(&mut self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let ids: Vec<u32> = x.flatten_all()?.to_vec1()?;
        if !self.store.drafting() {
            self.store.note_tokens(ids.clone());
        }
        let n = ids.len();
        let mut xs = self.table.rows(&ids)?.to_dtype(DType::F32)?.to_device(&self.device)?;
        let n_layers = self.layers.len();
        let (heads, kv_heads, head_dim, window) = (self.heads, self.kv_heads, self.head_dim, self.window);
        let group = heads / kv_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();
        let cos = self.cos.narrow(0, offset, n)?;
        let sin = self.sin.narrow(0, offset, n)?;
        // Lab timing switches (text is junk with either on): skip the experts, or skip attention.
        let no_moe = std::env::var("COACHWHIP_GPTOSS_NOMOE").is_ok();
        let no_attn = std::env::var("COACHWHIP_GPTOSS_NOATTN").is_ok();
        for l in 0..n_layers {
            let layer = &mut self.layers[l];
            if !no_attn {
            // Attention with sinks; sliding layers see the last `window` tokens only.
            let h = layer.attn_norm.forward(&xs)?;
            // Heads first, (1, heads, n, d), so the rotation is candle's one fused kernel and the
            // cache grows along its own axis without a transpose per word.
            let q = layer.wq.forward(&h)?.broadcast_add(&layer.bq)?.reshape((1, n, heads, head_dim))?.transpose(1, 2)?.contiguous()?;
            let k = layer.wk.forward(&h)?.broadcast_add(&layer.bk)?.reshape((1, n, kv_heads, head_dim))?.transpose(1, 2)?.contiguous()?;
            let v = layer.wv.forward(&h)?.broadcast_add(&layer.bv)?.reshape((1, n, kv_heads, head_dim))?.transpose(1, 2)?.contiguous()?;
            let q = candle_nn::rotary_emb::rope(&q, &cos, &sin)?;
            let k = candle_nn::rotary_emb::rope(&k, &cos, &sin)?;
            // The cache holds keys and values in f16, as GGML does: half the memory, which the
            // Wrangler hands to the bank. They are widened back to f32 for the maths.
            let (k, v) = (k.to_dtype(DType::F16)?, v.to_dtype(DType::F16)?);
            let k = match &layer.k_cache { Some(c) => Tensor::cat(&[c, &k], 2)?, None => k };
            let v = match &layer.v_cache { Some(c) => Tensor::cat(&[c, &v], 2)?, None => v };
            // A sliding layer never looks further back than its window, so it keeps no more: one new
            // word needs exactly the last `window` keys (itself included), and then no mask at all.
            let keep = if n == 1 { window } else { window + n };
            let (k, v, first) = if layer.sliding && k.dim(2)? > keep {
                let drop = k.dim(2)? - keep;
                (k.narrow(2, drop, keep)?.contiguous()?, v.narrow(2, drop, keep)?.contiguous()?, layer.kv_start + drop)
            } else {
                (k, v, layer.kv_start)
            };
            layer.kv_start = first;
            layer.k_cache = Some(k.clone());
            layer.v_cache = Some(v.clone());
            let t = k.dim(2)?;
            // Grouped heads share their keys: q as (kv_heads, group * rows, d) against k as (kv_heads, t, d),
            // so nothing is copied out to all 64 heads.
            let kt = k.squeeze(0)?.to_dtype(DType::F32)?.t()?.contiguous()?;
            let vv = v.squeeze(0)?.to_dtype(DType::F32)?;
            // A long prompt is attended a block of rows at a time: its score table (heads x rows x
            // keys) then stays a few hundred MB instead of several GB on a 16 GB Mac.
            const ROWS: usize = 64;
            let mut parts = Vec::with_capacity(n.div_ceil(ROWS));
            let mut i0 = 0;
            while i0 < n {
                let rows = ROWS.min(n - i0);
                let qg = q.narrow(2, i0, rows)?.contiguous()?.reshape((kv_heads, group * rows, head_dim))?;
                let scores = (qg.matmul(&kt)? * scale)?.reshape((heads, rows, t))?;
                let scores = if n == 1 {
                    scores
                } else {
                    let mut mask = vec![0f32; rows * t];
                    for i in 0..rows {
                        let pos = offset + i0 + i;
                        for j in 0..t {
                            let at = first + j;
                            if at > pos || (layer.sliding && at + window <= pos) {
                                mask[i * t + j] = f32::NEG_INFINITY;
                            }
                        }
                    }
                    scores.broadcast_add(&Tensor::from_vec(mask, (1, rows, t), &self.device)?)?
                };
                let sinks = layer.sinks.reshape((heads, 1, 1))?.broadcast_as((heads, rows, 1))?.contiguous()?;
                let all = Tensor::cat(&[&scores, &sinks], 2)?;
                let probs = candle_nn::ops::softmax_last_dim(&all)?.narrow(2, 0, t)?.contiguous()?.reshape((kv_heads, group * rows, t))?;
                parts.push(probs.matmul(&vv)?.reshape((heads, rows, head_dim))?.transpose(0, 1)?.reshape((rows, heads * head_dim))?);
                i0 += rows;
            }
            let ctx = if parts.len() == 1 { parts.pop().unwrap() } else { Tensor::cat(&parts, 0)? };
            let o = layer.wo.forward(&ctx)?.broadcast_add(&layer.bo)?;
            xs = (xs + o)?;
            }
            if no_moe {
                continue;
            }
            // The experts, through the core.
            let post = layer.post_norm.forward(&xs)?;
            let width = post.dim(1)?;
            let post = if self.xpad > width { post.pad_with_zeros(1, 0, self.xpad - width)? } else { post };
            let post = post.unsqueeze(0)?;
            let routed = layer.moe.forward(&post)?.squeeze(0)?;
            xs = (xs + routed)?;
            if n > 1 {
                if let Some(p) = &mut self.progress {
                    p(crate::model::Step::Layer(l + 1, n_layers));
                }
            }
        }
        let last = self.norm.forward(&xs.narrow(0, n - 1, 1)?)?;
        self.output.forward(&last)?.to_dtype(DType::F32)
    }

    pub fn clear_kv_cache(&mut self) {
        for l in &mut self.layers {
            l.k_cache = None;
            l.v_cache = None;
            l.kv_start = 0;
        }
    }

    pub fn report(&self) {
        self.store.report_and_reset();
    }
}

/// Lab, step 4 gate: every expert matrix of one layer, multiplied on `gpu`, against the same
/// matrix decoded to f32 and multiplied exactly on the CPU. Prints the worst relative error.
pub fn check_experts(path: &Path, layer: usize, gpu: &Device) -> Result<()> {
    let mut file = std::fs::File::open(path)?;
    let ct = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(path))?;
    let mut worst = (0f32, 0usize, "gate");
    let mut count = 0;
    for name in ["gate", "up", "down"] {
        let info = &ct.tensor_infos[&format!("blk.{layer}.ffn_{name}_exps.weight")];
        let dims = info.shape.dims().to_vec();
        let (n, rows, cols) = (dims[0], dims[1], dims[2]);
        let bytes = rows * cols / info.ggml_dtype.block_size() * info.ggml_dtype.type_size();
        let x = Tensor::randn(0f32, 1f32, (3, cols), &Device::Cpu)?;
        count = n;
        for e in 0..n {
            let mut buf = vec![0u8; bytes];
            file.read_exact_at(&mut buf, ct.tensor_data_offset + info.offset + (e * bytes) as u64)?;
            let w = qtensor_from_ggml(info.ggml_dtype, &buf, vec![rows, cols], &Device::Cpu)?.dequantize(&Device::Cpu)?;
            let exact = x.matmul(&w.t()?)?;
            let q = QMatMul::from_weights(Arc::new(qtensor_from_ggml(info.ggml_dtype, &buf, vec![rows, cols], gpu)?))?;
            let got = q.forward(&x.to_device(gpu)?)?.to_device(&Device::Cpu)?;
            let err = (got - &exact)?.abs()?.max_all()?.to_scalar::<f32>()? / exact.abs()?.max_all()?.to_scalar::<f32>()?;
            if err > worst.0 {
                worst = (err, e, name);
            }
        }
    }
    eprintln!("coachwhip: expert check, layer {layer}, {count} experts x 3 matrices on {gpu:?}: worst relative error {:.2e} (expert {} {})", worst.0, worst.1, worst.2);
    Ok(())
}
