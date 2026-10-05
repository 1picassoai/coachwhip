//! Qwen3-Next, read from a GGUF file. The same expert streaming as Qwen3 MoE; what is new is the
//! attention. Three layers in four keep a fixed-size running memory per head (Gated DeltaNet)
//! instead of a cache that grows with every word; every fourth layer is classic attention with a
//! gate on its output. Each layer also has a shared expert that every word uses.
//!
//! Follows llama.cpp's qwen3next graph for the tensor layout and the maths.

use crate::experts::{ExpertStore, Settings, StreamedMoe};
use candle::quantized::gguf_file;
use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::kv_cache::KvCache;
use candle_nn::{Embedding, Linear};
use candle_transformers::models::quantized_qwen3::Gguf;
use candle_transformers::models::with_tracing::QMatMul;
use candle_transformers::quantized_nn::RmsNorm;
use std::path::Path;
use std::sync::Arc;

/// x / sqrt(sum(x^2) + eps) over the last dimension.
fn l2_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let n = (x.sqr()?.sum_keepdim(D::Minus1)? + eps)?.sqrt()?;
    x.broadcast_div(&n)
}

/// Give each layer the router of the layer `ahead` on, so it can guess that layer's experts early.
/// The last `ahead` layers have no later router and fall back to the learned route table.
fn wire_lookahead(layers: &mut [Layer], ahead: usize) {
    let ahead = ahead.max(1);
    for i in ahead..layers.len() {
        layers[i - ahead].moe.next_gate = Some(layers[i].moe.gate.clone());
    }
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    (x.exp()? + 1.0)?.log()
}

/// Words per block when a running-memory layer reads a prompt.
pub const DELTA_CHUNK: usize = 64;

/// The delta rule over a whole prompt, a block of words at a time (Gated DeltaNet, Yang, Kautz and
/// Hatamizadeh, ICLR 2025, section 3). Gives the same answer as stepping word by word, but each
/// block is a few large matrix products instead of a dozen small GPU jobs per word. Inside a block,
/// every word's correction depends on the corrections before it; that chain is a small triangular
/// system, solved once per block on the CPU. Decays are kept as logarithms so long blocks never
/// divide by a vanishing number.
/// q, k: (l, n_v, dk); v: (l, n_v, dv); beta, log_decay: (l, n_v); s: (n_v, dk, dv).
pub fn delta_chunked(q: &Tensor, k: &Tensor, v: &Tensor, beta: &Tensor, log_decay: &Tensor, mut s: Tensor) -> Result<(Tensor, Tensor)> {
    let (l, n_v, dk) = q.dims3()?;
    let dv = v.dim(2)?;
    let c = DELTA_CHUNK;
    let nc = l.div_ceil(c);
    let pad = nc * c - l;
    let device = q.device();
    // Padding words have no key, no value, no write strength and no decay, so they change nothing.
    let blocks = |t: &Tensor, width: usize| -> Result<Tensor> {
        let t = if pad > 0 { t.pad_with_zeros(0, 0, pad)? } else { t.clone() };
        t.reshape((nc, c, n_v, width))?.permute((2, 0, 1, 3))?.contiguous()?.reshape((n_v * nc, c, width))
    };
    let (q, k, v) = (blocks(q, dk)?, blocks(k, dk)?, blocks(v, dv)?);
    let beta = blocks(&beta.unsqueeze(2)?, 1)?; // (n, c, 1)
    let cum = blocks(&log_decay.unsqueeze(2)?, 1)?.squeeze(2)?.cumsum(1)?; // (n, c): decay since block start, inclusive

    let mask = |strict: bool| -> Result<Tensor> {
        let m: Vec<f32> = (0..c * c).map(|x| if x % c < x / c || (!strict && x % c == x / c) { 0.0 } else { -1e30 }).collect();
        Tensor::from_vec(m, (c, c), device)
    };
    let diff = cum.unsqueeze(2)?.broadcast_sub(&cum.unsqueeze(1)?)?; // [t, i] = decay from i to t
    let fade_strict = diff.broadcast_add(&mask(true)?)?.exp()?;
    let fade_incl = diff.broadcast_add(&mask(false)?)?.exp()?;
    let kt = k.transpose(1, 2)?.contiguous()?;
    let a = (k.matmul(&kt)? * fade_strict)?.broadcast_mul(&beta)?;
    let p = (q.matmul(&kt)? * fade_incl)?;
    let gam = cum.exp()?.unsqueeze(2)?; // (n, c, 1)
    let last = cum.narrow(1, c - 1, 1)?; // (n, 1)
    let gam_last = last.exp()?.unsqueeze(2)?; // (n, 1, 1)
    let k_to_end = k.broadcast_mul(&last.broadcast_sub(&cum)?.exp()?.unsqueeze(2)?)?;
    let q_gam = q.broadcast_mul(&gam)?;

    // T = (I + A)^-1, A strictly lower triangular: forward substitution, one small matrix per block and head.
    let a_rows = a.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    let mut t_rows = vec![0f32; a_rows.len()];
    use rayon::prelude::*;
    t_rows.par_chunks_mut(c * c).zip(a_rows.par_chunks(c * c)).for_each(|(t, a)| {
        for r in 0..c {
            t[r * c + r] = 1.0;
            for i in 0..r {
                let air = a[r * c + i];
                if air != 0.0 {
                    for j in 0..=i {
                        t[r * c + j] -= air * t[i * c + j];
                    }
                }
            }
        }
    });
    let tri = Tensor::from_vec(t_rows, (n_v * nc, c, c), device)?;
    let w = tri.matmul(&k.broadcast_mul(&(beta.broadcast_mul(&gam))?)?)?; // (n, c, dk)
    let u = tri.matmul(&v.broadcast_mul(&beta)?)?; // (n, c, dv)

    let per_block = |t: Tensor, width: usize| t.reshape((n_v, nc, c, width));
    let (w, u, p, q_gam) = (per_block(w, dk)?, per_block(u, dv)?, per_block(p, c)?, per_block(q_gam, dk)?);
    let k_to_end_t = per_block(k_to_end, dk)?.transpose(2, 3)?.contiguous()?; // (n_v, nc, dk, c)
    let gam_last = gam_last.reshape((n_v, nc, 1, 1))?;
    let mut outs = Vec::with_capacity(nc);
    for b in 0..nc {
        let at = |t: &Tensor| t.narrow(1, b, 1)?.squeeze(1);
        let d = (at(&u)? - at(&w)?.matmul(&s)?)?; // (n_v, c, dv): each word's correction
        outs.push((at(&q_gam)?.matmul(&s)? + at(&p)?.contiguous()?.matmul(&d)?)?);
        s = (s.broadcast_mul(&at(&gam_last)?)? + at(&k_to_end_t)?.matmul(&d)?)?;
    }
    let o = Tensor::cat(&outs, 1)?.narrow(1, 0, l)?.transpose(0, 1)?.contiguous()?; // (l, n_v, dv)
    Ok((o, s))
}

/// A running-memory layer: per value head a 128 x 128 matrix that decays, is corrected by each
/// new key and value (the delta rule), and is read out with the query.
struct LinearAttn {
    in_proj: QMatMul,
    ba: QMatMul,
    conv_w: Tensor,
    dt_bias: Tensor,
    a: Tensor,
    norm: RmsNorm,
    out: QMatMul,
    n_k: usize,
    n_v: usize,
    dk: usize,
    dv: usize,
    eps: f64,
    conv_state: Option<Tensor>,
    state: Option<Tensor>,
}

impl LinearAttn {
    fn reset(&mut self) {
        self.conv_state = None;
        self.state = None;
    }

    fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let (b, l, h) = x.dims3()?;
        let device = x.device();
        let (n_k, n_v, dk, dv) = (self.n_k, self.n_v, self.dk, self.dv);
        let per_group = n_v / n_k;
        let x2 = x.reshape((l, h))?;

        // One projection holds, per key head: q, k, then the values and gates of its value heads.
        let mixed = self.in_proj.forward(&x2)?.reshape((l, n_k, 2 * dk + 2 * dv * per_group))?;
        let q = mixed.narrow(2, 0, dk)?.reshape((l, n_k * dk))?;
        let k = mixed.narrow(2, dk, dk)?.reshape((l, n_k * dk))?;
        let v = mixed.narrow(2, 2 * dk, dv * per_group)?.reshape((l, n_v * dv))?;
        let z = mixed.narrow(2, 2 * dk + dv * per_group, dv * per_group)?.reshape((l, n_v, dv))?;
        let qkv = Tensor::cat(&[&q, &k, &v], 1)?; // (l, channels)
        let channels = qkv.dim(1)?;

        let ba = self.ba.forward(&x2)?.reshape((l, n_k, 2 * per_group))?;
        let beta = candle_nn::ops::sigmoid(&ba.narrow(2, 0, per_group)?.reshape((l, n_v))?)?;
        let alpha = ba.narrow(2, per_group, per_group)?.reshape((l, n_v))?;
        let log_decay = softplus(&alpha.broadcast_add(&self.dt_bias)?)?.broadcast_mul(&self.a)?; // (l, n_v)

        // Causal depthwise convolution over the last `taps` inputs, carried across calls.
        let taps = self.conv_w.dim(1)?;
        let past = match &self.conv_state {
            Some(s) => s.clone(),
            None => Tensor::zeros((taps - 1, channels), DType::F32, device)?,
        };
        let padded = Tensor::cat(&[&past, &qkv], 0)?; // (taps - 1 + l, channels)
        let mut conv = padded.narrow(0, 0, l)?.broadcast_mul(&self.conv_w.narrow(1, 0, 1)?.t()?)?;
        for w in 1..taps {
            conv = (conv + padded.narrow(0, w, l)?.broadcast_mul(&self.conv_w.narrow(1, w, 1)?.t()?)?)?;
        }
        self.conv_state = Some(padded.narrow(0, l, taps - 1)?.contiguous()?);
        let conv = candle_nn::ops::silu(&conv)?;

        let qc = l2_norm(&conv.narrow(1, 0, n_k * dk)?.reshape((l, n_k, dk))?, self.eps)?;
        let kc = l2_norm(&conv.narrow(1, n_k * dk, n_k * dk)?.reshape((l, n_k, dk))?, self.eps)?;
        let vc = conv.narrow(1, 2 * n_k * dk, n_v * dv)?.reshape((l, n_v, dv))?;
        // Each key head serves `per_group` value heads, side by side.
        let widen = |t: Tensor| -> Result<Tensor> { t.unsqueeze(2)?.broadcast_as((l, n_k, per_group, dk))?.reshape((l, n_v, dk)) };
        let qc = (widen(qc)? * (1.0 / (dk as f64).sqrt()))?;
        let kc = widen(kc)?;

        let mut s = match &self.state {
            Some(s) => s.clone(),
            None => Tensor::zeros((n_v, dk, dv), DType::F32, device)?,
        };
        let (o, s) = if l > 1 {
            delta_chunked(&qc, &kc, &vc, &beta, &log_decay, s)?
        } else {
            let decay = log_decay.exp()?;
            let mut outs = Vec::with_capacity(l);
            for t in 0..l {
                let kt = kc.get(t)?.unsqueeze(1)?.contiguous()?; // (n_v, 1, dk)
                let qt = qc.get(t)?.unsqueeze(1)?.contiguous()?;
                let vt = vc.get(t)?.unsqueeze(1)?; // (n_v, 1, dv)
                let bt = beta.get(t)?.reshape((n_v, 1, 1))?;
                let gt = decay.get(t)?.reshape((n_v, 1, 1))?;
                s = s.broadcast_mul(&gt)?;
                let sk = kt.matmul(&s)?; // (n_v, 1, dv)
                let d = (vt - sk)?.broadcast_mul(&bt)?;
                s = (s + kt.transpose(1, 2)?.contiguous()?.matmul(&d)?)?;
                outs.push(qt.matmul(&s)?.squeeze(1)?); // (n_v, dv)
            }
            (Tensor::stack(&outs, 0)?, s) // (l, n_v, dv)
        };
        self.state = Some(s);

        let o = (self.norm.forward(&o)? * candle_nn::ops::silu(&z)?)?;
        self.out.forward(&o.reshape((l, n_v * dv))?)?.reshape((b, l, h))
    }
}

/// Classic attention with a sigmoid gate on its output and rotary position on a quarter of each head.
struct FullAttn {
    wq: QMatMul,
    wk: QMatMul,
    wv: QMatMul,
    wo: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    n_head: usize,
    n_kv: usize,
    hd: usize,
    rot: usize,
    freq_base: f64,
    kv_cache: KvCache,
}

impl FullAttn {
    fn rope(&self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let (_b, _h, l, _d) = x.dims4()?;
        let half = self.rot / 2;
        let inv: Vec<f32> = (0..half).map(|i| (1.0 / self.freq_base.powf(2.0 * i as f64 / self.rot as f64)) as f32).collect();
        let pos: Vec<f32> = (offset..offset + l).map(|p| p as f32).collect();
        let inv = Tensor::from_vec(inv, (1, half), x.device())?;
        let pos = Tensor::from_vec(pos, (l, 1), x.device())?;
        let freqs = pos.broadcast_mul(&inv)?; // (l, half)
        let (cos, sin) = (freqs.cos()?, freqs.sin()?);
        let rotated = candle_nn::rotary_emb::rope(&x.narrow(3, 0, self.rot)?.contiguous()?, &cos, &sin)?;
        Tensor::cat(&[&rotated, &x.narrow(3, self.rot, self.hd - self.rot)?], 3)
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let full = self.wq.forward(x)?.reshape((l, self.n_head, 2 * self.hd))?;
        let q = self.q_norm.forward(&full.narrow(2, 0, self.hd)?.contiguous()?)?;
        let gate = full.narrow(2, self.hd, self.hd)?.reshape((l, self.n_head * self.hd))?;
        let k = self.k_norm.forward(&self.wk.forward(x)?.reshape((l, self.n_kv, self.hd))?)?;
        let v = self.wv.forward(x)?.reshape((l, self.n_kv, self.hd))?;

        let heads = |t: Tensor| -> Result<Tensor> { t.transpose(0, 1)?.unsqueeze(0)?.contiguous() }; // (1, H, l, D)
        let q = self.rope(&heads(q)?, offset)?;
        let k = self.rope(&heads(k)?, offset)?;
        let v = heads(v)?;
        let (k, v) = self.kv_cache.append(&k.contiguous()?, &v)?;
        let s_len = k.dim(2)?;

        let g = self.n_head / self.n_kv;
        let q = q.contiguous()?.reshape((1, self.n_kv, g * l, self.hd))?;
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * (1.0 / (self.hd as f64).sqrt()))?;
        if let Some(m) = mask {
            scores = scores
                .reshape((1, self.n_kv, g, l, s_len))?
                .broadcast_add(&m.to_dtype(scores.dtype())?.unsqueeze(1)?)?
                .reshape((1, self.n_kv, g * l, s_len))?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?.reshape((1, self.n_head, l, self.hd))?;
        let ctx = ctx.transpose(1, 2)?.reshape((l, self.n_head * self.hd))?;
        let ctx = (ctx * candle_nn::ops::sigmoid(&gate)?)?;
        self.wo.forward(&ctx)?.reshape((b, l, ()))
    }
}

enum Mixer {
    Linear(LinearAttn),
    Full(FullAttn),
}

/// The expert every word uses, with its own sigmoid gate.
struct SharedExpert {
    gate_inp: Tensor,
    gate: QMatMul,
    up: QMatMul,
    down: QMatMul,
}

impl SharedExpert {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = (candle_nn::ops::silu(&self.gate.forward(x)?)? * self.up.forward(x)?)?;
        let y = self.down.forward(&hidden)?;
        let g = candle_nn::ops::sigmoid(&x.broadcast_mul(&self.gate_inp)?.sum_keepdim(D::Minus1)?)?;
        y.broadcast_mul(&g)
    }
}

struct Layer {
    mixer: Mixer,
    attn_norm: RmsNorm,
    post_norm: RmsNorm,
    moe: StreamedMoe,
    shared: SharedExpert,
}

pub struct Qwen3Next {
    embeddings: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    output: QMatMul,
    store: Arc<ExpertStore>,
    device: Device,
    pub thinks: bool,
    pub progress: Option<crate::model::Progress>,
}

impl Qwen3Next {
    pub fn load(path: &Path, device: &Device, settings: &Settings) -> Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let ct = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(path))?;
        let md = |s: &str| match ct.metadata.get(s) {
            None => candle::bail!("cannot find {s} in the GGUF metadata"),
            Some(v) => Ok(v.clone()),
        };
        let md_u = |s: &str| -> Result<usize> { Ok(md(&format!("qwen3next.{s}"))?.to_u32()? as usize) };
        let n_head = md_u("attention.head_count")?;
        let n_kv = md_u("attention.head_count_kv")?;
        let hd = md_u("attention.key_length")?;
        let rot = md_u("rope.dimension_count")?;
        let block_count = md_u("block_count")?;
        let top_k = md_u("expert_used_count")?;
        let embedding_length = md_u("embedding_length")?;
        let d_state = md_u("ssm.state_size")?;
        let n_group = md_u("ssm.group_count")?;
        let dt_rank = md_u("ssm.time_step_rank")?;
        let inner = md_u("ssm.inner_size")?;
        let interval = md_u("full_attention_interval").unwrap_or(4);
        let eps = md("qwen3next.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        // Some conversions leave the rope base out; Qwen3-Next's is ten million.
        let freq_base = md("qwen3next.rope.freq_base").ok().and_then(|v| v.to_f32().ok()).map_or(1e7, |f| f as f64);
        let thinks = md("tokenizer.chat_template").and_then(|t| t.to_string().cloned()).map_or(false, |t| t.contains("enable_thinking"));

        let store = ExpertStore::new(&ct, device, block_count, path, settings)?;
        let mut gg = Gguf::new(ct, &mut file, device.clone());
        let dense = |gg: &mut Gguf<&mut std::fs::File>, name: &str| -> Result<Tensor> { gg.tensor(name)?.dequantize(device)?.to_dtype(DType::F32) };

        let embeddings = gg.tensor("token_embd.weight")?.dequantize(device)?.to_dtype(DType::F16)?.to_device(&Device::Cpu)?;
        let norm = gg.rms_norm("output_norm.weight", eps)?;
        let output = match gg.qmatmul("output.weight") {
            Ok(v) => v,
            _ => gg.qmatmul("token_embd.weight")?,
        };

        let mut layers = Vec::with_capacity(block_count);
        for layer in 0..block_count {
            let p = format!("blk.{layer}");
            let mixer = if (layer + 1) % interval == 0 {
                Mixer::Full(FullAttn {
                    wq: gg.qmatmul(&format!("{p}.attn_q.weight"))?,
                    wk: gg.qmatmul(&format!("{p}.attn_k.weight"))?,
                    wv: gg.qmatmul(&format!("{p}.attn_v.weight"))?,
                    wo: gg.qmatmul(&format!("{p}.attn_output.weight"))?,
                    q_norm: gg.rms_norm(&format!("{p}.attn_q_norm.weight"), eps)?,
                    k_norm: gg.rms_norm(&format!("{p}.attn_k_norm.weight"), eps)?,
                    n_head,
                    n_kv,
                    hd,
                    rot,
                    freq_base,
                    kv_cache: KvCache::new(2, 4096),
                })
            } else {
                let conv_w = dense(&mut gg, &format!("{p}.ssm_conv1d.weight"))?; // (channels, taps)
                Mixer::Linear(LinearAttn {
                    in_proj: gg.qmatmul(&format!("{p}.ssm_in.weight"))?,
                    ba: gg.qmatmul(&format!("{p}.ssm_ba.weight"))?,
                    conv_w,
                    dt_bias: dense(&mut gg, &format!("{p}.ssm_dt.bias"))?,
                    a: dense(&mut gg, &format!("{p}.ssm_a"))?,
                    norm: gg.rms_norm(&format!("{p}.ssm_norm.weight"), eps)?,
                    out: gg.qmatmul(&format!("{p}.ssm_out.weight"))?,
                    n_k: n_group,
                    n_v: dt_rank,
                    dk: d_state,
                    dv: inner / dt_rank,
                    eps,
                    conv_state: None,
                    state: None,
                })
            };
            let gate = dense(&mut gg, &format!("{p}.ffn_gate_inp.weight"))?;
            let moe = StreamedMoe {
                layer,
                gate: Linear::new(gate, None),
                next_gate: None,
                ahead: settings.ahead.max(1),
                store: store.clone(),
                top_k,
                norm_topk_prob: true,
                prefetch_k: settings.prefetch,
            };
            let shared = SharedExpert {
                gate_inp: dense(&mut gg, &format!("{p}.ffn_gate_inp_shexp.weight"))?,
                gate: gg.qmatmul(&format!("{p}.ffn_gate_shexp.weight"))?,
                up: gg.qmatmul(&format!("{p}.ffn_up_shexp.weight"))?,
                down: gg.qmatmul(&format!("{p}.ffn_down_shexp.weight"))?,
            };
            layers.push(Layer {
                mixer,
                attn_norm: gg.rms_norm(&format!("{p}.attn_norm.weight"), eps)?,
                post_norm: gg.rms_norm(&format!("{p}.post_attention_norm.weight"), eps)?,
                moe,
                shared,
            });
        }
        wire_lookahead(&mut layers, settings.ahead);
        Ok(Self { embeddings: Embedding::new(embeddings, embedding_length), layers, norm, output, store, device: device.clone(), thinks, progress: None })
    }

    pub fn report(&self) {
        self.store.report_and_reset();
    }

    /// Forget the conversation: the classic layers' caches and the running memories together.
    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            match &mut layer.mixer {
                Mixer::Full(a) => a.kv_cache.reset(),
                Mixer::Linear(a) => a.reset(),
            }
        }
    }

    pub fn forward(&mut self, x: &Tensor, offset: usize) -> Result<Tensor> {
        let mut xs = self.embeddings.forward(&x.to_device(&Device::Cpu)?)?.to_device(&self.device)?.to_dtype(DType::F32)?;
        let (_b, l) = x.dims2()?;
        let mask = if l == 1 {
            None
        } else {
            Some(candle_transformers::utils::build_additive_causal_mask(l, offset, None, &self.device, DType::F32)?)
        };
        let n = self.layers.len();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let normed = layer.attn_norm.forward(&xs)?;
            let mixed = match &mut layer.mixer {
                Mixer::Linear(a) => a.forward(&normed)?,
                Mixer::Full(a) => a.forward(&normed, mask.as_ref(), offset)?,
            };
            let h = (mixed + &xs)?;
            let post = layer.post_norm.forward(&h)?;
            let routed = layer.moe.forward(&post)?;
            let shared = layer.shared.forward(&post)?;
            xs = ((routed + shared)? + h)?;
            if l > 1 {
                if let Some(p) = &mut self.progress {
                    p(crate::model::Step::Layer(i + 1, n));
                }
            }
        }
        let xs = self.norm.forward(&xs.narrow(1, l - 1, 1)?)?;
        self.output.forward(&xs)?.to_dtype(DType::F32)?.squeeze(1)
    }
}
