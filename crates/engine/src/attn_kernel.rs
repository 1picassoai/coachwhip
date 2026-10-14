//! Lab (`COACHWHIP_ATTN_KERNEL=1`): the whole middle of a Gated DeltaNet layer for ONE word in one Metal
//! kernel: causal conv taps + silu, the two L2 norms, the key-head widening and query scale, the delta
//! rule over the memory, the output RMS norm and the silu gate. Only the two big projections stay as
//! their own matmul kernels. Same arithmetic as the op-by-op path in `model_next.rs`.
//! One threadgroup per value head, one thread per value column.

use candle::backend::BackendStorage;
use candle::{CustomOp3, DType, Layout, MetalStorage, Result, Shape, Storage, Tensor, D};
use candle_metal_kernels::metal::ComputePipeline;
use candle_metal_kernels::{set_params, Output};
use objc2_metal::MTLSize;
use std::sync::{Arc, OnceLock};

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

inline float silu_f(float x) { return x / (1.0f + exp(-x)); }

// qkv: (channels) this word's projection: q heads (n_k*dk), k heads (n_k*dk), v heads (n_v*dv).
// z: (n_v*dv) gate projection. ba: beta (n_v) then alpha (n_v), raw.
// conv_state: (taps-1, channels) past inputs. conv_w: (channels, taps). dt_bias, a: (n_v). norm_w: (dv).
// s: (n_v, dk, dv) memory. out: o (n_v*dv) | s' (n_v*dk*dv) | conv_state' ((taps-1)*channels).
kernel void coachwhip_gdn_word(
    device const float* qkv        [[buffer(0)]],
    device const float* z          [[buffer(1)]],
    device const float* ba         [[buffer(2)]],
    device const float* conv_state [[buffer(3)]],
    device const float* conv_w     [[buffer(4)]],
    device const float* dt_bias    [[buffer(5)]],
    device const float* a_vec      [[buffer(6)]],
    device const float* norm_w     [[buffer(7)]],
    device const float* s          [[buffer(8)]],
    device float* out              [[buffer(9)]],
    constant uint& n_k             [[buffer(10)]],
    constant uint& n_v             [[buffer(11)]],
    constant uint& dk              [[buffer(12)]],
    constant uint& dv              [[buffer(13)]],
    constant uint& taps            [[buffer(14)]],
    constant uint& cycle           [[buffer(15)]],
    constant float& eps            [[buffer(16)]],
    uint h   [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]])
{
    threadgroup float q_sh[512];
    threadgroup float k_sh[512];
    threadgroup float red[2];
    const uint channels = 2 * n_k * dk + n_v * dv;
    const uint per_group = n_v / n_k;
    const uint kh = cycle ? (h % n_k) : (h / per_group);
    device float* o_out = out;
    device float* s_out = out + n_v * dv;
    device float* c_out = s_out + n_v * dk * dv;

    // 1. conv + silu for this head's q, k channels (into shared memory) and its v channels (registers).
    for (uint i = tid; i < dk; i += tgs) {
        uint cq = kh * dk + i;
        uint ck = n_k * dk + kh * dk + i;
        float xq = 0.0f, xk = 0.0f;
        for (uint t = 0; t + 1 < taps; t++) {
            xq += conv_w[cq * taps + t] * conv_state[t * channels + cq];
            xk += conv_w[ck * taps + t] * conv_state[t * channels + ck];
        }
        xq += conv_w[cq * taps + taps - 1] * qkv[cq];
        xk += conv_w[ck * taps + taps - 1] * qkv[ck];
        q_sh[i] = silu_f(xq);
        k_sh[i] = silu_f(xk);
        // shift this channel's conv window: rows 1..taps-1 move up, the new word is the last row.
        for (uint t = 0; t + 2 < taps; t++) {
            c_out[t * channels + cq] = conv_state[(t + 1) * channels + cq];
            c_out[t * channels + ck] = conv_state[(t + 1) * channels + ck];
        }
        c_out[(taps - 2) * channels + cq] = qkv[cq];
        c_out[(taps - 2) * channels + ck] = qkv[ck];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 2. L2 norms of q and k (one thread sums; dk is small), query scale 1/sqrt(dk).
    if (tid == 0) {
        float sq = 0.0f, sk = 0.0f;
        for (uint i = 0; i < dk; i++) { sq += q_sh[i] * q_sh[i]; sk += k_sh[i] * k_sh[i]; }
        red[0] = sqrt(sq + eps);
        red[1] = sqrt(sk + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float qscale = 1.0f / (red[0] * sqrt((float)dk));
    const float kscale = 1.0f / red[1];
    for (uint i = tid; i < dk; i += tgs) { q_sh[i] *= qscale; k_sh[i] *= kscale; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. gates for this head.
    const float beta = 1.0f / (1.0f + exp(-ba[h]));
    const float alpha = ba[n_v + h];
    const float g = exp(log(exp(alpha + dt_bias[h]) + 1.0f) * a_vec[h]);

    // 4. the delta rule, one column per thread; 5. output RMS norm and silu gate.
    device const float* sh = s + h * dk * dv;
    device float* so = s_out + h * dk * dv;
    for (uint j = tid; j < dv; j += tgs) {
        uint cv = 2 * n_k * dk + h * dv + j;
        float xv = 0.0f;
        for (uint t = 0; t + 1 < taps; t++) xv += conv_w[cv * taps + t] * conv_state[t * channels + cv];
        xv += conv_w[cv * taps + taps - 1] * qkv[cv];
        const float v = silu_f(xv);
        for (uint t = 0; t + 2 < taps; t++) c_out[t * channels + cv] = conv_state[(t + 1) * channels + cv];
        c_out[(taps - 2) * channels + cv] = qkv[cv];

        float skv = 0.0f;
        for (uint i = 0; i < dk; i++) skv += k_sh[i] * sh[i * dv + j];
        skv *= g;
        const float d = (v - skv) * beta;
        float acc = 0.0f;
        for (uint i = 0; i < dk; i++) {
            const float ns = g * sh[i * dv + j] + k_sh[i] * d;
            so[i * dv + j] = ns;
            acc += q_sh[i] * ns;
        }
        o_out[h * dv + j] = acc;
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
        float ss = 0.0f;
        for (uint j = 0; j < dv; j++) { float x = o_out[h * dv + j]; ss += x * x; }
        red[0] = rsqrt(ss / (float)dv + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float rms = red[0];
    for (uint j = tid; j < dv; j += tgs) {
        float x = o_out[h * dv + j] * rms * norm_w[j];
        o_out[h * dv + j] = x * silu_f(z[h * dv + j]);
    }
}
"#;

fn pipeline(device: &candle::MetalDevice) -> Result<Arc<ComputePipeline>> {
    static PIPE: OnceLock<Arc<ComputePipeline>> = OnceLock::new();
    if let Some(p) = PIPE.get() {
        return Ok(p.clone());
    }
    let lib = device.device().new_library_with_source(SOURCE, None).map_err(candle::MetalError::from)?;
    let f = lib.get_function("coachwhip_gdn_word", None).map_err(candle::MetalError::from)?;
    let p = Arc::new(device.device().new_compute_pipeline_state_with_function(&f).map_err(candle::MetalError::from)?);
    let _ = PIPE.set(p.clone());
    Ok(p)
}

pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("COACHWHIP_ATTN_KERNEL").map(|v| v != "0").unwrap_or(true))
}

/// The layer's constants and this word's state, read straight from their tensors' Metal buffers.
pub struct GdnWord {
    pub conv_state: Tensor,
    pub conv_w: Tensor,
    pub dt_bias: Tensor,
    pub a: Tensor,
    pub norm_w: Tensor,
    pub s: Tensor,
    pub n_k: usize,
    pub n_v: usize,
    pub dk: usize,
    pub dv: usize,
    pub taps: usize,
    pub cycle: bool,
    pub eps: f32,
}

fn check(t: &Tensor, what: &str) -> Result<()> {
    if t.dtype() != DType::F32 {
        candle::bail!("coachwhip-gdn-word: {what} must be f32, got {:?}", t.dtype())
    }
    if !t.is_contiguous() {
        candle::bail!("coachwhip-gdn-word: {what} must be contiguous")
    }
    Ok(())
}

impl CustomOp3 for GdnWord {
    fn name(&self) -> &'static str {
        "coachwhip-gdn-word"
    }

    fn cpu_fwd(&self, _: &candle::CpuStorage, _: &Layout, _: &candle::CpuStorage, _: &Layout, _: &candle::CpuStorage, _: &Layout) -> Result<(candle::CpuStorage, Shape)> {
        candle::bail!("coachwhip-gdn-word runs on Metal only")
    }

    fn metal_fwd(&self, qkv: &MetalStorage, ql: &Layout, z: &MetalStorage, zl: &Layout, ba: &MetalStorage, bl: &Layout) -> Result<(MetalStorage, Shape)> {
        for (st, l, what) in [(qkv, ql, "qkv"), (z, zl, "z"), (ba, bl, "ba")] {
            if st.dtype() != DType::F32 {
                candle::bail!("coachwhip-gdn-word: {what} must be f32")
            }
            if l.contiguous_offsets().is_none() {
                candle::bail!("coachwhip-gdn-word: {what} must be contiguous")
            }
        }
        let (n_k, n_v, dk, dv, taps) = (self.n_k, self.n_v, self.dk, self.dv, self.taps);
        let channels = 2 * n_k * dk + n_v * dv;
        let n_out = n_v * dv + n_v * dk * dv + (taps - 1) * channels;
        let device = qkv.device().clone();
        let dst = device.new_buffer_builder().with_size_for(n_out, DType::F32).with_label("gdn-word").build()?;
        let pipe = pipeline(&device)?;
        // Hold the read guards of the layer's own tensors for the encode.
        let cs = self.conv_state.storage_and_layout();
        let cw = self.conv_w.storage_and_layout();
        let db = self.dt_bias.storage_and_layout();
        let av = self.a.storage_and_layout();
        let nw = self.norm_w.storage_and_layout();
        let ss = self.s.storage_and_layout();
        fn m<'a>(g: &'a Storage, name: &str) -> Result<&'a MetalStorage> {
            match g {
                Storage::Metal(m) => Ok(m),
                _ => candle::bail!("coachwhip-gdn-word: {name} is not on Metal"),
            }
        }
        let (cs_m, cw_m, db_m, av_m, nw_m, ss_m) = (m(&cs.0, "conv_state")?, m(&cw.0, "conv_w")?, m(&db.0, "dt_bias")?, m(&av.0, "a")?, m(&nw.0, "norm_w")?, m(&ss.0, "s")?);
        {
            let encoder = device.command_encoder()?;
            let encoder = encoder.as_ref();
            encoder.set_compute_pipeline_state(&pipe);
            let (n_k_u, n_v_u, dk_u, dv_u, taps_u, cycle_u) = (n_k as u32, n_v as u32, dk as u32, dv as u32, taps as u32, self.cycle as u32);
            set_params!(
                encoder,
                (
                    (qkv.buffer(), ql.start_offset() * 4),
                    (z.buffer(), zl.start_offset() * 4),
                    (ba.buffer(), bl.start_offset() * 4),
                    (cs_m.buffer(), cs.1.start_offset() * 4),
                    (cw_m.buffer(), cw.1.start_offset() * 4),
                    (db_m.buffer(), db.1.start_offset() * 4),
                    (av_m.buffer(), av.1.start_offset() * 4),
                    (nw_m.buffer(), nw.1.start_offset() * 4),
                    (ss_m.buffer(), ss.1.start_offset() * 4),
                    Output::with_offset(&dst, 0),
                    n_k_u,
                    n_v_u,
                    dk_u,
                    dv_u,
                    taps_u,
                    cycle_u,
                    self.eps
                )
            );
            let threads = dv.min(256);
            encoder.dispatch_thread_groups(MTLSize { width: n_v, height: 1, depth: 1 }, MTLSize { width: threads, height: 1, depth: 1 });
        }
        Ok((MetalStorage::new(dst, device, n_out, DType::F32), Shape::from(n_out)))
    }
}

/// One word through the middle of the layer. Returns `(o_gated (1, n_v*dv), s' (n_v, dk, dv), conv_state' (taps-1, channels))`.
pub fn gdn_word(op: GdnWord, qkv: &Tensor, z: &Tensor, beta: &Tensor, alpha: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
    let (n_k, n_v, dk, dv, taps) = (op.n_k, op.n_v, op.dk, op.dv, op.taps);
    let channels = 2 * n_k * dk + n_v * dv;
    for (t, w) in [(&op.conv_state, "conv_state"), (&op.conv_w, "conv_w"), (&op.dt_bias, "dt_bias"), (&op.a, "a"), (&op.norm_w, "norm_w"), (&op.s, "s")] {
        check(t, w)?;
    }
    let f = |t: &Tensor| -> Result<Tensor> { t.flatten_all()?.to_dtype(DType::F32)?.contiguous() };
    let qkv = f(qkv)?;
    let z = f(z)?;
    let ba = Tensor::cat(&[&f(beta)?, &f(alpha)?], D::Minus1)?.contiguous()?;
    let out = qkv.apply_op3_no_bwd(&z, &ba, &op)?;
    let o = out.narrow(0, 0, n_v * dv)?.reshape((1, n_v * dv))?;
    let s_new = out.narrow(0, n_v * dv, n_v * dk * dv)?.reshape((n_v, dk, dv))?;
    let conv_new = out.narrow(0, n_v * dv + n_v * dk * dv, (taps - 1) * channels)?.reshape((taps - 1, channels))?;
    Ok((o, s_new, conv_new))
}
