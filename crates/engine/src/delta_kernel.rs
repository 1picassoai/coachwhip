//! Lab (`COACHWHIP_DELTA_KERNEL=1`): the gated delta rule for ONE word, all value heads, in one Metal
//! kernel. Replaces the dozen small ops of `delta_steps` when `upto == 1`. Same arithmetic:
//! s = s * g; sk = k . s; d = (v - sk) * beta; s = s + k^T d; o = q . s.
//! One threadgroup per value head, one thread per value column j: the column of the memory is read,
//! corrected and written by the thread that owns it, so no reduction crosses threads.

use candle::backend::BackendStorage;
use candle::{CustomOp3, DType, Layout, MetalStorage, Result, Shape, Tensor, D};
use candle_metal_kernels::metal::ComputePipeline;
use candle_metal_kernels::{set_params, Output};
use objc2_metal::MTLSize;
use std::sync::{Arc, OnceLock};

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// qk: (n_v, 2*dk) q then k per head. vbd: (n_v, dv + 2) v, beta, decay per head. s: (n_v, dk, dv).
// out: (n_v, dk*dv + dv): the new memory, then the read-out.
kernel void coachwhip_delta_word(
    device const float* qk    [[buffer(0)]],
    device const float* vbd   [[buffer(1)]],
    device const float* s     [[buffer(2)]],
    device float* out         [[buffer(3)]],
    constant uint& dk         [[buffer(4)]],
    constant uint& dv         [[buffer(5)]],
    uint h   [[threadgroup_position_in_grid]],
    uint j   [[thread_position_in_threadgroup]],
    uint tgs [[threads_per_threadgroup]])
{
    device const float* q = qk + h * 2 * dk;
    device const float* k = q + dk;
    device const float* v = vbd + h * (dv + 2);
    const float beta = v[dv];
    const float g = v[dv + 1];
    device const float* sh = s + h * dk * dv;
    device float* so = out + h * (dk * dv + dv);
    device float* o = so + dk * dv;
    for (uint col = j; col < dv; col += tgs) {
        float sk = 0.0f;
        for (uint i = 0; i < dk; i++) sk += k[i] * sh[i * dv + col];
        sk *= g;
        const float d = (v[col] - sk) * beta;
        float acc = 0.0f;
        for (uint i = 0; i < dk; i++) {
            const float ns = g * sh[i * dv + col] + k[i] * d;
            so[i * dv + col] = ns;
            acc += q[i] * ns;
        }
        o[col] = acc;
    }
}
"#;

fn pipeline(device: &candle::MetalDevice) -> Result<Arc<ComputePipeline>> {
    static PIPE: OnceLock<Arc<ComputePipeline>> = OnceLock::new();
    if let Some(p) = PIPE.get() {
        return Ok(p.clone());
    }
    let lib = device.device().new_library_with_source(SOURCE, None).map_err(candle::MetalError::from)?;
    let f = lib.get_function("coachwhip_delta_word", None).map_err(candle::MetalError::from)?;
    let p = Arc::new(device.device().new_compute_pipeline_state_with_function(&f).map_err(candle::MetalError::from)?);
    let _ = PIPE.set(p.clone());
    Ok(p)
}

pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("COACHWHIP_DELTA_KERNEL").map(|v| v == "1").unwrap_or(false))
}

struct DeltaWord {
    dk: usize,
    dv: usize,
}

impl CustomOp3 for DeltaWord {
    fn name(&self) -> &'static str {
        "coachwhip-delta-word"
    }

    fn cpu_fwd(&self, _: &candle::CpuStorage, _: &Layout, _: &candle::CpuStorage, _: &Layout, _: &candle::CpuStorage, _: &Layout) -> Result<(candle::CpuStorage, Shape)> {
        candle::bail!("coachwhip-delta-word runs on Metal only")
    }

    fn metal_fwd(&self, qk: &MetalStorage, ql: &Layout, vbd: &MetalStorage, vl: &Layout, s: &MetalStorage, sl: &Layout) -> Result<(MetalStorage, Shape)> {
        if qk.dtype() != DType::F32 || vbd.dtype() != DType::F32 || s.dtype() != DType::F32 {
            candle::bail!("coachwhip-delta-word: inputs must be f32")
        }
        if ql.contiguous_offsets().is_none() || vl.contiguous_offsets().is_none() || sl.contiguous_offsets().is_none() {
            candle::bail!("coachwhip-delta-word: inputs must be contiguous")
        }
        let n_v = sl.shape().dims()[0];
        let (dk, dv) = (self.dk, self.dv);
        let device = s.device().clone();
        let n_out = n_v * (dk * dv + dv);
        let dst = device.new_buffer_builder().with_size_for(n_out, DType::F32).with_label("delta-word").build()?;
        let pipe = pipeline(&device)?;
        {
            let encoder = device.command_encoder()?;
            let encoder = encoder.as_ref();
            encoder.set_compute_pipeline_state(&pipe);
            let dk_u = dk as u32;
            let dv_u = dv as u32;
            set_params!(
                encoder,
                (
                    (qk.buffer(), ql.start_offset() * 4),
                    (vbd.buffer(), vl.start_offset() * 4),
                    (s.buffer(), sl.start_offset() * 4),
                    Output::with_offset(&dst, 0),
                    dk_u,
                    dv_u
                )
            );
            let threads = dv.min(256);
            encoder.dispatch_thread_groups(MTLSize { width: n_v, height: 1, depth: 1 }, MTLSize { width: threads, height: 1, depth: 1 });
        }
        Ok((MetalStorage::new(dst, device, n_out, DType::F32), Shape::from((n_v, dk * dv + dv))))
    }
}

/// One word of the delta rule. `q, k: (n_v, dk)`, `v: (n_v, dv)`, `beta, decay: (n_v)`, `s: (n_v, dk, dv)`.
/// Returns `(o (n_v, dv), s' (n_v, dk, dv))`.
pub fn delta_word(q: &Tensor, k: &Tensor, v: &Tensor, beta: &Tensor, decay: &Tensor, s: &Tensor) -> Result<(Tensor, Tensor)> {
    let (n_v, dk, dv) = s.dims3()?;
    let f = |t: &Tensor| -> Result<Tensor> { t.to_dtype(DType::F32)?.contiguous() };
    let qk = Tensor::cat(&[&f(q)?.reshape((n_v, dk))?, &f(k)?.reshape((n_v, dk))?], D::Minus1)?.contiguous()?;
    let vbd = Tensor::cat(&[&f(v)?.reshape((n_v, dv))?, &f(beta)?.reshape((n_v, 1))?, &f(decay)?.reshape((n_v, 1))?], D::Minus1)?.contiguous()?;
    let s = f(s)?;
    let out = qk.apply_op3_no_bwd(&vbd, &s, &DeltaWord { dk, dv })?;
    let s_new = out.narrow(1, 0, dk * dv)?.reshape((n_v, dk, dv))?;
    let o = out.narrow(1, dk * dv, dv)?.contiguous()?;
    Ok((o, s_new))
}
