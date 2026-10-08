//! One Metal dispatch for every selected expert of every token: ggml's `kernel_mul_mv_id`,
//! which ships inside Candle's Metal kernel library but has no Rust entry point there.

use candle_metal_kernels::metal::{Buffer, Device};
use candle_metal_kernels::source::Source;
use candle_metal_kernels::utils::EncoderProvider;
use candle_metal_kernels::{set_params, GgmlDType, Kernels, MetalKernelError, Output};
use objc2_metal::MTLSize;

/// For every token and each of its `nei0` selected experts, multiplies one row of `src1` by that
/// expert's quantised matrix.
/// - `src0s` holds `ne02` expert matrices of `n` rows x `k` columns, `nb02` bytes apart.
/// - `ids` holds `nei1 x nei0` int32 expert indices into `src0s`.
/// - `src1` rows are f32; row `i11 = slot % ne11` of token `i12` sits at `i11*nb11 + i12*nb12` bytes.
/// - `dst` gets f32 `[nei1][nei0][n]`.
#[allow(clippy::too_many_arguments)]
pub fn matvec_id(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    (n, k, ne02, nb02): (usize, usize, usize, usize),
    src0s: &Buffer,
    src1: &Buffer,
    src1_offset: usize,
    (ne11, nb11, nb12): (usize, usize, usize),
    ids: &Buffer,
    ids_offset: usize,
    (nei0, nei1): (usize, usize),
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    let (nth0, nth1, align) = match dtype {
        GgmlDType::Q4_0 | GgmlDType::Q4_1 | GgmlDType::Q5_0 | GgmlDType::Q5_1 | GgmlDType::Q8_0 | GgmlDType::Q8_1 => (8, 8, 8),
        GgmlDType::Q2K => (2, 32, 8),
        GgmlDType::Q4K => (4, 8, 4),
        GgmlDType::Q3K | GgmlDType::Q5K => (2, 32, 4),
        GgmlDType::Q6K => (2, 32, 2),
        GgmlDType::F16 | GgmlDType::BF16 | GgmlDType::Q8K | GgmlDType::F32 => (32, 1, 8),
    };
    let name = match dtype {
        GgmlDType::Q4_0 => "kernel_mul_mv_id_q4_0_f32",
        GgmlDType::Q4_1 => "kernel_mul_mv_id_q4_1_f32",
        GgmlDType::Q5_0 => "kernel_mul_mv_id_q5_0_f32",
        GgmlDType::Q5_1 => "kernel_mul_mv_id_q5_1_f32",
        GgmlDType::Q8_0 => "kernel_mul_mv_id_q8_0_f32",
        GgmlDType::Q2K => "kernel_mul_mv_id_q2_K_f32",
        GgmlDType::Q3K => "kernel_mul_mv_id_q3_K_f32",
        GgmlDType::Q4K => "kernel_mul_mv_id_q4_K_f32",
        GgmlDType::Q5K => "kernel_mul_mv_id_q5_K_f32",
        GgmlDType::Q6K => "kernel_mul_mv_id_q6_K_f32",
        GgmlDType::F16 => "kernel_mul_mv_id_f16_f32",
        GgmlDType::F32 => "kernel_mul_mv_id_f32_f32",
        other => return Err(MetalKernelError::LoadLibraryError(format!("no mul_mv_id kernel for {other:?}"))),
    };
    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let ne00 = k as i64;
    let ne01 = n as i64;
    let ne02 = ne02 as i64;
    let nb00 = 0u64;
    let nb01 = 0u64;
    let nb02 = nb02 as u64;
    let ne10 = k as i64;
    let ne11 = ne11 as i64;
    let ne12 = nei1 as i64;
    let ne13 = 1i64;
    let nb10 = 4u64;
    let nb11 = nb11 as u64;
    let nb12 = nb12 as u64;
    let ne0 = n as i64;
    let ne1 = nei0 as i64;
    let nb1 = (n * 4) as u64;
    let nei0 = nei0 as i64;
    let nei1 = nei1 as i64;
    let nbi1 = (nei0 * 4) as u64;

    set_params!(
        encoder,
        (
            src0s,
            (src1, src1_offset),
            Output::with_offset(dst, 0),
            (ids, ids_offset),
            nei0,
            nei1,
            nbi1,
            ne00,
            ne01,
            ne02,
            nb00,
            nb01,
            nb02,
            ne10,
            ne11,
            ne12,
            ne13,
            nb10,
            nb11,
            nb12,
            ne0,
            ne1,
            nb1
        )
    );
    let thread_groups_count = MTLSize { width: n.div_ceil(align), height: 1, depth: (nei0 * nei1) as usize };
    let threads_per_threadgroup = MTLSize { width: nth0, height: nth1, depth: 1 };
    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}
