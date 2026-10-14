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
        GgmlDType::Q4_0 | GgmlDType::Q4_1 | GgmlDType::Q5_0 | GgmlDType::Q5_1 | GgmlDType::Q8_0 | GgmlDType::Q8_1 | GgmlDType::Mxfp4 => (8, 8, 8),
        GgmlDType::Q2K => (2, 32, 8),
        GgmlDType::Q4K => (4, 8, 4),
        GgmlDType::Q3K | GgmlDType::Q5K => (2, 32, 4),
        GgmlDType::Q6K => (2, 32, 2),
        GgmlDType::F16 | GgmlDType::BF16 | GgmlDType::Q8K | GgmlDType::F32 => (32, 1, 8),
        GgmlDType::Iq2Xxs | GgmlDType::Iq3Xxs => (4, 16, 8),
        GgmlDType::Iq4Xs => (4, 16, 4),
    };
    let name = match dtype {
        GgmlDType::Q4_0 => "kernel_mul_mv_id_q4_0_f32",
        GgmlDType::Q4_1 => "kernel_mul_mv_id_q4_1_f32",
        GgmlDType::Q5_0 => "kernel_mul_mv_id_q5_0_f32",
        GgmlDType::Q5_1 => "kernel_mul_mv_id_q5_1_f32",
        GgmlDType::Q8_0 => "kernel_mul_mv_id_q8_0_f32",
        GgmlDType::Mxfp4 => "kernel_mul_mv_id_mxfp4_f32",
        GgmlDType::Iq2Xxs => "kernel_mul_mv_id_iq2_xxs_f32",
        GgmlDType::Iq3Xxs => "kernel_mul_mv_id_iq3_xxs_f32",
        GgmlDType::Iq4Xs => "kernel_mul_mv_id_iq4_xs_f32",
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
    // IQ kernels keep ggml's grids and signs in threadgroup memory.
    let iq_shmem: usize = match dtype { GgmlDType::Iq2Xxs => 256 * 8 + 128, GgmlDType::Iq3Xxs => 256 * 4 + 128, GgmlDType::Iq4Xs => 32 * 4, _ => 0 };
    if iq_shmem > 0 {
        encoder.set_threadgroup_memory_length(0, iq_shmem);
    }

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


/// Lab switch: ggml's current mul_mv_id kernels for the IQ expert types.
pub fn new_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("COACHWHIP_NEWMV").map(|v| v == "1").unwrap_or(false))
}

#[repr(C)]
struct KargsMvId {
    nei0: i32, nei1: i32, nbi1: u64,
    ne00: i32, ne01: i32, ne02: i32, nb00: u64, nb01: u64, nb02: u64,
    ne10: i32, ne11: i32, ne12: i32, ne13: i32, nb10: u64, nb11: u64, nb12: u64,
    ne0: i32, ne1: i32, nb1: u64, nr0: i32,
}

fn new_pipe(device: &Device, name: &str, split: bool) -> Result<std::sync::Arc<candle_metal_kernels::metal::ComputePipeline>, MetalKernelError> {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};
    static P: OnceLock<Mutex<HashMap<(String, bool), Arc<candle_metal_kernels::metal::ComputePipeline>>>> = OnceLock::new();
    let mut g = P.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap();
    if let Some(p) = g.get(&(name.to_string(), split)) {
        return Ok(p.clone());
    }
    let src = if split { include_str!("ggml_mv_split.metal") } else { include_str!("ggml_mv_plain.metal") };
    let lib = device.new_library_with_source(src, None)?;
    let f = lib.get_function(name, None)?;
    let p = Arc::new(device.new_compute_pipeline_state_with_function(&f)?);
    g.insert((name.to_string(), split), p.clone());
    Ok(p)
}

/// Same contract as [`matvec_id`], IQ types only; returns Ok(false) for any other type.
#[allow(clippy::too_many_arguments)]
pub fn matvec_id_new(
    device: &Device,
    ep: impl EncoderProvider,
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
) -> Result<bool, MetalKernelError> {
    let nb32 = k / 32;
    let short_row = nb32 < 32 && 32 % nb32 == 0;
    let (name, block, nr0, split, smem) = match dtype {
        GgmlDType::Iq2Xxs => ("kernel_mul_mv_id_iq2_xxs_f32", 66usize, if short_row { 8 } else { 4 }, short_row, 256 * 8 + 128),
        GgmlDType::Iq3Xxs => ("kernel_mul_mv_id_iq3_xxs_f32", 98, if short_row { 8 } else { 4 }, short_row, 256 * 4 + 128),
        GgmlDType::Iq4Xs => ("kernel_mul_mv_id_iq4_xs_f32", 136, 2, false, 32 * 4),
        _ => return Ok(false),
    };
    let nsg = 2usize;
    let pipeline = new_pipe(device, name, split)?;
    let encoder = ep.encoder();
    let encoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_threadgroup_memory_length(0, smem);
    let args = KargsMvId {
        nei0: nei0 as i32, nei1: nei1 as i32, nbi1: (nei0 * 4) as u64,
        ne00: k as i32, ne01: n as i32, ne02: ne02 as i32,
        nb00: block as u64, nb01: (k / 256 * block) as u64, nb02: nb02 as u64,
        ne10: k as i32, ne11: ne11 as i32, ne12: nei1 as i32, ne13: 1,
        nb10: 4, nb11: nb11 as u64, nb12: nb12 as u64,
        ne0: n as i32, ne1: nei0 as i32, nb1: (n * 4) as u64, nr0: nr0 as i32,
    };
    encoder.set_bytes(0, &args);
    encoder.set_input_buffer(1, Some(src0s), 0);
    encoder.set_input_buffer(2, Some(src1), src1_offset);
    encoder.set_output_buffer(3, Some(dst), 0);
    encoder.set_input_buffer(4, Some(ids), ids_offset);
    let groups = MTLSize { width: n.div_ceil(nr0 * nsg), height: 1, depth: nei0 * nei1 };
    encoder.dispatch_thread_groups(groups, MTLSize { width: 32, height: nsg, depth: 1 });
    Ok(true)
}


pub fn group_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("COACHWHIP_GROUP").map(|v| v == "1").unwrap_or(false))
}

/// Same contract as [`matvec_id`], for a batch of words: one threadgroup column per expert gathers the
/// words that picked it and multiplies them as one small matrix. Returns Ok(false) when not supported.
#[allow(clippy::too_many_arguments)]
pub fn matmul_id_grouped(
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
) -> Result<bool, MetalKernelError> {
    let (name, block) = match dtype {
        GgmlDType::Iq2Xxs => ("kernel_mul_mm_id_iq2_xxs_f32", 66usize),
        GgmlDType::Iq3Xxs => ("kernel_mul_mm_id_iq3_xxs_f32", 98),
        GgmlDType::Iq4Xs => ("kernel_mul_mm_id_iq4_xs_f32", 136),
        _ => return Ok(false),
    };
    let smem = (8192 + 4 * nei0 * nei1 + 15) / 16 * 16;
    if smem > 32768 || k % 256 != 0 {
        return Ok(false);
    }
    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_threadgroup_memory_length(0, smem);
    let (nei0_, nei1_) = (nei0 as i64, nei1 as i64);
    let nbi1 = (nei0 * 4) as u64;
    let ne00 = k as i64;
    let ne02_ = ne02 as i64;
    let nb01 = (k / 256 * block) as u64;
    let nb02_ = nb02 as u64;
    let ne11_ = ne11 as i64;
    let ne12 = nei1 as i64;
    let ne13 = 1i64;
    let nb10 = 4u64;
    let nb11_ = nb11 as u64;
    let nb12_ = nb12 as u64;
    let ne0 = n as i64;
    let ne1 = nei0 as i64;
    let nb1 = (n * 4) as u64;
    set_params!(
        encoder,
        (
            src0s,
            (src1, src1_offset),
            Output::with_offset(dst, 0),
            (ids, ids_offset),
            nei0_, nei1_, nbi1, ne00, ne02_, nb01, nb02_, ne11_, ne12, ne13, nb10, nb11_, nb12_, ne0, ne1, nb1
        )
    );
    let groups = MTLSize { width: nei1.div_ceil(32), height: n.div_ceil(64), depth: ne02 };
    encoder.dispatch_thread_groups(groups, MTLSize { width: 128, height: 1, depth: 1 });
    Ok(true)
}


pub fn group2_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("COACHWHIP_GROUP").map(|v| v == "2").unwrap_or(false))
}

/// Grouped multiply from piles sorted on the CPU: `piles` (slot, word) u16 pairs, `heads` (expert,
/// first, count) u32 triples, `used` experts, `most` = the largest pile.
#[allow(clippy::too_many_arguments)]
pub fn matmul_id_piles(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    (n, k, ne02, nb02): (usize, usize, usize, usize),
    src0s: &Buffer,
    src1: &Buffer,
    src1_offset: usize,
    (ne11, nb11, nb12): (usize, usize, usize),
    piles: &Buffer,
    heads: &Buffer,
    (used, most, nei0): (usize, usize, usize),
    dst: &Buffer,
) -> Result<bool, MetalKernelError> {
    let (name, block) = match dtype {
        GgmlDType::Iq2Xxs => ("kernel_mul_mm_idg_iq2_xxs_f32", 66usize),
        GgmlDType::Iq3Xxs => ("kernel_mul_mm_idg_iq3_xxs_f32", 98),
        GgmlDType::Iq4Xs => ("kernel_mul_mm_idg_iq4_xs_f32", 136),
        _ => return Ok(false),
    };
    let smem = (8192 + 4 * most + 15) / 16 * 16;
    if smem > 32768 || k % 256 != 0 {
        return Ok(false);
    }
    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_threadgroup_memory_length(0, smem);
    let ne00 = k as i64;
    let ne02_ = ne02 as i64;
    let nb01 = (k / 256 * block) as u64;
    let nb02_ = nb02 as u64;
    let ne11_ = ne11 as i64;
    let ne12 = 1i64;
    let nb10 = 4u64;
    let nb11_ = nb11 as u64;
    let nb12_ = nb12 as u64;
    let ne0 = n as i64;
    let ne1 = nei0 as i64;
    set_params!(
        encoder,
        (src0s, (src1, src1_offset), Output::with_offset(dst, 0), piles, heads, ne00, ne02_, nb01, nb02_, ne11_, ne12, nb10, nb11_, nb12_, ne0, ne1)
    );
    let groups = MTLSize { width: most.div_ceil(32), height: n.div_ceil(64), depth: used };
    encoder.dispatch_thread_groups(groups, MTLSize { width: 128, height: 1, depth: 1 });
    Ok(true)
}
