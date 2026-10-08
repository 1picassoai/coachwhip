//! The GPU shrink (Captain's design, 8 Oct): an expert arriving from the file at Q3_K or Q5_K is
//! shrunk to Q2_K on the GPU the moment its bytes land, so the bank holds Q2 copies and parks
//! more experts in the same memory. One thread per 256-weight super-block: dequantise with the
//! same code the matvec kernels use, then fit 16 sub-blocks of 16 to 2 bits by min and max.
//! Its own command queue, so reader threads can run it without touching Candle's queue.

use candle::{MetalDevice, Result};
use candle_metal_kernels::metal::{Buffer, ComputePipeline, Device};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLDevice, MTLSize};
use std::time::Instant;

pub const QK_K: usize = 256;
pub const Q2K_BLOCK_BYTES: usize = 84;
pub const Q3K_BLOCK_BYTES: usize = 110;
pub const Q5K_BLOCK_BYTES: usize = 176;

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
#define QK_K 256
#define K_SCALE_SIZE 12

typedef struct {
    uint8_t scales[QK_K/16];
    uint8_t qs[QK_K/4];
    half d;
    half dmin;
} block_q2_K;

typedef struct {
    uint8_t hmask[QK_K/8];
    uint8_t qs[QK_K/4];
    uint8_t scales[12];
    half d;
} block_q3_K;

typedef struct {
    half d;
    half dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K/8];
    uint8_t qs[QK_K/2];
} block_q5_K;

static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)), uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

// 16 values of a Q3_K block (il = 0..15), as the matvec kernels read them.
void dq_q3(device const block_q3_K *xb, short il, thread float *out) {
    const half d_all = xb->d;
    device const uint8_t * q = (device const uint8_t *)xb->qs;
    device const uint8_t * h = (device const uint8_t *)xb->hmask;
    device const int8_t * scales = (device const int8_t *)xb->scales;
    q = q + 32 * (il/8) + 16 * (il&1);
    h = h + 16 * (il&1);
    uint8_t m = 1 << (il/2);
    uint16_t kmask1 = (il/4)>1 ? ((il/4)>2 ? 192 : 48) : ((il/4)>0 ? 12 : 3);
    uint16_t kmask2 = il/8 ? 0xF0 : 0x0F;
    uint16_t scale_2 = scales[il%8], scale_1 = scales[8 + il%4];
    int16_t dl_int = (il/4)&1 ? (scale_2&kmask2) | ((scale_1&kmask1) << 2) : (scale_2&kmask2) | ((scale_1&kmask1) << 4);
    float dl = il<8 ? d_all * (dl_int - 32.f) : d_all * (dl_int / 16.f - 32.f);
    const float ml = 4.f * dl;
    short il2 = (il/2) & 3;
    const half coef = il2>1 ? (il2>2 ? 1/64.h : 1/16.h) : (il2>0 ? 1/4.h : 1.h);
    const uint8_t mask = il2>1 ? (il2>2 ? 192 : 48) : (il2>0 ? 12 : 3);
    dl *= coef;
    for (int i = 0; i < 16; ++i) out[i] = dl * (q[i] & mask) - (h[i] & m ? 0 : ml);
}

typedef struct {
    half d;
    half dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K/2];
} block_q4_K;

void dq_q4(device const block_q4_K *xb, short il, thread float *out) {
    device const uchar * q = xb->qs;
    short is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il&1);
    short il2 = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il2/2, xb->scales);
    const float d   = il2 < 2 ? xb->d : xb->d / 16.h;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];
    const ushort mask = il2<2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) out[i] = dl * (q[i] & mask) - ml;
}

void dq_q5(device const block_q5_K *xb, short il, thread float *out) {
    device const uint8_t * q  = xb->qs;
    device const uint8_t * qh = xb->qh;
    short is = (il/4) * 2;
    q  = q + 32 * (il/4) + 16 * (il&1);
    qh = qh + 16 * (il&1);
    uint8_t ul = 1 << (il/2);
    short il2 = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il2/2, xb->scales);
    const float d = il2 < 2 ? xb->d : xb->d / 16.f;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];
    const ushort mask = il2<2 ? 0x0F : 0xF0;
    const float qh_val = il2<2 ? 16.f : 256.f;
    for (int i = 0; i < 16; ++i) out[i] = dl * ((q[i] & mask) + (qh[i] & ul ? qh_val : 0)) - ml;
}

// Fit 256 values to a Q2_K block: per sub-block of 16, scale = (max-min)/3 and min; the 16 scales
// and mins go to 4 bits against the block's largest; quants are 2 bits.
// Weighted fit of one sub-block of 16: given a trial (scale, min), the quants are fixed by rounding,
// then scale and min are re-solved by weighted least squares on those quants. Two rounds.
static inline void fit16(thread const float *x, thread const float *w, thread float &scale, thread float &mn) {
    for (int round = 0; round < 2; ++round) {
        float sw = 0.f, swl = 0.f, swll = 0.f, swx = 0.f, swlx = 0.f;
        for (int i = 0; i < 16; ++i) {
            int l = 0;
            if (scale > 0.f) { l = (int)((x[i] + mn) / scale + 0.5f); l = clamp(l, 0, 3); }
            const float wi = w[i];
            sw += wi; swl += wi * l; swll += wi * l * l; swx += wi * x[i]; swlx += wi * l * x[i];
        }
        // x ~ scale*l - mn : solve for (scale, mn) weighted
        const float det = sw * swll - swl * swl;
        if (det > 1e-12f) {
            const float a = (sw * swlx - swl * swx) / det;      // scale
            const float b = (swl * swlx - swll * swx) / det;    // = mn
            if (a > 0.f && b >= 0.f) { scale = a; mn = b; }
        }
    }
}

void q2k_encode_w(thread const float *x, device const float *w, uint w_offset, device block_q2_K *y) {
    float scales[16], mins[16];
    float max_scale = 0.f, max_min = 0.f;
    for (int j = 0; j < 16; ++j) {
        float lo = x[16*j], hi = x[16*j];
        for (int i = 1; i < 16; ++i) { lo = min(lo, x[16*j+i]); hi = max(hi, x[16*j+i]); }
        if (lo > 0.f) lo = 0.f;
        float sc = (hi - lo) / 3.f;
        float mn = -lo;
        float wl[16];
        for (int i = 0; i < 16; ++i) wl[i] = sqrt(max(w[w_offset + 16*j + i], 0.f)) + 1e-6f;
        fit16(x + 16*j, wl, sc, mn);
        scales[j] = sc;
        mins[j] = mn;
        max_scale = max(max_scale, scales[j]);
        max_min = max(max_min, mins[j]);
    }
    float inv_scale = max_scale > 0.f ? 15.f / max_scale : 0.f;
    float inv_min = max_min > 0.f ? 15.f / max_min : 0.f;
    uint8_t ls[16], lm[16];
    for (int j = 0; j < 16; ++j) {
        ls[j] = (uint8_t)min(15, (int)(scales[j] * inv_scale + 0.5f));
        lm[j] = (uint8_t)min(15, (int)(mins[j] * inv_min + 0.5f));
        y->scales[j] = ls[j] | (lm[j] << 4);
    }
    y->d = half(max_scale / 15.f);
    y->dmin = half(max_min / 15.f);
    const float d = float(y->d), dmin = float(y->dmin);
    uint8_t L[256];
    for (int j = 0; j < 16; ++j) {
        const float dj = d * ls[j];
        const float mj = dmin * lm[j];
        for (int i = 0; i < 16; ++i) {
            int l = 0;
            if (dj > 0.f) { l = (int)((x[16*j+i] + mj) / dj + 0.5f); l = clamp(l, 0, 3); }
            L[16*j+i] = (uint8_t)l;
        }
    }
    for (int j = 0; j < QK_K; j += 128) {
        for (int l = 0; l < 32; ++l) {
            y->qs[j/4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
        }
    }
}

void q2k_encode(thread const float *x, device block_q2_K *y) {
    float scales[16], mins[16];
    float max_scale = 0.f, max_min = 0.f;
    const float ones[16] = {1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f,1.f};
    for (int j = 0; j < 16; ++j) {
        float lo = x[16*j], hi = x[16*j];
        for (int i = 1; i < 16; ++i) { lo = min(lo, x[16*j+i]); hi = max(hi, x[16*j+i]); }
        if (lo > 0.f) lo = 0.f;
        float sc = (hi - lo) / 3.f;
        float mn = -lo;
        fit16(x + 16*j, ones, sc, mn);
        scales[j] = sc;
        mins[j] = mn;
        max_scale = max(max_scale, scales[j]);
        max_min = max(max_min, mins[j]);
    }
    float inv_scale = max_scale > 0.f ? 15.f / max_scale : 0.f;
    float inv_min = max_min > 0.f ? 15.f / max_min : 0.f;
    uint8_t ls[16], lm[16];
    for (int j = 0; j < 16; ++j) {
        ls[j] = (uint8_t)min(15, (int)(scales[j] * inv_scale + 0.5f));
        lm[j] = (uint8_t)min(15, (int)(mins[j] * inv_min + 0.5f));
        y->scales[j] = ls[j] | (lm[j] << 4);
    }
    y->d = half(max_scale / 15.f);
    y->dmin = half(max_min / 15.f);
    const float d = float(y->d), dmin = float(y->dmin);
    uint8_t L[256];
    for (int j = 0; j < 16; ++j) {
        const float dj = d * ls[j];
        const float mj = dmin * lm[j];
        for (int i = 0; i < 16; ++i) {
            int l = 0;
            if (dj > 0.f) {
                l = (int)((x[16*j+i] + mj) / dj + 0.5f);
                l = clamp(l, 0, 3);
            }
            L[16*j+i] = (uint8_t)l;
        }
    }
    for (int j = 0; j < QK_K; j += 128) {
        for (int l = 0; l < 32; ++l) {
            y->qs[j/4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
        }
    }
}

kernel void coachwhip_requant_q3k_to_q2k(
    device const block_q3_K *src [[buffer(0)]],
    device block_q2_K *dst       [[buffer(1)]],
    constant uint& n_blocks      [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n_blocks) return;
    float x[256];
    for (short il = 0; il < 16; ++il) dq_q3(src + tid, il, x + 16*il);
    q2k_encode(x, dst + tid);
}

kernel void coachwhip_requant_q3k_to_q2k_w(
    device const block_q3_K *src [[buffer(0)]],
    device block_q2_K *dst       [[buffer(1)]],
    constant uint& n_blocks      [[buffer(2)]],
    device const float *shape    [[buffer(3)]],
    constant uint& blocks_per_row [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n_blocks) return;
    float x[256];
    for (short il = 0; il < 16; ++il) dq_q3(src + tid, il, x + 16*il);
    q2k_encode_w(x, shape, (tid % blocks_per_row) * 256, dst + tid);
}

kernel void coachwhip_requant_q4k_to_q2k(
    device const block_q4_K *src [[buffer(0)]],
    device block_q2_K *dst       [[buffer(1)]],
    constant uint& n_blocks      [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n_blocks) return;
    float x[256];
    for (short il = 0; il < 16; ++il) dq_q4(src + tid, il, x + 16*il);
    q2k_encode(x, dst + tid);
}

kernel void coachwhip_requant_q4k_to_q2k_w(
    device const block_q4_K *src [[buffer(0)]],
    device block_q2_K *dst       [[buffer(1)]],
    constant uint& n_blocks      [[buffer(2)]],
    device const float *shape    [[buffer(3)]],
    constant uint& blocks_per_row [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n_blocks) return;
    float x[256];
    for (short il = 0; il < 16; ++il) dq_q4(src + tid, il, x + 16*il);
    q2k_encode_w(x, shape, (tid % blocks_per_row) * 256, dst + tid);
}

kernel void coachwhip_requant_q5k_to_q2k(
    device const block_q5_K *src [[buffer(0)]],
    device block_q2_K *dst       [[buffer(1)]],
    constant uint& n_blocks      [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n_blocks) return;
    float x[256];
    for (short il = 0; il < 16; ++il) dq_q5(src + tid, il, x + 16*il);
    q2k_encode(x, dst + tid);
}

kernel void coachwhip_requant_q5k_to_q2k_w(
    device const block_q5_K *src [[buffer(0)]],
    device block_q2_K *dst       [[buffer(1)]],
    constant uint& n_blocks      [[buffer(2)]],
    device const float *shape    [[buffer(3)]],
    constant uint& blocks_per_row [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n_blocks) return;
    float x[256];
    for (short il = 0; il < 16; ++il) dq_q5(src + tid, il, x + 16*il);
    q2k_encode_w(x, shape, (tid % blocks_per_row) * 256, dst + tid);
}
"#;

/// The shrink: two pipelines and a command queue of its own.
/// Which quantisation the arriving bytes are in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SrcKind {
    Q3K,
    Q4K,
    Q5K,
}

impl SrcKind {
    pub fn of(dtype: candle::quantized::GgmlDType) -> Option<Self> {
        use candle::quantized::GgmlDType as G;
        match dtype {
            G::Q3K => Some(Self::Q3K),
            G::Q4K => Some(Self::Q4K),
            G::Q5K => Some(Self::Q5K),
            _ => None,
        }
    }
}

pub struct Requant {
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    q3: ComputePipeline,
    q4: ComputePipeline,
    q5: ComputePipeline,
    q3w: ComputePipeline,
    q4w: ComputePipeline,
    q5w: ComputePipeline,
}

unsafe impl Send for Requant {}
unsafe impl Sync for Requant {}

impl Requant {
    pub fn new(metal: &MetalDevice) -> Result<Self> {
        let device: &Device = metal.device();
        let lib = device.new_library_with_source(SOURCE, None).map_err(candle::MetalError::from)?;
        let mut get = |name: &str| -> Result<ComputePipeline> {
            let f = lib.get_function(name, None).map_err(candle::MetalError::from)?;
            Ok(device.new_compute_pipeline_state_with_function(&f).map_err(candle::MetalError::from)?)
        };
        let q3 = get("coachwhip_requant_q3k_to_q2k")?;
        let q4 = get("coachwhip_requant_q4k_to_q2k")?;
        let q5 = get("coachwhip_requant_q5k_to_q2k")?;
        let q3w = get("coachwhip_requant_q3k_to_q2k_w")?;
        let q4w = get("coachwhip_requant_q4k_to_q2k_w")?;
        let q5w = get("coachwhip_requant_q5k_to_q2k_w")?;
        let queue = device.as_ref().newCommandQueue().ok_or_else(|| candle::Error::Msg("no Metal command queue for the shrink".into()))?;
        Ok(Self { queue, q3, q4, q5, q3w, q4w, q5w })
    }

    /// Shrink `n_blocks` super-blocks from `src` (Q3_K or Q5_K bytes, from `src_offset`) into
    /// `dst` at `dst_offset` (Q2_K bytes). Waits for the GPU, so the caller may mark the slot ready.
    pub fn run(&self, kind: SrcKind, src: &Buffer, src_offset: usize, dst: &Buffer, dst_offset: usize, n_blocks: usize) -> Result<()> {
        let cb = self.queue.commandBuffer().ok_or_else(|| candle::Error::Msg("no command buffer for the shrink".into()))?;
        let enc = cb.computeCommandEncoder().ok_or_else(|| candle::Error::Msg("no encoder for the shrink".into()))?;
        let pipe = match kind { SrcKind::Q3K => &self.q3, SrcKind::Q4K => &self.q4, SrcKind::Q5K => &self.q5 };
        enc.setComputePipelineState(pipe.as_ref());
        unsafe {
            enc.setBuffer_offset_atIndex(Some(src.as_ref()), src_offset, 0);
            enc.setBuffer_offset_atIndex(Some(dst.as_ref()), dst_offset, 1);
            let n = n_blocks as u32;
            enc.setBytes_length_atIndex(std::ptr::NonNull::new(&n as *const u32 as *mut std::ffi::c_void).unwrap(), 4, 2);
        }
        let tg = 64usize;
        let groups = (n_blocks + tg - 1) / tg;
        enc.dispatchThreadgroups_threadsPerThreadgroup(MTLSize { width: groups, height: 1, depth: 1 }, MTLSize { width: tg, height: 1, depth: 1 });
        enc.endEncoding();
        cb.commit();
        unsafe { cb.waitUntilCompleted() };
        Ok(())
    }
}

impl Requant {
    /// The shaped shrink: like `run`, but each input column's error is weighted by `shape`
    /// (one float per column, `k` of them), the running importance of that column at this layer.
    pub fn run_shaped(&self, kind: SrcKind, src: &Buffer, src_offset: usize, dst: &Buffer, dst_offset: usize, n_blocks: usize, shape: &Buffer, k: usize) -> Result<()> {
        let cb = self.queue.commandBuffer().ok_or_else(|| candle::Error::Msg("no command buffer for the shrink".into()))?;
        let enc = cb.computeCommandEncoder().ok_or_else(|| candle::Error::Msg("no encoder for the shrink".into()))?;
        let pipe = match kind { SrcKind::Q3K => &self.q3w, SrcKind::Q4K => &self.q4w, SrcKind::Q5K => &self.q5w };
        enc.setComputePipelineState(pipe.as_ref());
        unsafe {
            enc.setBuffer_offset_atIndex(Some(src.as_ref()), src_offset, 0);
            enc.setBuffer_offset_atIndex(Some(dst.as_ref()), dst_offset, 1);
            let n = n_blocks as u32;
            enc.setBytes_length_atIndex(std::ptr::NonNull::new(&n as *const u32 as *mut std::ffi::c_void).unwrap(), 4, 2);
            enc.setBuffer_offset_atIndex(Some(shape.as_ref()), 0, 3);
            let bpr = (k / QK_K) as u32;
            enc.setBytes_length_atIndex(std::ptr::NonNull::new(&bpr as *const u32 as *mut std::ffi::c_void).unwrap(), 4, 4);
        }
        let tg = 64usize;
        let groups = (n_blocks + tg - 1) / tg;
        enc.dispatchThreadgroups_threadsPerThreadgroup(MTLSize { width: groups, height: 1, depth: 1 }, MTLSize { width: tg, height: 1, depth: 1 });
        enc.endEncoding();
        cb.commit();
        unsafe { cb.waitUntilCompleted() };
        Ok(())
    }
}

/// Lab timing, run by the `gpu_shrink` test: one expert's three matrices through the shrink.
pub fn time_one_expert(metal: &MetalDevice, parts: &[(SrcKind, Vec<u8>, usize)]) -> Result<(f64, usize, usize)> {
    let rq = Requant::new(metal)?;
    let mut bytes_in = 0;
    let mut bytes_out = 0;
    let started = Instant::now();
    for (kind, data, n_blocks) in parts {
        let src = metal.new_buffer_builder().with_size(data.len()).with_label("shrink_src").build()?;
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), src.contents() as *mut u8, data.len()) };
        let dst = metal.new_buffer_builder().with_size(n_blocks * Q2K_BLOCK_BYTES).with_label("shrink_dst").build()?;
        rq.run(*kind, &src, 0, &dst, 0, *n_blocks)?;
        bytes_in += data.len();
        bytes_out += n_blocks * Q2K_BLOCK_BYTES;
    }
    Ok((started.elapsed().as_secs_f64() * 1e3, bytes_in, bytes_out))
}
