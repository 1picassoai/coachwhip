//! Lab (`COACHWHIP_FUSE=1`): fewer GPU jobs per layer on the resident expert path.
//! - `route`: softmax over the router's scores, top-k picks (first index wins ties), weights
//!   renormalised over the picks when asked: one dispatch instead of about seven.
//! - `silu_mul`: hidden = silu(gate) * up: one dispatch instead of two.
//! - `weighted_sum`: out[t][j] = sum over the k picks of w[t][e] * down[t][e][j]: one instead of three.

use candle::backend::BackendStorage;
use candle::{CustomOp1, CustomOp2, DType, Layout, MetalStorage, Result, Shape, Tensor};
use candle_metal_kernels::metal::ComputePipeline;
use candle_metal_kernels::{set_params, Output};
use objc2_metal::MTLSize;
use std::sync::{Arc, OnceLock};

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// One threadgroup per token, one thread per expert (E <= 256), k <= 16. All threads take part in
// each pick: a max-reduction over (probability, lowest index) — the order a stable descending sort gives.
kernel void cw_route(
    device const float* logits [[buffer(0)]],
    device uint* outp          [[buffer(1)]],
    constant uint& E           [[buffer(2)]],
    constant uint& K           [[buffer(3)]],
    constant uint& norm        [[buffer(4)]],
    constant uint& which       [[buffer(5)]],
    uint t    [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint tgs  [[threads_per_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]])
{
    threadgroup float rv[8];
    threadgroup uint ri[8];
    threadgroup float picked_w[16];
    threadgroup uint picked_i[16];
    device const float* x = logits + t * E;
    float v = tid < E ? x[tid] : -INFINITY;
    // softmax pieces: max and sum over the experts
    float m = simd_max(v);
    if (lane == 0) rv[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mm = rv[0]; for (uint q = 1; q < (tgs + 31) / 32; q++) mm = max(mm, rv[q]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float e = tid < E ? exp(v - mm) : 0.0f;
    float s = simd_sum(e);
    if (lane == 0) rv[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ss = 0.0f; for (uint q = 0; q < (tgs + 31) / 32; q++) ss += rv[q];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float pr = tid < E ? e / ss : -1.0f;
    bool taken = tid >= E;
    for (uint j = 0; j < K; j++) {
        float bv = taken ? -2.0f : pr;
        uint bi = tid;
        for (uint off = 16; off > 0; off >>= 1) {
            float ov = simd_shuffle_down(bv, off);
            uint oi = simd_shuffle_down(bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { rv[sg] = bv; ri[sg] = bi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float b = rv[0]; uint bix = ri[0];
            for (uint q = 1; q < (tgs + 31) / 32; q++) { if (rv[q] > b || (rv[q] == b && ri[q] < bix)) { b = rv[q]; bix = ri[q]; } }
            picked_w[j] = b; picked_i[j] = bix;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == picked_i[j]) taken = true;
    }
    if (tid == 0) {
        float tot = 0.0f; for (uint j = 0; j < K; j++) tot += picked_w[j];
        for (uint j = 0; j < K; j++) outp[t * K + j] = which == 0 ? picked_i[j] : as_type<uint>(norm ? picked_w[j] / tot : picked_w[j]);
    }
}

kernel void cw_silu_mul(
    device const float* g [[buffer(0)]],
    device const float* u [[buffer(1)]],
    device float* o       [[buffer(2)]],
    constant uint& n      [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i < n) { float x = g[i]; o[i] = x / (1.0f + exp(-x)) * u[i]; }
}

// gu: rows of [gate (I) | up (I)]; out: rows of I. out[r][j] = silu(gu[r][j]) * gu[r][I + j].
kernel void cw_silu_mul_gu(
    device const float* gu [[buffer(0)]],
    device float* o        [[buffer(1)]],
    constant uint& I       [[buffer(2)]],
    constant uint& n       [[buffer(3)]],
    uint i [[thread_position_in_grid]])
{
    if (i < n) { uint r = i / I, j = i % I; float x = gu[r * 2 * I + j]; o[i] = x / (1.0f + exp(-x)) * gu[r * 2 * I + I + j]; }
}

// First index of the largest value (what the CPU loop `if v > best` picks). One threadgroup, 1024 threads.
kernel void cw_argmax(
    device const float* x [[buffer(0)]],
    device uint* out      [[buffer(1)]],
    constant uint& n      [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]])
{
    threadgroup float bv_s[32];
    threadgroup uint bi_s[32];
    float bv = -INFINITY; uint bi = 0xFFFFFFFFu;
    for (uint i = tid; i < n; i += 1024) { float v = x[i]; if (v > bv || bi == 0xFFFFFFFFu) { bv = v; bi = i; } }
    for (uint off = 16; off > 0; off >>= 1) {
        float ov = simd_shuffle_down(bv, off); uint oi = simd_shuffle_down(bi, off);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    if (lane == 0) { bv_s[sg] = bv; bi_s[sg] = bi; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float b = bv_s[0]; uint ix = bi_s[0];
        for (uint q = 1; q < 32; q++) { if (bv_s[q] > b || (bv_s[q] == b && bi_s[q] < ix)) { b = bv_s[q]; ix = bi_s[q]; } }
        out[0] = ix;
    }
}

// One row of a Q5_K table, unpacked exactly as Candle's CPU to_float does, then rounded through half.
kernel void cw_embed_q5k(
    device const uchar* table [[buffer(0)]],
    device const uint* ids    [[buffer(1)]],
    device float* out         [[buffer(2)]],
    constant uint& hidden     [[buffer(3)]],
    constant uint& row_bytes  [[buffer(4)]],
    uint2 g [[thread_position_in_grid]])
{
    #pragma clang fp contract(off)
    uint e = g.x, t = g.y;
    if (e >= hidden) return;
    device const uchar* blk = table + (ulong)ids[t] * row_bytes + (e / 256) * 176;
    uint pos = e % 256;
    float d = float(as_type<half>(*(device const ushort*)(blk)));
    float mn = float(as_type<half>(*(device const ushort*)(blk + 2)));
    device const uchar* sc = blk + 4;
    device const uchar* qh = blk + 16;
    device const uchar* ql = blk + 48;
    uint chunk = pos / 64, hi = (pos % 64) / 32, l = pos % 32;
    uint j = 2 * chunk + hi;
    uint s, m;
    if (j < 4) { s = sc[j] & 63; m = sc[j + 4] & 63; }
    else { s = (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4); m = (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4); }
    float dd = d * float(s);
    float mm = mn * float(m);
    uchar q = ql[chunk * 32 + l];
    uint u = (hi ? 2u : 1u) << (2 * chunk);
    float add = (qh[l] & u) ? 16.0f : 0.0f;
    float qv = float(hi ? (q >> 4) : (q & 0xF)) + add;
    float prod = dd * qv;
    float v = prod - mm;
    out[t * hidden + e] = float(half(v));
}

// y[r] = dot(w[r], x), w rows f16, x f32; one simdgroup per row, 4 rows per threadgroup.
kernel void cw_mv_f16(
    device const half* w  [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* y       [[buffer(2)]],
    constant uint& k      [[buffer(3)]],
    constant uint& rows   [[buffer(4)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]])
{
    uint r = gid * 4 + sg;
    if (r >= rows) return;
    device const half4* w4 = (device const half4*)(w + (ulong)r * k);
    device const float4* x4 = (device const float4*)x;
    float acc = 0.0f;
    for (uint i = lane; i < k / 4; i += 32) acc += dot(float4(w4[i]), x4[i]);
    acc = simd_sum(acc);
    if (lane == 0) y[r] = acc;
}

// Piles: thread e counts the (slot, word) picks of expert e, finds where its pile starts, writes it.
// heads[3e] = e, heads[3e+1] = first pair, heads[3e+2] = count. Pairs keep the words' order.
kernel void cw_pile(
    device const uint* ids  [[buffer(0)]],
    device ushort2* piles   [[buffer(1)]],
    device uint* heads      [[buffer(2)]],
    constant uint& n_ids    [[buffer(3)]],
    constant uint& nei0     [[buffer(4)]],
    constant uint& experts  [[buffer(5)]],
    uint e [[thread_position_in_threadgroup]])
{
    threadgroup uint counts[256];
    uint c = 0;
    if (e < experts) { for (uint i = 0; i < n_ids; i++) if (ids[i] == e) c++; }
    counts[e] = c;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e >= experts) return;
    uint off = 0;
    for (uint j = 0; j < e; j++) off += counts[j];
    heads[3 * e] = e; heads[3 * e + 1] = off; heads[3 * e + 2] = c;
    uint w = off;
    for (uint i = 0; i < n_ids; i++) if (ids[i] == e) { piles[w] = ushort2(i % nei0, i / nei0); w++; }
}

// The small piles' words: thread e copies its pile when it has fewer than min_cnt words.
kernel void cw_small(
    device const ushort2* piles [[buffer(0)]],
    device const uint* heads    [[buffer(1)]],
    device ushort2* outp        [[buffer(2)]],
    device uint* npairs         [[buffer(3)]],
    constant uint& experts      [[buffer(4)]],
    constant uint& min_cnt      [[buffer(5)]],
    uint e [[thread_position_in_threadgroup]])
{
    threadgroup uint counts[256];
    uint c = 0, first = 0;
    if (e < experts) { c = heads[3 * e + 2]; first = heads[3 * e + 1]; if (c >= min_cnt) c = 0; }
    counts[e] = c;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint off = 0;
    for (uint j = 0; j < e; j++) off += counts[j];
    for (uint i = 0; i < c; i++) outp[off + i] = piles[first + i];
    if (e == 255) { uint tot = off + c; npairs[0] = tot; }
}

// Top-k, one pass: group g takes elements [g*4096, (g+1)*4096), 256 threads x 16 values in registers,
// and writes its k best (value, index) in order. has_ids: the input carries its own indices (pass 2).
kernel void cw_topk(
    device const float* xv [[buffer(0)]],
    device const uint* xi  [[buffer(1)]],
    device float* ov       [[buffer(2)]],
    device uint* oi        [[buffer(3)]],
    constant uint& n       [[buffer(4)]],
    constant uint& k       [[buffer(5)]],
    constant uint& has_ids [[buffer(6)]],
    uint g    [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]])
{
    const uint T = 256;
    const uint start = g * T * 16;
    float v[16]; uint id[16];
    for (uint j = 0; j < 16; j++) {
        uint i = start + j * T + tid;
        if (i < n) { v[j] = xv[i]; id[j] = has_ids ? xi[i] : i; } else { v[j] = -INFINITY; id[j] = 0xFFFFFFFFu; }
    }
    threadgroup float bv[8];
    threadgroup uint bi[8];
    threadgroup uint win;
    for (uint r = 0; r < k; r++) {
        float best = -INFINITY; uint bid = 0xFFFFFFFFu; uint bj = 99;
        for (uint j = 0; j < 16; j++) {
            if (id[j] != 0xFFFFFFFFu && (bj == 99 || v[j] > best || (v[j] == best && id[j] < bid))) { best = v[j]; bid = id[j]; bj = j; }
        }
        float sb = best; uint si = bid;
        for (uint off = 16; off > 0; off >>= 1) {
            float o_v = simd_shuffle_down(sb, off); uint o_i = simd_shuffle_down(si, off);
            if (o_i != 0xFFFFFFFFu && (si == 0xFFFFFFFFu || o_v > sb || (o_v == sb && o_i < si))) { sb = o_v; si = o_i; }
        }
        if (lane == 0) { bv[sg] = sb; bi[sg] = si; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float b = bv[0]; uint ix = bi[0];
            for (uint q = 1; q < 8; q++) {
                if (bi[q] != 0xFFFFFFFFu && (ix == 0xFFFFFFFFu || bv[q] > b || (bv[q] == b && bi[q] < ix))) { b = bv[q]; ix = bi[q]; }
            }
            ov[g * k + r] = b; oi[g * k + r] = ix; win = ix;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint w = win;
        if (bj < 16 && bid == w) { v[bj] = -INFINITY; id[bj] = 0xFFFFFFFFu; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// down: (T, K, H) rows; w: (T, K); out: (T, H). One thread per (t, j).
kernel void cw_weighted_sum(
    device const float* down [[buffer(0)]],
    device const float* w    [[buffer(1)]],
    device float* o          [[buffer(2)]],
    constant uint& T         [[buffer(3)]],
    constant uint& K         [[buffer(4)]],
    constant uint& H         [[buffer(5)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= T * H) return;
    uint t = i / H, j = i % H;
    float acc = 0.0f;
    for (uint e = 0; e < K; e++) acc += down[(t * K + e) * H + j] * w[t * K + e];
    o[i] = acc;
}
"#;

fn pipe(dev: &candle::MetalDevice, name: &str) -> Result<Arc<ComputePipeline>> {
    static LIB: OnceLock<std::sync::Mutex<std::collections::HashMap<String, Arc<ComputePipeline>>>> = OnceLock::new();
    let lib = LIB.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut g = lib.lock().unwrap();
    if let Some(p) = g.get(name) {
        return Ok(p.clone());
    }
    let l = dev.device().new_library_with_source(SOURCE, None).map_err(candle::MetalError::from)?;
    let f = l.get_function(name, None).map_err(candle::MetalError::from)?;
    let p = Arc::new(dev.device().new_compute_pipeline_state_with_function(&f).map_err(candle::MetalError::from)?);
    g.insert(name.to_string(), p.clone());
    Ok(p)
}

pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("COACHWHIP_FUSE").map(|v| v == "1").unwrap_or(false))
}

fn need(st: &MetalStorage, l: &Layout, what: &str) -> Result<()> {
    if st.dtype() != DType::F32 || l.contiguous_offsets().is_none() {
        candle::bail!("cw fuse: {what} must be contiguous f32")
    }
    Ok(())
}

/// logits (T, E) -> ids (T, K) u32 (`which` 0) or weights (T, K) f32 (`which` 1). Two dispatches
/// replace about seven; the picks never leave the GPU.
struct Route { k: usize, norm: bool, which: u32 }
impl CustomOp1 for Route {
    fn name(&self) -> &'static str { "cw-route" }
    fn cpu_fwd(&self, _: &candle::CpuStorage, _: &Layout) -> Result<(candle::CpuStorage, Shape)> { candle::bail!("metal only") }
    fn metal_fwd(&self, x: &MetalStorage, l: &Layout) -> Result<(MetalStorage, Shape)> {
        need(x, l, "logits")?;
        let (t, e) = l.shape().dims2()?;
        if e > 256 || self.k > 16 { candle::bail!("cw-route: {e} experts / k {}", self.k) }
        let dev = x.device().clone();
        let n = t * self.k;
        let dt = if self.which == 0 { DType::U32 } else { DType::F32 };
        let dst = crate::arena::take_for(&dev, n, dt, "cw-route")?;
        let p = pipe(&dev, "cw_route")?;
        {
            let enc = dev.command_encoder()?;
            let enc = enc.as_ref();
            enc.set_compute_pipeline_state(&p);
            let (e_u, k_u, nm, wh) = (e as u32, self.k as u32, self.norm as u32, self.which);
            set_params!(enc, ((x.buffer(), l.start_offset() * 4), Output::with_offset(&dst, 0), e_u, k_u, nm, wh));
            enc.dispatch_thread_groups(MTLSize { width: t, height: 1, depth: 1 }, MTLSize { width: 256, height: 1, depth: 1 });
        }
        Ok((MetalStorage::new(dst, dev, n, dt), Shape::from((t, self.k))))
    }
}

pub fn route(logits: &Tensor, k: usize, norm: bool) -> Result<(Tensor, Tensor)> {
    let lg = logits.to_dtype(DType::F32)?.contiguous()?;
    let ids = lg.apply_op1_no_bwd(&Route { k, norm, which: 0 })?;
    let w = lg.apply_op1_no_bwd(&Route { k, norm, which: 1 })?;
    Ok((ids, w))
}

struct SiluMul;
impl CustomOp2 for SiluMul {
    fn name(&self) -> &'static str { "cw-silu-mul" }
    fn cpu_fwd(&self, _: &candle::CpuStorage, _: &Layout, _: &candle::CpuStorage, _: &Layout) -> Result<(candle::CpuStorage, Shape)> { candle::bail!("metal only") }
    fn metal_fwd(&self, g: &MetalStorage, gl: &Layout, u: &MetalStorage, ul: &Layout) -> Result<(MetalStorage, Shape)> {
        need(g, gl, "gate")?; need(u, ul, "up")?;
        let n = gl.shape().elem_count();
        let dev = g.device().clone();
        let dst = crate::arena::take_for(&dev, n, DType::F32, "cw-silu-mul")?;
        let p = pipe(&dev, "cw_silu_mul")?;
        {
            let enc = dev.command_encoder()?;
            let enc = enc.as_ref();
            enc.set_compute_pipeline_state(&p);
            let n_u = n as u32;
            set_params!(enc, ((g.buffer(), gl.start_offset() * 4), (u.buffer(), ul.start_offset() * 4), Output::with_offset(&dst, 0), n_u));
            enc.dispatch_thread_groups(MTLSize { width: n.div_ceil(256), height: 1, depth: 1 }, MTLSize { width: 256, height: 1, depth: 1 });
        }
        Ok((MetalStorage::new(dst, dev, n, DType::F32), gl.shape().clone()))
    }
}
pub fn silu_mul(g: &Tensor, u: &Tensor) -> Result<Tensor> { g.contiguous()?.apply_op2_no_bwd(&u.contiguous()?, &SiluMul) }

struct WSum;
impl CustomOp2 for WSum {
    fn name(&self) -> &'static str { "cw-wsum" }
    fn cpu_fwd(&self, _: &candle::CpuStorage, _: &Layout, _: &candle::CpuStorage, _: &Layout) -> Result<(candle::CpuStorage, Shape)> { candle::bail!("metal only") }
    fn metal_fwd(&self, d: &MetalStorage, dl: &Layout, w: &MetalStorage, wl: &Layout) -> Result<(MetalStorage, Shape)> {
        need(d, dl, "down")?; need(w, wl, "weights")?;
        let (t, k, h) = dl.shape().dims3()?;
        let dev = d.device().clone();
        let dst = crate::arena::take_for(&dev, t * h, DType::F32, "cw-wsum")?;
        let p = pipe(&dev, "cw_weighted_sum")?;
        {
            let enc = dev.command_encoder()?;
            let enc = enc.as_ref();
            enc.set_compute_pipeline_state(&p);
            let (t_u, k_u, h_u) = (t as u32, k as u32, h as u32);
            set_params!(enc, ((d.buffer(), dl.start_offset() * 4), (w.buffer(), wl.start_offset() * 4), Output::with_offset(&dst, 0), t_u, k_u, h_u));
            enc.dispatch_thread_groups(MTLSize { width: (t * h).div_ceil(256), height: 1, depth: 1 }, MTLSize { width: 256, height: 1, depth: 1 });
        }
        Ok((MetalStorage::new(dst, dev, t * h, DType::F32), Shape::from((t, h))))
    }
}
pub fn weighted_sum(down: &Tensor, w: &Tensor) -> Result<Tensor> { down.contiguous()?.apply_op2_no_bwd(&w.contiguous()?, &WSum) }

struct SiluMulGu;
impl CustomOp1 for SiluMulGu {
    fn name(&self) -> &'static str { "cw-silu-mul-gu" }
    fn cpu_fwd(&self, _: &candle::CpuStorage, _: &Layout) -> Result<(candle::CpuStorage, Shape)> { candle::bail!("metal only") }
    fn metal_fwd(&self, g: &MetalStorage, gl: &Layout) -> Result<(MetalStorage, Shape)> {
        need(g, gl, "gate|up")?;
        let mut dims = gl.shape().dims().to_vec();
        let two_i = *dims.last().unwrap();
        let i = two_i / 2;
        *dims.last_mut().unwrap() = i;
        let n = gl.shape().elem_count() / 2;
        let dev = g.device().clone();
        let dst = crate::arena::take_for(&dev, n, DType::F32, "cw-silu-mul-gu")?;
        let p = pipe(&dev, "cw_silu_mul_gu")?;
        {
            let enc = dev.command_encoder()?;
            let enc = enc.as_ref();
            enc.set_compute_pipeline_state(&p);
            let (i_u, n_u) = (i as u32, n as u32);
            set_params!(enc, ((g.buffer(), gl.start_offset() * 4), Output::with_offset(&dst, 0), i_u, n_u));
            enc.dispatch_thread_groups(MTLSize { width: n.div_ceil(256), height: 1, depth: 1 }, MTLSize { width: 256, height: 1, depth: 1 });
        }
        Ok((MetalStorage::new(dst, dev, n, DType::F32), Shape::from(dims)))
    }
}
/// gu (.., 2I) with gate in the first half of each row and up in the second -> silu(gate) * up (.., I).
pub fn silu_mul_gu(gu: &Tensor) -> Result<Tensor> { gu.contiguous()?.apply_op1_no_bwd(&SiluMulGu) }

pub fn addnorm_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| on() && std::env::var("COACHWHIP_ADDNORM").map(|v| v == "1").unwrap_or(false))
}

/// (a + b (+ c), rms_norm(a + b (+ c)) * alpha) in one dispatch, Candle's rms_norm block size.
pub fn add_norm(a: &Tensor, b: &Tensor, c: Option<&Tensor>, alpha: &Tensor, eps: f32) -> Result<(Tensor, Tensor)> {
    use candle::Storage;
    let a = a.contiguous()?;
    let b = b.contiguous()?;
    let c = match c { Some(c) => Some(c.contiguous()?), None => None };
    let alpha = alpha.contiguous()?;
    let shape = a.shape().clone();
    let h = *shape.dims().last().unwrap();
    let numel = shape.elem_count();
    let (sa, la) = a.storage_and_layout();
    let (sb, lb) = b.storage_and_layout();
    let (sw, lw) = alpha.storage_and_layout();
    let cs = c.as_ref().map(|c| c.storage_and_layout());
    let (Storage::Metal(ma), Storage::Metal(mb), Storage::Metal(mw)) = (&*sa, &*sb, &*sw) else { candle::bail!("add_norm: metal only") };
    if ma.dtype() != DType::F32 || mb.dtype() != DType::F32 || mw.dtype() != DType::F32 { candle::bail!("add_norm: f32 only") }
    let (mc, lc_off) = match &cs {
        Some((sc, lc)) => match &**sc { Storage::Metal(m) => (m.buffer().clone(), lc.start_offset() * 4), _ => candle::bail!("add_norm: metal only") },
        None => (ma.buffer().clone(), la.start_offset() * 4),
    };
    let dev = ma.device().clone();
    let sum = dev.new_buffer_builder().with_size_for(numel, DType::F32).with_label("cw-add").build()?;
    let dst = dev.new_buffer_builder().with_size_for(numel, DType::F32).with_label("cw-add-norm").build()?;
    let p = dev.kernels().load_pipeline(dev.device(), candle_metal_kernels::source::Source::Reduce, "add_rms_norm_f32").map_err(candle::MetalError::from)?;
    let rp = dev.kernels().load_pipeline(dev.device(), candle_metal_kernels::source::Source::Reduce, "rmsnorm_f32").map_err(candle::MetalError::from)?;
    let width = p.max_total_threads_per_threadgroup().min((h / 2).next_power_of_two());
    let rwidth = rp.max_total_threads_per_threadgroup().min((h / 2).next_power_of_two());
    if width != rwidth { candle::bail!("add_norm: block {width} differs from rms_norm's {rwidth}") }
    {
        let enc = dev.command_encoder()?;
        let enc = enc.as_ref();
        enc.set_compute_pipeline_state(&p);
        let (n_u, h_u, has_c) = (numel as u32, h as u32, cs.is_some() as u32);
        set_params!(enc, (n_u, h_u, (ma.buffer(), la.start_offset() * 4), (mb.buffer(), lb.start_offset() * 4), (&mc, lc_off), has_c,
            Output::with_offset(&sum, 0), Output::with_offset(&dst, 0), (mw.buffer(), lw.start_offset() * 4), eps));
        enc.dispatch_thread_groups(MTLSize { width: numel / h, height: 1, depth: 1 }, MTLSize { width, height: 1, depth: 1 });
    }
    drop((sa, sb, sw, cs));
    let st = |buf| candle::Storage::Metal(MetalStorage::new(buf, dev.clone(), numel, DType::F32));
    let s_t = Tensor::from_storage(st(sum), shape.clone(), candle::op::BackpropOp::none(), false);
    let d_t = Tensor::from_storage(st(dst), shape, candle::op::BackpropOp::none(), false);
    Ok((s_t, d_t))
}

pub fn gpupick_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("COACHWHIP_GPUPICK").map(|v| v == "1").unwrap_or(false))
}

struct Argmax;
impl CustomOp1 for Argmax {
    fn name(&self) -> &'static str { "cw-argmax" }
    fn cpu_fwd(&self, _: &candle::CpuStorage, _: &Layout) -> Result<(candle::CpuStorage, Shape)> { candle::bail!("metal only") }
    fn metal_fwd(&self, x: &MetalStorage, l: &Layout) -> Result<(MetalStorage, Shape)> {
        need(x, l, "logits")?;
        let n = l.shape().elem_count();
        let dev = x.device().clone();
        let dst = dev.new_buffer_builder().with_size_for(1, DType::U32).with_label("cw-argmax").build()?;
        let p = pipe(&dev, "cw_argmax")?;
        {
            let enc = dev.command_encoder()?;
            let enc = enc.as_ref();
            enc.set_compute_pipeline_state(&p);
            let n_u = n as u32;
            set_params!(enc, ((x.buffer(), l.start_offset() * 4), Output::with_offset(&dst, 0), n_u));
            enc.dispatch_thread_groups(MTLSize { width: 1, height: 1, depth: 1 }, MTLSize { width: 1024, height: 1, depth: 1 });
        }
        Ok((MetalStorage::new(dst, dev, 1, DType::U32), Shape::from((1, 1))))
    }
}
/// The first index of the largest logit, as a (1, 1) u32 tensor that stays on the GPU.
pub fn argmax(logits: &Tensor) -> Result<Tensor> { logits.flatten_all()?.to_dtype(DType::F32)?.contiguous()?.apply_op1_no_bwd(&Argmax) }

struct EmbedQ5k { hidden: usize, row_bytes: usize }
impl CustomOp2 for EmbedQ5k {
    fn name(&self) -> &'static str { "cw-embed-q5k" }
    fn cpu_fwd(&self, _: &candle::CpuStorage, _: &Layout, _: &candle::CpuStorage, _: &Layout) -> Result<(candle::CpuStorage, Shape)> { candle::bail!("metal only") }
    fn metal_fwd(&self, t: &MetalStorage, tl: &Layout, ids: &MetalStorage, il: &Layout) -> Result<(MetalStorage, Shape)> {
        let n = il.shape().elem_count();
        let dev = t.device().clone();
        let elems = n * self.hidden;
        let dst = dev.new_buffer_builder().with_size_for(elems, DType::F32).with_label("cw-embed").build()?;
        let p = pipe(&dev, "cw_embed_q5k")?;
        {
            let enc = dev.command_encoder()?;
            let enc = enc.as_ref();
            enc.set_compute_pipeline_state(&p);
            let (h_u, rb_u) = (self.hidden as u32, self.row_bytes as u32);
            set_params!(enc, ((t.buffer(), tl.start_offset()), (ids.buffer(), il.start_offset() * 4), Output::with_offset(&dst, 0), h_u, rb_u));
            enc.dispatch_thread_groups(MTLSize { width: self.hidden.div_ceil(256), height: n, depth: 1 }, MTLSize { width: 256, height: 1, depth: 1 });
        }
        Ok((MetalStorage::new(dst, dev, elems, DType::F32), Shape::from((n, self.hidden))))
    }
}
/// Rows `ids` (u32, on the GPU) of a packed Q5_K word table held on the GPU as bytes.
pub fn embed_q5k(table: &Tensor, ids: &Tensor, hidden: usize, row_bytes: usize) -> Result<Tensor> {
    table.apply_op2_no_bwd(&ids.flatten_all()?.contiguous()?, &EmbedQ5k { hidden, row_bytes })
}

pub fn router16_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("COACHWHIP_ROUTER16").map(|v| v == "1").unwrap_or(false))
}

struct MvF16;
impl CustomOp2 for MvF16 {
    fn name(&self) -> &'static str { "cw-mv-f16" }
    fn cpu_fwd(&self, _: &candle::CpuStorage, _: &Layout, _: &candle::CpuStorage, _: &Layout) -> Result<(candle::CpuStorage, Shape)> { candle::bail!("metal only") }
    fn metal_fwd(&self, w: &MetalStorage, wl: &Layout, x: &MetalStorage, xl: &Layout) -> Result<(MetalStorage, Shape)> {
        let (rows, k) = wl.shape().dims2()?;
        if w.dtype() != DType::F16 || x.dtype() != DType::F32 || k % 4 != 0 || xl.shape().elem_count() != k { candle::bail!("cw-mv-f16: shapes") }
        let dev = w.device().clone();
        let dst = dev.new_buffer_builder().with_size_for(rows, DType::F32).with_label("cw-mv-f16").build()?;
        let p = pipe(&dev, "cw_mv_f16")?;
        {
            let enc = dev.command_encoder()?;
            let enc = enc.as_ref();
            enc.set_compute_pipeline_state(&p);
            let (k_u, r_u) = (k as u32, rows as u32);
            set_params!(enc, ((w.buffer(), wl.start_offset() * 2), (x.buffer(), xl.start_offset() * 4), Output::with_offset(&dst, 0), k_u, r_u));
            enc.dispatch_thread_groups(MTLSize { width: rows.div_ceil(4), height: 1, depth: 1 }, MTLSize { width: 32, height: 4, depth: 1 });
        }
        Ok((MetalStorage::new(dst, dev, rows, DType::F32), Shape::from((1, rows))))
    }
}
/// (1, k) f32 times an f16 (rows, k) table -> (1, rows) f32.
pub fn mv_f16(w16: &Tensor, x: &Tensor) -> Result<Tensor> { w16.apply_op2_no_bwd(&x.to_dtype(DType::F32)?.contiguous()?, &MvF16) }

pub fn group3_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("COACHWHIP_GROUP").map(|v| v == "3").unwrap_or(false))
}

/// The piles for a batch of picks, made on the GPU: (piles, heads) buffers.
pub fn piles(dev: &candle::MetalDevice, ids: &candle_metal_kernels::metal::Buffer, ids_offset: usize, n_ids: usize, nei0: usize, experts: usize)
    -> Result<(Arc<candle_metal_kernels::metal::Buffer>, Arc<candle_metal_kernels::metal::Buffer>)> {
    if experts > 256 { candle::bail!("cw-pile: {experts} experts") }
    let pb = crate::arena::take_for(&dev, n_ids, DType::U32, "cw-piles")?;
    let hb = crate::arena::take_for(&dev, 3 * experts, DType::U32, "cw-heads")?;
    let p = pipe(dev, "cw_pile")?;
    {
        let enc = dev.command_encoder()?;
        let enc = enc.as_ref();
        enc.set_compute_pipeline_state(&p);
        let (n_u, k_u, e_u) = (n_ids as u32, nei0 as u32, experts as u32);
        set_params!(enc, ((ids, ids_offset), Output::with_offset(&pb, 0), Output::with_offset(&hb, 0), n_u, k_u, e_u));
        enc.dispatch_thread_groups(MTLSize { width: 1, height: 1, depth: 1 }, MTLSize { width: 256, height: 1, depth: 1 });
    }
    Ok((pb, hb))
}

struct TopK { k: usize }
impl CustomOp1 for TopK {
    fn name(&self) -> &'static str { "cw-topk" }
    fn cpu_fwd(&self, _: &candle::CpuStorage, _: &Layout) -> Result<(candle::CpuStorage, Shape)> { candle::bail!("metal only") }
    fn metal_fwd(&self, x: &MetalStorage, l: &Layout) -> Result<(MetalStorage, Shape)> {
        need(x, l, "scores")?;
        let n = l.shape().elem_count();
        let k = self.k;
        if k == 0 || k > 64 { candle::bail!("cw-topk: k {k}") }
        let groups = n.div_ceil(4096);
        let n2 = groups * k;
        if n2 > 4096 { candle::bail!("cw-topk: {n} scores too many for two passes") }
        let dev = x.device().clone();
        let v1 = dev.new_buffer_builder().with_size_for(n2, DType::F32).with_label("cw-topk-v1").build()?;
        let i1 = dev.new_buffer_builder().with_size_for(n2, DType::U32).with_label("cw-topk-i1").build()?;
        let dst = dev.new_buffer_builder().with_size_for(2 * k, DType::U32).with_label("cw-topk").build()?;
        let p = pipe(&dev, "cw_topk")?;
        {
            let enc = dev.command_encoder()?;
            let enc = enc.as_ref();
            enc.set_compute_pipeline_state(&p);
            let (n_u, k_u, no) = (n as u32, k as u32, 0u32);
            set_params!(enc, ((x.buffer(), l.start_offset() * 4), (x.buffer(), l.start_offset() * 4), Output::with_offset(&v1, 0), Output::with_offset(&i1, 0), n_u, k_u, no));
            enc.dispatch_thread_groups(MTLSize { width: groups, height: 1, depth: 1 }, MTLSize { width: 256, height: 1, depth: 1 });
        }
        {
            let enc = dev.command_encoder()?;
            let enc = enc.as_ref();
            enc.set_compute_pipeline_state(&p);
            let (n_u, k_u, yes) = (n2 as u32, k as u32, 1u32);
            set_params!(enc, ((&*v1, 0usize), (&*i1, 0usize), Output::with_offset(&dst, 0), Output::with_offset(&dst, k * 4), n_u, k_u, yes));
            enc.dispatch_thread_groups(MTLSize { width: 1, height: 1, depth: 1 }, MTLSize { width: 256, height: 1, depth: 1 });
        }
        Ok((MetalStorage::new(dst, dev, 2 * k, DType::U32), Shape::from(2 * k)))
    }
}
/// The k best scores and their word ids, best first (first id on equal scores); only 2k numbers come back.
pub fn topk(logits: &Tensor, k: usize) -> Result<(Vec<f32>, Vec<u32>)> {
    let x = logits.flatten_all()?.to_dtype(DType::F32)?.contiguous()?;
    let out: Vec<u32> = x.apply_op1_no_bwd(&TopK { k })?.to_vec1()?;
    let vals = out[..k].iter().map(|b| f32::from_bits(*b)).collect();
    Ok((vals, out[k..].to_vec()))
}

/// From the piles: the list of words in piles smaller than `min_cnt`, and its length (on the GPU).
pub fn small_pairs(dev: &candle::MetalDevice, piles: &candle_metal_kernels::metal::Buffer, heads: &candle_metal_kernels::metal::Buffer, n_pairs: usize, experts: usize, min_cnt: u32)
    -> Result<(Arc<candle_metal_kernels::metal::Buffer>, Arc<candle_metal_kernels::metal::Buffer>)> {
    let ob = dev.new_buffer_builder().with_size_for(n_pairs.max(1), DType::U32).with_label("cw-small").build()?;
    let nb = dev.new_buffer_builder().with_size_for(1, DType::U32).with_label("cw-small-n").build()?;
    let p = pipe(dev, "cw_small")?;
    {
        let enc = dev.command_encoder()?;
        let enc = enc.as_ref();
        enc.set_compute_pipeline_state(&p);
        let (e_u, m_u) = (experts as u32, min_cnt);
        set_params!(enc, ((piles, 0usize), (heads, 0usize), Output::with_offset(&ob, 0), Output::with_offset(&nb, 0), e_u, m_u));
        enc.dispatch_thread_groups(MTLSize { width: 1, height: 1, depth: 1 }, MTLSize { width: 256, height: 1, depth: 1 });
    }
    Ok((ob, nb))
}
