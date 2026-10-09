//! The parallel path (S4): the GPU never drains for the CPU inside a word.
//!
//! Per layer, a tiny kernel copies the router's picks (and the early guess) into memory the CPU
//! can read, and its command buffer is committed at once so it completes the moment the GPU
//! reaches it. The expert maths is queued behind a wait on a shared event. A service thread sees
//! the picks, places and reads the experts, updates the GPU's expert-to-slot table and signals
//! the event; a second kernel then looks the slots up on the GPU and the maths runs. The CPU
//! never waits for the queue inside a word; the GPU waits only for bytes.

use candle::backend::BackendStorage;
use candle::{CustomOp1, CustomOp2, DType, Layout, MetalStorage, Result, Shape};
use candle_metal_kernels::metal::{Buffer, Commands, ComputePipeline, Device};
use candle_metal_kernels::{set_params, Output};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLDevice, MTLEvent, MTLSharedEvent, MTLSize};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

pub const NO_SLOT: u32 = u32::MAX;

const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void coachwhip_publish(
    device const uint* ids      [[buffer(0)]],
    device const uint* guess    [[buffer(1)]],
    device uint* picks          [[buffer(2)]],
    device atomic_uint* flag    [[buffer(3)]],
    device uint* out            [[buffer(4)]],
    constant uint& n_ids        [[buffer(5)]],
    constant uint& n_guess      [[buffer(6)]],
    constant uint& tag          [[buffer(7)]],
    uint tid [[thread_position_in_grid]],
    uint tgs [[threads_per_threadgroup]])
{
    // Each pick carries its job's stamp in the top half of the same word, so a reader can never
    // take a mix of this word's picks and the last word's left in the buffer.
    for (uint i = tid; i < n_ids; i += tgs) picks[i] = (tag << 16) | ids[i];
    for (uint i = tid; i < n_guess; i += tgs) picks[n_ids + i] = (tag << 16) | guess[i];
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
        atomic_store_explicit(flag, 1u + tag, memory_order_relaxed);
        out[0] = 1u;
    }
}

kernel void coachwhip_resolve(
    device const uint* ids      [[buffer(0)]],
    device const uint* table    [[buffer(1)]],
    device uint* slots          [[buffer(2)]],
    device atomic_uint* missing [[buffer(3)]],
    constant uint& n_ids        [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint tgs [[threads_per_threadgroup]])
{
    for (uint i = tid; i < n_ids; i += tgs) {
        uint s = table[ids[i]];
        if (s == 0xFFFFFFFFu) {
            atomic_fetch_add_explicit(missing, 1u, memory_order_relaxed);
            s = 0u;
        }
        slots[i] = s;
    }
}
"#;

/// The GPU's view of one layer's bank, in memory both sides can touch.
pub struct GpuBank {
    table: Arc<Buffer>,
    ready: Arc<Buffer>,
    picks: Arc<Buffer>,
    flag: Arc<Buffer>,
    /// How many picks the resolve kernel found no slot for (should stay 0).
    missing: Arc<Buffer>,
    pub experts: usize,
    pub slots: usize,
    pub max_picks: usize,
}

fn atom(buffer: &Buffer, i: usize) -> &AtomicU32 {
    unsafe { AtomicU32::from_ptr((buffer.contents() as *mut u32).add(i)) }
}

impl GpuBank {
    pub fn new(metal: &candle::MetalDevice, experts: usize, slots: usize, max_picks: usize) -> Result<Self> {
        // Plain shared buffers, zeroed from the CPU: no GPU work queued, so making a bank never
        // has to wait for the queue (which, mid-word, may be waiting on us).
        let make = |n: usize, label: &str| -> Result<Arc<Buffer>> {
            let buf = metal.new_buffer_builder().with_size_for(n, DType::U32).with_label(label).build()?;
            unsafe { std::ptr::write_bytes(buf.contents() as *mut u8, 0, n * 4) };
            Ok(buf)
        };
        let table = make(experts, "s4_table")?;
        let ready = make(slots, "s4_ready")?;
        let picks = make(max_picks, "s4_picks")?;
        let flag = make(1, "s4_flag")?;
        let missing = make(1, "s4_missing")?;
        let bank = Self { table, ready, picks, flag, missing, experts, slots, max_picks };
        for e in 0..experts {
            bank.clear(e as u32);
        }
        for s in 0..slots {
            bank.set_ready(s, false);
        }
        Ok(bank)
    }

    pub fn set_slot(&self, expert: u32, slot: usize) {
        atom(&self.table, expert as usize).store(slot as u32, Ordering::Release);
    }

    pub fn clear(&self, expert: u32) {
        atom(&self.table, expert as usize).store(NO_SLOT, Ordering::Release);
    }

    pub fn set_ready(&self, slot: usize, ready: bool) {
        atom(&self.ready, slot).store(ready as u32, Ordering::Release);
    }

    pub fn reset_flag(&self) {
        atom(&self.flag, 0).store(0, Ordering::Release);
    }

    /// 0 = the picks are not out yet; otherwise 1 + the sum of the picks.
    pub fn flag(&self) -> u32 {
        atom(&self.flag, 0).load(Ordering::SeqCst)
    }

    pub fn picks(&self, n: usize) -> Vec<u32> {
        (0..n).map(|i| atom(&self.picks, i).load(Ordering::SeqCst)).collect()
    }

    /// The picks of job `tag`, once every one of them carries that stamp; None until then.
    pub fn picks_tagged(&self, n: usize, tag: u32) -> Option<Vec<u32>> {
        if self.flag() != 1 + tag {
            return None;
        }
        let picks = self.picks(n);
        if picks.iter().all(|&p| p >> 16 == tag) { Some(picks.iter().map(|&p| p & 0xFFFF).collect()) } else { None }
    }

    /// Picks the resolve kernel found no slot for so far.
    pub fn missing(&self) -> u32 {
        atom(&self.missing, 0).load(Ordering::SeqCst)
    }
}

/// The two kernels, compiled once per device.
pub struct Pipelines {
    pub publish: Arc<ComputePipeline>,
    pub resolve: Arc<ComputePipeline>,
}

pub fn pipelines(device: &Device) -> Result<Pipelines> {
    let lib = device.new_library_with_source(SOURCE, None).map_err(candle::MetalError::from)?;
    let mut get = |name: &str| -> Result<Arc<ComputePipeline>> {
        let f = lib.get_function(name, None).map_err(candle::MetalError::from)?;
        Ok(Arc::new(device.new_compute_pipeline_state_with_function(&f).map_err(candle::MetalError::from)?))
    };
    Ok(Pipelines { publish: get("coachwhip_publish")?, resolve: get("coachwhip_resolve")? })
}

/// A Metal shared event: the CPU signals values, the GPU queue waits for them.
pub struct GpuEvent(Retained<ProtocolObject<dyn MTLSharedEvent>>);
unsafe impl Send for GpuEvent {}
unsafe impl Sync for GpuEvent {}

impl GpuEvent {
    pub fn new(device: &Device) -> Result<Self> {
        let e = device.as_ref().newSharedEvent().ok_or_else(|| candle::Error::Msg("Metal gave no shared event".into()))?;
        Ok(Self(e))
    }

    pub fn signal(&self, value: u64) {
        self.0.setSignaledValue(value);
    }

    pub fn value(&self) -> u64 {
        self.0.signaledValue()
    }

    /// Queue a wait: everything encoded after this runs only once the CPU has signalled `value`.
    pub fn gpu_wait(&self, value: u64) -> Result<()> {
        let commands = Commands::current().ok_or_else(|| candle::Error::Msg("no Metal command queue in use yet".into()))?;
        let event: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*self.0);
        commands.encode_wait_for_event(event, value).map_err(candle::MetalError::from)?;
        Ok(())
    }
}

/// Commit the work encoded so far, so it runs and completes without waiting for anything later.
pub fn flush() -> Result<()> {
    let commands = Commands::current().ok_or_else(|| candle::Error::Msg("no Metal command queue in use yet".into()))?;
    commands.flush().map_err(candle::MetalError::from)?;
    Ok(())
}

/// `ids (1, k) u32, guess (n_guess,) u32`: copies both into the bank's shared picks buffer and
/// raises the flag. The output is a one-element dummy.
pub struct Publish {
    pub bank: Arc<GpuBank>,
    pub pipeline: Arc<ComputePipeline>,
    pub n_guess: usize,
    /// This job's stamp, the low 16 bits of its sequence number.
    pub tag: u32,
}

impl CustomOp2 for Publish {
    fn name(&self) -> &'static str {
        "coachwhip-publish"
    }

    fn cpu_fwd(&self, _: &candle::CpuStorage, _: &Layout, _: &candle::CpuStorage, _: &Layout) -> Result<(candle::CpuStorage, Shape)> {
        candle::bail!("coachwhip-publish runs on Metal only")
    }

    fn metal_fwd(&self, ids: &MetalStorage, il: &Layout, guess: &MetalStorage, gl: &Layout) -> Result<(MetalStorage, Shape)> {
        if ids.dtype() != DType::U32 || guess.dtype() != DType::U32 {
            candle::bail!("coachwhip-publish: ids and guess must be u32")
        }
        let n_ids = il.shape().elem_count();
        if n_ids + self.n_guess > self.bank.max_picks {
            candle::bail!("coachwhip-publish: {} picks, room for {}", n_ids + self.n_guess, self.bank.max_picks)
        }
        let device = ids.device().clone();
        let dst = device.new_buffer_builder().with_size_for(1, DType::U32).with_label("publish").build()?;
        {
            let encoder = device.command_encoder()?;
            let encoder = encoder.as_ref();
            encoder.set_compute_pipeline_state(&self.pipeline);
            let n_ids_u = n_ids as u32;
            let n_guess_u = self.n_guess as u32;
            set_params!(
                encoder,
                (
                    (ids.buffer(), il.start_offset() * 4),
                    (guess.buffer(), gl.start_offset() * 4),
                    self.bank.picks.as_ref(),
                    self.bank.flag.as_ref(),
                    Output::with_offset(&dst, 0),
                    n_ids_u,
                    n_guess_u,
                    self.tag
                )
            );
            encoder.dispatch_thread_groups(MTLSize { width: 1, height: 1, depth: 1 }, MTLSize { width: 64, height: 1, depth: 1 });
        }
        Ok((MetalStorage::new(dst, device, 1, DType::U32), Shape::from(1)))
    }
}

/// `ids (1, k) u32` -> `slots (1, k) u32`, looked up in the bank's expert-to-slot table on the GPU.
pub struct Resolve {
    pub bank: Arc<GpuBank>,
    pub pipeline: Arc<ComputePipeline>,
}

impl CustomOp1 for Resolve {
    fn name(&self) -> &'static str {
        "coachwhip-resolve"
    }

    fn cpu_fwd(&self, _: &candle::CpuStorage, _: &Layout) -> Result<(candle::CpuStorage, Shape)> {
        candle::bail!("coachwhip-resolve runs on Metal only")
    }

    fn metal_fwd(&self, ids: &MetalStorage, il: &Layout) -> Result<(MetalStorage, Shape)> {
        if ids.dtype() != DType::U32 {
            candle::bail!("coachwhip-resolve: ids must be u32")
        }
        let n_ids = il.shape().elem_count();
        let device = ids.device().clone();
        let dst = device.new_buffer_builder().with_size_for(n_ids, DType::U32).with_label("resolve").build()?;
        {
            let encoder = device.command_encoder()?;
            let encoder = encoder.as_ref();
            encoder.set_compute_pipeline_state(&self.pipeline);
            let n_ids_u = n_ids as u32;
            set_params!(encoder, ((ids.buffer(), il.start_offset() * 4), self.bank.table.as_ref(), Output::with_offset(&dst, 0), self.bank.missing.as_ref(), n_ids_u));
            encoder.dispatch_thread_groups(MTLSize { width: 1, height: 1, depth: 1 }, MTLSize { width: 64, height: 1, depth: 1 });
        }
        Ok((MetalStorage::new(dst, device, n_ids, DType::U32), il.shape().clone()))
    }
}
