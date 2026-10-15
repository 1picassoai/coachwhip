//! The engine's own GPU scratch: buffers allocated once and reused whenever free, never given
//! back. A buffer is free when only the arena holds it. `COACHWHIP_ARENA=0` = Candle's pool.
use candle::{DType, MetalDevice, Result};
use candle_metal_kernels::metal::Buffer;
use std::sync::{Arc, Mutex, OnceLock};

static BUFS: Mutex<Vec<(usize, Arc<Buffer>)>> = Mutex::new(Vec::new());

pub fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("COACHWHIP_ARENA").map(|v| v != "0").unwrap_or(true))
}

/// A buffer of at least `bytes`, free or new; rounded up to 1 MB steps so sizes repeat.
pub fn take(dev: &MetalDevice, bytes: usize, label: &str) -> Result<Arc<Buffer>> {
    if !on() {
        return Ok(dev.new_buffer_builder().with_size(bytes).with_label(label).build()?);
    }
    let want = if bytes >= 1 << 20 { (bytes + (1 << 20) - 1) & !((1 << 20) - 1) } else { bytes.next_power_of_two().max(256) };
    let mut v = BUFS.lock().unwrap();
    let mut best: Option<usize> = None;
    for (i, (sz, b)) in v.iter().enumerate() {
        if *sz >= want && Arc::strong_count(b) == 1 && best.map_or(true, |j| v[j].0 > *sz) {
            best = Some(i);
        }
    }
    if let Some(i) = best {
        return Ok(v[i].1.clone());
    }
    let b = dev.new_raw_buffer(want)?;
    v.push((want, b.clone()));
    Ok(b)
}

pub fn take_for(dev: &MetalDevice, elems: usize, dt: DType, label: &str) -> Result<Arc<Buffer>> {
    take(dev, elems * dt.size_in_bytes(), label)
}

/// (bytes held, buffers) for the log.
pub fn stats() -> (usize, usize) {
    let v = BUFS.lock().unwrap();
    (v.iter().map(|(s, _)| *s).sum(), v.len())
}
