//! The expert store: run a mixture-of-experts model whose experts do not fit in memory.
//!
//! Apple silicon has one memory for the CPU and the GPU. Every expert is a slice of bytes inside
//! the GGUF file, and the OS page cache is the shelf. Each layer owns a bank: for gate, up and
//! down one shared Metal buffer holding `bank` expert slots side by side, allocated once and
//! overwritten in place, so a load is one read from the file straight into the memory the GPU
//! uses. A layer's maths is three dispatches of ggml's `mul_mv_id` kernel (every selected expert
//! of every token at once) plus the activation and the weighted sum.
//!
//! While the GPU works on one layer, reader threads fetch the experts the next layer will
//! probably need, predicted from a table of routes the model took before (the hot path). The
//! table keeps learning: every answer's routes are added to it at once and saved, so the
//! predictions follow what this machine is actually asked.

use candle::quantized::metal::QMetalStorage;
use candle::quantized::{ggml_file::qtensor_from_ggml, gguf_file, GgmlDType, QMatMul, QTensor};
use candle::{CustomOp2, DType, Device, Layout, MetalDevice, MetalStorage, Module, Result, Shape, Tensor, D};
use candle::backend::BackendStorage;
use candle_nn::Linear;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::os::unix::fs::FileExt;
use rayon::prelude::*;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex, Weak};
use std::time::Instant;

/// Where one matrix of one expert lives in the file.
#[derive(Clone)]
struct Slice {
    dtype: GgmlDType,
    offset: u64,
    bytes: usize,
    dims: Vec<usize>,
}

/// One matrix of a layer's bank: `per_layer` expert slots in one shared buffer.
struct BankMat {
    storage: Arc<QMetalStorage>,
    base: usize,
    bytes: usize,
    dtype: GgmlDType,
    n: usize,
    k: usize,
}

/// Two kinds of slot (`COACHWHIP_SHARP=<n>`, Captain's design): the experts the model leans on
/// keep the file's own bytes in `n` sharp slots per layer; the rest go through the GPU shrink
/// into the small (Q2) bank. A sharp slot's index carries `SHARP_FLAG`.
pub const SHARP_FLAG: usize = 1 << 30;
/// A prefetch job's expert id with this bit set goes to the sharp bank (decided by weight).
pub const SHARP_TAG: u32 = 1 << 31;

struct SharpBank {
    mats: [BankMat; 3],
    slots: Slots,
    ready: Vec<Arc<AtomicBool>>,
}

struct Bank {
    mats: [BankMat; 3],
    slots: Slots,
    sharp: Option<SharpBank>,
    /// False while the prefetch thread is still reading an expert into the slot.
    ready: Vec<Arc<AtomicBool>>,
    /// The GPU's own copy of which slot holds which expert, and which slots are ready (S4).
    gpu: Arc<crate::gpu_sync::GpuBank>,
    /// S4: slots the GPU is about to read for the word in flight; a prefetch may not take them.
    pinned: Vec<AtomicBool>,
}

impl Bank {
    /// Rewrite the GPU's table from the CPU's: every expert's slot, or none.
    fn sync_table(&self) {
        for e in 0..self.gpu.experts as u32 {
            match self.slots.slot_of.get(&e) {
                Some(&s) => self.gpu.set_slot(e, s),
                None => self.gpu.clear(e),
            }
        }
    }
}

/// A bank's bookkeeping, apart from the GPU memory it describes: which expert sits in which
/// slot, and the order slots were last used in. Pure, so it can be tested without a GPU.
#[derive(Clone, Debug)]
pub struct Slots {
    pub holder: Vec<Option<u32>>,
    pub slot_of: HashMap<u32, usize>,
    /// Slots from least to most recently used.
    pub lru: VecDeque<usize>,
    /// Experts a prefetch brought in that no token has asked for yet.
    pub prefetched: HashSet<u32>,
}

/// A zeroed quantised buffer of exactly `elems` weights, outside candle's power-of-two rounding.
fn exact_zeros(metal: &MetalDevice, elems: usize, dtype: GgmlDType) -> Result<QMetalStorage> {
    use candle::quantized::k_quants::{BlockQ2K, BlockQ3K, BlockQ4K, BlockQ5K, BlockQ6K, GgmlType};
    use candle::quantized::metal::load_quantized;
    if std::env::var("COACHWHIP_POOLED_BANK").map(|v| v == "1").unwrap_or(false) {
        return QMetalStorage::zeros(metal, elems, dtype);
    }
    let n = elems / dtype.block_size();
    let q = match dtype {
        GgmlDType::Q2K => load_quantized(metal, &vec![BlockQ2K::zeros(); n])?,
        GgmlDType::Q3K => load_quantized(metal, &vec![BlockQ3K::zeros(); n])?,
        GgmlDType::Q4K => load_quantized(metal, &vec![BlockQ4K::zeros(); n])?,
        GgmlDType::Q5K => load_quantized(metal, &vec![BlockQ5K::zeros(); n])?,
        GgmlDType::Q6K => load_quantized(metal, &vec![BlockQ6K::zeros(); n])?,
        _ => return QMetalStorage::zeros(metal, elems, dtype),
    };
    match q {
        candle::quantized::QStorage::Metal(s) => Ok(s),
        _ => candle::bail!("exact bank: not a Metal buffer"),
    }
}

/// The chat's longest conversation, in tokens (the server's limit); the governor keeps room for it.
const WRANGLER_CONTEXT: f64 = 16384.0;
/// Kept free under macOS's GPU working-set limit, and the working memory a forward needs on top of
/// the weights, banks and keys. Measured 9 Oct 2026 on the squeezed 122B, 16 GB Mac mini: bank 40
/// with a 15.7K prompt peaked at 11.34 GB of 12.71 GB and ran; bank 44 stalled.
const WRANGLER_MARGIN: f64 = 1.3e9;
const WRANGLER_WORKING: f64 = 1.0e9;

/// Wrangler: the most bank slots per layer that fit a full chat under the GPU limit, or None when
/// switched off (`COACHWHIP_WRANGLER=0`) or not measurable.
fn wrangler_fit(ct: &gguf_file::Content, metal: &MetalDevice, layers: usize, expert_bytes: f64) -> Option<usize> {
    use objc2_metal::MTLDevice as _;
    if std::env::var("COACHWHIP_WRANGLER").map(|v| v == "0").unwrap_or(false) {
        return None;
    }
    let limit = metal.device().as_ref().recommendedMaxWorkingSetSize() as f64;
    let md_u = |k: &str| ct.metadata.get(k).and_then(|v| v.to_u32().ok()).map(|v| v as f64);
    let arch = ct.metadata.get("general.architecture").and_then(|v| v.to_string().ok().cloned()).unwrap_or_default();
    // The weights that live on the GPU: everything but the experts and the packed word table.
    let dense: f64 = ct
        .tensor_infos
        .iter()
        .filter(|(n, _)| !n.contains("_exps") && n.as_str() != "token_embd.weight")
        .map(|(_, i)| (i.shape.elem_count() / i.ggml_dtype.block_size() * i.ggml_dtype.type_size()) as f64)
        .sum();
    // Keys and values for a full chat, f32, on the layers that keep them.
    let interval = md_u(&format!("{arch}.full_attention_interval")).unwrap_or(1.0).max(1.0);
    let n_kv = md_u(&format!("{arch}.attention.head_count_kv")).unwrap_or(1.0);
    let hd = md_u(&format!("{arch}.attention.key_length")).unwrap_or(128.0);
    let kv = (layers as f64 / interval).floor() * 2.0 * n_kv * hd * 4.0 * WRANGLER_CONTEXT;
    let room = limit - WRANGLER_MARGIN - WRANGLER_WORKING - dense - kv;
    let fit = (room / (layers as f64 * expert_bytes)).floor().max(8.0) as usize;
    eprintln!(
        "coachwhip: wrangler: GPU limit {:.2} GB; weights {:.2} GB, keys for a {}-token chat {:.2} GB, working {:.1} GB, margin {:.1} GB; room for {fit} slots per layer",
        limit / 1e9, dense / 1e9, WRANGLER_CONTEXT as usize, kv / 1e9, WRANGLER_WORKING / 1e9, WRANGLER_MARGIN / 1e9
    );
    Some(fit)
}

/// macOS memory pressure: 1 normal, 2 warning, 4 critical (0 when it cannot be read).
fn memory_pressure() -> i32 {
    let mut level: libc::c_int = 0;
    let mut size = std::mem::size_of::<libc::c_int>();
    let name = b"kern.memorystatus_vm_pressure_level\0";
    let r = unsafe { libc::sysctlbyname(name.as_ptr() as *const libc::c_char, &mut level as *mut _ as *mut libc::c_void, &mut size, std::ptr::null_mut(), 0) };
    if r == 0 { level } else { 0 }
}

/// What one forward needs from a layer's bank: the experts already there and the ones to read in.
#[derive(Debug, Default, PartialEq)]
pub struct Plan {
    pub hits: Vec<(u32, usize)>,
    pub misses: Vec<(u32, usize)>,
    /// Hits that a prefetch had brought in.
    pub prefetch_hits: u64,
}

impl Slots {
    pub fn new(n: usize) -> Self {
        Self { holder: vec![None; n], slot_of: HashMap::new(), lru: (0..n).collect(), prefetched: HashSet::new() }
    }

    /// The least recently used slot that is not being filled (`ready`) and does not hold one of `keep`.
    fn victim(&self, ready: &impl Fn(usize) -> bool, keep: &[u32]) -> Option<usize> {
        self.lru.iter().position(|&slot| ready(slot) && self.holder[slot].map_or(true, |h| !keep.contains(&h)))
    }

    fn put(&mut self, pos: usize, e: u32) -> usize {
        let slot = self.lru.remove(pos).unwrap();
        if let Some(old) = self.holder[slot] {
            self.slot_of.remove(&old);
            self.prefetched.remove(&old);
        }
        self.holder[slot] = Some(e);
        self.slot_of.insert(e, slot);
        self.lru.push_back(slot);
        slot
    }

    /// Give every expert in `needed` a slot: hits move to most recently used, misses take the least
    /// recently used slot that is ready and holds nothing this forward needs. None, and nothing
    /// changed, when too few such slots are free (the caller then reads the experts fresh).
    pub fn assign(&mut self, needed: &[u32], ready: impl Fn(usize) -> bool) -> Option<Plan> {
        let missing = needed.iter().filter(|e| !self.slot_of.contains_key(e)).count();
        let free = self.lru.iter().filter(|&&slot| ready(slot) && self.holder[slot].map_or(true, |h| !needed.contains(&h))).count();
        if free < missing {
            return None;
        }
        let mut plan = Plan::default();
        for &e in needed {
            if let Some(&slot) = self.slot_of.get(&e) {
                self.lru.retain(|&x| x != slot);
                self.lru.push_back(slot);
                if self.prefetched.remove(&e) {
                    plan.prefetch_hits += 1;
                }
                plan.hits.push((e, slot));
                continue;
            }
            let pos = self.victim(&ready, needed).expect("a bank always has a slot that this forward does not need");
            plan.misses.push((e, self.put(pos, e)));
        }
        Some(plan)
    }

    /// Let an expert go: its slot is emptied and goes to the front of the eviction line.
    pub fn release(&mut self, e: u32) {
        if let Some(slot) = self.slot_of.remove(&e) {
            self.holder[slot] = None;
            self.prefetched.remove(&e);
            self.lru.retain(|&x| x != slot);
            self.lru.push_front(slot);
        }
    }

    /// The toaster: once a layer has used these experts, their slots go to the front of the
    /// eviction line, so the next reads take them first instead of the longest-unused slots.
    pub fn pop(&mut self, used: &[u32]) {
        for &e in used {
            if let Some(&slot) = self.slot_of.get(&e) {
                self.lru.retain(|&x| x != slot);
                self.lru.push_front(slot);
            }
        }
    }

    /// Reserve slots for the predicted experts not already held, best first, as far as free slots
    /// go; never evicts another predicted expert or a slot still being filled.
    pub fn reserve(&mut self, predicted: &[u32], ready: impl Fn(usize) -> bool) -> Vec<(u32, usize)> {
        let mut out = Vec::new();
        for &e in predicted {
            if self.slot_of.contains_key(&e) {
                continue;
            }
            let Some(pos) = self.victim(&ready, predicted) else { continue };
            let slot = self.put(pos, e);
            self.prefetched.insert(e);
            out.push((e, slot));
        }
        out
    }
}

pub type Key = (usize, u32);

/// Scores for layer `layer + 1`'s experts given `layer`'s picks (expert, weight), from the route
/// table: each pick hands its weight on in proportion to where it has led before. Best first;
/// equal scores in expert order, so the same table always gives the same guess.
pub fn next_scores(routes: &Counts, layer: usize, now: &[(u32, f32)]) -> Vec<(u32, f32)> {
    let mut score: HashMap<u32, f32> = HashMap::new();
    for &(a, w) in now {
        if let Some(row) = routes.get(&(layer, a)) {
            let total = row.values().sum::<u64>().max(1) as f32;
            for (&b, &n) in row {
                *score.entry(b).or_insert(0.0) += w * n as f32 / total;
            }
        }
    }
    let mut v: Vec<(u32, f32)> = score.into_iter().collect();
    v.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap().then(x.0.cmp(&y.0)));
    v
}

/// How the store behaves. `Default` is what the Mac mini M4 with 16 GB was tuned on.
#[derive(Clone, Debug)]
pub struct Settings {
    /// Expert slots kept per layer (one expert each: about 3 MB for the 30B, 2 MB for the 80B, at Q4_K_M).
    pub bank: usize,
    /// Experts guessed and fetched ahead per layer; 0 turns prefetch off.
    pub prefetch: usize,
    /// How many layers ahead the guess looks: layer n runs layer n + ahead's own router on its
    /// input, so the reads have that many layers of GPU work to land in.
    pub ahead: usize,
    /// Reader threads that fetch predicted experts.
    pub readers: usize,
    /// Folder of route logs the predictions start from.
    pub routes: Option<std::path::PathBuf>,
    /// Folder where the routes of every answer are saved, and read back at the next start.
    pub learned: Option<std::path::PathBuf>,
    /// Print where each word's time went after every answer.
    pub profile: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self { bank: 44, prefetch: 10, ahead: 4, readers: 3, routes: None, learned: None, profile: false }
    }
}

/// Route table: for each (layer, expert), how often each expert of the next layer followed it.
pub type Counts = HashMap<Key, HashMap<u32, u64>>;

/// Adds one layer-to-layer step to the table: every expert picked now, for every expert picked
/// in the layer before, token by token.
pub fn count_step(counts: &mut Counts, layer: usize, before: &[Vec<u32>], now: &[Vec<u32>]) {
    for (a_row, b_row) in before.iter().zip(now.iter()) {
        // Nothing followed (a broken log line): leave no empty row behind to look like knowledge.
        if b_row.is_empty() {
            continue;
        }
        for &a in a_row {
            let row = counts.entry((layer, a)).or_default();
            for &b in b_row {
                *row.entry(b).or_insert(0) += 1;
            }
        }
    }
}

/// Reads route logs from folders. A log is a "T ..." line followed by one line per layer:
/// comma-separated tokens, each the space-separated experts the router picked for it.
pub fn load_routes(dirs: &[&Path], layers: usize) -> Counts {
    let mut counts = Counts::new();
    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(dir) else { continue };
        read_logs(&mut counts, entries, layers);
    }
    counts
}

fn read_logs(counts: &mut Counts, entries: std::fs::ReadDir, layers: usize) {
    for entry in entries.flatten() {
        if entry.path().extension().and_then(|x| x.to_str()) != Some("log") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(entry.path()) else { continue };
        let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
        let mut i = 0;
        while i + layers < lines.len() {
            if !lines[i].starts_with("T ") {
                i += 1;
                continue;
            }
            let per_layer: Vec<Vec<Vec<u32>>> = (0..layers)
                .map(|l| lines[i + 1 + l].split(',').map(|t| t.split_whitespace().filter_map(|e| e.parse().ok()).collect()).collect())
                .collect();
            for l in 0..layers - 1 {
                count_step(counts, l, &per_layer[l], &per_layer[l + 1]);
            }
            i += 1 + layers;
        }
    }
}

/// What the current forward pass has picked so far, for learning and for the log.
#[derive(Default)]
struct Trail {
    /// The layer before, and the experts each token picked there.
    before: Option<(usize, Vec<Vec<u32>>)>,
    /// One line per layer of the forward pass in progress.
    lines: Vec<String>,
    /// Finished forward passes, waiting to be saved.
    blocks: Vec<String>,
    /// The words of the forward pass in progress, for the block's T line.
    tokens: Vec<u32>,
}

/// Our register: per layer, what each expert adds to the running vector (`d`), what the layer adds
/// regardless (`d0`), and the last prediction waiting to be checked against the truth.
struct Register {
    d: Vec<Tensor>,
    d0: Vec<Tensor>,
    lr: f32,
    lr0: f32,
    last: Option<(usize, Tensor, Tensor, Tensor)>,
    /// The input the last prediction was made from (measurement only).
    last_x: Tensor,
}

impl Default for Register {
    fn default() -> Self {
        Self { d: Vec::new(), d0: Vec::new(), lr: 1.0, lr0: 0.05, last: None, last_x: Tensor::zeros(1, DType::F32, &Device::Cpu).unwrap() }
    }
}

#[derive(Default)]
struct Stats {
    hits: u64,
    loads: u64,
    load_seconds: f64,
    bytes_loaded: u64,
    fresh: u64,
    fresh_read_seconds: f64,
    fresh_build_seconds: f64,
    prefetch_copies: u64,
    prefetch_hits: u64,
    /// Guessed reads thrown away because the word had moved past their layer.
    stale_dropped: u64,
    /// Time a forward spent waiting for a guessed expert still being read in.
    prefetch_wait_seconds: f64,
    /// Forwards that fit the bank but found too few free slots (guesses still landing).
    crowded: u64,
    words: u64,
    word_started: Option<Instant>,
    word_seconds: f64,
    expert_seconds: f64,
    load_phase_seconds: f64,
    maths_seconds: f64,
    /// S6 measurement: how often a bank-only route (experts already resident) would have picked
    /// the same experts as the real route. Layers seen, summed agreement fraction, layers fully
    /// agreeing, words where every layer agreed, and whether the current word has agreed so far.
    draft_layers: u64,
    draft_agree: f64,
    draft_layers_full: u64,
    draft_words_full: u64,
    draft_word_ok: bool,
    /// S6 measurement: drafts run, words drafted, words the plain path agreed with.
    spec_passes: u64,
    spec_drafted: u64,
    spec_accepted: u64,
    /// Register measurement: relative error of its predicted state against the truth, summed per
    /// layer step, and the same for the plain carry-over (x alone) as the yardstick.
    /// Where the CPU time around the loads goes, per layer step (profile mode, plain path):
    /// waiting for the GPU to hand the picks back, placing the experts in the bank (no reads),
    /// the reads themselves, the bookkeeping (learn, log, guess, prefetch requests), and the encode.
    t_readback: f64,
    t_place: f64,
    t_books: f64,
    t_encode: f64,
    reg_steps: u64,
    reg_err: f64,
    reg_err_plain: f64,
    reg_err_by_layer: Vec<f64>,
    reg_plain_by_layer: Vec<f64>,
}

static PROFILE: AtomicBool = AtomicBool::new(false);

fn profiling() -> bool {
    PROFILE.load(Ordering::Relaxed)
}

fn toaster() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("COACHWHIP_TOASTER").is_ok())
}

/// Lab (`COACHWHIP_STAGE=1`): file reads land in ordinary memory and are copied into the bank.
fn staged() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("COACHWHIP_STAGE").map(|v| v == "1").unwrap_or(false))
}

/// Lab (`COACHWHIP_QRULE=<p>`): prefetch every early-guess candidate whose router probability is
/// at least `p`, instead of a flat `prefetch` per layer. `COACHWHIP_QCAP` caps the width (16).
fn q_rule() -> Option<f32> {
    static V: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("COACHWHIP_QRULE").ok().and_then(|v| v.parse().ok()))
}

fn q_cap() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("COACHWHIP_QCAP").ok().and_then(|v| v.parse().ok()).unwrap_or(16))
}

/// Lab (`COACHWHIP_SPLIT=1`): a layer's maths starts on the resident experts while its misses
/// are read, then finishes with the misses. Same sum, two halves.
fn split_maths() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("COACHWHIP_SPLIT").map(|v| v == "1").unwrap_or(false))
}

/// The GPU shrink (`COACHWHIP_GPUQ2=1`): the bank holds Q2_K copies, shrunk on arrival.
pub fn gpuq2() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("COACHWHIP_GPUQ2").map(|v| v == "1").unwrap_or(false))
}

/// The shaped shrink (`COACHWHIP_SHAPE=1`): the shrink's fit is weighted by the token's shape.
pub fn shaped() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("COACHWHIP_SHAPE").map(|v| v == "1").unwrap_or(false))
}

/// With the GPU shrink on, squeeze the down matrix too (`COACHWHIP_Q2_DOWN=1`). Off by default:
/// down is the fat, sensitive one, and the Q2 file that read well last night kept it sharp.
fn shrink_down() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("COACHWHIP_Q2_DOWN").map(|v| v == "1").unwrap_or(false))
}

/// By weight (`COACHWHIP_SHARP_W=<w>`, Captain's board): with two kinds of slot, the bank an
/// expert goes to is chosen by its router weight for this token: at or above `w` it fetches
/// sharp, below it Q2. Guesses use their router probability the same way. Off: the hot list rules.
fn sharp_weight() -> Option<f32> {
    static V: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("COACHWHIP_SHARP_W").ok().and_then(|v| v.parse().ok()))
}

/// Admission (`COACHWHIP_ADMIT_W=<w>`, Captain's board): with use-sharp on, a miss whose router
/// weight for this token is below `w` is used from its sharp staging and let go — no slot kept,
/// no squeeze. Above it, it earns a slot and is squeezed for reuse.
fn admit_w() -> Option<f32> {
    static V: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("COACHWHIP_ADMIT_W").ok().and_then(|v| v.parse().ok()))
}

/// Mechanics test (`COACHWHIP_Q2_NONE=1`): with the GPU shrink on, nothing is actually shrunk —
/// the bank holds the file's own bytes and the "squeeze" is a plain copy. The text must then be
/// identical to the plain engine's; if it is not, the flow has a bug.
fn q2_none() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("COACHWHIP_Q2_NONE").map(|v| v == "1").unwrap_or(false))
}

/// Heavy reuse re-read (`COACHWHIP_REREAD_W=<w>`): with use-sharp on, a hit whose router weight for
/// this token is at least `w` is re-read sharp and computed from the fresh bytes; the Q2 copy
/// stays in the bank for the light reuses. Heavy words exact, light ones pay the Q2.
fn reread_w() -> Option<f32> {
    static V: std::sync::OnceLock<Option<f32>> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("COACHWHIP_REREAD_W").ok().and_then(|v| v.parse().ok()))
}

/// Register repair (`COACHWHIP_REPAIR=1`, Captain's board): at the squeeze, the engine measures
/// what the squeeze did to that expert's output on the input it was used for (sharp minus Q2),
/// keeps the difference per expert, and adds it back whenever the Q2 copy is reused.
fn repair_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("COACHWHIP_REPAIR").map(|v| v == "1").unwrap_or(false))
}

/// Use sharp, then squeeze (`COACHWHIP_USE_SHARP=1`): misses compute from their fresh bytes
/// and are squeezed into the bank afterwards.
pub fn use_sharp() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("COACHWHIP_USE_SHARP").map(|v| v == "1").unwrap_or(false))
}

/// S4: the GPU never stops for the CPU (`COACHWHIP_PARALLEL=1`).
pub fn parallel() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("COACHWHIP_PARALLEL").is_ok())
}

pub struct ExpertStore {
    file: std::fs::File,
    device: Device,
    metal: MetalDevice,
    per_layer: usize,
    ahead: usize,
    slices: Vec<Vec<[Slice; 3]>>,
    banks: Mutex<HashMap<usize, Bank>>,
    stats: Mutex<Stats>,
    layers: usize,
    routes: Mutex<Counts>,
    trail: Mutex<Trail>,
    learned: Option<std::path::PathBuf>,
    learned_file: String,
    /// Prefetch jobs: the layer to fill, the word they were guessed for, the experts.
    prefetch_tx: Mutex<Option<mpsc::Sender<(usize, u64, Vec<u32>)>>>,
    /// The word being written (one-word steps only) and the layer it has reached.
    word: AtomicU64,
    now_layer: AtomicUsize,
    /// S6: routing confined to the bank, nothing read or learned.
    draft: AtomicBool,
    /// Foresight: the draft pass also tells the bank what the real pass of this word will need.
    foresee: AtomicBool,
    /// Which word ahead the foresight draft is on (0 = the word about to be written).
    foresee_step: AtomicU64,
    foresight_lists: Mutex<Vec<(usize, Tensor)>>,
    /// Lab (`COACHWHIP_DUMP=file`): every layer's MoE input and router scores, one record per
    /// layer per word, to fit the expert predictor from.
    dump: Mutex<Option<std::io::BufWriter<std::fs::File>>>,
    /// Our register (`COACHWHIP_REGISTER=1`): what each expert adds to the running vector, per
    /// layer, learned on the GPU as the words go by; the early guess reads the state it predicts.
    register: Mutex<Option<Register>>,
    /// S4: the resolve-and-wait kernel, when the parallel path is on (`COACHWHIP_PARALLEL=1`).
    /// The parallel path (S4), when `COACHWHIP_PARALLEL` is set.
    s4: Option<S4>,
    /// The GPU shrink (`COACHWHIP_GPUQ2=1`): arriving experts are shrunk to Q2_K on the GPU on
    /// their way into the bank, which holds Q2 copies and so parks more of them.
    requant: Option<crate::requant::Requant>,
    /// Staging buffers for the shrink, by size, reused across reads.
    staging: Mutex<HashMap<usize, Vec<Arc<candle_metal_kernels::metal::Buffer>>>>,
    /// Use-sharp-then-squeeze (`COACHWHIP_USE_SHARP=1`, Captain's board): a miss is read into a
    /// sharp one-slot staging (the file's own dtype), this token computes from it, and once the
    /// GPU has drained at the next layer it is squeezed into its Q2 slot. The stagings, pooled by
    /// (dtype, elements); the misses waiting for their squeeze; and this forward's fresh experts.
    sharp_pool: Mutex<HashMap<(GgmlDType, usize), Vec<Arc<QMetalStorage>>>>,
    pending_squeeze: Mutex<Vec<(usize, u32, usize, [Arc<QMetalStorage>; 3], f32, bool)>>,
    fresh: Mutex<HashMap<(usize, u32), [Arc<QMetalStorage>; 3]>>,
    /// Register repair: the input each layer last saw (1, h), and per (layer, expert) the
    /// measured difference sharp minus Q2 of that expert's output (h,).
    last_x: Mutex<HashMap<usize, Tensor>>,
    repair: Mutex<HashMap<(usize, u32), Tensor>>,
    /// Sharp slots per layer (0 = one kind of slot) and the hot experts per layer that get them,
    /// from the route logs at load: the most-used experts of each layer.
    sharp_n: usize,
    hot: Vec<Vec<u32>>,
    /// The token's shape per layer (`COACHWHIP_SHAPE=1`): running mean of x² over the hidden
    /// size (for gate and up) and of the expert's hidden² over the intermediate size (for down),
    /// in shared buffers the shaped shrink reads. Count of updates alongside.
    shape: Mutex<HashMap<usize, (Arc<candle_metal_kernels::metal::Buffer>, Arc<candle_metal_kernels::metal::Buffer>, u64)>>,
}

/// The parallel path's fixed parts: the two kernels, the shared event, the sequence of waits and
/// the service thread's queue.
struct S4 {
    pipes: crate::gpu_sync::Pipelines,
    event: crate::gpu_sync::GpuEvent,
    seq: AtomicU64,
    tx: Mutex<Option<mpsc::Sender<S4Job>>>,
}

/// One layer's picks for the service thread: read them, place the experts, signal the GPU.
struct S4Job {
    layer: usize,
    seq: u64,
    n_ids: usize,
    n_guess: usize,
    prefetch_k: usize,
    ahead: usize,
}

/// Expert ids ordered by a router's scores, best first. Equal scores keep expert order, so the
/// same scores always give the same guess.
pub fn rank(scores: &[f32]) -> Vec<u32> {
    let mut ids: Vec<u32> = (0..scores.len() as u32).collect();
    ids.sort_by(|&a, &b| scores[b as usize].partial_cmp(&scores[a as usize]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
    ids
}

/// A guessed read is stale once its word is over, or the word has reached its layer: from then
/// on that layer reads its own misses, and a late guess would only evict experts still to be used.
pub fn is_stale(job_layer: usize, job_word: u64, now_layer: usize, now_word: u64) -> bool {
    job_word < now_word || (job_word == now_word && job_layer <= now_layer)
}

impl ExpertStore {
    /// Reads the file's table of contents only; experts are read when a token first asks for them.
    pub fn new(ct: &gguf_file::Content, device: &Device, layers: usize, path: &Path, settings: &Settings) -> Result<Arc<Self>> {
        let Device::Metal(metal) = device else {
            candle::bail!("Coachwhip needs a Metal device (Apple silicon)")
        };
        PROFILE.store(settings.profile, Ordering::Relaxed);
        let mut per_layer = settings.bank.max(1);

        let mut slices = Vec::with_capacity(layers);
        for layer in 0..layers {
            let parts = ["gate", "up", "down"].map(|m| {
                let info = &ct.tensor_infos[&format!("blk.{layer}.ffn_{m}_exps.weight")];
                let dims = info.shape.dims().to_vec();
                let per_expert_elems: usize = dims[1..].iter().product();
                let bytes = per_expert_elems / info.ggml_dtype.block_size() * info.ggml_dtype.type_size();
                (info.ggml_dtype, ct.tensor_data_offset + info.offset, bytes, dims[1..].to_vec(), dims[0])
            });
            let experts = parts[0].4;
            let per_expert: Vec<[Slice; 3]> = (0..experts)
                .map(|e| {
                    parts.clone().map(|(dtype, base, bytes, dims, _)| Slice {
                        dtype,
                        offset: base + (e * bytes) as u64,
                        bytes,
                        dims,
                    })
                })
                .collect();
            slices.push(per_expert);
        }
        let file = std::fs::File::open(path)?;
        // Lab (`COACHWHIP_NOCACHE=1`): read experts straight from the flash, bypassing the page
        // cache. On a 16 GB box the cache has nothing to give back and costs a copy per read.
        if std::env::var("COACHWHIP_NOCACHE").map(|v| v == "1").unwrap_or(false) {
            use std::os::unix::io::AsRawFd;
            let r = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) };
            eprintln!("coachwhip: page cache bypassed for expert reads (fcntl F_NOCACHE -> {r})");
        }
        let experts = slices[0].len();
        let per_expert_mb = slices[0][0].iter().map(|s| s.bytes).sum::<usize>() as f64 / 1e6;
        if let Some(fit) = wrangler_fit(ct, metal, layers, per_expert_mb * 1e6) {
            if fit < per_layer {
                eprintln!("coachwhip: wrangler: bank {per_layer} would not leave room for a full chat on this Mac; using {fit}");
                per_layer = fit;
            }
        }
        eprintln!(
            "coachwhip: {layers} layers x {experts} experts, {per_expert_mb:.1} MB each; bank {per_layer} slots per layer ({:.2} GB)",
            per_layer as f64 * layers as f64 * per_expert_mb / 1e3
        );
        let dirs: Vec<&Path> = settings.routes.iter().chain(settings.learned.iter()).map(|p| p.as_path()).collect();
        let routes = load_routes(&dirs, layers);
        eprintln!("coachwhip: route table for {} (layer, expert) pairs", routes.len());
        let sharp_n: usize = std::env::var("COACHWHIP_SHARP").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let hot: Vec<Vec<u32>> = (0..layers)
            .map(|l| {
                let mut use_count: Vec<(u32, u64)> = (0..slices[l].len() as u32).map(|e| (e, routes.get(&(l, e)).map_or(0, |row| row.values().sum()))).collect();
                use_count.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                use_count.into_iter().take(sharp_n).filter(|(_, n)| *n > 0).map(|(e, _)| e).collect()
            })
            .collect();
        if sharp_n > 0 {
            let known: usize = hot.iter().map(|h| h.len()).sum();
            eprintln!("coachwhip: two kinds of slot: {sharp_n} sharp per layer for the hot experts ({known} known from the route logs), {per_layer} small");
        }
        let started = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let store = Arc::new(Self {
            file,
            device: device.clone(),
            metal: metal.clone(),
            per_layer,
            ahead: settings.ahead,
            slices,
            banks: Mutex::new(HashMap::new()),
            stats: Mutex::new(Stats::default()),
            layers,
            routes: Mutex::new(routes),
            trail: Mutex::new(Trail::default()),
            learned: settings.learned.clone(),
            learned_file: format!("{started}.log"),
            prefetch_tx: Mutex::new(None),
            word: AtomicU64::new(0),
            now_layer: AtomicUsize::new(0),
            draft: AtomicBool::new(false),
            foresee: AtomicBool::new(false),
            foresee_step: AtomicU64::new(0),
            foresight_lists: Mutex::new(Vec::new()),
            dump: Mutex::new(std::env::var("COACHWHIP_DUMP").ok().and_then(|p| std::fs::File::create(p).ok()).map(std::io::BufWriter::new)),
            register: Mutex::new(if std::env::var("COACHWHIP_REGISTER").map(|v| v == "1").unwrap_or(false) { Some(Register::default()) } else { None }),
            s4: if parallel() {
                Some(S4 { pipes: crate::gpu_sync::pipelines(metal.device())?, event: crate::gpu_sync::GpuEvent::new(metal.device())?, seq: AtomicU64::new(0), tx: Mutex::new(None) })
            } else {
                None
            },
            requant: if gpuq2() { Some(crate::requant::Requant::new(metal)?) } else { None },
            staging: Mutex::new(HashMap::new()),
            shape: Mutex::new(HashMap::new()),
            sharp_pool: Mutex::new(HashMap::new()),
            pending_squeeze: Mutex::new(Vec::new()),
            fresh: Mutex::new(HashMap::new()),
            last_x: Mutex::new(HashMap::new()),
            repair: Mutex::new(HashMap::new()),
            sharp_n,
            hot,
        });
        if gpuq2() {
            let q2_mb = store.slices[0][0].iter().enumerate().map(|(i, s)| if !q2_none() && (i < 2 || shrink_down()) { s.dims.iter().product::<usize>() / crate::requant::QK_K * crate::requant::Q2K_BLOCK_BYTES } else { s.bytes }).sum::<usize>() as f64 / 1e6;
            eprintln!("coachwhip: GPU shrink on: experts are shrunk to Q2_K on arrival, {q2_mb:.1} MB each in the bank ({:.2} GB)", per_layer as f64 * layers as f64 * q2_mb / 1e3);
        }
        if let Some(s4) = &store.s4 {
            eprintln!("coachwhip: parallel path on: the GPU waits on a shared event for its experts; the CPU never waits for the GPU inside a word");
            let (tx, rx) = mpsc::channel::<S4Job>();
            *s4.tx.lock().unwrap() = Some(tx);
            let weak: Weak<Self> = Arc::downgrade(&store);
            std::thread::spawn(move || {
                for job in rx {
                    let Some(st) = weak.upgrade() else { break };
                    st.s4_service(job);
                }
            });
        }
        // The prefetch threads read guessed experts into free slots while the GPU works on the
        // current layer. They hold only a weak handle, so the store still shuts down with the model.
        let (tx, rx) = mpsc::channel::<(usize, u64, Vec<u32>)>();
        *store.prefetch_tx.lock().unwrap() = Some(tx);
        let rx = Arc::new(Mutex::new(rx));
        for _ in 0..settings.readers {
            let weak: Weak<Self> = Arc::downgrade(&store);
            let rx = rx.clone();
            std::thread::spawn(move || loop {
                let job = rx.lock().unwrap().recv();
                let Ok((layer, word, experts)) = job else { break };
                let Some(st) = weak.upgrade() else { break };
                if is_stale(layer, word, st.now_layer.load(Ordering::Acquire), st.word.load(Ordering::Acquire)) {
                    st.stats.lock().unwrap().stale_dropped += experts.len() as u64;
                    continue;
                }
                if let Err(e) = st.prefetch(layer, &experts) {
                    eprintln!("coachwhip: prefetch failed: {e}");
                }
            });
        }
        Ok(store)
    }

    fn next_scores(&self, layer: usize, now: &[(u32, f32)]) -> Vec<(u32, f32)> {
        next_scores(&self.routes.lock().unwrap(), layer, now)
    }

    /// The `k` likeliest experts of `layer` that are not in its bank yet, one layer ahead of `now`
    /// (`ahead` = 1) or two (`ahead` = 2, composing the table twice).
    fn predict_missing(&self, layer: usize, now: &[u32], ahead: usize, k: usize) -> Vec<u32> {
        let mut scores: Vec<(u32, f32)> = now.iter().map(|&e| (e, 1.0)).collect();
        for step in 0..ahead {
            scores = self.next_scores(layer + step, &scores);
            scores.truncate(32);
        }
        let banks = self.banks.lock().unwrap();
        let present = banks.get(&(layer + ahead));
        scores
            .into_iter()
            .map(|(e, _)| e)
            .filter(|e| present.map_or(true, |b| !b.slots.slot_of.contains_key(e) && !b.sharp.as_ref().map_or(false, |sb| sb.slots.slot_of.contains_key(e))))
            .take(k)
            .collect()
    }

    pub fn register_on(&self) -> bool {
        self.register.lock().unwrap().is_some()
    }

    /// One step of the register at `layer`: learn from the layer before (its prediction against
    /// this layer's real input), then predict this layer's output state from the experts it chose.
    /// `xs` is (1, h), `ids` the chosen experts, `top_w` their weights (1, k). Returns x̂ (1, h).
    fn register_step(&self, layer: usize, xs: &Tensor, ids: &[u32], top_w: &Tensor) -> Result<Tensor> {
        let mut guard = self.register.lock().unwrap();
        let Some(reg) = guard.as_mut() else { return Ok(xs.clone()) };
        let h = xs.dim(1)?;
        let experts = self.slices[layer].len();
        let device = xs.device();
        if reg.d.is_empty() {
            for l in 0..self.layers {
                reg.d.push(Tensor::zeros((experts, h), DType::F32, device)?);
                reg.d0.push(Tensor::zeros(h, DType::F32, device)?);
            }
            reg.lr = std::env::var("COACHWHIP_REGISTER_LR").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
            reg.lr0 = std::env::var("COACHWHIP_REGISTER_LR0").ok().and_then(|v| v.parse().ok()).unwrap_or(0.05);
            eprintln!("coachwhip: register on: {} layers x {experts} experts x {h}, lr {} / {}", self.layers, reg.lr, reg.lr0);
        }
        // Learn: the layer before predicted x̂; this layer's input is the truth.
        if let Some((l, x_hat, ids_t, w_t)) = reg.last.take() {
            if l + 1 == layer {
                let err = (xs - &x_hat)?; // (1, h)
                if profiling() {
                    let truth = xs.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt().max(1e-6);
                    let e = err.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
                    let plain = (xs - &reg.last_x)?.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
                    let mut st = self.stats.lock().unwrap();
                    if st.reg_err_by_layer.len() < self.layers {
                        st.reg_err_by_layer = vec![0.0; self.layers];
                        st.reg_plain_by_layer = vec![0.0; self.layers];
                    }
                    st.reg_steps += 1;
                    st.reg_err += (e / truth) as f64;
                    st.reg_err_plain += (plain / truth) as f64;
                    st.reg_err_by_layer[layer] += (e / truth) as f64;
                    st.reg_plain_by_layer[layer] += (plain / truth) as f64;
                }
                let delta = w_t.broadcast_mul(&err)?.affine(reg.lr as f64, 0.0)?; // (k, h)
                reg.d[l] = reg.d[l].index_add(&ids_t, &delta, 0)?;
                reg.d0[l] = (&reg.d0[l] + err.squeeze(0)?.affine(reg.lr0 as f64, 0.0)?)?;
            }
        }
        // Predict: x̂ = x + d0 + Σ w_e d[e].
        let ids_t = Tensor::new(ids, device)?;
        let w_t = top_w.reshape((ids.len(), 1))?.to_dtype(DType::F32)?;
        let adds = reg.d[layer].index_select(&ids_t, 0)?; // (k, h)
        let x_hat = (xs.broadcast_add(&reg.d0[layer])? + w_t.broadcast_mul(&adds)?.sum_keepdim(0)?)?;
        reg.last = Some((layer, x_hat.clone(), ids_t, w_t));
        reg.last_x = xs.clone();
        Ok(x_hat)
    }

    /// The register step with the picks still on the GPU (the parallel path never reads them
    /// back): `top_ids` (1, k) u32, `top_w` (1, k). Learns from the layer before, predicts x̂.
    fn register_step_gpu(&self, layer: usize, xs: &Tensor, top_ids: &Tensor, top_w: &Tensor) -> Result<Tensor> {
        let mut guard = self.register.lock().unwrap();
        let Some(reg) = guard.as_mut() else { return Ok(xs.clone()) };
        let h = xs.dim(1)?;
        let k = top_ids.elem_count();
        let experts = self.slices[layer].len();
        let device = xs.device();
        if reg.d.is_empty() {
            for _ in 0..self.layers {
                reg.d.push(Tensor::zeros((experts, h), DType::F32, device)?);
                reg.d0.push(Tensor::zeros(h, DType::F32, device)?);
            }
            reg.lr = std::env::var("COACHWHIP_REGISTER_LR").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
            reg.lr0 = std::env::var("COACHWHIP_REGISTER_LR0").ok().and_then(|v| v.parse().ok()).unwrap_or(0.05);
            eprintln!("coachwhip: register on (GPU): {} layers x {experts} experts x {h}, lr {} / {}", self.layers, reg.lr, reg.lr0);
        }
        if let Some((l, x_hat, ids_t, w_t)) = reg.last.take() {
            if l + 1 == layer {
                let err = (xs - &x_hat)?;
                let delta = w_t.broadcast_mul(&err)?.affine(reg.lr as f64, 0.0)?;
                reg.d[l] = reg.d[l].index_add(&ids_t, &delta, 0)?;
                reg.d0[l] = (&reg.d0[l] + err.squeeze(0)?.affine(reg.lr0 as f64, 0.0)?)?;
            }
        }
        let ids_t = top_ids.flatten_all()?.contiguous()?;
        let w_t = top_w.reshape((k, 1))?.to_dtype(DType::F32)?;
        let adds = reg.d[layer].index_select(&ids_t, 0)?;
        let x_hat = (xs.broadcast_add(&reg.d0[layer])? + w_t.broadcast_mul(&adds)?.sum_keepdim(0)?)?;
        reg.last = Some((layer, x_hat.clone(), ids_t, w_t));
        Ok(x_hat)
    }

    /// Chain one more layer on the GPU: route x̂ with `gate` (the next layer's router), then add
    /// what those experts typically add. No read-back.
    fn register_chain(&self, layer: usize, x_hat: &Tensor, gate: &Linear, k: usize) -> Result<Tensor> {
        let guard = self.register.lock().unwrap();
        let Some(reg) = guard.as_ref() else { return Ok(x_hat.clone()) };
        if reg.d.is_empty() || layer >= reg.d.len() {
            return Ok(x_hat.clone());
        }
        let scores = gate.forward(x_hat)?; // (1, e)
        let top = scores.arg_sort_last_dim(false)?.narrow(D::Minus1, 0, k)?.contiguous()?; // (1, k)
        let p = candle_nn::ops::softmax_last_dim(&scores.gather(&top, D::Minus1)?)?; // (1, k)
        let adds = reg.d[layer].index_select(&top.flatten_all()?, 0)?; // (k, h)
        Ok((x_hat.broadcast_add(&reg.d0[layer])? + p.reshape((k, 1))?.broadcast_mul(&adds)?.sum_keepdim(0)?)?)
    }

    /// Lab: how many experts beyond the top k the foresight peek fetches per layer
    /// (`COACHWHIP_FORESIGHT_EXTRA`, default 0).
    fn foresight_extra() -> usize {
        static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *V.get_or_init(|| std::env::var("COACHWHIP_FORESIGHT_EXTRA").ok().and_then(|v| v.parse().ok()).unwrap_or(0))
    }

    /// Hand a layer's guessed experts to the prefetch threads.
    fn request_prefetch(&self, layer: usize, predicted: Vec<u32>) {
        if predicted.is_empty() {
            return;
        }
        if let Some(tx) = self.prefetch_tx.lock().unwrap().as_ref() {
            // A draft pass runs ahead of the real pass of the same word: its reads belong to that word.
            let word = self.word.load(Ordering::Acquire) + if self.drafting() { 1 + self.foresee_step.load(Ordering::Acquire) } else { 0 };
            let _ = tx.send((layer, word, predicted));
        }
    }

    /// Prefetch those of the top `k` experts in `guess` (best first) that `layer`'s bank does not
    /// hold, in pairs across the reader threads so they are read side by side.
    fn prefetch_guess(&self, layer: usize, guess: &[u32], k: usize) {
        let missing: Vec<u32> = {
            let banks = self.banks.lock().unwrap();
            let present = banks.get(&layer);
            guess.iter().take(k).copied().filter(|e| present.map_or(true, |b| !b.slots.slot_of.contains_key(e) && !b.sharp.as_ref().map_or(false, |sb| sb.slots.slot_of.contains_key(e)))).collect()
        };
        for pair in missing.chunks(2) {
            self.request_prefetch(layer, pair.to_vec());
        }
    }

    /// Mark where the word being written has got to; layer 0 starts a new word.
    fn reached(&self, layer: usize) {
        if layer == 0 {
            self.word.fetch_add(1, Ordering::AcqRel);
        }
        self.now_layer.store(layer, Ordering::Release);
    }

    /// Prefetch thread: reserve free slots for the predicted experts, then read them in with the
    /// bank unlocked, so the main thread is never held up by a read it did not ask for.
    fn prefetch(&self, layer: usize, predicted_all: &[u32]) -> Result<()> {
        // Two kinds: a guess tagged SHARP_TAG by the forward (its probability was above the line)
        // goes to the sharp bank; untagged guesses go by the hot list when no tags are present,
        // else to the small bank.
        let tagged = predicted_all.iter().any(|&e| e & SHARP_TAG != 0);
        let (hot, cold): (Vec<u32>, Vec<u32>) = if tagged {
            let hot: Vec<u32> = predicted_all.iter().filter(|&&e| e & SHARP_TAG != 0).map(|&e| e & !SHARP_TAG).collect();
            let cold: Vec<u32> = predicted_all.iter().filter(|&&e| e & SHARP_TAG == 0).copied().collect();
            (hot, cold)
        } else {
            predicted_all.iter().partition(|&&e| self.is_hot(layer, e))
        };
        if !hot.is_empty() {
            let mut sjobs: Vec<(u32, usize, [(usize, usize); 3], Arc<AtomicBool>)> = Vec::new();
            {
                let mut banks = self.banks.lock().unwrap();
                if let Some(bank) = banks.get_mut(&layer) {
                    if let Some(sb) = bank.sharp.as_mut() {
                        let ready = &sb.ready;
                        let reserved = sb.slots.reserve(&hot, |s| ready[s].load(Ordering::Acquire));
                        for &(_, slot) in &reserved {
                            ready[slot].store(false, Ordering::Release);
                        }
                        for (e, slot) in reserved {
                            let ptrs = [0, 1, 2].map(|i| (sb.mats[i].base + slot * sb.mats[i].bytes, sb.mats[i].bytes));
                            sjobs.push((e, slot, ptrs, ready[slot].clone()));
                        }
                    }
                }
            }
            for (e, _slot, ptrs, ready) in sjobs {
                for (i, s) in self.slices[layer][e as usize].iter().enumerate() {
                    let dst = unsafe { std::slice::from_raw_parts_mut(ptrs[i].0 as *mut u8, s.bytes) };
                    self.file.read_exact_at(dst, s.offset)?;
                }
                ready.store(true, Ordering::Release);
                self.stats.lock().unwrap().prefetch_copies += 1;
            }
        }
        let predicted: &[u32] = &cold;
        if predicted.is_empty() {
            return Ok(());
        }
        let mut jobs: Vec<(u32, usize, [(usize, usize); 3], Arc<AtomicBool>, Arc<crate::gpu_sync::GpuBank>)> = Vec::new();
        {
            let mut banks = self.banks.lock().unwrap();
            let Some(bank) = banks.get_mut(&layer) else { return Ok(()) };
            let ready = &bank.ready;
            let pinned = &bank.pinned;
            let reserved = bank.slots.reserve(predicted, |s| ready[s].load(Ordering::Acquire) && !pinned[s].load(Ordering::Acquire));
            for &(_, slot) in &reserved {
                ready[slot].store(false, Ordering::Release);
                bank.gpu.set_ready(slot, false);
            }
            // The GPU's table follows the CPU's: evicted experts gone, new ones present but not ready.
            bank.sync_table();
            let gpu = bank.gpu.clone();
            for (e, slot) in reserved {
                let ptrs = [0, 1, 2].map(|i| (bank.mats[i].base + slot * bank.mats[i].bytes, bank.mats[i].bytes));
                jobs.push((e, slot, ptrs, ready[slot].clone(), gpu.clone()));
            }
        }
        if self.requant.is_some() {
            let mats: [(Arc<QMetalStorage>, usize); 3] = {
                let banks = self.banks.lock().unwrap();
                let bank = &banks[&layer];
                [0, 1, 2].map(|i| (bank.mats[i].storage.clone(), bank.mats[i].bytes))
            };
            for (e, slot, _ptrs, ready, gpu) in jobs {
                self.shrink_into(layer, e, slot, &mats)?;
                ready.store(true, Ordering::Release);
                gpu.set_ready(slot, true);
                self.stats.lock().unwrap().prefetch_copies += 1;
            }
            return Ok(());
        }
        for (e, slot, ptrs, ready, gpu) in jobs {
            for (i, s) in self.slices[layer][e as usize].iter().enumerate() {
                let dst = unsafe { std::slice::from_raw_parts_mut(ptrs[i].0 as *mut u8, s.bytes) };
                if staged() {
                    let mut tmp = vec![0u8; s.bytes];
                    self.file.read_exact_at(&mut tmp, s.offset)?;
                    dst.copy_from_slice(&tmp);
                } else {
                    self.file.read_exact_at(dst, s.offset)?;
                }
            }
            ready.store(true, Ordering::Release);
            gpu.set_ready(slot, true);
            self.stats.lock().unwrap().prefetch_copies += 1;
        }
        Ok(())
    }

    fn take_staging(&self, bytes: usize) -> Result<Arc<candle_metal_kernels::metal::Buffer>> {
        if let Some(b) = self.staging.lock().unwrap().entry(bytes).or_default().pop() {
            return Ok(b);
        }
        Ok(self.metal.new_buffer_builder().with_size(bytes).with_label("shrink_staging").build()?)
    }

    fn put_staging(&self, bytes: usize, b: Arc<candle_metal_kernels::metal::Buffer>) {
        self.staging.lock().unwrap().entry(bytes).or_default().push(b);
    }

    fn take_sharp(&self, dtype: GgmlDType, elems: usize) -> Result<Arc<QMetalStorage>> {
        if let Some(b) = self.sharp_pool.lock().unwrap().entry((dtype, elems)).or_default().pop() {
            return Ok(b);
        }
        let st = QMetalStorage::zeros(&self.metal, elems, dtype)?;
        self.device.synchronize()?;
        Ok(Arc::new(st))
    }

    fn put_sharp(&self, dtype: GgmlDType, elems: usize, b: Arc<QMetalStorage>) {
        self.sharp_pool.lock().unwrap().entry((dtype, elems)).or_default().push(b);
    }

    /// Read one miss into three sharp stagings (the file's bytes as they are) and remember it as
    /// this forward's fresh expert; its Q2 slot stays not-ready until the squeeze.
    fn stage_sharp(&self, layer: usize, e: u32, slot: usize, weight: f32, squeeze: bool) -> Result<()> {
        let mut parts: Vec<Arc<QMetalStorage>> = Vec::with_capacity(3);
        for sl in self.slices[layer][e as usize].iter() {
            let elems: usize = sl.dims.iter().product();
            let st = self.take_sharp(sl.dtype, elems)?;
            let dst = unsafe { std::slice::from_raw_parts_mut(st.buffer().contents() as *mut u8, sl.bytes) };
            self.file.read_exact_at(dst, sl.offset)?;
            parts.push(st);
        }
        let parts: [Arc<QMetalStorage>; 3] = match parts.try_into() {
            Ok(p) => p,
            Err(_) => candle::bail!("an expert has three matrices"),
        };
        self.fresh.lock().unwrap().insert((layer, e), parts.clone());
        self.pending_squeeze.lock().unwrap().push((layer, e, slot, parts, weight, squeeze));
        Ok(())
    }

    /// After the GPU has drained: squeeze every pending miss from its sharp staging into its Q2
    /// slot, mark the slot ready, return the stagings to the pool.
    fn squeeze_pending(&self) -> Result<()> {
        let pending: Vec<(usize, u32, usize, [Arc<QMetalStorage>; 3], f32, bool)> = std::mem::take(&mut *self.pending_squeeze.lock().unwrap());
        if pending.is_empty() {
            return Ok(());
        }
        let Some(rq) = self.requant.as_ref() else { return Ok(()) };
        for (layer, e, slot, parts, weight, squeeze) in pending {
            // A heavy re-read: the bank already holds its copy; just give the stagings back.
            if !squeeze {
                for (i, sl) in self.slices[layer][e as usize].iter().enumerate() {
                    self.put_sharp(sl.dtype, sl.dims.iter().product(), parts[i].clone());
                }
                self.fresh.lock().unwrap().remove(&(layer, e));
                let _ = slot;
                continue;
            }
            // Below the admission line: used, done, let go. No squeeze, no slot.
            if admit_w().map_or(false, |line| weight < line) {
                for (i, sl) in self.slices[layer][e as usize].iter().enumerate() {
                    self.put_sharp(sl.dtype, sl.dims.iter().product(), parts[i].clone());
                }
                self.fresh.lock().unwrap().remove(&(layer, e));
                let mut banks = self.banks.lock().unwrap();
                if let Some(bank) = banks.get_mut(&layer) {
                    bank.slots.release(e);
                    bank.ready[slot].store(true, Ordering::Release);
                    bank.gpu.clear(e);
                    bank.gpu.set_ready(slot, false);
                }
                self.stats.lock().unwrap().stale_dropped += 0;
                continue;
            }
            let mats: [(Arc<QMetalStorage>, usize); 3] = {
                let banks = self.banks.lock().unwrap();
                let bank = &banks[&layer];
                [0, 1, 2].map(|i| (bank.mats[i].storage.clone(), bank.mats[i].bytes))
            };
            for (i, sl) in self.slices[layer][e as usize].iter().enumerate() {
                let elems: usize = sl.dims.iter().product();
                if q2_none() || (i == 2 && !shrink_down()) {
                    let dst = unsafe { std::slice::from_raw_parts_mut((mats[i].0.buffer().contents() as usize + slot * mats[i].1) as *mut u8, sl.bytes) };
                    let src = unsafe { std::slice::from_raw_parts(parts[i].buffer().contents() as *const u8, sl.bytes) };
                    dst.copy_from_slice(src);
                } else {
                    let Some(kind) = crate::requant::SrcKind::of(sl.dtype) else { candle::bail!("the GPU shrink takes Q3_K, Q4_K or Q5_K experts") };
                    let n_blocks = elems / crate::requant::QK_K;
                    rq.run(kind, parts[i].buffer(), 0, mats[i].0.buffer(), slot * mats[i].1, n_blocks)?;
                }
            }
            if repair_on() {
                self.measure_repair(layer, e, slot, &parts)?;
            }
            for (i, sl) in self.slices[layer][e as usize].iter().enumerate() {
                self.put_sharp(sl.dtype, sl.dims.iter().product(), parts[i].clone());
            }
            self.fresh.lock().unwrap().remove(&(layer, e));
            let banks = self.banks.lock().unwrap();
            if let Some(bank) = banks.get(&layer) {
                bank.ready[slot].store(true, Ordering::Release);
                bank.gpu.set_ready(slot, true);
            }
        }
        Ok(())
    }

    /// One expert's output y = down(silu(gate x) * up x) for input x (1, h), from three storages
    /// at `slot` (the sharp stagings at slot 0, or the bank at the expert's slot).
    fn expert_out(&self, x: &Tensor, layer: usize, ops: [MoeMatvec; 3], slot: u32) -> Result<Tensor> {
        let device = x.device();
        let slots = Tensor::from_vec(vec![slot], (1, 1), device)?;
        let [g, u, d] = ops;
        let gate = x.apply_op2_no_bwd(&slots, &g)?;
        let up = x.apply_op2_no_bwd(&slots, &u)?;
        let inter = gate.dim(2)?;
        let hidden = (candle_nn::ops::silu(&gate)? * &up)?.reshape((1, inter))?;
        let down = hidden.apply_op2_no_bwd(&slots, &d)?; // (1, 1, h)
        down.flatten_all()?.to_dtype(DType::F32)
    }

    /// Measure the squeeze's damage to one expert on the input it was used for, and keep it.
    fn measure_repair(&self, layer: usize, e: u32, slot: usize, parts: &[Arc<QMetalStorage>; 3]) -> Result<()> {
        let x = { self.last_x.lock().unwrap().get(&layer).cloned() };
        let Some(x) = x else { return Ok(()) };
        let sharp = [0, 1, 2].map(|i| self.fresh_op(&parts[i], layer, i, 1, i == 2));
        let y_sharp = self.expert_out(&x, layer, sharp, 0)?;
        let q2 = [0, 1, 2].map(|i| self.bank_op(layer, i, 1, i == 2));
        let y_q2 = self.expert_out(&x, layer, q2, slot as u32)?;
        let d = (y_sharp - y_q2)?;
        self.repair.lock().unwrap().insert((layer, e), d);
        Ok(())
    }

    /// A one-slot `mul_mv_id` over a sharp staging (a fresh miss, the file's own dtype).
    fn fresh_op(&self, st: &Arc<QMetalStorage>, layer: usize, which: usize, nei0: usize, rows_per_expert: bool) -> MoeMatvec {
        let sl = &self.slices[layer][0][which];
        MoeMatvec { bank: st.clone(), dtype: sl.dtype, n: sl.dims[0], k: sl.dims[1], slots: 1, bytes: sl.bytes, nei0, rows_per_expert }
    }

    /// Read one expert of `layer` from the file and shrink it on the GPU straight into `slot`
    /// of the bank's three matrices (`mats`: storage and bytes per slot). Returns when the slot
    /// holds the shrunk bytes.
    fn shrink_into(&self, layer: usize, e: u32, slot: usize, mats: &[(Arc<QMetalStorage>, usize); 3]) -> Result<()> {
        let Some(rq) = self.requant.as_ref() else { candle::bail!("shrink_into without the GPU shrink") };
        for (i, sl) in self.slices[layer][e as usize].iter().enumerate() {
            if q2_none() || (i == 2 && !shrink_down()) {
                // The down matrix keeps the file's bytes: read it straight into its slot.
                let dst = unsafe { std::slice::from_raw_parts_mut((mats[i].0.buffer().contents() as usize + slot * mats[i].1) as *mut u8, sl.bytes) };
                self.file.read_exact_at(dst, sl.offset)?;
                continue;
            }
            let Some(kind) = crate::requant::SrcKind::of(sl.dtype) else {
                candle::bail!("the GPU shrink takes Q3_K, Q4_K or Q5_K experts, this file has {:?}", sl.dtype)
            };
            let staging = self.take_staging(sl.bytes)?;
            let dst = unsafe { std::slice::from_raw_parts_mut(staging.contents() as *mut u8, sl.bytes) };
            self.file.read_exact_at(dst, sl.offset)?;
            let n_blocks = sl.dims.iter().product::<usize>() / crate::requant::QK_K;
            let shape_buf = if shaped() {
                let shapes = self.shape.lock().unwrap();
                shapes.get(&layer).filter(|(_, _, n)| *n >= 32).map(|(xs, hs, _)| if i == 2 { hs.clone() } else { xs.clone() })
            } else {
                None
            };
            match shape_buf {
                Some(sh) => rq.run_shaped(kind, &staging, 0, mats[i].0.buffer(), slot * mats[i].1, n_blocks, &sh, sl.dims[1], )?,
                None => rq.run(kind, &staging, 0, mats[i].0.buffer(), slot * mats[i].1, n_blocks)?,
            }
            self.put_staging(sl.bytes, staging);
        }
        Ok(())
    }

    /// Fold one token's shape into `layer`'s running shape: x² over the hidden size, hidden² over
    /// the intermediate size (averaged over the experts the token used). CPU side, ~20k floats.
    fn note_shape(&self, layer: usize, xs: &Tensor, hidden: &Tensor) -> Result<()> {
        let x: Vec<f32> = xs.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
        let h2: Vec<Vec<f32>> = hidden.to_dtype(DType::F32)?.to_vec2()?;
        let inter = h2.first().map_or(0, |r| r.len());
        let mut shapes = self.shape.lock().unwrap();
        let entry = match shapes.get(&layer) {
            Some(_) => shapes.get_mut(&layer).unwrap(),
            None => {
                let xb = self.metal.new_buffer_builder().with_size(x.len() * 4).with_label("shape_x").build()?;
                let hb = self.metal.new_buffer_builder().with_size(inter * 4).with_label("shape_h").build()?;
                unsafe {
                    std::ptr::write_bytes(xb.contents() as *mut u8, 0, x.len() * 4);
                    std::ptr::write_bytes(hb.contents() as *mut u8, 0, inter * 4);
                }
                shapes.insert(layer, (xb, hb, 0));
                shapes.get_mut(&layer).unwrap()
            }
        };
        let n = entry.2 as f32;
        let a = 1.0 / (n + 1.0);
        let xp = entry.0.contents() as *mut f32;
        let hp = entry.1.contents() as *mut f32;
        unsafe {
            for (i, v) in x.iter().enumerate() {
                let old = *xp.add(i);
                *xp.add(i) = old + a * (v * v - old);
            }
            for (i, row) in h2.iter().enumerate() {
                let w = 1.0 / h2.len() as f32;
                for (j, v) in row.iter().enumerate() {
                    let old = *hp.add(j);
                    let target = v * v;
                    // average over the experts used this token, then fold into the running mean
                    *hp.add(j) = old + a * w * (target - old) * if i == 0 { 1.0 } else { 1.0 };
                }
            }
        }
        entry.2 += 1;
        Ok(())
    }

    /// A layer's bank: three shared buffers, each `per_layer` expert slots of one matrix.
    fn make_bank(&self, layer: usize) -> Result<Bank> {
        let mut mats = Vec::with_capacity(3);
        for (i, s) in self.slices[layer][0].iter().enumerate() {
            let elems: usize = s.dims.iter().product();
            let (dtype, bytes) = if self.requant.is_some() && !q2_none() && (i < 2 || shrink_down()) {
                (GgmlDType::Q2K, elems / crate::requant::QK_K * crate::requant::Q2K_BLOCK_BYTES)
            } else {
                (s.dtype, s.bytes)
            };
            let storage = exact_zeros(&self.metal, elems * self.per_layer, dtype)?;
            let base = storage.buffer().contents() as usize;
            if base == 0 {
                candle::bail!("Metal gave a buffer the CPU cannot write to");
            }
            mats.push(BankMat {
                storage: Arc::new(storage),
                base,
                bytes,
                dtype,
                n: s.dims[0],
                k: s.dims[1],
            });
        }
        let mats: [BankMat; 3] = match mats.try_into() {
            Ok(m) => m,
            Err(_) => candle::bail!("a bank has three matrices"),
        };
        // `zeros` queues a GPU fill; let it run now, or it would wipe the experts written in afterwards.
        self.device.synchronize()?;
        let gpu = Arc::new(crate::gpu_sync::GpuBank::new(&self.metal, self.slices[layer].len(), self.per_layer, 8 * 16 + 32)?);
        let sharp = if self.sharp_n > 0 {
            let mut smats = Vec::with_capacity(3);
            for s in self.slices[layer][0].iter() {
                let elems: usize = s.dims.iter().product();
                let storage = QMetalStorage::zeros(&self.metal, elems * self.sharp_n, s.dtype)?;
                let base = storage.buffer().contents() as usize;
                if base == 0 {
                    candle::bail!("Metal gave a buffer the CPU cannot write to");
                }
                smats.push(BankMat { storage: Arc::new(storage), base, bytes: s.bytes, dtype: s.dtype, n: s.dims[0], k: s.dims[1] });
            }
            let smats: [BankMat; 3] = match smats.try_into() {
                Ok(m) => m,
                Err(_) => candle::bail!("a bank has three matrices"),
            };
            self.device.synchronize()?;
            Some(SharpBank { mats: smats, slots: Slots::new(self.sharp_n), ready: (0..self.sharp_n).map(|_| Arc::new(AtomicBool::new(true))).collect() })
        } else {
            None
        };
        Ok(Bank {
            mats,
            slots: Slots::new(self.per_layer),
            sharp,
            ready: (0..self.per_layer).map(|_| Arc::new(AtomicBool::new(true))).collect(),
            gpu,
            pinned: (0..self.per_layer).map(|_| AtomicBool::new(false)).collect(),
        })
    }

    /// Many fresh experts outside the bank, for a forward that needs more experts than a bank
    /// holds (reading a prompt touches most of them). All the reads go to the SSD at once, and
    /// `each` gets every expert the moment its bytes land, so the GPU computes while the rest of
    /// the layer is still being read. `each` is called on this thread, in arrival order.
    fn fresh_stream(&self, layer: usize, experts: &[u32], mut each: impl FnMut(usize, [Arc<QTensor>; 3]) -> Result<()>) -> Result<()> {
        let started = Instant::now();
        let (tx, rx) = std::sync::mpsc::channel::<(usize, Result<[Vec<u8>; 3]>)>();
        let mut build_seconds = 0.0;
        let mut waited = 0.0;
        std::thread::scope(|scope| -> Result<()> {
            scope.spawn(move || {
                experts.par_iter().enumerate().for_each_with(tx, |tx, (j, &e)| {
                    let read = |i: usize| -> Result<Vec<u8>> {
                        let s = &self.slices[layer][e as usize][i];
                        let mut b = vec![0u8; s.bytes];
                        self.file.read_exact_at(&mut b, s.offset)?;
                        Ok(b)
                    };
                    let bytes = (|| Ok([read(0)?, read(1)?, read(2)?]))();
                    let _ = tx.send((j, bytes));
                });
            });
            for _ in 0..experts.len() {
                let wait = Instant::now();
                let (j, bytes) = rx.recv().map_err(|e| candle::Error::Msg(format!("expert reader stopped: {e}")))?;
                waited += wait.elapsed().as_secs_f64();
                let bytes = bytes?;
                let build = Instant::now();
                let sl = &self.slices[layer][experts[j] as usize];
                let [a, b, c] = [0, 1, 2].map(|i| qtensor_from_ggml(sl[i].dtype, &bytes[i], sl[i].dims.clone(), &self.device).map(Arc::new));
                let mats = [a?, b?, c?];
                build_seconds += build.elapsed().as_secs_f64();
                each(j, mats)?;
            }
            Ok(())
        })?;
        let mut st = self.stats.lock().unwrap();
        st.fresh += experts.len() as u64;
        st.fresh_read_seconds += waited;
        st.fresh_build_seconds += build_seconds;
        st.load_seconds += started.elapsed().as_secs_f64();
        Ok(())
    }

    /// Makes sure every needed expert of one layer sits in the bank; returns expert -> slot.
    /// None when the bank is too small for this forward.
    fn is_hot(&self, layer: usize, e: u32) -> bool {
        self.sharp_n > 0 && self.hot[layer].contains(&e)
    }

    /// Where an expert goes with two kinds of slot: by its weight for this token when the weight
    /// line is set (and the expert is not already parked somewhere), else by the hot list.
    fn goes_sharp(&self, layer: usize, e: u32, weight: Option<f32>, bank: &Bank) -> bool {
        if self.sharp_n == 0 {
            return false;
        }
        if let Some(sb) = bank.sharp.as_ref() {
            if sb.slots.slot_of.contains_key(&e) {
                return true;
            }
        }
        if bank.slots.slot_of.contains_key(&e) {
            return false;
        }
        match (sharp_weight(), weight) {
            (Some(line), Some(w)) => w >= line,
            _ => self.is_hot(layer, e),
        }
    }

    /// The sharp bank's half of a forward: give the hot experts sharp slots, read the missing
    /// ones straight from the file (no shrink). Returns expert -> slot | SHARP_FLAG, or None when
    /// the sharp bank cannot hold them (the caller then treats them as cold).
    fn ensure_sharp(&self, layer: usize, bank: &mut Bank, hot: &[u32]) -> Result<Option<HashMap<u32, usize>>> {
        let Some(sb) = bank.sharp.as_mut() else { return Ok(None) };
        if hot.is_empty() {
            return Ok(Some(HashMap::new()));
        }
        if hot.len() > self.sharp_n {
            return Ok(None);
        }
        let ready = &sb.ready;
        let Some(plan) = sb.slots.assign(hot, |s| ready[s].load(Ordering::Acquire)) else { return Ok(None) };
        for &(_, slot) in &plan.hits {
            while !sb.ready[slot].load(Ordering::Acquire) {
                std::thread::yield_now();
            }
        }
        self.stats.lock().unwrap().hits += plan.hits.len() as u64;
        if !plan.misses.is_empty() {
            let places: [(usize, usize); 3] = [0, 1, 2].map(|i| (sb.mats[i].base, sb.mats[i].bytes));
            let reads: Vec<(usize, usize, u64)> = plan
                .misses
                .iter()
                .flat_map(|&(e, slot)| self.slices[layer][e as usize].iter().enumerate().map(move |(i, s)| (places[i].0 + slot * places[i].1, s.bytes, s.offset)).collect::<Vec<_>>())
                .collect();
            let started = Instant::now();
            reads.par_iter().try_for_each(|&(dst, len, offset)| {
                let dst = unsafe { std::slice::from_raw_parts_mut(dst as *mut u8, len) };
                self.file.read_exact_at(dst, offset)
            })?;
            let mut st = self.stats.lock().unwrap();
            st.loads += plan.misses.len() as u64;
            st.bytes_loaded += reads.iter().map(|r| r.1 as u64).sum::<u64>();
            st.load_seconds += started.elapsed().as_secs_f64();
        }
        Ok(Some(hot.iter().map(|&e| (e, sb.slots.slot_of[&e] | SHARP_FLAG)).collect()))
    }

    fn ensure(&self, layer: usize, needed_all: &[u32]) -> Result<Option<HashMap<u32, usize>>> {
        self.ensure_weighted(layer, needed_all, None)
    }

    fn ensure_weighted(&self, layer: usize, needed_all: &[u32], weights: Option<&HashMap<u32, f32>>) -> Result<Option<HashMap<u32, usize>>> {
        // A forward that needs more experts than the banks hold (reading a prompt) goes the fresh
        // way and must not make the banks here: their 8 GB would land on top of the prompt's
        // transient reads and push the GPU out of memory. Found 8 Oct: the two-kinds change had
        // moved this check below the bank's creation.
        if needed_all.len() > self.per_layer + self.sharp_n {
            return Ok(None);
        }
        let mut banks = self.banks.lock().unwrap();
        if !banks.contains_key(&layer) {
            banks.insert(layer, self.make_bank(layer)?);
        }
        let bank = banks.get_mut(&layer).unwrap();
        // Two kinds: by weight when the line is set, else the hot experts go to the sharp bank.
        let (hot, mut cold): (Vec<u32>, Vec<u32>) = needed_all.iter().partition(|&&e| self.goes_sharp(layer, e, weights.and_then(|m| m.get(&e).copied()), bank));
        let sharp_map = if self.sharp_n > 0 {
            match self.ensure_sharp(layer, bank, &hot)? {
                Some(m) => m,
                None => {
                    cold.extend(hot.iter().copied());
                    HashMap::new()
                }
            }
        } else {
            HashMap::new()
        };
        let needed: &[u32] = &cold;
        if needed.len() > self.per_layer {
            return Ok(None);
        }
        if needed.is_empty() {
            return Ok(Some(sharp_map));
        }
        // A slot still being filled by a prefetch cannot be taken; if too few are free for this
        // layer's misses, read them fresh instead, before the bank is touched.
        let ready = &bank.ready;
        let Some(plan) = bank.slots.assign(needed, |s| ready[s].load(Ordering::Acquire)) else {
            self.stats.lock().unwrap().crowded += 1;
            return Ok(None);
        };
        // S4: the GPU reads these slots later than now; keep the prefetch readers off them until
        // this layer is placed again for the next word.
        if self.s4.is_some() {
            for p in &bank.pinned {
                p.store(false, Ordering::Release);
            }
            for &(_, slot) in plan.hits.iter().chain(plan.misses.iter()) {
                bank.pinned[slot].store(true, Ordering::Release);
            }
        }
        // A prefetch may still be reading a hit in; wait for it rather than read it twice.
        let mut waited = 0.0;
        for &(_, slot) in &plan.hits {
            if !bank.ready[slot].load(Ordering::Acquire) {
                let wait = Instant::now();
                while !bank.ready[slot].load(Ordering::Acquire) {
                    std::thread::yield_now();
                }
                waited += wait.elapsed().as_secs_f64();
            }
        }
        {
            let mut st = self.stats.lock().unwrap();
            st.hits += plan.hits.len() as u64;
            st.prefetch_hits += plan.prefetch_hits;
            st.prefetch_wait_seconds += waited;
        }
        let misses = plan.misses;
        // Heavy reuse: a hit this token leans on is re-read sharp; its Q2 copy stays for the light ones.
        if self.requant.is_some() && use_sharp() {
            if let (Some(line), Some(w)) = (reread_w(), weights) {
                let heavy: Vec<(u32, usize)> = plan.hits.iter().copied().filter(|(e, _)| w.get(e).map_or(false, |&x| x >= line)).collect();
                if !heavy.is_empty() {
                    let started = Instant::now();
                    heavy.par_iter().try_for_each(|&(e, slot)| self.stage_sharp(layer, e, slot, w[&e], false))?;
                    let mut st = self.stats.lock().unwrap();
                    st.loads += heavy.len() as u64;
                    st.load_seconds += started.elapsed().as_secs_f64();
                }
            }
        }
        // The GPU's table follows: misses are placed but not ready until their bytes land.
        for &(_, slot) in &misses {
            bank.gpu.set_ready(slot, false);
        }
        bank.sync_table();
        if !misses.is_empty() && self.requant.is_some() && use_sharp() {
            let started = Instant::now();
            for &(_, slot) in &misses {
                bank.ready[slot].store(false, Ordering::Release);
            }
            misses.par_iter().try_for_each(|&(e, slot)| self.stage_sharp(layer, e, slot, weights.and_then(|m| m.get(&e).copied()).unwrap_or(1.0), true))?;
            let mut st = self.stats.lock().unwrap();
            st.loads += misses.len() as u64;
            st.bytes_loaded += misses.len() as u64 * self.slices[layer][0].iter().map(|s| s.bytes as u64).sum::<u64>();
            st.load_seconds += started.elapsed().as_secs_f64();
            // The slots are claimed but not ready; the squeeze at the next layer fills them.
            let mut map = bank.slots.slot_of.clone();
            map.extend(sharp_map);
            return Ok(Some(map));
        } else if !misses.is_empty() && self.requant.is_some() {
            let mats: [(Arc<QMetalStorage>, usize); 3] = [0, 1, 2].map(|i| (bank.mats[i].storage.clone(), bank.mats[i].bytes));
            let started = Instant::now();
            misses.par_iter().try_for_each(|&(e, slot)| self.shrink_into(layer, e, slot, &mats))?;
            let mut st = self.stats.lock().unwrap();
            st.loads += misses.len() as u64;
            st.bytes_loaded += misses.len() as u64 * self.slices[layer][0].iter().map(|s| s.bytes as u64).sum::<u64>();
            st.load_seconds += started.elapsed().as_secs_f64();
        } else if !misses.is_empty() {
            // All of this layer's misses go to the SSD at once: it serves several requests in
            // parallel far faster than one after another. Each matrix of each expert is one read.
            let places: [(usize, usize); 3] = [0, 1, 2].map(|i| (bank.mats[i].base, bank.mats[i].bytes));
            let reads: Vec<(usize, usize, u64)> = misses
                .iter()
                .flat_map(|&(e, slot)| {
                    self.slices[layer][e as usize]
                        .iter()
                        .enumerate()
                        .map(move |(i, s)| (places[i].0 + slot * places[i].1, s.bytes, s.offset))
                        .collect::<Vec<_>>()
                })
                .collect();
            let started = Instant::now();
            reads.par_iter().try_for_each(|&(dst, len, offset)| {
                // Each read fills its own slot of a shared buffer that no GPU work in flight uses.
                let dst = unsafe { std::slice::from_raw_parts_mut(dst as *mut u8, len) };
                if staged() {
                    // Lab: read into ordinary memory first, then copy; the shared buffer may be
                    // slower to write straight from the file.
                    let mut tmp = vec![0u8; len];
                    self.file.read_exact_at(&mut tmp, offset)?;
                    dst.copy_from_slice(&tmp);
                    Ok(())
                } else {
                    self.file.read_exact_at(dst, offset)
                }
            })?;
            let mut st = self.stats.lock().unwrap();
            st.loads += misses.len() as u64;
            st.bytes_loaded += reads.iter().map(|r| r.1 as u64).sum::<u64>();
            st.load_seconds += started.elapsed().as_secs_f64();
        }
        for &(_, slot) in &misses {
            bank.gpu.set_ready(slot, true);
        }
        let mut map = bank.slots.slot_of.clone();
        map.extend(sharp_map);
        Ok(Some(map))
    }

    /// Like `ensure`, but hands the miss reads back instead of doing them, so the caller can
    /// start the GPU on the hits first. Returns expert -> slot, the misses, and their reads
    /// (destination, length, file offset); None when the bank is too small or crowded.
    fn ensure_split(&self, layer: usize, needed: &[u32]) -> Result<Option<(HashMap<u32, usize>, Vec<(u32, usize)>, Vec<(usize, usize, u64)>)>> {
        if needed.len() > self.per_layer {
            return Ok(None);
        }
        let mut banks = self.banks.lock().unwrap();
        if !banks.contains_key(&layer) {
            banks.insert(layer, self.make_bank(layer)?);
        }
        let bank = banks.get_mut(&layer).unwrap();
        let ready = &bank.ready;
        let Some(plan) = bank.slots.assign(needed, |s| ready[s].load(Ordering::Acquire)) else {
            self.stats.lock().unwrap().crowded += 1;
            return Ok(None);
        };
        let mut waited = 0.0;
        for &(_, slot) in &plan.hits {
            if !bank.ready[slot].load(Ordering::Acquire) {
                let wait = Instant::now();
                while !bank.ready[slot].load(Ordering::Acquire) {
                    std::thread::yield_now();
                }
                waited += wait.elapsed().as_secs_f64();
            }
        }
        {
            let mut st = self.stats.lock().unwrap();
            st.hits += plan.hits.len() as u64;
            st.prefetch_hits += plan.prefetch_hits;
            st.prefetch_wait_seconds += waited;
        }
        let misses = plan.misses;
        for &(_, slot) in &misses {
            bank.gpu.set_ready(slot, false);
            // The prefetchers must not take a slot whose bytes are still to come.
            bank.ready[slot].store(false, Ordering::Release);
        }
        bank.sync_table();
        let places: [(usize, usize); 3] = [0, 1, 2].map(|i| (bank.mats[i].base, bank.mats[i].bytes));
        let reads: Vec<(usize, usize, u64)> = misses
            .iter()
            .flat_map(|&(e, slot)| {
                self.slices[layer][e as usize]
                    .iter()
                    .enumerate()
                    .map(move |(i, s)| (places[i].0 + slot * places[i].1, s.bytes, s.offset))
                    .collect::<Vec<_>>()
            })
            .collect();
        Ok(Some((bank.slots.slot_of.clone(), misses, reads)))
    }

    /// The second half of `ensure_split`: read the misses in, then mark their slots ready.
    fn read_misses(&self, layer: usize, misses: &[(u32, usize)], reads: &[(usize, usize, u64)]) -> Result<()> {
        if misses.is_empty() {
            return Ok(());
        }
        let started = Instant::now();
        reads.par_iter().try_for_each(|&(dst, len, offset)| {
            let dst = unsafe { std::slice::from_raw_parts_mut(dst as *mut u8, len) };
            self.file.read_exact_at(dst, offset)
        })?;
        {
            let mut st = self.stats.lock().unwrap();
            st.loads += misses.len() as u64;
            st.bytes_loaded += reads.iter().map(|r| r.1 as u64).sum::<u64>();
            st.load_seconds += started.elapsed().as_secs_f64();
        }
        let banks = self.banks.lock().unwrap();
        if let Some(bank) = banks.get(&layer) {
            for &(_, slot) in misses {
                bank.ready[slot].store(true, Ordering::Release);
                bank.gpu.set_ready(slot, true);
            }
        }
        Ok(())
    }

    /// S4: the GPU's view of `layer`'s bank, making the bank if the layer has none yet.
    fn gpu_bank(&self, layer: usize) -> Result<Arc<crate::gpu_sync::GpuBank>> {
        let mut banks = self.banks.lock().unwrap();
        if !banks.contains_key(&layer) {
            banks.insert(layer, self.make_bank(layer)?);
        }
        Ok(banks[&layer].gpu.clone())
    }

    /// S4: every layer's bank must exist before a word starts, so none is made mid-word while the
    /// GPU queue is waiting on the service thread. Called at layer 0; cheap once they exist.
    fn ensure_all_banks(&self) -> Result<()> {
        let mut banks = self.banks.lock().unwrap();
        if banks.len() >= self.layers {
            return Ok(());
        }
        for layer in 0..self.layers {
            if !banks.contains_key(&layer) {
                let bank = self.make_bank(layer)?;
                banks.insert(layer, bank);
            }
        }
        Ok(())
    }

    /// S4 service: wait for the GPU to publish a layer's picks, place and read its experts, hand
    /// the early guess to the readers, then let the GPU go on. Never touches the GPU queue.
    fn s4_service(&self, job: S4Job) {
        let Some(s4) = &self.s4 else { return };
        let dbg = std::env::var("COACHWHIP_S4_DEBUG").is_ok();
        let Ok(gpu) = self.gpu_bank(job.layer) else { return };
        let started = Instant::now();
        if dbg {
            eprintln!("s4: job layer {} seq {} received", job.layer, job.seq);
        }
        let mut picks: Option<Vec<u32>> = None;
        while picks.is_none() {
            picks = gpu.picks_tagged(job.n_ids + job.n_guess, (job.seq & 0xFFFF) as u32);
            if picks.is_none() && started.elapsed().as_secs() > 15 {
                eprintln!("coachwhip: parallel: layer {} never published its picks; releasing the GPU", job.layer);
                break;
            }
            std::hint::spin_loop();
        }
        if let Some(picks) = picks {
            if dbg {
                eprintln!("s4: layer {} flag seen after {:.1} ms", job.layer, started.elapsed().as_secs_f64() * 1e3);
            }
            self.reached(job.layer);
            let ids = vec![picks[..job.n_ids].to_vec()];
            self.learn(job.layer, &ids);
            let mut needed: Vec<u32> = ids[0].clone();
            needed.sort_unstable();
            needed.dedup();
            let placed = Instant::now();
            loop {
                match self.ensure(job.layer, &needed) {
                    Ok(Some(_)) => break,
                    Ok(None) => {
                        if placed.elapsed().as_secs() > 15 {
                            eprintln!("coachwhip: parallel: layer {} found no free slots; releasing the GPU", job.layer);
                            break;
                        }
                        std::thread::yield_now();
                    }
                    Err(e) => {
                        eprintln!("coachwhip: parallel: layer {}: {e}", job.layer);
                        break;
                    }
                }
            }
            if dbg {
                // Every needed expert must sit ready in a slot the GPU's table knows about.
                let banks = self.banks.lock().unwrap();
                if let Some(bank) = banks.get(&job.layer) {
                    for e in &needed {
                        match bank.slots.slot_of.get(e) {
                            Some(&slot) if bank.ready[slot].load(Ordering::Acquire) => {}
                            Some(&slot) => eprintln!("s4: MISSING layer {} expert {e}: slot {slot} not ready", job.layer),
                            None => eprintln!("s4: MISSING layer {} expert {e}: no slot", job.layer),
                        }
                    }
                }
            }
            if job.n_guess > 0 && job.layer + job.ahead < self.layers {
                self.prefetch_guess(job.layer + job.ahead, &picks[job.n_ids..], job.prefetch_k);
            }
            let mut st = self.stats.lock().unwrap();
            st.load_phase_seconds += started.elapsed().as_secs_f64();
        }
        s4.event.signal(job.seq);
        if dbg {
            eprintln!("s4: layer {} signalled seq {} after {:.1} ms", job.layer, job.seq, started.elapsed().as_secs_f64() * 1e3);
        }
        if job.layer + 1 == self.layers {
            let missing: u32 = (0..self.layers).filter_map(|l| self.banks.lock().unwrap().get(&l).map(|b| b.gpu.missing())).sum();
            if missing > 0 {
                eprintln!("coachwhip: parallel: {missing} expert picks so far had no slot when the GPU looked them up");
            }
        }
    }

    /// S6 measurement bookkeeping: one draft of `k` words, `accepted` of them matched the plain path.
    pub fn note_speculation(&self, k: usize, accepted: usize) {
        let mut st = self.stats.lock().unwrap();
        st.spec_passes += 1;
        st.spec_drafted += k as u64;
        st.spec_accepted += accepted as u64;
    }

    /// S6 draft mode: route only among experts already in the bank, read nothing, learn nothing.
    pub fn set_draft(&self, on: bool) {
        self.draft.store(on, Ordering::Release);
    }

    pub fn drafting(&self) -> bool {
        self.draft.load(Ordering::Acquire)
    }

    /// Foresight (`COACHWHIP_FORESIGHT=1`): during the draft pass, each layer's unrestricted route
    /// is handed to the readers a whole word before the real pass asks for it.
    pub fn set_foresee(&self, on: bool, step: u64) {
        self.foresee.store(on, Ordering::Release);
        self.foresee_step.store(step, Ordering::Release);
    }

    pub fn foreseeing(&self) -> bool {
        self.foresee.load(Ordering::Acquire)
    }

    /// The bank's map for the GPU: a mask that shuts out every expert not sitting ready in a slot
    /// (0 / -1e30, one per expert) and the slot each expert sits in (u32, one per expert).
    fn peek_tables(&self, layer: usize, top_k: usize, device: &Device) -> Result<(Tensor, Tensor)> {
        let (mask, table) = {
            let banks = self.banks.lock().unwrap();
            let Some(bank) = banks.get(&layer) else { candle::bail!("draft: layer {layer} has no bank yet") };
            let experts = self.slices[layer].len();
            let mut mask = vec![-1e30f32; experts];
            let mut table = vec![0u32; experts];
            let mut ready = 0;
            for (&e, &s) in bank.slots.slot_of.iter() {
                if bank.ready[s].load(Ordering::Acquire) {
                    mask[e as usize] = 0.0;
                    table[e as usize] = s as u32;
                    ready += 1;
                }
            }
            if ready < top_k {
                candle::bail!("draft: layer {layer} holds only {ready} ready experts")
            }
            (mask, table)
        };
        let n = mask.len();
        Ok((Tensor::from_vec(mask, n, device)?, Tensor::from_vec(table, n, device)?))
    }

    /// A layer's foresight list, kept on the GPU until `flush_foresight`.
    fn keep_foresight(&self, layer: usize, list: Tensor) {
        self.foresight_lists.lock().unwrap().push((layer, list));
    }

    /// After a peek pass: read every layer's foresight list back (one wait) and start the reads.
    pub fn flush_foresight(&self, top_k: usize) -> Result<()> {
        let lists: Vec<(usize, Tensor)> = std::mem::take(&mut *self.foresight_lists.lock().unwrap());
        for (layer, list) in lists {
            let ids: Vec<u32> = list.to_vec1()?;
            self.prefetch_guess(layer, &ids, top_k + ExpertStore::foresight_extra());
        }
        Ok(())
    }

    /// The experts of `layer` sitting in a ready slot right now, expert -> slot; None before the
    /// layer has a bank at all.
    pub fn resident_ready(&self, layer: usize) -> Option<HashMap<u32, usize>> {
        let banks = self.banks.lock().unwrap();
        let bank = banks.get(&layer)?;
        Some(bank.slots.slot_of.iter().filter(|(_, &s)| bank.ready[s].load(Ordering::Acquire)).map(|(&e, &s)| (e, s)).collect())
    }

    /// The toaster (`COACHWHIP_TOASTER=1`): after a layer has computed, its experts' slots become
    /// the first to be evicted.
    fn pop_used(&self, layer: usize, used: &[u32]) {
        if let Some(bank) = self.banks.lock().unwrap().get_mut(&layer) {
            bank.slots.pop(used);
        }
    }

    /// The experts of `layer` in the bank right now, before this layer's loads.
    fn resident(&self, layer: usize) -> Vec<u32> {
        let banks = self.banks.lock().unwrap();
        banks.get(&layer).map(|b| b.slots.slot_of.keys().copied().collect()).unwrap_or_default()
    }

    /// S6 measurement: compare the real route with the route a draft confined to the bank would
    /// take, and keep the score.
    fn measure_draft(&self, layer: usize, scores: &[f32], real: &[u32], top_k: usize) {
        let mut resident = self.resident(layer);
        resident.sort_by(|a, b| scores[*b as usize].partial_cmp(&scores[*a as usize]).unwrap_or(std::cmp::Ordering::Equal));
        let draft: Vec<u32> = resident.into_iter().take(top_k).collect();
        let agree = real.iter().filter(|e| draft.contains(e)).count();
        let mut st = self.stats.lock().unwrap();
        if layer == 0 {
            st.draft_word_ok = true;
        }
        st.draft_layers += 1;
        st.draft_agree += agree as f64 / top_k as f64;
        if agree == top_k {
            st.draft_layers_full += 1;
        } else {
            st.draft_word_ok = false;
        }
        if layer + 1 == self.slices.len() && st.draft_word_ok {
            st.draft_words_full += 1;
        }
    }

    /// One `mul_mv_id` dispatch over the sharp bank's matrix.
    fn bank_op_sharp(&self, layer: usize, which: usize, nei0: usize, rows_per_expert: bool) -> MoeMatvec {
        let banks = self.banks.lock().unwrap();
        let m = &banks[&layer].sharp.as_ref().expect("a sharp bank").mats[which];
        MoeMatvec { bank: m.storage.clone(), dtype: m.dtype, n: m.n, k: m.k, slots: self.sharp_n, bytes: m.bytes, nei0, rows_per_expert }
    }

    /// One `mul_mv_id` dispatch over a bank matrix.
    fn bank_op(&self, layer: usize, which: usize, nei0: usize, rows_per_expert: bool) -> MoeMatvec {
        let banks = self.banks.lock().unwrap();
        let m = &banks[&layer].mats[which];
        MoeMatvec {
            bank: m.storage.clone(),
            dtype: m.dtype,
            n: m.n,
            k: m.k,
            slots: self.per_layer,
            bytes: m.bytes,
            nei0,
            rows_per_expert,
        }
    }

    /// Learn from one layer's picks: add the step from the layer before to the route table at once,
    /// and keep the line for the log.
    /// Lab: append one record (layer, the MoE input, the router's scores) to the dump file.
    fn dump_record(&self, layer: usize, xs: &[f32], scores: &[f32]) {
        use std::io::Write;
        let mut d = self.dump.lock().unwrap();
        if let Some(f) = d.as_mut() {
            let mut buf = Vec::with_capacity(4 + 4 * (xs.len() + scores.len()));
            buf.extend_from_slice(&(layer as u32).to_le_bytes());
            for v in xs.iter().chain(scores.iter()) {
                buf.extend_from_slice(&v.to_le_bytes());
            }
            let _ = f.write_all(&buf);
        }
    }

    /// The words a forward pass is about to route, so the log's T line can name them.
    pub fn note_tokens(&self, tokens: Vec<u32>) {
        self.trail.lock().unwrap().tokens = tokens;
    }

    fn learn(&self, layer: usize, ids: &[Vec<u32>]) {
        let mut trail = self.trail.lock().unwrap();
        if layer == 0 {
            trail.lines.clear();
        } else if let Some((before_layer, before)) = trail.before.take() {
            if before_layer + 1 == layer && before.len() == ids.len() {
                count_step(&mut self.routes.lock().unwrap(), before_layer, &before, ids);
            }
        }
        let line = ids.iter().map(|t| t.iter().map(|e| e.to_string()).collect::<Vec<_>>().join(" ")).collect::<Vec<_>>().join(",");
        trail.lines.push(line);
        if layer + 1 == self.layers {
            let words = if trail.tokens.is_empty() { "-".to_string() } else { trail.tokens.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" ") };
            let block = format!("T {words}
{}
", trail.lines.join("
"));
            trail.blocks.push(block);
            trail.lines.clear();
        } else {
            trail.before = Some((layer, ids.to_vec()));
        }
    }

    /// Append the routes learned since the last save to this session's log file.
    fn save_learned(&self) {
        let Some(dir) = &self.learned else { return };
        let blocks: Vec<String> = std::mem::take(&mut self.trail.lock().unwrap().blocks);
        if blocks.is_empty() {
            return;
        }
        let saved = std::fs::create_dir_all(dir).and_then(|_| {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(dir.join(&self.learned_file))?;
            f.write_all(blocks.concat().as_bytes())
        });
        if let Err(e) = saved {
            eprintln!("coachwhip: could not save learned routes: {e}");
        }
    }

    /// Print the profile so far, save what was learned, and start counting afresh; the chat calls
    /// this after each answer.
    pub fn report_and_reset(&self) {
        self.print_report();
        self.save_learned();
        *self.stats.lock().unwrap() = Stats::default();
    }

    fn print_report(&self) {
        let s = self.stats.lock().unwrap();
        if s.hits + s.loads == 0 {
            return;
        }
        let requests = (s.hits + s.loads).max(1) as f64;
        eprintln!(
            "coachwhip: {} expert requests: bank {:.1}%, file {:.1}% ({} outside the bank); {:.2} GB read; {:.2} s loading; {:.2} ms per load; prefetched {} of which used {}",
            requests as u64,
            100.0 * s.hits as f64 / requests,
            100.0 * s.loads as f64 / requests,
            s.fresh,
            s.bytes_loaded as f64 / 1e9,
            s.load_seconds,
            1000.0 * s.load_seconds / s.loads.max(1) as f64,
            s.prefetch_copies,
            s.prefetch_hits
        );
        eprintln!(
            "coachwhip: waited {:.2} s on guesses still landing; dropped {} stale guessed reads; {} layers read fresh because the bank was crowded{}",
            s.prefetch_wait_seconds, s.stale_dropped, s.crowded,
            match memory_pressure() {
                2 => "; wrangler: macOS memory pressure is at WARNING, close other apps or start with a smaller --bank",
                4 => "; wrangler: macOS memory pressure is CRITICAL, start with a smaller --bank",
                _ => "",
            }
        );
        if s.fresh > 0 {
            eprintln!(
                "coachwhip: prompt experts: {} read outside the bank, {:.2} s reading, {:.2} s copying to the GPU",
                s.fresh, s.fresh_read_seconds, s.fresh_build_seconds
            );
        }
        if s.spec_passes > 0 {
            eprintln!(
                "coachwhip S6: {} drafts of {} words: {} of {} drafted words accepted ({:.1}%), {:.2} accepted per draft",
                s.spec_passes,
                s.spec_drafted / s.spec_passes,
                s.spec_accepted,
                s.spec_drafted,
                100.0 * s.spec_accepted as f64 / s.spec_drafted.max(1) as f64,
                s.spec_accepted as f64 / s.spec_passes as f64
            );
        }
        if profiling() && s.draft_layers > 0 {
            eprintln!(
                "coachwhip S6: bank-only route agrees with the real route on {:.1}% of experts; {:.1}% of layers fully; {} of {} words fully",
                100.0 * s.draft_agree / s.draft_layers as f64,
                100.0 * s.draft_layers_full as f64 / s.draft_layers as f64,
                s.draft_words_full,
                s.words
            );
        }
        if profiling() && s.reg_steps > 0 {
            let n = s.reg_steps as f64;
            eprintln!(
                "coachwhip register: predicted state is {:.1}% off the truth on average (carrying the input unchanged: {:.1}% off); by layer:",
                100.0 * s.reg_err / n,
                100.0 * s.reg_err_plain / n
            );
            let per = n / s.reg_err_by_layer.len().max(1) as f64;
            let line: Vec<String> = s.reg_err_by_layer.iter().zip(s.reg_plain_by_layer.iter()).enumerate().filter(|(l, _)| *l > 0).map(|(l, (e, p))| format!("{l}:{:.0}/{:.0}", 100.0 * e / per, 100.0 * p / per)).collect();
            eprintln!("coachwhip register by layer (register/plain, % off): {}", line.join(" "));
        }
        if profiling() && s.words > 2 {
            let words = (s.words - 1) as f64;
            eprintln!(
                "coachwhip cpu split per word: readback {:.1} ms, placing {:.1} ms, bookkeeping {:.1} ms (the rest of router+loads is the reads and the encode)",
                1000.0 * s.t_readback / words,
                1000.0 * s.t_place / words,
                1000.0 * s.t_books / words
            );
            eprintln!(
                "coachwhip profile: {} written words after the first, {:.1} ms per word in total, {:.1} ms in the expert layers: router+loads {:.1} (of which file reads {:.1}), expert maths {:.1}",
                s.words - 1,
                1000.0 * s.word_seconds / words,
                1000.0 * s.expert_seconds / words,
                1000.0 * s.load_phase_seconds / words,
                1000.0 * s.load_seconds / words,
                1000.0 * s.maths_seconds / words
            );
        }
    }
}

impl Drop for ExpertStore {
    fn drop(&mut self) {
        self.print_report();
        self.save_learned();
    }
}

/// `x (rows, k) , ids (tokens, nei0)` -> `(tokens, nei0, n)`: for each token and each of its
/// selected slots, one row of x times that slot's quantised matrix from the bank.
/// With `rows_per_expert`, x has one row per (token, slot) instead of one per token.
struct MoeMatvec {
    bank: Arc<QMetalStorage>,
    dtype: GgmlDType,
    n: usize,
    k: usize,
    slots: usize,
    bytes: usize,
    nei0: usize,
    rows_per_expert: bool,
}

impl CustomOp2 for MoeMatvec {
    fn name(&self) -> &'static str {
        "coachwhip-moe-matvec"
    }

    fn cpu_fwd(&self, _: &candle::CpuStorage, _: &Layout, _: &candle::CpuStorage, _: &Layout) -> Result<(candle::CpuStorage, Shape)> {
        candle::bail!("coachwhip-moe-matvec runs on Metal only")
    }

    fn metal_fwd(&self, x: &MetalStorage, xl: &Layout, ids: &MetalStorage, il: &Layout) -> Result<(MetalStorage, Shape)> {
        if !xl.is_contiguous() || !il.is_contiguous() {
            candle::bail!("coachwhip-moe-matvec needs contiguous inputs")
        }
        if x.dtype() != DType::F32 || ids.dtype() != DType::U32 {
            candle::bail!("coachwhip-moe-matvec: x must be f32 and ids u32")
        }
        let (tokens, nei0) = il.shape().dims2()?;
        if nei0 != self.nei0 {
            candle::bail!("coachwhip-moe-matvec: {} ids per token, expected {}", nei0, self.nei0)
        }
        let (rows, k) = xl.shape().dims2()?;
        if k != self.k || rows != if self.rows_per_expert { tokens * nei0 } else { tokens } {
            candle::bail!("coachwhip-moe-matvec: x is {rows}x{k}, expected k={} for {tokens} tokens", self.k)
        }
        let (ne11, nb11, nb12) = if self.rows_per_expert { (nei0, k * 4, nei0 * k * 4) } else { (1, 0, k * 4) };
        let device = x.device().clone();
        let elems = tokens * nei0 * self.n;
        let dst = device.new_buffer_builder().with_size_for(elems, DType::F32).with_label("moe_mv_id").build()?;
        {
            let encoder = device.command_encoder()?;
            crate::mv_id::matvec_id(
                device.device(),
                &encoder,
                device.kernels(),
                self.dtype.into(),
                (self.n, self.k, self.slots, self.bytes),
                self.bank.buffer(),
                x.buffer(),
                xl.start_offset() * 4,
                (ne11, nb11, nb12),
                ids.buffer(),
                il.start_offset() * 4,
                (nei0, tokens),
                &dst,
            )
            .map_err(candle::MetalError::from)?;
        }
        Ok((MetalStorage::new(dst, device.clone(), elems, DType::F32), Shape::from((tokens, nei0, self.n))))
    }
}

/// Router scores with every expert not `allowed` pushed far below the rest, so `route` picks
/// only among the allowed ones (S6 draft: the experts already in the bank).
pub fn restrict(logits: &Tensor, allowed: impl Fn(u32) -> bool) -> Result<Tensor> {
    let n = logits.dim(D::Minus1)?;
    let mask: Vec<f32> = (0..n as u32).map(|e| if allowed(e) { 0.0 } else { -1e30 }).collect();
    let mask = Tensor::from_vec(mask, n, logits.device())?.to_dtype(logits.dtype())?;
    logits.broadcast_add(&mask)
}

/// The router's choice for each token: the `top_k` likeliest experts from the router's scores,
/// best first, and their weights, rescaled to sum to 1 when `norm`.
pub fn route(logits: &Tensor, top_k: usize, norm: bool) -> Result<(Tensor, Tensor)> {
    let probs = candle_nn::ops::softmax_last_dim(logits)?;
    let top_ids = probs.arg_sort_last_dim(false)?.narrow(D::Minus1, 0, top_k)?.contiguous()?;
    let mut top_w = probs.gather(&top_ids, D::Minus1)?;
    if norm {
        top_w = top_w.broadcast_div(&top_w.sum_keepdim(D::Minus1)?)?;
    }
    Ok((top_ids, top_w))
}

/// A MoE layer whose experts live in the ExpertStore instead of on the device.
pub struct StreamedMoe {
    pub layer: usize,
    pub gate: Linear,
    /// A later layer's router (`ahead` layers on), run early on this layer's input to guess
    /// that layer's experts while there is still time to read them in.
    pub next_gate: Option<Linear>,
    /// The very next layer's router, for the register to chain through when `ahead` is 2.
    pub gate1: Option<Linear>,
    pub ahead: usize,
    pub store: Arc<ExpertStore>,
    pub top_k: usize,
    pub norm_topk_prob: bool,
    pub prefetch_k: usize,
}

impl StreamedMoe {
    /// The plain path: one quantised matmul per expert, experts read fresh from the file.
    fn reference(&self, xs: &Tensor, ids: &[Vec<u32>], top_w: &Tensor) -> Result<Tensor> {
        let device = xs.device();
        let weights: Vec<Vec<f32>> = top_w.to_vec2()?;
        let mut tokens_per_expert: BTreeMap<u32, (Vec<u32>, Vec<f32>)> = BTreeMap::new();
        for (t, (row_ids, row_w)) in ids.iter().zip(weights.iter()).enumerate() {
            for (e, w) in row_ids.iter().zip(row_w.iter()) {
                let entry = tokens_per_expert.entry(*e).or_default();
                entry.0.push(t as u32);
                entry.1.push(*w);
            }
        }
        let experts: Vec<u32> = tokens_per_expert.keys().copied().collect();
        let work: Vec<(Vec<u32>, Vec<f32>)> = tokens_per_expert.into_values().collect();
        // Each expert's result is kept and summed in expert order at the end, so the answer does
        // not depend on the order the reads happened to land in.
        let mut parts: Vec<Option<(Tensor, Tensor)>> = (0..experts.len()).map(|_| None).collect();
        self.store.fresh_stream(self.layer, &experts, |j, [g, u, d]| {
            let (tokens, w) = &work[j];
            let (gate, up, down) = (QMatMul::from_arc(g)?, QMatMul::from_arc(u)?, QMatMul::from_arc(d)?);
            let idx = Tensor::new(tokens.as_slice(), device)?;
            let x = xs.index_select(&idx, 0)?;
            let hidden = (candle_nn::ops::silu(&gate.forward(&x)?)? * up.forward(&x)?)?;
            let y = down.forward(&hidden)?;
            let w = Tensor::new(w.as_slice(), device)?.unsqueeze(1)?;
            parts[j] = Some((idx, y.broadcast_mul(&w)?));
            Ok(())
        })?;
        // One add for the whole layer: adding expert by expert would build a full copy of the
        // output for every expert, a gigabyte of short-lived GPU memory on a long prompt.
        let (idx, ys): (Vec<Tensor>, Vec<Tensor>) = parts.into_iter().flatten().unzip();
        if idx.is_empty() {
            return xs.zeros_like();
        }
        xs.zeros_like()?.index_add(&Tensor::cat(&idx, 0)?, &Tensor::cat(&ys, 0)?, 0)
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, s, h) = xs.dims3()?;
        let prof = profiling() && b * s == 1 && self.store.s4.is_none();
        if prof && self.layer == 0 && !self.store.drafting() {
            self.store.device.synchronize()?;
            let mut st = self.store.stats.lock().unwrap();
            if let Some(t) = st.word_started {
                if st.words >= 2 {
                    st.word_seconds += t.elapsed().as_secs_f64();
                }
            }
            st.word_started = Some(Instant::now());
            st.words += 1;
        }
        let started = Instant::now();
        let dtype = xs.dtype();
        let n = b * s;
        if n == 1 && !self.store.drafting() && self.store.s4.is_none() {
            self.store.reached(self.layer);
        }
        let xs = xs.reshape((n, h))?.to_dtype(DType::F32)?.contiguous()?;

        // S6 draft: the bank as it stands is the whole model for this word. No reads, no learning,
        // no prefetch, no bookkeeping; the LRU order is left alone too.
        if self.store.drafting() {
            if n != 1 {
                candle::bail!("draft: one word at a time")
            }
            // No round trip to the CPU: the bank's map goes up to the GPU as two small tensors, the
            // GPU picks among the resident experts and looks their slots up itself. The foresight
            // list (the unrestricted route) is kept on the GPU and read back once, after the pass.
            let device = xs.device();
            let logits = self.gate.forward(&xs)?;
            let (mask, table) = self.store.peek_tables(self.layer, self.top_k, device)?;
            if self.store.foreseeing() {
                let want = (self.top_k + ExpertStore::foresight_extra()).min(logits.dim(D::Minus1)?);
                let list = logits.arg_sort_last_dim(false)?.narrow(D::Minus1, 0, want)?.flatten_all()?.contiguous()?;
                self.store.keep_foresight(self.layer, list);
            }
            let (top_ids, top_w) = route(&logits.broadcast_add(&mask)?, self.top_k, self.norm_topk_prob)?;
            let slots = table.gather(&top_ids.flatten_all()?, 0)?.reshape((n, self.top_k))?;
            let gate = xs.apply_op2_no_bwd(&slots, &self.store.bank_op(self.layer, 0, self.top_k, false))?;
            let up = xs.apply_op2_no_bwd(&slots, &self.store.bank_op(self.layer, 1, self.top_k, false))?;
            let inter = gate.dim(2)?;
            let hidden = (candle_nn::ops::silu(&gate)? * &up)?.reshape((n * self.top_k, inter))?;
            let down = hidden.apply_op2_no_bwd(&slots, &self.store.bank_op(self.layer, 2, self.top_k, true))?;
            let out = down.broadcast_mul(&top_w.unsqueeze(2)?)?.sum(1)?;
            return out.reshape((b, s, h))?.to_dtype(dtype);
        }

        let logits = self.gate.forward(&xs)?;
        let (top_ids, top_w) = route(&logits, self.top_k, self.norm_topk_prob)?;
        if n == 1 && repair_on() {
            self.store.last_x.lock().unwrap().insert(self.layer, xs.clone());
        }

        // S4, the parallel path: nothing here waits for the GPU. The picks stay on the GPU; a tiny
        // kernel publishes them for the service thread, the maths is queued behind a wait on the
        // shared event, and a second kernel looks the slots up once the service thread has placed
        // the experts and signalled. The CPU moves straight on to the next layer.
        if n == 1 && self.store.s4.is_some() {
            let s4 = self.store.s4.as_ref().unwrap();
            if self.layer == 0 {
                self.store.ensure_all_banks()?;
            }
            let device = xs.device();
            let (guess, n_guess) = match &self.next_gate {
                Some(g) if self.prefetch_k > 0 => {
                    // With the register on, guess from the state it predicts, all on the GPU.
                    let x_in = if self.store.register_on() {
                        let mut x_hat = self.store.register_step_gpu(self.layer, &xs, &top_ids, &top_w)?;
                        if self.ahead >= 2 {
                            if let Some(g1) = &self.gate1 {
                                x_hat = self.store.register_chain(self.layer + 1, &x_hat, g1, self.top_k)?;
                            }
                        }
                        x_hat
                    } else {
                        xs.clone()
                    };
                    let scores = g.forward(&x_in)?;
                    let n_guess = 32.min(scores.dim(D::Minus1)?);
                    (scores.arg_sort_last_dim(false)?.narrow(D::Minus1, 0, n_guess)?.flatten_all()?.contiguous()?, n_guess)
                }
                _ => (Tensor::zeros(1, DType::U32, device)?, 0),
            };
            // Metal's rule: a command buffer must be committed after the CPU writes the data it
            // reads. The buffer holding the layer before's maths is committed by this layer's flush,
            // so wait here until the service thread has placed and written that layer's experts.
            // The GPU keeps its queue; only the CPU's run-ahead is capped at one layer.
            let last = s4.seq.load(Ordering::Acquire);
            let held = Instant::now();
            while s4.event.value() < last {
                if held.elapsed().as_secs() > 20 {
                    candle::bail!("parallel: layer {} waited too long for the layer before", self.layer)
                }
                std::hint::spin_loop();
            }
            let gpu = self.store.gpu_bank(self.layer)?;
            gpu.reset_flag();
            if std::env::var("COACHWHIP_S4_DEBUG").is_ok() {
                eprintln!("s4: encode layer {}", self.layer);
            }
            let seq = s4.seq.fetch_add(1, Ordering::AcqRel) + 1;
            let publish = crate::gpu_sync::Publish { bank: gpu.clone(), pipeline: s4.pipes.publish.clone(), n_guess, tag: (seq & 0xFFFF) as u32 };
            let _published = top_ids.apply_op2_no_bwd(&guess, &publish)?;
            crate::gpu_sync::flush()?;
            s4.event.gpu_wait(seq)?;
            let resolve = crate::gpu_sync::Resolve { bank: gpu.clone(), pipeline: s4.pipes.resolve.clone() };
            let slots = top_ids.apply_op1_no_bwd(&resolve)?;
            let gate = xs.apply_op2_no_bwd(&slots, &self.store.bank_op(self.layer, 0, self.top_k, false))?;
            let up = xs.apply_op2_no_bwd(&slots, &self.store.bank_op(self.layer, 1, self.top_k, false))?;
            let inter = gate.dim(2)?;
            let hidden = (candle_nn::ops::silu(&gate)? * &up)?.reshape((n * self.top_k, inter))?;
            let down = hidden.apply_op2_no_bwd(&slots, &self.store.bank_op(self.layer, 2, self.top_k, true))?;
            let out = down.broadcast_mul(&top_w.unsqueeze(2)?)?.sum(1)?;
            if let Some(tx) = s4.tx.lock().unwrap().as_ref() {
                let _ = tx.send(S4Job { layer: self.layer, seq, n_ids: self.top_k, n_guess, prefetch_k: self.prefetch_k, ahead: self.ahead });
            }
            return out.reshape((b, s, h))?.to_dtype(dtype);
        }
        // S6 measurement (profile mode, one word at a time): what a bank-only draft would pick.
        let draft_scores: Option<Vec<f32>> = if prof { Some(logits.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?) } else { None };
        if n == 1 && self.store.dump.lock().unwrap().is_some() {
            let x: Vec<f32> = xs.flatten_all()?.to_vec1()?;
            let s: Vec<f32> = logits.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
            self.store.dump_record(self.layer, &x, &s);
        }
        // The guess for a later layer is queued before the read-back below, so it costs no wait.
        let register = n == 1 && self.store.register_on();
        let early = match &self.next_gate {
            Some(g) if n == 1 && self.prefetch_k > 0 && !register => Some(g.forward(&xs)?),
            _ => None,
        };
        // Reading the picks back waits for the GPU to finish everything queued so far, so no
        // kernel is still reading a bank slot when `ensure` and the prefetchers overwrite it.
        // Keep this wait if this code ever changes.
        let t_rb = Instant::now();
        let ids: Vec<Vec<u32>> = top_ids.to_vec2()?;
        let t_rb = t_rb.elapsed().as_secs_f64();
        // The GPU has drained: the misses the layer before used sharp can be squeezed now.
        if use_sharp() {
            self.store.squeeze_pending()?;
        }
        let t_bk = Instant::now();
        if let Some(scores) = &draft_scores {
            self.store.measure_draft(self.layer, scores, &ids[0], self.top_k);
        }
        self.store.learn(self.layer, &ids);
        let mut needed: Vec<u32> = ids.iter().flatten().copied().collect();
        needed.sort_unstable();
        needed.dedup();
        let books_a = t_bk.elapsed().as_secs_f64();
        let loads_before = if prof { self.store.stats.lock().unwrap().load_seconds } else { 0.0 };
        let t_pl = Instant::now();
        // Split path: assign now, read the misses after the hits' maths has been queued.
        let split = n == 1 && split_maths();
        let mut pending: Option<(Vec<(u32, usize)>, Vec<(usize, usize, u64)>)> = None;
        let slot_of = if split {
            match self.store.ensure_split(self.layer, &needed)? {
                Some((slot_of, misses, reads)) => {
                    pending = Some((misses, reads));
                    Some(slot_of)
                }
                None => None,
            }
        } else if n == 1 && ((self.store.sharp_n > 0 && sharp_weight().is_some()) || (use_sharp() && (admit_w().is_some() || reread_w().is_some()))) {
            let w: Vec<f32> = top_w.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
            let weights: HashMap<u32, f32> = ids[0].iter().copied().zip(w.into_iter()).collect();
            self.store.ensure_weighted(self.layer, &needed, Some(&weights))?
        } else {
            self.store.ensure(self.layer, &needed)?
        };
        let t_pl = t_pl.elapsed().as_secs_f64();
        let loads_in = if prof { self.store.stats.lock().unwrap().load_seconds - loads_before } else { 0.0 };
        let t_bk2 = Instant::now();
        // Our register: predict the state this layer hands on, chain to the layer before the
        // target if need be, and run the target layer's router on the predicted state.
        let early = match (&self.next_gate, register) {
            (Some(g), true) if self.prefetch_k > 0 => {
                let mut x_hat = self.store.register_step(self.layer, &xs, &ids[0], &top_w)?;
                if self.ahead >= 2 {
                    if let Some(g1) = &self.gate1 {
                        x_hat = self.store.register_chain(self.layer + 1, &x_hat, g1, self.top_k)?;
                    }
                }
                Some(g.forward(&x_hat)?)
            }
            _ => early,
        };
        if let Some(logits) = early {
            let scores: Vec<f32> = logits.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
            if let Some(p_min) = q_rule() {
                // The probability rule: a candidate is read iff its router probability, calibrated
                // on the route dumps, says it is more likely needed than its bytes cost. On the
                // 122B that line sits at a router probability of about 1%.
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let ex: Vec<f32> = scores.iter().map(|&z| (z - m).exp()).collect();
                let sum: f32 = ex.iter().sum();
                let ranked = rank(&scores);
                let cands: Vec<u32> = ranked.iter().copied().take_while(|&e| ex[e as usize] / sum >= p_min).take(q_cap()).collect();
                if let (Some(line), true) = (sharp_weight(), self.store.sharp_n > 0) {
                    // By weight: the guess's share among the candidates stands in for the token's
                    // weight; above the line it is tagged for the sharp bank. Reader threads do the reads.
                    let tot: f32 = cands.iter().map(|&e| ex[e as usize]).sum::<f32>().max(1e-9);
                    let missing: Vec<u32> = {
                        let banks = self.store.banks.lock().unwrap();
                        let present = banks.get(&(self.layer + self.ahead));
                        cands.iter().copied().filter(|e| present.map_or(true, |b| !b.slots.slot_of.contains_key(e) && !b.sharp.as_ref().map_or(false, |sb| sb.slots.slot_of.contains_key(e)))).collect()
                    };
                    let tagged: Vec<u32> = missing.iter().map(|&e| if ex[e as usize] / tot >= line { e | SHARP_TAG } else { e }).collect();
                    for pair in tagged.chunks(2) {
                        self.store.request_prefetch(self.layer + self.ahead, pair.to_vec());
                    }
                } else {
                    let k = cands.len().max(1);
                    self.store.prefetch_guess(self.layer + self.ahead, &cands, k);
                }
            } else {
                self.store.prefetch_guess(self.layer + self.ahead, &rank(&scores), self.prefetch_k);
            }
        } else if n == 1 && self.prefetch_k > 0 {
            for ahead in 1..=self.store.ahead {
                if self.layer + ahead < self.store.slices.len() {
                    let predicted = self.store.predict_missing(self.layer, &needed, ahead, self.prefetch_k);
                    self.store.request_prefetch(self.layer + ahead, predicted);
                }
            }
        }
        let books_b = t_bk2.elapsed().as_secs_f64();
        if prof {
            self.store.device.synchronize()?;
            let mut st = self.store.stats.lock().unwrap();
            if st.words > 1 {
                st.load_phase_seconds += started.elapsed().as_secs_f64();
                st.t_readback += t_rb;
                st.t_place += (t_pl - loads_in).max(0.0);
                st.t_books += books_a + books_b;
            }
        }
        let maths_started = Instant::now();
        let device = xs.device();

        let out = if let (Some(slot_of), Some((misses, reads))) = (&slot_of, pending.take()) {
            // The resident experts first, queued and flushed so the GPU works while the CPU reads.
            let miss_set: HashSet<u32> = misses.iter().map(|&(e, _)| e).collect();
            let row = &ids[0];
            let part = |positions: &[usize]| -> Result<Option<Tensor>> {
                if positions.is_empty() {
                    return Ok(None);
                }
                let k = positions.len();
                let slots: Vec<u32> = positions.iter().map(|&p| slot_of[&row[p]] as u32).collect();
                let slots = Tensor::from_vec(slots, (1, k), device)?;
                let idx = Tensor::new(positions.iter().map(|&p| p as u32).collect::<Vec<_>>(), device)?;
                let w = top_w.index_select(&idx, 1)?; // (1, k)
                let gate = xs.apply_op2_no_bwd(&slots, &self.store.bank_op(self.layer, 0, k, false))?;
                let up = xs.apply_op2_no_bwd(&slots, &self.store.bank_op(self.layer, 1, k, false))?;
                let inter = gate.dim(2)?;
                let hidden = (candle_nn::ops::silu(&gate)? * &up)?.reshape((k, inter))?;
                let down = hidden.apply_op2_no_bwd(&slots, &self.store.bank_op(self.layer, 2, k, true))?;
                Ok(Some(down.broadcast_mul(&w.unsqueeze(2)?)?.sum(1)?))
            };
            let hit_pos: Vec<usize> = (0..row.len()).filter(|&p| !miss_set.contains(&row[p])).collect();
            let miss_pos: Vec<usize> = (0..row.len()).filter(|&p| miss_set.contains(&row[p])).collect();
            let hits_out = part(&hit_pos)?;
            if hits_out.is_some() && !misses.is_empty() {
                crate::gpu_sync::flush()?;
            }
            self.store.read_misses(self.layer, &misses, &reads)?;
            let miss_out = part(&miss_pos)?;
            match (hits_out, miss_out) {
                (Some(a), Some(b)) => (a + b)?,
                (Some(a), None) | (None, Some(a)) => a,
                (None, None) => xs.zeros_like()?,
            }
        } else if let (Some(slot_of), true) = (&slot_of, n == 1 && use_sharp() && self.store.requant.is_some()) {
            // Use sharp, then squeeze: fresh misses compute from their stagings, the rest from the bank.
            let row = &ids[0];
            let fresh: HashMap<u32, [Arc<QMetalStorage>; 3]> = {
                let f = self.store.fresh.lock().unwrap();
                row.iter().filter_map(|e| f.get(&(self.layer, *e)).map(|p| (*e, p.clone()))).collect()
            };
            let mut out: Option<Tensor> = None;
            let mut add = |t: Tensor| -> Result<()> {
                out = Some(match out.take() {
                    Some(a) => (a + t)?,
                    None => t,
                });
                Ok(())
            };
            // the bank part
            let bank_pos: Vec<usize> = (0..row.len()).filter(|&p| !fresh.contains_key(&row[p])).collect();
            if !bank_pos.is_empty() {
                let k = bank_pos.len();
                let slots: Vec<u32> = bank_pos.iter().map(|&p| (slot_of[&row[p]] & !SHARP_FLAG) as u32).collect();
                let slots = Tensor::from_vec(slots, (1, k), device)?;
                let idx = Tensor::new(bank_pos.iter().map(|&p| p as u32).collect::<Vec<_>>(), device)?;
                let w = top_w.index_select(&idx, 1)?;
                let gate = xs.apply_op2_no_bwd(&slots, &self.store.bank_op(self.layer, 0, k, false))?;
                let up = xs.apply_op2_no_bwd(&slots, &self.store.bank_op(self.layer, 1, k, false))?;
                let inter = gate.dim(2)?;
                let hidden = (candle_nn::ops::silu(&gate)? * &up)?.reshape((k, inter))?;
                let mut down = hidden.apply_op2_no_bwd(&slots, &self.store.bank_op(self.layer, 2, k, true))?; // (1, k, h)
                if repair_on() {
                    let h = down.dim(2)?;
                    let rep = self.store.repair.lock().unwrap();
                    let rows: Vec<Tensor> = bank_pos
                        .iter()
                        .map(|&p| rep.get(&(self.layer, row[p])).cloned().map(|t| t.to_dtype(down.dtype())).transpose())
                        .collect::<Result<Vec<Option<Tensor>>>>()?
                        .into_iter()
                        .map(|t| t.map_or_else(|| Tensor::zeros(h, down.dtype(), device), Ok))
                        .collect::<Result<Vec<Tensor>>>()?;
                    let corr = Tensor::stack(&rows, 0)?.unsqueeze(0)?; // (1, k, h)
                    down = (down + corr)?;
                }
                add(down.broadcast_mul(&w.unsqueeze(2)?)?.sum(1)?)?;
            }
            // each fresh miss, one slot wide, from its sharp staging
            for p in 0..row.len() {
                let Some(parts) = fresh.get(&row[p]) else { continue };
                let slots = Tensor::from_vec(vec![0u32], (1, 1), device)?;
                let idx = Tensor::new(vec![p as u32], device)?;
                let w = top_w.index_select(&idx, 1)?;
                let gate = xs.apply_op2_no_bwd(&slots, &self.store.fresh_op(&parts[0], self.layer, 0, 1, false))?;
                let up = xs.apply_op2_no_bwd(&slots, &self.store.fresh_op(&parts[1], self.layer, 1, 1, false))?;
                let inter = gate.dim(2)?;
                let hidden = (candle_nn::ops::silu(&gate)? * &up)?.reshape((1, inter))?;
                let down = hidden.apply_op2_no_bwd(&slots, &self.store.fresh_op(&parts[2], self.layer, 2, 1, true))?;
                add(down.broadcast_mul(&w.unsqueeze(2)?)?.sum(1)?)?;
            }
            match out {
                Some(t) => t,
                None => xs.zeros_like()?,
            }
        } else if let (Some(slot_of), true) = (&slot_of, n == 1 && self.store.sharp_n > 0) {
            // Two kinds of slot: the sharp experts and the small ones each get their dispatches; the sums add.
            let row = &ids[0];
            let part = |positions: &[usize], sharp: bool| -> Result<Option<Tensor>> {
                if positions.is_empty() {
                    return Ok(None);
                }
                let k = positions.len();
                let slots: Vec<u32> = positions.iter().map(|&p| (slot_of[&row[p]] & !SHARP_FLAG) as u32).collect();
                let slots = Tensor::from_vec(slots, (1, k), device)?;
                let idx = Tensor::new(positions.iter().map(|&p| p as u32).collect::<Vec<_>>(), device)?;
                let w = top_w.index_select(&idx, 1)?;
                let op = |which: usize, nei0: usize, rpe: bool| if sharp { self.store.bank_op_sharp(self.layer, which, nei0, rpe) } else { self.store.bank_op(self.layer, which, nei0, rpe) };
                let gate = xs.apply_op2_no_bwd(&slots, &op(0, k, false))?;
                let up = xs.apply_op2_no_bwd(&slots, &op(1, k, false))?;
                let inter = gate.dim(2)?;
                let hidden = (candle_nn::ops::silu(&gate)? * &up)?.reshape((k, inter))?;
                let down = hidden.apply_op2_no_bwd(&slots, &op(2, k, true))?;
                Ok(Some(down.broadcast_mul(&w.unsqueeze(2)?)?.sum(1)?))
            };
            let sharp_pos: Vec<usize> = (0..row.len()).filter(|&p| slot_of[&row[p]] & SHARP_FLAG != 0).collect();
            let small_pos: Vec<usize> = (0..row.len()).filter(|&p| slot_of[&row[p]] & SHARP_FLAG == 0).collect();
            match (part(&sharp_pos, true)?, part(&small_pos, false)?) {
                (Some(a), Some(b)) => (a + b)?,
                (Some(a), None) | (None, Some(a)) => a,
                (None, None) => xs.zeros_like()?,
            }
        } else if let Some(slot_of) = slot_of {
            let slots: Vec<u32> = ids.iter().flatten().map(|e| slot_of[e] as u32).collect();
            let slots = Tensor::from_vec(slots, (n, self.top_k), device)?;
            let gate = xs.apply_op2_no_bwd(&slots, &self.store.bank_op(self.layer, 0, self.top_k, false))?; // (n, k, inter)
            let up = xs.apply_op2_no_bwd(&slots, &self.store.bank_op(self.layer, 1, self.top_k, false))?;
            let inter = gate.dim(2)?;
            let hidden = (candle_nn::ops::silu(&gate)? * &up)?.reshape((n * self.top_k, inter))?;
            let down = hidden.apply_op2_no_bwd(&slots, &self.store.bank_op(self.layer, 2, self.top_k, true))?; // (n, k, h)
            if n == 1 && shaped() && self.store.requant.is_some() {
                self.store.note_shape(self.layer, &xs, &hidden)?;
            }
            down.broadcast_mul(&top_w.unsqueeze(2)?)?.sum(1)?
        } else {
            self.reference(&xs, &ids, &top_w)?
        };
        if n == 1 && toaster() {
            self.store.pop_used(self.layer, &needed);
        }
        if prof {
            self.store.device.synchronize()?;
            let mut st = self.store.stats.lock().unwrap();
            if st.words > 1 {
                st.expert_seconds += started.elapsed().as_secs_f64();
                st.maths_seconds += maths_started.elapsed().as_secs_f64();
            }
        }
        out.reshape((b, s, h))?.to_dtype(dtype)
    }
}
