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
use std::sync::atomic::{AtomicBool, Ordering};
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

struct Bank {
    mats: [BankMat; 3],
    slots: Slots,
    /// False while the prefetch thread is still reading an expert into the slot.
    ready: Vec<Arc<AtomicBool>>,
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
    /// Experts predicted and fetched ahead for the next layer; 0 turns prefetch off.
    pub prefetch: usize,
    /// How many layers ahead to predict.
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
        Self { bank: 44, prefetch: 2, ahead: 1, readers: 3, routes: None, learned: None, profile: false }
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
    words: u64,
    word_started: Option<Instant>,
    word_seconds: f64,
    expert_seconds: f64,
    load_phase_seconds: f64,
    maths_seconds: f64,
}

static PROFILE: AtomicBool = AtomicBool::new(false);

fn profiling() -> bool {
    PROFILE.load(Ordering::Relaxed)
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
    prefetch_tx: Mutex<Option<mpsc::Sender<(usize, Vec<u32>)>>>,
}

impl ExpertStore {
    /// Reads the file's table of contents only; experts are read when a token first asks for them.
    pub fn new(ct: &gguf_file::Content, device: &Device, layers: usize, path: &Path, settings: &Settings) -> Result<Arc<Self>> {
        let Device::Metal(metal) = device else {
            candle::bail!("Coachwhip needs a Metal device (Apple silicon)")
        };
        PROFILE.store(settings.profile, Ordering::Relaxed);
        let per_layer = settings.bank.max(1);

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
        let experts = slices[0].len();
        let per_expert_mb = slices[0][0].iter().map(|s| s.bytes).sum::<usize>() as f64 / 1e6;
        eprintln!(
            "coachwhip: {layers} layers x {experts} experts, {per_expert_mb:.1} MB each; bank {per_layer} slots per layer ({:.2} GB)",
            per_layer as f64 * layers as f64 * per_expert_mb / 1e3
        );
        let dirs: Vec<&Path> = settings.routes.iter().chain(settings.learned.iter()).map(|p| p.as_path()).collect();
        let routes = load_routes(&dirs, layers);
        eprintln!("coachwhip: route table for {} (layer, expert) pairs", routes.len());
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
        });
        // The prefetch thread reads predicted experts into free slots while the GPU works on the
        // current layer. It holds only a weak handle, so the store still shuts down when the model does.
        let (tx, rx) = mpsc::channel::<(usize, Vec<u32>)>();
        *store.prefetch_tx.lock().unwrap() = Some(tx);
        let rx = Arc::new(Mutex::new(rx));
        for _ in 0..settings.readers {
            let weak: Weak<Self> = Arc::downgrade(&store);
            let rx = rx.clone();
            std::thread::spawn(move || loop {
                let job = rx.lock().unwrap().recv();
                let Ok((layer, experts)) = job else { break };
                let Some(st) = weak.upgrade() else { break };
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
            .filter(|e| present.map_or(true, |b| !b.slots.slot_of.contains_key(e)))
            .take(k)
            .collect()
    }

    /// Hand the next layer's predicted experts to the prefetch thread.
    fn request_prefetch(&self, layer: usize, predicted: Vec<u32>) {
        if predicted.is_empty() {
            return;
        }
        if let Some(tx) = self.prefetch_tx.lock().unwrap().as_ref() {
            let _ = tx.send((layer, predicted));
        }
    }

    /// Prefetch thread: reserve free slots for the predicted experts, then read them in with the
    /// bank unlocked, so the main thread is never held up by a read it did not ask for.
    fn prefetch(&self, layer: usize, predicted: &[u32]) -> Result<()> {
        let mut jobs: Vec<(u32, [(usize, usize); 3], Arc<AtomicBool>)> = Vec::new();
        {
            let mut banks = self.banks.lock().unwrap();
            let Some(bank) = banks.get_mut(&layer) else { return Ok(()) };
            let ready = &bank.ready;
            for (e, slot) in bank.slots.reserve(predicted, |s| ready[s].load(Ordering::Acquire)) {
                ready[slot].store(false, Ordering::Release);
                let ptrs = [0, 1, 2].map(|i| (bank.mats[i].base + slot * bank.mats[i].bytes, bank.mats[i].bytes));
                jobs.push((e, ptrs, ready[slot].clone()));
            }
        }
        for (e, ptrs, ready) in jobs {
            for (i, s) in self.slices[layer][e as usize].iter().enumerate() {
                let dst = unsafe { std::slice::from_raw_parts_mut(ptrs[i].0 as *mut u8, s.bytes) };
                self.file.read_exact_at(dst, s.offset)?;
            }
            ready.store(true, Ordering::Release);
            self.stats.lock().unwrap().prefetch_copies += 1;
        }
        Ok(())
    }

    /// A layer's bank: three shared buffers, each `per_layer` expert slots of one matrix.
    fn make_bank(&self, layer: usize) -> Result<Bank> {
        let mut mats = Vec::with_capacity(3);
        for s in self.slices[layer][0].iter() {
            let elems: usize = s.dims.iter().product();
            let storage = QMetalStorage::zeros(&self.metal, elems * self.per_layer, s.dtype)?;
            let base = storage.buffer().contents() as usize;
            if base == 0 {
                candle::bail!("Metal gave a buffer the CPU cannot write to");
            }
            mats.push(BankMat {
                storage: Arc::new(storage),
                base,
                bytes: s.bytes,
                dtype: s.dtype,
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
        Ok(Bank {
            mats,
            slots: Slots::new(self.per_layer),
            ready: (0..self.per_layer).map(|_| Arc::new(AtomicBool::new(true))).collect(),
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
    fn ensure(&self, layer: usize, needed: &[u32]) -> Result<Option<HashMap<u32, usize>>> {
        if needed.len() > self.per_layer {
            return Ok(None);
        }
        let mut banks = self.banks.lock().unwrap();
        if !banks.contains_key(&layer) {
            banks.insert(layer, self.make_bank(layer)?);
        }
        let bank = banks.get_mut(&layer).unwrap();
        // A slot still being filled by a prefetch cannot be taken; if too few are free for this
        // layer's misses, read them fresh instead, before the bank is touched.
        let ready = &bank.ready;
        let Some(plan) = bank.slots.assign(needed, |s| ready[s].load(Ordering::Acquire)) else {
            return Ok(None);
        };
        // A prefetch may still be reading a hit in; wait for it rather than read it twice.
        for &(_, slot) in &plan.hits {
            while !bank.ready[slot].load(Ordering::Acquire) {
                std::thread::yield_now();
            }
        }
        {
            let mut st = self.stats.lock().unwrap();
            st.hits += plan.hits.len() as u64;
            st.prefetch_hits += plan.prefetch_hits;
        }
        let misses = plan.misses;
        if !misses.is_empty() {
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
                self.file.read_exact_at(dst, offset)
            })?;
            let mut st = self.stats.lock().unwrap();
            st.loads += misses.len() as u64;
            st.bytes_loaded += reads.iter().map(|r| r.1 as u64).sum::<u64>();
            st.load_seconds += started.elapsed().as_secs_f64();
        }
        Ok(Some(bank.slots.slot_of.clone()))
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
            let block = format!("T -\n{}\n", trail.lines.join("\n"));
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
        if s.fresh > 0 {
            eprintln!(
                "coachwhip: prompt experts: {} read outside the bank, {:.2} s reading, {:.2} s copying to the GPU",
                s.fresh, s.fresh_read_seconds, s.fresh_build_seconds
            );
        }
        if profiling() && s.words > 2 {
            let words = (s.words - 1) as f64;
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
        let prof = profiling() && b * s == 1;
        if prof && self.layer == 0 {
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
        let xs = xs.reshape((n, h))?.to_dtype(DType::F32)?.contiguous()?;

        let (top_ids, top_w) = route(&self.gate.forward(&xs)?, self.top_k, self.norm_topk_prob)?;
        // Reading the picks back waits for the GPU to finish everything queued so far, so no
        // kernel is still reading a bank slot when `ensure` and the prefetchers overwrite it.
        // Keep this wait if this code ever changes.
        let ids: Vec<Vec<u32>> = top_ids.to_vec2()?;
        self.store.learn(self.layer, &ids);
        let mut needed: Vec<u32> = ids.iter().flatten().copied().collect();
        needed.sort_unstable();
        needed.dedup();
        let slot_of = self.store.ensure(self.layer, &needed)?;
        if n == 1 && self.prefetch_k > 0 {
            for ahead in 1..=self.store.ahead {
                if self.layer + ahead < self.store.slices.len() {
                    let predicted = self.store.predict_missing(self.layer, &needed, ahead, self.prefetch_k);
                    self.store.request_prefetch(self.layer + ahead, predicted);
                }
            }
        }
        if prof {
            self.store.device.synchronize()?;
            let mut st = self.store.stats.lock().unwrap();
            if st.words > 1 {
                st.load_phase_seconds += started.elapsed().as_secs_f64();
            }
        }
        let maths_started = Instant::now();
        let device = xs.device();

        let out = if let Some(slot_of) = slot_of {
            let slots: Vec<u32> = ids.iter().flatten().map(|e| slot_of[e] as u32).collect();
            let slots = Tensor::from_vec(slots, (n, self.top_k), device)?;
            let gate = xs.apply_op2_no_bwd(&slots, &self.store.bank_op(self.layer, 0, self.top_k, false))?; // (n, k, inter)
            let up = xs.apply_op2_no_bwd(&slots, &self.store.bank_op(self.layer, 1, self.top_k, false))?;
            let inter = gate.dim(2)?;
            let hidden = (candle_nn::ops::silu(&gate)? * &up)?.reshape((n * self.top_k, inter))?;
            let down = hidden.apply_op2_no_bwd(&slots, &self.store.bank_op(self.layer, 2, self.top_k, true))?; // (n, k, h)
            down.broadcast_mul(&top_w.unsqueeze(2)?)?.sum(1)?
        } else {
            self.reference(&xs, &ids, &top_w)?
        };
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
