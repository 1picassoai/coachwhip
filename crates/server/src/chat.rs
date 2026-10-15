//! The chat: the model loads once and stays hot; a page on loopback sends each prompt in.
//!
//! One thread, no framework - `std::net::TcpListener` and a string of HTML. Nothing leaves the
//! machine: the page is served on 127.0.0.1 only. The answer streams: each word goes out as an
//! HTTP chunk the moment it is written.
//!
//! A chat is one long conversation the model has already read: a follow-up reads only its own new
//! words, then the model carries on from where it stopped.
//!
//! The same server also speaks the OpenAI chat API (`/v1/chat/completions`, `/v1/models`), so
//! coding tools on this machine can use the model. Those clients send the whole conversation every
//! time; when it begins with what the model has already read, only the new part is read.

use anyhow::Result;
use std::cell::RefCell;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::rc::Rc;

use candle::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use coachwhip_engine::model::{Model, Progress, Step};
use tokenizers::Tokenizer;

/// Largest request body accepted: a prompt of a few hundred pages.
const MAX_BODY: usize = 1 << 20;

/// Longest piece of a prompt read in one forward pass.
const PROMPT_CHUNK_DEFAULT: usize = 512;
fn fast_chat() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("COACHWHIP_FAST_CHAT").map(|v| v == "1").unwrap_or(false))
}
/// Lab (`COACHWHIP_PROMPT_CHUNK=n`): smaller pieces when the experts are all resident and memory is tight.
fn prompt_chunk() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("COACHWHIP_PROMPT_CHUNK").ok().and_then(|v| v.parse().ok()).filter(|&n| n > 0).unwrap_or(PROMPT_CHUNK_DEFAULT))
}

/// When a chat would grow past this many words, it is summarised first to make room.
pub const CONTEXT_LIMIT_MAX: usize = 16384;
/// The most a chat can ever hold (the reserved space may set less).
pub const CONTEXT_LIMIT: usize = CONTEXT_LIMIT_MAX;
/// The chat limit: the reserved conversation space when the wrangler set one, else 16384.
pub fn context_limit() -> usize {
    std::env::var("COACHWHIP_CONV_TOKENS").ok().and_then(|v| v.parse().ok()).filter(|&n: &usize| n > 0).map_or(CONTEXT_LIMIT_MAX, |n: usize| n.min(CONTEXT_LIMIT_MAX))
}

/// Longest summary written when making room.
const SUMMARY_TOKENS: usize = 500;

const SUMMARY_ASK: &str = "Summarise our conversation so far in at most 300 words, for someone who will continue it: what was asked, what was decided, and every name, file, function or fact the next answer will need. Plain text, no preamble.";

/// The start of every gpt-oss chat: the model's own default system message from its chat
/// template, at its default reasoning level.
const HARMONY_SYSTEM: &str = "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\n\nReasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>";

/// The chat format the model was trained on: ChatML (the Qwen families) or harmony (gpt-oss).
struct Fmt {
    harmony: bool,
}

impl Fmt {
    fn of(model: &Model) -> Self {
        Self { harmony: model.harmony() }
    }

    /// The words that end the assistant's turn. A stop word is never part of the answer.
    fn stops(&self, tokenizer: &Tokenizer) -> Vec<u32> {
        // Harmony: <|end|> only closes a message (the thinking ends with it); <|return|> ends the answer.
        let names: &[&str] = if self.harmony { &["<|return|>", "<|call|>"] } else { &["<|im_end|>"] };
        let vocab = tokenizer.get_vocab(true);
        names.iter().filter_map(|n| vocab.get(*n).copied()).collect()
    }

    /// What a new chat opens with, before its first turn.
    fn start(&self) -> &'static str {
        if self.harmony { HARMONY_SYSTEM } else { "" }
    }

    /// What closes the last answer before the next turn (the stop word itself was never read).
    fn close(&self) -> &'static str {
        if self.harmony { "<|end|>" } else { "<|im_end|>\n" }
    }

    fn user(&self, text: &str) -> String {
        if self.harmony { format!("<|start|>user<|message|>{text}<|end|>") } else { format!("<|im_start|>user\n{text}<|im_end|>\n") }
    }

    /// Opens the assistant's turn. gpt-oss with thinking on picks its own channel (analysis first);
    /// with thinking off it goes straight to its final channel.
    fn assistant(&self, model: &Model, think: bool) -> String {
        match (self.harmony, think) {
            (true, true) => "<|start|>assistant".to_string(),
            (true, false) => "<|start|>assistant<|channel|>final<|message|>".to_string(),
            _ => format!("<|im_start|>assistant\n{}", model.assistant_open(think)),
        }
    }
}

/// What the page shows of a gpt-oss answer that thinks: its analysis channel as a think block,
/// then its final channel. Format markers never show; one still arriving holds the text back.
fn harmony_view(raw: &str) -> String {
    let t = raw
        .replace("<|channel|>analysis<|message|>", "<think>\n")
        .replace("<|end|><|start|>assistant<|channel|>final<|message|>", "\n</think>\n\n")
        .replace("<|channel|>final<|message|>", "");
    match t.find("<|") {
        Some(i) => t[..i].to_string(),
        None => t,
    }
}

/// Separates the streamed text from the closing stats line.
pub const STATS_MARK: &str = "\u{1}STATS:";

/// A progress line sent while the prompt is read, before any text: `\u{1}PROG:<json>\n`.
pub const PROG_MARK: &str = "\u{1}PROG:";

/// Ends the stream when the answer could not be finished; the page shows what follows it.
pub const ERROR_MARK: &str = "\u{1}ERROR:";

/// After the GPU has run out of memory once, Metal does not recover in this process.
fn gpu_out_of_memory(e: &anyhow::Error) -> bool {
    let m = format!("{e:#}");
    m.contains("OutOfMemory") || m.contains("Insufficient Memory")
}

pub fn serve(port: u16, model: &mut Model, model_name: &str, tokenizer: Tokenizer, device: &Device, max_new: usize, sampling: Sampling) -> Result<()> {
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    println!("\n  coachwhip chat on http://127.0.0.1:{port} (model hot, loopback only)\n");
    let mut session = Session::default();
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        if let Err(e) = handle(stream, model, model_name, &tokenizer, device, max_new, &sampling, &mut session) {
            eprintln!("  request failed: {e}");
        }
    }
    Ok(())
}

/// Where the model is in the current chat.
#[derive(Default)]
pub struct Session {
    /// Words the model has read so far, its own answers included.
    pos: usize,
    /// The last word of an answer cut off at the length limit: written, but not yet read back.
    pending: Option<u32>,
    /// Every word the model has read so far, in order: `pos` of them.
    fed: Vec<u32>,
    /// Who wrote the conversation the model holds: the page continues only its own.
    by_api: bool,
}

pub struct Answer {
    /// New words read for this question; the earlier chat was not read again.
    pub prompt_tokens: usize,
    /// Words in the chat after this answer.
    pub context: usize,
    pub fresh: bool,
    /// The chat was summarised to make room before this answer.
    pub compacted: bool,
    pub written: usize,
    pub read_tps: f64,
    pub write_tps: f64,
    /// The model ended its answer itself, rather than hitting the length limit.
    pub finished: bool,
}

/// Runs the model and hands every new piece of text to `emit` as soon as it exists.
#[allow(clippy::too_many_arguments)]

/// Foresight (lab): COACHWHIP_FORESIGHT=N drafts N words from the bank before each real word, only
/// to start the reads those words will need; the text is unchanged.
fn foresight() -> usize {
    std::env::var("COACHWHIP_FORESIGHT").ok().and_then(|v| v.parse().ok()).unwrap_or(0)
}

pub fn generate(
    model: &mut Model,
    tokenizer: &Tokenizer,
    device: &Device,
    session: &mut Session,
    new_chat: bool,
    prompt: &str,
    think: bool,
    max_new: usize,
    sampling: &Sampling,
    progress: Option<Progress>,
    emit: &mut dyn FnMut(&str) -> Result<()>,
) -> Result<Answer> {
    let think = think && model.thinks();
    let fmt = Fmt::of(model);
    // The template's way of switching thinking on or off is in what opens the assistant's turn.
    let open = fmt.assistant(model, think);
    let encode = |text: String| -> Result<Vec<u32>> { Ok(tokenizer.encode(text, true).map_err(anyhow::Error::msg)?.get_ids().to_vec()) };
    let stops = fmt.stops(tokenizer);
    let tell = progress.map(|p| Rc::new(RefCell::new(p)));

    // A follow-up closes the last answer and adds the new question; nothing earlier is read again.
    let turn = format!("{}{open}", fmt.user(prompt));
    let mut tokens = Vec::new();
    let mut fresh = new_chat || session.pos == 0 || session.by_api;
    let mut summary = None;
    if !fresh {
        tokens.extend(session.pending);
        tokens.extend(encode(format!("{}{turn}", fmt.close()))?);
        // Too long to go on: the model summarises the chat, and the summary opens a new one.
        if session.pos + tokens.len() + max_new > context_limit() {
            summary = Some(summarise(model, tokenizer, device, session, &stops, &tell)?);
            fresh = true;
        }
    }
    if fresh {
        model.clear_kv_cache();
        *session = Session::default();
        tokens = encode(match &summary {
            Some(s) => format!("{}{}{open}", fmt.start(), fmt.user(&format!("Summary of our conversation so far, for context:\n{s}\n\nNow: {prompt}"))),
            None => format!("{}{turn}", fmt.start()),
        })?;
    }

    let a = answer(model, tokenizer, device, session, tokens, fresh, summary.is_some(), think, max_new, sampling, &tell, emit);
    session.by_api = false;
    a
}


/// Compaction reuse (`COACHWHIP_SNAPSHOTS=1`): conversation states saved on the SSD, keyed by the exact
/// words they cover, kept to a disk budget, the least used first out.
mod snaps {
    use std::io::{Read, Write};
    use std::path::PathBuf;
    pub struct Entry { pub tokens: Vec<u32>, pub file: PathBuf, pub bytes: u64, pub hits: u64, pub used: u64 }
    pub struct Store { pub dir: PathBuf, pub budget: u64, pub list: Vec<Entry>, pub clock: u64 }
    pub fn on() -> bool { std::env::var("COACHWHIP_SNAPSHOTS").map(|v| v == "1").unwrap_or(false) }
    impl Store {
        pub fn open() -> Store {
            let dir = std::env::var("COACHWHIP_SNAP_DIR").map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into())).join(".coachwhip").join("snapshots"));
            let _ = std::fs::create_dir_all(&dir);
            let gb: f64 = std::env::var("COACHWHIP_SNAP_GB").ok().and_then(|v| v.parse().ok()).unwrap_or(2.0);
            let mut list = Vec::new();
            if let Ok(rd) = std::fs::read_dir(&dir) {
                for e in rd.flatten() {
                    let p = e.path();
                    if p.extension().map(|x| x == "tok").unwrap_or(false) {
                        if let Ok(b) = std::fs::read(&p) {
                            let tokens: Vec<u32> = b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                            let file = p.with_extension("state");
                            let bytes = std::fs::metadata(&file).map(|m| m.len()).unwrap_or(0);
                            if bytes > 0 { list.push(Entry { tokens, file, bytes, hits: 0, used: 0 }); }
                        }
                    }
                }
            }
            Store { dir, budget: (gb * 1e9) as u64, list, clock: 0 }
        }
        /// The longest saved conversation the prompt begins with (at least `min` words, shorter than the prompt).
        pub fn find(&mut self, prompt: &[u32], min: usize) -> Option<usize> {
            let mut best: Option<usize> = None;
            for (i, e) in self.list.iter().enumerate() {
                let n = e.tokens.len();
                if n >= min && n < prompt.len() && prompt[..n] == e.tokens[..] && best.map_or(true, |b| self.list[b].tokens.len() < n) {
                    best = Some(i);
                }
            }
            if let Some(i) = best { self.clock += 1; self.list[i].hits += 1; self.list[i].used = self.clock; }
            best
        }
        pub fn read(&self, i: usize) -> std::io::Result<Vec<(Vec<usize>, Vec<f32>)>> {
            let mut b = Vec::new();
            std::fs::File::open(&self.list[i].file)?.read_to_end(&mut b)?;
            let mut at = 0usize;
            let mut u64_at = |b: &[u8], at: &mut usize| -> u64 { let v = u64::from_le_bytes(b[*at..*at + 8].try_into().unwrap()); *at += 8; v };
            let parts = u64_at(&b, &mut at) as usize;
            let mut out = Vec::with_capacity(parts);
            for _ in 0..parts {
                let rank = u64_at(&b, &mut at) as usize;
                let dims: Vec<usize> = (0..rank).map(|_| u64_at(&b, &mut at) as usize).collect();
                let n: usize = if rank == 0 { 0 } else { dims.iter().product() };
                let vals: Vec<f32> = b[at..at + 4 * n].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                at += 4 * n;
                out.push((dims, vals));
            }
            Ok(out)
        }
        pub fn save(&mut self, tokens: &[u32], parts: &[(Vec<usize>, Vec<f32>)]) -> std::io::Result<()> {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            tokens.hash(&mut h);
            let name = format!("{:016x}", h.finish());
            let file = self.dir.join(format!("{name}.state"));
            let mut w = std::io::BufWriter::new(std::fs::File::create(&file)?);
            w.write_all(&(parts.len() as u64).to_le_bytes())?;
            for (dims, vals) in parts {
                w.write_all(&(dims.len() as u64).to_le_bytes())?;
                for d in dims { w.write_all(&(*d as u64).to_le_bytes())?; }
                for v in vals { w.write_all(&v.to_le_bytes())?; }
            }
            w.flush()?;
            let tok: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
            std::fs::write(self.dir.join(format!("{name}.tok")), tok)?;
            let bytes = std::fs::metadata(&file)?.len();
            self.clock += 1;
            self.list.retain(|e| e.tokens != tokens);
            self.list.push(Entry { tokens: tokens.to_vec(), file, bytes, hits: 0, used: self.clock });
            // over budget: the least used go first (fewest hits, then oldest)
            while self.list.iter().map(|e| e.bytes).sum::<u64>() > self.budget && self.list.len() > 1 {
                let (i, _) = self.list.iter().enumerate().min_by_key(|(_, e)| (e.hits, e.used)).unwrap();
                let e = self.list.remove(i);
                let _ = std::fs::remove_file(&e.file);
                let _ = std::fs::remove_file(e.file.with_extension("tok"));
            }
            Ok(())
        }
    }
    pub fn store() -> &'static std::sync::Mutex<Store> {
        static S: std::sync::OnceLock<std::sync::Mutex<Store>> = std::sync::OnceLock::new();
        S.get_or_init(|| std::sync::Mutex::new(Store::open()))
    }
}

/// Answers a whole conversation sent at once, as the OpenAI API does. When it begins with what the
/// model has already read, only the rest is read; otherwise the model starts afresh.
#[allow(clippy::too_many_arguments)]
pub fn generate_full(
    model: &mut Model,
    tokenizer: &Tokenizer,
    device: &Device,
    session: &mut Session,
    prompt: Vec<u32>,
    max_new: usize,
    sampling: &Sampling,
    emit: &mut dyn FnMut(&str) -> Result<()>,
) -> Result<Answer> {
    if prompt.len() + max_new > context_limit() {
        anyhow::bail!("the conversation and its answer would be over {} tokens; start a new one", context_limit());
    }
    let known = session.fed.len();
    let mut fresh = known == 0 || prompt.len() <= known || !prompt.starts_with(&session.fed);
    if fresh {
        model.clear_kv_cache();
        *session = Session::default();
        // A conversation that begins with words already saved starts from that snapshot.
        if snaps::on() {
            let mut st = snaps::store().lock().unwrap();
            if let Some(i) = st.find(&prompt, 256) {
                let n = st.list[i].tokens.len();
                match st.read(i).map_err(anyhow::Error::from).and_then(|parts| model.load_state(&parts).map_err(anyhow::Error::from)) {
                    Ok(true) => {
                        session.fed = prompt[..n].to_vec();
                        session.pos = n;
                        fresh = false;
                        eprintln!("  snapshot: restored {n} words, reading only the {} new", prompt.len() - n);
                    }
                    Ok(false) => {}
                    Err(e) => { eprintln!("  snapshot: could not restore ({e}); reading afresh"); model.clear_kv_cache(); *session = Session::default(); }
                }
            }
        }
    } else {
        // The word an earlier answer was cut off on is part of what the client sent back.
        session.pending = None;
    }
    let tokens = prompt[session.fed.len()..].to_vec();
    let think = model.harmony();
    let a = answer(model, tokenizer, device, session, tokens, fresh, false, think, max_new, sampling, &None, emit);
    session.by_api = true;
    // Kept for the next conversation that begins the same way.
    if snaps::on() && a.is_ok() && session.fed.len() >= 256 {
        if let Ok(Some(parts)) = model.save_state() {
            if let Err(e) = snaps::store().lock().unwrap().save(&session.fed, &parts) {
                eprintln!("  snapshot: could not save ({e})");
            }
        }
    }
    a
}

/// Reads `tokens` after what the session has read, then writes the answer, streaming it to `emit`.
#[allow(clippy::too_many_arguments)]
fn answer(
    model: &mut Model,
    tokenizer: &Tokenizer,
    device: &Device,
    session: &mut Session,
    tokens: Vec<u32>,
    fresh: bool,
    compacted: bool,
    think: bool,
    max_new: usize,
    sampling: &Sampling,
    tell: &Option<Rc<RefCell<Progress>>>,
    emit: &mut dyn FnMut(&str) -> Result<()>,
) -> Result<Answer> {
    // Lab (`COACHWHIP_QOS=1`): this thread at user-interactive priority, so macOS wakes it at once
    // when the GPU finishes a word.
    if std::env::var("COACHWHIP_QOS").map(|v| v == "1").unwrap_or(false) {
        extern "C" { fn pthread_set_qos_class_self_np(qos: u32, rel: i32) -> i32; }
        unsafe { pthread_set_qos_class_self_np(0x21, 0); }
    }
    let fmt = Fmt::of(model);
    let stops = fmt.stops(tokenizer);
    // gpt-oss thinking: the channels are read from the format markers, so keep them when decoding.
    let channels = fmt.harmony && think;
    let mut sampler = LogitsProcessor::from_sampling(42, sampling.clone());
    let t0 = std::time::Instant::now();
    let logits = read_prompt(model, device, session.pos, &tokens, tell)?;
    session.fed.extend(&tokens);
    let base = session.pos + tokens.len();
    let mut next = sampler.sample(&logits)?;
    let read = t0.elapsed().as_secs_f64();

    // The answer so far; an end-of-turn word is never part of it.
    let mut out = if stops.contains(&next) { Vec::new() } else { vec![next] };
    let mut sent = 0usize;
    let mut stream_text = |out: &[u32], sent: &mut usize| -> Result<()> {
        let full = if channels {
            harmony_view(&tokenizer.decode(out, false).map_err(anyhow::Error::msg)?)
        } else {
            tokenizer.decode(out, true).map_err(anyhow::Error::msg)?
        };
        // A token can be half of a multi-byte character; wait until the text is whole.
        if full.ends_with('\u{FFFD}') {
            return Ok(());
        }
        // With thinking off the model still opens and closes an empty think block; skip it.
        let start = if !think && full.starts_with("<think>") {
            match full.find("</think>") {
                Some(i) => {
                    let after = &full[i + "</think>".len()..];
                    i + "</think>".len() + (after.len() - after.trim_start_matches('\n').len())
                }
                None => return Ok(()),
            }
        } else {
            0
        };
        let from = start.max(*sent);
        // A later decode can re-space earlier text; wait for a clean place to carry on from.
        if !full.is_char_boundary(from) {
            return Ok(());
        }
        if full.len() > from {
            emit(&full[from..])?;
            *sent = full.len();
        }
        Ok(())
    };
    stream_text(&out, &mut sent)?;
    let t1 = std::time::Instant::now();
    // S6 (lab): COACHWHIP_SPECULATE=K drafts K words from the bank and checks them in one pass;
    // COACHWHIP_SPECULATE_MEASURE=K only measures, writing the answer the plain way.
    let speculate: usize = std::env::var("COACHWHIP_SPECULATE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let measure: usize = std::env::var("COACHWHIP_SPECULATE_MEASURE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let greedy = matches!(sampling, Sampling::ArgMax);
    while out.len() < max_new && !stops.contains(&next) {
        if speculate > 0 && greedy && max_new - out.len() > speculate {
            if let Some(seq) = model.speculate(next, base + out.len() - 1, speculate)? {
                for &w in &seq {
                    next = w;
                    if stops.contains(&w) {
                        break;
                    }
                    out.push(w);
                    stream_text(&out, &mut sent)?;
                }
                continue;
            }
        }
        if measure > 0 && greedy && max_new - out.len() >= measure {
            if let Some((plain, _accepted)) = model.speculate_measure(next, base + out.len() - 1, measure)? {
                for &w in &plain {
                    next = w;
                    if stops.contains(&w) {
                        break;
                    }
                    out.push(w);
                    stream_text(&out, &mut sent)?;
                }
                continue;
            }
        }
        if foresight() > 0 {
            model.foresee(next, base + out.len() - 1, foresight())?;
        }
        // Lab (`COACHWHIP_FAST_CHAT=1`): the word goes in from the CPU (no GPU round trip before the
        // pass) and, when greedy, the GPU picks the next one and only 4 bytes come back.
        let fast = fast_chat() && device.is_metal();
        let ta = std::time::Instant::now();
        let logits = model.forward(&Tensor::new(&[next], if fast { &Device::Cpu } else { device })?.unsqueeze(0)?, base + out.len() - 1)?.squeeze(0)?;
        let tb = std::time::Instant::now();
        next = if fast && greedy { coachwhip_engine::fuse_kernel::argmax(&logits)?.flatten_all()?.to_vec1::<u32>()?[0] } else { sampler.sample(&logits)? };
        let tc = std::time::Instant::now();
        if !stops.contains(&next) {
            out.push(next);
            stream_text(&out, &mut sent)?;
        }
        if std::env::var("COACHWHIP_CHAT_TIME").is_ok() {
            static T: std::sync::Mutex<[f64; 4]> = std::sync::Mutex::new([0.0; 4]);
            let mut g = T.lock().unwrap();
            g[0] += (tb - ta).as_secs_f64(); g[1] += (tc - tb).as_secs_f64(); g[2] += tc.elapsed().as_secs_f64(); g[3] += 1.0;
            if g[3] as u64 % 100 == 0 {
                eprintln!("coachwhip chat time per word over {}: queue {:.2} ms, pick (GPU wait) {:.2} ms, decode+send {:.2} ms", g[3], 1000.0 * g[0] / g[3], 1000.0 * g[1] / g[3], 1000.0 * g[2] / g[3]);
            }
        }
    }
    let write = t1.elapsed().as_secs_f64();
    let written = out.len();
    // Every word of the answer was read back, except the last one when the length limit cut it off.
    let finished = stops.contains(&next);
    let read_back = if finished { out.len() } else { out.len() - 1 };
    session.pos = base + read_back;
    session.pending = if finished { None } else { out.last().copied() };
    session.fed.extend(&out[..read_back]);
    Ok(Answer {
        prompt_tokens: tokens.len(),
        context: session.pos,
        fresh,
        compacted,
        finished,
        written,
        read_tps: tokens.len() as f64 / read.max(1e-9),
        // The first word comes out of reading the prompt; the rate counts the ones written after it.
        write_tps: written.saturating_sub(1) as f64 / write.max(1e-9),
    })
}

/// Reads a prompt that continues the chat at `pos`, and returns the logits for the next word.
/// The chunker: a long prompt is read in pieces, so the memory a piece needs stays bounded.
/// While it reads, every finished layer is reported, counted across all the pieces, so the page
/// is never silent.
fn read_prompt(model: &mut Model, device: &Device, pos: usize, tokens: &[u32], tell: &Option<Rc<RefCell<Progress>>>) -> Result<Tensor> {
    // The pieces shrink as the chat grows, so the attention scores of a piece (piece x chat) never
    // need more memory than a full piece at 4K of chat.
    let piece_at = |at: usize| -> usize { let full = prompt_chunk(); if at <= 4096 { full } else { (full * 4096 / at).max(32) } };
    let mut bounds = Vec::new();
    let mut at = 0usize;
    while at < tokens.len() {
        let n = piece_at(pos + at + piece_at(pos + at)).min(tokens.len() - at);
        bounds.push((at, n));
        at += n;
    }
    let pieces = bounds.len();
    let read = (|| -> Result<Tensor> {
    let mut logits = None;
    for (i, &(start, n)) in bounds.iter().enumerate() {
        let piece = &tokens[start..start + n];
        model.set_progress(tell.clone().map(|t| -> Progress {
            Box::new(move |step| {
                if let Step::Layer(layer, of) = step {
                    (t.borrow_mut())(Step::Layer(i * of + layer, pieces * of))
                }
            })
        }));
        logits = Some(model.forward(&Tensor::new(piece, device)?.unsqueeze(0)?, pos + start)?.squeeze(0)?);
    }
    logits.ok_or_else(|| anyhow::anyhow!("empty prompt"))
    })();
    // Taken back on every path: the hook holds the page's connection open.
    model.set_progress(None);
    read
}

/// Asks the model, in the chat as it stands, for a summary of the chat; the chat is then thrown
/// away and the summary opens the next one. Greedy, so it is the same every time.
fn summarise(model: &mut Model, tokenizer: &Tokenizer, device: &Device, session: &Session, stops: &[u32], tell: &Option<Rc<RefCell<Progress>>>) -> Result<String> {
    let mut tokens: Vec<u32> = session.pending.into_iter().collect();
    let fmt = Fmt::of(model);
    let ask = format!("{}{}{}", fmt.close(), fmt.user(SUMMARY_ASK), fmt.assistant(model, false));
    tokens.extend(tokenizer.encode(ask, true).map_err(anyhow::Error::msg)?.get_ids().to_vec());
    let mut sampler = LogitsProcessor::from_sampling(42, Sampling::ArgMax);
    let logits = read_prompt(model, device, session.pos, &tokens, tell)?;
    let mut next = sampler.sample(&logits)?;
    let base = session.pos + tokens.len();
    let mut out = Vec::new();
    while !stops.contains(&next) && out.len() < SUMMARY_TOKENS {
        out.push(next);
        if let Some(t) = tell {
            if out.len() % 8 == 0 {
                (t.borrow_mut())(Step::Summary(out.len()));
            }
        }
        if foresight() > 0 {
            model.foresee(next, base + out.len() - 1, foresight())?;
        }
        let logits = model.forward(&Tensor::new(&[next], device)?.unsqueeze(0)?, base + out.len() - 1)?.squeeze(0)?;
        next = sampler.sample(&logits)?;
    }
    let text = tokenizer.decode(&out, true).map_err(anyhow::Error::msg)?;
    let text = match text.find("</think>") {
        Some(i) if text.starts_with("<think>") => text[i + "</think>".len()..].to_string(),
        _ => text,
    };
    let text = text.trim().to_string();
    println!("  made room: {} tokens of chat summarised into {} tokens", session.pos, out.len());
    Ok(text)
}

/// The text of an OpenAI message: a plain string, or a list of parts of which the text ones count.
pub fn message_text(content: &serde_json::Value) -> String {
    match content {
        serde_json::Value::String(t) => t.clone(),
        serde_json::Value::Array(parts) => parts.iter().filter_map(|p| p["text"].as_str()).collect::<Vec<_>>().join("\n"),
        _ => String::new(),
    }
}

/// An OpenAI conversation in harmony (gpt-oss), ready for the assistant's turn. The client's own
/// system message becomes harmony's developer message; the model thinks before its final answer.
fn harmony_prompt(messages: &[serde_json::Value]) -> String {
    let mut text = HARMONY_SYSTEM.to_string();
    for m in messages {
        let content = message_text(&m["content"]);
        text.push_str(&match m["role"].as_str().unwrap_or("user") {
            "system" | "developer" => format!("<|start|>developer<|message|># Instructions\n\n{content}<|end|>"),
            "assistant" => format!("<|start|>assistant<|channel|>final<|message|>{content}<|end|>"),
            _ => format!("<|start|>user<|message|>{content}<|end|>"),
        });
    }
    text.push_str("<|start|>assistant");
    text
}

/// An OpenAI conversation in the model's chat format, ready for the assistant's turn; `open` is
/// what the model's template puts at the start of that turn (see `Model::assistant_open`).
pub fn chat_prompt(messages: &[serde_json::Value], open: &str) -> String {
    let mut text = String::new();
    for m in messages {
        let role = match m["role"].as_str().unwrap_or("user") {
            r @ ("system" | "user" | "assistant") => r,
            "developer" => "system",
            _ => "user",
        };
        text.push_str(&format!("<|im_start|>{role}\n{}<|im_end|>\n", message_text(&m["content"])));
    }
    text.push_str("<|im_start|>assistant\n");
    text.push_str(open);
    text
}

/// `/v1/chat/completions`: the OpenAI chat API, streamed (server-sent events) or in one reply. Any
/// API key is accepted and ignored: only programs on this machine can reach this port.
#[allow(clippy::too_many_arguments)]
fn openai(mut s: TcpStream, body: &[u8], model: &mut Model, model_name: &str, tokenizer: &Tokenizer, device: &Device, max_new: usize, sampling: &Sampling, session: &mut Session) -> Result<()> {
    let reply = |s: &mut TcpStream, status: &str, json: serde_json::Value| -> Result<()> {
        let body = json.to_string();
        write!(s, "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len())?;
        Ok(())
    };
    let bad = |m: &str| serde_json::json!({"error": {"message": m, "type": "invalid_request_error"}});
    let Ok(q) = serde_json::from_slice::<serde_json::Value>(body) else {
        return reply(&mut s, "400 Bad Request", bad("the body is not JSON"));
    };
    let Some(messages) = q["messages"].as_array().filter(|m| !m.is_empty()) else {
        return reply(&mut s, "400 Bad Request", bad("`messages` must be a non-empty list"));
    };
    let stream = q["stream"].as_bool().unwrap_or(false);
    let asked = q["max_completion_tokens"].as_u64().or(q["max_tokens"].as_u64());
    let text = if model.harmony() { harmony_prompt(messages) } else { chat_prompt(messages, model.assistant_open(false)) };
    let prompt = tokenizer.encode(text, true).map_err(anyhow::Error::msg)?.get_ids().to_vec();
    let max_new = answer_budget(asked, max_new, prompt.len());
    // The requests own temperature and top_p win over the doors defaults; 0 means the likeliest word.
    let asked_sampling: Sampling = match (q["temperature"].as_f64(), q["top_p"].as_f64()) {
        (Some(t), _) if t <= 0.0 => Sampling::ArgMax,
        (Some(t), Some(p)) => Sampling::TopP { p, temperature: t },
        (Some(t), None) => match sampling { Sampling::TopP { p, .. } => Sampling::TopP { p: *p, temperature: t }, _ => Sampling::All { temperature: t } },
        (None, _) => sampling.clone(),
    };
    let sampling = &asked_sampling;
    let id = format!("chatcmpl-{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos()));
    let created = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let prompt_len = prompt.len();
    println!("  api: {} messages, {} tokens, {}", messages.len(), prompt_len, if stream { "streamed" } else { "one reply" });

    let event = |delta: serde_json::Value, finish: Option<&str>| {
        serde_json::json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model_name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
    };
    let mut text = String::new();
    let result = if stream {
        write!(s, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nCache-Control: no-cache\r\n\r\n")?;
        chunk(&mut s, format!("data: {}\n\n", event(serde_json::json!({"role": "assistant", "content": ""}), None)).as_bytes())?;
        let mut out_s = s.try_clone()?;
        let mut emit = |t: &str| chunk(&mut out_s, format!("data: {}\n\n", event(serde_json::json!({"content": t}), None)).as_bytes());
        generate_full(model, tokenizer, device, session, prompt, max_new, sampling, &mut emit)
    } else {
        let mut emit = |t: &str| -> Result<()> {
            text.push_str(t);
            Ok(())
        };
        generate_full(model, tokenizer, device, session, prompt, max_new, sampling, &mut emit)
    };
    let a = match result {
        Ok(a) => a,
        Err(e) => {
            *session = Session::default();
            let oom = gpu_out_of_memory(&e);
            let msg = if oom {
                "The GPU ran out of memory. Coachwhip has stopped: close some apps, or start it again with a smaller --bank.".to_string()
            } else {
                format!("{e}")
            };
            let too_long = format!("{e}").contains("start a new one");
            let err = serde_json::json!({"error": {"message": msg, "type": if too_long { "invalid_request_error" } else { "server_error" }}});
            if stream {
                let _ = chunk(&mut s, format!("data: {err}\n\ndata: [DONE]\n\n").as_bytes());
                let _ = s.write_all(b"0\r\n\r\n");
            } else {
                let status = if too_long { "400 Bad Request" } else { "500 Internal Server Error" };
                let _ = reply(&mut s, status, err);
            }
            let _ = s.flush();
            if oom {
                eprintln!("\n  coachwhip: {msg}\n");
                std::process::exit(1);
            }
            return Err(e);
        }
    };
    println!("  api: read {} new tokens at {:.1} tok/s, wrote {} at {:.1} tok/s, chat now {} tokens", a.prompt_tokens, a.read_tps, a.written, a.write_tps, a.context);
    model.report();
    if model.harmony() {
        if let Some(i) = text.find("</think>") {
            text = text[i + "</think>".len()..].trim_start().to_string();
        }
    }
    let finish = if a.finished { "stop" } else { "length" };
    let usage = serde_json::json!({"prompt_tokens": prompt_len, "completion_tokens": a.written, "total_tokens": prompt_len + a.written});
    if stream {
        let mut last = event(serde_json::json!({}), Some(finish));
        last["usage"] = usage;
        chunk(&mut s, format!("data: {last}\n\ndata: [DONE]\n\n").as_bytes())?;
        s.write_all(b"0\r\n\r\n")?;
        s.flush()?;
        Ok(())
    } else {
        reply(&mut s, "200 OK", serde_json::json!({"id": id, "object": "chat.completion", "created": created, "model": model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}], "usage": usage}))
    }
}

/// Only this machine's own page, or a program on this machine, may talk to the model: the Host
/// must name this port on loopback, and an Origin, when a browser sends one, must too. Another
/// site open in the browser could otherwise post prompts here. Headers arrive lower-cased.
pub fn own_page(host: &str, origin: &str, port: u16) -> bool {
    let local = |h: &str| h == format!("127.0.0.1:{port}") || h == format!("localhost:{port}");
    local(host) && (origin.is_empty() || origin.strip_prefix("http://").is_some_and(local))
}

/// How many tokens an API answer may write: what the client asked for, capped by Coachwhip's own
/// limit; with no limit asked for, whatever room the 16K conversation leaves.
pub fn answer_budget(asked: Option<u64>, max_new: usize, prompt_len: usize) -> usize {
    let room = context_limit().saturating_sub(prompt_len);
    asked.map_or(max_new.min(room), |m| (m as usize).min(max_new))
}

pub fn html_escape(t: &str) -> String {
    t.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

pub fn chunk(s: &mut TcpStream, bytes: &[u8]) -> Result<()> {
    if bytes.is_empty() {
        return Ok(());
    }
    write!(s, "{:x}\r\n", bytes.len())?;
    s.write_all(bytes)?;
    s.write_all(b"\r\n")?;
    s.flush()?;
    Ok(())
}

fn handle(mut s: TcpStream, model: &mut Model, model_name: &str, tokenizer: &Tokenizer, device: &Device, max_new: usize, sampling: &Sampling, session: &mut Session) -> Result<()> {
    // A client that stops sending must not hold the one-at-a-time server forever.
    s.set_read_timeout(Some(std::time::Duration::from_secs(30)))?;
    let mut r = BufReader::new(s.try_clone()?).take(64 * 1024 + MAX_BODY as u64);
    let mut line = String::new();
    r.read_line(&mut line)?;
    let path = line.split_whitespace().nth(1).unwrap_or("/").to_string();
    let mut len = 0usize;
    let mut host = String::new();
    let mut origin = String::new();
    loop {
        let mut h = String::new();
        if r.read_line(&mut h)? == 0 || h.trim().is_empty() {
            break;
        }
        let lower = h.to_lowercase();
        if let Some(v) = lower.strip_prefix("content-length:") {
            len = v.trim().parse().unwrap_or(0);
        } else if let Some(v) = lower.strip_prefix("host:") {
            host = v.trim().to_string();
        } else if let Some(v) = lower.strip_prefix("origin:") {
            origin = v.trim().to_string();
        }
    }
    let own_page = own_page(&host, &origin, s.local_addr()?.port());
    let path = path.split('?').next().unwrap_or("/").to_string();
    let api = path.starts_with("/v1/");
    if (path == "/ask" || api) && (!own_page || len > MAX_BODY) {
        write!(s, "HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")?;
        return Ok(());
    }
    if path == "/v1/models" {
        let body = serde_json::json!({"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "coachwhip"}]}).to_string();
        write!(s, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len())?;
        return Ok(());
    }
    if path == "/v1/chat/completions" {
        let mut body = vec![0u8; len];
        r.read_exact(&mut body)?;
        return openai(s, &body, model, model_name, tokenizer, device, max_new, sampling, session);
    }
    if api {
        write!(s, "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")?;
        return Ok(());
    }
    if path == "/ask" {
        let mut body = vec![0u8; len];
        r.read_exact(&mut body)?;
        let Ok(q) = serde_json::from_slice::<serde_json::Value>(&body) else {
            write!(s, "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n")?;
            return Ok(());
        };
        let prompt = q["prompt"].as_str().unwrap_or("");
        let think = q["think"].as_bool().unwrap_or(false);
        let new_chat = q["new_chat"].as_bool().unwrap_or(false);
        let max_new = q["max_new"].as_u64().map(|m| (m as usize).min(max_new)).unwrap_or(max_new);
        println!("  prompt: {} chars, thinking {}", prompt.len(), if think { "on" } else { "off" });
        write!(
            s,
            "HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\nTransfer-Encoding: chunked\r\nCache-Control: no-cache\r\nX-Accel-Buffering: no\r\n\r\n"
        )?;
        s.flush()?;
        let mut out_s = s.try_clone()?;
        let mut emit = |t: &str| chunk(&mut out_s, t.as_bytes());
        let mut prog_s = s.try_clone()?;
        let progress: Progress = Box::new(move |step| {
            let line = match step {
                Step::Layer(layer, of) => format!("{PROG_MARK}{{\"layer\":{layer},\"of\":{of}}}\n"),
                Step::Summary(words) => format!("{PROG_MARK}{{\"summary\":{words}}}\n"),
            };
            let _ = chunk(&mut prog_s, line.as_bytes());
        });
        let a = match generate(model, tokenizer, device, session, new_chat, prompt, think, max_new, sampling, Some(progress), &mut emit) {
            Ok(a) => a,
            Err(e) => {
                // A broken answer leaves the model part-way through it, so the next question starts afresh.
                *session = Session::default();
                let oom = gpu_out_of_memory(&e);
                let msg = if oom {
                    "The GPU ran out of memory. Coachwhip has stopped: close some apps, or start it again with a smaller --bank.".to_string()
                } else {
                    format!("The answer could not be finished: {e}")
                };
                let _ = chunk(&mut s, format!("{ERROR_MARK}{msg}").as_bytes());
                let _ = s.write_all(b"0\r\n\r\n");
                let _ = s.flush();
                if oom {
                    eprintln!("\n  coachwhip: {msg}\n");
                    std::process::exit(1);
                }
                return Err(e);
            }
        };
        println!("  read {} new tokens at {:.1} tok/s, wrote {} at {:.1} tok/s, chat now {} tokens", a.prompt_tokens, a.read_tps, a.written, a.write_tps, a.context);
        model.report();
        let stats = serde_json::json!({
            "prompt_tokens": a.prompt_tokens, "written": a.written, "context": a.context, "fresh": a.fresh, "compacted": a.compacted,
            "read_tps": a.read_tps, "write_tps": a.write_tps,
        });
        chunk(&mut s, format!("{STATS_MARK}{stats}").as_bytes())?;
        s.write_all(b"0\r\n\r\n")?;
    } else {
        // gpt-oss is a reasoning model: its thinking switch starts ticked.
        let page = if model.harmony() {
            PAGE.replace(r#"id="think">"#, r#"id="think" checked>"#)
        } else if model.thinks() {
            PAGE.to_string()
        } else {
            PAGE.replace(THINK_BOX, "")
        };
        let page = page.replace("MODEL_NAME", &html_escape(model_name));
        write!(s, "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\n\r\n", page.len())?;
        s.write_all(page.as_bytes())?;
    }
    s.flush()?;
    Ok(())
}

/// The page's thinking switch, left out for a model that has no thinking mode.
pub const THINK_BOX: &str = r#"<label><input type="checkbox" id="think"> let it think first (slower)</label>"#;

pub const PAGE: &str = r##"<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Coachwhip</title><style>
 :root{--bg:#0d1117;--panel:#161b22;--line:#30363d;--text:#e6edf3;--dim:#8b949e;--on:#3fb950;--off:#6e7681}
 *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);
   font:15px/1.55 ui-monospace,"Cascadia Code",Consolas,monospace}
 .wrap{max-width:900px;margin:0 auto;padding:28px 16px 60px}
 h1{font-size:22px;margin:0 0 4px} h1 b{color:var(--on)}
 .sub{color:var(--dim);font-size:13px;margin-bottom:22px}
 textarea{width:100%;height:160px;background:var(--panel);color:var(--text);border:1px solid var(--line);
   border-radius:8px;padding:14px;font:14px/1.5 ui-monospace,Consolas,monospace;resize:vertical}
 .row{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin:12px 0 22px}
 button{background:var(--on);color:#08130b;border:0;border-radius:7px;padding:10px 22px;font-weight:700;
   font-size:14px;cursor:pointer} button:disabled{background:var(--off);color:#0d1117;cursor:default}
 label{color:var(--dim);font-size:13px;display:flex;gap:7px;align-items:center;cursor:pointer}
 .card{background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden}
 .card h2{margin:0;padding:11px 15px;font-size:12px;letter-spacing:.09em;text-transform:uppercase;
   border-bottom:1px solid var(--line);color:var(--on)}
 pre{margin:0;padding:15px;white-space:pre-wrap;word-break:break-word;min-height:40px;font-size:14px}
 .you{color:var(--dim);white-space:pre-wrap;word-break:break-word;margin:18px 0 8px;font-size:14px}
 .you b{color:var(--on)} #log .card{margin-bottom:6px} button.ghost{background:transparent;color:var(--dim);border:1px solid var(--line)}
 .ms{color:var(--dim);font-size:13px;padding:0 15px 12px} .ms b{color:var(--text)}
</style></head><body><div class="wrap">
 <h1>coachwhip <b>·</b> MODEL_NAME</h1>
 <div class="sub">the experts stream from the SSD; everything runs on this machine</div>
 <div id="log"></div>
 <textarea id="p" spellcheck="false">Write a JavaScript function debounce(fn, ms) and explain in two sentences how it works.</textarea>
 <div class="row">
   <button id="go">Send</button>
   <button id="new" class="ghost">New chat</button>
   <label><input type="checkbox" id="think"> let it think first (slower)</label>
 </div>
</div><script>
const $=i=>document.getElementById(i);
const MARK = '\u0001STATS:';
const ERR = '\u0001ERROR:';
const PROG = /\u0001PROG:(\{[^\n]*\})\n/g;
let newChat = true;
const add = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; e.innerHTML = html; $('log').appendChild(e); return e; };
$('new').onclick = () => { newChat = true; $('log').innerHTML = ''; $('p').focus(); };
$('go').onclick = async () => {
  const prompt = $('p').value.trim();
  if (!prompt) return;
  $('go').disabled = true; $('new').disabled = true;
  add('div', 'you', '<b>you ›</b> ').appendChild(document.createTextNode(prompt));
  const card = add('div', 'card', '<pre></pre><div class="ms"></div>');
  const A = card.querySelector('pre'), T = card.querySelector('.ms');
  T.textContent = 'reading…'; $('p').value = '';
  const t0 = performance.now();
  let text = '', first = 0;
  try{
    const r = await fetch('/ask',{method:'POST',headers:{'content-type':'application/json'},
      body:JSON.stringify({prompt, new_chat:newChat, think:!!($('think')&&$('think').checked)})});
    newChat = false;
    const reader = r.body.getReader(), dec = new TextDecoder();
    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      text += dec.decode(value, {stream:true});
      let p = null;
      text = text.replace(PROG, (m, j) => { p = JSON.parse(j); return ''; });
      const x = text.indexOf(ERR);
      if (x >= 0) { A.textContent = text.slice(0, x); T.textContent = text.slice(x + ERR.length); newChat = true; break; }
      if (p && !first) {
        const secs = ((performance.now()-t0)/1000).toFixed(1) + ' s';
        T.textContent = p.summary !== undefined
          ? 'making room: summarising the chat so far · ' + p.summary + ' tokens · ' + secs
          : 'reading your prompt · layer ' + p.layer + ' of ' + p.of + ' · ' + secs;
        continue;
      }
      if (!text) continue;
      if (!first) { first = performance.now(); T.textContent = 'writing…'; }
      const k = text.indexOf(MARK);
      A.textContent = k < 0 ? text : text.slice(0, k);
    }
    const k = text.indexOf(MARK);
    if (k >= 0 && text.indexOf(ERR) < 0) {
      const j = JSON.parse(text.slice(k + MARK.length));
      T.innerHTML = 'writing <b>' + j.write_tps.toFixed(1) + ' tok/s</b> (' + j.written + ' tokens) · read <b>' +
        j.prompt_tokens + ' new tokens</b> at ' + j.read_tps.toFixed(1) + ' tok/s · first word after <b>' +
        ((first - t0)/1000).toFixed(1) + ' s</b> · chat so far ' + j.context + ' tokens' +
        (j.compacted ? ' · the chat was summarised to make room' : '');
    }
  }catch(e){ A.textContent = 'failed: ' + e; newChat = true; }
  $('go').disabled = false; $('new').disabled = false;
  card.scrollIntoView({block:'end'});
};
</script></body></html>"##;
