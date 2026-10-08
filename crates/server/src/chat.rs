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
const PROMPT_CHUNK: usize = 512;

/// When a chat would grow past this many words, it is summarised first to make room.
pub const CONTEXT_LIMIT: usize = 16384;

/// Longest summary written when making room.
const SUMMARY_TOKENS: usize = 500;

const SUMMARY_ASK: &str = "Summarise our conversation so far in at most 300 words, for someone who will continue it: what was asked, what was decided, and every name, file, function or fact the next answer will need. Plain text, no preamble.";

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
    // The template's way of switching thinking on or off: what opens the assistant's turn.
    let open = model.assistant_open(think);
    let encode = |text: String| -> Result<Vec<u32>> { Ok(tokenizer.encode(text, true).map_err(anyhow::Error::msg)?.get_ids().to_vec()) };
    let eos = *tokenizer.get_vocab(true).get("<|im_end|>").unwrap();
    let tell = progress.map(|p| Rc::new(RefCell::new(p)));

    // A follow-up closes the last answer and adds the new question; nothing earlier is read again.
    let turn = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{open}");
    let mut tokens = Vec::new();
    let mut fresh = new_chat || session.pos == 0 || session.by_api;
    let mut summary = None;
    if !fresh {
        tokens.extend(session.pending);
        tokens.extend(encode(format!("<|im_end|>\n{turn}"))?);
        // Too long to go on: the model summarises the chat, and the summary opens a new one.
        if session.pos + tokens.len() + max_new > CONTEXT_LIMIT {
            summary = Some(summarise(model, tokenizer, device, session, eos, &tell)?);
            fresh = true;
        }
    }
    if fresh {
        model.clear_kv_cache();
        *session = Session::default();
        tokens = encode(match &summary {
            Some(s) => format!("<|im_start|>user\nSummary of our conversation so far, for context:\n{s}\n\nNow: {prompt}<|im_end|>\n<|im_start|>assistant\n{open}"),
            None => turn,
        })?;
    }

    let a = answer(model, tokenizer, device, session, tokens, fresh, summary.is_some(), think, max_new, sampling, &tell, emit);
    session.by_api = false;
    a
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
    if prompt.len() + max_new > CONTEXT_LIMIT {
        anyhow::bail!("the conversation and its answer would be over {CONTEXT_LIMIT} tokens; start a new one");
    }
    let known = session.fed.len();
    let fresh = known == 0 || prompt.len() <= known || !prompt.starts_with(&session.fed);
    if fresh {
        model.clear_kv_cache();
        *session = Session::default();
    } else {
        // The word an earlier answer was cut off on is part of what the client sent back.
        session.pending = None;
    }
    let tokens = prompt[session.fed.len()..].to_vec();
    let a = answer(model, tokenizer, device, session, tokens, fresh, false, false, max_new, sampling, &None, emit);
    session.by_api = true;
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
    let eos = *tokenizer.get_vocab(true).get("<|im_end|>").unwrap();
    let mut sampler = LogitsProcessor::from_sampling(42, sampling.clone());
    let t0 = std::time::Instant::now();
    let logits = read_prompt(model, device, session.pos, &tokens, tell)?;
    session.fed.extend(&tokens);
    let base = session.pos + tokens.len();
    let mut next = sampler.sample(&logits)?;
    let read = t0.elapsed().as_secs_f64();

    // The answer so far; an end-of-turn word is never part of it.
    let mut out = if next == eos { Vec::new() } else { vec![next] };
    let mut sent = 0usize;
    let mut stream_text = |out: &[u32], sent: &mut usize| -> Result<()> {
        let full = tokenizer.decode(out, true).map_err(anyhow::Error::msg)?;
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
    while out.len() < max_new && next != eos {
        if speculate > 0 && greedy && max_new - out.len() > speculate {
            if let Some(seq) = model.speculate(next, base + out.len() - 1, speculate)? {
                for &w in &seq {
                    next = w;
                    if w == eos {
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
                    if w == eos {
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
        let logits = model.forward(&Tensor::new(&[next], device)?.unsqueeze(0)?, base + out.len() - 1)?.squeeze(0)?;
        next = sampler.sample(&logits)?;
        if next != eos {
            out.push(next);
            stream_text(&out, &mut sent)?;
        }
    }
    let write = t1.elapsed().as_secs_f64();
    let written = out.len();
    // Every word of the answer was read back, except the last one when the length limit cut it off.
    let finished = next == eos;
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
    let pieces = tokens.len().div_ceil(PROMPT_CHUNK);
    let read = (|| -> Result<Tensor> {
    let mut logits = None;
    for (i, piece) in tokens.chunks(PROMPT_CHUNK).enumerate() {
        model.set_progress(tell.clone().map(|t| -> Progress {
            Box::new(move |step| {
                if let Step::Layer(layer, of) = step {
                    (t.borrow_mut())(Step::Layer(i * of + layer, pieces * of))
                }
            })
        }));
        logits = Some(model.forward(&Tensor::new(piece, device)?.unsqueeze(0)?, pos + i * PROMPT_CHUNK)?.squeeze(0)?);
    }
    logits.ok_or_else(|| anyhow::anyhow!("empty prompt"))
    })();
    // Taken back on every path: the hook holds the page's connection open.
    model.set_progress(None);
    read
}

/// Asks the model, in the chat as it stands, for a summary of the chat; the chat is then thrown
/// away and the summary opens the next one. Greedy, so it is the same every time.
fn summarise(model: &mut Model, tokenizer: &Tokenizer, device: &Device, session: &Session, eos: u32, tell: &Option<Rc<RefCell<Progress>>>) -> Result<String> {
    let mut tokens: Vec<u32> = session.pending.into_iter().collect();
    let ask = format!("<|im_end|>\n<|im_start|>user\n{SUMMARY_ASK}<|im_end|>\n<|im_start|>assistant\n{}", model.assistant_open(false));
    tokens.extend(tokenizer.encode(ask, true).map_err(anyhow::Error::msg)?.get_ids().to_vec());
    let mut sampler = LogitsProcessor::from_sampling(42, Sampling::ArgMax);
    let logits = read_prompt(model, device, session.pos, &tokens, tell)?;
    let mut next = sampler.sample(&logits)?;
    let base = session.pos + tokens.len();
    let mut out = Vec::new();
    while next != eos && out.len() < SUMMARY_TOKENS {
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
    let prompt = tokenizer.encode(chat_prompt(messages, model.assistant_open(false)), true).map_err(anyhow::Error::msg)?.get_ids().to_vec();
    let max_new = answer_budget(asked, max_new, prompt.len());
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
    let room = CONTEXT_LIMIT.saturating_sub(prompt_len);
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
        let page = if model.thinks() { PAGE.to_string() } else { PAGE.replace(THINK_BOX, "") };
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
