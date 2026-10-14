//! Coachwhip: run mixture-of-experts models that are bigger than the memory, on a Mac. A 122B
//! model runs on 16 GB: the experts stream from the SSD into a small bank on the GPU, and the
//! ones the next layer will probably need are fetched while the current layer computes.

use coachwhip_engine::{experts, model};
use coachwhip_server::chat;

use anyhow::Result;
use candle::Device;
use candle_transformers::generation::Sampling;
use clap::Parser;
use std::io::Write;
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(version, about = "Run mixture-of-experts models bigger than your memory on a Mac: a 122B model on 16 GB")]
struct Args {
    /// The model, as a GGUF file: Qwen3 MoE or Qwen3-Next (Q4_K_M tested).
    #[arg(long)]
    model: PathBuf,

    /// The model's tokenizer.json.
    #[arg(long)]
    tokenizer: PathBuf,

    /// Serve the chat page on 127.0.0.1 at this port.
    #[arg(long, conflicts_with = "prompt")]
    chat: Option<u16>,

    /// Lab: read the prompt as raw text, no chat template, and decode greedily.
    #[arg(long, hide = true)]
    raw: bool,

    /// Answer one prompt and exit.
    #[arg(long)]
    prompt: Option<String>,

    /// Let the model think before it answers (slower).
    #[arg(long)]
    think: bool,

    /// Longest answer, in tokens.
    #[arg(long, default_value_t = 4000)]
    max_tokens: usize,

    /// Sampling temperature; 0 always picks the likeliest token.
    #[arg(long, default_value_t = 0.7)]
    temperature: f64,

    /// Nucleus sampling: keep the likeliest tokens up to this probability mass.
    #[arg(long, default_value_t = 0.8)]
    top_p: f64,

    /// Expert slots kept on the GPU per layer.
    #[arg(long, default_value_t = 44)]
    bank: usize,

    /// Experts guessed and fetched ahead per layer; 0 turns prefetch off. Default 10, or 6 for
    /// the 122B, the settings each model's numbers were measured with.
    #[arg(long)]
    prefetch: Option<usize>,

    /// How many layers ahead the guess looks (each layer is about 2 ms of GPU work to read in).
    /// Default 4, or 2 for the 122B.
    #[arg(long)]
    ahead: Option<usize>,

    /// Reader threads for the prefetch.
    #[arg(long, default_value_t = 3)]
    readers: usize,

    /// Folder of starting route logs, one sub-folder per model, named after the model file.
    #[arg(long, default_value = "data/routes")]
    routes: PathBuf,

    /// Folder where the routes of every answer are saved, one sub-folder per model, named after
    /// the model file, so each model's hot path keeps learning on its own.
    #[arg(long, default_value = "learned")]
    learned: PathBuf,

    /// Print where each word's time went after every answer.
    #[arg(long)]
    profile: bool,

    /// Ask the router for this many experts per word instead of the model's own number (Qwen3.5-122B:
    /// 4 instead of 8). Fewer experts means fewer bytes per word and faster writing; the answers
    /// change, so check them for your use.
    #[arg(long)]
    experts: Option<usize>,

    /// The parallel path: the GPU waits on a shared event for its experts instead of the CPU
    /// re-queuing each layer. Faster on big models; the answers do not change.
    #[arg(long)]
    parallel: bool,

    /// The register: learns, as it runs, what each expert adds to the running state, and guesses
    /// the next layers' experts from the state it predicts. Fewer reads; the answers do not change.
    #[arg(long)]
    register: bool,

    /// Stand-ins (gpt-oss): a light pick that is not in the bank is replaced by the best expert
    /// that is, by the router's own scores. The first pick is always read for real, and the second
    /// whenever the router gives it at least 0.3 of the weight, and a substitute is used only when its
    /// router score is close to the real pick's. Fewer reads; the answers change, so check them for
    /// your use.
    #[arg(long)]
    stand_ins: bool,

    /// Flash reads: experts are read straight from the SSD, past the page cache, which has nothing
    /// to give back on a 16 GB Mac. The answers do not change.
    #[arg(long)]
    flash_reads: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    // The engine reads these as switches; the flags are their front door.
    if let Some(k) = args.experts {
        std::env::set_var("COACHWHIP_TOPK", k.to_string());
    }
    if args.parallel {
        std::env::set_var("COACHWHIP_PARALLEL", "1");
    }
    // The parallel path commits GPU work itself, layer by layer, once the CPU has written what
    // that work reads; Candle must not commit on its own count in between.
    if coachwhip_engine::experts::parallel() && std::env::var("CANDLE_METAL_COMPUTE_PER_BUFFER").is_err() {
        std::env::set_var("CANDLE_METAL_COMPUTE_PER_BUFFER", "1000000");
    }
    if args.register {
        std::env::set_var("COACHWHIP_REGISTER", "1");
    }
    if args.stand_ins {
        std::env::set_var("COACHWHIP_STANDIN", "1");
        if std::env::var("COACHWHIP_STANDIN_W").is_err() {
            std::env::set_var("COACHWHIP_STANDIN_W", "0.3");
        }
        // A substitute more than this far below its real pick (router logits) is not used; the
        // real expert is read. Three exam runs at 1.0 scored 5.0, 3.8 and 4.8; without it 4.0, 4.0, 0.8.
        if std::env::var("COACHWHIP_STANDIN_GAP").is_err() {
            std::env::set_var("COACHWHIP_STANDIN_GAP", "1.0");
        }
    }
    if args.flash_reads {
        std::env::set_var("COACHWHIP_NOCACHE", "1");
    }
    let device = Device::new_metal(0)?;
    // A hot path belongs to one model: its routes live in a folder named after the model file.
    let model_name = args.model.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "model".into());
    let routes = args.routes.join(&model_name);
    let learned = args.learned.join(&model_name);
    std::fs::create_dir_all(&learned)?;
    // The 122B (48-layer qwen35moe) guesses best at 6 ahead by 2, gpt-oss at 2 by 2 (its 13 MB
    // experts make a wrong guess dear); the others at 10 by 4.
    let (big, oss) = {
        let mut f = std::fs::File::open(&args.model)?;
        let ct = candle::quantized::gguf_file::Content::read(&mut f)?;
        let arch = ct.metadata.get("general.architecture").and_then(|v| v.to_string().ok().cloned()).unwrap_or_default();
        let layers = ct.metadata.get(&format!("{arch}.block_count")).and_then(|v| v.to_u32().ok()).unwrap_or(0);
        (arch == "qwen35moe" && layers >= 48, arch == "gpt-oss")
    };
    // gpt-oss guesses about half the experts it uses per word (13 MB each makes a wrong guess dear).
    // With stand-ins most picks are already in the bank; one guess a layer keeps the guessed reads
    // out of the way of the ones a word needs (1,000 words: 8.4 tok/s at 2 guesses, 10.9 at 1).
    let oss_guess = if args.stand_ins { 1 } else { args.experts.map_or(2, |k| (k / 2).max(1)) };
    let (prefetch, ahead) = match (args.prefetch, args.ahead) {
        (Some(p), Some(a)) => (p, a),
        (Some(p), None) => (p, if big || oss { 2 } else { 4 }),
        (None, Some(a)) => (if oss { oss_guess } else if big { 6 } else { 10 }, a),
        (None, None) => if oss { (oss_guess, 2) } else if big { (6, 2) } else { (10, 4) },
    };
    let settings = experts::Settings {
        bank: args.bank,
        prefetch,
        ahead: ahead.max(1),
        readers: args.readers,
        routes: routes.is_dir().then_some(routes),
        learned: Some(learned.clone()),
        profile: args.profile,
    };
    eprintln!("coachwhip: hot path for {model_name} learns in {}", learned.display());

    let started = std::time::Instant::now();
    let mut model = model::Model::load(&args.model, &device, &settings)?;
    if let Some((used, max)) = coachwhip_engine::gpu_sync::gpu_memory(&device) {
        eprintln!("coachwhip: memlog after load: {:.2} GB of {:.2} GB", used as f64 / 1e9, max as f64 / 1e9);
    }
    eprintln!("coachwhip: model ready in {:.1} s", started.elapsed().as_secs_f64());
    let tokenizer = Tokenizer::from_file(&args.tokenizer).map_err(anyhow::Error::msg)?;
    let sampling = if args.temperature <= 0.0 {
        Sampling::ArgMax
    } else {
        Sampling::TopP { p: args.top_p, temperature: args.temperature }
    };

    if let Some(port) = args.chat {
        return chat::serve(port, &mut model, &model_name, tokenizer, &device, args.max_tokens, sampling);
    }
    let Some(prompt) = args.prompt else {
        anyhow::bail!("give --chat <port> or --prompt <text>")
    };
    if args.raw {
        let ids = tokenizer.encode(prompt.as_str(), false).map_err(anyhow::Error::msg)?.get_ids().to_vec();
        eprintln!("coachwhip: raw prompt, {} tokens", ids.len());
        model.clear_kv_cache();
        // Read the prompt in 512-token pieces, as the chat does; the last piece goes through the loop.
        let mut offset = 0usize;
        let last = ids.len() - (ids.len() - 1) % 512 - 1;
        for piece in ids[..last].chunks(512) {
            model.forward(&candle::Tensor::new(piece, &device)?.unsqueeze(0)?, offset)?;
            offset += piece.len();
        }
        let mut input = candle::Tensor::new(&ids[last..], &candle::Device::Cpu)?.unsqueeze(0)?;
        let mut outs = Vec::new();
        // Lab (`COACHWHIP_BALLAST_MB=n`): hold n MB of GPU memory, to price running near the limit.
        let _ballast = match std::env::var("COACHWHIP_BALLAST_MB").ok().and_then(|v| v.parse::<usize>().ok()) {
            Some(mb) => Some(candle::Tensor::ones(mb * 1024 * 256, candle::DType::F32, &device)?),
            None => None,
        };
        let t0 = std::time::Instant::now();
        if coachwhip_engine::fuse_kernel::gpupick_on() && device.is_metal() {
            // The GPU picks each word and feeds it straight back in; the CPU reads the words back
            // every 16, so it is never waiting for the GPU between two words.
            let mut pending: Vec<candle::Tensor> = Vec::new();
            for step in 0..args.max_tokens {
                let logits = model.forward(&input, offset)?;
                offset += input.dim(1)?;
                let next = coachwhip_engine::fuse_kernel::argmax(&logits)?;
                pending.push(next.clone());
                input = next;
                if step == 0 || pending.len() == 16 || step + 1 == args.max_tokens {
                    for t in pending.drain(..) {
                        outs.push(t.flatten_all()?.to_vec1::<u32>()?[0]);
                    }
                    if step == 0 {
                        eprintln!("coachwhip: first token after {:.1} s", t0.elapsed().as_secs_f64());
                    }
                }
            }
            println!("{:?}", tokenizer.decode(&outs, false).map_err(anyhow::Error::msg)?);
            eprintln!("coachwhip: {} tokens in {:.1} s", outs.len(), t0.elapsed().as_secs_f64());
            return Ok(());
        }
        for step in 0..args.max_tokens {
            let logits: Vec<f32> = model.forward(&input, offset)?.flatten_all()?.to_dtype(candle::DType::F32)?.to_vec1()?;
            offset += input.dim(1)?;
            // The likeliest token, first of equals (what a stable descending sort put first), in one pass.
            let mut best = 0usize;
            for (i, &v) in logits.iter().enumerate() {
                if v > logits[best] {
                    best = i;
                }
            }
            if step == 0 {
                let m = logits[best];
                let lse = m + logits.iter().map(|v| (v - m).exp()).sum::<f32>().ln();
                let mut idx: Vec<usize> = (0..logits.len()).collect();
                idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
                let top: Vec<(String, f32)> = idx[..3].iter().map(|&i| (tokenizer.decode(&[i as u32], false).unwrap_or_default(), logits[i] - lse)).collect();
                eprintln!("coachwhip: first token top-3 {top:?} after {:.1} s", t0.elapsed().as_secs_f64());
                debug_assert_eq!(idx[0], best);
            }
            if std::env::var("COACHWHIP_POOLLOG").is_ok() && step % 25 == 0 {
                if let candle::Device::Metal(m) = &device {
                    let (pooled, in_use, count) = m.pool_stats();
                    eprintln!("coachwhip: pool at word {step}: {:.2} GB pooled in {count} buffers, {:.2} GB in use", pooled as f64 / 1e9, in_use as f64 / 1e9);
                }
            }
            let next = best as u32;
            outs.push(next);
            let bk: usize = std::env::var("COACHWHIP_BATCH_TIME").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
            input = candle::Tensor::new(vec![next; bk].as_slice(), &candle::Device::Cpu)?.unsqueeze(0)?;
        }
        println!("{:?}", tokenizer.decode(&outs, false).map_err(anyhow::Error::msg)?);
        eprintln!("coachwhip: {} tokens in {:.1} s", outs.len(), t0.elapsed().as_secs_f64());
        return Ok(());
    }
    let mut out = std::io::stdout();
    let mut emit = |t: &str| -> Result<()> {
        out.write_all(t.as_bytes())?;
        out.flush()?;
        Ok(())
    };
    let a = chat::generate(&mut model, &tokenizer, &device, &mut chat::Session::default(), true, &prompt, args.think, args.max_tokens, &sampling, None, &mut emit)?;
    if let Some((used, max)) = coachwhip_engine::gpu_sync::gpu_memory(&device) {
        eprintln!("coachwhip: memlog after answer: {:.2} GB of {:.2} GB", used as f64 / 1e9, max as f64 / 1e9);
    }
    eprintln!(
        "\n\ncoachwhip: read {} tokens at {:.1} tok/s, wrote {} at {:.1} tok/s",
        a.prompt_tokens, a.read_tps, a.written, a.write_tps
    );
    model.report();
    Ok(())
}
