//! Coachwhip: run mixture-of-experts models that are bigger than the memory, on a Mac. An 80B
//! model runs on 16 GB: the experts stream from the SSD into a small bank on the GPU, and the
//! ones the next layer will probably need are fetched while the current layer computes.

mod chat;
mod experts;
mod model;
mod model_next;
mod mv_id;
#[cfg(test)]
mod tests;

use anyhow::Result;
use candle::Device;
use candle_transformers::generation::Sampling;
use clap::Parser;
use std::io::Write;
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(version, about = "Run mixture-of-experts models bigger than your memory on a Mac: an 80B model on 16 GB")]
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

    /// Experts fetched ahead for the next layer; 0 turns prefetch off.
    #[arg(long, default_value_t = 2)]
    prefetch: usize,

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
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = Device::new_metal(0)?;
    // A hot path belongs to one model: its routes live in a folder named after the model file.
    let model_name = args.model.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "model".into());
    let routes = args.routes.join(&model_name);
    let learned = args.learned.join(&model_name);
    std::fs::create_dir_all(&learned)?;
    let settings = experts::Settings {
        bank: args.bank,
        prefetch: args.prefetch,
        ahead: 1,
        readers: args.readers,
        routes: routes.is_dir().then_some(routes),
        learned: Some(learned.clone()),
        profile: args.profile,
    };
    eprintln!("coachwhip: hot path for {model_name} learns in {}", learned.display());

    let started = std::time::Instant::now();
    let mut model = model::Model::load(&args.model, &device, &settings)?;
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
    let mut out = std::io::stdout();
    let mut emit = |t: &str| -> Result<()> {
        out.write_all(t.as_bytes())?;
        out.flush()?;
        Ok(())
    };
    let a = chat::generate(&mut model, &tokenizer, &device, &mut chat::Session::default(), true, &prompt, args.think, args.max_tokens, &sampling, None, &mut emit)?;
    eprintln!(
        "\n\ncoachwhip: read {} tokens at {:.1} tok/s, wrote {} at {:.1} tok/s",
        a.prompt_tokens, a.read_tps, a.written, a.write_tps
    );
    model.report();
    Ok(())
}
