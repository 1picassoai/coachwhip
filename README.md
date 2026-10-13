<p align="center"><img src="docs/coachwhip.png" alt="A coachwhip snake on sand" width="720"></p>

<p align="center">
  <a href="https://github.com/1picassoai/coachwhip/releases/latest"><img src="https://img.shields.io/github/v/release/1picassoai/coachwhip" alt="Latest release"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/1picassoai/coachwhip" alt="License: Apache-2.0"></a>
  <img src="https://img.shields.io/badge/macOS-Apple%20silicon-black?logo=apple" alt="macOS on Apple silicon"></p>

# Coachwhip

**The 80-billion-parameter Qwen3-Coder-Next on a 16 GB Mac, at 11.5 to 13.4 tok/s. Fully offline. Nothing leaves your machine.**

**Requirements:** Apple silicon Mac (M1 to M4; M4 measured, earlier chips slower), 16 GB unified memory, macOS 15 or later, about 35 GB of free disk.

**Install the binary:**

```sh
curl -fsSL https://raw.githubusercontent.com/1picassoai/coachwhip/main/install.sh | bash
```

**Download the pre-squeezed model (29 GB) and the tokenizer:**

```sh
mkdir -p ~/coachwhip/models && cd ~/coachwhip/models
curl -L -O https://huggingface.co/vinnj/coachwhip-models/resolve/main/Qwen3-Coder-Next.Q2X.gguf
curl -L -O https://huggingface.co/Qwen/Qwen3-Coder-Next/resolve/main/tokenizer.json
```

sha256 of the model file: `cdc0584c1e4d30134cbb83f18e4c7607af90bd1145a64940c7a357b02a0216af` (`shasum -a 256 Qwen3-Coder-Next.Q2X.gguf`). It was squeezed once from [MaziyarPanahi's Q3_K_M](https://huggingface.co/MaziyarPanahi/Qwen3-Coder-Next-GGUF) with the `coachwhip-prepare` tool that ships in the install; the squeeze is reproducible, same bytes every time.

**Start the chat in your browser:**

```sh
~/coachwhip/coachwhip.sh ~/coachwhip/models/Qwen3-Coder-Next.Q2X.gguf ~/coachwhip/models/tokenizer.json 109 8090 --parallel --flash-reads --stand-ins
```

The chat opens at http://127.0.0.1:8090. Measured on a Mac mini M4 with 16 GB: 12.8 to 13.4 tok/s on short prompts, 11.5 tok/s with a 13,000-token prompt, three runs each, with 3 GB of memory left for macOS at its tightest. Drop `--stand-ins` for every expert read exactly, at about 5 tok/s. Numbers and the quality check: [release notes](docs/releases/0.5.1.md).

<p align="center"><img src="docs/coachwhip-demo-coder.gif" alt="The 80B Qwen3-Coder-Next writing a TypeScript function in the Coachwhip chat on a 16 GB Mac mini, in real time, at 13 tok/s" width="720"></p>

**Use it from coding tools, through the local OpenAI API:**

| Setting | Value |
|---|---|
| Base URL | `http://127.0.0.1:8090/v1` |
| API key | anything; it is ignored |
| Model | `Qwen3-Coder-Next.Q2X.gguf` |

Aider in the terminal:

```sh
aider --openai-api-base http://127.0.0.1:8090/v1 --openai-api-key anything --model openai/Qwen3-Coder-Next.Q2X.gguf --map-tokens 0 src/queue.ts
```

**How it works:**

- Keeps the part of the model every word needs on the GPU, and streams the experts each word asks for from the SSD.
- Guesses the next layers' experts with the model's own routers and reads them in while the GPU is busy on this one.
- Reads experts straight from the SSD, past the page cache (`--flash-reads`), into memory the GPU reads directly; no copies.
- Other models, gpt-oss-120b among them, and what each needs: [docs/MODELS.md](docs/MODELS.md).

License: Apache-2.0 (see [LICENSE](LICENSE)).

---

## Why we built it

Two reasons, and both had to hold at once.

**The code stays on the machine.** A coding assistant reads your code. If that assistant runs on someone else's server, your code goes there with every question. Coachwhip runs the model on your Mac, listens only to programs on your Mac, and sends nothing anywhere. Unplug the network and it works the same.

**It had to be fast enough to use.** A model big enough to be worth asking does not fit in a laptop's memory, and the usual answer, a smaller model, is a worse assistant. Coachwhip's answer is to stop loading the whole model. A mixture-of-experts model is made of many small experts, and a router picks only a few of them for each word. Coachwhip reads exactly those, keeps the ones it has used lately on the GPU, and guesses the next layer's picks while the GPU is busy on this one. The big model stays big; only its working set is in memory.

## Use

**Chat in your browser:**

```sh
~/coachwhip/coachwhip.sh /path/to/model.gguf /path/to/tokenizer.json 56
```

The last number is the bank, the expert slots kept on the GPU per layer (44 if you leave it out; each model's number is on the models page). The chat opens in your browser. Type, press Send, and the answer streams as it is written. Follow-up questions read only your new words; press **New chat** to start again. Stop Coachwhip with Ctrl+C.

**Thinking.** A model that thinks before it answers shows its thinking in a box above the answer. A model with an optional thinking mode shows a **let it think first** box on the page. Either way Coachwhip only puts the model's own template text at the start of its turn; each word comes at the same speed, and thinking makes the reply longer, not slower per word. In the terminal it is `--think`; through the API it is always off, so coding tools get the answer.

**Through the API,** a conversation holds up to 16K tokens: the history, the question and the answer together. That limit is sized to fit a 16 GB Mac; the models themselves support longer.

Only programs on this Mac can reach it. Follow-ups that resend the whole conversation read only the new part. The chat page and your tools share one model, one request at a time; when a tool has used it, the page's next question starts a new chat.

Tested with the official `openai` Python client and with [Aider](https://aider.chat). `--map-tokens 0` in the Aider line turns off its repository map, which adds up to about a thousand tokens to each request by default.

<p align="center"><img src="docs/coachwhip-aider.png" alt="Aider using an 80-billion-parameter coder through Coachwhip on a 16 GB Mac: it finds a sort-order bug and rewrites the file" width="720"></p>

**One answer in the terminal:**

```sh
cd ~/coachwhip
./target/release/coachwhip --model /path/to/model.gguf --tokenizer /path/to/tokenizer.json --prompt "Write a Rust function that reverses a linked list."
```

**Useful options:**

| Option | What it does |
|---|---|
| `--chat 8090` | Serve the chat page on that port, on this machine only |
| `--max-tokens 4000` | Longest answer, thinking included |
| `--temperature 0` | Always pick the likeliest word (same answer every time) |
| `--bank 44` | Expert slots kept on the GPU per layer |
| `--prefetch 10` | Experts guessed and read in ahead, per layer; 0 turns the guess off |
| `--ahead 4` | How many layers ahead the guess looks |
| `--think` | Let a thinking model reason before it answers (`--prompt` runs) |
| `--parallel` | The GPU waits on a shared event for its experts; faster on big models, same answers |
| `--flash-reads` | Read experts straight from the SSD, past the page cache; same answers |
| `--experts 4` | Use fewer experts per word than the model was trained with; faster, the answers change |
| `--stand-ins` | Fast mode for models that support it: stand-ins for the light picks not in the bank; the answers change |
| `--profile` | Print where the time went after each answer |
| `--help` | Everything else |

## Tell us how it ran

Good or bad, especially if it was slow, the fan worked hard, or it ran hot. [Open an issue](https://github.com/1picassoai/coachwhip/issues) with your Mac (chip and memory), macOS version, the model file, the `--bank` you used, and the speed line the chat page shows under each answer. Questions and ideas are welcome there too.

Release notes, every version: [docs/releases](docs/releases).

## Roadmap

- More MoE families: Mixtral, Qwen2-MoE, OLMoE, DeepSeek, GLM, Qwen3.8.
- Fused-attention GGUF files.
- An adapter that reads your own repository, on your own machine.
- More platforms.

## Credits

Built on [Candle](https://github.com/huggingface/candle); `vendor/candle-core` is its core crate with an MXFP4 kernel for Metal. gpt-oss-120b is [OpenAI's](https://huggingface.co/openai/gpt-oss-120b), under Apache-2.0. `crates/engine/src/model.rs` is derived from Candle's Qwen3 model, and the fused expert kernel is ggml's `mul_mv_id`, which ships inside Candle. The block-at-a-time linear attention follows [Gated Delta Networks](https://arxiv.org/abs/2412.06464) (Yang, Kautz and Hatamizadeh, ICLR 2025).

To build from source instead of taking the ready-built binary, run the install line with `COACHWHIP_FROM_SOURCE=1` in front of `bash`; it installs Apple's command line tools and Rust if they are missing, about 10 minutes.
