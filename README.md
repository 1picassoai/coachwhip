<p align="center"><img src="docs/coachwhip.png" alt="A coachwhip snake on sand" width="720"></p>

# Coachwhip

**Run mixture-of-experts models that are bigger than your memory.**

An 80B model, running comfortably on a 16 GB Mac. Coachwhip never needs the whole model in memory, because a mixture-of-experts model never needs the whole model for one word.

One Rust binary. Everything on the GPU. Nothing leaves your machine.

**Tried it? Tell us how it ran**, good or bad, especially if it was slow, the fan worked hard, or it ran hot. [Open an issue](https://github.com/1picassoai/coachwhip/issues) with your Mac (chip and memory), macOS version, the model file, the `--bank` you used, and the speed line the chat page shows under each answer. Questions and ideas are welcome there too.

<p align="center"><img src="docs/coachwhip-demo.gif" alt="Qwen3-Coder-Next (80B) writing TypeScript in the Coachwhip chat on a 16 GB Mac" width="720"></p>

## Why it works

A mixture-of-experts (MoE) model is made of many small experts, and a router picks only a few of them for each word. Qwen3-Coder-Next has 80 billion parameters, but each word wakes up only about 3 billion of them.

Coachwhip is built around that fact:

- **Streams experts, not files.** The small, always-used part of the model lives on the GPU. The experts stay on the SSD, and for each word Coachwhip reads exactly the experts the router picked.
- **An expert bank on the GPU.** Recently used experts stay in a bank in the GPU's memory and are reused, so the SSD is only touched for experts the bank does not hold.
- **The early guess.** While the GPU works on one layer, Coachwhip runs the router of a layer four steps ahead on the same input, so it already knows most of the experts that layer will want, and reads them in while the GPU is busy. A guess that would land too late is dropped rather than evicting an expert still in use. The last few layers, which have no later router, use the routes the model has taken before, and every chat teaches that table more.
- **One GPU dispatch per layer.** All the experts a layer picked run together in a single fused kernel.
- **Reads while it computes.** When a prompt needs more experts than the bank holds, the GPU starts on each expert the moment its bytes land, while the rest are still being read.
- **Unified memory, used properly.** On Apple silicon the CPU and GPU share one memory, so an expert read from the SSD lands straight where the GPU reads it. No copies.

## Numbers

Mac mini M4, 16 GB, both models at Q4_K_M.

| | Qwen3-Coder-30B-A3B | Qwen3-Coder-Next (80B) |
|---|---|---|
| Model file | 18.6 GB | 48.4 GB |
| Follow-up in a chat | reads only your new words | reads only your new words |

While it reads your prompt, the page shows the layer it is on, from the first second.

## What it does to your Mac

Honest numbers, so you can decide.

| | 30B on 16 GB (bank 44) | 80B on 16 GB (bank 56) |
|---|---|---|
| Held by Coachwhip while the model is hot | about 11 GB | about 7.5 GB |
| Left for everything else | about 3 GB | about 7 GB |

The 30B's experts are bigger than the 80B's, so it holds more. If memory runs short, macOS squeezes your other apps, not Coachwhip. If the GPU runs out of room, Coachwhip says so on the page and stops; close some apps, or start it again with a smaller `--bank`.

With 24 GB or more the 80B runs with room to spare. `--bank` trades GPU memory for speed in either direction; on 16 GB, 56 is the most the 80B should use.

## Models

Coachwhip is an engine for MoE models, in GGUF format.

| Model | Architecture | Status |
|---|---|---|
| [Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) | qwen3moe | Tested. |
| [Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next) (80B) | qwen3next | Tested. See below for the file to use. |
| [Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) | qwen3next | Tested. |
| Qwen3-30B-A3B, other Qwen3 MoE | qwen3moe | Supported. |
| Mixtral, Qwen2-MoE, OLMoE, DeepSeek, gpt-oss, GLM | | Coming next. |

Speed and a quality check for each model file, measured on a 16 GB Mac: [docs/MODELS.md](docs/MODELS.md).

Try other MoE models and tell us what happens. If a model's architecture is not supported yet, Coachwhip says so, and an issue is the fastest way to move it up the list.

**Which GGUF file.** Coachwhip reads the standard GGUF quantisations (Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, F16). Some newer conversions store a few tensors in MXFP4 or fuse the attention weights into one tensor; Coachwhip does not read those yet. For Qwen3-Coder-Next, [MaziyarPanahi's Q3_K_M and Q4_K_M](https://huggingface.co/MaziyarPanahi/Qwen3-Coder-Next-GGUF) load as is.

## What you need

| | 30B | 80B |
|---|---|---|
| Mac | Apple silicon (M1 or later). Tested on a Mac mini M4. | the same |
| Memory | 16 GB | 16 GB; 24 GB has room to spare |
| Disk | 19 GB for the model, 3 GB for the build | 39 GB (Q3_K_M) or 49 GB (Q4_K_M) for the model, 3 GB for the build |
| macOS | Tested on macOS 26 | the same |

## Install

One line in Terminal:

```sh
curl -fsSL https://raw.githubusercontent.com/1picassoai/coachwhip/main/install.sh | bash
```

It checks your Mac, installs what is missing (Apple's command line tools, Rust) and builds Coachwhip into `~/coachwhip`, in about 10 minutes. Run it again at any time: it skips whatever is already done.

If a window asks to install Apple's command line tools, finish it and run the line again.

## Bring a model

Coachwhip is the engine; the model is yours. Any GGUF of a supported model works (see Models).

**Recommended: Qwen3-Coder-Next, the 80B coder, in its Q3_K_M file.** Download the [Q3_K_M GGUF](https://huggingface.co/MaziyarPanahi/Qwen3-Coder-Next-GGUF) (38 GB) and its [tokenizer.json](https://huggingface.co/Qwen/Qwen3-Coder-Next), then start it with a bank of 70:

```sh
~/coachwhip/coachwhip.sh ~/Downloads/Qwen3-Coder-Next.Q3_K_M.gguf ~/Downloads/tokenizer.json 70
```

The Q3_K_M file writes faster than Q4_K_M on a 16 GB Mac, and the two scored the same on our quality check; the measurements are in [docs/MODELS.md](docs/MODELS.md).

| Model | GGUF file | Tokenizer | Bank on 16 GB |
|---|---|---|---|
| Qwen3-Coder-Next (80B), recommended | [MaziyarPanahi Q3_K_M](https://huggingface.co/MaziyarPanahi/Qwen3-Coder-Next-GGUF) | [tokenizer.json](https://huggingface.co/Qwen/Qwen3-Coder-Next) | 70 |
| Qwen3-Coder-Next (80B), the 4-bit file | [MaziyarPanahi Q4_K_M](https://huggingface.co/MaziyarPanahi/Qwen3-Coder-Next-GGUF) | [tokenizer.json](https://huggingface.co/Qwen/Qwen3-Coder-Next) | 56 |
| Qwen3-Coder-30B-A3B, a smaller download | [unsloth Q4_K_M](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF) | [tokenizer.json](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) | 44 |

## Use

**Chat in your browser:**

```sh
~/coachwhip/coachwhip.sh /path/to/model.gguf /path/to/tokenizer.json 56
```

The last number is the bank (44 if you leave it out). The chat opens in your browser. Type, press Send, and the answer streams as it is written. Follow-up questions read only your new words; press **New chat** to start again. Stop Coachwhip with Ctrl+C.

**From your coding tools:** while Coachwhip runs, it also speaks the OpenAI chat API, so any tool that takes an OpenAI base URL can use the model.

| Setting | Value |
|---|---|
| Base URL | `http://127.0.0.1:8090/v1` |
| API key | anything; it is ignored |
| Model | the model's file name, as listed at `/v1/models` |

A conversation holds up to 16K tokens: the history, the question and the answer together. That limit is sized to fit a 16 GB Mac; the models themselves support longer.

Only programs on this Mac can reach it. Follow-ups that resend the whole conversation read only the new part. The chat page and your tools share one model, one request at a time; when a tool has used it, the page's next question starts a new chat.

Tested with the official `openai` Python client and with [Aider](https://aider.chat), the coding agent for the terminal:

```sh
aider --openai-api-base http://127.0.0.1:8090/v1 --openai-api-key anything \
      --model openai/Qwen3-Coder-Next-Q4_K_M --map-tokens 0 src/queue.ts
```

<p align="center"><img src="docs/coachwhip-aider.png" alt="Aider using Qwen3-Coder-Next (80B) through Coachwhip on a 16 GB Mac: it finds a sort-order bug and rewrites the file" width="720"></p>

<p align="center"><sub>Aider on a Windows laptop, reaching the Mac through an SSH tunnel, so the requests arrive as a local program.</sub></p>

`--map-tokens 0` turns off Aider's repository map, which adds up to about a thousand tokens to each request by default; with it on, each request takes longer to read.

**One answer in the terminal:**

```sh
cd ~/coachwhip
./target/release/coachwhip --model /path/to/model.gguf --tokenizer /path/to/tokenizer.json --prompt "Write a Rust function that reverses a linked list."
```

**Useful options:**

| Option | What it does |
|---|---|
| `--chat 8090` | Serve the chat page on that port, on this machine only |
| `--max-tokens 4000` | Longest answer |
| `--temperature 0` | Always pick the likeliest word (same answer every time) |
| `--bank 44` | Expert slots kept on the GPU per layer (56 for the 80B, 70 for its Q3_K_M file) |
| `--prefetch 10` | Experts guessed and read in ahead, per layer; 0 turns the guess off |
| `--ahead 4` | How many layers ahead the guess looks |
| `--profile` | Print where the time went after each answer |
| `--help` | Everything else |

## Roadmap

- More MoE families: Mixtral, Qwen2-MoE, OLMoE, DeepSeek, gpt-oss, GLM.
- MXFP4 and fused-attention GGUF files.
- An adapter that reads your own repository, on your own machine.
- More platforms.

## Credits

Built on [Candle](https://github.com/huggingface/candle). `crates/engine/src/model.rs` is derived from Candle's Qwen3 model, and the fused expert kernel is ggml's `mul_mv_id`, which ships inside Candle. The block-at-a-time linear attention follows [Gated Delta Networks](https://arxiv.org/abs/2412.06464) (Yang, Kautz and Hatamizadeh, ICLR 2025).

## Licence

Apache-2.0. See [LICENSE](LICENSE).
