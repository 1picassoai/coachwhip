# Models

Every model Coachwhip runs: which file to download, how to start it, what it needs from your Mac, and what it does on a 16 GB Mac, measured the same way every time. These are measurements, not promises: your Mac, your prompt and your other apps will move them. Run the same checks yourself (at the end) and post what you get in an [issue](https://github.com/1picassoai/coachwhip/issues).

## Supported models

Coachwhip is an engine for mixture-of-experts models in GGUF format.

| Model | Architecture | Status |
|---|---|---|
| [gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) | gpt-oss | Tested, after `prepare` (see [gpt-oss-120b](#gpt-oss-120b)). Thinks before it answers. |
| [Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) | qwen35moe | Tested, after `prepare` (see [Qwen3.5-122B-A10B](#qwen35-122b-a10b)). |
| [Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) | qwen3moe | Tested. |
| [Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next) (80B) | qwen3next | Tested. See [Qwen3-Coder-Next](#qwen3-coder-next-80b) for the file to use. |
| [Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) | qwen3next | Tested. |
| [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | qwen35moe | Tested. Thinks when asked to. |
| Qwen3-30B-A3B, other Qwen3 MoE | qwen3moe | Supported. |
| Qwen3.5 MoE, Qwen3.6-14B-A3B (community) | qwen35moe | Supported. The 14B fits in 16 GB on its own; Coachwhip does not make it faster. |

Try other MoE models and tell us what happens. If a model's architecture is not supported yet, Coachwhip says so, and an issue is the fastest way to move it up the list.

**Which GGUF file.** Coachwhip reads the standard GGUF quantisations (Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, F16). gpt-oss ships in MXFP4, which `prepare` reads and squeezes. Some newer Qwen conversions store a few tensors in MXFP4 or fuse the attention weights into one tensor; Coachwhip does not read those yet. For Qwen3-Coder-Next, [MaziyarPanahi's Q3_K_M and Q4_K_M](https://huggingface.co/MaziyarPanahi/Qwen3-Coder-Next-GGUF) load as is.

## Files and start lines

| Model | GGUF file | Tokenizer | Bank on 16 GB |
|---|---|---|---|
| gpt-oss-120b, after `prepare` | [ggml-org MXFP4](https://huggingface.co/ggml-org/gpt-oss-120b-GGUF) | [tokenizer.json](https://huggingface.co/openai/gpt-oss-120b) | 25 |
| Qwen3.5-122B-A10B, after `prepare` | [mradermacher Q3_K_M](https://huggingface.co/mradermacher/Qwen3.5-122B-A10B-GGUF) | [tokenizer.json](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) | 40 |
| Qwen3-Coder-Next (80B) | [MaziyarPanahi Q3_K_M](https://huggingface.co/MaziyarPanahi/Qwen3-Coder-Next-GGUF) | [tokenizer.json](https://huggingface.co/Qwen/Qwen3-Coder-Next) | 70 |
| Qwen3-Coder-Next (80B), the 4-bit file | [MaziyarPanahi Q4_K_M](https://huggingface.co/MaziyarPanahi/Qwen3-Coder-Next-GGUF) | [tokenizer.json](https://huggingface.co/Qwen/Qwen3-Coder-Next) | 56 |
| Qwen3.6-35B-A3B | [unsloth UD-Q4_K_M](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) | [tokenizer.json](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | 44 |
| Qwen3-Coder-30B-A3B, a smaller download | [unsloth Q4_K_M](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF) | [tokenizer.json](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) | 44 |

Any model that needs no squeeze starts with one line, the bank last:

```sh
~/coachwhip/coachwhip.sh ~/Downloads/Qwen3-Coder-Next.Q3_K_M.gguf ~/Downloads/tokenizer.json 70
```

## What each model needs

| | gpt-oss-120b | 122B | 80B | 30B |
|---|---|---|---|---|
| Mac | Apple silicon, M4 or later recommended; M1 to M3 run it slower. Tested on a Mac mini M4. | the same | the same | the same |
| Memory | 16 GB | 16 GB | 16 GB; 24 GB has room to spare | 16 GB |
| Disk | 106 GB free while squeezing, 43 GB once you delete the download | 105 GB free while squeezing, 46 GB once you delete the download | 39 GB (Q3_K_M) or 49 GB (Q4_K_M) | 19 GB |
| macOS | Tested on macOS 26 | the same | the same | the same |

**What it does to your Mac.** While a model is hot, Coachwhip holds its bank and the model's always-used part on the GPU; the rest of the memory is yours.

| | 30B on 16 GB (bank 44) | 80B on 16 GB (bank 56) | 122B on 16 GB (bank 40) |
|---|---|---|---|
| Held by Coachwhip while the model is hot | about 11 GB | about 7.5 GB | 11.2 to 12.8 GB |
| Left for everything else | about 3 GB | about 7 GB | 1.5 to 2.3 GB at its tightest |

The 30B's experts are bigger than the 80B's, so it holds more. If memory runs short, macOS squeezes your other apps, not Coachwhip. If the GPU runs out of room, Coachwhip says so on the page and stops; close some apps, or start it again with a smaller `--bank`. With 24 GB or more the 80B runs with room to spare. `--bank` trades GPU memory for speed in either direction; on 16 GB, 56 is the most the 80B should use.

## How we measure

- **Machine:** Mac mini M4, 16 GB, macOS 26.
- **Engine:** the version named in each section, with its default settings, temperature 0. The 80B and the 35B were measured on 0.3.0 (`--prefetch 10 --ahead 4`); the 122B on 0.4.0 (`--prefetch 6 --ahead 2`, its own default); gpt-oss-120b on 0.5.0.
- **Speed:** the five coding prompts in [`bench/speed/prompts.json`](../bench/speed/prompts.json), each a fresh conversation capped at 300 tokens, run by [`bench/speed/run.py`](../bench/speed/run.py). "Writing" is answer tokens per second from the first token to the last, the same thing the chat shows under each answer; the table gives the mean over the five prompts, with the slowest and fastest in brackets.
- **Quality:** 8 small coding tasks in [`bench/quality/tasks.json`](../bench/quality/tasks.json). Each answer's code is run against tests the model never sees ([`bench/quality/tests.py`](../bench/quality/tests.py)); a task passes only if every test does. Quality depends on the model file, not on Coachwhip's settings: the bank and the prefetch change how fast an answer comes, never what it says. Fast modes (`--experts`, `--stand-ins`) are the exception and are measured separately.

## gpt-oss-120b

OpenAI's open model (architecture `gpt-oss`): 117 billion parameters, 36 layers, 128 experts of which 4 are used per word. It thinks before it answers, and the chat page shows the thinking.

**Set up.** It runs from a squeezed file that `coachwhip-prepare` makes on your Mac from the official file.

1. Download the official [MXFP4 GGUF](https://huggingface.co/ggml-org/gpt-oss-120b-GGUF) (63.4 GB) and the model's [tokenizer.json](https://huggingface.co/openai/gpt-oss-120b).
2. Squeeze it. About 13 minutes on an M4. You need about **106 GB free** while it runs: the 63 GB download and the 43 GB squeezed file sit side by side.

```sh
~/coachwhip/target/release/coachwhip-prepare ~/Downloads/gpt-oss-120b-MXFP4.gguf ~/Downloads/gpt-oss-120b.Q2A.gguf
```

   The squeeze takes every expert matrix to Q2_K, rows padded to whole 256-weight blocks; everything else in the file is copied as it is. Nothing is downloaded from us.

3. Delete the download. Coachwhip only uses the squeezed file:

```sh
rm ~/Downloads/gpt-oss-120b-MXFP4.gguf
```

4. Start it in one of two modes. Coachwhip picks the bank that fits your Mac (25 on 16 GB):

```sh
# exact: the model's own 4 experts per word, every one read
~/coachwhip/coachwhip.sh ~/Downloads/gpt-oss-120b.Q2A.gguf ~/Downloads/tokenizer.json 25 8090 --parallel --flash-reads --max-tokens 8000

# fast: stand-ins for the light picks that are not in the bank
~/coachwhip/coachwhip.sh ~/Downloads/gpt-oss-120b.Q2A.gguf ~/Downloads/tokenizer.json 25 8090 --parallel --flash-reads --stand-ins --max-tokens 8000
```

`--max-tokens 8000` gives its thinking room: the cap counts thinking and answer together, and the default of 4,000 can end a hard task inside the thinking.

**Measured.** Mac mini M4, 16 GB, `--parallel --flash-reads`, bank 25, temperature 0, on one coding task, a cancellable async task pool in TypeScript (about 4,000 words of thinking and code), against 20 hidden checks:

| Mode | Writing | Score of 5 |
|---|---|---|
| Fast, `--stand-ins` | 9.6 to 10.5 tok/s, three runs | 5.0, 3.8, 4.8 (mean 4.5) |
| Exact, the model's own 4 experts | 4.2 tok/s | 4.8 |

This model is measured on its own task, not the five-prompt speed set: its thinking makes 300-token answers mostly thinking. Every setup measured, what fast mode does, how it behaves on everyday and on hard tasks: [GPT-OSS.md](GPT-OSS.md).

- **Why the squeeze:** at MXFP4's 13.2 MB per expert the bank holds 15 experts a layer; at Q2_K's 8.7 MB it holds 25. On the task above the squeezed file scores 4.8 against the original download's 5.0, which runs at 2.2 tok/s.
- **Fast mode changes the answers.** A word's first expert is always read, and so is its second whenever the router gives it at least 0.3 of the weight. A lighter pick that is not already in the bank is replaced by the best expert that is, by the router's own scores, and only when that expert's score is close to the real pick's; otherwise the real one is read. It is a choice, never the default.
- Tokenizer: [tokenizer.json](https://huggingface.co/openai/gpt-oss-120b) from the model's own page.

## Qwen3.5-122B-A10B

The biggest model Coachwhip runs on 16 GB (architecture `qwen35moe`): 48 layers, 256 experts of which 8 are used per word. Its experts are too big to stream fast as they come, so Coachwhip squeezes the file once, on your Mac.

**Set up.**

1. Download the publisher's [Q3_K_M GGUF](https://huggingface.co/mradermacher/Qwen3.5-122B-A10B-GGUF) (59 GB) and the model's [tokenizer.json](https://huggingface.co/Qwen/Qwen3.5-122B-A10B).
2. Squeeze it. About 5 to 9 minutes on an M4. You need about **105 GB free** while it runs: the 59 GB download and the 46 GB squeezed file sit side by side.

```sh
~/coachwhip/target/release/coachwhip-prepare ~/Downloads/Qwen3.5-122B-A10B.Q3_K_M.gguf ~/Downloads/Qwen3.5-122B-A10B.Q2X.gguf
```

   The squeeze takes each expert's gate and up matrices to Q2_K on the GPU and its down matrix to Q3_K; everything else in the file is copied as it is. Nothing is downloaded from us.

3. Delete the download. Coachwhip only uses the squeezed file, and keeping both takes about 105 GB:

```sh
rm ~/Downloads/Qwen3.5-122B-A10B.Q3_K_M.gguf
```

4. Start it in one of two modes, bank 40. If your Mac cannot hold a long chat at that size, Coachwhip says so and uses less:

```sh
# exact: the model's own 8 experts per word
~/coachwhip/coachwhip.sh ~/Downloads/Qwen3.5-122B-A10B.Q2X.gguf ~/Downloads/tokenizer.json 40

# fast: the router's top 4 experts per word, and the GPU runs ahead of the CPU
~/coachwhip/coachwhip.sh ~/Downloads/Qwen3.5-122B-A10B.Q2X.gguf ~/Downloads/tokenizer.json 40 8090 --experts 4 --parallel
```

**Measured.** Fast mode (`--experts 4 --parallel`), Mac mini M4, 16 GB, bank 40:

| Tokens | Speed | Coachwhip uses | Left for macOS and your apps |
|---|---|---|---|
| 6,000-token answer | 4.2 tok/s | 11.2 GB | 2.3 GB at its tightest |
| 12,000-token prompt | 4.3 tok/s | 12.8 GB | 1.5 GB at its tightest |

Measured 9 Oct 2026 on Coachwhip 0.4.0 by the tester, greedy. Exact mode (the model's own 8 experts, no `--experts`) wrote at 2.2 tok/s after the same 12,000-token prompt. Agent tasks through Aider, six of them in Python, TypeScript and Rust with hidden tests: fast mode passed 5 of 6, and exact mode passed the sixth. The 122B is tested to a 12,000-token prompt; longer is untested.

- **Why the squeeze:** at the file's own 4.9 MB per expert, 8 experts a word, the SSD cannot feed a 16 GB Mac faster than about 2 tok/s. At 3.4 MB per expert the same bank holds more and each read lands sooner.
- **Fast mode changes the answers:** the model was trained to combine 8 experts per word; asking for 4 drops the four lightest, and its answers can differ from what the full model would say. The one agent task fast mode failed had a built-in contradiction: it argued with itself until it ran out of room, where exact mode solved it in one go. It is a choice, never the default.
- Tokenizer: [tokenizer.json](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) from the model's own page.

## Qwen3-Coder-Next (80B)

The 80B coder has no thinking mode. Download the file below and its [tokenizer.json](https://huggingface.co/Qwen/Qwen3-Coder-Next), then start it with a bank of 70 for the Q3_K_M file or 56 for the Q4_K_M:

```sh
~/coachwhip/coachwhip.sh ~/Downloads/Qwen3-Coder-Next.Q3_K_M.gguf ~/Downloads/tokenizer.json 70
```

| GGUF file | Size | `--bank` on 16 GB | Bank memory | Writing, 0.3.0 | Writing, 0.2.0 | Quality |
|---|---|---|---|---|---|---|
| [MaziyarPanahi Q3_K_M](https://huggingface.co/MaziyarPanahi/Qwen3-Coder-Next-GGUF) | 38.2 GB | 70 | 5.5 GB | **5.4 to 5.7 tok/s** over three runs (prompts from 5.0 to 5.9) | 4.5 tok/s | 7 / 8 |
| [MaziyarPanahi Q4_K_M](https://huggingface.co/MaziyarPanahi/Qwen3-Coder-Next-GGUF) | 48.4 GB | 56 | 5.5 GB | **4.1 tok/s**, one run (prompts from 3.9 to 4.4) | 3.4 tok/s | 7 / 8 |

0.3.0 speed measured 5 and 6 Oct 2026 over the five prompts; two of them finish before the 300-token cap, so their speed is measured over a shorter answer. The Q3_K_M figure is three runs by two people on the same Mac: 5.66 on an otherwise idle machine, 5.38 and 5.39 with a background load of about 2. Expect a few percent less when your Mac is doing other things. 0.2.0 speed measured 3 Oct 2026 on the first prompt only. Quality measured 4 and 5 Oct 2026, the same result both times.

- **Why 0.3.0 is faster:** it guesses each layer's experts four layers early with the model's own routers and reads them in while the GPU is busy, so fewer reads happen with the GPU waiting.
- **Why Q3_K_M is faster:** its experts are 1.6 MB instead of 2.0 MB, so each one read from the SSD arrives sooner, and the same bank memory holds 70 of them per layer instead of 56. The two files scored the same on the quality check.
- **Quality:** both files failed one task, a different one each. Q4_K_M's word wrap counted a space before the first word of each line; Q3_K_M's duration parser accepted a repeated unit (`1h1h`). Eight tasks is a small sample: treat it as "no difference found", not "no difference".
- Both use the same [tokenizer.json](https://huggingface.co/Qwen/Qwen3-Coder-Next).
- **On 0.4.0:** the Q3_K_M file was run again on the release build at bank 70 and answered correctly; the full speed and quality checks above were not repeated.

## Qwen3.6-35B-A3B

Qwen's newest MoE generation (architecture `qwen35moe`): 40 layers, 256 experts of which 8 are used per word, plus a shared expert. It has a thinking mode; these runs have it off, the default.

| GGUF file | Size | `--bank` on 16 GB | Bank memory | Writing, 0.3.0 | Quality |
|---|---|---|---|---|---|
| [unsloth UD-Q4_K_M](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) | 22.1 GB | 44 | 3.3 GB | **6.2 tok/s**, one run (prompts from 5.8 to 6.5) | 8 / 8 |
| the same file | 22.1 GB | 56 | 4.3 GB | 6.0 tok/s, one run (prompts from 5.8 to 6.5) | 8 / 8 |

Measured 6 Oct 2026: one person, one run per bank, on the same Mac with a background load of about 1.8, so these are a first measurement, not a settled range like the 80B's. Expect a few percent less when your Mac is doing other things, and tell us what you get. The first word came after 2.5 to 3.9 seconds on each prompt.

- **Why bank 44, not 56:** the two banks write at the same speed (a 2,500-token thinking answer ran at 6.1 tok/s on both), but this model's non-expert part is heavier than the 80B's, and at bank 56 Coachwhip held about 7 GB against 5 GB at 44. On a 16 GB Mac, 56 ran the GPU out of memory once, on a long thinking answer that followed an earlier answer in the same chat. 44 leaves that room.
- **Quality:** all eight tasks passed, including the duration parser that both 80B files got wrong. Eight tasks is a small sample.
- Tokenizer: [tokenizer.json](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) from the model's own page.
- **On 0.4.0:** not re-measured. Its code path is unchanged, but nobody has run it on the release build.

## Qwen3-Coder-30B-A3B

The smaller download (architecture `qwen3moe`), 18.6 GB at Q4_K_M. Bank 44 on 16 GB. Its experts are bigger than the 80B's, so it holds about 11 GB while hot and leaves about 3 GB for everything else.

## Not measured yet

Qwen3-Coder-30B-A3B and Qwen3-Next-80B-A3B-Instruct run on Coachwhip but have not been through the speed and quality checks yet.

## Run it yourself

```sh
~/coachwhip/target/release/coachwhip --model /path/to/model.gguf --tokenizer /path/to/tokenizer.json \
    --bank 56 --temperature 0 --chat 8090
python3 bench/speed/run.py
python3 bench/quality/run.py
```

`bench/speed/run.py` streams the five prompts and prints each answer's writing speed and the mean. `bench/quality/run.py` asks the 8 tasks, runs each answer against the tests, and prints pass or fail per task. Both need only Python 3.
