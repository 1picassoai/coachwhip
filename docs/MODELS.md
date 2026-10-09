# Models

What each model file does on a 16 GB Mac, measured the same way every time. These are measurements, not promises: your Mac, your prompt and your other apps will move them. Run the same checks yourself (below) and post what you get in an [issue](https://github.com/1picassoai/coachwhip/issues).

## How we measure

- **Machine:** Mac mini M4, 16 GB, macOS 26.
- **Engine:** the version named in each section, with its default settings, temperature 0. The 80B and the 35B were measured on 0.3.0 (`--prefetch 10 --ahead 4`); the 122B on 0.4.0 (`--prefetch 6 --ahead 2`, its own default).
- **Speed:** the five coding prompts in [`bench/speed/prompts.json`](../bench/speed/prompts.json), each a fresh conversation capped at 300 tokens, run by [`bench/speed/run.py`](../bench/speed/run.py). "Writing" is answer tokens per second from the first token to the last, the same thing the chat shows under each answer; the table gives the mean over the five prompts, with the slowest and fastest in brackets.
- **Quality:** 8 small coding tasks in [`bench/quality/tasks.json`](../bench/quality/tasks.json). Each answer's code is run against tests the model never sees ([`bench/quality/tests.py`](../bench/quality/tests.py)); a task passes only if every test does. Quality depends on the model file, not on Coachwhip's settings: the bank and the prefetch change how fast an answer comes, never what it says.

## Qwen3.5-122B-A10B

The biggest model Coachwhip runs on 16 GB (architecture `qwen35moe`): 48 layers, 256 experts of which 8 are used per word. It runs from a **squeezed file** that `coachwhip-prepare` makes on your Mac from the publisher's Q3_K_M (59 GB): each expert's gate and up matrices to Q2_K on the GPU, its down matrix to Q3_K, the rest untouched; 45 GB written in 5 to 9 minutes on an M4, and about 105 GB free needed while both files exist. See the README, The 122B.

Fast mode (`--experts 4 --parallel`), Mac mini M4, 16 GB, bank 40:

| Tokens | Speed | Coachwhip uses | Left for macOS and your apps |
|---|---|---|---|
| 6,000-token answer | 4.2 tok/s | 11.2 GB | 2.3 GB at its tightest |
| 12,000-token prompt | 4.3 tok/s | 12.8 GB | 1.5 GB at its tightest |

Measured 9 Oct 2026 on Coachwhip 0.4.0 by the tester, greedy. Exact mode (the model's own 8 experts, no `--experts`) wrote at 2.2 tok/s after the same 12,000-token prompt. Agent tasks through Aider, six of them in Python, TypeScript and Rust with hidden tests: fast mode passed 5 of 6, and exact mode passed the sixth.

- **Why the squeeze:** at the file's own 4.9 MB per expert, 8 experts a word, the SSD cannot feed a 16 GB Mac faster than about 2 tok/s. At 3.4 MB per expert the same bank holds more and each read lands sooner.
- **Why fast mode changes answers:** the model was trained to combine 8 experts per word; asking for 4 drops the four lightest. The one agent task fast mode failed had a built-in contradiction: it argued with itself until it ran out of room, where exact mode solved it in one go.
- Tokenizer: [tokenizer.json](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) from the model's own page.

## Qwen3-Coder-Next (80B)

| GGUF file | Size | `--bank` on 16 GB | Bank memory | Writing, 0.3.0 | Writing, 0.2.0 | Quality |
|---|---|---|---|---|---|---|
| [MaziyarPanahi Q3_K_M](https://huggingface.co/MaziyarPanahi/Qwen3-Coder-Next-GGUF), recommended | 38.2 GB | 70 | 5.5 GB | **5.4 to 5.7 tok/s** over three runs (prompts from 5.0 to 5.9) | 4.5 tok/s | 7 / 8 |
| [MaziyarPanahi Q4_K_M](https://huggingface.co/MaziyarPanahi/Qwen3-Coder-Next-GGUF) | 48.4 GB | 56 | 5.5 GB | **4.1 tok/s**, one run (prompts from 3.9 to 4.4) | 3.4 tok/s | 7 / 8 |

0.3.0 speed measured 5 and 6 Oct 2026 over the five prompts; two of them finish before the 300-token cap, so their speed is measured over a shorter answer. The Q3_K_M figure is three runs by two people on the same Mac: 5.66 on an otherwise idle machine, 5.38 and 5.39 with a background load of about 2. Expect a few percent less when your Mac is doing other things. 0.2.0 speed measured 3 Oct 2026 on the first prompt only. Quality measured 4 and 5 Oct 2026, the same result both times.

- **Why 0.3.0 is faster:** it guesses each layer's experts four layers early with the model's own routers and reads them in while the GPU is busy, so fewer reads happen with the GPU waiting.
- **Why Q3_K_M is faster:** its experts are 1.6 MB instead of 2.0 MB, so each one read from the SSD arrives sooner, and the same bank memory holds 70 of them per layer instead of 56.
- **Quality:** both files failed one task, a different one each. Q4_K_M's word wrap counted a space before the first word of each line; Q3_K_M's duration parser accepted a repeated unit (`1h1h`). Eight tasks is a small sample: treat it as "no difference found", not "no difference".
- Both use the same [tokenizer.json](https://huggingface.co/Qwen/Qwen3-Coder-Next).
- **On 0.4.0:** the Q3_K_M file was run again on the release build at bank 70 and answered correctly; the full speed and quality checks above were not repeated.

## Qwen3.6-35B-A3B

Qwen's newest MoE generation (architecture `qwen35moe`): 40 layers, 256 experts of which 8 are used per word, plus a shared expert. It has a thinking mode; these runs have it off, the default.

| GGUF file | Size | `--bank` on 16 GB | Bank memory | Writing, 0.3.0 | Quality |
|---|---|---|---|---|---|
| [unsloth UD-Q4_K_M](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF), recommended | 22.1 GB | 44 | 3.3 GB | **6.2 tok/s**, one run (prompts from 5.8 to 6.5) | 8 / 8 |
| the same file | 22.1 GB | 56 | 4.3 GB | 6.0 tok/s, one run (prompts from 5.8 to 6.5) | 8 / 8 |

Measured 6 Oct 2026: one person, one run per bank, on the same Mac with a background load of about 1.8, so these are a first measurement, not a settled range like the 80B's. Expect a few percent less when your Mac is doing other things, and tell us what you get. The first word came after 2.5 to 3.9 seconds on each prompt.

- **Why bank 44, not 56:** the two banks write at the same speed (a 2,500-token thinking answer ran at 6.1 tok/s on both), but this model's non-expert part is heavier than the 80B's, and at bank 56 Coachwhip held about 7 GB against 5 GB at 44. On a 16 GB Mac, 56 ran the GPU out of memory once, on a long thinking answer that followed an earlier answer in the same chat. 44 leaves that room.
- **Quality:** all eight tasks passed, including the duration parser that both 80B files got wrong. Eight tasks is a small sample.
- Tokenizer: [tokenizer.json](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) from the model's own page.
- **On 0.4.0:** not re-measured. Its code path is unchanged, but nobody has run it on the release build.

## Not measured yet

Qwen3-Coder-30B-A3B and Qwen3-Next-80B-A3B-Instruct run on Coachwhip (see the README) but have not been through these checks yet.

## Run it yourself

```sh
~/coachwhip/target/release/coachwhip --model /path/to/model.gguf --tokenizer /path/to/tokenizer.json \
    --bank 56 --temperature 0 --chat 8090
python3 bench/speed/run.py
python3 bench/quality/run.py
```

`bench/speed/run.py` streams the five prompts and prints each answer's writing speed and the mean. `bench/quality/run.py` asks the 8 tasks, runs each answer against the tests, and prints pass or fail per task. Both need only Python 3.
