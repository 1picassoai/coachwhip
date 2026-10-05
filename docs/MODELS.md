# Models

What each model file does on a 16 GB Mac, measured the same way every time. These are measurements, not promises: your Mac, your prompt and your other apps will move them. Run the same checks yourself (below) and post what you get in an [issue](https://github.com/1picassoai/coachwhip/issues).

## How we measure

- **Machine:** Mac mini M4, 16 GB, macOS 26.
- **Engine:** Coachwhip 0.3.0 with its default settings (`--prefetch 10 --ahead 4`), temperature 0.
- **Speed:** the five coding prompts in [`bench/speed/prompts.json`](../bench/speed/prompts.json), each a fresh conversation capped at 300 tokens, run by [`bench/speed/run.py`](../bench/speed/run.py). "Writing" is answer tokens per second from the first token to the last, the same thing the chat shows under each answer; the table gives the mean over the five prompts, with the slowest and fastest in brackets.
- **Quality:** 8 small coding tasks in [`bench/quality/tasks.json`](../bench/quality/tasks.json). Each answer's code is run against tests the model never sees ([`bench/quality/tests.py`](../bench/quality/tests.py)); a task passes only if every test does. Quality depends on the model file, not on Coachwhip's settings: the bank and the prefetch change how fast an answer comes, never what it says.

## Qwen3-Coder-Next (80B)

| GGUF file | Size | `--bank` on 16 GB | Bank memory | Writing, 0.3.0 | Writing, 0.2.0 | Quality |
|---|---|---|---|---|---|---|
| [MaziyarPanahi Q3_K_M](https://huggingface.co/MaziyarPanahi/Qwen3-Coder-Next-GGUF), recommended | 38.2 GB | 70 | 5.5 GB | **5.7 tok/s** (5.5 to 5.9) | 4.5 tok/s | 7 / 8 |
| [MaziyarPanahi Q4_K_M](https://huggingface.co/MaziyarPanahi/Qwen3-Coder-Next-GGUF) | 48.4 GB | 56 | 5.5 GB | **4.1 tok/s** (3.9 to 4.4) | 3.4 tok/s | 7 / 8 |

0.3.0 speed measured 5 Oct 2026 over the five prompts; 0.2.0 speed measured 3 Oct 2026 on the first prompt only. Quality measured 4 and 5 Oct 2026, the same result both times.

- **Why 0.3.0 is faster:** it guesses each layer's experts four layers early with the model's own routers and reads them in while the GPU is busy, so fewer reads happen with the GPU waiting.
- **Why Q3_K_M is faster:** its experts are 1.6 MB instead of 2.0 MB, so each one read from the SSD arrives sooner, and the same bank memory holds 70 of them per layer instead of 56.
- **Quality:** both files failed one task, a different one each. Q4_K_M's word wrap counted a space before the first word of each line; Q3_K_M's duration parser accepted a repeated unit (`1h1h`). Eight tasks is a small sample: treat it as "no difference found", not "no difference".
- Both use the same [tokenizer.json](https://huggingface.co/Qwen/Qwen3-Coder-Next).

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
