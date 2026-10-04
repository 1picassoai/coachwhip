# Models

What each model file does on a 16 GB Mac, measured the same way every time. These are measurements, not promises: your Mac, your prompt and your other apps will move them. Run the same checks yourself (below) and post what you get in an [issue](https://github.com/1picassoai/coachwhip/issues).

## How we measure

- **Machine:** Mac mini M4, 16 GB, macOS 26.
- **Engine:** Coachwhip 0.2.0, temperature 0.
- **Speed:** one answer to *"Write a Python function that merges overlapping intervals, with a short explanation."*, capped at 300 tokens. "Writing" is tokens written per second once the answer has started, as Coachwhip counts it: the same figure the chat shows under each answer.
- **Quality:** 8 small coding tasks in [`bench/quality/tasks.json`](../bench/quality/tasks.json). Each answer's code is run against tests the model never sees ([`bench/quality/tests.py`](../bench/quality/tests.py)); a task passes only if every test does. Quality depends on the model file, not on Coachwhip's settings: the bank and the prefetch change how fast an answer comes, never what it says.

## Qwen3-Coder-Next (80B)

| GGUF file | Size | `--bank` on 16 GB | Bank memory | Writing | Quality |
|---|---|---|---|---|---|
| [MaziyarPanahi Q4_K_M](https://huggingface.co/MaziyarPanahi/Qwen3-Coder-Next-GGUF) | 48.4 GB | 56 | 5.5 GB | 3.4 tok/s | 7 / 8 |
| [MaziyarPanahi Q3_K_M](https://huggingface.co/MaziyarPanahi/Qwen3-Coder-Next-GGUF) | 38.2 GB | 70 | 5.5 GB | 4.5 tok/s | 7 / 8 |

Speed measured 3 Oct 2026; quality measured 4 Oct 2026. At bank 56, Q3_K_M wrote 4.4 tok/s.

- **Why Q3_K_M is faster:** its experts are 1.6 MB instead of 2.0 MB, so each one read from the SSD arrives sooner, and the same bank memory holds 70 of them per layer instead of 56.
- **Quality:** both files failed one task, a different one each. Q4_K_M's word wrap counted a space before the first word of each line; Q3_K_M's duration parser accepted a repeated unit (`1h1h`). Eight tasks is a small sample: treat it as "no difference found", not "no difference".
- Both use the same [tokenizer.json](https://huggingface.co/Qwen/Qwen3-Coder-Next).

## Not measured yet

Qwen3-Coder-30B-A3B and Qwen3-Next-80B-A3B-Instruct run on Coachwhip (see the README) but have not been through these checks yet.

## Run it yourself

```sh
~/coachwhip/target/release/coachwhip --model /path/to/model.gguf --tokenizer /path/to/tokenizer.json \
    --bank 56 --temperature 0 --chat 8090
python3 bench/quality/run.py
```

`run.py` asks the 8 tasks through Coachwhip's API, runs each answer against the tests, and prints pass or fail per task. It needs only Python 3. The chat's speed line gives the writing speed for any prompt you like.
