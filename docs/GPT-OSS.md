# gpt-oss-120b on a 16 GB Mac

OpenAI's gpt-oss-120b has 117 billion parameters and 128 experts in each of its 36 layers, of which it uses 4 for each word. On a Mac mini M4 with 16 GB, Coachwhip runs it at about 4.2 tok/s with every expert read for real, and at about 10 tok/s in fast mode.

All numbers here: Mac mini M4, 16 GB, macOS 26, the squeezed file (`prepare`, every expert matrix at Q2_K), `--parallel --flash-reads`, temperature 0. The task is one prompt, a generic cancellable async task pool in TypeScript with dependency resolution; the model thinks, then writes the code, about 4,000 words in all. The code is run against 20 hidden checks (ordering, the concurrency limit, failure cascades, cancellation, a task that ignores its abort signal, an empty list, and others), 4 checks to a point. A second version of the task, the same job worded as a build-pipeline runner, checks that a score is not a fluke.

## Speed and score, setup by setup

| Setup | Speed | Score | Second version |
|---|---|---|---|
| 4 experts, all read for real | 4.2 tok/s | 4.8 | |
| 3 experts, all read for real | 5.6 tok/s | 4.5 | |
| 4 experts, the first 2 real, stand-ins for the rest | 7.8 tok/s | 4.5 | 3.5 (7.9 tok/s) |
| **4 experts, stand-ins by weight (`--stand-ins`)** | **9.8 tok/s** | **4.0** | **4.0 (10.0 tok/s)** |
| 4 experts, only the first real | 9.6 tok/s | 4.5 | 0.8 (10.2 tok/s) |
| 2 experts | 7.1 tok/s | 0 (the code does not compile) | |

`--stand-ins` is the only setup near 10 tok/s that held its score on both versions of the task. Reading fewer experts (3 or 2) costs more than standing in for the light ones: the model was trained to use 4.

## What makes it fast

Speed on a 16 GB Mac is set by one thing: how many bytes each word reads from the SSD. Everything below cuts those bytes or hides them.

- **The squeeze.** `prepare` takes each expert from 13.2 MB (MXFP4) to 8.7 MB (Q2_K, rows padded to whole 256-weight blocks), so the bank on the GPU holds 25 experts a layer instead of 15. On this task the squeezed file scores the same as the original download.
- **The expert bank.** Recently used experts stay on the GPU and are reused; with every expert read for real, about 4 picks in 5 are already there.
- **The early guess.** While the GPU works on one layer, Coachwhip runs the router of a later layer on the same input and reads the experts it will want; for gpt-oss, 2 guesses a layer, 2 layers ahead.
- **Flash reads (`--flash-reads`).** Experts are read straight from the SSD, past macOS's page cache.
- **Stand-ins (`--stand-ins`).** A word's first pick is always read for real, and so is the second whenever the router gives it at least 0.3 of the weight. A lighter pick that is not in the bank is replaced by the best expert that is, by the router's own scores, and the weights are recomputed over the experts actually used. Bank hits go from about 78% to about 97%, and the bytes read per word fall by about seven-eighths.
- **Lean attention.** gpt-oss looks back only 128 words on every other layer; those layers keep only that window, and the shared keys are never copied out per head.

## Its chat format

gpt-oss speaks OpenAI's harmony format, with an analysis channel for its thinking and a final channel for the answer. The chat page shows the thinking in a box above the answer; the OpenAI-compatible API returns only the final answer.

## Not tested

- Fast mode on tasks other than the one above.
- Chats past 16K tokens, and Macs other than an M4 with 16 GB.
