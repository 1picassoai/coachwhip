# gpt-oss-120b on a 16 GB Mac

OpenAI's gpt-oss-120b has 117 billion parameters and 128 experts in each of its 36 layers, of which it uses 4 for each word. On a Mac mini M4 with 16 GB, Coachwhip runs it at about 4.2 tok/s with every expert read for real, and at about 10 tok/s in fast mode.

All numbers here: Mac mini M4, 16 GB, macOS 26, the squeezed file (`prepare`, every expert matrix at Q2_K), `--parallel --flash-reads`, temperature 0. The task is one prompt, a generic cancellable async task pool in TypeScript with dependency resolution; the model thinks, then writes the code, about 4,000 words in all. The code is run against 20 hidden checks (ordering, the concurrency limit, failure cascades, cancellation, a task that ignores its abort signal, an empty list, and others), 4 checks to a point. A second version of the task, the same job worded as a build-pipeline runner, checks that a score is not a fluke.

## Speed and score, setup by setup

| Setup | Speed | Score | Second version |
|---|---|---|---|
| 4 experts, all read for real | 4.2 tok/s | 4.8 | |
| 3 experts, all read for real | 5.6 tok/s | 4.5 | |
| 4 experts, the first 2 real, stand-ins for the rest | 7.8 tok/s | 4.5 | 3.5 (7.9 tok/s) |
| 4 experts, stand-ins by weight, no gap | 9.8 tok/s | 4.0 | 4.0 (10.0 tok/s) |
| **4 experts, stand-ins by weight with the gap (`--stand-ins` as shipped), three runs** | **9.8, 10.5, 9.6 tok/s** | **5.0, 3.8, 4.8** | not run |
| 4 experts, only the first real | 9.6 tok/s | 4.5 | 0.8 (10.2 tok/s) |
| 2 experts | 7.1 tok/s | 0 (the code does not compile) | |

`--stand-ins` as shipped is the only setup near 10 tok/s whose every run compiled and scored above 3.8. Without the gap, one run in three broke its file on a one-character slip (a run scored 0.8 before the gap was added). Reading fewer experts (3 or 2) costs more than standing in for the light ones: the model was trained to use 4.

## What makes it fast

Speed on a 16 GB Mac is set by one thing: how many bytes each word reads from the SSD. Everything below cuts those bytes or hides them.

- **The squeeze.** `prepare` takes each expert from 13.2 MB (MXFP4) to 8.7 MB (Q2_K, rows padded to whole 256-weight blocks), so the bank on the GPU holds 25 experts a layer instead of 15. On this task the squeezed file scores the same as the original download.
- **The expert bank.** Recently used experts stay on the GPU and are reused; with every expert read for real, about 4 picks in 5 are already there.
- **The early guess.** While the GPU works on one layer, Coachwhip runs the router of a later layer on the same input and reads the experts it will want; for gpt-oss, 2 guesses a layer, 2 layers ahead; with `--stand-ins` 1 guess a layer, since most picks are already in the bank and extra guesses only queue reads in front of the ones a word needs (1,000-word answers: 8.4 to 10.9 tok/s).
- **Flash reads (`--flash-reads`).** Experts are read straight from the SSD, past macOS's page cache.
- **Stand-ins (`--stand-ins`).** A word's first pick is always read for real, and so is the second whenever the router gives it at least 0.3 of the weight. A lighter pick that is not in the bank is replaced by the best expert that is, by the router's own scores, and the weights are recomputed over the experts actually used. The substitute is used only when its router score is within 1.0 of the real pick's (logits); a pick that far ahead of every expert in the bank is read for real. Bank hits go from about 78% to about 97%, and the bytes read per word fall by about seven-eighths.
- **Lean attention.** gpt-oss looks back only 128 words on every other layer; those layers keep only that window, and the shared keys are never copied out per head.

## The task, and what it wrote

This is the prompt every score on this page comes from. It asks for a piece of production code with several things that have to be right at once: a dependency graph with cycle checks, a concurrency limit, cancellation through `AbortSignal`, and failures that cascade without running the dependants.

> Implement a generic, cancellable asynchronous task execution pool in TypeScript with dependency resolution.
> 
> ### Requirements:
> 
> 1. **Task Definition & Dependencies:**
>    - Each task has an `id: string`, optional `dependencies: string[]` (task IDs that must successfully complete first), and an `execute: (signal: AbortSignal) => Promise<T>`.
>    - Before running any task, validate the dependency graph. Throw an `Error` if circular dependencies or references to non-existent tasks exist.
> 
> 2. **Concurrency & Execution:**
>    - Accept a `concurrency: number` limit. At no point should more than `concurrency` tasks run in parallel.
>    - Tasks must run as soon as a concurrency slot is available and all their dependencies have completed with `fulfilled` status.
> 
> 3. **Cancellation & Cascade Failures:**
>    - Provide a `cancel(reason?: string)` method that aborts all actively executing tasks via their `AbortSignal` and cancels all pending tasks.
>    - If any task fails (`rejected`) or is aborted (`cancelled`), any downstream dependent tasks must automatically be marked as `cancelled` without calling their `execute` function.
> 
> 4. **Return Value:**
>    - `run()` must resolve to a `Map<string, TaskResult<unknown>>`, where:
>      ```typescript
>      type TaskStatus = 'fulfilled' | 'rejected' | 'cancelled';
> 
>      interface TaskResult<T unknown> {
>        id: string;
>        status: TaskStatus;
>        value?: T;
>        error?: unknown;
>      }
>      ```

**What gpt-oss-120b wrote in fast mode, at 9.8 tok/s:** [gpt-oss-task-pool.ts](gpt-oss-task-pool.ts), 338 lines, taken from one of the three runs. The whole answer was 7,319 words, thinking included, and took 747 s on the Mac mini. That answer passed all 20 hidden checks; the other two runs passed 15 and 19.

The 20 checks the code is run against (4 to a point), none of which the model sees:

1. missing dependency throws
2. cycle throws
3. chain runs in order with values
4. no task runs twice
5. concurrency limit holds
6. diamond completes
7. failure cancels dependants without running them
8. rejected keeps its error
9. cancel aborts running and skips pending
10. every id in the result map
11. cancel before run: nothing runs, all cancelled
12. cancel with a task that ignores its signal: run still resolves, task cancelled
13. a synchronous throw is rejected and the pool carries on
14. an empty task list resolves to an empty map
15. a bad graph runs nothing
16. a deep cascade cancels every level, the independent branch finishes
17. 30 tasks at concurrency 3 use every slot and never more
18. a diamond at concurrency 1 runs in a valid order
19. cancel twice and after the end is harmless
20. one late failure cancels a task that waits on two

## Its chat format

gpt-oss speaks OpenAI's harmony format, with an analysis channel for its thinking and a final channel for the answer. The chat page shows the thinking in a box above the answer; the OpenAI-compatible API returns only the final answer.

## Not tested

- Fast mode on tasks other than the one above.
- Chats past 16K tokens, and Macs other than an M4 with 16 GB.
