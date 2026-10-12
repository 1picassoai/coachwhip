# gpt-oss-120b on a 16 GB Mac

OpenAI's gpt-oss-120b has 117 billion parameters: 36 layers, 128 experts in each, 4 of them used for every word. It thinks before it answers.

On a Mac mini M4 with 16 GB, Coachwhip runs it two ways:

| Mode | Speed | What you get |
|---|---|---|
| Exact (`--parallel --flash-reads`) | 4.2 tok/s | every expert the model asks for is read |
| Fast (`--stand-ins` added) | 9.6 to 10.5 tok/s | light experts not in memory are stood in for by ones that are |

Setup is on the models page under [gpt-oss-120b](MODELS.md#gpt-oss-120b): download the official MXFP4 file, squeeze it once with `coachwhip-prepare`, start the chat.

## What to expect

**It thinks first.** The chat page shows the thinking in a box above the answer. On everyday coding tasks the thinking is 50 to 300 words. On hard tasks, a job queue with cancellation and retries, a combinatorics proof, it thinks for 3,000 to 5,000 words before the first line of the answer. At 10 tok/s that is five to eight minutes of thinking.

**The word cap counts the thinking.** `--max-tokens` is the total for thinking and answer together. The README starts the chat at 8,000, which covered every task on this page. A lower cap on a hard task can end the run inside the thinking, with no answer.

**Everyday code in fast mode.** Working code at 8 to 10 tok/s. One example, measured on the release build: an arithmetic expression parser, asked for with a tokenizer, precedence climbing, right-associative `^`, unary minus, error positions and no `eval`, came back as 2,744 words at 8.3 tok/s. Run against 11 cases from its own requirements it passed 10. The one it missed, `-2^2` giving 4 instead of -4, the original file with every expert read missed in exactly the same way. On harder tasks it can get a detail like that wrong, in either mode. Read what it writes, and run it.

**The squeeze and fast mode, checked.** The same prompt in four configurations: original file exact, 3.5 of 5 by reading; squeezed file fast mode on the release build, 10 of 11 cases; the task pool, 5.0 original against 4.8 squeezed exact and 4.5 fast. Neither the squeeze nor fast mode has turned working code into broken code in any measured run of the release build.

**Hard tasks.** A rate-limiter specification with asynchronous waiting scored 1.5 of 5 in fast mode and 1.1 with every expert read. The loss there is the model's reasoning on the problem, not the mode.

## A hard problem, and how it thinks

Running a model this close is also a way of measuring the model. The numbers above say how fast it writes; this section says what it does with a problem it cannot do in one pass. Two cases, both in fast mode at 8 to 10 tok/s.

**A counting problem.** Count the lines through at least three points of a 20 by 20 grid, and the sum over those lines of the number of ways to choose three of their points. The second sum is the number of collinear triples, and the exact answers are 6,778 lines and 104,676.

At its default reasoning level the model thought for 4,000 words and was still thinking when the run was capped. The thinking was on the right road: inside the first 300 words it saw that the second sum counts collinear triples, then set up the count by primitive direction vectors, and by the end had the right formula for the triples in one direction, (b minus 1) times (20 minus b dx) times (20 minus b dy) summed over the step b. It never got to add the directions up. The cap, not the method, ended it.

At the lowest reasoning level it stopped thinking early and wrote an answer within the same cap. The answer explained the method correctly for the first part, printed a garbled box where the number should have been, and gave a wrong formula for the second part. Less thinking got an answer out in four minutes, and the answer was wrong.

**A job queue with cancellation.** A TypeScript class with a concurrency limit, three priority levels, cancellation before and during a job through `AbortSignal`, retries with backoff that must not hold a slot, and a drain that resolves when idle. The model thought for 3,500 words and was capped before writing code. The plan it had written was complete and sound: three queues, a flag to settle the race between a job finishing and its signal aborting, the slot released before the backoff timer starts. On the same task at a cap of 7,000 words the code would follow the plan; at 10 tok/s that is twelve minutes.

What this says about the model: on a problem with several interacting rules it plans in full before it writes, and the plan is usually right. The cost is time, thousands of words of it, and a word cap that ends the run inside the plan leaves nothing. Give it the cap and the minutes, and it does the work. Take the thinking away, and it writes something that looks like an answer.

## Speed and score on the task-pool prompt

The task is one prompt: a generic cancellable asynchronous task pool in TypeScript with dependency resolution. The model thinks, then writes the code, about 4,000 words in all. The code is run against 20 hidden checks, 4 checks to a point. A second wording of the same job, a build-pipeline runner, checks that a score is not a fluke.

All rows: Mac mini M4, 16 GB, macOS 26, the squeezed file (every expert matrix at Q2_K), `--parallel --flash-reads`, temperature 0.

| Setup | Speed | Score | Second wording |
|---|---|---|---|
| 4 experts, all read | 4.2 tok/s | 4.8 | |
| 3 experts, all read | 5.6 tok/s | 4.5 | |
| 4 experts, the first 2 read, stand-ins for the rest | 7.8 tok/s | 4.5 | 3.5 at 7.9 tok/s |
| 4 experts, stand-ins by weight, no gap | 9.8 tok/s | 4.0 | 4.0 at 10.0 tok/s |
| **4 experts, stand-ins by weight with the gap (`--stand-ins` as shipped), three runs** | **9.8, 10.5, 9.6 tok/s** | **5.0, 3.8, 4.8** | |
| 4 experts, only the first read | 9.6 tok/s | 4.5 | 0.8 at 10.2 tok/s |
| 2 experts | 7.1 tok/s | 0, the code does not compile | |

`--stand-ins` as shipped is the setup near 10 tok/s whose every run compiled and scored 3.8 or more. Without the gap, one run in three broke its file on a one-character slip. Reading fewer experts costs more than standing in for the light ones: the model was trained to use 4.

## What makes it fast

On a 16 GB Mac the speed is set by one number: how many bytes each word reads from the SSD. Everything here cuts those bytes or hides them.

- **The squeeze.** `coachwhip-prepare` takes each expert from 13.2 MB (MXFP4) to 8.7 MB (Q2_K, rows padded to whole 256-weight blocks), so the expert bank on the GPU holds 25 experts a layer instead of 15. On the task-pool prompt the squeezed file scores 4.8 against 5.0 for the original download, which runs at 2.2 tok/s.
- **The expert bank.** Recently used experts stay on the GPU and are reused. With every expert read, about 4 picks in 5 are already there.
- **The early guess.** While the GPU works on one layer, Coachwhip runs the router of a later layer on the same input and reads the experts it will want. For gpt-oss: 2 guesses a layer, 2 layers ahead; with `--stand-ins`, 1 guess a layer, since most picks are already in the bank.
- **Flash reads (`--flash-reads`).** Experts are read straight from the SSD, past the page cache.
- **Stand-ins (`--stand-ins`).** A word's first pick is always read, and so is the second whenever the router gives it at least 0.3 of the weight. A lighter pick that is not in the bank is replaced by the best expert that is, by the router's own scores, and the weights are recomputed over the experts used. The substitute is used only when its router score is within 1.0 of the real pick's; a pick further ahead than that is read. Bank hits go from about 78% to about 97%, and the bytes read per word fall by about seven-eighths.
- **Lean attention.** gpt-oss looks back only 128 words on every other layer; those layers keep only that window, and the shared keys are never copied out per head.

## The task-pool prompt, and what it wrote

This is the prompt every score in the table above comes from. It asks for production code with several things that have to be right at once: a dependency graph with cycle checks, a concurrency limit, cancellation through `AbortSignal`, and failures that cascade without running the dependants.

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

**What gpt-oss-120b wrote in fast mode at 9.8 tok/s:** [gpt-oss-task-pool.ts](gpt-oss-task-pool.ts), 338 lines, from one of the three runs. The whole answer was 7,319 words, thinking included, and took 747 s. It passed all 20 hidden checks; the other two runs passed 15 and 19.

The 20 checks, none of which the model sees:

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

gpt-oss speaks OpenAI's harmony format: an analysis channel for its thinking, a final channel for the answer. The chat page shows both. The OpenAI-compatible API returns the final answer only.

## Not tested

- Chats past 16K tokens.
- Macs other than an M4 with 16 GB.
