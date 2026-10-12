/* -------------------------------------------------------------
   Types
   ------------------------------------------------------------- */
type TaskStatus = 'fulfilled' | 'rejected' | 'cancelled';

interface TaskResult<T = unknown> {
  id: string;
  status: TaskStatus;
  /** value is present only when status === 'fulfilled' */
  value?: T;
  /** error is present only when status === 'rejected' or 'cancelled' */
  error?: unknown;
}

/** A single unit of work. */
interface Task<T = unknown> {
  /** Unique identifier */
  id: string;
  /** Ids of tasks that must finish successfully before this one starts */
  dependencies?: string[];
  /** Called with an AbortSignal that is aborted when the pool is cancelled */
  execute: (signal: AbortSignal) => Promise<T>;
}

/* -------------------------------------------------------------
   Semaphore – limits the number of concurrently running tasks
   ------------------------------------------------------------- */
class Semaphore {
  private readonly max: number;
  private current = 0;
  private readonly waiters: (() => void)[] = [];

  constructor(max: number) {
    if (max < 1) throw new Error('Concurrency must be >= 1');
    this.max = max;
  }

  async acquire(): Promise<void> {
    if (this.current < this.max) {
      this.current++;
      return;
    }
    return new Promise<void>(resolve => {
      this.waiters.push(() => {
        this.current++;
        resolve();
      });
    });
  }

  release(): void {
    this.current--;
    if (this.waiters.length) {
      const next = this.waiters.shift()!;
      next();
    }
  }
}

/* -------------------------------------------------------------
   TaskPool – core implementation
   ------------------------------------------------------------- */
export class TaskPool {
  /** All tasks indexed by their id */
  private readonly tasks = new Map<string, Task>();
  /** Map of task id → list of dependent task ids */
  private readonly dependents = new Map<string, string[]>();
  /** Remaining dependencies count for each task */
  private readonly remaining = new Map<string, number>();
  /** Results produced by the pool */
  private readonly results = new Map<string, TaskResult>();
  /** AbortController for every *running* task */
  private readonly controllers = new Map<string, AbortController>();
  /** Set of tasks that have already been started */
  private readonly started = new Set<string>();
  /** Whether the whole pool has been cancelled */
  private cancelled = false;
  /** Optional cancellation reason */
  private cancelReason?: string;
  /** Concurrency limiter */
  private readonly semaphore: Semaphore;

  /**
   * @param tasks       List of tasks to run.
   * @param concurrency Maximum number of tasks that may run in parallel.
   */
  constructor(tasks: Task[], private readonly concurrency: number) {
    this.semaphore = new Semaphore(concurrency);
    // store tasks by id
    for (const t of tasks) {
      if (this.tasks.has(t.id)) {
        throw new Error(`Duplicate task id "${t.id}"`);
      }
      this.tasks.set(t.id, t);
    }
    this.validateGraph();
    this.buildDependencyMaps();
    this.initRemainingCounts();
  }

  /* ---------------------------------------------------------
     Public API
     --------------------------------------------------------- */
  /** Cancel the whole pool – aborts all running tasks and marks the rest as cancelled. */
  cancel(reason?: string): void {
    if (this.cancelled) return;
    this.cancelled = true;
    this.cancelReason = reason;

    // abort currently running tasks
    for (const ctrl of this.controllers.values()) {
      ctrl.abort();
    }

    // mark every task that has not yet produced a result as cancelled
    for (const id of this.tasks.keys()) {
      if (!this.results.has(id)) {
        this.results.set(id, { id, status: 'cancelled' });
      }
    }
  }

  /** Run the whole pool. Returns a map from task id → its final result. */
  async run(): Promise<Map<string, TaskResult>> {
    // If the pool was cancelled before run() was called, we are done.
    if (this.cancelled) return this.results;

    // Kick off all tasks that have no dependencies.
    for (const [id, remaining] of this.remaining.entries()) {
      if (remaining === 0) this.schedule(id);
    }

    // Wait until every task has a result (fulfilled / rejected / cancelled).
    while (this.results.size < this.tasks.size) {
      // The loop yields control to the event loop; the actual work happens
      // inside the async `runTask` method.
      await new Promise(resolve => setTimeout(resolve, 0));
    }
    return this.results;
  }

  /* ---------------------------------------------------------
     Internals
     --------------------------------------------------------- */
  /** Validate the dependency graph – throws on missing or circular deps. */
  private validateGraph(): void {
    // 1️⃣  Missing dependencies
    for (const task of this.tasks.values()) {
      for (const dep of task.dependencies ?? []) {
        if (!this.tasks.has(dep)) {
          throw new Error(
            `Task "${task.id}" depends on unknown task "${dep}"`,
          );
        }
      }
    }

    // 2️⃣  Detect cycles (DFS)
    const visited = new Set<string>();
    const stack = new Set<string>();

    const dfs = (id: string): boolean => {
      if (!visited.has(id)) {
        visited.add(id);
        stack.add(id);
        for (const dep of this.tasks.get(id)!.dependencies ?? []) {
          if (!visited.has(dep) && dfs(dep)) return true;
          if (stack.has(dep)) return true; // back‑edge → cycle
        }
      }
      stack.delete(id);
      return false;
    };

    for (const id of this.tasks.keys()) {
      if (dfs(id)) {
        throw new Error('Circular dependency detected');
      }
    }
  }

  /** Build the "dependents" adjacency list. */
  private buildDependencyMaps(): void {
    // initialise empty list for every task
    for (const id of this.tasks.keys()) {
      this.dependents.set(id, []);
    }
    // fill it
    for (const [id, task] of this.tasks.entries()) {
      for (const dep of task.dependencies ?? []) {
        this.dependents.get(dep)!.push(id);
      }
    }
  }

  /** Initialise the remaining‑dependency counter for each task. */
  private initRemainingCounts(): void {
    for (const [id, task] of this.tasks.entries()) {
      this.remaining.set(id, (task.dependencies ?? []).length);
    }
  }

  /** Schedule a task for execution (if it hasn't been started yet). */
  private schedule(id: string): void {
    if (this.started.has(id) || this.results.has(id)) return;
    this.started.add(id);
    this.runTask(id).catch(() => {
      // Errors are already recorded inside runTask – we just swallow them
    });
  }

  /** Core runner for a single task. */
  private async runTask(id: string): Promise<void> {
    // If the whole pool has been cancelled, we simply mark this task as cancelled.
    if (this.cancelled) {
      this.results.set(id, { id, status: 'cancelled' });
      return;
    }

    // Acquire a concurrency slot.
    await this.semaphore.acquire();

    // AbortSignal for this task.
    const controller = new AbortController();
    this.controllers.set(id, controller);

    const task = this.tasks.get(id)!;

    try {
      const value = await task.execute(controller.signal);
      this.results.set(id, { id, status: 'fulfilled', value });
      this.propagateSuccess(id);
    } catch (err) {
      // Distinguish abort vs real failure
      if (controller.signal.aborted) {
        this.results.set(id, { id, status: 'cancelled', error: err });
      } else {
        this.results.set(id, { id, status: 'rejected', error: err });
      }
      this.propagateFailureOrCancel(id);
    } finally {
      // Clean‑up
      this.controllers.delete(id);
      this.semaphore.release();
    }
  }

  /** After a task succeeds – schedule its dependents if they become ready. */
  private propagateSuccess(completedId: string): void {
    for (const dependentId of this.dependents.get(completedId) ?? []) {
      // Only consider tasks that are still pending.
      if (this.results.has(dependentId)) continue;

      const remaining = this.remaining.get(dependentId)! - 1;
      this.remaining.set(dependentId, remaining);
      if (remaining === 0) this.schedule(dependentId);
    }
  }

  /** After a task fails or is cancelled – cascade cancellation to all downstream tasks. */
  private propagateFailureOrCancel(failedId: string): void {
    for (const dependentId of this.dependents.get(failedId) ?? []) {
      // If the dependent already has a result we do nothing.
      if (this.results.has(dependentId)) continue;

      // Mark the dependent as cancelled.
      this.results.set(dependentId, { id: dependentId, status: 'cancelled' });

      // Recursively propagate further down the graph.
      this.propagateFailureOrCancel(dependentId);
    }
  }
}

/* -------------------------------------------------------------
   Example usage
   ------------------------------------------------------------- */
async function demo() {
  const tasks: Task<number>[] = [
    {
      id: 'a',
      execute: async signal => {
        await delay(1000, signal);
        return 1;
      },
    },
    {
      id: 'b',
      dependencies: ['a'],
      execute: async signal => {
        await delay(500, signal);
        return 2;
      },
    },
    {
      id: 'c',
      dependencies: ['a'],
      execute: async signal => {
        await delay(800, signal);
        return 3;
      },
    },
    {
      id: 'd',
      dependencies: ['b', 'c'],
      execute: async signal => {
        await delay(300, signal);
        return 4;
      },
    },
  ];

  const pool = new TaskPool(tasks, 2); // at most 2 tasks in parallel

  // Cancel after 2 seconds (just to illustrate cancellation)
  setTimeout(() => pool.cancel('demo timeout'), 2000);

  const results = await pool.run();
  console.log('Results:');
  for (const [id, r] of results.entries()) {
    console.log(`${id}:`, r);
  }
}

/** Helper – resolves after ms unless aborted */
function delay(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, ms);
    signal.addEventListener('abort', () => {
      clearTimeout(timer);
      reject(new Error('aborted'));
    });
  });
}

// Uncomment to run the demo
// demo();
export { Semaphore };
