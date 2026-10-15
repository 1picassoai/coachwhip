// The judge: eight executed checks against the developer's LruCache. Same file for every run, no AI.
// The developer's code is written to ./main.ts and must export the class LruCache.
import { LruCache } from "./main";

let pass = 0, fail = 0;
function check(name: string, fn: () => boolean) {
  let ok = false;
  try { ok = fn(); } catch (e) { ok = false; }
  console.log((ok ? "PASS  " : "FAIL  ") + name);
  ok ? pass++ : fail++;
}
const sleep = (ms: number) => { const t = Date.now(); while (Date.now() - t < ms) {} };

check("capacity < 1 throws RangeError", () => { try { new LruCache<string, number>(0, 1000); return false; } catch (e) { return e instanceof RangeError; } });
check("get returns the value and marks it most recently used", () => {
  const c = new LruCache<string, number>(2, 10000); c.set("a", 1); c.set("b", 2); c.get("a"); c.set("c", 3);
  return c.get("a") === 1 && c.get("b") === undefined && c.get("c") === 3;
});
check("peek does not change recency", () => {
  const c = new LruCache<string, number>(2, 10000); c.set("a", 1); c.set("b", 2); c.peek("a"); c.set("c", 3);
  return c.get("a") === undefined && c.get("b") === 2;
});
check("has() is true for a stored undefined", () => { const c = new LruCache<string, number | undefined>(2, 10000); c.set("u", undefined); return c.has("u") === true; });
check("has() is false for a missing key", () => { const c = new LruCache<string, number>(2, 10000); return c.has("nope") === false; });
check("set on an existing key resets expiry and recency", () => {
  const c = new LruCache<string, number>(2, 60); c.set("a", 1); c.set("b", 2); sleep(40); c.set("a", 9); sleep(40);
  return c.get("a") === 9 && c.has("b") === false;
});
check("expired entry is gone, lazily, no timers", () => {
  const c = new LruCache<string, number>(2, 30); c.set("a", 1); sleep(50);
  return c.has("a") === false && c.get("a") === undefined;
});
check("get does NOT extend expiry", () => {
  const c = new LruCache<string, number>(2, 60); c.set("a", 1); sleep(40); c.get("a"); sleep(40);
  return c.has("a") === false;
});
console.log(`SCORE ${pass}/${pass + fail}`);
