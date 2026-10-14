"""Coachwhip Crew — four CrewAI agents (Analyst -> Architect -> Developer -> Tester) build and review a piece
of code through an OpenAI-compatible door, shown as a pipeline of robots on one screen, every word streaming.
Then the code is RUN (a runner, no AI) and SCORED by a judge: seven executed checks, the same file for every run.

  python crew_app.py            then open http://127.0.0.1:8072/

Doors:
  local  (default)  COACHWHIP_BASE=http://127.0.0.1:8090/v1   — a Coachwhip door; bytes to the internet: 0.
  cloud             CREW_CLOUD=1 CREW_CLOUD_BASE=https://api.openai.com/v1 CREW_CLOUD_MODEL=<name>
                    CREW_CLOUD_KEY_FILE=<path to a file holding the key>  — the key is read inside, never printed.
                    CREW_PRICE_IN / CREW_PRICE_OUT: $ per million tokens, for the cost chip.
Task: CREW_TASK_FILE (default task_lru.txt). Judge: judge_lru.ts (expects `export class LruCache` in the code).
Nothing leaves this machine on the local door: the page listens on 127.0.0.1 only.
"""
import json, os, re, sys, time, threading, queue, subprocess, tempfile, shutil
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

from crewai import Agent, Task, Crew, Process, LLM
from crewai.events import BaseEventListener
from crewai.events.types.llm_events import LLMStreamChunkEvent, LLMCallStartedEvent, LLMCallCompletedEvent
from crewai.events.types.agent_events import AgentExecutionStartedEvent, AgentExecutionCompletedEvent

HERE = os.path.dirname(os.path.abspath(__file__))
CLOUD = os.environ.get("CREW_CLOUD") == "1"
BASE = os.environ.get("CREW_CLOUD_BASE", "https://api.openai.com/v1") if CLOUD else os.environ.get("COACHWHIP_BASE", "http://127.0.0.1:8090/v1")
PORT = int(os.environ.get("CREW_PORT", "8072"))
MAX_TOKENS = int(os.environ.get("CREW_MAX_TOKENS", "3000"))
PRICE_IN = float(os.environ.get("CREW_PRICE_IN", "0"))
PRICE_OUT = float(os.environ.get("CREW_PRICE_OUT", "0"))
TASK_FILE = os.environ.get("CREW_TASK_FILE", os.path.join(HERE, "task_lru.txt"))
JUDGE_FILE = os.environ.get("CREW_JUDGE_FILE", os.path.join(HERE, "judge_lru.ts"))
HOST = urlparse(BASE).hostname or ""
LOCAL = HOST in ("127.0.0.1", "localhost")

events: "queue.Queue[dict]" = queue.Queue()
state = {"running": False, "model": os.environ.get("CREW_CLOUD_MODEL") if CLOUD else os.environ.get("COACHWHIP_MODEL"),
         "seat": None, "started": {}, "chars": {}, "tps": {}, "first": {}, "call_t0": None, "tok_in": 0, "tok_out": 0}

SEATS = {"Business Analyst": "ba", "Software Architect": "arch", "Senior Developer": "dev", "Strict Tester": "tester"}


def emit(**kw):
    kw["t"] = time.time()
    events.put(kw)


def api_key():
    if not CLOUD:
        return "anything"
    path = os.environ.get("CREW_CLOUD_KEY_FILE", "")
    return open(path, encoding="utf-8").read().strip()


def model_name():
    if state["model"]:
        return state["model"]
    import urllib.request
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never a system proxy for a loopback door
    req = urllib.request.Request(BASE.rstrip("/") + "/models", headers={"Host": "127.0.0.1:8090"})
    with opener.open(req, timeout=10) as r:
        data = json.load(r)
    state["model"] = data["data"][0]["id"]
    return state["model"]


def task_text():
    return open(TASK_FILE, encoding="utf-8").read().strip()


class Streamer(BaseEventListener):
    def setup_listeners(self, bus):
        @bus.on(AgentExecutionStartedEvent)
        def started(source, event):
            role = getattr(event.agent, "role", None) if hasattr(event, "agent") else None
            seat = SEATS.get(role)
            if seat:
                state["seat"] = seat
                state["started"][seat] = time.time()
                state["chars"][seat] = 0
                state["tps"].pop(seat, None)
                state["first"].pop(seat, None)
                emit(kind="start", seat=seat)

        @bus.on(LLMStreamChunkEvent)
        def chunk(source, event):
            seat = SEATS.get(getattr(event, "agent_role", None)) or state["seat"]
            if seat and event.chunk:
                if state["chars"].get(seat, 0) == 0:
                    state["first"][seat] = time.time()  # the first word: everything before it was reading the prompt
                state["chars"][seat] = state["chars"].get(seat, 0) + len(event.chunk)
                emit(kind="chunk", seat=seat, text=event.chunk)

        @bus.on(LLMCallStartedEvent)
        def call_started(source, event):
            seat = SEATS.get(getattr(event, "agent_role", None)) or state["seat"]
            if seat:
                state["call_t0"] = (seat, time.time())

        @bus.on(LLMCallCompletedEvent)
        def call_done(source, event):
            # Measured, not estimated: completion tokens reported for the call over the call's wall time.
            seat = SEATS.get(getattr(event, "agent_role", None)) or state["seat"]
            t0 = state.get("call_t0")
            usage = getattr(event, "usage", None) or {}
            if not isinstance(usage, dict):
                usage = {}
            toks = usage.get("completion_tokens")
            if seat and t0 and t0[0] == seat and toks:
                now = time.time()
                first = state["first"].get(seat) or t0[1]
                state["tps"][seat] = (toks, now - t0[1], usage.get("prompt_tokens") or 0, first - t0[1], now - first)
            state["tok_in"] += usage.get("prompt_tokens") or 0
            state["tok_out"] += toks or 0
            if not LOCAL:
                # Rough bytes sent: prompt tokens ~ 4 bytes each; the real wire count is higher, never lower.
                emit(kind="bytes", bytes=state["tok_in"] * 4)
                emit(kind="cost", usd=(state["tok_in"] * PRICE_IN + state["tok_out"] * PRICE_OUT) / 1e6)

        @bus.on(AgentExecutionCompletedEvent)
        def done(source, event):
            role = getattr(event.agent, "role", None) if hasattr(event, "agent") else None
            seat = SEATS.get(role)
            if seat:
                secs = time.time() - state["started"].get(seat, time.time())
                toks, csecs, ptoks, read_s, write_s = state["tps"].get(seat, (None, None, 0, 0, 0))
                tps = round(toks / write_s, 1) if toks and write_s and write_s > 0.5 else None
                rps = round(ptoks / read_s, 1) if ptoks and read_s and read_s > 0.2 else None
                emit(kind="done", seat=seat, secs=round(secs, 1), tokens=toks, tps=tps, ptokens=ptoks, rps=rps, output=str(getattr(event, "output", "")))


def extract_code(text):
    m = re.search(r"```(\w+)?\s*\n(.*?)```", text, re.S)
    if not m:
        return None, None
    return (m.group(1) or "").lower(), m.group(2)


def run_and_judge(lang, code):
    """Runner: execute the code's own usage example. Judge: the seven checks. Returns (ok, out, pass, total, lines)."""
    d = tempfile.mkdtemp(prefix="crew-run-")
    try:
        if lang not in ("ts", "typescript"):
            return None, f"no runner for language {lang!r}", 0, 8, "judge expects TypeScript"
        path = os.path.join(d, "main.ts")
        open(path, "w", encoding="utf-8").write(code)
        p = subprocess.run(["npx", "--yes", "tsx", path], capture_output=True, text=True, timeout=120, shell=(os.name == "nt"))
        run_out = ((p.stdout + p.stderr).strip() or "(no output)") + f"\n\nexit code {p.returncode}"
        shutil.copy(JUDGE_FILE, os.path.join(d, "judge.ts"))
        j = subprocess.run(["npx", "--yes", "tsx", os.path.join(d, "judge.ts")], capture_output=True, text=True, timeout=120, shell=(os.name == "nt"))
        jout = (j.stdout + j.stderr).strip()
        m = re.search(r"SCORE (\d+)/(\d+)", jout)
        if m:
            passed, total = int(m.group(1)), int(m.group(2))
        else:
            passed, total = 0, 8
            jout = "judge could not run the code:\n" + jout[-1200:]
        lines = "\n".join(l for l in jout.splitlines() if l.startswith(("PASS", "FAIL"))) or jout[-1200:]
        return p.returncode == 0, run_out, passed, total, lines
    except subprocess.TimeoutExpired:
        return False, "timed out after 120 s", 0, 8, "timed out"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def run_crew():
    state["running"] = True
    state["tok_in"] = state["tok_out"] = 0
    try:
        task = task_text()
        model = model_name()
        emit(kind="model", model=model, where=("local door, port " + str(urlparse(BASE).port)) if LOCAL else ("cloud " + HOST), local=LOCAL, host=HOST, task=task)
        headers = {"Host": "127.0.0.1:8090"} if LOCAL else None
        limit = {"max_completion_tokens": MAX_TOKENS} if CLOUD else {"max_tokens": MAX_TOKENS}
        temp = {} if CLOUD else {"temperature": 0}  # newer cloud models reject temperature and the old limit name
        llm = LLM(model=f"openai/{model}", base_url=BASE, api_key=api_key(), stream=True, extra_headers=headers, **limit, **temp)
        ba = Agent(role="Business Analyst", goal="Turn the task into precise, testable requirements.",
                   backstory="You write short numbered requirement lists (8 at most) that name the language and the public API exactly. No code, no preamble.",
                   llm=llm, verbose=False, allow_delegation=False)
        arch = Agent(role="Software Architect", goal="Decide the design before any code is written.",
                     backstory="You read the requirements and write a short design: the data structure, the invariants, the complexity of each operation, and the two or three traps to avoid. Under 200 words, no code.",
                     llm=llm, verbose=False, allow_delegation=False)
        dev = Agent(role="Senior Developer", goal="Implement the design exactly.",
                    backstory="You answer with one fenced ```typescript code block only: `export class LruCache<K, V>` with the constructor (capacity, ttlMs), then a short runnable usage example at the end that prints its results. No dependencies. No prose outside the block.",
                    llm=llm, verbose=False, allow_delegation=False)
        tester = Agent(role="Strict Tester", goal="Find every requirement the code fails.",
                       backstory="You read the task, the requirements, the design and the code. You list each failed requirement on its own line with the line of code at fault, most serious first. If nothing fails you reply exactly: ALL REQUIREMENTS MET. You never rewrite the code. Be brief.",
                       llm=llm, verbose=False, allow_delegation=False)
        t1 = Task(description=f"Task from the user:\n{task}\n\nWrite the requirements.", expected_output="A numbered list of at most 8 testable requirements.", agent=ba)
        t2 = Task(description="Write the design for the requirements.", expected_output="A short design note: structure, invariants, complexity, traps.", agent=arch, context=[t1])
        t3 = Task(description="Implement the design and the requirements, exactly. Export the class.", expected_output="One fenced typescript block: `export class LruCache<K, V>` and a runnable usage example.", agent=dev, context=[t1, t2])
        t4 = Task(description=f"The original task:\n{task}\n\nCheck the developer's code against the requirements and the design.", expected_output="Either the failed requirements, one per line, or exactly: ALL REQUIREMENTS MET.", agent=tester, context=[t1, t2, t3])
        crew = Crew(agents=[ba, arch, dev, tester], tasks=[t1, t2, t3, t4], process=Process.sequential, verbose=False)
        result = crew.kickoff()
        dev_out = str(t3.output.raw) if t3.output else ""
        verdict = str(t4.output.raw) if t4.output else str(result)
        lang, code = extract_code(dev_out)
        if code:
            emit(kind="run_start", lang=lang)
            ok, out, passed, total, lines = run_and_judge(lang, code)
            emit(kind="run_done", ok=ok, output=out)
            emit(kind="judge", **{"pass": passed, "total": total, "lines": lines})
        else:
            emit(kind="run_done", ok=None, output="no code block found in the developer's answer")
            emit(kind="judge", **{"pass": 0, "total": 8, "lines": "no code to judge"})
        head = re.sub(r"[^A-Z ]", "", verdict.strip().upper()[:40])
        emit(kind="verdict", text=verdict.strip(), met=head.startswith("ALL REQUIREMENTS MET"))
    except Exception as e:
        emit(kind="error", text=f"{type(e).__name__}: {e}")
    finally:
        state["running"] = False
        emit(kind="end")


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            body = open(os.path.join(HERE, "crew_app.html"), "rb").read()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        elif self.path == "/task":
            self.send_json({"task": task_text()})
        elif self.path == "/events":
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-cache"); self.end_headers()
            try:
                if state["model"]:
                    hello = {"kind": "model", "model": state["model"], "where": ("local door, port " + str(urlparse(BASE).port)) if LOCAL else ("cloud " + HOST), "local": LOCAL, "host": HOST}
                    self.wfile.write(("data: " + json.dumps(hello) + "\n\n").encode()); self.wfile.flush()
                while True:
                    try:
                        ev = events.get(timeout=15)
                        self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode()); self.wfile.flush()
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n"); self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        if self.path == "/run":
            if state["running"]:
                self.send_response(409); self.end_headers(); return
            threading.Thread(target=run_crew, daemon=True).start()
            self.send_response(202); self.end_headers()
        else:
            self.send_response(404); self.end_headers()


if __name__ == "__main__":
    Streamer()
    print(f"Coachwhip Crew on http://127.0.0.1:{PORT}/  ({'CLOUD ' + HOST if CLOUD else 'local door ' + BASE})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
