"""Quality check: ask the running Coachwhip the 8 tasks in tasks.json, run each answer against tests.py.

Start Coachwhip with --temperature 0 --chat 8090 first, then:  python3 bench/quality/run.py
Standard library only. Answers and their code go to ./coachwhip-quality/ in the current folder.
"""
import json, os, re, subprocess, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath("coachwhip-quality")
URL = os.environ.get("COACHWHIP_URL", "http://127.0.0.1:8090/v1/chat/completions")

RUNNER = r'''
import sys
sys.path.insert(0, sys.argv[1])
import tests
ns = {}
exec(open(sys.argv[2], encoding="utf-8").read(), ns)
getattr(tests, "t_" + sys.argv[3])(ns)
print("PASS")
'''


def ask(prompt):
    body = json.dumps({"model": "m", "max_tokens": 1200, "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))["choices"][0]["message"]["content"]


def code_of(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return "\n\n".join(blocks) if blocks else text


def main():
    tasks = json.load(open(os.path.join(HERE, "tasks.json")))
    os.makedirs(OUT, exist_ok=True)
    answers, passed = {}, 0
    for t in tasks:
        answers[t["id"]] = ask(t["prompt"])
        path = os.path.join(OUT, f"answer_{t['id']}.py")
        open(path, "w", encoding="utf-8").write(code_of(answers[t["id"]]))
        try:
            p = subprocess.run([sys.executable, "-c", RUNNER, HERE, path, t["id"]], capture_output=True, text=True, timeout=10)
            ok = "PASS" in p.stdout
            why = "" if ok else (p.stderr.strip().splitlines() or ["?"])[-1][:100]
        except subprocess.TimeoutExpired:
            ok, why = False, "timeout"
        passed += ok
        print(f"{t['id']:12s} {'pass' if ok else 'FAIL'}  {why}", flush=True)
    json.dump(answers, open(os.path.join(OUT, "answers.json"), "w"), indent=1)
    print(f"total {passed} / {len(tasks)}")


if __name__ == "__main__":
    main()
