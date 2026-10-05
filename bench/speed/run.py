"""Speed check: five coding prompts through the running Coachwhip, each as a fresh conversation,
and the writing speed of each answer.

Start Coachwhip with --temperature 0 --chat 8090 first, then:  python3 bench/speed/run.py
Standard library only. Writing speed = answer tokens / seconds from the first to the last token,
the same thing the chat page shows under each answer.
"""
import json, os, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
URL = os.environ.get("COACHWHIP_URL", "http://127.0.0.1:8090/v1/chat/completions")
MAX_TOKENS = 300


def stream(prompt):
    body = json.dumps({"model": "m", "stream": True, "max_tokens": MAX_TOKENS,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    sent = time.time()
    first = last = None
    tokens = 0
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if "usage" in event and event["usage"]:
                tokens = event["usage"].get("completion_tokens", tokens)
            delta = event["choices"][0]["delta"].get("content")
            if delta:
                now = time.time()
                first = first or now
                last = now
    return sent, first, last, tokens


def main():
    prompts = json.load(open(os.path.join(HERE, "prompts.json")))
    speeds = []
    print(f"{'prompt':12s} {'tokens':>6s} {'first word':>10s} {'writing':>9s}")
    for p in prompts:
        sent, first, last, tokens = stream(p["prompt"])
        if not first or tokens < 2 or last == first:
            print(f"{p['id']:12s} no answer")
            continue
        tps = (tokens - 1) / (last - first)
        speeds.append(tps)
        print(f"{p['id']:12s} {tokens:6d} {first - sent:9.1f}s {tps:7.2f} tok/s")
    if speeds:
        print(f"writing: mean {sum(speeds) / len(speeds):.2f} tok/s, min {min(speeds):.2f}, max {max(speeds):.2f}, over {len(speeds)} prompts")


if __name__ == "__main__":
    main()
