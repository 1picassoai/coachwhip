# Coachwhip Crew

Four CrewAI agents (analyst, architect, developer, tester) work a coding task through Coachwhip's local OpenAI door, on one screen, every word streaming. A runner (no AI) executes the developer's code; a judge runs a fixed set of checks the agents never see.

- `crew_app.py` — the server and the crew. `python crew_app.py`, then open http://127.0.0.1:8072/ and press Run. Needs `crewai` (`pip install crewai`) and, for the runner and judge, Node with `npx tsx`.
- `crew_app.html` — the page.
- `task_lru.txt` — the task. Change it for your own work.
- `judge_lru.ts` — the judge: eight executed checks for the LRU task. Write your own for your task; it imports `./main`, the developer's code.
- `record_app.py` — records a run to MP4 at 1280x720 (needs `playwright` and its Chromium).

Env: `COACHWHIP_BASE` (default `http://127.0.0.1:8090/v1`), `CREW_PORT` (8072), `CREW_MAX_TOKENS` (3000). For a cloud door: `CREW_CLOUD=1`, `CREW_CLOUD_BASE`, `CREW_CLOUD_MODEL`, `CREW_CLOUD_KEY_FILE` (a file holding the key; never the key itself), `CREW_PRICE_IN` / `CREW_PRICE_OUT` in dollars per million tokens for the cost chip.

Nothing leaves your machine on the local door: the page listens on 127.0.0.1 only.
