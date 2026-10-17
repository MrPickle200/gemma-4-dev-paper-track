# AGENTS.md — How to work in this project

Research project: a **micro-model-native harness** for Gemma 4 E2B/E4B on SWE-bench Verified.
`IDEA.md` is the specification (hypothesis, 3 modules, ablation matrix, metrics). Read it before changing design.
`PROGRESS.md` is an append-only log of what was actually done — never put plans there.

## Hard rules

1. **Always run Python from `.venv`.** Never use the system interpreter.
   ```powershell
   .\.venv\Scripts\python.exe scripts\<script>.py     # or: .\.venv\Scripts\Activate.ps1
   uv pip install --python .\.venv\Scripts\python.exe <pkg>   # install into the venv
   ```
2. **Report at every step.** After each action, state what changed, what the output proved, and what is next.
3. Keep changes minimal and consistent with existing scripts; do not add frameworks or abstractions.
4. Never commit model weights, `data/`, `logs/`, `.venv/`, or `tools/downloads/`.

## Environment (this machine, verified)

| | |
|---|---|
| GPU | RTX 4050 Laptop, **6 GB VRAM** (driver 596.21, CUDA 13.2) |
| Host | 15.7 GB RAM, Windows, CUDA 13.0 toolkit |
| Python | 3.11.9 in `.venv` (created with `uv`) |
| Docker | 29.7.2 (needed for SWE-bench eval, Bước 5) |

**Constraint that drives every decision:** 6 GB VRAM cannot hold bf16 Gemma 4 (E2B = 10.25 GB,
E4B = 16 GB). The project therefore runs **quantised GGUF via llama.cpp**, not vLLM, and the
harness only ever talks to an **OpenAI-compatible HTTP endpoint** so the backend stays swappable.

## Layout

```
IDEA.md                 spec
AGENTS.md               this file
PROGRESS.md             append-only log of completed work
requirements.txt        venv deps
data/subset_50.json     the 50-task SWE-bench Verified Mini subset
data/subset_50.meta.json  reproducibility manifest (filter + seed)
harness/config.py       RunConfig (model, variant, limits, thinking) + log paths
harness/llm.py          OpenAI-compatible /v1/chat/completions client
harness/tools.py        tool schemas + implementations, metric counters
harness/agent.py        the agent loop and the per-turn JSONL trace
harness/runner.py       worktree setup, task loading, patch collection
scripts/filter_subset.py     rebuilds the subset
scripts/fetch_repos.py       clones the 4 repos and verifies every base_commit
scripts/setup_inference.ps1  downloads llama.cpp + GGUFs
scripts/serve.ps1            starts llama-server for E2B or E4B
scripts/smoke_test.py        backend acceptance test (endpoint/thinking/JSON/tools)
scripts/diag_toolcalls.py    probes the Gemma 4 tool-call -> JSON conversion
scripts/run_task.py          runs the agent on one or more tasks
scripts/summarize_run.py     reads one .jsonl run trace
scripts/summarize_batch.py   aggregates all runs into the ablation table
repos/<owner>__<name>   full clones (gitignored)
worktrees/<instance_id> per-task checkouts, created and removed per run (gitignored)
models/*.gguf           Gemma 4 weights (gitignored)
tools/llama.cpp/        llama-server.exe (gitignored)
logs/runs/*.jsonl       one trace per run; logs/patches/*.patch the model's diff
```

## Common commands

```powershell
# 1. rebuild the 50-task subset
.\.venv\Scripts\python.exe scripts\filter_subset.py --n 50 --seed 0

# 2. download binaries + weights (idempotent, ~9 GB total)
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_inference.ps1

# 3. serve Gemma 4 (background; OpenAI-compatible on :8080)
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\serve.ps1 -Model E2B -Ctx 32768
#   E4B also fits 6 GB at -GpuLayers 99 -Ctx 32768 (measured: ~3.5 GB VRAM in use).

# 4. verify the backend (endpoint, thinking control, constrained JSON, tool calls)
.\.venv\Scripts\python.exe scripts\smoke_test.py

# 5. Bước 2: clone repos, then run the agent loop
.\.venv\Scripts\python.exe scripts\fetch_repos.py
.\.venv\Scripts\python.exe scripts\run_task.py --list
.\.venv\Scripts\python.exe scripts\run_task.py --instance-id psf__requests-1724 --thinking on
.\.venv\Scripts\python.exe scripts\run_batch.py --model gemma-4-E2B --thinking on --workers 8 `
    --base-url https://<name>.trycloudflare.com --api-key <key>   # Colab GPU backend (see colab/)
.\.venv\Scripts\python.exe scripts\check_endpoint.py --base-url <url> --api-key <key>   # verify a backend

# 6. inspect the results
.\.venv\Scripts\python.exe scripts\summarize_run.py logs\runs\<run>.jsonl --show-tool apply_edit
.\.venv\Scripts\python.exe scripts\summarize_batch.py --csv logs\batch.csv
```

## Bước 2 harness facts

- **Variants** (IDEA.md §3 ablation matrix): `B0` is implemented. `M1`/`M2`/`M3` are accepted by
  `--variant` and recorded in the log, but their mechanisms are Bước 3 work.
- **Thinking mode must be explicit.** Measured `--thinking off` over 3 runs: **0 edits, 26 repeated
  identical calls**. With `--thinking on` over 4 runs: **1 run reached a clean edit and `finish`**.
  E2B frequently omits the comma between JSON arguments when thinking is off
  (`{"pattern=...\"\npath": ...}`), which the tool dispatcher rejects. Treat thinking as a variable
  of the experiment, not a default.
- `read_file` prints **no line numbers** by default (`--line-numbers` opts in). B0 is specified as
  `cat`, and line-numbered output makes E2B copy the `314: ` prefixes into `search_block`.
  Do not turn this on for baseline runs.
- Counters in every `run_end` record, feeding IDEA.md §4: `edits_attempted`, `edits_malformed`
  (metric #2), `malformed_arguments`, `repeated_identical_calls` (metric #4), `peak_prompt_tokens`
  and `total_tokens` (metric #3).
- `reasoning_content` is logged but never echoed back into the next request: the Gemma 4 template
  strips prior thinking anyway, and re-injecting it is the context bloat Module 3 targets.
- **Parallel and remote grid.** `run_batch.py --workers N` runs N cells at once and should be set to
  the backend's slot count. Measured on this build (`b11696`): **`-c` is the per-slot context** and
  **`-np` defaults to 4**, so `serve.ps1 -Ctx 32768` already serves **4 slots x 32768** (`GET /slots`
  confirms it) - running the first local pass at `--workers 1` left three slots idle. On Colab pass
  `-np` explicitly and budget by **host RAM** (~1.27 GB of RSS per slot), not VRAM.
  `harness/runner.py` serialises `git worktree add/remove` per repo, and keys each worktree by
  `(instance_id, variant)` because the variants of one task run concurrently.
  `--base-url` + `--api-key` point the same harness at a remote llama-server (e.g. a Colab GPU
  behind a tunnel) — see `colab/README.md`. `run_start` logs `base_url`, and `api_key_set` (never
  the secret), so local and remote traces stay distinguishable under one `model` label.
- A turn cut off by the **context limit** returns `finish_reason=length` with no tool call; the loop
  nudges the model and retries (up to 2 times, `length_retry`) instead of ending the run empty-handed.
  Seen on E4B before its context was raised to 32768.
- A run's diff is written to `logs/patches/<run_id>.patch` — that is the Bước 5 evaluator input.

## Backend facts (measured, E2B Q4_0 on this GPU)

- Load time ~10 s; generation **~85 tok/s**; prompt processing ~13 tok/s (first pass, uncached).
- **E4B Q4_0 also fits 6 GB**: `-ngl 99 -c 32768` uses ~3.5 GB VRAM and loads in ~13 s.
  Generation is slower than E2B (a 5-turn M3 run took ~80 s).
- `temperature` defaults to **1.0** on the server — always send `temperature=0.0` explicitly (IDEA.md §3).
- **Thinking mode:** the template's `enable_thinking` defaults to false, but thinking switches on when
  `tools` are passed or the first message has role `system`/`developer`. Control it per request with
  `chat_template_kwargs={"enable_thinking": true|false}`.
  Measured on one arithmetic prompt: thinking=**311** completion tokens vs no-thinking=**4**.
  Chain-of-thought arrives in `reasoning_content`; the answer arrives in `content`.
  This is a first-order variable for Module 3's token metrics — record both fields.
- Constrained decoding works via `response_format={"type":"json_schema", ...}` (Module 2) and native
  function calling works via `tools=[...]` → `message.tool_calls` (agent loop).


## Traps already hit (do not rediscover)

- SWE-bench `repo` field uses `sympy/sympy`, **not** `sympy__sympy`.
- `FAIL_TO_PASS` / `PASS_TO_PASS` are JSON-encoded **strings**, not lists.
- vLLM is Linux-only; the WSL2 fallback has only 7 GB RAM — that is why llama.cpp is the backend.
- Model downloads are ~3.4 GB (E2B) / 5.2 GB (E4B): run them in the background, never in the foreground.
- **Never launch llama-server by piping its output** (`... 2>&1 | Tee-Object`) in a script with
  `$ErrorActionPreference='Stop'`: its stderr banner becomes a terminating `NativeCommandError` and
  kills the server immediately. `scripts/serve.ps1` uses `Start-Process` redirection instead.
- Windows consoles are cp1252: scripts that print non-ASCII (e.g. "BƯỚC") must call
  `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` first.
