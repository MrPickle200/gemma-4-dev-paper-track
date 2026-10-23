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
4. **Organise every experiment under `logs/experiments/<tag>/`.** Use `run_batch.py --experiment <tag>`
   (and `run_task.py --experiment <tag>` for single-cell follow-ups) so traces, patches, manifest,
   results.csv and DIAGNOSTICS.md land in `logs/experiments/<tag>/{runs,patches}/` together. Never
   drop experiment traces into the shared `logs/runs/` + `logs/patches/` — those hold only the
   committed 50-task baseline grid. See `logs/experiments/README.md`.
5. Never commit model weights, environments, or regenerable checkouts: `models/`, `.venv/`,
   `repos/`, `worktrees/`, `tools/downloads/`, and server/console logs (`logs/*.log`,
   `logs/*.log.*`, `logs/*.bak`). **Do commit the run evidence:** `logs/runs/*.jsonl` traces and
   `logs/patches/*.patch` diffs are the primary record behind every reported number, and the
   aggregate `logs/*.csv` tables are cited by `E2B_DIAGNOSTICS_REPORT.md`. `data/subset_50*.json`
   (the task definition) is tracked for the same reason. The same applies to every experiment
   directory: `logs/experiments/<tag>/runs/*.jsonl`, `patches/*.patch`, `manifest.json`,
   `results.csv` and `DIAGNOSTICS.md` are committed together. The evidence is deliberately public
   so any number in the report can be recomputed from `scripts/diagnostics_e2b.py`.

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
logs/runs/*.jsonl       one trace per run; logs/patches/*.patch the model's diff (committed evidence)
logs/experiments/<tag>/ optional per-experiment bundle: runs/, patches/, manifest.json,
                        results.csv, DIAGNOSTICS.md (committed; see logs/experiments/README.md)
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

# 7. a named experiment keeps its evidence out of the committed 50-task grid
#    (logs/experiments/<tag>/runs|patches, plus manifest.json and results.csv)
.\.venv\Scripts\python.exe scripts\run_batch.py --model gemma-4-E2B --thinking on --workers 4 `
    --experiment 20261021-fixes-smoke-e2b --instance-id psf__requests-1724 --instance-id scikit-learn-13142
.\.venv\Scripts\python.exe scripts\diagnostics_e2b.py --experiment 20261021-fixes-smoke-e2b
.\.venv\Scripts\python.exe scripts\summarize_batch.py --experiment 20261021-fixes-smoke-e2b
```

## Bước 2 harness facts

- **Variants** (IDEA.md §3 ablation matrix): all four are implemented — `B0` (cat/search_text),
  `M1` (structured edit schema), `M2` (constrained symbol navigation), `M3` (compaction at
  `lock_targets`). The 2026-10-21 harness fixes below apply to every variant; they are execution-layer
  repairs, not new mechanisms, so the variant meanings are unchanged.
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
  and `total_tokens` (metric #3). A tool call whose arguments do not match the tool's schema counts
  as `malformed_arguments` (it never reached the tool) and the model is told the parameters it may
  use, not our Python signature.
- **Read M1's column with care.** M1's request shape (structured edit schema + the
  `_EDIT_STRUCTURED` prompt) makes E2B garble *unrelated* calls - 9 of its 50 cells wrote
  `search_text` arguments as `pattern="x"\npath` with Gemma 4's template tokens (`<|"|>`,
  `<channel|><|tool_call|>`) inside them, which no B0/M2/M3 cell did at temperature 0. See
  PROGRESS.md before attributing M1's numbers to the syntax gate alone.
- `reasoning_content` is logged but never echoed back into the next request: the Gemma 4 template
  strips prior thinking anyway, and re-injecting it is the context bloat Module 3 targets.
- **Parallel and remote grid.** `run_batch.py --workers N` runs N cells at once and should be set to
  the backend's slot count. `-np` defaults to 4 on this build (`b11696`). On Colab pass `-np`
  explicitly and budget by **host RAM** (~1.27 GB of RSS per slot), not VRAM.
  **`-c` is the context of the whole server, not of a slot** (see the trap below): each slot
  really gets `-c / -np`, and only when that division leaves enough for per-slot caches does the
  server report `kv_unified = 'false'`. To give every one of N concurrent runs a full window,
  launch `-c $((32768 * N)) -np N`, e.g. `serve.ps1 -Model E2B -Ctx 131072 -Slots 4`, which
  measured 4 independent 32768-token caches for +558 MiB of VRAM.
  `run_batch.py` and `run_task.py` therefore **probe the server for real** before every batch
  (`llm.verify_run_context`: N concurrent full-window prompts) and log the answer as
  `ctx_per_run`, because `/props` reports 32768 in both configurations. A run that ended on
  `context_exceeded` is only scored as a Module 3 result when `ctx_per_run >= 32768`; otherwise it
  is a pool artefact, excluded and re-run (see `config.ctx_is_adequate`).
  `harness/runner.py` serialises `git worktree add/remove` per repo, and keys each worktree by
  `(instance_id, variant)` because the variants of one task run concurrently.
  `--base-url` + `--api-key` point the same harness at a remote llama-server (e.g. a Colab GPU
  behind a tunnel) — see `colab/README.md`. `run_start` logs `base_url`, and `api_key_set` (never
  the secret), so local and remote traces stay distinguishable under one `model` label.
- A turn cut off by the **context limit** returns `finish_reason=length` with no tool call; the loop
  nudges the model and retries (up to 2 times, `length_retry`) instead of ending the run empty-handed.
  Seen on E4B before its context was raised to 32768.
- A run's diff is written to `logs/patches/<run_id>.patch` — that is the Bước 5 evaluator input.

## Execution-layer repairs (2026-10-21, in every variant)

These change what the model *sees*, so keep them constant across a comparison grid; do not read a
before/after difference as a variant effect.

- **Paths** go through `tools.resolve_repo_path`: exact, then drop the first directory component
  (`psf/requests/sessions.py` → `requests/sessions.py`), then a unique suffix match in the worktree.
  Ambiguous or missing paths raise `PathError`, which `ToolBox.execute` turns into a tool result
  listing the closest matches. Applies to every `path`/`file`/lock argument.
- **Tool schemas** all carry `"additionalProperties": false`. The argument parser also repairs the
  glued-key shape `{"pattern=\"x\"\npath": ...}`; unrepairable shapes return "Invalid JSON keys" and
  count as `malformed_arguments`.
- **`apply_edit`** matches on a CRLF-normalised copy (real `\r\n` and literal `\\r\\n` both → `\n`)
  in three tiers: exact, trailing-whitespace-insensitive, indentation-flexible (unique site only,
  indentation offset applied to `replace_block` before the AST gate). Writes go through the original
  text and preserve the file's newline style; the mismatch error keeps the `search_block not found`
  prefix used by the diagnostics taxonomy and appends the closest region with line numbers.
  Tiers 1/2 keep the old first-match semantics and count `edits_ambiguous`; tier 3 refuses an
  ambiguous site instead of guessing.
- **Symbols** accept `file.py::Class.method`, `Class.method`, or a bare name unique in `file`; a miss
  lists the file's valid symbols (or repo-wide close matches) instead of an empty error.
- **`lock_targets` is a gate (M3):** if any locked file/symbol does not resolve to non-empty source,
  compaction is refused (`lock_rejected`), Phase 1 is kept and suggestions are returned. The Phase-2
  message carries the per-symbol source and the "Do NOT call get_repo_map or read_symbol" directive,
  and its full text is logged as `compacted_prompt_preview`.
- **Loop-breaker:** an identical `(tool, canonical_json(args))` to the previous turn is not executed;
  the previous result is replayed with a leading `{"LOOP DETECTED": ...}` key so the model's own
  context still contains the refusal. Counted as `loop_blocked`. Argument text that `json.loads`
  rejects with `ValueError` rather than `JSONDecodeError` (Python 3.11's 4,300-digit int limit) is a
  malformed call too, not a harness error.
- `run_batch.py --workers N` halves N until `llm.verify_run_context` passes, so an under-provisioned
  server cannot silently produce `context_exceeded` cells.
- `--experiment <tag>` sequences evidence under `logs/experiments/<tag>/`; the default paths are
  unchanged, so committed grid numbers remain reproducible.

## Bước 5 evaluation facts (2026-10-23, swebench 5.0.2)

- **Dataset is `SWE-bench/SWE-bench_Verified`**, not `princeton-nlp/SWE-bench_Verified`. The old
  name has no `image` column and `make_test_spec` dies with `KeyError: 'image'`. The new one ships
  prebuilt images, so nothing is built locally: one image per task (~3.6-6.9 GB), pulled once and
  reused by every variant and by both models. The 50-task subset needs **50 images**.
- **One predictions file per (model, variant).** `run_evaluation.py:743` rewrites the list as
  `{instance_id: pred}`, so a file holding B0/M1/M2/M3 of the same task keeps only the last row.
  `scripts/build_predictions.py --experiment <tag>` writes `predictions/<model>_<variant>.jsonl`
  with 50 unique ids each; `scripts/run_swebench_eval.py` runs one cell per `--run_id` and merges
  the reports into `swebench_results.csv`.
- **`scripts/patch_swebench_windows.py` is mandatory on Windows** and the runner calls it before
  every run. `Path.write_text` writes `eval.sh` with CRLF, so inside the Linux container
  `cd /testbed\r` fails, conda never activates (`ModuleNotFoundError: mpmath`) and `bin/test ...\r`
  collects nothing; `test_output.txt` is also opened in the cp1252 locale and raises
  `UnicodeEncodeError` on box-drawing characters. Either bug silently turns a whole cell into
  `resolved = 0`.
- Warm evaluation is ~7-20 s per instance, so a 50-cell file takes 4-9 minutes at `--max_workers 4`;
  the first pass pays the image pulls. `--workers` has nothing to do with the 6 GB GPU — this stage
  is CPU/disk bound.
- Do **not** read "ambiguous failures" as infrastructure faults: the classifier greps for
  `no tests ran` / `ModuleNotFoundError`, which pytest's own test suite prints as test data. The
  instances stay in `unresolved`, so the denominator stays at 50.

## Backend facts (measured, E2B Q4_0 on this GPU)

- Load time ~10 s; generation **~85 tok/s**; prompt processing ~13 tok/s (first pass, uncached).
- **E4B Q4_0 also fits 6 GB**: `-ngl 99 -c 32768 -np 1` uses ~3.5 GB VRAM and loads in ~13 s (with
  `-np 4` that same flag means 8192 per slot — see the shared-pool trap).
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
- **Always stream (`stream: true`).** A buffered response carries no byte until the turn has finished
  generating, so behind any proxy with a read timeout a long turn is killed in flight; a streamed one
  answers in ~1 s and then keeps dribbling. `harness/llm.py` sends `stream_options.include_usage`, so
  `usage` and `timings` still ride the final chunk and the counters are unchanged; tool-call
  `arguments` arrive as fragments to be concatenated per `index`. `--no-stream` restores the old
  buffered request. Streaming is a transport detail, not an experimental variable: the folded result
  is identical to the buffered one.


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
- **A Cloudflare quick tunnel kills any request that stays silent for 125 s (HTTP 524).** That is a
  time-to-first-byte limit, not a duration limit, so it cannot be fixed by shortening a turn - only by
  streaming. Buffered, it cost ~34% of a Colab grid (35 llm_error in 104 attempts) against 2 in 126
  locally. The 524 arrives as an HTML page, so the trace's `error` field is the only place it shows.
- **`-c` is the whole server's context, and llama-server hands out `-c / -np` per slot.** Read as
  "per-slot", `serve.ps1 -Ctx 32768` (and therefore every `--workers 4` batch) left each run
  ~8192 tokens instead of 32768: runs hit `decode: Context size has been exceeded` (HTTP 500, four
  slots failing *together* because they shared `kv_unified = true` one 32768-token cache) and long
  runs died for a reason we chose, not one the model chose. `GET /slots` and `/props` both report
  32768 in *both* configurations — the banner's `kv_unified` field is the only tell, and it is not
  exposed over HTTP. Fix: `-c $((32768 * workers)) -np workers` (`serve.ps1 -Ctx 131072 -Slots 4`
  measured 4 independent 32768 caches for +558 MiB of VRAM, leaving ~3.7 GB free), and let
  `llm.verify_run_context` prove it by serving N full-window prompts at once. A `context_exceeded`
  run on a server that fails that probe is a backend artefact: excluded and re-run, never scored.
