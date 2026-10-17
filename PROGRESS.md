# PROGRESS.md — log of completed work

Append-only. One entry per completed step: what was done, the evidence, and the artifact produced.
Plans and ideas belong in `IDEA.md`, not here.

---

## 2026-10-16 — Bước 1 (Data & Infra), in progress

### Reconnaissance
- Repo contained only `IDEA.md`; `AGENTS.md` and `PROGRESS.md` were empty.
- Hardware probed: RTX 4050 Laptop **6141 MiB VRAM**, 15.7 GB host RAM, driver 596.21 (CUDA 13.2),
  CUDA 13.0 toolkit, Docker 29.7.2, WSL2 Ubuntu (CUDA passthrough OK, only 7 GB RAM inside).
- `uv` present; Python 3.13.1 system default, 3.11.9 also installed.
- Model facts confirmed on HF: `google/gemma-4-E2B-it` (10.25 GB bf16, 5.1B total / 2.3B effective
  with Per-Layer Embeddings, 128K ctx) and `google/gemma-4-E4B-it` (16.0 GB bf16, 8B total).
  Both ungated, Apache-2.0. vLLM supports `Gemma4ForCausalLM`/`Gemma4ForConditionalGeneration`.
- **Decision (with user):** 6 GB VRAM cannot hold bf16 weights → run **llama.cpp CUDA + Q4_0 QAT GGUF**,
  harness speaks only an OpenAI-compatible API so the backend remains swappable. Scope this session = Bước 1.

### Environment setup
- Created `.venv` with `uv` on **Python 3.11.9**: `.venv/Scripts/python.exe`.
- Installed into the venv: `datasets`, `openai`, `pydantic`, `huggingface_hub`, `tree-sitter`,
  `tree-sitter-language-pack`, `tqdm`, `rich`, `pyyaml`, `pytest`, `requests`.
- Verified imports: `datasets 5.1.0`, `tree_sitter 0.26.0`, and a working Python parser via
  `tree_sitter_language_pack.get_parser('python')` (needed by Module 1 AST-Nav).
- Created directories `scripts/`, `data/`, `models/`, `tools/`, `logs/`.

### Subset of 50 tasks — `scripts/filter_subset.py`
- Loaded `princeton-nlp/SWE-bench_Verified` (500 instances, split `test`).
- Filters applied: repo ∈ {`sympy/sympy`, `scikit-learn/scikit-learn`, `pytest-dev/pytest`, `psf/requests`},
  gold patch touches exactly **1 file**, patch **< 50 changed lines**.
- Result: **117 of 500** instances pass; sampled **50** with `seed=0`.
- Composition: sympy 29, scikit-learn 13, pytest 5, requests 3. Changed lines: min 1 / max 38 / mean 10.0.
- Artifacts: `data/subset_50.json` (453 KB, full SWE-bench fields incl. `FAIL_TO_PASS`,
  `PASS_TO_PASS`, `patch`, `test_patch`) and `data/subset_50.meta.json` (filter + seed manifest).
- Bug fixed along the way: SWE-bench `repo` values use `/` (e.g. `sympy/sympy`), not `__`.

### Inference stack (working)
- Wrote `scripts/setup_inference.ps1` (idempotent): downloaded llama.cpp build **b11696**
  (`llama-b11696-bin-win-cuda-13.4-x64.zip` + `cudart-llama-bin-win-cuda-13.4-x64.zip`) into
  `tools/llama.cpp/`, plus the official **QAT Q4_0** GGUFs into `models/`:
  `gemma-4-E2B_q4_0-it.gguf` (3.12 GiB) and `gemma-4-E4B_q4_0-it.gguf` (4.80 GiB).
  (Replaced IDEA.md's vLLM server: vLLM is Linux-only and 6 GB VRAM cannot hold bf16 weights.
  Approved with the user. The harness will speak only the OpenAI `/v1` API, so vLLM stays a
  drop-in swap on a larger GPU.)
- Verified the binary sees the GPU: `llama-cli --list-devices` → `CUDA0: RTX 4050 Laptop GPU (6140 MiB)`.
- Wrote `scripts/serve.ps1` to launch llama-server with `Start-Process` stream redirection.
- Wrote `scripts/smoke_test.py`, the Bước 1 acceptance test.
- **Bước 1 ACCEPTANCE: PASS** (`logs/smoke_test.log`), covering:
  - endpoint alive, model loads in **9.6 s**, serves `gemma-4-E2B_q4_0-it.gguf` on `:8080`;
  - plain generation at `temperature=0.0` → `HARNESS OK`, **~85 tok/s** (96 tokens in 1.5 s);
  - thinking control works: same prompt costs **311** completion tokens with
    `enable_thinking=true` vs **4** with `false`;
  - JSON-schema constrained decoding returns all 4 required Module 2 keys;
  - native function calling emits `get_repo_map({"path":"harness/agent.py"})`.

### Findings that change the design
- **Thinking mode is a first-order cost variable.** The Gemma 4 template defaults
  `enable_thinking=false`, but thinking activates whenever `tools` are passed or the first message
  has role `system`/`developer` — i.e. exactly the harness's situation. Chain-of-thought arrives in
  `reasoning_content` and the answer in `content`; the harness must log **both** token streams, and
  must set `enable_thinking` explicitly per phase.
- The server's default `temperature` is **1.0**, so `temperature=0.0` must be sent on every request.
- Measured throughput ~85 tok/s with ~13 tok/s uncached prefill makes the IDEA.md budget
  (50 tasks × 15 turns) tractable on this laptop.

### Bugs found and fixed
- Subset filter matched 0 tasks: SWE-bench `repo` values use `/` (`sympy/sympy`), not `__`.
- First server launch died instantly: piping llama-server's stderr banner through `Tee-Object` while
  `$ErrorActionPreference='Stop'` raised a terminating `NativeCommandError`. Fixed with
  `Start-Process -RedirectStandardOutput/-RedirectStandardError`.
- `smoke_test.py` crashed printing "BƯỚC" under the cp1252 Windows console; fixed by reconfiguring
  stdout to UTF-8.

### Documentation
- `AGENTS.md`: venv rule, verified environment table, the 6 GB VRAM constraint and why llama.cpp was
  chosen, layout, common commands, measured backend facts (thinking mode, tok/s), and traps already hit.
- `PROGRESS.md` created as an append-only log (this file).
- `requirements.txt` (pinned venv deps) and `.gitignore` (weights, logs, venv, downloads).

### Left for Bước 2
- Clone each of the 50 repos at its `base_commit` (nothing reads code yet).
- Agent loop with per-turn `.jsonl` logging of prompt, response, both token counts and tool status.
- `swebench` + Docker evaluation harness is Bước 5.

---

## 2026-10-17 — IDEA.md updated, Bước 2 (Core Loop) complete

### IDEA.md edits (as requested)
- Bước 1 checkbox ticked `[x]`, with the real numbers recorded inline.
- All three vLLM references rewritten for the llama.cpp reality: Module 2's `guided_json` now names
  llama.cpp's `response_format={"type":"json_schema"}` as the equivalent, and §3's Models line now
  states the backend, the reason vLLM was dropped (6 GB VRAM, Linux-only), the measured throughput,
  and the thinking-mode trap.

### Repos — `scripts/fetch_repos.py`
- Full clone per repo into `repos/<owner>__<name>` (full history is required so pinned `base_commit`s
  exist locally): requests 19 MB, pytest 55 MB, scikit-learn 228 MB, sympy 250 MB.
- **All 4 repos cloned, all 50 `base_commit`s verified present** via `git cat-file -e`.
- Per-task checkouts are created lazily as detached **git worktrees** (`worktrees/<instance_id>`) and
  removed after each run, so the clones are never mutated and disk stays bounded.

### Core loop — `harness/`
- `config.py` `RunConfig` (model, variant, base_url, temperature, max_tokens, max_turns, thinking,
  line_numbers) with a `run_id` of `<instance>__<model>__<variant>__<timestamp>`.
- `llm.py` an OpenAI-compatible client written on `requests` so llama.cpp-only fields
  (`timings`, `reasoning_content`) survive into the logs; swapping `base_url` reaches vLLM/SGLang.
- `tools.py` B0 tool set per IDEA.md §3 row 1 — `search_text` (grep), `read_file` (cat),
  `list_dir` (find), `apply_edit` (raw search/replace, no syntax gate), `finish` — plus the metric
  counters for IDEA.md §4.
- `agent.py` the loop plus the per-turn JSONL trace; `runner.py` worktree + patch collection;
  `scripts/run_task.py` the CLI.

### JSONL trace (Bước 2's actual deliverable)
One `logs/runs/<run_id>.jsonl` per run, appended live, with `run_start` (full config, tool list),
one `turn` record each (request shape, full `content`, `reasoning_content`, tool calls,
prompt/completion/cached tokens, server timings, wall time), one `tool` record each (args, status,
error, meta, output size and preview, `repeat_of_turn`), an `artifacts` record (patch path, diff
stat) and a `run_end` record carrying the §4 aggregates: `turns_used`, token totals,
`peak_prompt_tokens`, `edits_attempted`, `edits_malformed`, `malformed_arguments`,
`repeated_identical_calls`, `tool_errors`, `stop_reason`.
Patches are written to `logs/patches/<run_id>.patch`.

### 9 runs executed (2 models not yet; E2B only) — evidence in `logs/`
| config | runs | runs reaching an edit | malformed edit rate | repeats |
|---|---|---|---|---|
| B0, thinking off | 3 | 0 | n/a | 26 |
| B0, thinking on | 4 | 1 | 0.0 | 14 |
| B0, pre-flag runs | 2 | 1 | 1.0 | 7 |
- Successful run: `psf__requests-1724`, 5 turns, 19 119 tokens, `finish`, diff
  `1 file changed, 1 insertion(+), 1 deletion(-)`. The model changed
  `method = method.upper()` to `method = str(method).upper()`, semantically equivalent to the gold
  patch's `builtin_str(method)` — a plausible, real fix.
- Table reproducible via `scripts/summarize_batch.py` (`logs/batch_baseline.csv`).

### Findings
1. **`thinking` is a first-order variable, not a default.** With thinking off, E2B repeatedly emitted
   tool arguments missing the separator between key/value pairs
   (`{"pattern=...\"\npath": ...}`), which the dispatcher rejects; 3/3 runs produced no edit at all
   and looped (26 identical repeated calls). With thinking on, 1/4 runs solved the task cleanly.
2. **Localization, not editing, is the bottleneck.** Several 15-turn runs searched successfully with
   zero errors and simply never committed to an edit — the failure mode Module 1/Module 3 target.
3. A control (`scripts/diag_toolcalls.py`) proved llama.cpp's Gemma 4 tool-call→JSON conversion is
   correct (3/3 probes clean), so the malformed arguments are the **model's** syntactic fragility,
   not a backend defect. This is IDEA.md hypothesis #2 observed directly.

### Confound found and removed
- The first `read_file` printed line numbers, and the model copied the `314: ` prefixes into
  `search_block`, giving a fake 11/11 malformed edit rate. IDEA.md defines B0 as `cat`, which has no
  line numbers, so `read_file` now omits them by default (`--line-numbers` opts in) and that run is
  labelled `n/a` rather than `ln=False` in the aggregate. Baseline numbers above are post-fix.

### Note on a 6 GB mistake worth avoiding again
- The GPU is 6 GB; an E2B Q4_0 server holds ~2 GB VRAM and ~2 GB host RAM, so the agent loop and the
  server coexist comfortably. E4B (4.8 GiB) has **not** been run yet and will need a reduced `-ngl`
  or a smaller `-c`.

### Scripts added
`scripts/fetch_repos.py`, `scripts/run_task.py`, `scripts/summarize_run.py`,
`scripts/summarize_batch.py`, `scripts/diag_toolcalls.py`.

### Left for Bước 3
- `M1`: constrained JSON edit schema + in-memory `ast.parse()` syntax gate before writing.
- `M2`: `tree_sitter` navigation tools `get_repo_map` / `read_symbol` / `find_references`.
- `M3`: `lock_targets()` + episodic context compaction between the localization and patching phases.

---

## 2026-10-17 — Bước 3 (Implement Modules M1/M2/M3) complete

### What was built
The three ablation modules now exist and are selected by `--variant` alone; `B0` code paths are
untouched so the baseline stays honest.

- **New `harness/astnav.py`** — tree-sitter Python navigation (Module 1). Owns the repo file
  walker (`SKIP_DIRS`/`SKIP_EXTS`/`iter_files`, which `tools.py` now imports) and returns
  `Symbol` records (name, qualified name, byte span, line span, nesting depth, signature,
  docstring). `find_references` matches only tree-sitter `identifier` nodes, so a name inside a
  comment or string literal can never produce a false hit — the grep failure mode Module 1 targets.
- **`harness/tools.py`** — now variant-aware. `ToolBox(variant=...)` exposes:
  - `M2/M3`: `get_repo_map`, `read_symbol`, `find_references` (hiding the bash tools);
  - `M1+`: `apply_edit` gains `target_symbol` and runs an **in-memory `ast.parse()` syntax gate** —
    a syntactically broken edit is rejected and *nothing is written* (`_syntax_error`);
  - `M3`: `lock_targets(files, symbols, root_cause_note)`.
  New counters for IDEA.md §4: `edits_syntax_rejected`, `nav_calls`, `lock_calls`,
  `symbols_locked`, `compactions`.
- **`harness/agent.py`** — per-variant system prompts, phase tracking (`localize`→`patch` for M3),
  and `compact_messages()`: the moment `lock_targets` fires, the whole scratch history is replaced
  by a clean Phase-2 context of *issue + root-cause note + verbatim source of the locked symbols*
  (Module 3). Each compaction is written as a `compaction` record (before/after char counts,
  files, symbols); `turn`/`tool` records now carry `phase`.
- **`harness/runner.py`** passes `variant` through; `scripts/summarize_run.py` and
  `summarize_batch.py` print the new counters/columns.

### Tests — `tests/test_modules.py` (6 passing, no server needed)
Syntax gate rejects + writes nothing / B0 has no gate; `find_references` ignores comment and string
literals; `read_symbol` (incl. dotted `Thing.method`) and `get_repo_map`; exact per-variant tool sets;
and an M3 end-to-end with a scripted fake client asserting the post-`lock_targets` request is the
clean 2-message Phase-2 context and the file is edited.

### Real runs — same-session slice, E2B Q4_0, `thinking on`, max_turns 12, 3 `psf/requests` tasks
| variant | finish | runs w/ edit | malformed edits | mean tokens | mean peak ctx | compactions | syntax-rejected |
|---|---|---|---|---|---|---|---|
| B0 | 1/3 | 1/3 | 0 | 38 249 | 5 894 | 0 | 0 |
| M1 | 2/3 | 2/3 | 0 | 42 930 | 5 499 | 0 | 0 |
| M2 | 2/3 | 2/3 | 0 | **30 822** | 5 528 | 0 | 0 |
| M3 | **3/3** | **3/3** | 0 | 32 766 | **4 308** | 3 | 0 |

- M3 gives the best localization→patch outcome and the smallest peak context; the trace shows
  compaction working directly (`requests-1724`: prompt tokens 5 186 on the pre-lock turn → 4 874
  after compaction). M2 is the cheapest per task (AST navigation returns structure instead of
  raw file text).
- **"finish/edit" is not "resolve".** Manual gold comparison of M3's three patches: `requests-1724`
  matches (`str(method).upper()` ≈ gold's `builtin_str(method)`), but `requests-2931` and
  `requests-6028` are plausible-looking edits in unrelated code. Resolve rate must come from the
  Bước 5 Docker evaluator; this table only shows how far the harness got the model.

### Findings
1. **Compaction must be a one-way phase transition.** Compacting on *every* `lock_targets` call
   (the literal reading of IDEA.md) let E2B lock 6 times in a row, each time re-reading the same
   clean context and re-locking — 64 k tokens to finish. Compacting once, and answering later locks
   with "you are already in the patching phase", removed the loop (same task: 7 turns, 34 k tokens).
2. **Small models name nested symbols with dots** (`read_symbol("Session.request")`). The first M2
   run burned 9 turns on that one rejection. `find_symbol` now accepts bare, `Class.method` and
   deeper qualified suffixes, and the not-found error lists all available qualified names.
3. **The syntax gate never fired** in these 9 runs (`edits_syntax_rejected=0`): every edit the
   model committed was parseable. The gate is exercised by unit tests, not yet by the model —
   the malformed-edit failure mode shows up earlier, as bad tool *arguments*.
4. **`thinking on` degrades llama.cpp's tool-call → JSON conversion.** With thinking off the probe
   in `scripts/diag_toolcalls.py` was 3/3 clean; with thinking on, several `search_text` calls
   arrived as a single JSON *key* equal to the whole argument string
   (`bad arguments ... unexpected keyword argument 'pattern="sessions.py'`), which the dispatcher
   rejects. This is IDEA.md hypothesis #2 (syntactic fragility) observed in the wild; the AST tools
   (M2/M3) submit different, simpler arguments and avoid most of it.

### Left for Bước 4 / 5
- Bước 4: the full 4×2 grid on 50 tasks (and E4B, which still needs a reduced `-ngl`/`-c` on 6 GB).
- Bước 5: `swebench` Docker evaluation for the resolve rate, then the ablation table and the
  error-vs-context-depth plot.

---

## 2026-10-17 — Bước 4 (Execution) started: grid runner + E4B brought up

### Batch runner — `scripts/run_batch.py`
- Runs one variant × task grid per invocation against whatever `llama-server` is up (the 6 GB GPU
  holds one model at a time, so a `--model` label names the served weights).
- **Resumable**: before each cell it looks in `logs/runs/` for a *completed* trace whose config
  matches `(model, variant, thinking, max_turns, line_numbers)` and skips it, so an interrupted
  batch continues where it stopped (`--no-skip` forces a re-run).
- Guards against a dead server: it pings `/health` before every cell and aborts the batch after 3
  consecutive failures instead of burning through the grid writing error stubs.
- `--dry-run` prints the plan. Verified: the E2B 4×50 grid is 200 cells, of which the 4 B0
  thinking-on traces from Bước 2 are already counted as done → 196 to run.

### E4B is now known to fit 6 GB (was an open question)
- `scripts/serve.ps1 -Model E4B -Ctx 32768 -GpuLayers 99` loads in ~13 s and uses **~3.5 GB VRAM**
  (measured at 8192/16384/32768 ctx: 3157/3293/3565 MiB). The earlier "E4B needs a reduced
  `-ngl`/`-c`" note is superseded — 32768 ctx fits comfortably.
- One E4B M3 run on `psf__requests-1724` finished in 5 turns / 30 546 tokens with a *correct* patch
  (`str(method).upper()`), compaction firing (21 508 → 13 110 chars).

### Bug found and fixed: context-limit truncation was treated as a decision to stop
- At `-Ctx 8192`, E4B filled the window (prompt 7 503 + completion 689 = 8192) and its turn came
  back with `finish_reason=length` and **no tool call**. The old loop read "no tool call" as
  "answered" and ended the run with `no_tool_call` and zero edits.
- `harness/agent.py` now treats a truncated turn as retryable: it appends the partial assistant
  message plus a nudge ("call `apply_edit` … or `finish`") and continues, up to 2 times
  (`stop_reason=length_retry`). Covered by `tests/test_modules.py` (now 7 passing).

### Launched
- The E2B grid (`B0 M1 M2 M3` × 50 tasks, `thinking on`, `max_turns 15`) is running in the
  background via `scripts/run_batch.py`. Results will be recorded here and aggregated by
  `scripts/summarize_batch.py` when the batch completes.

### Measurement gap found while the batch ran: no-op edits
- Several cells reported `apply_edit ok` yet `diff: (empty)`. Inspecting the traces showed the model
  emitted **identical `search_block` and `replace_block`** (a do-nothing "edit") and then called
  `finish` — e.g. `psf__requests-6028` M1 turn 13, `requests-6028` M2 turn 9, `pytest-10081` M1
  turn 11. The harness was right to report success; the metric was wrong to count it as an edit.
- Added an `edits_noop` counter to `harness/tools.py` (flagged in the tool `meta`) for future runs.
  Because the running batch imported its modules at launch, the authoritative count for the whole
  grid is derived **post-hoc** in `scripts/summarize_batch.py` from each trace's `apply_edit`
  arguments, so old and new runs are measured the same way. The table now shows `edited` (reached
  an edit) vs `realed` (actually changed text) and a `noop` count.
- Effect on the first 5 M1 grid cells: `edited=4` but `realed=2` — half of M1's "edits" changed
  nothing. This is the kind of inflation Bước 5's resolve rate will settle, but the harness can
  already separate it from the trace.

### Interruption and resume
- At ~12:26 the E2B `llama-server` and the batch process were both killed by an external event
  (an unrelated Kaggle job started on the machine at the same minute; the GPU was idle afterwards).
  Nothing was lost: all completed traces are on disk.
- Recovered by restarting E2B and re-running `scripts/run_batch.py`, which skipped the 43 finished
  cells and resumed the remaining 157. The batch's console output now also goes to
  `logs/batch_e2b.log` so it survives a lost shell buffer.

### Progress trigger — `scripts/batch_progress.py`
- Prints a snapshot (grid cells done per variant with finish/edit/real/noop, mean tokens and peak
  context, plus backend health); `--append` also writes it to `logs/batch_progress.log`.
- Registered a Windows Scheduled Task **`Gemma4BatchProgress`** (`schtasks`, `/SC MINUTE /MO 15`)
  that runs it every 15 minutes, so the progress trail is durable even if this session ends.

### Moving the grid to a Colab GPU (endpoint-only, in preparation)
- Motivation: the 6 GB laptop GPU serves one slot, so the 400-cell grid is serialised; measured
  2-5 min/cell (a single 16384-token truncated turn alone costs ~217 s). A Colab GPU runs the same
  llama.cpp build and Q4_0 GGUF with several slots, so the same grid can run N-way parallel.
- Design choice (operator): keep the agent loop local and move only the model server to Colab
  behind a tunnel. The harness already only speaks OpenAI-compatible HTTP, and `run_start` records
  `base_url`, so local and Colab traces are distinguishable even under the same `model` label.
- `scripts/run_batch.py`: added `--workers N` (concurrent cells; `1` = the previous sequential
  behaviour with unchanged output) and `--api-key` (bearer token, for a publicly reachable tunnel).
- `harness/runner.py`: `git worktree add/remove` is now serialised per repo with a reentrant lock,
  because the grid can touch one clone from several threads. Verified with a scratch-clone stress
  test (24 threads, mixed add/remove/prune): 24/24 locked ops succeeded and, on this git version,
  48/48 unlocked ops succeeded too - so the lock is cheap insurance rather than a demonstrated fix.
- `harness/llm.py` / `harness/config.py`: optional `api_key` -> `Authorization: Bearer`; the secret
  is never logged (`as_dict()` records only `api_key_set: true|false`).
- `colab/gemma4_backend.ipynb` + `colab/README.md`: builds llama.cpp tag `b11696` with CUDA, pulls
  the *identical* QAT Q4_0 GGUF, serves with `-np <slots>` and `-c slots*32768` (so each slot keeps
  the laptop's 32768 window), and publishes it with a `cloudflared` quick tunnel; the final cells
  print `BASE_URL` and `API_KEY` for the laptop command. Parity table is in `colab/README.md`.
- Status: code compile-checked, `tests/test_modules.py` 7/7, grid skip logic re-verified via
  `--dry-run`, and the runner path stress-tested - but **no run has yet gone through a Colab
  endpoint**; that waits on the operator starting the notebook and pasting `BASE_URL`/`API_KEY`.

### Switched the grid to the Colab endpoint - and two `--workers` bugs found and fixed
- Backend verified before using it: `/health` ok, `total_slots: 8`, `model_path` = the same
  `gemma-4-E2B_q4_0-it.gguf`, per-slot `n_ctx: 32768` (parity with the laptop), bearer auth enforced
  (401 without the key), and a tool call round-trips over the tunnel (`add(a=2,b=3)` parsed, thinking
  on yields `reasoning_content`). Build `b1-ceefd16` from tag `b11696`.
- The first Colab attempt failed for a reason worth recording: **not VRAM but port 8080 in use**
  (`llama_server: exiting due to HTTP server error` / `couldn't bind HTTP server socket`), which left
  a `<defunct>` zombie. The notebook now kills stale server/tunnel processes and picks a free port,
  caches build+weights in Google Drive (so a runtime restart is cheap), auto-sizes slots to VRAM, and
  falls back from `--flash-attn on` to `auto` (Turing is picky).
- **Bug 1 (correctness - would have corrupted data):** worktrees were keyed by `instance_id` only, so
  the 4 variants of one task shared a single directory. Under `--workers 8` four workers tore down and
  re-created each other's checkout, mixing edits across variants. Caught within ~30 s of launch by
  reading the log; the 8 traces from that window were deleted and the worktrees reset. Fixed:
  `runner.worktree_path()` now keys by `(instance_id, variant)`.
- **Bug 2 (crash risk):** `astnav._parser` was a module-level singleton tree-sitter parser, and
  tree-sitter parsers are not thread-safe; eight workers parsing at once could segfault. Now
  thread-local.
- Regression tests added to `tests/test_modules.py`: variant-scoped worktree isolation (four
  concurrent setups stay distinct and never touch each other or the clone) and per-thread parser
  identity. Suite is 9/9.
- Relaunched against the tunnel: 8 workers, 108 cells already done, 92 to run; 8 distinct live
  worktrees confirmed. The local batch was stopped (its `llama-server` is left warm as a fallback).
- Throughput measured on the parallel run: **17 cells in 15 min (~1.1/min)** vs ~0.4/min on the
  laptop - roughly 3x even with the tunnel in the path.
- Mid-run the Colab `llama-server` died: Cloudflare returned **502** (edge alive, origin gone), so
  in-flight requests failed, 10 runs ended with `stop_reason=llm_error`, and `run_batch.py` aborted
  on its health check after 29/92 cells. Cause not yet confirmed (OOM under load vs the runtime being
  recycled); the diagnosis path is recorded in `colab/README.md`.
- **Infrastructure failures no longer count as results.** New `config.is_infra_failure()` is used by
  both `run_batch.py` (such cells are re-run, never skipped as done) and `summarize_batch.py` (they
  are excluded from the table, with a count printed). Effect: the grid went from a misleading
  "137 done" back to the honest **125 done / 75 to run**, and 12 runs left the aggregate.
- `colab/gemma4_backend.ipynb` now caps slots with `MAX_SLOTS` derived from a conservative KV budget
  (~3 GiB per 32768-token slot, minus weights + CUDA context) instead of blindly trying 8 first, so
  it cannot pick a config that OOMs once contexts fill. T4 -> 3, L4 -> 6, A100 -> 8.
- Added `scripts/check_endpoint.py`: verifies health, auth, per-slot context, and a real tool call in
  seconds - run it before every batch and after every Colab restart.
- Root cause of the crash, read off the Colab VM: **host RAM, not VRAM**. The T4 instance has only
  ~12.7 GB of system RAM and it was saturated at 12/12.7 GB; the host OOM-killed `llama-server` while
  `nvidia-smi` still looked comfortable. Because the runtime survived, the symptom was a **502** (the
  edge reached a live tunnel whose origin was dead) and only the origin died. Restarted with **3
  slots** on port 8081 - the T4's VRAM budget is (14.6 - 4) / ~3.05 = 3.
- Therefore the slot count is bounded by **host RAM as well as VRAM**: a T4 gets 3 slots, not 8,
  whatever the VRAM arithmetic suggests. Failure-mode ladder, now in `colab/README.md`:
  **404/530** = tunnel gone, **502** = tunnel up but origin dead, `couldn't bind` = stale port.
- Quick-tunnel URLs are single-use: a restart produces a new hostname, and a stale one fails
  permanently (530). The in-kernel `BASE_URL` variable can outlive the tunnel it named, so the
  printed URL must always be re-confirmed with `scripts/check_endpoint.py` before a batch.
- **Parity check on the laptop server passed, and it changed the throughput picture.** `GET /slots`
  on the running E2B server reports **4 slots, each with n_ctx = 32768** - not one slot. This build
  treats `-c` as the **per-slot** context and defaults `-np` to **4**; the Colab formula
  `-c slots*CTX_PER_SLOT` was therefore being clamped back to the model's 32768, which is why both
  servers ended up at `n_ctx_slot = 32768`. Consequences: (1) every local cell already ran at 32768
  per slot, so **the local traces are parity-correct and need no re-run**; (2) the whole local E2B
  grid ran with `--workers 1` while three slots sat idle, so 2.5x of available batching was unused.
- Measured batching on the laptop GPU (4 *identical* short prompts, so prompt caching flatters it):
  1 request 3.40 s vs 4 concurrent 5.32 s = **1.56x**, i.e. genuine parallelism rather than a queue.
- The Colab crash was **host RAM, not VRAM**: `dmesg` shows
  `Memory cgroup out of memory: Killed process 58999 (llama-server) ... anon-rss:10186332kB` on a
  12.97 GB VM (it also killed Colab's `node` and `language_server`). That is ~1.27 GB of host RSS per
  slot, so the real slot limit on this VM is RAM (~4), not the VRAM budget the notebook was using -
  the notebook's slot heuristic is being rewritten around host RAM, and `serve.ps1` gains a `-Slots`
  flag so slot count is explicit and reproducible instead of an undocumented build default.
- Re-verified the replacement Colab endpoint (`/health` ok, 3 slots x 32768, auth enforced, tool-call
  round-trip) and relaunched the grid on it at `--workers 3`; 75 cells to run.
- `scripts/batch_progress.py` now honours `is_infra_failure()` too, so the 15-minute report agrees
  with `run_batch.py` (it had been claiming 137/200 against the true 125/200) and it prints which
  base_url it probed for health.


