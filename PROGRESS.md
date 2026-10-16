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


