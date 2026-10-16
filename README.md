```text
 ██████╗  ███████╗ ███╗   ███╗ ███╗   ███╗  █████╗  ██╗  ██╗
██╔════╝  ██╔════╝ ████╗ ████║ ████╗ ████║ ██╔══██╗ ██║  ██║
██║  ███╗ █████╗   ██╔████╔██║ ██╔████╔██║ ███████║ ███████║
██║   ██║ ██╔══╝   ██║╚██╔╝██║ ██║╚██╔╝██║ ██╔══██║ ╚════██║
╚██████╔╝ ███████╗ ██║ ╚═╝ ██║ ██║ ╚═╝ ██║ ██║  ██║      ██║
 ╚═════╝  ╚══════╝ ╚═╝     ╚═╝ ╚═╝     ╚═╝ ╚═╝  ╚═╝      ╚═╝

        M I C R O - M O D E L   N A T I V E   H A R N E S S
```

**Do small models fail SWE-bench because they cannot reason — or because the harness drowns them?**

![python](https://img.shields.io/badge/python-3.11-3776AB)
![backend](https://img.shields.io/badge/backend-llama.cpp%20%2B%20Q4__0%20QAT-2E8B57)
![benchmark](https://img.shields.io/badge/benchmark-SWE--bench%20Verified%20Mini%20(N%3D50)-8A2BE2)
![status](https://img.shields.io/badge/progress-B%C6%B0%E1%BB%9Bc%201--2%20complete-success)

A research harness that tests one hypothesis: a 2B-effective model like **Gemma 4 E2B** fails
real repository bugs not for lack of logic, but because of two *systemic* bottlenecks —

1. **Context degradation** — attention quality collapses once the window is flooded with raw
   `grep`/`cat` output and long stack traces.
2. **Syntactic fragility** — high rates of malformed output when asked to emit structured
   formats (JSON tool calls, line-number-based patches).

So we move the burden from **model reasoning** to a **deterministic harness**: constrain the
action space by AST, force the output through a schema, and compact context at phase boundaries.

---

## The harness

```text
┌─────────────────────────────────────────────────────────────────────────┐
│ SWE-bench Verified Mini · 50 tasks · 1 file per patch · <50 lines       │
└─────────────────────────────────────────────────────────────────────────┘
                                     │  issue text
                                     ▼
╔═ PHASE 1 · LOCALIZATION ════════════════════════════════════════════════╗
║ M2 · AST-Nav   get_repo_map() · read_symbol() · find_references()       ║
║ symbol-level navigation instead of grep/cat context dumps               ║
╚════════════════════════════════════╤════════════════════════════════════╝
                                     │  lock_targets(files, symbols, note)
╔═ M3 · EPISODIC COMPACTION   scratch history discarded ══════════════════╗
║ context := issue + locked symbol sources + root_cause_note              ║
║ everything the agent grepped and cat'ed before is gone                  ║
╚════════════════════════════════════╤════════════════════════════════════╝
                                     ▼
╔═ PHASE 2 · PATCHING & VERIFICATION ═════════════════════════════════════╗
║ M1 · AST-Edit   constrained JSON schema + in-memory ast.parse() gate    ║
║ nothing reaches disk unless the file still parses                       ║
╚════════════════════════════════════╤════════════════════════════════════╝
                                     │  unified diff
                                     ▼
                           logs/patches/<run_id>.patch
```

| # | Variant | Navigation | Editing | Context | Isolates |
|---|---------|-----------|---------|---------|----------|
| 1 | `B0` | `grep` / `cat` / `find` | raw search/replace, no gate | full history | baseline |
| 2 | `M1` | `grep` / `cat` / `find` | **+** JSON-schema decoding **+** `ast.parse()` gate | full history | AST-Edit |
| 3 | `M2` | **+** tree-sitter symbols | **+** gate | full history | AST-Nav |
| 4 | `M3` | **+** tree-sitter symbols | **+** gate | **+** phase compaction | full |

Every run writes a complete `.jsonl` transcript (full prompt, raw response, `reasoning_content`,
token counts, tool status, timings) plus the model's diff. Nothing is asserted that the trace
does not show.

---

## Field notes (measured, not assumed)

These are the failure modes actually observed on this hardware, with the traces to back them.

```text
gemma-4-E2B · B0 · thinking = off · turn 1
   {"pattern="req.method = method.upper()"
    path": ...}                      ← the comma between keys: never heard of her
   ✗ 15 turns · 0 edits · 26 identical retries

gemma-4-E2B · B0 · thinking = off
   search_block := "  314:         req = Request("
   ✗ line-number prefixes copied straight out of read_file into the patch

gemma-4-E2B · B0 · thinking = on
   search_block := "            method = method.upper(),"
   replace_block := "            method = str(method).upper(),"
   ✓ 5 turns · 1 file changed · equivalent to the gold patch's builtin_str(method)
```

- **Thinking mode is a first-order variable, not a default.** With thinking off, 3/3 runs
  produced *no edit at all*. With it on, 1/4 produced a clean, plausible fix.
- **Localization is the real bottleneck.** Several 15-turn runs searched successfully with zero
  tool errors and simply never committed to an edit — exactly what M1/M3 target.
- **The malformed arguments are the model, not the backend.** A control probe
  (`scripts/diag_toolcalls.py`) showed llama.cpp's Gemma 4 tool-call → JSON conversion is clean
  (3/3). The corruption is hypothesis #2, caught in the act.

> ⚠️ One confound was found and removed: the first `read_file` printed line numbers, and the
> model copied them into `search_block`, manufacturing a fake 11/11 malformed-edit rate.
> `B0` is specified as `cat`, which has no line numbers, so the tool now matches `cat`.
> Baseline numbers below are post-fix. See `PROGRESS.md`.

### Baseline so far — E2B only, small n

| config | runs | reached an edit | malformed edit rate | identical retries |
|---|---:|---:|---:|---:|
| `B0`, thinking off | 3 | **0** | n/a | 26 |
| `B0`, thinking on | 4 | **1** | 0.0 | 14 |

Reproduce with `scripts/summarize_batch.py`. **`edited` ≠ `resolved`** — resolution rate needs the
SWE-bench Docker evaluator (Bước 5). No claim of a solved task is made here yet.

---

## Quickstart

Backend is **llama.cpp** + official **QAT Q4_0** GGUF, not vLLM: vLLM is Linux-only and the
development GPU has 6 GB of VRAM, which cannot hold the bf16 weights (E2B = 10.25 GB,
E4B = 16 GB). The harness speaks only the OpenAI `/v1` API, so pointing `--base-url` at a
vLLM or SGLang server is the only change needed on a bigger GPU.

```powershell
# 0. environment (venv is mandatory; never the system interpreter)
uv venv .venv --python 3.11
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt

# 1. build the N=50 subset from SWE-bench Verified
.\.venv\Scripts\python.exe scripts\filter_subset.py --n 50 --seed 0

# 2. fetch llama.cpp + the Gemma 4 quantised weights (~9 GB, idempotent)
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_inference.ps1

# 3. serve the model (OpenAI-compatible on :8080)
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\serve.ps1 -Model E2B -Ctx 32768

# 4. prove the backend works: endpoint, thinking control, constrained JSON, tool calls
.\.venv\Scripts\python.exe scripts\smoke_test.py

# 5. clone the repos and run the agent
.\.venv\Scripts\python.exe scripts\fetch_repos.py
.\.venv\Scripts\python.exe scripts\run_task.py --list
.\.venv\Scripts\python.exe scripts\run_task.py --instance-id psf__requests-1724 --thinking on

# 6. read the results
.\.venv\Scripts\python.exe scripts\summarize_run.py logs\runs\<run>.jsonl --show-tool apply_edit
.\.venv\Scripts\python.exe scripts\summarize_batch.py --csv logs\batch.csv
```

---

## Layout

```text
harness/config.py        RunConfig (model, variant, limits, thinking) + log paths
harness/llm.py           OpenAI-compatible /v1/chat/completions client
harness/tools.py         tool schemas + implementations + metric counters
harness/agent.py         the agent loop and the per-turn JSONL trace
harness/runner.py        git worktree setup, task loading, patch collection

scripts/filter_subset.py     builds data/subset_50.json from SWE-bench Verified
scripts/fetch_repos.py       clones the repos, verifies every base_commit
scripts/setup_inference.ps1  downloads llama.cpp + Gemma 4 GGUFs
scripts/serve.ps1            starts llama-server
scripts/smoke_test.py        backend acceptance test
scripts/diag_toolcalls.py    probes the Gemma 4 tool-call -> JSON conversion
scripts/run_task.py          runs the agent on one or more tasks
scripts/summarize_run.py     reads one .jsonl trace
scripts/summarize_batch.py   aggregates runs into the ablation table

data/subset_50.json      the 50 tasks   ·  data/subset_50.meta.json  filter+seed manifest
logs/runs/*.jsonl        per-turn traces (evidence)
logs/patches/*.patch     the model's diff, fed to the SWE-bench evaluator
repos/ · worktrees/      clones + per-task detached checkouts (regenerated, not committed)
```

Each task runs in its own detached **git worktree** pinned at `base_commit`, so the shared clones
are never mutated and disk use stays bounded.

---

## Metrics tracked

Extracted from the traces by `scripts/summarize_batch.py`:

| # | Metric | Field in the trace | Tests the claim |
|---|--------|-------------------|-----------------|
| 1 | Resolve rate | *(Docker evaluator, Bước 5)* | overall |
| 2 | Malformed edit rate | `edits_malformed / edits_attempted` | M1 |
| 3 | Peak context & total tokens | `peak_prompt_tokens`, `total_tokens` | M2, M3 |
| 4 | Error rate vs. context depth | `repeated_identical_calls`, prompt tokens per turn | E2B/E4B limits |

Token accounting is split deliberately: `reasoning_content` (chain of thought) is logged
separately from `content`, because thinking mode is the dominant cost driver and would otherwise
make the token charts meaningless.

---

## Hardware reality check

This was developed on a laptop, not a cluster:

```text
GPU     RTX 4050 Laptop · 6141 MiB VRAM · driver 596.21 (CUDA 13.2)
Host    15.7 GB RAM · Windows · CUDA 13.0 toolkit
Model   gemma-4-E2B-it QAT Q4_0 (3.12 GiB) via llama.cpp CUDA build b11696
Speed   ~10 s load · ~85 tok/s generation · ~13 tok/s uncached prefill
```

That last number is the interesting one: **prefill, not decode, is the bottleneck** — which is
itself an argument for Module 1, since dumping raw files into context is paid for in prefill.

E4B (4.8 GiB) has not been run yet and will need a reduced `-ngl` or a smaller `-c`.

---

## What this repo does *not* claim

- No resolve rate yet. One clean diff is not one solved task.
- Sample sizes are 3–4 runs per config. The thinking-mode result is a **lead**, not a conclusion.
- Gemma 4 is multimodal; this harness is **text-only**.
- The repos cloned under `repos/` belong to their respective upstream projects and carry their
  own licenses; nothing from them is redistributed here.

## Progress

- [x] **Bước 1** — venv, N=50 subset (117/500 passed the filter), llama.cpp backend, acceptance test
- [x] **Bước 2** — agent loop, per-turn `.jsonl` traces, worktrees, baseline measured
- [ ] **Bước 3** — implement the three modules: AST-Edit, AST-Nav, `compact_context()`
- [ ] **Bước 4** — batch 4 variants × 2 models × 50 tasks
- [ ] **Bước 5** — SWE-bench Docker evaluation, ablation table, context-depth degradation chart

Full step-by-step history, including the bugs and the confound retraction: [`PROGRESS.md`](PROGRESS.md).
The research spec: [`IDEA.md`](IDEA.md). Working notes for agents: [`AGENTS.md`](AGENTS.md).
