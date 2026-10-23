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

![backend](https://img.shields.io/badge/backend-llama.cpp%20%2B%20Q4__0%20QAT-2E8B57)
![benchmark](https://img.shields.io/badge/benchmark-SWE--bench%20Verified%20Mini%20(N%3D50)-8A2BE2)
![status](https://img.shields.io/badge/progress-B%C6%B0%E1%BB%9Bc%201--5%20complete-success)

Gemma 4 **E2B / E4B**, 50 SWE-bench Verified tasks, 4 harness variants, **400 Docker-evaluated
cells** on one 6 GB laptop GPU. Short answer: the harness matters — symbol navigation more than
doubles E4B's resolve rate (6 % → 16 %) — but it does not close the gap to a frontier model.

## Hypothesis

A 2B-effective model fails real repository bugs not for lack of logic, but because of two
*systemic* bottlenecks:

1. **Context degradation** — attention collapses once the window is flooded with raw `grep`/`cat`
   output and long stack traces.
2. **Syntactic fragility** — high malformed-output rates when asked to emit structured formats
   (JSON tool calls, line-number patches).

So move the burden from **model reasoning** to a **deterministic harness**: constrain the action
space by AST, force output through a schema, compact context at phase boundaries.

| # | Variant | Navigation | Editing | Context | Isolates |
|---|---------|-----------|---------|---------|----------|
| 1 | `B0` | `grep` / `cat` / `find` | raw search/replace, no gate | full history | baseline |
| 2 | `M1` | `grep` / `cat` / `find` | **+** JSON-schema decoding **+** `ast.parse()` gate | full history | AST-Edit |
| 3 | `M2` | **+** AST symbols | **+** gate | full history | AST-Nav |
| 4 | `M3` | **+** AST symbols | **+** gate | **+** phase compaction | full |

Every run writes a per-turn `.jsonl` trace (full prompt, raw response, `reasoning_content`, token
counts, tool status) and the model's diff. Nothing is claimed that the trace does not show.

## Results — resolve rate, n = 50 per cell

| model | `B0` | `M1` | `M2` | `M3` | total |
|---|---:|---:|---:|---:|---:|
| Gemma 4 **E2B** | 3 | 4 | 5 | 3 | **15 / 200** (7.5 %) |
| Gemma 4 **E4B** | 3 | 6 | **8** | **8** | **25 / 200** (12.5 %) |

Reproduce from the committed bundles (`experiments/20261023-{e2b,e4b}-grid/`):

```powershell
.\.venv\Scripts\python.exe scripts\summarize_batch.py --experiment 20261023-e2b-grid
.\.venv\Scripts\python.exe scripts\build_predictions.py --experiment 20261023-e2b-grid
.\.venv\Scripts\python.exe scripts\run_swebench_eval.py --experiment 20261023-e2b-grid --max-workers 4
```

### What actually moved

- **Symbol navigation is the win.** `M2`/`M3` go from 6 % to 16 % on E4B — the only change that
  survives contact with all 50 tasks.
- **`M3`'s gain is stopping cleanly, not being right.** On E4B it compacts at `lock_targets`, ends
  runs deliberately (`finish` 42/50 vs 30/50 for `B0`) and cuts tokens, but resolves no more tasks
  than `M2`.
- **Thinking mode is a first-order variable, not a default.** With it off, E2B produced 0 edits in
  3 runs and 26 identical retries. All grid cells run thinking on.
- **E2B has a ceiling the harness cannot lift.** 6–10 % across every variant; the malformed-edit
  rate stays 33–67 %.
- ⚠️ **Read E2B's `M1` column with care.** The structured-edit prompt also garbles *unrelated*
  calls (9/50 cells wrote Gemma template tokens into `search_text` args, which no other variant
  did at temperature 0), so its number is not a clean measure of the syntax gate.
- A run that ended on `context_exceeded` is only scored when the server proved 32768 tokens *per
  slot*; otherwise it is a backend-pool artefact and is re-run.

## Quickstart

Backend is **llama.cpp** + official **QAT Q4_0** GGUF, not vLLM: vLLM is Linux-only and the dev GPU
has 6 GB of VRAM (bf16 needs 10.25 GB for E2B, 16 GB for E4B). The harness speaks only the OpenAI
`/v1` API — pointing `--base-url` at vLLM/SGLang on a bigger GPU is the only change needed.

```powershell
uv venv .venv --python 3.11
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt

.\.venv\Scripts\python.exe scripts\filter_subset.py --n 50 --seed 0
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_inference.ps1   # ~9 GB, idempotent
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\serve.ps1 -Model E2B -Ctx 32768
.\.venv\Scripts\python.exe scripts\smoke_test.py        # endpoint, thinking, JSON, tool calls

.\.venv\Scripts\python.exe scripts\fetch_repos.py
.\.venv\Scripts\python.exe scripts\run_task.py --instance-id psf__requests-1724 --thinking on `
    --experiment 20261023-e2b-grid
```

## Layout

```text
harness/config.py   RunConfig (model, variant, limits, thinking) + experiment paths
harness/llm.py      OpenAI-compatible /v1 client (streaming, usage accounting)
harness/tools.py    tool schemas + implementations + metric counters
harness/agent.py    the agent loop and the per-turn JSONL trace
harness/runner.py   git worktrees, task loading, patch collection

scripts/            filter_subset · fetch_repos · serve.ps1 · smoke_test · run_task/run_batch
                    summarize_run/summarize_batch · diagnostics_e2b · build_predictions
                    run_swebench_eval (+ patch_swebench_windows)

data/subset_50.json the 50 tasks      ·  data/subset_50.meta.json  filter + seed manifest
experiments/<tag>/  one directory per experiment, everything that belongs to it:
                    runs/ patches/ predictions/ reports/ manifest.json results.csv
                    DIAGNOSTICS.md swebench_results.csv   (run_evaluation/ is gitignored scratch)
```

Each task runs in its own detached **git worktree** pinned at `base_commit`, so the shared clones
are never mutated. See [`experiments/README.md`](experiments/README.md).

## Metrics (IDEA.md §4)

| # | Metric | Field in the trace | Tests |
|---|--------|-------------------|-------|
| 1 | Resolve rate | Docker evaluator (`swebench_results.csv`) | overall |
| 2 | Malformed edit rate | `edits_malformed / edits_attempted` | `M1` |
| 3 | Peak context & total tokens | `peak_prompt_tokens`, `total_tokens` | `M2`, `M3` |
| 4 | Error rate vs. context depth | `repeated_identical_calls` per turn | E2B/E4B limits |

`reasoning_content` is logged separately from `content`: thinking is the dominant cost driver and
would otherwise make the token charts meaningless.

## Hardware reality check

```text
GPU     RTX 4050 Laptop · 6141 MiB VRAM · driver 596.21 (CUDA 13.2)
Host    15.7 GB RAM · Windows · CUDA 13.0 toolkit
Model   gemma-4-E2B-it (3.12 GiB) and E4B (4.8 GiB) QAT Q4_0 via llama.cpp b11696
Speed   ~10-13 s load · ~85 tok/s generation · ~13 tok/s uncached prefill
```

**Prefill, not decode, is the bottleneck** — which is itself an argument for `M1`/`M2`: dumping raw
files into context is paid for up front. `-c` is the *server's* context and each slot gets `-c/np`,
so a 4-worker grid needs `serve.ps1 -Ctx 131072 -Slots 4` to give every run a real 32768-token
window.

## What this repo does *not* claim

- Single seed, temperature 0, `n = 50` per cell — a lead, not a benchmark leaderboard entry.
- No frontier-model comparison; no claim that the harness closes the gap.
- Gemma 4 is multimodal; this harness is **text-only**.
- Repos under `repos/` belong to their upstream projects and keep their own licenses; nothing from
  them is redistributed here.

## Progress

- [x] **Bước 1** — venv, N=50 subset (117/500 passed the filter), llama.cpp backend, acceptance test
- [x] **Bước 2** — agent loop, per-turn `.jsonl` traces, worktrees, baseline measured
- [x] **Bước 3** — AST-Edit, AST-Nav, `compact_context()`, all four variants
- [x] **Bước 4** — batch 4 variants × 2 models × 50 tasks (400 traces)
- [x] **Bước 5** — SWE-bench Docker evaluation (400 cells, ~40 min) and the ablation table

History, bugs and retractions: [`PROGRESS.md`](PROGRESS.md). Spec: [`IDEA.md`](IDEA.md).
Working notes for agents: [`AGENTS.md`](AGENTS.md).
