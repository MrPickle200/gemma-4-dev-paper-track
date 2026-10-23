# Experiments

Every run set lives under `experiments/<tag>/`. The first level is the experiment name (free-form
but chronological, e.g. `20261023-e2b-grid`); the second level is everything that belongs to that
experiment. Nothing is shared between experiments, so a report can be regenerated from one
directory and evidence is never mixed.

## Layout

```
experiments/<tag>/
  runs/<run_id>.jsonl            one per-turn trace per cell
  patches/<run_id>.patch         the model's diff for that cell
  manifest.json                  first invocation (command, model, variants, thinking, max_turns,
                                 workers, ctx_per_run, task ids, planned cells, git commit) plus
                                 `invocations`, one entry per resume or single-cell re-run, and the
                                 union of task ids
  results.csv                    one row per finished cell (turns, tokens, counters, error class),
                                 written by scripts/summarize_batch.py
  DIAGNOSTICS.md                 scripts/diagnostics_e2b.py --experiment <tag>
  predictions/<model>_<var>.jsonl  Bước 5 input: one file per (model, variant), 50 unique ids
  reports/<model>.<run_id>.json      Bước 5 output: resolved/unresolved/ambiguous ids per cell
  swebench_results.csv           resolved / unresolved / empty-patch counts per cell
  run_evaluation/<run_id>/       swebench's per-instance Docker logs (gitignored, regenerable)
```

`run_evaluation/` is scratch: `scripts/run_swebench_eval.py` moves it out of swebench's hardcoded
`logs/run_evaluation/<run_id>` into the bundle after each cell, and `--force` reproduces it.

## Producing one

```powershell
# Bước 2 — traces and patches
.\.venv\Scripts\python.exe scripts\run_batch.py --model gemma-4-E2B --thinking on --workers 4 `
    --experiment 20261023-e2b-grid `
    --instance-id psf__requests-1724 --instance-id scikit-learn-13142
.\.venv\Scripts\python.exe scripts\summarize_batch.py --experiment 20261023-e2b-grid `
    --csv experiments\20261023-e2b-grid\results.csv
.\.venv\Scripts\python.exe scripts\diagnostics_e2b.py --experiment 20261023-e2b-grid

# Bước 5 — SWE-bench resolve rate
.\.venv\Scripts\python.exe scripts\build_predictions.py --experiment 20261023-e2b-grid
.\.venv\Scripts\python.exe scripts\run_swebench_eval.py --experiment 20261023-e2b-grid --max-workers 4
```

## Rules

- **Commit the whole bundle** except `run_evaluation/`: traces, patches, manifest, results,
  DIAGNOSTICS, predictions/reports and the CSVs are the record behind every quoted number.
- A script run **without** `--experiment` falls back to `experiments/_default/`, which is scratch.
  Named experiments are the only things cited in the report.
- Do **not** copy an experiment's traces into another experiment's `runs/`; a second trace for the
  same `(instance_id, variant)` changes that cell's count and its report.
- `--experiment` only rebinds `EXPERIMENTS`/`RUNS`/`PATCHES`; the harness is otherwise identical,
  so bundles are directly comparable. Compare error classes by *count per cell*, not raw count.
- Trace line endings are LF (`.gitattributes`: `*.jsonl text eol=lf`), as are patches since
  2026-10-21; older local patch files may still be CRLF.
- A resume or re-run **merges** the manifest: `invocations` grows and `task_ids` is the union, so a
  one-cell follow-up cannot make a five-task experiment look like a one-task run.
