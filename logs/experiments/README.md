# Experiment bundles

`logs/runs/` and `logs/patches/` hold the committed 50-task ablation grid (the baseline behind
`E2B_DIAGNOSTICS_REPORT.md`). Every later run set gets its own directory here so evidence is never
mixed with that grid, and so a report can be regenerated per experiment.

## Layout

```
logs/experiments/<tag>/
  runs/<run_id>.jsonl     one per-turn trace per cell (same format as logs/runs/)
  patches/<run_id>.patch  the model's diff for that cell
  manifest.json           first invocation (command, model, variants, thinking, max_turns,
                          workers, ctx_per_run, task ids, planned cells, git commit) plus
                          `invocations`, one entry per resume or single-cell re-run, and the
                          union of task ids
  results.csv             one row per finished cell (cells, turns, tokens, counters, error class)
  DIAGNOSTICS.md          output of scripts/diagnostics_e2b.py --experiment <tag>
```

`<tag>` is free-form but should sort chronologically, e.g. `20261021-fixes-smoke-e2b`. It is written
by `scripts/run_batch.py --experiment <tag>`; `run_task.py --experiment <tag>` writes into the same
directory for single-cell follow-ups.

## Producing one

```powershell
.\.venv\Scripts\python.exe scripts\run_batch.py --model gemma-4-E2B --thinking on --workers 4 `
    --experiment 20261021-fixes-smoke-e2b `
    --instance-id psf__requests-1724 --instance-id scikit-learn-13142
.\.venv\Scripts\python.exe scripts\diagnostics_e2b.py --experiment 20261021-fixes-smoke-e2b
.\.venv\Scripts\python.exe scripts\summarize_batch.py --experiment 20261021-fixes-smoke-e2b
```

## Rules

- **Commit the whole bundle** (traces, patches, manifest, results, DIAGNOSTICS). AGENTS.md rule 4
  treats it exactly like `logs/runs/`: it is the record behind any number quoted from it.
- Do **not** copy experiment traces into `logs/runs/`. That grid is scored as 50 tasks × 4 variants;
  a second trace for the same `(instance_id, variant)` would change its cell count and the report.
- `--experiment` only rebinds `RUNS`/`PATCHES`; the harness is otherwise identical, so a bundle is
  directly comparable to the baseline grid. Compare error classes by *count per cell*, not raw
  count, because a smoke test has 5 tasks and the grid has 50.
- Trace line endings are LF (`.gitattributes`: `*.jsonl text eol=lf`), as are patches since
  2026-10-21; older local patch files may still be CRLF.
- A resume or re-run **merges** the manifest: `invocations` grows and `task_ids` is the union, so a
  one-cell follow-up cannot make a five-task experiment look like a one-task run.
