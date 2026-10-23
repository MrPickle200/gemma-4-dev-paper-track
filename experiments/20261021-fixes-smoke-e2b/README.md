# 20261021-fixes-smoke-e2b

Smoke test of the six TASK.md repairs: 5 tasks that failed hardest in the 50-task grid, all four
variants, E2B, thinking on, `max_turns=15`, local llama-server at `-c 131072 -np 4` (4 × 32,768,
probe served 124,552 tokens across 4 concurrent requests).

Tasks: `psf__requests-1724`, `scikit-learn-10908`, `scikit-learn-13142`, `scikit-learn-14087`,
`sympy__sympy-21930`.

```
run_batch.py --model gemma-4-E2B --thinking on --workers 4 --experiment 20261021-fixes-smoke-e2b \
    --instance-id psf__requests-1724 --instance-id scikit-learn__scikit-learn-10908 \
    --instance-id scikit-learn__scikit-learn-13142 --instance-id scikit-learn__scikit-learn-14087 \
    --instance-id sympy__sympy-21930
```

## Files

- `runs/` — 21 traces for 20 cells. `scikit-learn-10908 M1 __20261021-220457` has **no `run_end`**:
  it is the partial trace of a harness crash (Python 3.11's 4300-digit int limit while parsing tool
  arguments). The fix is in `harness/agent.py::_parse_args`; the cell was re-run as
  `__20261021-224342`, which finished. Diagnostics keep only the scoreable trace.
- `patches/` — final diff per cell (12 non-empty of 20; the baseline grid had 8 for the same 20).
- `manifest.json` — two invocations: the 20-cell batch and the one-cell M1 re-run.
- `results.csv` — one row per finished cell, including the re-run.
- `DIAGNOSTICS.md` — `diagnostics_e2b.py --experiment 20261021-fixes-smoke-e2b`.

## Headline numbers (same 20 cells, baseline grid → this run)

| variant | failed calls | SEARCH_BLOCK_MISMATCH | JSON_SCHEMA_ERROR | repeats (all blocked) | non-empty patch |
|---|---|---|---|---|---|
| B0 | 24 → 10 | 16 → 6 | 0 → 0 | 16 → 10 | 1/5 → 2/5 |
| M1 | 48 → 8 | 12 → 8 | 32 → 0 | 20 → 8 | 1/5 → 3/5 |
| M2 | 44 → 32 | 32 → 19 | 0 → 0 | 7 → 21 | 1/5 → 3/5 |
| M3 | 12 → 13 | 1 → 3 | 0 → 0 | 4 → 3 | 5/5 → 4/5 |

- `psf__requests-1724 M3`: `psf/requests` resolved, `Session.request` loaded, lock accepted
  (`requests/sessions.py::Session.request`, 3,737 chars), context 22,325 → 13,511 chars, one
  `apply_edit`, `finish`; no Phase-2 `get_repo_map`.
- Repeats are counted when detected; every one of the 42 was blocked, so no duplicate call executed.
  The *detection* count barely moved, so do not read this as the model looping less.
- 3 of the 4 compacted M3 cells still called `get_repo_map`/`read_symbol` in Phase 2 despite the
  directive; `scikit-learn-14087` also grew 15,135 → 16,599 chars because the locked `GaussianMixture`
  body hits the 12,000-char source cap.
