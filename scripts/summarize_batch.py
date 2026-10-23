"""Aggregate every run in experiments/<tag>/runs/*.jsonl into an ablation table.

Seeds the Bước 5 table in IDEA.md §4: per (model, variant, thinking) group it reports the
run count, how many runs produced an edit, malformed edit rate, total and peak context
tokens, and loop/repeat behaviour. Resolution rate needs the Docker evaluator and is
reported later, not here.

Usage:
  .\\.venv\\Scripts\\python.exe scripts\\summarize_batch.py
  .\\.venv\\Scripts\\python.exe scripts\\summarize_batch.py --experiment 20261023-e2b-grid
  .\\.venv\\Scripts\\python.exe scripts\\summarize_batch.py --experiment 20261023-e2b-grid --csv experiments\\20261023-e2b-grid\\results.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import config  # noqa: E402
from harness.config import (is_infra_failure, is_proxy_capped,   # noqa: E402
                            model_side_stop_reason, refined_stop_reason, turn_error)


def load(path: str) -> tuple[dict, dict, list[dict]]:
    recs = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    start = next((r for r in recs if r["type"] == "run_start"), {})
    end = next((r for r in recs if r["type"] == "run_end"), {})
    return start, end, recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=str(config.RUNS / "*.jsonl"))
    ap.add_argument("--experiment", help="read experiments/<TAG>/runs/*.jsonl instead")
    ap.add_argument("--csv", help="also write the table to this CSV path")
    ap.add_argument("--thinking", choices=["on", "off", "auto"],
                    help="only runs with this thinking setting (the grid's other settings are probes)")
    ap.add_argument("--max-turns", type=int, help="only runs with this turn budget")
    args = ap.parse_args()
    if args.experiment:
        args.glob = str(config.EXPERIMENTS / args.experiment / "runs" / "*.jsonl")

    groups: dict[tuple, list[tuple]] = defaultdict(list)
    # Excluded/starved are counted in *traces* (every trace that should not have ended that way,
    # even when its cell was re-run later); `missing` is counted in *cells*, which is what decides
    # whether any work is left. Conflating the two reads as "32 runs still to re-run" after the
    # grid is in fact complete.
    excluded = 0
    starved = 0
    superseded = 0
    capped = 0
    valid_cells: set[tuple] = set()      # (instance, model, variant, thinking) with a scoreable trace
    bad_cells: dict[tuple, bool] = {}    # ... with only excluded/crashed traces; True if starved
    # One trace per *cell* - a cell being (task, variant, config). A re-run supersedes the
    # earlier trace instead of adding to it; otherwise re-running a cell would double-count it.
    entries: dict[tuple, tuple] = {}
    for path in sorted(glob.glob(args.glob)):
        start, end, recs = load(path)
        cfg = start.get("config", {})
        if args.thinking and cfg.get("thinking") != args.thinking:
            continue
        if args.max_turns is not None and cfg.get("max_turns") != args.max_turns:
            continue
        cell_id = (cfg.get("instance_id"), cfg.get("model"), cfg.get("variant"), cfg.get("thinking"))
        if not end:
            bad_cells.setdefault(cell_id, False)   # crashed mid-run: no run_end to score
            continue
        err = turn_error(recs)
        # `llm_error` alone is not enough: a run the MODEL ended by overflowing its context or
        # emitting unparsable tool JSON is a result to score (Modules 1 and 3) - but only if the
        # backend really gave the run the window it asked for, or the pool ends it instead.
        if is_infra_failure(end.get("stop_reason"), err, cfg.get("ctx_per_run")):
            excluded += 1  # backend/harness failure, not an experimental outcome
            bad_cells[cell_id] = bad_cells.get(cell_id, False) or (
                model_side_stop_reason(err) == "context_exceeded")
            if model_side_stop_reason(err) == "context_exceeded":
                starved += 1  # the server's shared pool cut it short; re-run with a bigger -c
            continue
        # Runs predating a config field must show up as "n/a", not silently as False:
        # the first run used line-numbered read_file before the flag existed.
        key = (cfg.get("model"), cfg.get("variant"), cfg.get("thinking"),
               cfg.get("line_numbers"), cfg.get("max_turns"))
        cell = (cfg.get("instance_id"), key)
        prev = entries.get(cell)
        if prev is not None:
            superseded += 1
            if prev[1].get("ts", 0) >= end.get("ts", 0):
                continue                       # the trace already held is the newer one
        # Derive the no-op edit count from the trace itself (identical search/replace), so it
        # works for runs made before the counter existed in run_end.
        apply_calls = [r for r in recs if r["type"] == "tool" and r.get("name") == "apply_edit"]
        noop = sum(1 for r in apply_calls
                   if (r.get("args") or {}).get("search_block") == (r.get("args") or {}).get("replace_block"))
        entries[cell] = (key, end, len(apply_calls), noop, is_proxy_capped(cfg),
                         refined_stop_reason(end.get("stop_reason"), err, cfg.get("ctx_per_run")))
        valid_cells.add(cell_id)

    missing = sorted(c for c in bad_cells if c not in valid_cells)      # cells still to re-run
    starved_left = sum(1 for c in missing if bad_cells[c])
    missing_note = (f"note: {len(missing)} cell(s) have NO valid trace ({starved_left} of them starved "
                    f"by the shared KV pool) and must re-run against -c >= 32768 * workers.")
    for key, end, n_apply, noop, was_capped, eff in entries.values():
        groups[key].append((end, n_apply, noop, eff))
        capped += 1 if was_capped else 0

    if not groups:
        print("no completed runs found")
        if missing:      # every trace was an infrastructure failure: name the cells to re-run
            print(missing_note)
        return 1

    header = ("model", "var", "think", "ln", "runs", "edited", "malf_rate",
              "mean_tok", "mean_peak", "mean_turns", "finish", "ctxover", "repeats")
    rows = []
    for key, pairs in sorted(groups.items(),
                            key=lambda kv: tuple("~" if x is None else str(x) for x in kv[0])):
        model, variant, thinking, ln, max_turns = key
        ends = [e for e, _, _, _ in pairs]
        attempted = sum(e.get("edits_attempted", 0) for e in ends)
        malformed = sum(e.get("edits_malformed", 0) for e in ends)
        edited = sum(1 for e in ends if e.get("edits_attempted", 0) > 0)
        real_edited = sum(1 for _, a, n, _ in pairs if a - n > 0)
        noop = sum(n for _, _, n, _ in pairs)
        rows.append({
            "model": model, "variant": variant, "thinking": thinking, "line_numbers": ln,
            "runs": len(ends), "runs_with_edit": edited, "runs_real_edit": real_edited,
            "noop_edits": noop,
            "malformed_edit_rate": round(malformed / attempted, 3) if attempted else None,
            "mean_total_tokens": round(statistics.mean(e.get("total_tokens", 0) for e in ends)),
            "mean_peak_context": round(statistics.mean(e.get("peak_prompt_tokens", 0) for e in ends)),
            "mean_turns": round(statistics.mean(e.get("turns_used", 0) for e in ends), 1),
            "finished": sum(1 for _, _, _, eff in pairs if eff == "finish"),
            "ctx_over": sum(1 for _, _, _, eff in pairs if eff == "context_exceeded"),
            "syntax_rejected": sum(e.get("edits_syntax_rejected", 0) for e in ends),
            "compactions": sum(e.get("compactions", 0) for e in ends),
            "repeated_identical_calls": sum(e.get("repeated_identical_calls", 0) for e in ends),
        })

    widths = {"model": 12, "variant": 4, "thinking": 6, "line_numbers": 6, "runs": 4,
              "runs_with_edit": 6, "runs_real_edit": 6, "noop_edits": 5, "malformed_edit_rate": 8,
              "mean_total_tokens": 10, "mean_peak_context": 9, "mean_turns": 6, "finished": 6,
              "ctx_over": 7, "syntax_rejected": 7, "compactions": 6,
              "repeated_identical_calls": 7}
    title = {"model": "model", "variant": "var", "thinking": "think", "line_numbers": "ln",
             "runs": "runs", "runs_with_edit": "edited", "runs_real_edit": "realed",
             "noop_edits": "noop", "malformed_edit_rate": "malf_rate",
             "mean_total_tokens": "mean_tok", "mean_peak_context": "mean_peak",
             "mean_turns": "turns", "finished": "finish", "ctx_over": "ctxover",
             "syntax_rejected": "synrej",
             "compactions": "compct", "repeated_identical_calls": "repeats"}
    print(" ".join(f"{title[k]:<{widths[k]}}" for k in widths))
    for r in rows:
        print(" ".join(
            f"{('n/a' if r[k] is None else str(r[k])):<{widths[k]}}" for k in widths
        ))

    total_runs = sum(r["runs"] for r in rows)
    total_edited = sum(r["runs_with_edit"] for r in rows)
    print(f"\nruns={total_runs}  runs that produced >=1 edit: {total_edited}  "
          f"(-> {total_edited}/{total_runs} = {total_edited/total_runs:.0%})")
    print("note: 'edited' counts runs that reached apply_edit, not runs that resolve the issue.")
    print("note: 'realed' counts runs whose edit actually changed text; 'noop' are identical")
    print("      search/replace edits the model issued without changing anything.")
    print("note: 'ctxover' counts runs the model ended by outgrowing the context window - a real")
    print("      result for Module 3 (metric #3), not an infrastructure failure. Runs the backend")
    print("      killed (transport/proxy) are excluded instead, and re-run.")
    if starved:
        print(f"note: {starved} trace(s) hit 'Context size has been exceeded' on a server whose "
              f"shared KV pool was smaller than pool/workers - that ends a run for a reason we chose,")
        print(f"      not one the model chose. See AGENTS.md.")
    if missing:
        print(missing_note)
    elif excluded:
        print(f"note: all {excluded} excluded trace(s) are superseded by a valid re-run of the same "
              f"cell - no cell is missing, nothing left to re-run on that account.")
    print("note: resolve rate requires the SWE-bench Docker evaluator (Bước 5).")
    if superseded:
        print(f"note: {superseded} superseded trace(s) ignored - one trace per cell (newest wins).")
    if capped:
        print(f"note: {capped} of {total_runs} cell(s) ran BUFFERED over a tunnel, so the proxy's read "
              f"timeout may have dropped their long turns (asymmetric sample; see PROGRESS.md).")

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.csv).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
