"""Aggregate every run in logs/runs/*.jsonl into an ablation table.

Seeds the Bước 5 table in IDEA.md §4: per (model, variant, thinking) group it reports the
run count, how many runs produced an edit, malformed edit rate, total and peak context
tokens, and loop/repeat behaviour. Resolution rate needs the Docker evaluator and is
reported later, not here.

Usage:
  .\\.venv\\Scripts\\python.exe scripts\\summarize_batch.py
  .\\.venv\\Scripts\\python.exe scripts\\summarize_batch.py --csv logs\\batch.csv
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


def load(path: str) -> tuple[dict, dict, list[dict]]:
    recs = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    start = next((r for r in recs if r["type"] == "run_start"), {})
    end = next((r for r in recs if r["type"] == "run_end"), {})
    return start, end, recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="logs/runs/*.jsonl")
    ap.add_argument("--csv", help="also write the table to this CSV path")
    args = ap.parse_args()

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for path in sorted(glob.glob(args.glob)):
        start, end, _ = load(path)
        if not end:
            continue
        cfg = start.get("config", {})
        # Runs predating a config field must show up as "n/a", not silently as False:
        # the first run used line-numbered read_file before the flag existed.
        key = (cfg.get("model"), cfg.get("variant"), cfg.get("thinking"),
               cfg.get("line_numbers"), cfg.get("max_turns"))
        groups[key].append(end)

    if not groups:
        print("no completed runs found")
        return 1

    header = ("model", "var", "think", "ln", "runs", "edited", "malf_rate",
              "mean_tok", "mean_peak", "mean_turns", "finish", "repeats")
    rows = []
    for key, ends in sorted(groups.items(),
                            key=lambda kv: tuple("~" if x is None else str(x) for x in kv[0])):
        model, variant, thinking, ln, max_turns = key
        attempted = sum(e.get("edits_attempted", 0) for e in ends)
        malformed = sum(e.get("edits_malformed", 0) for e in ends)
        edited = sum(1 for e in ends if e.get("edits_attempted", 0) > 0)
        rows.append({
            "model": model, "variant": variant, "thinking": thinking, "line_numbers": ln,
            "runs": len(ends), "runs_with_edit": edited,
            "malformed_edit_rate": round(malformed / attempted, 3) if attempted else None,
            "mean_total_tokens": round(statistics.mean(e.get("total_tokens", 0) for e in ends)),
            "mean_peak_context": round(statistics.mean(e.get("peak_prompt_tokens", 0) for e in ends)),
            "mean_turns": round(statistics.mean(e.get("turns_used", 0) for e in ends), 1),
            "finished": sum(1 for e in ends if e.get("stop_reason") == "finish"),
            "repeated_identical_calls": sum(e.get("repeated_identical_calls", 0) for e in ends),
        })

    widths = {"model": 12, "variant": 4, "thinking": 6, "line_numbers": 6, "runs": 4,
              "runs_with_edit": 6, "malformed_edit_rate": 8, "mean_total_tokens": 10,
              "mean_peak_context": 9, "mean_turns": 6, "finished": 6, "repeated_identical_calls": 7}
    title = {"model": "model", "variant": "var", "thinking": "think", "line_numbers": "ln",
             "runs": "runs", "runs_with_edit": "edited", "malformed_edit_rate": "malf_rate",
             "mean_total_tokens": "mean_tok", "mean_peak_context": "mean_peak",
             "mean_turns": "turns", "finished": "finish", "repeated_identical_calls": "repeats"}
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
    print("note: resolve rate requires the SWE-bench Docker evaluator (Bước 5).")

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
