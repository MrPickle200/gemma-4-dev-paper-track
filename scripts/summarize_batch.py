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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.config import is_infra_failure   # noqa: E402


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

    groups: dict[tuple, list[tuple]] = defaultdict(list)
    excluded = 0
    for path in sorted(glob.glob(args.glob)):
        start, end, recs = load(path)
        if not end:
            continue
        if is_infra_failure(end.get("stop_reason")):
            excluded += 1  # backend/harness failure, not an experimental outcome
            continue
        cfg = start.get("config", {})
        # Runs predating a config field must show up as "n/a", not silently as False:
        # the first run used line-numbered read_file before the flag existed.
        key = (cfg.get("model"), cfg.get("variant"), cfg.get("thinking"),
               cfg.get("line_numbers"), cfg.get("max_turns"))
        # Derive the no-op edit count from the trace itself (identical search/replace), so it
        # works for runs made before the counter existed in run_end.
        apply_calls = [r for r in recs if r["type"] == "tool" and r.get("name") == "apply_edit"]
        noop = sum(1 for r in apply_calls
                   if (r.get("args") or {}).get("search_block") == (r.get("args") or {}).get("replace_block"))
        groups[key].append((end, len(apply_calls), noop))

    if not groups:
        print("no completed runs found")
        return 1

    header = ("model", "var", "think", "ln", "runs", "edited", "malf_rate",
              "mean_tok", "mean_peak", "mean_turns", "finish", "repeats")
    rows = []
    for key, pairs in sorted(groups.items(),
                            key=lambda kv: tuple("~" if x is None else str(x) for x in kv[0])):
        model, variant, thinking, ln, max_turns = key
        ends = [e for e, _, _ in pairs]
        attempted = sum(e.get("edits_attempted", 0) for e in ends)
        malformed = sum(e.get("edits_malformed", 0) for e in ends)
        edited = sum(1 for e in ends if e.get("edits_attempted", 0) > 0)
        real_edited = sum(1 for _, a, n in pairs if a - n > 0)
        noop = sum(n for _, _, n in pairs)
        rows.append({
            "model": model, "variant": variant, "thinking": thinking, "line_numbers": ln,
            "runs": len(ends), "runs_with_edit": edited, "runs_real_edit": real_edited,
            "noop_edits": noop,
            "malformed_edit_rate": round(malformed / attempted, 3) if attempted else None,
            "mean_total_tokens": round(statistics.mean(e.get("total_tokens", 0) for e in ends)),
            "mean_peak_context": round(statistics.mean(e.get("peak_prompt_tokens", 0) for e in ends)),
            "mean_turns": round(statistics.mean(e.get("turns_used", 0) for e in ends), 1),
            "finished": sum(1 for e in ends if e.get("stop_reason") == "finish"),
            "syntax_rejected": sum(e.get("edits_syntax_rejected", 0) for e in ends),
            "compactions": sum(e.get("compactions", 0) for e in ends),
            "repeated_identical_calls": sum(e.get("repeated_identical_calls", 0) for e in ends),
        })

    widths = {"model": 12, "variant": 4, "thinking": 6, "line_numbers": 6, "runs": 4,
              "runs_with_edit": 6, "runs_real_edit": 6, "noop_edits": 5, "malformed_edit_rate": 8,
              "mean_total_tokens": 10, "mean_peak_context": 9, "mean_turns": 6, "finished": 6,
              "syntax_rejected": 7, "compactions": 6, "repeated_identical_calls": 7}
    title = {"model": "model", "variant": "var", "thinking": "think", "line_numbers": "ln",
             "runs": "runs", "runs_with_edit": "edited", "runs_real_edit": "realed",
             "noop_edits": "noop", "malformed_edit_rate": "malf_rate",
             "mean_total_tokens": "mean_tok", "mean_peak_context": "mean_peak",
             "mean_turns": "turns", "finished": "finish", "syntax_rejected": "synrej",
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
    print("note: resolve rate requires the SWE-bench Docker evaluator (Bước 5).")
    if excluded:
        print(f"note: {excluded} run(s) excluded - they ended on a backend/harness error and will be re-run.")

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
