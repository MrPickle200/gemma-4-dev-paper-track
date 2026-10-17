"""Bước 4 batch progress snapshot: grid cells done per variant, plus server health.

Used both by the 15-minute trigger and for manual checks. Prints one block; with --append it
also appends that block (timestamped) to logs/batch_progress.log, so the trail survives even if
the console is gone.

Usage:
  .\\.venv\\Scripts\\python.exe scripts\\batch_progress.py                 # snapshot now
  .\\.venv\\Scripts\\python.exe scripts\\batch_progress.py --append        # also append to the log
  .\\.venv\\Scripts\\python.exe scripts\\batch_progress.py --delay 900     # wait, then snapshot (timer)
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.config import is_infra_failure, turn_error   # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

SUBSET = ROOT / "data" / "subset_50.json"
PROGRESS_LOG = ROOT / "logs" / "batch_progress.log"


def _matches(cfg: dict, model: str, thinking: str, max_turns: int) -> bool:
    return (cfg.get("model") == model and cfg.get("thinking") == thinking
            and cfg.get("max_turns") == max_turns and cfg.get("line_numbers") is False)


def grid_stats(model: str, variants: list[str], thinking: str, max_turns: int) -> dict:
    """Per-variant counts over **cells**, not traces: one trace per (task, variant), newest wins.

    This is the same rule the batch itself uses to decide what still needs running, so the
    "N/200 done" line means what it says - counting raw traces reported 200/200 while 85 cells
    were still outstanding.
    """
    best: dict[tuple, tuple] = {}
    for path in glob.glob(str(ROOT / "logs" / "runs" / "*.jsonl")):
        try:
            recs = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, json.JSONDecodeError):
            continue
        start = next((r for r in recs if r["type"] == "run_start"), None)
        end = next((r for r in recs if r["type"] == "run_end"), None)
        if not start or not end:
            continue
        cfg = start.get("config", {})
        if not _matches(cfg, model, thinking, max_turns) or cfg.get("variant") not in variants:
            continue
        if is_infra_failure(end.get("stop_reason"), turn_error(recs), cfg.get("ctx_per_run")):
            continue  # backend died, or starved it of context: the cell still needs a real run
        cell = (cfg["variant"], cfg.get("instance_id"))
        prev = best.get(cell)
        if prev is None or prev[0].get("ts", 0) <= end.get("ts", 0):
            best[cell] = (end, recs)

    rows = {v: {"done": 0, "finish": 0, "edit": 0, "real": 0, "noop": 0, "tok": 0, "peak": 0}
            for v in variants}
    for (variant, _), (end, recs) in best.items():
        apply_calls = [r for r in recs if r["type"] == "tool" and r.get("name") == "apply_edit"]
        noop = sum(1 for r in apply_calls
                   if (r.get("args") or {}).get("search_block") == (r.get("args") or {}).get("replace_block"))
        d = rows[variant]
        d["done"] += 1
        d["finish"] += end.get("stop_reason") == "finish"
        d["edit"] += end.get("edits_attempted", 0) > 0
        d["real"] += (len(apply_calls) - noop) > 0
        d["noop"] += noop
        d["tok"] += end.get("total_tokens", 0)
        d["peak"] += end.get("peak_prompt_tokens", 0)
    return rows


def server_state(base_url: str) -> str:
    try:
        r = requests.get(f"{base_url.rstrip('/')}/health", timeout=8)
        return "ok" if r.status_code == 200 else f"HTTP {r.status_code}"
    except requests.RequestException:
        return "DOWN"


def build_report(args) -> str:
    n_tasks = len(json.loads(SUBSET.read_text(encoding="utf-8")))
    cells = n_tasks * len(args.variants)
    rows = grid_stats(args.model, args.variants, args.thinking, args.max_turns)
    done = sum(r["done"] for r in rows.values())

    lines = [
        f"=== Bước 4 batch progress  {time.strftime('%Y-%m-%d %H:%M:%S')} ===",
        f"model={args.model} thinking={args.thinking} max_turns={args.max_turns}",
        f"server={server_state(args.base_url)}   ({args.base_url})",
        f"grid {done}/{cells} cells done ({cells - done} left)",
        f"{'var':4} {'done':>4} {'finish':>6} {'edit':>4} {'real':>4} {'noop':>4} {'mean_tok':>8} {'mean_peak':>9}",
    ]
    for v in args.variants:
        d = rows[v]
        n = d["done"] or 1
        lines.append(f"{v:4} {d['done']:>4} {d['finish']:>6} {d['edit']:>4} {d['real']:>4} {d['noop']:>4} "
                     f"{round(d['tok'] / n):>8} {round(d['peak'] / n):>9}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-4-E2B")
    ap.add_argument("--variants", nargs="+", default=["B0", "M1", "M2", "M3"])
    ap.add_argument("--thinking", default="on")
    ap.add_argument("--max-turns", type=int, default=15)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--delay", type=int, default=0, help="seconds to wait before reporting (timer mode)")
    ap.add_argument("--append", action="store_true", help="also append the report to logs/batch_progress.log")
    args = ap.parse_args()

    if args.delay > 0:
        time.sleep(args.delay)
    report = build_report(args)
    print(report)
    if args.append:
        PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with PROGRESS_LOG.open("a", encoding="utf-8") as fh:
            fh.write(report + "\n\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
