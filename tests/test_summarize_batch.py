"""Tests for the cell accounting in summarize_batch.py.

The table reports two different populations and conflating them misleads: `starved` counts
*trace*s (every run the shared KV pool cut short, even when the cell was re-run afterwards, so
the count stays >0 on a complete grid), while what decides whether work is left is the number of
*cell*s with no scoreable trace. The 200-cell E2B grid is finished with `starved = 32`, so this
distinction is the difference between "done" and "32 runs to go".

Run with:  .\\.venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUMMARIZE = ROOT / "scripts" / "summarize_batch.py"


def _trace(path: Path, instance: str, variant: str, *, stop: str, error: str = "",
           ctx_per_run: int | None = 32768, ts: int = 1) -> None:
    """Write a minimal one-run trace: run_start + one turn + run_end."""
    cfg = {"instance_id": instance, "model": "gemma-4-E2B", "variant": variant,
           "thinking": "on", "line_numbers": False, "max_turns": 15, "ctx_per_run": ctx_per_run}
    recs = [{"type": "run_start", "ts": ts, "config": cfg},
            {"type": "turn", "ts": ts + 1, "error": error},
            {"type": "run_end", "ts": ts + 2, "stop_reason": stop, "turns_used": 1,
             "total_tokens": 10, "peak_prompt_tokens": 10, "edits_attempted": 0}]
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")


def _summarize(pattern: Path, rc: int = 0) -> str:
    proc = subprocess.run([sys.executable, str(SUMMARIZE), "--glob", str(pattern)],
                          capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    assert proc.returncode == rc, proc.stderr or proc.stdout
    return proc.stdout


STARVED = "Context size has been exceeded"   # metric #3 marker: the pool, not the model


def test_starved_cell_without_a_rerun_is_reported_as_missing(tmp_path: Path) -> None:
    _trace(tmp_path / "a.jsonl", "t1", "B0", stop="llm_error", error=STARVED, ctx_per_run=None)
    out = _summarize(tmp_path / "*.jsonl", rc=1)
    assert "no completed runs found" in out, "a starved trace is not a result and must not be scored"
    assert "1 cell(s) have NO valid trace (1 of them starved" in out


def test_rerun_supersedes_the_starved_trace_and_clears_the_cell(tmp_path: Path) -> None:
    # The real shape of the finished grid: the starved trace is still counted as a trace, but its
    # cell has a newer valid trace, so nothing is missing.
    _trace(tmp_path / "starved.jsonl", "t1", "B0", stop="llm_error", error=STARVED,
           ctx_per_run=None, ts=1)
    _trace(tmp_path / "rerun.jsonl", "t1", "B0", stop="finish", ts=9)
    out = _summarize(tmp_path / "*.jsonl")
    assert "runs=1" in out
    assert "1 trace(s) hit 'Context size has been exceeded'" in out
    assert "superseded by a valid re-run" in out
    assert "NO valid trace" not in out


def test_toolcall_rejection_is_a_result_when_the_run_had_its_window(tmp_path: Path) -> None:
    # Module 1's target: the model emitted unparsable tool JSON. With a full window that is an
    # outcome to score, not a cell to re-run.
    _trace(tmp_path / "a.jsonl", "t1", "M1", stop="llm_error",
           error="Failed to parse tool call arguments")
    out = _summarize(tmp_path / "*.jsonl")
    assert "runs=1" in out and "M1  " in out
    assert "NO valid trace" not in out
