"""Tests for scripts/diagnostics_e2b.py (the DIAGNOSTICS_SPEC report generator).

Two classes of mistake this pins, both of which *were* in the first draft and both of which
produce a confident-looking number that is wrong:

  * the taxonomy mapping. `bad arguments` and `search_block not found` are different failures with
    different owners (Module 1's schema target vs. localisation), and a miscategorised call moves
    a percentage in the ablation story without any trace changing.
  * the symbol comparison in the M3 compaction section. `lock_targets` stores
    `sympy/core/mod.py::Mod.doit` while `apply_edit` sends `doit`, so a raw string comparison
    reported 31/41 tasks as false localisations when the real count is 2/41.

The end-to-end test runs the generator over synthetic traces, so a change that breaks the report's
accounting (or divides by zero on a partial grid) fails here rather than in a document people cite.

Run with:  .\\.venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "diagnostics_e2b.py"
sys.path.insert(0, str(ROOT / "scripts"))

from diagnostics_e2b import sym_match, sym_parts, taxonomy  # noqa: E402


@pytest.mark.parametrize("name,status,error,expected", [
    # arguments that are valid JSON but name parameters the tool does not have (Module 1's target)
    ("search_text", "error", "bad arguments for search_text: got ['pattern=\"x\"\\npath']", "JSON_SCHEMA_ERROR"),
    # the same population when the parser could not build an object at all, e.g. a cut-off call
    ("apply_edit", "malformed_arguments", "Unterminated string starting at: line 1", "JSON_SCHEMA_ERROR"),
    # AST-Nav tools handed a symbol or path that does not exist
    ("read_symbol", "error", "no symbol named 'Thing' in psf/requests/sessions.py", "SYMBOL_NOT_FOUND"),
    ("read_symbol", "error", "not a file: psf/requests", "SYMBOL_NOT_FOUND"),
    # the Task 4 message shapes: a miss that lists the file's symbols, and an ambiguous bare name
    ("read_symbol", "error", "Symbol 'Session.sned' not found in 'requests/sessions.py'. "
                             "Available symbols in this file: ['Session', 'Session.request']",
     "SYMBOL_NOT_FOUND"),
    ("read_symbol", "error", "Symbol 'go' is ambiguous in 'two.py': 2 definitions match "
                             "['A.go', 'B.go'].", "SYMBOL_NOT_FOUND"),
    # the localisation failure, and the gate that runs after a successful match
    ("apply_edit", "error", "search_block not found in the file - copy it verbatim", "SEARCH_BLOCK_MISMATCH"),
    ("apply_edit", "error", "syntax gate: invalid syntax (line 3)", "AST_SYNTAX_REJECTED"),
    # the shell-less analogue of a non-zero exit code
    ("search_text", "error", "invalid regex: multiple repeat at position 123", "BASH_EXECUTION_ERROR"),
    ("list_dir", "error", "not a directory: psf/requests", "BASH_EXECUTION_ERROR"),
    # an edit whose path does not exist is nobody's schema bug and nobody's search miss
    ("apply_edit", "error", "not a file: sympy/sympy. Paths are relative to the repo root", "OTHER_ERROR"),
    ("finish", "ok", None, None),
])
def test_taxonomy_maps_each_failure_to_one_cause(name, status, error, expected):
    assert taxonomy(name, status, error) == expected


def test_symbol_match_sees_through_lock_targets_formatting():
    # the forms the model actually wrote, against the bare name apply_edit sends
    assert sym_match("doit", "sympy/core/mod.py::Mod.doit")
    assert sym_match("is_subset", "sympy/sets/sets.py::is_subset")
    assert sym_match("Add.flatten", "Add.flatten")
    assert sym_match("src/_pytest/unittest.py::addSkip", "src/_pytest/unittest.py::TestCaseFunction.addSkip")
    # a member of the locked class is inside the lock, not outside it
    assert sym_match("Commutator._latex", "sympy/physics/secondquant.Commutator")


def test_symbol_match_flags_a_genuinely_different_site():
    assert not sym_match("RepeatedStratifiedKFold", "sklearn/model_selection/_split.RepeatedKFold")
    assert not sym_match("LatexPrinter.__init__", "sympy/printing/latex.latex")
    assert not sym_match("Mod.eval", "sympy/core/mod.doit")
    # a path is not a symbol, so it contributes nothing to the comparison
    assert sym_parts("sympy/core/mod.py") == set()


def _trace(path: Path, instance: str, variant: str, recs: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")


def _cell(instance: str, variant: str, ts: int, *, depth_a: int, depth_b: int,
          fail_second: bool, compaction: bool) -> list[dict]:
    """One run: two turns, a second call that may fail, and an optional M3 lock."""
    cfg = {"instance_id": instance, "model": "gemma-4-E2B", "variant": variant,
           "thinking": "on", "max_turns": 15, "ctx_per_run": 32768}
    second = {"type": "tool", "ts": ts + 5, "turn": 2, "phase": "solve", "name": "apply_edit",
              "args": {"file": "a.py", "search_block": "x", "replace_block": "y",
                       "target_symbol": "f"},
              "status": "error" if fail_second else "ok",
              "error": "search_block not found in the file" if fail_second else None,
              "repeat_of_turn": None}
    recs = [
        {"type": "run_start", "ts": ts, "config": cfg},
        {"type": "turn", "ts": ts + 1, "turn": 1,
         "usage": {"prompt_tokens": depth_a}, "response": {"reasoning_chars": 10}},
        {"type": "tool", "ts": ts + 2, "turn": 1, "phase": "solve", "name": "read_symbol",
         "args": {"symbol": "f"}, "status": "ok", "error": None, "repeat_of_turn": None},
        {"type": "turn", "ts": ts + 4, "turn": 2,
         "usage": {"prompt_tokens": depth_b}, "response": {"reasoning_chars": 20},
         "tool_calls": [{"function": {"name": "lock_targets"}}] if compaction else []},
    ]
    if compaction:
        recs += [
            {"type": "tool", "ts": ts + 4.5, "turn": 2, "phase": "localize",
             "name": "lock_targets", "args": {"files": ["a.py"], "symbols": ["a.py::f"]},
             "status": "ok", "error": None, "repeat_of_turn": None},
            {"type": "compaction", "ts": ts + 4.6, "turn": 2, "files": ["a.py"],
             "symbols": ["a.py::f"], "root_cause_note": "n", "messages_after": 2,
             "context_chars_before": 500, "context_chars_after": 300},
            {"type": "turn", "ts": ts + 6, "turn": 3,
             "usage": {"prompt_tokens": 1200}, "response": {"reasoning_chars": 30}},
        ]
    recs += [
        second,
        {"type": "run_end", "ts": ts + 7, "stop_reason": "finish", "turns_used": 3,
         "total_completion_tokens": 100, "peak_prompt_tokens": depth_b},
        {"type": "artifacts", "ts": ts + 8, "patch_is_empty": False},
    ]
    return recs


def test_report_counts_the_synthetic_traces(tmp_path: Path):
    _trace(tmp_path / "a.jsonl", "t__t-1", "B0", _cell("t__t-1", "B0", 1, depth_a=1500,
                                                       depth_b=9000, fail_second=True,
                                                       compaction=False))
    _trace(tmp_path / "b.jsonl", "t__t-2", "M3", _cell("t__t-2", "M3", 10, depth_a=1500,
                                                       depth_b=3000, fail_second=False,
                                                       compaction=True))
    out = tmp_path / "report.md"
    proc = subprocess.run([sys.executable, str(SCRIPT), "--glob", str(tmp_path / "*.jsonl"),
                           "--out", str(out)], capture_output=True, text=True,
                          encoding="utf-8", cwd=ROOT)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    report = out.read_text(encoding="utf-8")

    assert "**2 cells**" in report
    # the failed call is binned by the depth of the turn that issued it, not by the run's peak
    assert "| `<=2K` | `B0` | 1 | 1 | 0 (0.0%) | 0 (0.0%) |" in report
    assert "| `8-12K` | `B0` | 1 | 1 | 1 (100.0%) | 0 (0.0%) |" in report
    # one failure, attributed once
    assert "### B0 - 1 failed call(s)" in report
    assert "| SEARCH_BLOCK_MISMATCH | 1 | 100.0% |" in report
    # M3: the lock compacted 3000 -> 1200 prompt tokens, i.e. 60%
    assert "| Compaction triggered | 1 / 1 (100.0%) |" in report
    assert "60.0%" in report
    # no M2 cell in this grid, so no M2 sub-table and no empty M2 column
    assert "which cause dominates" not in report
    # only the variants present get a column: a one-config grid must not invent empty ones
    assert "| **Valid Patch Rate (%)** (non-empty `git diff`) | 1/1 (100%) | 1/1 (100%) |" in report
