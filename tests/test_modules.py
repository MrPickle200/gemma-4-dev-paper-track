"""Unit tests for the Bước 3 modules (M1 syntax gate, M2 AST navigation, M3 compaction).

These need no llama-server: they drive ToolBox and run_agent directly, which is exactly
where the deterministic harness logic lives. Run with:

  .\\.venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.agent import JsonlLogger, run_agent           # noqa: E402
from harness.config import RunConfig                        # noqa: E402
from harness.llm import Completion                          # noqa: E402
from harness.tools import ToolBox                           # noqa: E402

SIMPLE = "def f():\n    return 1\n"

NAV_CODE = (
    "def helper(x):\n"
    '    """Docstring."""\n'
    "    return x + 1\n"
    "\n"
    "\n"
    "class Thing:\n"
    "    def method(self):\n"
    "        # helper is mentioned in this comment\n"
    '        s = "helper appears in a string"\n'
    "        return helper(3)\n"
)


def _tool_names(tb: ToolBox, variant: str) -> set[str]:
    return {s["function"]["name"] for s in tb.specs(variant)}


# ------------------------------------------------------------------ M1: gate
def test_syntax_gate_blocks_broken_edit(tmp_path: Path) -> None:
    src = tmp_path / "m.py"
    src.write_text(SIMPLE, encoding="utf-8")
    tb = ToolBox(tmp_path, variant="M1", repo_name="x/y")

    res = tb.execute("apply_edit", {
        "file": "m.py", "target_symbol": "f",
        "search_block": "return 1", "replace_block": "return (",
    })
    assert not res.ok and "syntax gate" in res.error
    assert src.read_text(encoding="utf-8") == SIMPLE, "nothing must be written on rejection"
    assert tb.stats["edits_syntax_rejected"] == 1
    assert tb.stats["edits_malformed"] == 1


def test_b0_has_no_gate_but_m1_accepts_good_edit(tmp_path: Path) -> None:
    src = tmp_path / "m.py"
    src.write_text(SIMPLE, encoding="utf-8")

    tb_b0 = ToolBox(tmp_path, variant="B0", repo_name="x/y")
    assert "syntax_gate" not in _tool_names(tb_b0, "B0") or True  # gate is behaviour, not a tool
    r0 = tb_b0.execute("apply_edit", {
        "file": "m.py", "search_block": "return 1", "replace_block": "return (",
    })
    assert r0.ok, "B0 writes the edit with no syntax check"
    assert "return (" in src.read_text(encoding="utf-8")

    src.write_text(SIMPLE, encoding="utf-8")
    tb_m1 = ToolBox(tmp_path, variant="M1", repo_name="x/y")
    r1 = tb_m1.execute("apply_edit", {
        "file": "m.py", "target_symbol": "f",
        "search_block": "return 1", "replace_block": "return x",
    })
    assert r1.ok and r1.meta["syntax_checked"] is True
    assert "return x" in src.read_text(encoding="utf-8")


# -------------------------------------------------------------- M2: AST nav
def test_find_references_ignores_comments_and_strings(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(NAV_CODE, encoding="utf-8")
    tb = ToolBox(tmp_path, variant="M2", repo_name="x/y")

    res = tb.execute("find_references", {"symbol_name": "helper"})
    assert res.ok
    lines = {int(line.split(":")[1]) for line in res.output.splitlines() if ":" in line}
    assert lines == {1, 10}, f"comment (8) and string (9) must not match; got {lines}"


def test_read_symbol_and_repo_map(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(NAV_CODE, encoding="utf-8")
    tb = ToolBox(tmp_path, variant="M2", repo_name="x/y")

    rs = tb.execute("read_symbol", {"file": "a.py", "symbol_name": "method"})
    assert rs.ok and "def method" in rs.output and "return helper(3)" in rs.output

    dotted = tb.execute("read_symbol", {"file": "a.py", "symbol_name": "Thing.method"})
    assert dotted.ok and "def method" in dotted.output

    missing = tb.execute("read_symbol", {"file": "a.py", "symbol_name": "nope"})
    assert not missing.ok and "Available symbols" in missing.error

    rm = tb.execute("get_repo_map", {"path": "a.py"})
    assert rm.ok
    assert "def helper" in rm.output and "class Thing" in rm.output and "def method" in rm.output
    assert "return helper(3)" not in rm.output, "bodies must be hidden in the skeleton"
    assert tb.stats["nav_calls"] == 4


def test_variant_tool_sets() -> None:
    tb = ToolBox(Path("."), variant="B0")
    assert _tool_names(tb, "B0") == {"search_text", "read_file", "list_dir", "apply_edit", "finish"}
    assert _tool_names(tb, "M1") == {"search_text", "read_file", "list_dir", "apply_edit", "finish"}
    assert _tool_names(tb, "M2") == {"get_repo_map", "read_symbol", "find_references", "apply_edit", "finish"}
    assert _tool_names(tb, "M3") == {"get_repo_map", "read_symbol", "find_references",
                                     "lock_targets", "apply_edit", "finish"}


# ------------------------------------------------------------ M3: compaction
class _FakeClient:
    def __init__(self, script: list[Completion]):
        self.script = script
        self.i = 0
        self.seen: list[list[dict]] = []

    def chat(self, messages: list[dict], tools=None) -> Completion:
        self.seen.append([dict(m) for m in messages])
        out = self.script[self.i]
        self.i += 1
        return out


def _call(name: str, args: dict, i: int = 0) -> dict:
    return {"id": f"call_{i}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def _comp(calls: list[dict]) -> Completion:
    return Completion(content="", tool_calls=calls, prompt_tokens=100,
                      completion_tokens=20, finish_reason="tool_calls")


def test_length_truncation_is_retried_not_fatal(tmp_path: Path) -> None:
    """A turn cut off by the context limit must be nudged, not treated as a decision to stop."""
    (tmp_path / "m.py").write_text(SIMPLE, encoding="utf-8")
    tb = ToolBox(tmp_path, variant="B0", repo_name="x/y")
    cfg = RunConfig(instance_id="t__t-2", variant="B0", model="fake")
    task = {"instance_id": "t__t-2", "repo": "x/y", "base_commit": "0" * 40,
            "problem_statement": "f returns the wrong value"}

    truncated = Completion(content="Let me think about", tool_calls=[], prompt_tokens=100,
                           completion_tokens=200, finish_reason="length")
    client = _FakeClient([
        truncated,
        _comp([_call("apply_edit", {"file": "m.py", "search_block": "return 1",
                                    "replace_block": "return 2"}, 0)]),
        _comp([_call("finish", {"summary": "fixed"}, 1)]),
    ])
    logger = JsonlLogger(tmp_path / "run.jsonl")
    try:
        end = run_agent(cfg, task, client, tb, logger, verbose=False)
    finally:
        logger.close()

    assert end["stop_reason"] == "finish" and end["turns_used"] == 3
    assert end["edits_attempted"] == 1
    # the nudge must appear as a user message right after the truncated assistant turn
    retry_req = client.seen[1]
    assert retry_req[-1]["role"] == "user" and "cut off" in retry_req[-1]["content"]


def test_m3_compaction_clears_history(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text(SIMPLE, encoding="utf-8")
    tb = ToolBox(tmp_path, variant="M3", repo_name="x/y")
    cfg = RunConfig(instance_id="t__t-1", variant="M3", model="fake")
    task = {"instance_id": "t__t-1", "repo": "x/y", "base_commit": "0" * 40,
            "problem_statement": "f returns the wrong value"}

    client = _FakeClient([
        _comp([_call("get_repo_map", {"path": "m.py"}, 0)]),                       # localize
        _comp([_call("lock_targets", {"files": ["m.py"], "symbols": ["m.py::f"],
                                      "root_cause_note": "off by one"}, 1)]),      # locks -> compaction
        _comp([_call("apply_edit", {"file": "m.py", "target_symbol": "f",
                                    "search_block": "return 1",
                                    "replace_block": "return 2"}, 2)]),            # patch
        _comp([_call("finish", {"summary": "fixed"}, 3)]),
    ])

    log = tmp_path / "run.jsonl"
    logger = JsonlLogger(log)
    try:
        end = run_agent(cfg, task, client, tb, logger, verbose=False)
    finally:
        logger.close()

    assert end["compacted"] is True and end["compactions"] == 1
    assert end["stop_reason"] == "finish"
    # the request right after lock_targets must be the clean Phase-2 context
    post = client.seen[2]
    assert len(post) == 2
    assert "Phase 2 - Patching" in post[1]["content"]
    assert "off by one" in post[1]["content"]
    assert "return 1" in post[1]["content"], "locked symbol source must be present verbatim"
    assert (tmp_path / "m.py").read_text(encoding="utf-8") == "def f():\n    return 2\n"

    recs = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    compact = [r for r in recs if r["type"] == "compaction"]
    assert len(compact) == 1 and compact[0]["symbols"] == ["m.py::f"]
    assert compact[0]["messages_after"] == 2
    assert compact[0]["context_chars_after"] > 0


# ------------------------------------- Bước 4 concurrency (run_batch.py --workers)
def _init_repo(path: Path) -> str:
    import subprocess

    path.mkdir(parents=True, exist_ok=True)

    def g(*args: str):
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True)

    g("init", "-q")
    (path / "f.py").write_text("x = 1\n", encoding="utf-8")
    g("add", ".")
    g("-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "init")
    return g("rev-parse", "HEAD").stdout.strip()


def test_worktrees_are_isolated_per_variant(tmp_path: Path, monkeypatch) -> None:
    """--workers runs several variants of one task at once: they must not share a checkout.

    Keying the worktree path by instance_id alone (the pre-fix behaviour) let one variant tear down
    another's working tree, which corrupted the collected patches."""
    import harness.runner as runner

    repos = tmp_path / "repos"
    base = _init_repo(repos / "x__y")
    monkeypatch.setattr(runner, "REPOS", repos)
    monkeypatch.setattr(runner, "WORKTREES", tmp_path / "worktrees")

    task = {"instance_id": "x__y-1", "repo": "x/y", "base_commit": base}
    variants = ["B0", "M1", "M2", "M3"]
    dirs = {v: runner.setup_worktree(task, v) for v in variants}

    assert len({str(d) for d in dirs.values()}) == len(variants), "paths must be distinct"
    for v, d in dirs.items():
        assert d.exists() and d.name == f"x__y-1__{v}"

    # a write in one variant's checkout must not appear in another's, nor in the clone
    (dirs["B0"] / "f.py").write_text("x = 99\n", encoding="utf-8")
    for v in ("M1", "M2", "M3"):
        assert (dirs[v] / "f.py").read_text(encoding="utf-8") == "x = 1\n"
    assert (repos / "x__y" / "f.py").read_text(encoding="utf-8") == "x = 1\n"

    for v in variants:
        runner.teardown_worktree(task, v)


def test_bad_arguments_are_a_malformed_call_not_a_tool_error(tmp_path: Path) -> None:
    """Arguments that do not match the tool's schema must be named as such, and answered usefully.

    The grid's M1 cells produced 83 such calls: under constrained decoding E2B writes Gemma 4's
    template tokens into the argument string, so valid JSON arrives with keys like
    ``pattern="x"\\npath``. The old dispatch let Python's TypeError escape
    (``ToolBox._search_text() got an unexpected keyword argument``), which both mis-filed the
    failure under metric #2 and told the model nothing it could act on.

    Since TASK.md Task 2 the harness first tries to *recover* the call with a regex fallback (the
    schema now forbids the shape at generation time, this is the belt to that suspenders), and
    only reports an invalid-key error when the glued names are not parameters of the tool.
    """
    (tmp_path / "y").write_text("x = 1\n", encoding="utf-8")
    tb = ToolBox(tmp_path, variant="M1", repo_name="x/y")

    res = tb.execute("search_text", {"pattern=\"x\"\npath": "y"})
    assert res.ok, "the glued key must be repaired, not rejected"
    assert "x = 1" in res.output
    assert tb.stats["arguments_repaired"] == 1
    assert tb.stats["malformed_arguments"] == 0

    # not a parameter of this tool -> no safe recovery, and the message must name the keys
    bad = tb.execute("search_text", {"pattern=\"x\"\nbogus": "y"})
    assert not bad.ok
    assert "Invalid JSON keys" in bad.error and "pattern" in bad.error
    assert "_search_text" not in bad.error, "our Python signature must not leak to the model"
    assert "unexpected keyword" not in bad.error
    assert tb.stats["malformed_arguments"] == 1


def test_tree_sitter_parser_is_per_thread() -> None:
    """A shared tree-sitter Parser is not thread-safe, so each worker thread needs its own."""
    import threading

    from harness import astnav

    main = astnav.parser()
    seen: dict = {}
    t = threading.Thread(target=lambda: seen.__setitem__("worker", astnav.parser()))
    t.start()
    t.join()
    assert seen["worker"] is not main
