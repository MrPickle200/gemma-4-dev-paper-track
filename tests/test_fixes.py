"""Regression tests for the five harness fixes in TASK.md (2026-10-21).

Each test pins one cause that was measured in `E2B_DIAGNOSTICS_REPORT.md` and is now handled by the
harness instead of being paid for by the model:

  1. `resolve_repo_path`  - 18 `not a file` calls from owner/repo-prefixed paths, and the M3 layout
                            that could not load its locked symbol into Phase 2.
  2. schema + parser      - 83 `JSON_SCHEMA_ERROR` calls whose single key was `pattern="x"\\npath`.
  3. three-tier matching  - 49/97 B0 failures were `SEARCH_BLOCK_MISMATCH`, mostly CRLF escapes.
  4. symbol references    - 79 (M2) / 84 (M3) `SYMBOL_NOT_FOUND` calls with no way to self-correct.
  5. compaction gate and  - 23/41 M3 tasks re-navigated after the lock, and 38.9% of deep calls
     loop breaker           were byte-identical repeats of the previous turn.

Run with:  .\\.venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from harness.agent import PHASE2_DIRECTIVE, JsonlLogger, _parse_args, run_agent   # noqa: E402
from harness.config import RunConfig                                   # noqa: E402
from harness.llm import Completion                                     # noqa: E402
from harness.tools import PathError, ToolBox, resolve_repo_path        # noqa: E402

REQUESTS = (
    "# sessions.py\n"
    "class Session:\n"
    "    def request(self, method, url):\n"
    "        return method.upper()\n"
    "\n"
    "    def send(self, request):\n"
    "        return request\n"
)


def _repo(tmp_path: Path) -> ToolBox:
    (tmp_path / "requests").mkdir(exist_ok=True)
    (tmp_path / "requests" / "sessions.py").write_text(REQUESTS, encoding="utf-8")
    return ToolBox(tmp_path, variant="M3", repo_name="psf/requests")


# ------------------------------------------------------------------ Task 1: paths
def test_resolve_repo_path_accepts_the_forms_the_model_wrote(tmp_path: Path) -> None:
    _repo(tmp_path)
    root = tmp_path

    assert resolve_repo_path(root, "requests/sessions.py") == root / "requests/sessions.py"
    # owner/repo prefix, and a redundant repeated repo name
    assert resolve_repo_path(root, "psf/requests/sessions.py") == root / "requests/sessions.py"
    assert resolve_repo_path(root, "requests/requests/sessions.py") == root / "requests/sessions.py"
    # leading whitespace and slashes are stripped
    assert resolve_repo_path(root, "  /requests/sessions.py ") == root / "requests/sessions.py"
    # any unique suffix resolves (the 'sympy/sympy/core/matrix.py' family)
    assert resolve_repo_path(root, "requests/sessions.py") == root / "requests/sessions.py"
    assert resolve_repo_path(root, ".") == root


def test_resolve_repo_path_explains_a_failure(tmp_path: Path) -> None:
    _repo(tmp_path)
    with pytest.raises(PathError) as miss:
        resolve_repo_path(tmp_path, "requests/session.py")
    assert "not a file" in str(miss.value)
    assert "requests/sessions.py" in str(miss.value), "the closest path must be offered"

    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "sessions.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(PathError) as ambiguous:
        resolve_repo_path(tmp_path, "sessions.py")
    assert "ambiguous" in str(ambiguous.value)


def test_every_path_taking_tool_resolves_the_same_way(tmp_path: Path) -> None:
    tb = _repo(tmp_path)
    prefixed = "psf/requests/sessions.py"

    assert isinstance(tb.execute("read_file", {"file": prefixed}).output, str)
    assert tb.execute("read_file", {"file": prefixed}).ok
    assert tb.execute("list_dir", {"path": "psf/requests"}).ok
    assert tb.execute("search_text", {"pattern": "def send", "path": prefixed}).ok
    assert tb.execute("get_repo_map", {"path": prefixed}).ok
    assert tb.execute("read_symbol", {"file": prefixed, "symbol_name": "Session.send"}).ok
    assert tb.execute("lock_targets", {"files": [prefixed],
                                       "symbols": [f"{prefixed}::Session.send"],
                                       "root_cause_note": "n"}).ok
    edited = tb.execute("apply_edit", {"file": prefixed, "target_symbol": "Session.send",
                                       "search_block": "        return request\n",
                                       "replace_block": "        return request or None\n"})
    assert edited.ok
    assert tb.locked["files"] == ["requests/sessions.py"]
    assert tb.locked["symbols"] == ["requests/sessions.py::Session.send"]


# --------------------------------------------------- Task 2: schema and parser
def test_every_tool_schema_forbids_unknown_keys() -> None:
    tb = ToolBox(Path("."), variant="B0")
    for variant in ("B0", "M1", "M2", "M3"):
        for spec in tb.specs(variant):
            params = spec["function"]["parameters"]
            assert params["type"] == "object"
            assert params["additionalProperties"] is False, spec["function"]["name"]


def test_glued_arguments_are_repaired_and_then_rejected_cleanly(tmp_path: Path) -> None:
    tb = _repo(tmp_path)
    res = tb.execute("search_text", {"pattern=\"def send\"\npath": "psf/requests/sessions.py"})
    assert res.ok and "sessions.py:6" in res.output
    assert tb.stats["arguments_repaired"] == 1
    assert tb.stats["malformed_arguments"] == 0

    bad = tb.execute("search_text", {"pattern=\"x\"\nbogus": "y"})
    assert not bad.ok and "Invalid JSON keys" in bad.error
    assert tb.stats["malformed_arguments"] == 1


def test_huge_numeric_literal_is_a_malformed_call_not_a_crash() -> None:
    # Python 3.11 refuses int<->str beyond 4300 digits; a model emitting such a number used to
    # raise ValueError straight through the tool loop and end the run as harness_error.
    args, err = _parse_args('{"pattern": ' + "9" * 5000 + "}")
    assert args is None
    assert err and err.startswith("arguments were not valid JSON")


# ------------------------------------------------ Task 3: three-tier edit match
def test_apply_edit_folds_escaped_and_real_crlf(tmp_path: Path) -> None:
    p = tmp_path / "m.py"
    p.write_bytes(b"def f():\r\n    x = 1\r\n    return x\r\n")
    tb = ToolBox(tmp_path, variant="B0")

    # the model wrote a literal backslash-r backslash-n where the file has a real newline
    res = tb.execute("apply_edit", {"file": "m.py",
                                    "search_block": "    x = 1\\r\\n    return x",
                                    "replace_block": "    x = 2\\r\\n    return x"})
    assert res.ok, res.error
    assert res.meta["tier"] == 1
    assert p.read_bytes() == b"def f():\r\n    x = 2\r\n    return x\r\n", "CRLF must be preserved"


def test_apply_edit_tier2_ignores_trailing_whitespace(tmp_path: Path) -> None:
    p = tmp_path / "m.py"
    p.write_text("def f():\n    x = 1   \n    return x\n", encoding="utf-8")
    tb = ToolBox(tmp_path, variant="M2")

    res = tb.execute("apply_edit", {"file": "m.py", "target_symbol": "f",
                                    "search_block": "    x = 1\n    return x\n",
                                    "replace_block": "    x = 2\n    return x\n"})
    assert res.ok, res.error
    assert res.meta["tier"] == 2
    assert p.read_text(encoding="utf-8") == "def f():\n    x = 2\n    return x\n"


def test_apply_edit_tier3_reindents_the_replacement(tmp_path: Path) -> None:
    p = tmp_path / "m.py"
    p.write_text("class A:\n    def f(self):\n        x = 1\n        return x\n",
                 encoding="utf-8")
    tb = ToolBox(tmp_path, variant="M2")

    res = tb.execute("apply_edit", {
        "file": "m.py", "target_symbol": "A.f",
        "search_block": "def f(self):\n    x = 1\n    return x",
        "replace_block": "def f(self):\n    x = 2\n    return x",
    })
    assert res.ok, res.error
    assert res.meta["tier"] == 3
    assert res.meta["indent_offset"] == 4
    assert p.read_text(encoding="utf-8") == (
        "class A:\n    def f(self):\n        x = 2\n        return x\n"
    ), "the replacement must be shifted to the file's indentation"


def test_apply_edit_tier3_refuses_an_ambiguous_site(tmp_path: Path) -> None:
    p = tmp_path / "m.py"
    p.write_text("def a():\n    x = 1\n    return x\n\ndef b():\n    x = 1\n    return x\n",
                 encoding="utf-8")
    tb = ToolBox(tmp_path, variant="M1")
    res = tb.execute("apply_edit", {"file": "m.py", "target_symbol": "a",
                                    "search_block": "x = 1\nreturn x",
                                    "replace_block": "x = 2\nreturn x"})
    assert not res.ok and "more than one place" in res.error
    assert p.read_text(encoding="utf-8").startswith("def a():\n    x = 1\n"), "nothing written"


def test_apply_edit_mismatch_shows_the_closest_region(tmp_path: Path) -> None:
    p = tmp_path / "m.py"
    p.write_text("def f():\n    x = 1\n    return x\n", encoding="utf-8")
    tb = ToolBox(tmp_path, variant="B0")
    before = p.read_text(encoding="utf-8")

    res = tb.execute("apply_edit", {"file": "m.py",
                                    "search_block": "    x = 1\n    return z",
                                    "replace_block": "    x = 2\n    return z"})
    assert not res.ok
    assert res.error.startswith("search_block not found")
    assert "closest region is lines 2-3" in res.error
    assert "|     return x" in res.error and "never copy them" in res.error
    assert tb.stats["edits_malformed"] == 1
    assert p.read_text(encoding="utf-8") == before


# ---------------------------------------------------- Task 4: symbol references
def test_read_symbol_accepts_full_reference_bare_name_and_ambiguity(tmp_path: Path) -> None:
    tb = _repo(tmp_path)

    full = tb.execute("read_symbol", {"file": "requests/sessions.py",
                                      "symbol_name": "requests/sessions.py::Session.send"})
    assert full.ok and "def send" in full.output
    dotted = tb.execute("read_symbol", {"file": "psf/requests/sessions.py",
                                        "symbol_name": "Session.send"})
    assert dotted.ok and "def send" in dotted.output
    bare = tb.execute("read_symbol", {"file": "requests/sessions.py", "symbol_name": "send"})
    assert bare.ok and "def send" in bare.output

    miss = tb.execute("read_symbol", {"file": "requests/sessions.py", "symbol_name": "Session.sned"})
    assert not miss.ok
    assert "Symbol 'Session.sned' not found in 'requests/sessions.py'" in miss.error
    assert "Available symbols in this file" in miss.error and "Session.send" in miss.error

    (tmp_path / "two.py").write_text("class A:\n    def go(self):\n        pass\n\n"
                                     "class B:\n    def go(self):\n        pass\n",
                                     encoding="utf-8")
    ambiguous = tb.execute("read_symbol", {"file": "two.py", "symbol_name": "go"})
    assert not ambiguous.ok and "ambiguous" in ambiguous.error
    assert "A.go" in ambiguous.error and "B.go" in ambiguous.error

    refs = tb.execute("find_references", {"symbol_name": "two.py::go", "path": "two.py"})
    assert refs.ok


# --------------------------------------- Task 5: lock gate, phase 2, loop-breaker
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


def _task(instance: str = "t__t-9") -> dict:
    return {"instance_id": instance, "repo": "psf/requests", "base_commit": "0" * 40,
            "problem_statement": "send() returns the wrong request"}


def _run(tmp_path: Path, tb: ToolBox, script: list[Completion], task: dict) -> tuple[dict, list]:
    cfg = RunConfig(instance_id=task["instance_id"], variant=tb.variant, model="fake")
    log = tmp_path / "run.jsonl"
    logger = JsonlLogger(log)
    try:
        end = run_agent(cfg, task, _FakeClient(script), tb, logger, verbose=False)
    finally:
        logger.close()
    recs = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    return end, recs


def test_lock_targets_rejects_a_symbol_it_cannot_load(tmp_path: Path) -> None:
    tb = _repo(tmp_path)
    res = tb.execute("lock_targets", {"files": ["psf/requests/sessions.py"],
                                      "symbols": ["psf/requests/sessions.py::Session.sned"],
                                      "root_cause_note": "n"})
    assert not res.ok and res.meta.get("lock_rejected") is True
    assert "Session.send" in res.error, "the gate must offer the symbols the file really has"
    assert tb.locked is None, "a rejected lock must not start Phase 2"
    assert tb.stats["lock_rejected"] == 1


def test_rejected_lock_keeps_phase_one_and_valid_lock_compacts_with_full_text(
        tmp_path: Path) -> None:
    tb = _repo(tmp_path)
    task = _task()
    script = [
        _comp([_call("get_repo_map", {"path": "psf/requests/sessions.py"}, 0)]),
        # wrong symbol -> the gate refuses, no compaction, Phase 1 history intact
        _comp([_call("lock_targets", {"files": ["psf/requests/sessions.py"],
                                      "symbols": ["psf/requests/sessions.py::Session.sned"],
                                      "root_cause_note": "n"}, 1)]),
        _comp([_call("lock_targets", {"files": ["psf/requests/sessions.py"],
                                      "symbols": ["psf/requests/sessions.py::Session.send"],
                                      "root_cause_note": "send loses the request"}, 2)]),
        _comp([_call("apply_edit", {"file": "psf/requests/sessions.py",
                                    "target_symbol": "Session.send",
                                    "search_block": "        return request\n",
                                    "replace_block": "        return request or None\n"}, 3)]),
        _comp([_call("finish", {"summary": "fixed"}, 4)]),
    ]
    end, recs = _run(tmp_path, tb, script, task)

    assert end["lock_rejected"] == 1 and end["compactions"] == 1
    comp = [r for r in recs if r["type"] == "compaction"][0]
    assert comp["turn"] == 3, "the bad lock at turn 2 must not compact"
    preview = comp["compacted_prompt_preview"]
    assert preview.startswith("# Phase 2 - Patching")
    assert PHASE2_DIRECTIVE in preview
    assert "=== requests/sessions.py :: Session.send (lines 6-7) ===" in preview
    assert "return request" in preview, "Phase 2 must carry the locked source itself"
    assert comp["phase2_directive"] == PHASE2_DIRECTIVE
    assert (tmp_path / "requests" / "sessions.py").read_text(encoding="utf-8").endswith(
        "return request or None\n")


def test_loop_breaker_blocks_a_byte_identical_repeat(tmp_path: Path) -> None:
    tb = _repo(tmp_path)
    wrong = {"file": "requests/sessions.py", "target_symbol": "Session.send",
             "search_block": "        return requestX\n",
             "replace_block": "        return None\n"}
    script = [
        _comp([_call("apply_edit", wrong, 0)]),
        _comp([_call("apply_edit", wrong, 1)]),          # byte-identical repeat
        _comp([_call("finish", {"summary": "gave up"}, 2)]),
    ]
    end, recs = _run(tmp_path, tb, script, _task())

    assert end["repeated_identical_calls"] == 1
    assert end["loop_blocked"] == 1
    assert end["edits_attempted"] == 1, "the repeated call must not reach the tool again"

    repeat = [r for r in recs if r["type"] == "tool" and r["turn"] == 2][0]
    assert repeat["loop_blocked"] is True and repeat["repeat_of_turn"] == 1
    assert "LOOP DETECTED" in repeat["loop_warning"]


def test_loop_warning_is_the_first_thing_in_the_tool_result(tmp_path: Path) -> None:
    tb = _repo(tmp_path)
    args = {"file": "requests/sessions.py", "target_symbol": "Session.send",
            "search_block": "        return requestX\n", "replace_block": "        return None\n"}
    client_script = [_comp([_call("apply_edit", args, 0)]),
                     _comp([_call("apply_edit", args, 1)]),
                     _comp([_call("finish", {"summary": "stop"}, 2)])]
    client = _FakeClient(client_script)
    cfg = RunConfig(instance_id="t__t-9", variant="M3", model="fake")
    logger = JsonlLogger(tmp_path / "r.jsonl")
    try:
        run_agent(cfg, _task(), client, _repo(tmp_path), logger, verbose=False)
    finally:
        logger.close()

    third_request = client.seen[2]
    tool_msg = [m for m in third_request if m["role"] == "tool"][-1]
    payload = json.loads(tool_msg["content"])
    assert list(payload)[0] == "LOOP DETECTED"
    assert "MUST modify your arguments" in payload["LOOP DETECTED"]


# ------------------------------------------------------- Task 6 / log organisation
def test_batch_can_be_pinned_to_an_experiment_directory(tmp_path: Path, monkeypatch) -> None:
    import harness.config as config

    monkeypatch.setattr(config, "EXPERIMENTS", tmp_path / "experiments")
    try:
        base = config.set_experiment("smoke-run")
        assert base == tmp_path / "experiments" / "smoke-run"
        cfg = RunConfig(instance_id="x__y-1", variant="B0", model="fake")
        assert cfg.log_path.parent == base / "runs"
        assert cfg.patch_path.parent == base / "patches"
        assert cfg.as_dict()["experiment"] == "smoke-run"
    finally:
        config.set_experiment(None)          # restore the default logs/runs + logs/patches


def test_manifest_merges_a_resume_instead_of_replacing_the_plan(tmp_path: Path, monkeypatch) -> None:
    import harness.config as config
    import run_batch

    monkeypatch.setattr(config, "EXPERIMENTS", tmp_path / "experiments")
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(run_batch, "git_commit", lambda: "abc123")
    monkeypatch.setattr(sys, "argv", ["run_batch.py", "--experiment", "e"])
    args = SimpleNamespace(experiment="e", model="m", variants=["B0", "M1"], thinking="on",
                           max_turns=15, workers=4, ctx_per_run=32768,
                           base_url="http://x", no_stream=False)
    config.set_experiment("e")
    try:
        run_batch.write_manifest(args, [{"instance_id": "a-1"}, {"instance_id": "b-2"}],
                                 todo=[1, 2], skipped=0)
        run_batch.write_manifest(args, [{"instance_id": "a-1"}], todo=[1], skipped=2)
    finally:
        config.set_experiment(None)

    manifest = json.loads((tmp_path / "experiments" / "e" / "manifest.json")
                          .read_text(encoding="utf-8"))
    assert manifest["task_ids"] == ["a-1", "b-2"]      # union, not the last invocation
    assert manifest["task_count"] == 2
    assert [inv["planned_cells"] for inv in manifest["invocations"]] == [2, 1]
    assert manifest["invocations"][1]["already_done"] == 2


def test_batch_lowers_concurrency_when_the_kv_pool_cannot_serve_it(monkeypatch) -> None:
    import run_batch

    monkeypatch.setattr(run_batch.LLMClient, "ctx_pool", lambda self: 32768)

    def fake_probe(base_url, workers, api_key=None, ctx=None, timeout=0):
        return 32768 if workers == 1 else None       # only one full window fits

    monkeypatch.setattr(run_batch, "verify_run_context", fake_probe)
    args = SimpleNamespace(base_url="http://x", model="fake", api_key=None, workers=4)
    assert run_batch.probe_workers(args) == (1, 32768)


def test_run_config_streams_by_default() -> None:
    """Task 6.1: streaming is the default transport, so a long thinking turn cannot be cut off."""
    assert RunConfig(instance_id="x__y-1").stream is True
