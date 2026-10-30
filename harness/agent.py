"""The agent loop and its per-turn JSONL trace.

Every turn appends one `turn` record and one `tool` record per invoked tool, so the
resulting .jsonl is a complete transcript: full prompt accounting, the raw response
(content + reasoning_content + tool calls), token counts, and tool status. The final
`run_end` record carries the aggregate numbers IDEA.md §4 asks for.

The loop serves every ablation variant; the variant selects the tool set (raised by
`ToolBox`) and, for M3, whether a `lock_targets` call triggers Episode Context Compaction
(IDEA.md §2 Module 3).

Design note: `reasoning_content` is logged but deliberately NOT fed back into the next
request. The Gemma 4 template strips prior thinking anyway, and re-injecting it is
exactly the context bloat Module 3's compaction is meant to remove.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import astnav
from .config import RunConfig, refined_stop_reason
from .llm import Completion, LLMClient
from .tools import ToolBox, ToolResult, split_symbol_ref

_BASE_PROMPT = """You are a coding agent that repairs bugs in a Python repository.

You may ONLY act through the provided tools. There is no free-form editing.

Rules:
- Change as little as possible: fix the reported bug only, in one file where possible.
- Never include line numbers in `search_block`.
- If `apply_edit` reports that the block is not found, re-read the code and copy it again.
- Do not explain your plan at length; act with tools."""

_LOCALIZE_BASH = """Work in this order:
1. `search_text` to find the relevant code. Search for distinctive strings from the bug report.
2. `read_file` the specific region you need. Do not read whole large files.
3. `apply_edit` with the exact text you saw.
4. `finish` when the fix is complete."""

_LOCALIZE_AST = """Work in this order:
1. `get_repo_map` to see the skeleton (classes, functions, signatures) of a file or directory.
2. `find_references` to see where a name is used in the repository.
3. `read_symbol` to load the full source of exactly the function or class you need.
4. `apply_edit` with the exact text you saw.
5. `finish` when the fix is complete."""

_EDIT_STRUCTURED = """`apply_edit` in this variant also asks for `target_symbol`: the name of the function or class
the edit belongs to. The edit is parsed with ast.parse() before it is written, so a
syntactically broken edit is rejected, nothing is written, and the error comes back to you to fix
in the next turn.

Every tool call is one JSON object whose keys are exactly the parameter names listed for that
tool. Never put a parameter name inside another parameter's value."""

_M3_PHASES = """Work in two phases.
Phase 1 - LOCALIZE: use the navigation tools to find the root cause. As soon as you know the
file(s) and symbol(s), call `lock_targets(files, symbols, root_cause_note)`.
Phase 2 - PATCH: after `lock_targets` your scratch history is cleared and you continue with a
clean context holding only the issue, your root-cause note and the locked symbols. Make the
edit and call `finish`."""

PHASE2_DIRECTIVE = ("=== PHASE 2: PATCHING === The full source code of your locked target symbols "
                    "is provided above. Do NOT call get_repo_map or read_symbol on these targets "
                    "again. Proceed directly to apply_edit.")

LOOP_WARNING = ("LOOP DETECTED: You just executed the exact same tool call with identical "
                "arguments as the previous turn. Repeating it again will not change the result. "
                "You MUST modify your arguments (e.g., fix the search_block/symbol name) or "
                "choose a different tool.")


def system_prompt(variant: str) -> str:
    parts = [_BASE_PROMPT]
    if variant in ("M3", "M4"):
        parts.append(_M3_PHASES)
        parts.append(_LOCALIZE_AST)
    elif variant == "M2":
        parts.append(_LOCALIZE_AST)
    else:
        parts.append(_LOCALIZE_BASH)
    if variant in ("M1", "M2", "M3", "M4"):
        parts.append(_EDIT_STRUCTURED)
    return "\n\n".join(parts)


def build_user_prompt(task: dict, seeded_text: str | None = None) -> str:
    base = (
        f"Repository: {task['repo']}\n\n"
        f"Bug report:\n{task['problem_statement']}\n\n"
        "Locate the cause and apply the fix."
    )
    if seeded_text:
        base = base + "\n\n" + seeded_text
    return base


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # newline="\n": on Windows text mode would turn every trace into CRLF. The evidence is
        # read by scripts (and by a fresh clone, where .gitattributes says LF), so keep one
        # convention everywhere and let git store it as LF.
        self._fh = self.path.open("a", encoding="utf-8", newline="\n")
        self.records = 0

    def write(self, record: dict) -> None:
        record = {"ts": round(time.time(), 3), **record}
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()
        self.records += 1

    def close(self) -> None:
        self._fh.close()


def _parse_args(raw: str) -> tuple[dict | None, str | None]:
    if raw is None or raw == "":
        return {}, None
    try:
        val = json.loads(raw)
    except (json.JSONDecodeError, RecursionError) as e:
        return None, f"arguments were not valid JSON: {e}"
    except ValueError as e:
        # Python 3.11 caps int<->str at 4300 digits; a model that emits a 15,000-digit number in
        # its arguments raises ValueError from json.loads, not JSONDecodeError. That is still a
        # malformed call (metric #2) and must not kill the run.
        return None, f"arguments were not valid JSON: {e}"
    if not isinstance(val, dict):
        return None, f"arguments must be a JSON object, got {type(val).__name__}"
    return val, None


def _locked_source(toolbox: ToolBox, locked: dict, max_chars: int = 12000) -> str:
    """Source of the locked symbols, for the post-compaction context (IDEA.md §2 Module 3).

    Each block is labelled with the *resolved* repo-relative path and the symbol's line span, so
    Phase 2 can copy text into `search_block` without re-reading the file. `lock_targets` has
    already proven that every entry yields source; the whole-file fallback covers a lock that
    named files but no symbols.
    """
    blocks: list[str] = []
    files = [str(f) for f in locked.get("files", [])]
    entries = [str(s) for s in locked.get("symbols", [])]
    for entry in entries:
        file_part, name = split_symbol_ref(entry)
        candidates = [file_part] if file_part else files
        for rel in candidates:
            try:
                p = toolbox._resolve(rel)
            except (ValueError, OSError):
                continue
            if not p.is_file():
                continue
            rel_path = p.relative_to(toolbox.workdir).as_posix()
            for sym in astnav.find_symbol(p, name, rel_path):
                blocks.append(
                    f"=== {rel_path} :: {sym.qualified} (lines {sym.start_line}-{sym.end_line}) ===\n"
                    + astnav.read_source(p, sym)
                )
    if not blocks:  # a lock on whole files (no symbols) still owes Phase 2 its code
        for rel in files:
            try:
                p = toolbox._resolve(rel)
            except (ValueError, OSError):
                continue
            if p.is_file():
                rel_path = p.relative_to(toolbox.workdir).as_posix()
                blocks.append(f"=== {rel_path} (whole file) ===\n"
                              + p.read_text(encoding="utf-8", errors="ignore"))
    text = "\n\n".join(blocks)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... [compacted source truncated at {max_chars} chars]"
    return text


def phase2_body(task: dict, toolbox: ToolBox, locked: dict) -> str:
    """The exact user message that starts the patching phase."""
    note = locked.get("root_cause_note") or "(none provided)"
    return (
        "# Phase 2 - Patching\n\n"
        "Your localization history has been cleared. Everything you need is below.\n\n"
        f"{PHASE2_DIRECTIVE}\n\n"
        f"## Issue\n{task['problem_statement']}\n\n"
        f"## Root cause (your note)\n{note}\n\n"
        "## Locked code (copy text verbatim into search_block; no line numbers)\n"
        + _locked_source(toolbox, locked)
    )


def compact_messages(sys_prompt: str, task: dict, toolbox: ToolBox,
                     locked: dict) -> tuple[list[dict], str]:
    """Build the clean Phase-2 context: issue + root-cause note + locked source only.

    Returns the messages and the body text, because the body is written to the trace under
    `compacted_prompt_preview` - a compaction whose content cannot be inspected is not evidence.
    """
    body = phase2_body(task, toolbox, locked)
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": body},
    ], body


def run_agent(cfg: RunConfig, task: dict, client: LLMClient, toolbox: ToolBox,
              logger: JsonlLogger, verbose: bool = True) -> dict:
    specs = toolbox.specs(cfg.variant)
    sys_prompt = system_prompt(cfg.variant)
    seeded_symbols = None
    seeded_text = None
    if cfg.variant == "M4":
        try:
            from harness.seeding import load_cache, format_block
            cache_path = Path("cache/seeding") / f"{task['instance_id']}.json"
            c = load_cache(cache_path)
            if c:
                seeded_symbols = c.get("top_symbols", [])[:8]
                seeded_text = format_block(seeded_symbols)
        except Exception:
            seeded_symbols = None
            seeded_text = None
    messages: list[dict] = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": build_user_prompt(task, seeded_text)},
    ]
    phase = "localize" if cfg.variant in ("M3", "M4") else "solve"
    compacted = False

    logger.write({
        "type": "run_start",
        "run_id": cfg.run_id,
        "instance_id": task["instance_id"],
        "repo": task["repo"],
        "base_commit": task["base_commit"],
        "difficulty": task.get("difficulty"),
        "config": cfg.as_dict(),
        "tools": [s["function"]["name"] for s in specs],
        "capabilities": {
            "ast_nav": toolbox.use_ast_nav,
            "structured_edit": toolbox.structured_edit,
            "syntax_gate": toolbox.syntax_gate,
            "lock_targets": toolbox.allow_lock,
        },
        "problem_statement_chars": len(task["problem_statement"]),
        "system_prompt_chars": len(sys_prompt),
        "seeded_symbols": seeded_symbols,
    })

    t_run = time.time()
    stop_reason = "max_turns"
    turns_used = 0
    total_prompt_tokens = total_completion_tokens = 0
    peak_prompt_tokens = 0
    length_retries = 0
    # Hard loop-breaker (TASK.md Task 5.3): the signatures executed in the *previous* turn, and
    # what they returned. A byte-identical repeat is not executed again - its old result is
    # replayed behind a mandatory warning, which is the only thing that has ever moved E2B off a
    # loop (report §5.1: the model held the refusal and re-sent the call regardless).
    prev_turn_sigs: set[tuple[str, str]] = set()
    prev_results: dict[tuple[str, str], ToolResult] = {}
    turn_sigs: set[tuple[str, str]] = set()
    turn_results: dict[tuple[str, str], ToolResult] = {}

    for turn in range(1, cfg.max_turns + 1):
        turns_used = turn
        prev_turn_sigs, prev_results = turn_sigs, turn_results
        turn_sigs, turn_results = set(), {}
        comp: Completion = client.chat(messages, tools=specs)

        if comp.error:
            logger.write({"type": "turn", "turn": turn, "phase": phase, "error": comp.error})
            # Name the failure while the message is in hand: the model overflowing its slot, or
            # emitting tool-call JSON the server cannot parse, is a RESULT of the experiment (it
            # is what M1 and M3 exist to change), not a backend accident to re-run. Anything else
            # stays `llm_error`, which callers treat as infrastructure. A context overflow only
            # counts as a result if this run actually had the whole window - with llama-server's
            # shared KV pool a 4-way batch on `-c 32768` gives each run ~8k, so see ctx_is_adequate.
            stop_reason = refined_stop_reason("llm_error", comp.error, cfg.ctx_per_run)
            break

        total_prompt_tokens += comp.prompt_tokens
        total_completion_tokens += comp.completion_tokens
        peak_prompt_tokens = max(peak_prompt_tokens, comp.prompt_tokens)

        logger.write({
            "type": "turn",
            "turn": turn,
            "phase": phase,
            "request": {"n_messages": len(messages), "tools_offered": len(specs)},
            "response": {
                "content": comp.content,
                "reasoning_content": comp.reasoning_content,
                "reasoning_chars": len(comp.reasoning_content),
                "tool_calls": comp.tool_calls,
                "finish_reason": comp.finish_reason,
            },
            "usage": {
                "prompt_tokens": comp.prompt_tokens,
                "completion_tokens": comp.completion_tokens,
                "cached_tokens": comp.cached_tokens,
                "total_tokens": comp.prompt_tokens + comp.completion_tokens,
            },
            "timings": comp.timings,
            "wall_s": round(comp.wall_s, 2),
        })
        if verbose:
            names = [tc["function"]["name"] for tc in comp.tool_calls] or ["<no tool call>"]
            print(f"  turn {turn:>2}: {comp.prompt_tokens:>6} prompt tok, "
                  f"{comp.completion_tokens:>5} out, {comp.wall_s:>5.1f}s -> {', '.join(names)}")

        if not comp.tool_calls:
            # A turn cut off by the token/context limit (finish_reason="length") is not a
            # decision to stop - the small models can ramble past the window without acting.
            # Nudge once or twice instead of silently ending the run with no edit.
            if comp.finish_reason == "length" and length_retries < 2:
                length_retries += 1
                messages.append({"role": "assistant", "content": comp.content or ""})
                messages.append({
                    "role": "user",
                    "content": ("Your last message was cut off by the token limit before you used a "
                                "tool. Act now: call `apply_edit` with the exact text, or `finish` if "
                                "the change is already applied."),
                })
                stop_reason = "length_retry"
                if verbose:
                    print("          [length] turn truncated; nudging the model to act")
                continue
            messages.append({"role": "assistant", "content": comp.content})
            stop_reason = "no_tool_call"
            break

        # echo the assistant turn back without the thinking trace (see module docstring)
        messages.append({
            "role": "assistant",
            "content": comp.content or "",
            "tool_calls": comp.tool_calls,
        })

        finished = False
        lock_this_turn = False
        for tc in comp.tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args, parse_err = _parse_args(fn.get("arguments"))
            t0 = time.time()

            if parse_err:
                result_json = json.dumps({"error": parse_err})
                toolbox.stats["tool_errors"] += 1
                toolbox.stats["tool_calls"] += 1
                toolbox.stats["malformed_arguments"] += 1
                logger.write({
                    "type": "tool", "turn": turn, "phase": phase, "name": name,
                    "args_raw": fn.get("arguments"),
                    "status": "malformed_arguments", "error": parse_err,
                    "duration_s": round(time.time() - t0, 2),
                })
                messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": result_json})
                continue

            # loop detection: IDEA.md §4 metric #4 (wrong/looping tool calls vs context depth)
            sig = (name, json.dumps(args, sort_keys=True))
            repeat_of = None
            if sig in prev_turn_sigs:
                repeat_of = turn - 1
            elif sig == toolbox._last_call_sig:
                repeat_of = turn
            if repeat_of is not None:
                toolbox.stats["repeated_identical_calls"] += 1

            blocked = sig in prev_turn_sigs and sig in prev_results
            if blocked:
                cached = prev_results[sig]
                toolbox.stats["tool_calls"] += 1
                toolbox.stats["loop_blocked"] += 1
                res = ToolResult(ok=cached.ok, output=cached.output, error=cached.error,
                                 meta={k: v for k, v in cached.meta.items() if k != "lock"})
            else:
                res = toolbox.execute(name, args)
            toolbox._last_call_sig = sig
            turn_sigs.add(sig)
            turn_results[sig] = res

            payload = {"ok": res.ok, "output": res.output}
            if res.error:
                payload["error"] = res.error
            if blocked:
                # The warning goes first in the JSON object, so it is the first thing the model
                # reads in the tool result (TASK.md Task 5.3).
                payload = {"LOOP DETECTED": LOOP_WARNING, **payload}
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": json.dumps(payload, ensure_ascii=False),
            })

            logger.write({
                "type": "tool",
                "turn": turn,
                "phase": phase,
                "name": name,
                "args": args,
                "status": "ok" if res.ok else "error",
                "error": res.error,
                "meta": res.meta,
                "repeat_of_turn": repeat_of,
                "loop_blocked": blocked,
                **({"loop_warning": LOOP_WARNING} if blocked else {}),
                "output_chars": len(res.output),
                "output_preview": res.output[:600],
                "duration_s": round(time.time() - t0, 3),
            })
            if verbose:
                flag = "ok " if res.ok else "ERR"
                print(f"          [{flag}] {name} {res.error or ''}".rstrip()
                      + (" [loop blocked]" if blocked else ""))
            if res.meta.get("finish"):
                stop_reason = "finish"
                finished = True
            if res.meta.get("lock"):
                lock_this_turn = True

        # Module 3: Episode Context Compaction. This is a one-way phase transition
        # (IDEA.md §2 Module 3): the first lock_targets replaces the scratch history with a
        # clean Phase-2 context. Later locks are acknowledged but do not compact again, so the
        # mechanism cannot trap the model in a lock loop.
        if toolbox.allow_lock and lock_this_turn and not compacted:
            chars_before = sum(len(m.get("content") or "") for m in messages)
            messages, body = compact_messages(sys_prompt, task, toolbox, toolbox.locked)
            compacted = True
            phase = "patch"
            toolbox._last_call_sig = None
            turn_sigs, turn_results = set(), {}
            toolbox.stats["compactions"] = toolbox.stats.get("compactions", 0) + 1
            chars_after = sum(len(m.get("content") or "") for m in messages)
            logger.write({
                "type": "compaction",
                "turn": turn,
                "files": toolbox.locked["files"],
                "symbols": toolbox.locked["symbols"],
                "root_cause_note": toolbox.locked["root_cause_note"],
                "messages_after": len(messages),
                "context_chars_before": chars_before,
                "context_chars_after": chars_after,
                # Full text handed to Phase 2 (TASK.md Task 5.2): without it a compaction could
                # only be audited by replaying compact_messages against a deleted worktree.
                "compacted_prompt_preview": body,
                "phase2_directive": PHASE2_DIRECTIVE,
                "locked_source_chars": len(_locked_source(toolbox, toolbox.locked)),
            })
            if verbose:
                print(f"          [compaction] context {chars_before} -> {chars_after} chars, "
                      f"{len(messages)} messages ({phase} phase)")

        if finished:
            break

    summary = {
        "type": "run_end",
        "run_id": cfg.run_id,
        "instance_id": task["instance_id"],
        "model": cfg.model,
        "variant": cfg.variant,
        "stop_reason": stop_reason,
        "compacted": compacted,
        "turns_used": turns_used,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "peak_prompt_tokens": peak_prompt_tokens,
        "wall_s": round(time.time() - t_run, 1),
        **toolbox.stats,
    }
    logger.write(summary)
    return summary
