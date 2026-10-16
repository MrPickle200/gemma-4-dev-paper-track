"""The agent loop (Bước 2) and its per-turn JSONL trace.

Every turn appends one `turn` record and one `tool` record per invoked tool, so the
resulting .jsonl is a complete transcript: full prompt accounting, the raw response
(content + reasoning_content + tool calls), token counts, and tool status. The final
`run_end` record carries the aggregate numbers IDEA.md §4 asks for.

Design note: `reasoning_content` is logged but deliberately NOT fed back into the next
request. The Gemma 4 template strips prior thinking anyway, and re-injecting it is
exactly the context bloat Module 3's compaction is meant to remove.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .config import RunConfig
from .llm import Completion, LLMClient
from .tools import ToolBox

SYSTEM_PROMPT = """You are a coding agent that repairs bugs in a Python repository.

You may ONLY act through the provided tools. There is no free-form editing.

Work in this order:
1. `search_text` to find the relevant code. Search for distinctive strings from the bug report.
2. `read_file` the specific region you need. Do not read whole large files.
3. `apply_edit` with the exact text you saw. `search_block` must be copied verbatim,
   including indentation. Never include line numbers in `search_block`.
4. `finish` when the fix is complete.

Rules:
- Change as little as possible: fix the reported bug only, in one file where possible.
- If `apply_edit` reports that the block is not found, re-read the file and copy the text again.
- Do not explain your plan at length; act with tools."""


def build_user_prompt(task: dict) -> str:
    return (
        f"Repository: {task['repo']}\n\n"
        f"Bug report:\n{task['problem_statement']}\n\n"
        "Locate the cause and apply the fix."
    )


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
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
    except json.JSONDecodeError as e:
        return None, f"arguments were not valid JSON: {e}"
    if not isinstance(val, dict):
        return None, f"arguments must be a JSON object, got {type(val).__name__}"
    return val, None


def run_agent(cfg: RunConfig, task: dict, client: LLMClient, toolbox: ToolBox,
              logger: JsonlLogger, verbose: bool = True) -> dict:
    specs = toolbox.specs(cfg.variant)
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(task)},
    ]

    logger.write({
        "type": "run_start",
        "run_id": cfg.run_id,
        "instance_id": task["instance_id"],
        "repo": task["repo"],
        "base_commit": task["base_commit"],
        "difficulty": task.get("difficulty"),
        "config": cfg.as_dict(),
        "tools": [s["function"]["name"] for s in specs],
        "problem_statement_chars": len(task["problem_statement"]),
        "system_prompt_chars": len(SYSTEM_PROMPT),
    })

    t_run = time.time()
    stop_reason = "max_turns"
    turns_used = 0
    total_prompt_tokens = total_completion_tokens = 0
    peak_prompt_tokens = 0

    for turn in range(1, cfg.max_turns + 1):
        turns_used = turn
        comp: Completion = client.chat(messages, tools=specs)

        if comp.error:
            logger.write({"type": "turn", "turn": turn, "phase": "solve", "error": comp.error})
            stop_reason = "llm_error"
            break

        total_prompt_tokens += comp.prompt_tokens
        total_completion_tokens += comp.completion_tokens
        peak_prompt_tokens = max(peak_prompt_tokens, comp.prompt_tokens)

        logger.write({
            "type": "turn",
            "turn": turn,
            "phase": "solve",
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
                    "type": "tool", "turn": turn, "name": name, "args_raw": fn.get("arguments"),
                    "status": "malformed_arguments", "error": parse_err,
                    "duration_s": round(time.time() - t0, 2),
                })
                messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": result_json})
                continue

            # loop detection: IDEA.md §4 metric #4 (wrong/looping tool calls vs context depth)
            sig = (name, json.dumps(args, sort_keys=True))
            repeat_of = None
            if sig == toolbox._last_call_sig:
                toolbox.stats["repeated_identical_calls"] += 1
                repeat_of = turn - 1
            toolbox._last_call_sig = sig

            res = toolbox.execute(name, args)
            payload = {"ok": res.ok, "output": res.output}
            if res.error:
                payload["error"] = res.error
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": json.dumps(payload, ensure_ascii=False),
            })

            logger.write({
                "type": "tool",
                "turn": turn,
                "name": name,
                "args": args,
                "status": "ok" if res.ok else "error",
                "error": res.error,
                "meta": res.meta,
                "repeat_of_turn": repeat_of,
                "output_chars": len(res.output),
                "output_preview": res.output[:600],
                "duration_s": round(time.time() - t0, 3),
            })
            if verbose:
                flag = "ok " if res.ok else "ERR"
                print(f"          [{flag}] {name} {res.error or ''}".rstrip())
            if res.meta.get("finish"):
                stop_reason = "finish"
                finished = True

        if finished:
            break

    summary = {
        "type": "run_end",
        "run_id": cfg.run_id,
        "instance_id": task["instance_id"],
        "model": cfg.model,
        "variant": cfg.variant,
        "stop_reason": stop_reason,
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
