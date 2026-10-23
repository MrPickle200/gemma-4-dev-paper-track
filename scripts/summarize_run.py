"""Summarize one run's .jsonl trace, and optionally drill into a specific turn.

This is the primitive Bước 5 will aggregate over (IDEA.md §4): resolve rate inputs,
malformed edit rate, peak context and token totals, and loop/error behaviour.

Usage:
  .\\.venv\\Scripts\\python.exe scripts\\summarize_run.py experiments\\<tag>\\runs\\<run>.jsonl
  .\\.venv\\Scripts\\python.exe scripts\\summarize_run.py experiments\\<tag>\\runs\\<run>.jsonl --show-tool apply_edit
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="path to a .jsonl run log")
    ap.add_argument("--show-tool", help="dump args/output of every call to this tool")
    ap.add_argument("--limit", type=int, default=3, help="how many tool calls to dump")
    args = ap.parse_args()

    recs = [json.loads(line) for line in Path(args.path).read_text(encoding="utf-8").splitlines() if line.strip()]
    kinds = Counter(r["type"] for r in recs)
    start = next((r for r in recs if r["type"] == "run_start"), {})
    end = next((r for r in recs if r["type"] == "run_end"), {})

    print(f"run_id   {start.get('run_id')}")
    print(f"task     {start.get('instance_id')}  ({start.get('repo')})")
    print(f"config   model={start.get('config',{}).get('model')} variant={start.get('config',{}).get('variant')} "
          f"thinking={start.get('config',{}).get('thinking')} max_turns={start.get('config',{}).get('max_turns')}")
    print(f"records  {dict(kinds)}")
    print(f"outcome  stop={end.get('stop_reason')} turns={end.get('turns_used')} "
          f"wall={end.get('wall_s')}s")
    print(f"tokens   total={end.get('total_tokens')} "
          f"(prompt {end.get('total_prompt_tokens')} / completion {end.get('total_completion_tokens')}) "
          f"peak_context={end.get('peak_prompt_tokens')}")
    print(f"edits    attempted={end.get('edits_attempted')} malformed={end.get('edits_malformed')} "
          f"ambiguous={end.get('edits_ambiguous')} syntax_rejected={end.get('edits_syntax_rejected')}")
    print(f"errors   tool_errors={end.get('tool_errors')} malformed_args={end.get('malformed_arguments')} "
          f"repeated_identical_calls={end.get('repeated_identical_calls')}")
    cap = start.get("capabilities", {})
    print(f"modules  ast_nav={cap.get('ast_nav')} structured_edit={cap.get('structured_edit')} "
          f"syntax_gate={cap.get('syntax_gate')} lock_targets={cap.get('lock_targets')}")
    print(f"nav/lock nav_calls={end.get('nav_calls')} locks={end.get('lock_calls')} "
          f"symbols_locked={end.get('symbols_locked')} compactions={end.get('compactions')} "
          f"compacted={end.get('compacted')}")

    tools = Counter(r.get("name") for r in recs if r["type"] == "tool")
    if tools:
        print(f"tools    {dict(tools)}")

    turns = [r for r in recs if r["type"] == "turn"]
    if turns:
        print("per-turn prompt tokens (context depth):")
        print("         " + " ".join(str(t["usage"]["prompt_tokens"]) for t in turns if "usage" in t))
        reasoning = sum(t.get("response", {}).get("reasoning_chars", 0) for t in turns)
        print(f"         reasoning_chars_total={reasoning}")

    if args.show_tool:
        shown = 0
        for r in recs:
            if r["type"] == "tool" and r.get("name") == args.show_tool:
                shown += 1
                if shown > args.limit:
                    break
                print(f"\n--- {args.show_tool} call #{shown} (turn {r.get('turn')}, status={r.get('status')})")
                print(f"    args   : {json.dumps(r.get('args') or r.get('args_raw'), indent=6)[:1200]}")
                if r.get("error"):
                    print(f"    error  : {r['error']}")
                print(f"    output : {(r.get('output_preview') or '')[:400]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
