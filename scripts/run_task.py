"""Run the agent loop on one or more tasks from data/subset_50.json.

Examples:
  .\\.venv\\Scripts\\python.exe scripts\\run_task.py --list
  .\\.venv\\Scripts\\python.exe scripts\\run_task.py --instance-id sympy__sympy-20590
  .\\.venv\\Scripts\\python.exe scripts\\run_task.py --repo sympy/sympy --limit 3 --variant B0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.config import MIN_RUN_CTX, RunConfig  # noqa: E402
from harness.llm import verify_run_context  # noqa: E402
from harness.runner import load_tasks, run_instance  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list candidate tasks and exit")
    ap.add_argument("--instance-id", action="append", default=[])
    ap.add_argument("--repo", help="restrict to one repo, e.g. psf/requests")
    ap.add_argument("--limit", type=int, help="max tasks to run")
    ap.add_argument("--model", default="gemma-4-E2B", help="label recorded in the log")
    ap.add_argument("--variant", default="B0", choices=["B0", "M1", "M2", "M3"])
    ap.add_argument("--thinking", default="off", choices=["auto", "on", "off"])
    ap.add_argument("--line-numbers", action="store_true",
                    help="print line numbers in read_file (OFF by default: B0 is `cat` fidelity)")
    ap.add_argument("--max-turns", type=int, default=15)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--api-key", default=None, help="bearer token, for a proxied/remote server")
    ap.add_argument("--no-stream", action="store_true", help="buffer whole responses instead of SSE")
    ap.add_argument("--keep-worktree", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    tasks = load_tasks()
    if args.instance_id:
        wanted = set(args.instance_id)
        tasks = [t for t in tasks if t["instance_id"] in wanted]
    if args.repo:
        tasks = [t for t in tasks if t["repo"] == args.repo]
    if args.limit:
        tasks = tasks[: args.limit]

    if args.list:
        for t in tasks:
            print(f"{t['instance_id']:<32} {t['repo']:<28} {t['_n_changed_lines']:>3} lines")
        print(f"\n{len(tasks)} task(s)")
        return 0

    if not tasks:
        print("no tasks selected")
        return 1

    summaries = []
    ctx = verify_run_context(args.base_url, 1, api_key=args.api_key)   # one run at a time
    if ctx is None:
        print(f"WARNING: {args.base_url} cannot serve a {MIN_RUN_CTX}-token prompt; a "
              f"context-exceeded run here is the backend's doing, not the model's.")
    for task in tasks:
        cfg = RunConfig(
            instance_id=task["instance_id"], model=args.model, variant=args.variant,
            base_url=args.base_url, thinking=args.thinking, max_turns=args.max_turns,
            line_numbers=args.line_numbers, workdir=None, stream=not args.no_stream,
            api_key=args.api_key, ctx_per_run=ctx,        # sole occupant: it gets the whole pool
        )
        try:
            summaries.append(run_instance(cfg, task, verbose=not args.quiet,
                                          keep_worktree=args.keep_worktree))
        except Exception as e:  # keep the batch going
            print(f"  FAILED {task['instance_id']}: {type(e).__name__}: {e}")
            summaries.append({"instance_id": task["instance_id"], "stop_reason": f"harness_error:{type(e).__name__}"})

    print(f"\n=== {len(summaries)} run(s) ===")
    for s in summaries:
        print(f"{s['instance_id']:<32} {s.get('stop_reason','?'):<22} "
              f"turns={s.get('turns_used','?'):>2} tok={s.get('total_tokens','?'):>7} "
              f"malformed={s.get('edits_malformed','?')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
