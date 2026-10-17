"""Bước 4: run the ablation grid (variant x task) against the currently served model.

One model per invocation, and now one *endpoint* per invocation: either the local 6 GB
llama-server (scripts/serve.ps1) or a remote one (e.g. llama-server on a Colab GPU behind a
tunnel, see colab/README.md). The harness only ever speaks OpenAI-compatible HTTP, so the two are
interchangeable; run_start records `base_url` so the traces can be told apart afterwards.

The grid is resumable: before each (task, variant) cell it looks for a *completed* trace in
logs/runs whose config matches (model, variant, thinking, max_turns, line_numbers); if one
exists the cell is skipped. Re-running after an interruption therefore picks up where it left off.

--workers N runs N cells concurrently. This only pays off against a backend with several slots
(llama-server -np N / vLLM continuous batching); against a single-slot server the requests just
queue. Git worktree add/remove is serialised per repo inside harness.runner.

Examples:
  .\\.venv\\Scripts\\python.exe scripts\\run_batch.py --model gemma-4-E2B --variants B0 M1 M2 M3 --thinking on
  .\\.venv\\Scripts\\python.exe scripts\\run_batch.py --model gemma-4-E2B --thinking on --workers 8 `
      --base-url https://<name>.trycloudflare.com --api-key <key>
  .\\.venv\\Scripts\\python.exe scripts\\run_batch.py --model gemma-4-E4B --variants M3 --thinking on --dry-run
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests                                          # noqa: E402

from harness.config import RUNS, RunConfig, is_infra_failure   # noqa: E402
from harness.runner import load_tasks, run_instance        # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

PRINT_LOCK = threading.Lock()


def completed_run(instance_id: str, model: str, variant: str, thinking: str,
                  max_turns: int, line_numbers: bool) -> str | None:
    """Return the run_id of a matching completed trace, or None."""
    pattern = str(RUNS / f"{instance_id}__{model}__{variant}__*.jsonl")
    for path in glob.glob(pattern):
        try:
            recs = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, json.JSONDecodeError):
            continue
        start = next((r for r in recs if r["type"] == "run_start"), None)
        end = next((r for r in recs if r["type"] == "run_end"), None)
        if not start or not end:
            continue
        if is_infra_failure(end.get("stop_reason")):
            continue  # the backend died mid-run: the cell still needs a real run
        cfg = start.get("config", {})
        if (cfg.get("thinking") == thinking and cfg.get("max_turns") == max_turns
                and cfg.get("line_numbers") == line_numbers
                and cfg.get("model") == model and cfg.get("variant") == variant):
            return start.get("run_id")
    return None


def server_ok(base_url: str) -> bool:
    try:
        return requests.get(f"{base_url.rstrip('/')}/health", timeout=10).status_code == 200
    except requests.RequestException:
        return False


def make_config(args, task: dict, variant: str) -> RunConfig:
    return RunConfig(instance_id=task["instance_id"], model=args.model, variant=variant,
                     base_url=args.base_url, thinking=args.thinking, max_turns=args.max_turns,
                     api_key=args.api_key)


def run_cell(args, task: dict, variant: str) -> tuple[dict, bool]:
    """Run one grid cell. Never raises: a failed cell simply leaves no completed trace.

    Returns (summary, failed)."""
    cfg = make_config(args, task, variant)
    try:
        return run_instance(cfg, task, verbose=args.workers == 1), False
    except Exception as e:  # keep the grid going; the cell simply has no trace
        with PRINT_LOCK:
            print(f"  FAILED {task['instance_id']} {variant}: {type(e).__name__}: {e}")
        return ({"instance_id": task["instance_id"], "variant": variant,
                 "stop_reason": f"harness_error:{type(e).__name__}"}, True)


def report(results: list[dict], todo: list, t0: float) -> None:
    print(f"\n=== batch done: {len(results)}/{len(todo)} run(s) in {time.time() - t0:.0f}s ===")
    by_stop: dict[str, int] = {}
    for s in results:
        by_stop[s.get("stop_reason", "?")] = by_stop.get(s.get("stop_reason", "?"), 0) + 1
    print(f"stop reasons: {by_stop}")
    edited = sum(1 for s in results if s.get("edits_attempted", 0) > 0)
    print(f"runs with >=1 edit: {edited}/{len(results)}")


def run_sequential(args, todo: list, t0: float) -> int:
    results: list[dict] = []
    consecutive_failures = 0
    for i, (task, variant) in enumerate(todo, 1):
        if not server_ok(args.base_url):
            print(f"\nABORT: server at {args.base_url} is not healthy; stopping the batch. "
                  f"{len(todo) - i + 1} cell(s) left (re-run to resume).")
            return 2
        print(f"\n[{i}/{len(todo)}] {task['instance_id']} {variant}  "
              f"(elapsed {time.time() - t0:.0f}s)")
        summary, failed = run_cell(args, task, variant)
        results.append(summary)
        if failed:
            consecutive_failures += 1
            if consecutive_failures >= 3:
                print(f"\nABORT: 3 consecutive failures; stopping the batch "
                      f"({len(todo) - i} cell(s) left, re-run to resume).")
                break
        else:
            consecutive_failures = 0
        if args.sleep:
            time.sleep(args.sleep)
    report(results, todo, t0)
    return 0


def run_concurrent(args, todo: list, t0: float) -> int:
    results: list[dict] = []
    stop = threading.Event()
    state = {"failures": 0}
    done = {"n": 0}

    def worker(task: dict, variant: str) -> dict | None:
        if stop.is_set():
            return None
        if not server_ok(args.base_url):
            with PRINT_LOCK:
                print(f"ABORT: server at {args.base_url} is not healthy; in-flight runs finish, "
                      f"no further cells start (re-run to resume).")
            stop.set()
            return None
        with PRINT_LOCK:
            print(f"[start] {task['instance_id']} {variant}")
        summary, failed = run_cell(args, task, variant)
        with PRINT_LOCK:
            done["n"] += 1
            if failed:
                state["failures"] += 1
                if state["failures"] >= 3:
                    print("ABORT: 3 failures; no further cells start (re-run to resume).")
                    stop.set()
            else:
                state["failures"] = 0
            print(f"[{done['n']}/{len(todo)}] {task['instance_id']} {variant} -> "
                  f"{summary.get('stop_reason', '?')} turns={summary.get('turns_used', '?')} "
                  f"tok={summary.get('total_tokens', '?')} edits={summary.get('edits_attempted', '?')}")
        return summary

    with ThreadPoolExecutor(max_workers=min(args.workers, len(todo))) as pool:
        futures = [pool.submit(worker, task, variant) for task, variant in todo]
        for future in as_completed(futures):
            summary = future.result()
            if summary is not None:
                results.append(summary)
    report(results, todo, t0)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-4-E2B")
    ap.add_argument("--variants", nargs="+", default=["B0", "M1", "M2", "M3"])
    ap.add_argument("--thinking", default="on", choices=["auto", "on", "off"])
    ap.add_argument("--max-turns", type=int, default=15)
    ap.add_argument("--repo", help="restrict to one repo")
    ap.add_argument("--limit", type=int, help="only the first N tasks")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--api-key", default=None,
                    help="bearer token when the server was started with --api-key (e.g. over a tunnel)")
    ap.add_argument("--workers", type=int, default=1,
                    help="run this many cells concurrently (needs a multi-slot backend)")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds to pause between runs")
    ap.add_argument("--no-skip", action="store_true", help="re-run cells that already have a trace")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = ap.parse_args()

    tasks = load_tasks()
    if args.repo:
        tasks = [t for t in tasks if t["repo"] == args.repo]
    if args.limit:
        tasks = tasks[: args.limit]

    cells = [(t, v) for t in tasks for v in args.variants]
    todo = []
    skipped = 0
    for task, variant in cells:
        if not args.no_skip and completed_run(task["instance_id"], args.model, variant,
                                              args.thinking, args.max_turns, False):
            skipped += 1
            continue
        todo.append((task, variant))

    print(f"grid: model={args.model} variants={args.variants} thinking={args.thinking} "
          f"max_turns={args.max_turns} workers={args.workers} base_url={args.base_url}")
    print(f"tasks={len(tasks)} cells={len(cells)}  already done={skipped}  to run={len(todo)}")
    if args.dry_run:
        for task, variant in todo[:20]:
            print(f"  would run {task['instance_id']:<32} {variant}")
        if len(todo) > 20:
            print(f"  ... and {len(todo) - 20} more")
        return 0

    if not todo:
        print("nothing to do")
        return 0

    t0 = time.time()
    if args.workers <= 1:
        return run_sequential(args, todo, t0)
    return run_concurrent(args, todo, t0)


if __name__ == "__main__":
    sys.exit(main())
