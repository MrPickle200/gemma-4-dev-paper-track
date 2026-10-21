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
import csv
import glob
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests                                          # noqa: E402

from harness import config                                      # noqa: E402
from harness.config import (MIN_RUN_CTX, RunConfig, is_infra_failure,   # noqa: E402
                            is_proxy_capped, turn_error)
from harness.llm import LLMClient, verify_run_context           # noqa: E402
from harness.runner import load_tasks, run_instance        # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

PRINT_LOCK = threading.Lock()

# The per-cell columns of the experiment's results.csv. Fixed order so the file can be opened as a
# table while the batch is still running, and joined to logs/runs/*.jsonl by run_id afterwards.
RESULT_FIELDS = ("ts", "run_id", "instance_id", "variant", "model", "thinking", "stop_reason",
                 "turns_used", "total_tokens", "peak_prompt_tokens", "edits_attempted",
                 "edits_malformed", "edits_syntax_rejected", "edits_noop", "edits_ambiguous",
                 "malformed_arguments", "arguments_repaired", "repeated_identical_calls",
                 "loop_blocked", "lock_calls", "lock_rejected", "symbols_locked", "compactions",
                 "patch_chars", "diff_stat", "log_path", "patch_path")


def completed_run(instance_id: str, model: str, variant: str, thinking: str,
                  max_turns: int, line_numbers: bool,
                  rerun_proxy_capped: bool = False) -> str | None:
    """Return the run_id of a matching completed trace, or None."""
    pattern = str(config.RUNS / f"{instance_id}__{model}__{variant}__*.jsonl")
    for path in glob.glob(pattern):
        try:
            recs = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, json.JSONDecodeError):
            continue
        start = next((r for r in recs if r["type"] == "run_start"), None)
        end = next((r for r in recs if r["type"] == "run_end"), None)
        if not start or not end:
            continue
        if is_infra_failure(end.get("stop_reason"), turn_error(recs),
                            start.get("config", {}).get("ctx_per_run")):
            continue  # the backend died - or starved this run of context: it needs a real run
        cfg = start.get("config", {})
        if rerun_proxy_capped and is_proxy_capped(cfg):
            continue  # complete, but the tunnel's read timeout may have dropped long turns
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


def probe_workers(args) -> tuple[int, int | None]:
    """The number of concurrent cells this server can really serve, and the context each gets.

    llama-server hands out `-c / -np` per slot only when `-c` is big enough for per-slot caches;
    started with a small `-c` it keeps ONE shared KV pool (`kv_unified = true`), and `/props`
    advertises `n_ctx = 32768` in *both* cases. So the only trustworthy answer comes from asking
    the server to serve that many full-window prompts at once (TASK.md Task 6.2). When it cannot,
    the batch *lowers its concurrency* instead of letting the pool end long runs - a
    `context_exceeded` produced that way is a reason we chose, not one the model chose.
    """
    advertised = LLMClient(args.base_url, args.model, api_key=args.api_key).ctx_pool()
    print(f"server advertises n_ctx = {advertised}")
    workers = max(1, args.workers)
    while True:
        share = verify_run_context(args.base_url, workers, api_key=args.api_key)
        if share:
            return workers, share
        if workers == 1:
            print(f"WARNING: {args.base_url} cannot serve even one {MIN_RUN_CTX}-token run. An "
                  f"'Context size has been exceeded' here is a backend artefact and those cells "
                  f"are excluded, not scored. Relaunch with -c {MIN_RUN_CTX} -np 1.")
            return 1, None
        nxt = max(1, workers // 2)
        print(f"WARNING: {args.base_url} cannot serve {workers} concurrent {MIN_RUN_CTX}-token "
              f"runs. Lowering --workers to {nxt} (relaunch with "
              f"-c {MIN_RUN_CTX * workers} -np {workers} to use {workers}).")
        workers = nxt


def experiment_dir(args) -> Path:
    return config.EXPERIMENTS / args.experiment if args.experiment else config.LOGS


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT,
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


INVOCATION_KEYS = (
    "created", "command", "git_commit", "model", "variants", "thinking", "max_turns",
    "workers", "ctx_per_run", "base_url", "stream", "task_count", "task_ids",
    "planned_cells", "already_done",
)


def write_manifest(args, tasks: list, todo: list, skipped: int) -> None:
    """Record what this invocation is, next to the traces it writes (so a run set is reusable).

    Written before the first cell, so a batch that is interrupted is still identifiable. A resume
    or a single-cell re-run appends to `invocations` and unions the task ids; it must not replace
    the plan that created the experiment, or the directory would look like a one-task run.
    """
    if not args.experiment:
        return
    out = experiment_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    invocation = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "command": subprocess.list2cmdline([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "git_commit": git_commit(),
        "model": args.model,
        "variants": args.variants,
        "thinking": args.thinking,
        "max_turns": args.max_turns,
        "workers": args.workers,
        "ctx_per_run": args.ctx_per_run,
        "base_url": args.base_url,
        "stream": not args.no_stream,
        "task_count": len(tasks),
        "task_ids": [t["instance_id"] for t in tasks],
        "planned_cells": len(todo),
        "already_done": skipped,
    }

    path = out / "manifest.json"
    prev = {}
    if path.exists():
        try:
            prev = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            prev = {}
    history = prev.get("invocations") or ([prev] if prev else [])
    history = [{k: inv.get(k) for k in INVOCATION_KEYS} for inv in history] + [invocation]

    task_ids = list(dict.fromkeys(
        [i for inv in history for i in (inv.get("task_ids") or [])]
    ))
    payload = {k: history[0].get(k) for k in INVOCATION_KEYS}
    payload.update({
        "experiment": args.experiment,
        "task_count": len(task_ids),
        "task_ids": task_ids,
        "runs_dir": str(config.RUNS.relative_to(config.ROOT)),
        "patches_dir": str(config.PATCHES.relative_to(config.ROOT)),
        "invocations": history,
    })
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_result(args, summary: dict, variant: str) -> None:
    """One results.csv row per finished cell - the batch table, written as the batch runs."""
    if not args.experiment:
        return
    out = experiment_dir(args) / "results.csv"
    row = {k: summary.get(k) for k in RESULT_FIELDS}
    row.update({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "variant": variant,
                "model": args.model, "thinking": args.thinking})
    with PRINT_LOCK:
        new = not out.exists()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(RESULT_FIELDS))
            if new:
                w.writeheader()
            w.writerow({k: ("" if row.get(k) is None else row[k]) for k in RESULT_FIELDS})


def make_config(args, task: dict, variant: str) -> RunConfig:
    return RunConfig(instance_id=task["instance_id"], model=args.model, variant=variant,
                     base_url=args.base_url, thinking=args.thinking, max_turns=args.max_turns,
                     api_key=args.api_key, stream=not args.no_stream,
                     ctx_per_run=getattr(args, "ctx_per_run", None))


def run_cell(args, task: dict, variant: str) -> tuple[dict, bool]:
    """Run one grid cell. Never raises: a failed cell simply leaves no completed trace.

    Returns (summary, failed)."""
    cfg = make_config(args, task, variant)
    try:
        summary, failed = run_instance(cfg, task, verbose=args.workers == 1), False
    except Exception as e:  # keep the grid going; the cell simply has no trace
        with PRINT_LOCK:
            print(f"  FAILED {task['instance_id']} {variant}: {type(e).__name__}: {e}")
        return ({"instance_id": task["instance_id"], "variant": variant,
                 "stop_reason": f"harness_error:{type(e).__name__}"}, True)
    append_result(args, summary, variant)
    return summary, failed


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
    ap.add_argument("--instance-id", action="append", default=[],
                    help="only this task (repeatable) - use it to re-run specific cells")
    ap.add_argument("--limit", type=int, help="only the first N tasks")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--api-key", default=None,
                    help="bearer token when the server was started with --api-key (e.g. over a tunnel)")
    ap.add_argument("--experiment", default=None,
                    help="write this batch to logs/experiments/<TAG>/ (runs, patches, manifest, "
                         "results.csv) instead of the shared logs/runs/ + logs/patches/")
    ap.add_argument("--workers", type=int, default=1,
                    help="run this many cells concurrently (needs a multi-slot backend; reduced "
                         "automatically if the server cannot serve that many full windows)")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds to pause between runs")
    ap.add_argument("--no-skip", action="store_true", help="re-run cells that already have a trace")
    ap.add_argument("--rerun-proxy-capped", action="store_true",
                    help="also re-run cells whose only complete trace ran buffered through a tunnel. "
                         "Such a run is complete, but the proxy's read timeout can have killed long "
                         "turns, which biases which runs survive; streaming removes that timeout")
    ap.add_argument("--no-stream", action="store_true",
                    help="buffer whole responses instead of SSE (pre-2026-10-17 behaviour; "
                         "dies on any turn longer than the proxy's read timeout)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = ap.parse_args()

    # Point RUNS/PATCHES at the experiment before any RunConfig is built, so the traces, the
    # patches, the manifest and results.csv of one batch live in one directory.
    exp = config.set_experiment(args.experiment) if args.experiment else None
    if exp is not None:
        print(f"experiment: {args.experiment} -> {exp.relative_to(config.ROOT)}")
    args.workers, args.ctx_per_run = probe_workers(args)

    tasks = load_tasks()
    if args.repo:
        tasks = [t for t in tasks if t["repo"] == args.repo]
    if args.instance_id:
        wanted = set(args.instance_id)
        tasks = [t for t in tasks if t["instance_id"] in wanted]
    if args.limit:
        tasks = tasks[: args.limit]

    cells = [(t, v) for t in tasks for v in args.variants]
    todo = []
    skipped = 0
    for task, variant in cells:
        if not args.no_skip and completed_run(task["instance_id"], args.model, variant,
                                              args.thinking, args.max_turns, False,
                                              args.rerun_proxy_capped):
            skipped += 1
            continue
        todo.append((task, variant))

    print(f"grid: model={args.model} variants={args.variants} thinking={args.thinking} "
          f"max_turns={args.max_turns} workers={args.workers} base_url={args.base_url}")
    print(f"tasks={len(tasks)} cells={len(cells)}  already done={skipped}  to run={len(todo)}")
    ctx_label = (f"{args.ctx_per_run} tokens per run, verified for {args.workers} concurrent run(s)"
                 if args.ctx_per_run else
                 "UNVERIFIED - any 'context_exceeded' here is a backend artefact, not the model's")
    print(f"context: {ctx_label}")
    if args.dry_run:
        for task, variant in todo[:20]:
            print(f"  would run {task['instance_id']:<32} {variant}")
        if len(todo) > 20:
            print(f"  ... and {len(todo) - 20} more")
        return 0

    write_manifest(args, tasks, todo, skipped)
    if not todo:
        print("nothing to do")
        return 0

    t0 = time.time()
    if args.workers <= 1:
        return run_sequential(args, todo, t0)
    return run_concurrent(args, todo, t0)


if __name__ == "__main__":
    sys.exit(main())
