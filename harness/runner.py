"""Task loading, per-task git worktree setup, patch collection, run orchestration."""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path

from .agent import JsonlLogger, run_agent
from .config import REPOS, ROOT, WORKTREES, RunConfig
from .llm import LLMClient
from .tools import ToolBox

SUBSET = ROOT / "data" / "subset_50.json"

# The Bước 4 grid can run several cells at once (run_batch.py --workers N). Git's worktree
# administration of one clone is not safe to touch from two threads at once, so add/remove are
# serialised per repo. Reentrant because setup_worktree() calls teardown_worktree() while holding it.
_WORKTREE_LOCKS: dict[str, threading.RLock] = {}
_WORKTREE_LOCKS_GUARD = threading.Lock()


def _repo_lock(repo: str) -> threading.RLock:
    with _WORKTREE_LOCKS_GUARD:
        lock = _WORKTREE_LOCKS.get(repo)
        if lock is None:
            lock = _WORKTREE_LOCKS[repo] = threading.RLock()
        return lock


def load_tasks() -> list[dict]:
    return json.loads(SUBSET.read_text(encoding="utf-8"))


def get_task(instance_id: str) -> dict:
    for t in load_tasks():
        if t["instance_id"] == instance_id:
            return t
    raise KeyError(f"{instance_id} is not in {SUBSET.name}")


def repo_path(repo: str) -> Path:
    return REPOS / repo.replace("/", "__")


def git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def worktree_path(task: dict, variant: str | None = None) -> Path:
    """A worktree per (task, variant).

    With run_batch.py --workers N, several variants of one task run at the same time; keying the
    path by instance_id alone would make them tear down each other's checkout."""
    name = task["instance_id"] if not variant else f"{task['instance_id']}__{variant}"
    return WORKTREES / name


def setup_worktree(task: dict, variant: str | None = None) -> Path:
    """Create a detached worktree pinned at the task's base_commit. Never mutates the clone."""
    src = repo_path(task["repo"])
    if not src.exists():
        raise FileNotFoundError(f"{src} missing - run scripts\\fetch_repos.py first")
    dest = worktree_path(task, variant)
    WORKTREES.mkdir(parents=True, exist_ok=True)
    with _repo_lock(task["repo"]):
        if dest.exists():
            teardown_worktree(task, variant, quiet=True)
        r = git(["worktree", "add", "--detach", "--force", str(dest), task["base_commit"]], cwd=src)
    if r.returncode != 0:
        raise RuntimeError(f"git worktree add failed: {r.stderr.strip()}")
    return dest


def teardown_worktree(task: dict, variant: str | None = None, quiet: bool = False) -> None:
    dest = worktree_path(task, variant)
    src = repo_path(task["repo"])
    with _repo_lock(task["repo"]):
        if dest.exists() and src.exists():
            r = git(["worktree", "remove", "--force", str(dest)], cwd=src)
            if r.returncode != 0 and not quiet:
                print(f"  warning: could not remove worktree: {r.stderr.strip()}")
        if src.exists():
            git(["worktree", "prune"], cwd=src)


def collect_patch(workdir: Path) -> str:
    """The model's diff — this is what Bước 5 feeds to the SWE-bench evaluator."""
    r = git(["diff"], cwd=workdir)
    return r.stdout


def run_instance(cfg: RunConfig, task: dict | None = None, verbose: bool = True,
                 keep_worktree: bool = False) -> dict:
    task = task or get_task(cfg.instance_id)
    workdir = setup_worktree(task, cfg.variant)
    cfg.workdir = workdir

    client = LLMClient(
        base_url=cfg.base_url, model=cfg.model, temperature=cfg.temperature,
        max_tokens=cfg.max_tokens, thinking=cfg.thinking, api_key=cfg.api_key,
    )
    toolbox = ToolBox(workdir=workdir, max_output_chars=cfg.max_tool_output_chars,
                      repo_name=task["repo"], line_numbers=cfg.line_numbers,
                      variant=cfg.variant)
    logger = JsonlLogger(cfg.log_path)

    print(f"run {cfg.run_id}")
    print(f"  repo {task['repo']} @ {task['base_commit'][:10]}  worktree {workdir.relative_to(ROOT)}")
    try:
        summary = run_agent(cfg, task, client, toolbox, logger, verbose=verbose)
        patch = collect_patch(workdir)
        cfg.patch_path.write_text(patch, encoding="utf-8")
        stat = git(["diff", "--shortstat"], cwd=workdir).stdout.strip()
        logger.write({
            "type": "artifacts",
            "patch_path": str(cfg.patch_path.relative_to(ROOT)),
            "patch_chars": len(patch),
            "diff_stat": stat,
            "patch_is_empty": patch.strip() == "",
        })
        summary["patch_chars"] = len(patch)
        summary["diff_stat"] = stat
        summary["log_path"] = str(cfg.log_path.relative_to(ROOT))
        summary["patch_path"] = str(cfg.patch_path.relative_to(ROOT))
    finally:
        logger.close()
        if not keep_worktree:
            teardown_worktree(task, cfg.variant)

    print(f"  stop={summary['stop_reason']} turns={summary['turns_used']} "
          f"tok={summary['total_tokens']} peak_ctx={summary['peak_prompt_tokens']} "
          f"edits={summary['edits_attempted']}/malformed={summary['edits_malformed']}")
    print(f"  diff: {summary.get('diff_stat') or '(empty)'}")
    print(f"  log:  {summary['log_path']}")
    return summary
