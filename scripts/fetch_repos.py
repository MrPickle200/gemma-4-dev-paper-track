"""Clone the repos needed by data/subset_50.json and verify every base_commit exists.

One full clone per repo into repos/<owner>__<name> (full history, because worktrees and
`git cat-file` need the pinned base_commit to be present locally). Per-task checkouts are
created lazily by harness/runner.py as git worktrees.

Usage:  .\\.venv\\Scripts\\python.exe scripts\\fetch_repos.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
REPOS = ROOT / "repos"


def repo_dir(repo: str) -> Path:
    return REPOS / repo.replace("/", "__")


def git(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def main() -> int:
    tasks = json.loads((ROOT / "data" / "subset_50.json").read_text(encoding="utf-8"))
    by_repo: dict[str, list[str]] = {}
    for t in tasks:
        by_repo.setdefault(t["repo"], []).append(t["base_commit"])

    REPOS.mkdir(exist_ok=True)
    failures = []

    for repo, commits in sorted(by_repo.items()):
        target = repo_dir(repo)
        if target.exists():
            print(f"[skip] {repo:<28} already cloned")
        else:
            print(f"[clone] {repo} -> {target.relative_to(ROOT)}")
            r = git(["clone", "--quiet", f"https://github.com/{repo}.git", str(target)])
            if r.returncode != 0:
                failures.append((repo, "clone failed", r.stderr.strip()[:300]))
                continue

        missing = [
            c for c in set(commits)
            if git(["cat-file", "-e", f"{c}^{{commit}}"], cwd=target).returncode != 0
        ]
        if missing:
            print(f"        fetching {len(missing)} missing commit(s) ...")
            git(["fetch", "--quiet", "origin"], cwd=target)
            missing = [
                c for c in missing
                if git(["cat-file", "-e", f"{c}^{{commit}}"], cwd=target).returncode != 0
            ]
        if missing:
            failures.append((repo, f"{len(missing)} commits not present", ", ".join(sorted(missing))[:200]))
            print(f"        MISSING: {missing}")
        else:
            size_mb = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / 1e6
            print(f"        ok: {len(set(commits))} commit(s) verified, {size_mb:.0f} MB, "
                  f"{len(commits)} task(s)")

    print()
    if failures:
        for repo, why, detail in failures:
            print(f"FAILED {repo}: {why} :: {detail}")
        return 1
    print(f"OK: {len(by_repo)} repos ready for {len(tasks)} tasks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
