"""Build the N=50 SWE-bench Verified Mini subset described in IDEA.md section 3.

Filter criteria (all must hold):
  1. repo is in the fast-test allow-list (pytest, requests, sympy, scikit-learn)
  2. gold patch touches exactly ONE file
  3. gold patch is < 50 changed lines (added + removed)

Output: data/subset_50.json  (list of SWE-bench instance dicts, plus a small
manifest under data/subset_50.meta.json for reproducibility).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset

DATASET = "princeton-nlp/SWE-bench_Verified"
ALLOWED_REPOS = {
    "pytest-dev/pytest",
    "psf/requests",
    "sympy/sympy",
    "scikit-learn/scikit-learn",
}
MAX_FILES = 1
MAX_LINES = 50

DIFF_FILE_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)


def patch_stats(patch: str) -> tuple[int, int, int]:
    """Return (n_files, n_added, n_removed) for a unified diff."""
    if not patch:
        return 0, 0, 0
    files = {m.group(2) for m in DIFF_FILE_RE.finditer(patch)}
    added = removed = 0
    for line in patch.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return len(files), added, removed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="size of the subset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/subset_50.json")
    args = ap.parse_args()

    print(f"[1/4] loading {DATASET} ...")
    ds = load_dataset(DATASET, split="test")
    print(f"      loaded {len(ds)} instances")

    print(f"[2/4] filtering (1 file, <{MAX_LINES} lines, repos={sorted(ALLOWED_REPOS)}) ...")
    kept, per_repo_all, per_repo_kept = [], Counter(), Counter()
    for row in ds:
        per_repo_all[row["repo"]] += 1
        if row["repo"] not in ALLOWED_REPOS:
            continue
        n_files, add, rem = patch_stats(row["patch"])
        if n_files == MAX_FILES and (add + rem) < MAX_LINES:
            rec = dict(row)
            rec["_n_files"] = n_files
            rec["_n_changed_lines"] = add + rem
            kept.append(rec)
            per_repo_kept[row["repo"]] += 1

    print(f"      {len(kept)} instances pass the filter")

    print(f"[3/4] sampling {args.n} (seed={args.seed}) ...")
    kept.sort(key=lambda r: r["instance_id"])
    rng = random.Random(args.seed)
    subset = rng.sample(kept, min(args.n, len(kept)))
    subset.sort(key=lambda r: r["instance_id"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(subset, indent=2), encoding="utf-8")

    meta = {
        "dataset": DATASET,
        "filter": {
            "allowed_repos": sorted(ALLOWED_REPOS),
            "max_files": MAX_FILES,
            "max_changed_lines": MAX_LINES,
        },
        "n_total_in_dataset": len(ds),
        "n_passing_filter": len(kept),
        "n_sampled": len(subset),
        "seed": args.seed,
        "repo_counts_total": dict(per_repo_all),
        "repo_counts_passing": dict(per_repo_kept),
        "instance_ids": [r["instance_id"] for r in subset],
    }
    Path("data/subset_50.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[4/4] subset composition")
    for repo, c in sorted(Counter(r["repo"] for r in subset).items()):
        print(f"      {repo:<32} {c:>3} / {per_repo_kept[repo]:<3} passing")
    lines = [r["_n_changed_lines"] for r in subset]
    print(f"      changed lines: min={min(lines)} max={max(lines)} mean={sum(lines)/len(lines):.1f}")
    print(f"      wrote {out} and data/subset_50.meta.json")


if __name__ == "__main__":
    main()
