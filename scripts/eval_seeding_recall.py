"""Evaluate Zero-Turn Seeding Recall (File & Symbol) on the 50-task SWE-bench Verified subset.

Compares top-K retrieved symbols in cache/seeding/{instance_id}.json against the ground-truth
modified files and AST symbols in the Gold Patch.

Usage:
  .\\.venv\\Scripts\\python.exe scripts\\eval_seeding_recall.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.astnav import symbols as get_file_symbols
from harness.runner import setup_worktree, teardown_worktree


def parse_patch_hunks(patch_text: str) -> dict[str, list[tuple[int, int]]]:
    """Extract {file_path: [(start_line, line_count), ...]} from a unified diff."""
    file_lines: dict[str, list[tuple[int, int]]] = {}
    current_file = None
    for line in patch_text.splitlines():
        if line.startswith("--- a/"):
            current_file = line[6:].strip()
        elif line.startswith("+++ b/"):
            current_file = line[6:].strip()
            if current_file not in file_lines:
                file_lines[current_file] = []
        elif line.startswith("@@") and current_file:
            m = re.search(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if m:
                old_start = int(m.group(1))
                old_len = int(m.group(2)) if m.group(2) else 1
                file_lines[current_file].append((old_start, max(old_len, 1)))
    return file_lines


def get_gold_for_task(task: dict) -> tuple[set[str], set[str]]:
    """Return (gold_files, gold_symbols) for a SWE-bench task."""
    hunks = parse_patch_hunks(task["patch"])
    gold_files = set(hunks.keys())
    gold_syms = set()

    workdir = setup_worktree(task, "recall_eval")
    try:
        for f, ranges in hunks.items():
            p = workdir / f
            if not p.is_file():
                continue
            file_symbols = get_file_symbols(p, f)
            for s in file_symbols:
                for start, length in ranges:
                    end = start + length - 1
                    # Overlap check
                    if not (s.end_line < start or s.start_line > end):
                        gold_syms.add(f"{s.file}::{s.qualified}")
                        # Also include bare class / bare function names as variants if needed
    finally:
        teardown_worktree(task, "recall_eval")

    return gold_files, gold_syms


def main():
    subset_path = ROOT / "data/subset_50.json"
    subset = json.loads(subset_path.read_text(encoding="utf-8"))
    cache_dir = ROOT / "cache/seeding"

    print(f"Loaded {len(subset)} tasks from {subset_path.relative_to(ROOT)}")

    gold_cache_file = ROOT / "cache/gold_symbols_50.json"
    if gold_cache_file.exists():
        print(f"Loading cached gold definitions from {gold_cache_file.relative_to(ROOT)}...")
        raw = json.loads(gold_cache_file.read_text(encoding="utf-8"))
        gold_data = {k: (set(v["files"]), set(v["symbols"])) for k, v in raw.items()}
    else:
        print("Extracting gold files & symbols across all 50 tasks (one-time setup)...")
        gold_data = {}
        serializable = {}
        for i, task in enumerate(subset, 1):
            iid = task["instance_id"]
            print(f"  [{i:>2}/50] Extracting gold for {iid}...")
            files, syms = get_gold_for_task(task)
            gold_data[iid] = (files, syms)
            serializable[iid] = {"files": sorted(files), "symbols": sorted(syms)}
        gold_cache_file.parent.mkdir(parents=True, exist_ok=True)
        gold_cache_file.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        print(f"Saved gold definitions cache to {gold_cache_file.relative_to(ROOT)}")

    # Metrics counters
    ks = [1, 3, 5, 8]
    file_hits = {k: 0 for k in ks}
    sym_hits = {k: 0 for k in ks}
    total = len(subset)

    per_task_results = []

    for task in subset:
        iid = task["instance_id"]
        gold_files, gold_syms = gold_data[iid]

        cache_path = cache_dir / f"{iid}.json"
        if not cache_path.exists():
            print(f"Warning: missing cache for {iid}")
            continue

        c = json.loads(cache_path.read_text(encoding="utf-8"))
        top_symbols = c.get("top_symbols", [])

        # Check hits at each K
        task_res = {"instance_id": iid, "gold_files": list(gold_files), "gold_symbols": list(gold_syms)}
        for k in ks:
            cand_k = top_symbols[:k]
            cand_sym_ids = {item["symbol_id"] for item in cand_k}
            cand_files = {item["symbol_id"].split("::")[0] for item in cand_k}

            f_hit = bool(cand_files & gold_files)
            # Symbol hit: either exact symbol_id match, or parent class match if class modified
            s_hit = bool(cand_sym_ids & gold_syms)

            if f_hit:
                file_hits[k] += 1
            if s_hit:
                sym_hits[k] += 1

            task_res[f"file_hit_{k}"] = f_hit
            task_res[f"sym_hit_{k}"] = s_hit

        per_task_results.append(task_res)

    print("\n" + "=" * 65)
    print("  BẢNG 1: HIỆU NĂNG ĐỊNH VỊ ZERO-TURN SEEDING (EMBEDDING GEMMA 2)")
    print("=" * 65)
    print(f"{'Metric':<30} | {'Count (Hits/50)':<16} | {'Recall Rate (%)':<15}")
    print("-" * 65)
    for k in ks:
        fh = file_hits[k]
        fr = (fh / total) * 100
        print(f"Top-{k:<2} File Hit Rate            | {fh:>2}/{total:<2}           | {fr:>6.1f}%")
    print("-" * 65)
    for k in ks:
        sh = sym_hits[k]
        sr = (sh / total) * 100
        print(f"Top-{k:<2} Symbol Hit Rate          | {sh:>2}/{total:<2}           | {sr:>6.1f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()
