#!/usr/bin/env python3
"""Build M4-Native SFT Dataset from SWE-smith and SWE-Gym.

Generates a 70% Phase-2 (Patching) / 30% Phase-1 (Anti-Anchoring Localization)
dataset in OpenAI chat completions format, matching the exact tool schemas
and prompt structures of the Micro-Model Native Harness (M4).
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import random
import re
import ssl
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow.parquet as pq

# Benchmark evaluation repositories to strictly avoid (zero contamination)
CONTAMINATED_REPO_KEYWORDS = ["requests", "pytest", "scikit-learn", "sympy"]

SWE_GYM_PARQUET_URL = (
    "https://huggingface.co/datasets/SWE-Gym/SWE-Gym/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
)

SWE_SMITH_PARQUET_URLS = [
    f"https://huggingface.co/datasets/SWE-bench/SWE-smith/resolve/refs%2Fconvert%2Fparquet/default/train/{i:04d}.parquet"
    for i in range(11)
]

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE


def fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=SSL_CTX) as resp:
        return resp.read()


def fetch_text(url: str) -> str:
    return fetch_bytes(url).decode("utf-8", errors="replace")


def parse_unified_diff(patch: str) -> Optional[Tuple[str, str, str, int, int]]:
    """Parse a single-file, single-hunk unified diff.
    
    Returns (target_file, search_block, replace_block, start_line, end_line)
    """
    files = re.findall(r"diff --git a/(.*?) b/", patch)
    if len(files) != 1:
        return None
    target_file = files[0]

    hunk_matches = list(re.finditer(r"@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@", patch))
    if len(hunk_matches) != 1:
        return None

    hunk = hunk_matches[0]
    start_line = int(hunk.group(1))
    line_count = int(hunk.group(2)) if hunk.group(2) else 1
    end_line = start_line + line_count - 1

    lines = patch.splitlines()
    search_lines = []
    replace_lines = []
    in_hunk = False

    for line in lines:
        if line.startswith("@@"):
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if line.startswith("---") or line.startswith("+++") or line.startswith("diff "):
            break
        if line.startswith("-"):
            search_lines.append(line[1:])
        elif line.startswith("+"):
            replace_lines.append(line[1:])
        elif line.startswith(" "):
            search_lines.append(line[1:])
            replace_lines.append(line[1:])

    search_block = "\n".join(search_lines)
    replace_block = "\n".join(replace_lines)

    if not search_block.strip() or search_block == replace_block:
        return None

    return target_file, search_block, replace_block, start_line, end_line


def find_enclosing_symbol(code: str, start_line: int, end_line: int) -> Tuple[Optional[str], Optional[Tuple[int, int]]]:
    try:
        tree = ast.parse(code)
    except Exception:
        return None, None

    best_sym = None
    best_range = (0, 10**9)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            s_lineno = getattr(node, "lineno", 0)
            e_lineno = getattr(node, "end_lineno", s_lineno + 50)
            if s_lineno <= start_line and e_lineno >= end_line:
                if (e_lineno - s_lineno) < (best_range[1] - best_range[0]):
                    best_range = (s_lineno, e_lineno)
                    best_sym = node.name

    return best_sym, best_range if best_sym else None


def fetch_file_content(dataset_type: str, row: Dict[str, Any], target_file: str) -> Optional[str]:
    try:
        if dataset_type == "swe_gym":
            repo = row["repo"]
            base_commit = row["base_commit"]
            raw_url = f"https://raw.githubusercontent.com/{repo}/{base_commit}/{target_file}"
            return fetch_text(raw_url)
        elif dataset_type == "swe_smith":
            repo_short = row["repo"].replace("swesmith/", "")
            raw_url = f"https://raw.githubusercontent.com/swesmith/{repo_short}/main/{target_file}"
            return fetch_text(raw_url)
    except Exception:
        return None
    return None


def build_phase2_example(
    instance_id: str,
    problem_statement: str,
    target_file: str,
    symbol_name: str,
    symbol_code_no_lines: str,
    start_line: int,
    end_line: int,
    search_block: str,
    replace_block: str,
    sys_prompt: str,
    phase2_directive: str,
) -> Dict[str, Any]:
    note = f"Fix bug in {target_file}::{symbol_name} as described in the issue."
    user_prompt = (
        "# Phase 2 - Patching\n\n"
        "Your localization history has been cleared. Everything you need is below.\n\n"
        f"{phase2_directive}\n\n"
        f"## Issue\n{problem_statement}\n\n"
        f"## Root cause (your note)\n{note}\n\n"
        "## Locked code (copy text verbatim into search_block; no line numbers)\n"
        f"=== {target_file} :: {symbol_name} (lines {start_line}-{end_line}) ===\n"
        f"{symbol_code_no_lines}"
    )

    assistant_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_edit_0",
                "type": "function",
                "function": {
                    "name": "apply_edit",
                    "arguments": json.dumps(
                        {
                            "file": target_file,
                            "target_symbol": symbol_name,
                            "search_block": search_block,
                            "replace_block": replace_block,
                        },
                        ensure_ascii=False,
                    ),
                },
            }
        ],
    }

    return {
        "instance_id": instance_id,
        "phase": 2,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
            assistant_message,
        ],
    }


def build_phase1_example(
    instance_id: str,
    problem_statement: str,
    target_file: str,
    symbol_name: str,
    anti_anchoring_mode: str,
    sys_prompt: str,
) -> Dict[str, Any]:
    dummy_candidates = [
        f"{target_file}::dummy_helper_{i}" for i in range(1, 8)
    ]

    if anti_anchoring_mode == "rank_4":
        candidates = dummy_candidates[:3] + [f"{target_file}::{symbol_name}"] + dummy_candidates[3:7]
    elif anti_anchoring_mode == "rank_7":
        candidates = dummy_candidates[:6] + [f"{target_file}::{symbol_name}"] + dummy_candidates[6:7]
    else:
        candidates = dummy_candidates[:8]

    candidate_block = "\n".join(f"{i+1}. `{c}`" for i, c in enumerate(candidates))

    user_prompt = (
        f"### Issue Description:\n{problem_statement}\n\n"
        "=== AUTO-RETRIEVED CANDIDATE SYMBOLS (Turn-0 AST-Semantic Seeding) ===\n"
        "The following existing symbols in the repository were automatically ranked as most relevant to the issue description. Use exact IDs from this list when calling `read_symbol` or `lock_targets`:\n"
        f"{candidate_block}\n\n"
        "Find the fault location and lock targets."
    )

    if anti_anchoring_mode in ["rank_4", "rank_7"]:
        assistant_turn1 = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_nav_0",
                    "type": "function",
                    "function": {
                        "name": "read_symbol",
                        "arguments": json.dumps(
                            {"file": target_file, "symbol_name": symbol_name},
                            ensure_ascii=False,
                        ),
                    },
                }
            ],
        }
        tool_result1 = {
            "role": "tool",
            "tool_call_id": "call_nav_0",
            "name": "read_symbol",
            "content": f"[Source for {target_file}::{symbol_name} successfully read]",
        }
        assistant_turn2 = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_lock_1",
                    "type": "function",
                    "function": {
                        "name": "lock_targets",
                        "arguments": json.dumps(
                            {
                                "files": [target_file],
                                "symbols": [f"{target_file}::{symbol_name}"],
                                "root_cause_note": f"Confirmed root cause is in {symbol_name}.",
                            },
                            ensure_ascii=False,
                        ),
                    },
                }
            ],
        }
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
            assistant_turn1,
            tool_result1,
            assistant_turn2,
        ]
    else:
        assistant_turn1 = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_nav_0",
                    "type": "function",
                    "function": {
                        "name": "get_repo_map",
                        "arguments": json.dumps({"path": target_file.split("/")[0]}),
                    },
                }
            ],
        }
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
            assistant_turn1,
        ]

    return {
        "instance_id": instance_id,
        "phase": 1,
        "anti_anchoring_mode": anti_anchoring_mode,
        "messages": messages,
    }


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from harness.agent import system_prompt, PHASE2_DIRECTIVE

    sys_m4 = system_prompt("M4")

    parser = argparse.ArgumentParser(description="Build M4 SFT dataset")
    parser.add_argument("--total", type=int, default=3000, help="Total examples to generate")
    parser.add_argument("--output", type=str, default="data/sft_m4_3000.jsonl", help="Output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    target_phase2 = int(args.total * 0.70)
    target_phase1 = args.total - target_phase2
    print(f"Target mixture: {target_phase2} Phase-2 (Patching) + {target_phase1} Phase-1 (Anti-Anchoring)")

    target_p2_gym = int(target_phase2 * 0.25)
    target_p2_smith = target_phase2 - target_p2_gym

    phase2_examples = []
    phase1_examples = []

    print("\n--- 1. Processing SWE-Gym Parquet ---")
    data_gym = fetch_bytes(SWE_GYM_PARQUET_URL)
    table_gym = pq.read_table(io.BytesIO(data_gym))
    df_gym = table_gym.to_pandas()

    gym_candidates = []
    for _, row in df_gym.iterrows():
        parsed = parse_unified_diff(row["patch"])
        if parsed:
            gym_candidates.append((row, parsed))

    random.shuffle(gym_candidates)
    print(f"Found {len(gym_candidates)} valid single-hunk tasks in SWE-Gym")

    for row, parsed in gym_candidates:
        if len(phase2_examples) >= target_p2_gym:
            break
        target_file, search_block, replace_block, s_line, e_line = parsed
        code = fetch_file_content("swe_gym", row, target_file)
        if not code:
            continue
        sym_name, s_range = find_enclosing_symbol(code, s_line, e_line)
        if not sym_name or not s_range:
            continue

        lines = code.splitlines()
        sym_lines = lines[max(0, s_range[0] - 1) : min(len(lines), s_range[1])]
        sym_code = "\n".join(sym_lines)

        ex = build_phase2_example(
            row["instance_id"],
            row["problem_statement"],
            target_file,
            sym_name,
            sym_code,
            s_range[0],
            s_range[1],
            search_block,
            replace_block,
            sys_m4,
            PHASE2_DIRECTIVE,
        )
        phase2_examples.append(ex)

    print(f"Successfully constructed {len(phase2_examples)} Phase-2 examples from SWE-Gym")

    print("\n--- 2. Processing SWE-smith Parquets ---")
    smith_candidates = []
    for p_url in SWE_SMITH_PARQUET_URLS[:4]:
        if len(smith_candidates) >= (target_p2_smith + target_phase1) * 2:
            break
        print(f"Fetching {p_url.split('/')[-1]}...")
        data_s = fetch_bytes(p_url)
        table_s = pq.read_table(io.BytesIO(data_s))
        df_s = table_s.to_pandas()
        for _, row in df_s.iterrows():
            if any(k in row["repo"].lower() for k in CONTAMINATED_REPO_KEYWORDS):
                continue
            parsed = parse_unified_diff(row["patch"])
            if parsed:
                smith_candidates.append((row, parsed))

    random.shuffle(smith_candidates)
    print(f"Found {len(smith_candidates)} valid candidates in processed SWE-smith chunks")

    for row, parsed in smith_candidates:
        if len(phase2_examples) >= target_phase2 and len(phase1_examples) >= target_phase1:
            break
        target_file, search_block, replace_block, s_line, e_line = parsed
        code = fetch_file_content("swe_smith", row, target_file)
        if not code:
            continue
        sym_name, s_range = find_enclosing_symbol(code, s_line, e_line)
        if not sym_name or not s_range:
            continue

        if len(phase2_examples) < target_phase2:
            lines = code.splitlines()
            sym_lines = lines[max(0, s_range[0] - 1) : min(len(lines), s_range[1])]
            sym_code = "\n".join(sym_lines)
            ex2 = build_phase2_example(
                row["instance_id"],
                row["problem_statement"],
                target_file,
                sym_name,
                sym_code,
                s_range[0],
                s_range[1],
                search_block,
                replace_block,
                sys_m4,
                PHASE2_DIRECTIVE,
            )
            phase2_examples.append(ex2)
        elif len(phase1_examples) < target_phase1:
            mode = random.choice(["rank_4", "rank_7", "omitted"])
            ex1 = build_phase1_example(
                row["instance_id"],
                row["problem_statement"],
                target_file,
                sym_name,
                mode,
                sys_m4,
            )
            phase1_examples.append(ex1)

    all_examples = phase2_examples + phase1_examples
    random.shuffle(all_examples)

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nSuccessfully generated {len(all_examples)} examples to {out_p}")
    print(f"  - Phase 2 (Patching): {len(phase2_examples)}")
    print(f"  - Phase 1 (Anti-Anchoring): {len(phase1_examples)}")


if __name__ == "__main__":
    main()
