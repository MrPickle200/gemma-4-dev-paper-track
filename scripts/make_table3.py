import csv
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

def read_swe(p):
    rows = {}
    if not Path(p).exists():
        return rows
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["model"]] = r
    return rows

def read_results(p):
    rows = {}
    if not Path(p).exists():
        return rows
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = f"{r['model']}_{r['variant']}"
            rows[k] = r
    return rows

e2b_swe = read_swe("experiments/20261023-e2b-grid/swebench_results.csv")
e4b_swe = read_swe("experiments/20261023-e4b-grid/swebench_results.csv")
e2b_m4_swe = read_swe("experiments/20261026-e2b-m4/swebench_results.csv")
e4b_m4_swe = read_swe("experiments/20261026-e4b-m4/swebench_results.csv")

e2b_res = read_results("experiments/20261023-e2b-grid/results.csv")
e4b_res = read_results("experiments/20261023-e4b-grid/results.csv")
e2b_m4_res = read_results("experiments/20261026-e2b-m4/results.csv")
e4b_m4_res = read_results("experiments/20261026-e4b-m4/results.csv")

all_swe = {**e2b_swe, **e4b_swe, **e2b_m4_swe, **e4b_m4_swe}
all_res = {**e2b_res, **e4b_res, **e2b_m4_res, **e4b_m4_res}

print("| Model & Variant | Resolved | Resolve rate (%) | Edited | Real edits | Valid Patch | Malformed | Mean tok | Finish | Resolved IDs |")
print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")

order = [
    "gemma-4-E2B_B0", "gemma-4-E2B_M1", "gemma-4-E2B_M2", "gemma-4-E2B_M3", "gemma-4-E2B_M4",
    "gemma-4-E4B_B0", "gemma-4-E4B_M1", "gemma-4-E4B_M2", "gemma-4-E4B_M3", "gemma-4-E4B_M4",
]

for k in order:
    if k not in all_swe:
        continue
    sw = all_swe[k]
    res = all_res.get(k, {})
    res_cnt = sw.get("resolved", "0")
    rate = f"{int(res_cnt) / 50 * 100:.1f}%"
    edited = res.get("runs_with_edit", "n/a")
    realed = res.get("runs_real_edit", "n/a")
    empty_p = sw.get("empty_patch", "0")
    valid_patch = f"{50 - int(empty_p)}/50" if empty_p.isdigit() else "n/a"
    malf = res.get("malformed_edit_rate", "n/a")
    tok = f"{int(res.get('mean_total_tokens', 0)):,}" if res.get('mean_total_tokens') else "n/a"
    fin = res.get("finished", "n/a")
    r_ids = sw.get("resolved_ids", "")
    id_list = ", ".join(f"`{x}`" for x in r_ids.split()) if r_ids else "-"
    print(f"| `{k}` | {res_cnt} / 50 | **{rate}** | {edited} | {realed} | {valid_patch} | {malf} | {tok} | {fin} | {id_list} |")
