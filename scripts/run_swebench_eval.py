"""Run the SWE-bench harness over every per-cell predictions file of an experiment.

`run_evaluation` keys predictions by instance_id, and our four variants share the
same instance_id, so each (model, variant) file must be its own run with its own
--run_id.  Reports land in --report_dir as <model>.<run_id>.json; this script then
merges them into results.csv, one row per cell.

    python scripts/run_swebench_eval.py --experiment 20261023-e2b-grid --max-workers 4
    python scripts/run_swebench_eval.py --experiment 20261023-e2b-grid --merge-only
"""

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys

DATASET = "SWE-bench/SWE-bench_Verified"
PYTHON = sys.executable


def report_path(report_dir, model_name, run_id):
    return os.path.join(report_dir, "{}.{}.json".format(model_name, run_id))


def harvest_eval_logs(run_id, log_root):
    """Move swebench's `logs/run_evaluation/<run_id>` into the experiment bundle."""
    src = os.path.join("logs", "run_evaluation", run_id)
    if not os.path.isdir(src):
        return
    os.makedirs(log_root, exist_ok=True)
    dst = os.path.join(log_root, run_id)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.move(src, dst)
    # swebench recreates logs/run_evaluation on every run; drop the empty shell so the repo
    # keeps a single top-level evidence directory.
    for d in (os.path.join("logs", "run_evaluation"), os.path.join("logs")):
        try:
            os.rmdir(d)
        except OSError:
            break


def run_one(pred_path, report_dir, run_id, max_workers, timeout, log_root):
    model_name = os.path.basename(pred_path)[: -len(".jsonl")]
    cmd = [
        PYTHON,
        "-m",
        "swebench.harness.run_evaluation",
        "-d",
        DATASET,
        "-s",
        "test",
        "-p",
        pred_path,
        "-id",
        run_id,
        "-t",
        str(timeout),
        "--max_workers",
        str(max_workers),
        "--report_dir",
        report_dir,
    ]
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    print("[eval] {} -> run_id={}".format(model_name, run_id), flush=True)
    proc = subprocess.run(cmd, env=env)
    # swebench hardcodes its per-instance scratch dir to logs/run_evaluation/<run_id>; move it
    # under experiments/<tag>/run_evaluation/ so an experiment owns everything about itself.
    harvest_eval_logs(run_id, log_root)
    out = report_path(report_dir, model_name, run_id)
    if not os.path.exists(out):
        print("[eval] {} produced no report (exit {})".format(model_name, proc.returncode))
        return None
    with open(out, encoding="utf-8") as f:
        return json.load(f)


def summarize(report, model_name):
    return {
        "model": model_name,
        "resolved": report.get("resolved_instances", 0),
        "unresolved": report.get("unresolved_instances", 0),
        "error": report.get("error_instances", 0),
        "empty_patch": report.get("empty_patch_instances", 0),
        "infra_failure": report.get("infra_failure_instances", 0),
        "submitted": report.get("submitted_instances", 0),
        "resolved_ids": " ".join(report.get("resolved_ids", [])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="discard existing reports and re-run")
    ap.add_argument("--only", default=None, help="substring filter on cell file name")
    a = ap.parse_args()

    root = os.path.join("experiments", a.experiment)
    pred_dir = os.path.join(root, "predictions")
    report_dir = os.path.join(root, "reports")
    log_root = os.path.join(root, "run_evaluation")
    os.makedirs(report_dir, exist_ok=True)

    if not a.merge_only:
        patcher = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patch_swebench_windows.py")
        subprocess.run([PYTHON, patcher], check=True)

    preds = sorted(glob.glob(os.path.join(pred_dir, "*.jsonl")))
    if a.only:
        preds = [p for p in preds if a.only in os.path.basename(p)]
    if not preds:
        raise SystemExit("no predictions in {}".format(pred_dir))

    rows = []
    for pred in preds:
        model_name = os.path.basename(pred)[: -len(".jsonl")]
        run_id = "swe-" + a.experiment + "-" + model_name
        out = report_path(report_dir, model_name, run_id)
        if a.force:
            for stale in (out, os.path.join(log_root, run_id)):
                if os.path.isdir(stale):
                    shutil.rmtree(stale)
                elif os.path.exists(stale):
                    os.remove(stale)
        if a.merge_only or os.path.exists(out):
            if os.path.exists(out):
                with open(out, encoding="utf-8") as f:
                    rows.append(summarize(json.load(f), model_name))
                print("[skip] {} already has a report".format(model_name))
            continue
        report = run_one(pred, report_dir, run_id, a.max_workers, a.timeout, log_root)
        if report is not None:
            rows.append(summarize(report, model_name))

    if not rows:
        raise SystemExit("no reports to summarise")

    out_csv = os.path.join(root, "swebench_results.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n{:<20} {:>9} {:>11} {:>6} {:>12}".format("cell", "resolved", "unresolved", "error", "empty_patch"))
    for r in rows:
        print(
            "{:<20} {:>9} {:>11} {:>6} {:>12}".format(
                r["model"], r["resolved"], r["unresolved"], r["error"], r["empty_patch"]
            )
        )
    print("\nwrote", out_csv)


if __name__ == "__main__":
    main()
