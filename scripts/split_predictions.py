"""Split per-cell SWE-bench predictions into per-machine files.

SWE-bench keys predictions by `instance_id` (run_evaluation.py collapses the list
into `{instance_id: pred}`), and the four variants of a task share the same
instance_id.  So the unit of work is one (model, variant) predictions file; it
must be evaluated in its own `run_evaluation` call with its own --run_id.  This
script slices each such file into N round-robin parts (round-robin keeps every
part mixed across repos instead of grouping them by instance_id order).

    python scripts/split_predictions.py \
        --pred-dir logs/experiments/20261023-e2b-grid/predictions \
        --chunks 2 --out-dir D:\\swe\\e2b_parts
"""

import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--chunks", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    if a.chunks < 1:
        raise SystemExit("--chunks must be >= 1")

    os.makedirs(a.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.pred_dir, "*.jsonl")))
    if not files:
        raise SystemExit("no *.jsonl in {}".format(a.pred_dir))

    for path in files:
        stem = os.path.basename(path)[: -len(".jsonl")]
        rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        buckets = [[] for _ in range(a.chunks)]
        for i, row in enumerate(rows):
            buckets[i % a.chunks].append(row)
        for i, bucket in enumerate(buckets):
            out = os.path.join(a.out_dir, "{}_part{}.jsonl".format(stem, i))
            with open(out, "w", encoding="utf-8") as f:
                for row in bucket:
                    f.write(json.dumps(row) + "\n")
            print(
                "{}: {} cell(s), {} with a patch".format(
                    os.path.basename(out),
                    len(bucket),
                    sum(1 for r in bucket if r.get("model_patch", "").strip()),
                )
            )


if __name__ == "__main__":
    main()
