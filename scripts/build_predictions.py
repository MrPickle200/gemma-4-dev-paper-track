"""Build one SWE-bench predictions file per (model, variant) cell.

SWE-bench matches on instance_id, so the four variants of one task cannot share a
file: each must be its own model_name_or_path. Traces are named
    <instance_id>__<model>__<variant>__<timestamp>.jsonl
and the patch for a trace is the same name with .patch.  If a cell was re-run we
keep the newest trace, and if no patch was written (empty diff) the prediction is
an empty model_patch, which SWE-bench scores as unresolved.

    python scripts/build_predictions.py --experiment 20261023-e2b-grid
"""

import argparse
import glob
import json
import os


def parse_name(path):
    base = os.path.basename(path)
    stem = base.rsplit(".", 1)[0]
    parts = stem.split("__")
    if len(parts) < 5:
        return None
    instance_id = "__".join(parts[:2])
    model = parts[2]
    variant = parts[3]
    stamp = "__".join(parts[4:])
    return instance_id, model, variant, stamp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()

    root = os.path.join("experiments", a.experiment)
    out_dir = a.out_dir or os.path.join(root, "predictions")
    os.makedirs(out_dir, exist_ok=True)

    newest = {}
    for trace in glob.glob(os.path.join(root, "runs", "*.jsonl")):
        parsed = parse_name(trace)
        if parsed is None:
            continue
        instance_id, model, variant, stamp = parsed
        key = (model, variant, instance_id)
        if key not in newest or stamp > newest[key][0]:
            newest[key] = (stamp, trace)

    cells = {}
    for (model, variant, instance_id), (_, trace) in newest.items():
        patch_name = os.path.basename(trace)[: -len(".jsonl")] + ".patch"
        patch_path = os.path.join(root, "patches", patch_name)
        patch = ""
        if os.path.exists(patch_path):
            with open(patch_path, encoding="utf-8", errors="replace") as f:
                patch = f.read()
        cells.setdefault((model, variant), []).append(
            (instance_id, "{}_{}".format(model, variant), patch)
        )

    for (model, variant), rows in sorted(cells.items()):
        rows.sort(key=lambda r: r[0])
        out = os.path.join(out_dir, "{}_{}.jsonl".format(model, variant))
        with open(out, "w", encoding="utf-8") as f:
            for instance_id, name, patch in rows:
                f.write(
                    json.dumps(
                        {
                            "instance_id": instance_id,
                            "model_name_or_path": name,
                            "model_patch": patch,
                        }
                    )
                    + "\n"
                )
        nonempty = sum(1 for r in rows if r[2].strip())
        print(
            "{}: {}/{} unique instances, {} with a non-empty patch".format(
                os.path.basename(out), len({r[0] for r in rows}), len(rows), nonempty
            )
        )


if __name__ == "__main__":
    main()
