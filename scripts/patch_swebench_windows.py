"""Patch swebench's Windows newline/encoding bugs in the local venv.

swebench 5.0.2 writes `eval.sh` with `Path.write_text(...)` and `test_output.txt`
with `open(..., "w")`.  On Windows text mode turns every `\\n` into `\\r\\n`, so the
script that runs *inside the Linux container* becomes

    cd $'/testbed\\r'   ->  "No such file or directory"
    conda activate $'testbed\\r'  ->  conda never activates
    bin/test $'...test_lambdify.py\\r'  ->  no tests collected

which silently marks whole cells unresolved/ambiguous.  And the pytest output is
written with the cp1252 locale, which raises UnicodeEncodeError on box-drawing
characters and aborts the instance.

This applies three idempotent edits to the installed harness:

    patch.diff      -> LF only
    eval.sh         -> LF only
    test_output.txt -> UTF-8, LF only

Run it once per machine before the first evaluation:

    python scripts/patch_swebench_windows.py [--check]
"""

import argparse
import os
import sys

EDITS = [
    (
        'patch_file.write_text(pred["model_patch"] or "")',
        'patch_file.write_text(pred["model_patch"] or "", newline="\\n")',
    ),
    (
        "eval_file.write_text(_inject_asset_restore(test_spec.eval_script, restore_cmds))",
        'eval_file.write_text(\n'
        "            _inject_asset_restore(test_spec.eval_script, restore_cmds),\n"
        '            newline="\\n",\n'
        "        )",
    ),
    (
        'with open(test_output_path, "w") as f:',
        'with open(test_output_path, "w", encoding="utf-8", newline="\\n") as f:',
    ),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only report, do not modify")
    a = ap.parse_args()

    import swebench

    path = os.path.join(os.path.dirname(swebench.__file__), "harness", "run_evaluation.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()

    pending = []
    for old, new in EDITS:
        if new in src:
            continue
        if old not in src:
            raise SystemExit("cannot find in {}: {}".format(path, old))
        pending.append((old, new))

    if not pending:
        print("already patched:", path)
        return
    if a.check:
        print("NOT patched ({} edit(s) needed): {}".format(len(pending), path))
        sys.exit(1)

    for old, new in pending:
        src = src.replace(old, new, 1)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(src)
    print("patched {} ({} edit(s))".format(path, len(pending)))


if __name__ == "__main__":
    main()
