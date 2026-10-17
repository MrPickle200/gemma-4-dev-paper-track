"""Metric #4 (IDEA.md §4): error and loop rate vs context depth.

The claim under test is that E2B/E4B degrade as their context fills up - that the PLE
architecture has a practical depth limit. The traces already hold both halves of that
measurement per turn: `usage.prompt_tokens` is how much context the model faced, and each
`tool` record says whether the call was malformed, failed, or repeated the previous call
byte-for-byte. So this script bins tool calls by the depth of the turn that issued them and
reports the two failure modes that metric #4 names:

  * fail rate  = (malformed arguments + tool call rejected) / calls   <- Module 1's target
  * loop rate  = identical repeat of the immediately preceding call    <- "lặp vô hạn"

Both are reported per variant, because the modules exist precisely to move them, and the
counts are printed with the rates: with 50 tasks and 15 turns, a bucket with a handful of
calls cannot support a claim, and the table has to say so rather than hide it.

Only scoreable runs count (same rule as summarize_batch: one trace per cell, newest wins,
infrastructure failures excluded), so a run a broken backend killed cannot bias the curve.

Usage:
  .\\.venv\\Scripts\\python.exe scripts\\plot_metric4.py --model gemma-4-E2B
  .\\.venv\\Scripts\\python.exe scripts\\plot_metric4.py --csv logs\\metric4_e2b.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.config import is_infra_failure, turn_error   # noqa: E402

# The edges IDEA.md §4 asks for. The last bucket is open-ended on purpose: a turn beyond the
# designed window should be visible as its own point, not folded into 12-16K.
BUCKETS = ((0, 2048, "<=2K"), (2048, 4096, "2-4K"), (4096, 8192, "4-8K"),
           (8192, 12288, "8-12K"), (12288, 16384, "12-16K"), (16384, 1 << 30, ">16K"))
LABELS = [label for _, _, label in BUCKETS]


def bucket_of(tokens: int) -> str:
    for lo, hi, label in BUCKETS:
        if lo <= tokens < hi:
            return label
    return LABELS[-1]


def load(path: str) -> tuple[dict, dict, list[dict]]:
    recs = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    start = next((r for r in recs if r["type"] == "run_start"), {})
    end = next((r for r in recs if r["type"] == "run_end"), {})
    return start, end, recs


def calls_with_depth(recs: list[dict]):
    """(prompt depth, tool record) for every tool call, in order.

    A tool call belongs to the depth of the turn that produced it: the model decided what to
    call while looking at that many tokens. A turn that errored carries no usage and no tool
    calls, so it cannot shift the depth of a later one.
    """
    depth = 0
    for r in recs:
        if r.get("type") == "turn":
            usage = r.get("usage") or {}
            if usage.get("prompt_tokens") is not None:
                depth = usage["prompt_tokens"]
        elif r.get("type") == "tool":
            yield depth, r


def failure_kind(status: str | None, error: str | None) -> str | None:
    """Name *why* a tool call failed, or None if it did not.

    One blob would hide the story. `malformed_json` (the server could not parse the arguments)
    and `schema` (arguments that are valid JSON but name parameters the tool does not have) are
    the same population Module 1 addresses: the model could not express the call. `search_miss`
    is a localisation failure, `syntax_gate` is M1's gate doing its job, and `path`/`symbol` are
    the model misreading the repo layout. Metric #4 is about the first two and the loop rate.
    """
    if status in (None, "ok"):
        return None
    if status == "malformed_arguments":
        return "malformed_json"
    e = str(error or "")
    if e.startswith("bad arguments for") or e.startswith("unknown tool"):
        return "schema"
    if "search_block not found" in e:
        return "search_miss"
    if "syntax gate" in e:
        return "syntax_gate"
    if "not a file:" in e or "not a directory:" in e:
        return "path"
    if "no symbol named" in e:
        return "symbol"
    return "other"


KINDS = ("malformed_json", "schema", "search_miss", "syntax_gate", "path", "symbol", "other")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-4-E2B")
    ap.add_argument("--variants", nargs="+", default=["B0", "M1", "M2", "M3"])
    ap.add_argument("--thinking", default="on")
    ap.add_argument("--max-turns", type=int, default=15)
    ap.add_argument("--glob", default="logs/runs/*.jsonl")
    ap.add_argument("--csv", help="write the per-bucket numbers to this CSV")
    ap.add_argument("--out", help="write the figure here (default figures/metric4_<model>.png)")
    ap.add_argument("--no-plot", action="store_true", help="numbers only, no figure (no matplotlib)")
    args = ap.parse_args()

    # ---- one trace per cell, newest wins, scoreable only (same rule as summarize_batch) ----
    cells: dict[tuple, tuple] = {}
    for path in sorted(glob.glob(args.glob)):
        start, end, recs = load(path)
        if not end:
            continue
        cfg = start.get("config", {})
        if (cfg.get("model") != args.model or cfg.get("thinking") != args.thinking
                or cfg.get("max_turns") != args.max_turns
                or cfg.get("variant") not in args.variants):
            continue
        if is_infra_failure(end.get("stop_reason"), turn_error(recs), cfg.get("ctx_per_run")):
            continue
        key = (cfg.get("instance_id"), cfg.get("variant"))
        if key in cells and cells[key][1].get("ts", 0) >= end.get("ts", 0):
            continue
        cells[key] = (recs, end)

    if not cells:
        print("no scoreable traces matched; check --model / --thinking / --max-turns")
        return 1

    # ---- bin every tool call of every selected run by the depth of its turn ----
    def blank() -> dict[str, int]:
        return {"calls": 0, "repeats": 0, **{k: 0 for k in KINDS}}

    stats: dict[tuple[str, str], dict[str, int]] = defaultdict(blank)
    turns_at_depth: dict[str, int] = defaultdict(int)
    for (_, variant), (recs, _) in cells.items():
        for depth, r in calls_with_depth(recs):
            s = stats[(variant, bucket_of(depth))]
            s["calls"] += 1
            kind = failure_kind(r.get("status"), r.get("error"))
            if kind:
                s[kind] += 1
            if r.get("repeat_of_turn") is not None:
                s["repeats"] += 1
        for r in recs:                       # how many turns were taken at each depth
            usage = r.get("usage") or {}
            if r.get("type") == "turn" and usage.get("prompt_tokens") is not None:
                turns_at_depth[bucket_of(usage["prompt_tokens"])] += 1

    rows = []
    for variant in args.variants:
        for label in LABELS:
            s = stats.get((variant, label))
            if not s or not s["calls"]:
                continue
            failed = sum(s[k] for k in KINDS)
            rows.append({
                "model": args.model, "variant": variant, "depth": label,
                "calls": s["calls"], "repeats": s["repeats"],
                "fail_rate": round(failed / s["calls"], 3),
                "loop_rate": round(s["repeats"] / s["calls"], 3),
                **{k: s[k] for k in KINDS},
            })

    if not rows:
        print("no tool calls found in the selected traces")
        return 1

    widths = {"model": 12, "variant": 4, "depth": 6, "calls": 5, "repeats": 5,
              "fail_rate": 6, "loop_rate": 6,
              "malformed_json": 5, "schema": 5, "search_miss": 5, "syntax_gate": 5,
              "path": 4, "symbol": 5, "other": 5}
    title = {"model": "model", "variant": "var", "depth": "depth", "calls": "calls",
             "repeats": "loop", "fail_rate": "fail%", "loop_rate": "loop%",
             "malformed_json": "mj", "schema": "sch", "search_miss": "miss",
             "syntax_gate": "gate", "path": "path", "symbol": "sym", "other": "oth"}
    print(" ".join(f"{title[k]:<{widths[k]}}" for k in widths))
    for r in rows:
        print(" ".join(f"{str(r[k]):<{widths[k]}}" for k in widths))

    print("\ndepth = the prompt tokens of the turn that issued the call (what the model faced).")
    print(f"fail% = any non-ok call; its reasons: " + ", ".join(
        f"{k} ({title[k]})" for k in KINDS))
    print("loop = byte-identical repeat of the immediately preceding call.")
    bad = [r for r in rows if r["calls"] < 20]
    if bad:
        print(f"note: {len(bad)} bucket(s) hold <20 calls - report them as sparse, not as a trend:")
        print("      " + ", ".join(f"{r['variant']} {r['depth']} (n={r['calls']})" for r in bad))
    grand = sum(r["calls"] for r in rows)
    print(f"total: {grand} tool calls over {len(cells)} run(s), "
          f"{sum(r[k] for r in rows for k in KINDS)} failed "
          f"({sum(r['malformed_json'] + r['schema'] for r in rows)} of them argument-side failures)")

    # The per-variant line is metric #4's headline for the ablation table; summing the CSV by hand
    # is how a table stops matching its own figure.
    print("\nper variant (all depths):")
    print(f"  {'var':<4}{'calls':>6}{'fail%':>7}{'loop%':>7}{'mj':>4}{'sch':>5}"
          f"{'miss':>6}{'gate':>6}{'path':>6}{'sym':>5}{'oth':>5}")
    for variant in args.variants:
        vrows = [r for r in rows if r["variant"] == variant]
        if not vrows:
            continue
        n = sum(r["calls"] for r in vrows)
        tot = {k: sum(r[k] for r in vrows) for k in KINDS}
        loops = sum(r["repeats"] for r in vrows)
        failed = sum(tot.values())
        print(f"  {variant:<4}{n:>6}{failed / n:>7.3f}{loops / n:>7.3f}"
              + "".join(f"{tot[k]:>4d}" if k == "malformed_json" else f"{tot[k]:>5d}"
                        if k in ("schema", "symbol", "other") else f"{tot[k]:>6d}"
                        for k in KINDS))

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.csv).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")

    if not args.no_plot:
        out = Path(args.out or f"figures/metric4_{args.model}.png")
        plot(rows, turns_at_depth, args.model, out)
        print(f"wrote {out}")
    return 0


def plot(rows: list[dict], turns_at_depth: dict[str, int], model: str, out: Path) -> None:
    """Two panels: failure rate and loop rate, one line per variant, x = depth bucket."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [lb for lb in LABELS if any(r["depth"] == lb for r in rows)]
    x = list(range(len(labels)))
    variants = sorted({r["variant"] for r in rows})
    colors = {"B0": "#1f77b4", "M1": "#d62728", "M2": "#2ca02c", "M3": "#9467bd"}
    calls_in = {lb: sum(r["calls"] for r in rows if r["depth"] == lb) for lb in labels}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
    for ax, metric, name in ((axes[0], "fail_rate", "any failed tool call"),
                             (axes[1], "loop_rate", "identical repeat of the previous call")):
        for variant in variants:
            by_depth = {r["depth"]: r for r in rows if r["variant"] == variant}
            ys = [by_depth[lb][metric] if lb in by_depth else None for lb in labels]
            ax.plot(x, ys, marker="o", label=variant, color=colors.get(variant),
                    linewidth=1.8, markersize=6)
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.25, linestyle=":")
        ax.set_ylim(bottom=0, top=1.05)
        ax.set_xticks(x)
        # The call count belongs on the axis, not as a label per point: the points cluster at low
        # rates (where the labels overlap) and a 1-call bucket must not look like a trend.
        ax.set_xticklabels([f"{lb}\n{turns_at_depth[lb]} turns\n{calls_in[lb]} calls" for lb in labels],
                           fontsize=8)
    axes[0].set_ylabel("rate per tool call")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"{model}: tool-call error and loop rate vs context depth (thinking on)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    fig.savefig(out.with_suffix(".pdf"))


if __name__ == "__main__":
    sys.exit(main())
