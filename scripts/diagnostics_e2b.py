"""Generate E2B_DIAGNOSTICS_REPORT.md from the run traces (DIAGNOSTICS_SPEC.md).

The report is a *derived artefact*: every number comes from logs/runs/*.jsonl, using the same
selection rule as the ablation table (one trace per cell, newest wins, infrastructure failures
excluded). That rule matters more here than anywhere else: the spec asks for sample sizes per
context bin, because a 78% failure rate computed over 18 calls is not the same claim as one
computed over 200.

Taxonomy note (the harness has no shell): B0's "bash" tools are search_text/read_file/list_dir,
so BASH_EXECUTION_ERROR means one of those returned ok=False (invalid regex, path not found) -
the analogue of a non-zero exit code. apply_edit failures are split by cause, not by tool, so a
search_block mismatch and a path that does not exist land in different buckets.

Usage:
  .\\.venv\\Scripts\\python.exe scripts\\diagnostics_e2b.py
  .\\.venv\\Scripts\\python.exe scripts\\diagnostics_e2b.py --out E2B_DIAGNOSTICS_REPORT.md
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from harness import config                        # noqa: E402
from harness.config import is_infra_failure, turn_error      # noqa: E402
from plot_metric4 import LABELS, bucket_of, calls_with_depth, load  # noqa: E402

VARIANTS = ("B0", "M1", "M2", "M3")
NAV_TOOLS = {"get_repo_map", "read_symbol", "find_references"}
BASH_TOOLS = {"search_text", "read_file", "list_dir"}
CATS = ("JSON_SCHEMA_ERROR", "SYMBOL_NOT_FOUND", "SEARCH_BLOCK_MISMATCH",
        "AST_SYNTAX_REJECTED", "BASH_EXECUTION_ERROR", "OTHER_ERROR")


def sym_parts(symbol: str | None) -> set[str]:
    """Identifier components of a symbol name, whatever shape the model gave it.

    `lock_targets` takes `sympy/core/mod.py::Mod.doit`, `sympy/core/mod.doit` or `Add.flatten`,
    while `apply_edit`'s `target_symbol` is usually `doit` or `Class.method`. Normalising to a
    component set is what makes those comparable: a bare `doit` overlaps the locked `Mod.doit`.
    """
    s = str(symbol or "").strip()
    if not s or s.endswith(".py"):
        return set()
    return {p for p in re.split(r"[./:\\]+", s) if p}


def sym_match(target: str | None, locked: str | None) -> bool:
    """True when an edit's target names the locked symbol or a member of it.

    `Commutator._latex` inside a lock on `Commutator` is not a false localisation; `doit` inside a
    lock on `Mod.doit` is not one either.
    """
    tp, lp = sym_parts(target), sym_parts(locked)
    return bool(tp & lp)


def taxonomy(name: str, status: str | None, error: str | None) -> str | None:
    """The spec's six-way cause for a failed call (None when the call succeeded)."""
    if status in (None, "ok"):
        return None
    e = str(error or "")
    if (status == "malformed_arguments" or e.startswith("bad arguments for")
            or e.startswith("unknown tool")):
        # includes 'Unterminated string': arguments cut off at max_tokens while thinking ran long
        return "JSON_SCHEMA_ERROR"
    if "syntax gate" in e:
        return "AST_SYNTAX_REJECTED"
    if "search_block not found" in e:
        return "SEARCH_BLOCK_MISMATCH"
    if name in NAV_TOOLS and ("no symbol named" in e or "not found in" in e
                              or "is ambiguous" in e or "not a file" in e
                              or "not a directory" in e):
        return "SYMBOL_NOT_FOUND"
    if name in BASH_TOOLS:
        return "BASH_EXECUTION_ERROR"
    return "OTHER_ERROR"


def select(model: str, thinking: str, max_turns: int,
           pattern: str = "logs/runs/*.jsonl") -> tuple[dict, Counter]:
    """Newest scoreable trace per (instance, variant), plus counts of what was dropped."""
    cells: dict[tuple, tuple] = {}
    dropped: Counter = Counter()
    for path in sorted(glob.glob(str(ROOT / pattern))):
        start, end, recs = load(path)
        cfg = start.get("config", {})
        if (cfg.get("model") != model or cfg.get("thinking") != thinking
                or cfg.get("max_turns") != max_turns or cfg.get("variant") not in VARIANTS):
            continue
        dropped["matched"] += 1
        if not end:
            dropped["no_run_end"] += 1
            continue
        if is_infra_failure(end.get("stop_reason"), turn_error(recs), cfg.get("ctx_per_run")):
            err = turn_error(recs) or ""
            if "Context size has been exceeded" in err:
                why = "server-side context 500 (shared KV pool)"
            elif "524" in err:
                why = "Cloudflare 524 (tunnel time-to-first-byte)"
            elif "502" in err:
                why = "http 502 (tunnel)"
            elif "truncated stream" in err:
                why = "truncated stream"
            else:
                why = "other"
            dropped["infra"] += 1
            dropped["infra: " + why] += 1
            continue
        key = (cfg.get("instance_id"), cfg.get("variant"))
        if key in cells and cells[key][1].get("ts", 0) >= end.get("ts", 0):
            dropped["superseded"] += 1
            continue
        if key in cells:
            dropped["superseded"] += 1
        cells[key] = (recs, end, path)
    return cells, dropped


def pct(n: int, d: int) -> str:
    return f"{n} ({n / d:.1%})" if d else "0 (-)"


def md_table(head: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(head) + " |",
           "| " + " | ".join(":---" if i == 0 else ":---:" for i in range(len(head))) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def line_of(path: str, ts: float) -> int:
    needle = json.dumps(ts)
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if needle in line:
            return i
    return 0


def raw(path: str, rec: dict, cap: int = 1500) -> str:
    """One trace record verbatim (long strings capped, with the line number to read the rest)."""
    txt = json.dumps(rec, ensure_ascii=False)
    ln = line_of(path, rec.get("ts", 0))
    if len(txt) > cap:
        txt = txt[:cap] + f" ... [{len(txt) - cap} chars truncated - full record at line {ln}]"
    return f"```json\n{txt}\n```"


def mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else 0.0


def one_line(text, cap: int = 140) -> str:
    """Keep commentary on one Markdown line, showing newlines as escapes rather than breaking it."""
    s = str(text or "").replace("\r", "\\r").replace("\n", "\\n")
    return s[:cap] + (" ..." if len(s) > cap else "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-4-E2B")
    ap.add_argument("--thinking", default="on")
    ap.add_argument("--max-turns", type=int, default=15)
    ap.add_argument("--out", default=None,
                    help="report path (default: E2B_DIAGNOSTICS_REPORT.md, or DIAGNOSTICS.md "
                         "inside logs/experiments/<TAG>/ when --experiment is used)")
    ap.add_argument("--glob", default="logs/runs/*.jsonl",
                    help="trace glob, relative to the repo root")
    ap.add_argument("--experiment", default=None,
                    help="read logs/experiments/<TAG>/runs/*.jsonl and write the report there")
    args = ap.parse_args()
    if args.experiment:
        args.glob = str(config.EXPERIMENTS / args.experiment / "runs" / "*.jsonl")
        if args.out is None:
            args.out = str(config.EXPERIMENTS / args.experiment / "DIAGNOSTICS.md")
    elif args.out is None:
        args.out = "E2B_DIAGNOSTICS_REPORT.md"

    cells, dropped = select(args.model, args.thinking, args.max_turns, args.glob)
    if not cells:
        print("no scoreable traces matched")
        return 1
    variants = tuple(v for v in VARIANTS if any(var == v for _, var in cells))

    # ---------------------------------------------------------------- one pass over every cell
    turns: Counter = Counter()          # (variant, bin) -> turns taken at that depth
    calls: Counter = Counter()          # (variant, bin) -> tool calls
    fails: Counter = Counter()          # (variant, bin) -> non-ok calls
    reps: Counter = Counter()           # (variant, bin) -> identical repeats
    cat_bin: dict[tuple[str, str], Counter] = defaultdict(Counter)   # taxonomy per bin
    cat_tot: dict[str, Counter] = defaultdict(Counter)               # taxonomy per config
    other_ex: dict[str, Counter] = defaultdict(Counter)              # OTHER_ERROR examples
    no_usage: Counter = Counter()                                    # turns with no usage
    per_run: dict[tuple[str, str], dict] = {}

    for (inst, variant), (recs, end, path) in cells.items():
        depth_of_turn: dict[int, int] = {}
        for r in recs:
            if r.get("type") != "turn":
                continue
            pt = (r.get("usage") or {}).get("prompt_tokens")
            if pt is None:
                no_usage[variant] += 1
            else:
                depth_of_turn[r["turn"]] = pt
                turns[(variant, bucket_of(pt))] += 1
        for depth, r in calls_with_depth(recs):
            b = bucket_of(depth)
            calls[(variant, b)] += 1
            kind = taxonomy(r.get("name", ""), r.get("status"), r.get("error"))
            if kind:
                fails[(variant, b)] += 1
                cat_bin[(variant, b)][kind] += 1
                cat_tot[variant][kind] += 1
                if kind == "OTHER_ERROR":
                    short = str(r.get("error") or "")[:70]
                    other_ex[variant][f"{r.get('name')}: {short}"] += 1
            if r.get("repeat_of_turn") is not None:
                reps[(variant, b)] += 1
        per_run[(inst, variant)] = {"recs": recs, "end": end, "path": path,
                                    "depth_of_turn": depth_of_turn}

    total_calls = sum(calls.values())
    total_fails = sum(fails.values())

    # ---------------------------------------------------------------------------- section 1
    s1 = []
    for label in LABELS:
        for v in variants:
            s1.append([f"`{label}`", f"`{v}`", turns[(v, label)], calls[(v, label)],
                       pct(fails[(v, label)], calls[(v, label)]),
                       pct(reps[(v, label)], calls[(v, label)])])

    # ---------------------------------------------------------------------------- section 2
    s2 = {}
    for v in variants:
        total = sum(cat_tot[v].values())
        s2[v] = [[c, cat_tot[v][c], f"{cat_tot[v][c] / total:.1%}" if total else "-"]
                 for c in CATS]
        s2[v].append(["**total**", f"**{total}**", "**100.0%**" if total else "-"])

    m2_rows = []

    def m2_row(name: str, labels: list[str]) -> list[str]:
        n = sum(calls[("M2", lb)] for lb in labels)
        row = [name, n]
        for c in CATS:
            k = sum(cat_bin[("M2", lb)][c] for lb in labels)
            row.append(f"{k} ({k / n:.0%})" if n else "0 (-)")
        f = sum(fails[("M2", lb)] for lb in labels)
        row.append(pct(f, n) if n else "0 (-)")
        return row

    m2_rows = [m2_row(f"`{lb}`", [lb]) for lb in ("8-12K", "12-16K")]
    m2_rows.append(m2_row("`8-16K` (combined)", ["8-12K", "12-16K"]))
    m2_rows.append(m2_row("`>16K`", [">16K"]))

    # ---------------------------------------------------------------------------- section 3
    m3 = [(inst, d) for (inst, v), d in per_run.items() if v == "M3"]
    compacted: list[tuple[str, dict, dict]] = []
    trig_turns: list[int] = []
    pre_tokens: list[int] = []
    post_tokens: list[int] = []
    chars_before: list[int] = []
    chars_after: list[int] = []
    re_nav = edit_outside = sym_outside = gave_up = grew = 0
    for inst, d in m3:
        comp = next((r for r in d["recs"] if r.get("type") == "compaction"), None)
        if not comp:
            continue
        compacted.append((inst, d, comp))
        trig_turns.append(comp["turn"])
        pre = d["depth_of_turn"].get(comp["turn"])
        post = d["depth_of_turn"].get(comp["turn"] + 1)
        if pre is not None and post is not None:
            pre_tokens.append(pre)
            post_tokens.append(post)
        chars_before.append(comp["context_chars_before"])
        chars_after.append(comp["context_chars_after"])
        after = [r for r in d["recs"] if r.get("type") == "tool" and r.get("turn", 0) > comp["turn"]]
        if any(r.get("name") in NAV_TOOLS for r in after):
            re_nav += 1
        locked_files = {str(f) for f in comp["files"]}
        locked_syms = [str(s) for s in comp["symbols"]]
        edits = [r for r in after if r.get("name") == "apply_edit"]
        if any((r.get("args") or {}).get("file") not in locked_files for r in edits):
            edit_outside += 1
        targets = [(r.get("args") or {}).get("target_symbol") for r in edits]
        if any(t and not any(sym_match(t, l) for l in locked_syms) for t in targets):
            sym_outside += 1
        real_edit = any((r.get("args") or {}).get("search_block")
                        != (r.get("args") or {}).get("replace_block") for r in edits)
        if d["end"].get("stop_reason") != "finish" or not real_edit:
            gave_up += 1
        if pre is not None and post is not None and post > pre:
            grew += 1

    reduction = [1 - p / q for p, q in zip(post_tokens, pre_tokens) if q]
    n_m3 = len(m3)
    n_comp = len(compacted)
    trig = (f"{n_comp} / {n_m3} ({n_comp / n_m3:.1%})" if n_m3
            else "n/a - this run set holds no M3 cells")

    def patch_ok(d: dict) -> bool:
        art = next((r for r in d["recs"] if r.get("type") == "artifacts"), None)
        return bool(art) and not art.get("patch_is_empty")

    comp_patch = sum(1 for _, d, _ in compacted if patch_ok(d))
    rest = [(i, d) for i, d in m3 if not any(i == j for j, _, _ in compacted)]
    rest_patch = sum(1 for _, d in rest if patch_ok(d))

    comp_rows = [
        ["Compaction triggered", trig],
        ["Turn at trigger (mean / median / min-max)",
         f"{mean(trig_turns):.1f} / {statistics.median(trig_turns):.0f} / "
         f"{min(trig_turns)}-{max(trig_turns)}" if trig_turns else "n/a"],
        ["Prompt tokens before compaction (mean, n)",
         f"{mean(pre_tokens):,.0f} (n={len(pre_tokens)})" if pre_tokens else "n/a"],
        ["Prompt tokens after compaction (mean, n)",
         f"{mean(post_tokens):,.0f} (n={len(post_tokens)})" if post_tokens else "n/a"],
        ["Token reduction per task (mean / min / max)",
         f"{mean(reduction):.1%} / {min(reduction):.1%} / {max(reduction):.1%}"
         if reduction else "n/a"],
        ["Context chars before -> after (mean, from the compaction record)",
         f"{mean(chars_before):,.0f} -> {mean(chars_after):,.0f}" if chars_before else "n/a"],
        ["Tasks where the post-lock prompt was *larger* than the pre-lock one",
         f"{grew} / {len(pre_tokens)}"],
        ["Phase-1 nav tool called again after compaction", f"{re_nav} / {n_comp}"],
        ["Edited a file outside the locked set", f"{edit_outside} / {n_comp}"],
        ["Edited a symbol other than the locked one (name-component overlap)",
         f"{sym_outside} / {n_comp}"],
        ["No real edit, or did not finish (gave up)", f"{gave_up} / {n_comp}"],
        ["Non-empty patch: compacted vs never-compacted M3 runs",
         f"{comp_patch}/{n_comp} vs {rest_patch}/{len(rest)}"],
    ]

    # ---------------------------------------------------------------------------- section 4
    def runs_of(v: str) -> list[dict]:
        return [d for (_, var), d in per_run.items() if var == v]

    def patches(v: str) -> str:
        runs = runs_of(v)
        if not runs:
            return "n/a"
        ok = sum(1 for d in runs if patch_ok(d))
        return f"{ok}/{len(runs)} ({ok / len(runs):.0%})"

    def stops(v: str) -> str:
        runs = runs_of(v)
        if not runs:
            return "n/a"
        mt = sum(1 for d in runs if d["end"].get("stop_reason") == "max_turns")
        cx = sum(1 for d in runs if d["end"].get("stop_reason") == "context_exceeded")
        hit = mt + cx
        return f"{hit}/{len(runs)} ({hit / len(runs):.0%}) - max_turns {mt}, ctx {cx}"

    def turns_col(v: str) -> str:
        t = [d["end"].get("turns_used", 0) for d in runs_of(v)]
        if not t:
            return "n/a"
        return f"{mean(t):.1f} / {statistics.median(t):.0f}"

    def peak_col(v: str) -> str:
        return f"{mean([d['end'].get('peak_prompt_tokens', 0) for d in runs_of(v)]):,.0f}"

    def gen_col(v: str) -> str:
        return f"{mean([d['end'].get('total_completion_tokens', 0) for d in runs_of(v)]):,.0f}"

    def think_col(v: str) -> str:
        chars = [r["response"]["reasoning_chars"] for d in runs_of(v) for r in d["recs"]
                 if r.get("type") == "turn"
                 and (r.get("response") or {}).get("reasoning_chars") is not None]
        return f"{mean(chars):,.0f} chars (n={len(chars)})" if chars else "n/a"

    s4_rows = [
        ["**Valid Patch Rate (%)** (non-empty `git diff`)"] + [patches(v) for v in variants],
        ["**Resolved Rate (%)**"] + ["not measured"] * len(variants),
        ["**Hit Max Turns / Context Rate (%)**"] + [stops(v) for v in variants],
        ["**Avg / Median Turns per Task**"] + [turns_col(v) for v in variants],
        ["**Avg Peak Context Tokens**"] + [peak_col(v) for v in variants],
        ["**Avg Total Generated Tokens** (completion)"] + [gen_col(v) for v in variants],
        ["**Avg Thinking Length per Turn**"] + [think_col(v) for v in variants],
    ]

    # ---------------------------------------------------------------------------- section 5
    def pick(pred) -> tuple | None:
        for (inst, var), d in per_run.items():
            for depth, r in calls_with_depth(d["recs"]):
                if pred(var, depth, r):
                    return inst, var, d, depth, r
        return None

    def turn_rec(d: dict, turn: int) -> dict | None:
        return next((x for x in d["recs"]
                     if x.get("type") == "turn" and x.get("turn") == turn), None)

    def prior_call(d: dict, r: dict) -> dict | None:
        """The earlier call this one repeated byte-for-byte, when the trace says so."""
        t = r.get("repeat_of_turn")
        if t is None:
            return None
        for x in d["recs"]:
            if (x.get("type") == "tool" and x.get("turn") == t
                    and x.get("name") == r.get("name") and x.get("args") == r.get("args")):
                return x
        return None

    s_a = (pick(lambda v, dep, r: (v == "M2" and dep >= 12288 and r.get("status") != "ok"
                                   and r.get("repeat_of_turn") is not None))
           or pick(lambda v, dep, r: (v == "M2" and dep >= 8192 and r.get("status") != "ok"
                                      and r.get("repeat_of_turn") is not None))
           or pick(lambda v, dep, r: (v == "M2" and dep >= 8192 and r.get("status") != "ok")))
    s_b = (pick(lambda v, dep, r: (v == "M1" and dep < 2048 and r.get("turn", 99) <= 2
                                   and taxonomy(r.get("name", ""), r.get("status"),
                                                r.get("error")) == "JSON_SCHEMA_ERROR")
               and r.get("status") != "ok")
           or pick(lambda v, dep, r: (v == "M1" and dep < 2048 and r.get("turn", 99) <= 2
                                      and r.get("status") != "ok"))
           or pick(lambda v, dep, r: v == "M1" and dep < 4096 and r.get("status") != "ok"))

    m3_sample = None
    for inst, d, comp in compacted:
        after = [r for r in d["recs"]
                 if r.get("type") == "tool" and r.get("turn", 0) > comp["turn"]]
        if any(r.get("name") in NAV_TOOLS for r in after):
            m3_sample = (inst, d, comp)
            break
    if not m3_sample and compacted:
        m3_sample = compacted[0]

    # ---------------------------------------------------------------------------- write it
    L: list[str] = []
    L.append(f"# {args.model} - diagnostics report\n")
    glob_label = args.glob
    try:
        glob_label = Path(args.glob).relative_to(ROOT).as_posix()
    except ValueError:
        pass
    L.append(f"Generated by `scripts/diagnostics_e2b.py` from `{glob_label}`; "
             "re-run the script to regenerate, never edit the numbers by hand.\n")
    L.append(f"- Model `{args.model}`, thinking `{args.thinking}`, `max_turns={args.max_turns}`"
             + (f", experiment `{args.experiment}`." if args.experiment else "."))
    L.append(f"- **{len(cells)} cells** = one newest scoreable trace per task x variant, out of "
             f"{dropped['matched']} matching traces ({dropped['superseded']} superseded, "
             f"{dropped['infra']} excluded as infrastructure failures, "
             f"{dropped['no_run_end']} without `run_end`).")
    infra_why = [(k.split("infra: ", 1)[1], n) for k, n in dropped.items()
                 if k.startswith("infra: ") and n]
    if infra_why:
        L.append("  - Excluded, by cause: " + ", ".join(
            f"`{why}` {n}" for why, n in sorted(infra_why, key=lambda x: -x[1]))
            + ". These are backend artefacts (AGENTS.md: a shared-KV-pool 500 or a tunnel 524 "
              "is not a model result), which is why they are re-run rather than scored.")
    L.append(f"- **{total_calls} tool calls**, {total_fails} failed "
             f"({total_fails / total_calls:.1%}), across {len(cells) // 4} tasks per config.")
    L.append("- Depth of a call = `usage.prompt_tokens` of the turn that issued it. A turn that "
             "errored before reporting usage has no depth and is counted separately.\n")

    L.append("## 1. Tool-call sample size per context bin\n")
    L.append("Per config, so every rate is readable against its own denominator. The `12-16K` and "
             "`>16K` rows are the ones to distrust: only runs that survived long enough reach "
             "them, which is the survivorship bias this table exists to expose.\n")
    L.append(md_table(["Context Bin", "Config", "Total Turns", "Total Tool Calls",
                       "Failed Calls (count & %)", "Identical Repeats (count & %)"], s1))
    if no_usage:
        L.append("\nTurns with no recorded usage (depth unknown, excluded from the bins): "
                 + ", ".join(f"`{v}` {n}" for v, n in sorted(no_usage.items())) + ".")
    L.append("")

    L.append("## 2. Error taxonomy\n")
    L.append("Every failed call, per config. The harness has no shell: B0's bash tools are "
             "`search_text`/`read_file`/`list_dir`, so `BASH_EXECUTION_ERROR` means one of those "
             "returned `ok=False` (invalid regex, path not found) - the analogue of a non-zero "
             "exit code. `apply_edit` failures are split by cause, not by tool. Two consequences "
             "of the real tool sets, which is why the columns do not follow the spec's "
             "parentheticals exactly: B0 also has `apply_edit`, so `SEARCH_BLOCK_MISMATCH` is "
             "populated there, while `SYMBOL_NOT_FOUND` and `AST_SYNTAX_REJECTED` are "
             "structurally impossible in B0 (no nav tools, no gate).\n")
    for v in variants:
        total = sum(cat_tot[v].values())
        L.append(f"### {v} - {total} failed call(s)\n")
        L.append(md_table(["Cause", "Count", "% of this config's errors"], s2[v]))
        if other_ex[v]:
            top = ", ".join(f"`{e}` x{n}" for e, n in other_ex[v].most_common(3))
            L.append(f"\n`OTHER_ERROR` examples for {v}: {top}.")
        L.append("")

    if any(var == "M2" for _, var in cells):
        L.append("### M2 - which cause dominates the 8-16K and >16K spike\n")
        L.append("Same taxonomy, restricted to M2 and to the bins where its failure rate jumps. "
                 "Read each cell against the call count in the same row.\n")
        L.append(md_table(["Context Bin", "Tool Calls"] + list(CATS) + ["Failed (all causes)"],
                          m2_rows))
        L.append("")

    L.append("## 3. M3 compaction\n")
    L.append("`lock_targets` is the phase transition: the first successful lock replaces the "
             "Phase-1 scratch history with a clean Phase-2 context. Token figures are the real "
             "`prompt_tokens` of the request *before* the lock and of the first request *after* "
             "it, not an estimate from characters.\n")
    L.append(md_table(["Metric", "Value"], comp_rows))
    L.append("\nTwo things to read carefully. First, the harness does **not** withdraw the "
             "Phase-1 tools after the lock (`toolbox.specs()` is computed once per run), so "
             "\"nav tool called again\" counts the model going back to search, not the harness "
             "refusing it - a phase-gating implementation would turn that number into refusals. "
             "Second, a negative reduction means the lock happened so early that the rebuilt "
             "Phase-2 context was *bigger* than the short history it replaced; compaction is not "
             "free, and triggering it on turn 3 is not the same mechanism as triggering it on "
             "turn 13. The compacted-vs-never-compacted patch line is confounded for the same "
             "reason a lock is a model action: the 9 runs that never compacted are the runs that "
             "never produced a valid `lock_targets`, so the line mixes the mechanism with the "
             "model's ability to enter it.")
    L.append("")

    L.append("## 4. Task-level and cost metrics (50 tasks per config)\n")
    L.append(md_table(["Metric"] + list(variants), s4_rows))
    L.append("\n`Avg Thinking Length` is in characters: the trace logs `reasoning_chars`, not "
             "reasoning tokens, and no char-to-token factor is assumed. `Avg Total Generated "
             "Tokens` is `total_completion_tokens`, which includes the thinking because "
             "llama.cpp counts it as output.")
    lens = [d["end"].get("stop_reason") for d in per_run.values()]
    L.append("\nStop reasons over all cells: "
             + ", ".join(f"`{k}` {n}" for k, n in Counter(lens).most_common()) + ".")
    L.append("")

    L.append("## 5. Qualitative failure samples\n")
    L.append("Verbatim trace records, with long strings capped and the source line given so the "
             "rest can be read in the trace itself.\n")

    if s_a:
        inst, v, d, dep, r = s_a
        L.append(f"### 5.1 `{v}` at {dep:,} prompt tokens, `{inst}` - failed call that repeats "
                 f"the previous one\n")
        L.append(f"Source: `{Path(d['path']).name}`, turn {r['turn']}.\n")
        prev = prior_call(d, r)
        if prev is not None:
            L.append(f"**the call it repeats (turn {prev['turn']}):**\n")
            L.append(raw(d["path"], prev))
        tr = turn_rec(d, r["turn"])
        if tr:
            L.append("\n**the turn (request sizes, thinking, tool call):**\n")
            L.append(raw(d["path"], tr))
        L.append("\n**the failed call:**\n")
        L.append(raw(d["path"], r))
        if prev is not None:
            L.append(f"\nThe call at turn {r['turn']} is a byte-identical repeat of turn "
                     f"{prev['turn']} (`repeat_of_turn={r['repeat_of_turn']}`): same "
                     f"`{r.get('name')}` and same arguments. At turn {prev['turn']} it was "
                     f"already `{prev.get('status')}` with "
                     f"`{one_line(prev.get('error'), 110)}`, so the model's context held the "
                     f"refusal and it issued the call again anyway.")
        L.append("")

    if s_b:
        inst, v, d, dep, r = s_b
        L.append(f"### 5.2 `{v}` at {dep:,} prompt tokens, `{inst}` - early failure\n")
        L.append(f"Source: `{Path(d['path']).name}`, turn {r['turn']}.\n")
        tr = turn_rec(d, r["turn"])
        if tr:
            L.append("**the turn (request sizes, thinking, tool call):**\n")
            L.append(raw(d["path"], tr))
        L.append("\n**the failed call:**\n")
        L.append(raw(d["path"], r))
        kind = taxonomy(r.get("name", ""), r.get("status"), r.get("error"))
        keys = list((r.get("args") or {}).keys())
        L.append(f"\nTurn {r['turn']} failed at {dep:,} prompt tokens with `{kind}`: "
                 f"`{one_line(r.get('error'), 130)}`.")
        if kind == "JSON_SCHEMA_ERROR" and len(keys) == 1 and "\n" in keys[0]:
            L.append(f" The single argument key `{one_line(keys[0], 90)}` packs both the pattern "
                     f"and the path into one name, so the tool received no parameter it accepts "
                     f"and nothing reached the file - the M1 request-shape failure described in "
                     f"the caveats, at the smallest context in the grid.")
        L.append("")

    if m3_sample:
        inst, d, comp = m3_sample
        L.append(f"### 5.3 `M3` compaction snapshot, `{inst}`\n")
        if comp.get("compacted_prompt_preview"):
            intro = ("The trace logs the compacted context's *text* under "
                     "`compacted_prompt_preview`, plus its size and message count, so Phase 2 "
                     "can be read exactly as the model received it.")
        else:
            intro = ("The trace logs the compacted context's *size and message count*, not its "
                     "text; `compact_messages` rebuilds it from the system prompt, the task and "
                     "the locked symbol source.")
        L.append(f"Source: `{Path(d['path']).name}`. {intro}\n")
        pre = turn_rec(d, comp["turn"])
        post = turn_rec(d, comp["turn"] + 1)
        if pre:
            L.append("**turn before compaction (the `lock_targets` call):**\n")
            L.append(raw(d["path"], pre))
        L.append("\n**compaction record - exactly what Phase 2 is handed:**\n")
        L.append(raw(d["path"], comp))
        if post:
            L.append("\n**first turn after compaction (Phase 2):**\n")
            L.append(raw(d["path"], post))
        after = [r for r in d["recs"]
                 if r.get("type") == "tool" and r.get("turn", 0) > comp["turn"]]
        empty = next((r.get("patch_is_empty") for r in d["recs"]
                      if r.get("type") == "artifacts"), None)
        delta = comp["context_chars_after"] - comp["context_chars_before"]
        L.append(f"\nPhase-2 tool calls after the lock: "
                 + (", ".join(f"`{r.get('name')}` (turn {r['turn']}, {r.get('status')})"
                              for r in after) or "none")
                 + f". Locked: `{', '.join(comp['files'])}` / `{', '.join(comp['symbols'])}`; "
                 f"ended `{d['end'].get('stop_reason')}`, patch "
                 f"{'empty' if empty else 'non-empty'}. Rebuilding the context moved the message "
                 f"text from {comp['context_chars_before']:,} to "
                 f"{comp['context_chars_after']:,} chars ({delta:+,}).")
        if any(r.get("name") == "lock_targets" for r in after):
            L.append("\nA second `lock_targets` after the lock is acknowledged but does not "
                     "compact again: the transition is one-way, so the model cannot re-lock its "
                     "way into a fresh Phase 1 - and cannot recover from a wrong lock either.")
        L.append("")

    L.append("## Caveats\n")
    L.append("- **Resolved Rate is empty on purpose.** It needs the SWE-bench Docker evaluator "
             "(Bước 5), which has not run; nothing here guesses it. Valid Patch Rate is the "
             "measured upper bound: a non-empty diff is not a correct fix.")
    if cat_tot.get("M1", Counter()).get("JSON_SCHEMA_ERROR", 0):
        L.append("- **M1's numbers are partly a request-shape artefact.** Under M1's structured edit "
                 "schema + `_EDIT_STRUCTURED` prompt, E2B writes `search_text` arguments as "
                 "`pattern=\"x\"\\npath` with Gemma 4 template tokens inside them (9 of 50 cells, 83 "
                 "calls; no B0/M2/M3 cell does this at temperature 0). Those are scored as "
                 "`JSON_SCHEMA_ERROR`, which is why M1 leads that column - they are evidence against "
                 "M1 as specified, not about the syntax gate (`AST_SYNTAX_REJECTED` measures that).")
    else:
        L.append("- **The M1 request-shape defect did not occur in this run set.** No call was "
                 "scored `JSON_SCHEMA_ERROR`, i.e. no argument key glued two parameters together. "
                 "The schema now carries `additionalProperties: false` and the harness repairs a "
                 "glued key with a regex fallback (TASK.md Task 2), so this is the expected result "
                 "of the fix - and a regression here means one of the two stopped working.")
    L.append("- **`>16K` holds very few calls.** Anything read from that row is an anecdote; "
             "`figures/metric4_e2b.png` prints the denominators for the same reason.")
    if any(comp.get("compacted_prompt_preview") for _, _, comp in compacted):
        L.append("- **Compaction text is in the trace.** Each `compaction` record carries the full "
                 "Phase-2 message body under `compacted_prompt_preview`, plus the directive and the "
                 "size of the locked source, so the rebuilt context can be read without replaying "
                 "`compact_messages` against a deleted worktree.")
    else:
        L.append("- **Compaction text is not in the trace.** `harness/agent.py` logs the compaction's "
                 "sizes, locked files/symbols and the root-cause note, not the reconstructed "
                 "messages; recovering the text would require replaying `compact_messages` against "
                 "the worktree. (Traces written after 2026-10-21 also log "
                 "`compacted_prompt_preview`, so a run set without it predates that field.)")
    L.append("- **One trace per cell.** A cell that was re-run keeps only its newest scoreable "
             "trace, so the 50 tasks are counted once each, not once per attempt.")

    out = ROOT / args.out
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(f"cells={len(cells)} calls={total_calls} failed={total_fails} "
          f"compacted={n_comp}/{n_m3}")
    for v in variants:
        tot = sum(cat_tot[v].values())
        print(f"  {v}: {tot} errors " + " ".join(f"{c}={cat_tot[v][c]}" for c in CATS))
    print(f"  samples: M2={'yes' if s_a else 'NO'} M1={'yes' if s_b else 'NO'} "
          f"M3={'yes' if m3_sample else 'NO'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
