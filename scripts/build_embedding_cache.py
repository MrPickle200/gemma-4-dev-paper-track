"""Build Turn-0 seeding cache for M4 using Ollama embeddinggemma-2:270m.

Usage (Ollama server must be running on :11434):
  .\.venv\Scripts\python.exe scripts\build_embedding_cache.py
"""
import argparse, json, sys, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.runner import setup_worktree, teardown_worktree
from harness.seeding import extract_identifiers, ollama_embed, save_cache
import numpy as np


def extract_symbols(workdir: Path):
    from tree_sitter_language_pack import get_parser
    parser = get_parser("python")
    out = []
    skip_dirs = {"tests", "test", ".git", "__pycache__", "docs", "doc"}
    for p in workdir.rglob("*.py"):
        rel = p.relative_to(workdir).as_posix()
        if any(part in skip_dirs for part in p.relative_to(workdir).parts):
            continue
        if p.name.startswith("test_"):
            continue
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        tree = parser.parse(src.encode())
        # simple: find function_definition / class_definition nodes
        def walk(node, qual=""):
            if node.type in ("function_definition", "class_definition"):
                name_node = node.child_by_field_name("name")
                name = name_node.text.decode() if name_node else "?"
                qname = f"{qual}.{name}" if qual else name
                # signature: first line
                start = node.start_point[0]
                lines = src.splitlines()
                sig = lines[start].strip() if start < len(lines) else ""
                # docstring
                body = node.child_by_field_name("body")
                doc = ""
                if body and body.children:
                    first = body.children[0]
                    if first.type == "expression_statement" and first.children and first.children[0].type == "string":
                        doc = first.children[0].text.decode().strip().strip('"""').strip("'''")[:300]
                # body preview
                preview_lines = [l for l in lines[start+1:start+12] if l.strip()][:10]
                preview = "\n".join(preview_lines)
                symbol_id = f"{rel}::{qname}"
                doc_text = (f"File: {rel}\nSymbol: {symbol_id}\nSignature: {sig}\n"
                            f"Docstring: {doc}\nBody Preview: {preview}")
                out.append({"symbol_id": symbol_id, "signature": sig, "summary": doc[:120],
                            "document_text": doc_text})
                # recurse into class for methods
                if node.type == "class_definition":
                    for c in (body.children if body else []):
                        walk(c, qname)
                return
            for c in node.children:
                walk(c, qual)
        walk(tree.root_node)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="embeddinggemma-2:270m")
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    subset = json.loads((ROOT / "data/subset_50.json").read_text(encoding="utf-8"))
    cache_dir = ROOT / "cache/seeding"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def process(task):
        iid = task["instance_id"]
        out = cache_dir / f"{iid}.json"
        if out.exists():
            print("skip", iid)
            return
        workdir = setup_worktree(task, "M4")
        try:
            symbols = extract_symbols(workdir)
            query = task["problem_statement"]
            if not symbols:
                print("no symbols for", iid)
                return
            docs = [s["document_text"] for s in symbols]
            dv = ollama_embed(args.model, docs, args.base_url)
            qv = ollama_embed(args.model, [query], args.base_url)[0]
            scores = dv @ qv
            # lexical boost: +0.1 for identifier match
            idents = set(extract_identifiers(query))
            for i, s in enumerate(symbols):
                low = s["symbol_id"].lower()
                if any(ident.lower() in low for ident in idents):
                    scores[i] += 0.1
            order = np.argsort(-scores)[:args.k]
            top = []
            for i in order:
                s = dict(symbols[i])
                s["score"] = float(scores[i])
                top.append(s)
            save_cache(out, {"instance_id": iid, "top_symbols": top})
            print("done", iid, len(top), "of", len(symbols))
        finally:
            teardown_worktree(task, "M4")

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(process, subset))


if __name__ == "__main__":
    main()
