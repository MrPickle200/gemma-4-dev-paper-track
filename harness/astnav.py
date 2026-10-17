"""AST-level navigation for the M2/M3 toolkits (IDEA.md §2 Module 1: AST-Guided Navigation).

Everything here is deterministic and only ever reads bytes that tree-sitter has labelled as
code (definitions, identifiers). Text inside comments or string literals can therefore never
produce a hit - which is precisely where `grep` misleads a small model.

This module also owns the repository file walker (skip-lists) so the B0 bash tools and the
AST tools traverse the tree identically.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import tree_sitter_language_pack as tslp

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", "build", "dist",
             ".mypy_cache", ".pytest_cache", ".tox", ".eggs"}
SKIP_EXTS = {".pyc", ".pyo", ".so", ".dll", ".pyd", ".png", ".jpg", ".jpeg", ".gif",
             ".zip", ".gz", ".whl", ".pdf", ".ico", ".npz", ".npy"}
MAX_SCAN_BYTES = 2_000_000

DEF_TYPES = {"function_definition", "class_definition"}

_tls = threading.local()


def parser():
    """A tree-sitter Python parser *per thread*.

    tree-sitter Parser objects are not thread-safe, and the Bước 4 grid runs several cells at once
    (run_batch.py --workers N), so each worker thread gets its own."""
    p = getattr(_tls, "parser", None)
    if p is None:
        p = tslp.get_parser("python")
        _tls.parser = p
    return p


def iter_files(root: Path):
    """Yield source files under `root`, skipping vendor/build noise and huge binaries."""
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.suffix.lower() in SKIP_EXTS:
            continue
        try:
            if p.stat().st_size > MAX_SCAN_BYTES:
                continue
        except OSError:
            continue
        yield p


@dataclass
class Symbol:
    name: str
    kind: str            # "class" | "def"
    file: str            # path relative to the walk base
    start_line: int      # 1-based, includes decorators
    end_line: int        # 1-based, inclusive
    start_byte: int
    end_byte: int
    depth: int           # 0 = module level, 1 = method, ...
    signature: str       # e.g. "def send(self, request, **kwargs):"
    docstring: str | None = None
    parent: str | None = None
    qualified: str = ""  # e.g. "Session.send"; lets a model ask for "Session.send"


def _clean_string(raw: str) -> str:
    """Strip a Python string literal down to its body (handles r/b/u/f prefixes, triple quotes)."""
    raw = raw.strip()
    i = 0
    while i < len(raw) and raw[i] not in "\"'":
        i += 1
    raw = raw[i:]
    for q in ('"""', "'''", '"', "'"):
        if raw.startswith(q):
            body = raw[len(q):]
            if body.endswith(q):
                body = body[:-len(q)]
            return body.strip()
    return raw


def _docstring(data: bytes, block) -> str | None:
    if block is None or not block.named_children:
        return None
    first = block.named_children[0]
    if first.type != "expression_statement" or not first.named_children:
        return None
    s = first.named_children[0]
    if s.type not in ("string", "concatenated_string"):
        return None
    return _clean_string(data[s.start_byte:s.end_byte].decode("utf-8", "replace"))


def symbols(path: Path, rel: str | None = None) -> list[Symbol]:
    """Every class/function definition in one file, in source order, with nesting depth."""
    data = path.read_bytes()
    tree = parser().parse(data)
    rel = rel or path.name
    out: list[Symbol] = []

    def walk(node, depth: int, parent: str | None) -> None:
        for child in node.named_children:
            inner = child
            if child.type == "decorated_definition":
                inner = child.child_by_field_name("definition")
            if inner is not None and inner.type in DEF_TYPES:
                name_node = inner.child_by_field_name("name")
                name = name_node.text.decode("utf-8", "replace") if name_node else "?"
                qualified = f"{parent}.{name}" if parent else name
                block = inner.child_by_field_name("body")
                sig_end = block.start_byte if block is not None else inner.end_byte
                sig = data[inner.start_byte:sig_end].decode("utf-8", "replace").strip()
                if not sig.endswith(":"):
                    sig += ":"
                out.append(Symbol(
                    name=name,
                    kind="class" if inner.type == "class_definition" else "def",
                    file=rel,
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                    start_byte=child.start_byte,
                    end_byte=child.end_byte,
                    depth=depth,
                    signature=" ".join(sig.split()),
                    docstring=_docstring(data, block),
                    parent=parent,
                    qualified=qualified,
                ))
                walk(inner, depth + 1, qualified)
            else:
                walk(child, depth, parent)

    walk(tree.root_node, 0, None)
    return out


def find_symbol(path: Path, name: str, rel: str | None = None) -> list[Symbol]:
    """Match by bare name ("send"), by qualified name ("Session.send"), or by a qualified
    suffix ("Session.send" matches a class nested deeper). Small models ask for methods as
    "Class.method", so that form has to work."""
    out = []
    for s in symbols(path, rel):
        if s.name == name or s.qualified == name:
            out.append(s)
        elif "." in name and s.qualified.endswith("." + name):
            out.append(s)
    return out


def read_source(path: Path, sym: Symbol) -> str:
    """The exact bytes of a symbol (decorators included). No line numbers, so it is safe to
    copy from directly into a `search_block`."""
    return path.read_bytes()[sym.start_byte:sym.end_byte].decode("utf-8", "replace")


def skeleton_lines(syms: list[Symbol], max_doc: int = 120) -> list[str]:
    lines = []
    for s in syms:
        pad = "    " * s.depth
        lines.append(f"{pad}{s.signature}")
        if s.docstring:
            first = s.docstring.strip().splitlines()[0][:max_doc]
            lines.append(f'{pad}    """{first}"""')
    return lines


def repo_map(root: Path, base: Path | None = None, max_files: int = 400) -> str:
    """Skeleton of a file, or of every .py file under a directory. Bodies are hidden."""
    base = base or root
    files = [root] if root.is_file() else sorted(p for p in iter_files(root) if p.suffix == ".py")
    chunks = []
    for f in files[:max_files]:
        rel = f.relative_to(base).as_posix() if _is_under(f, base) else f.name
        syms = symbols(f, rel)
        if not syms:
            continue
        chunks.append(rel + "\n" + "\n".join("  " + line for line in skeleton_lines(syms)))
    if len(files) > max_files:
        chunks.append(f"... [{len(files) - max_files} more files not shown]")
    return "\n\n".join(chunks) or "(no Python definitions found)"


def _is_under(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def find_references(root: Path, name: str, base: Path | None = None,
                     max_results: int = 40) -> list[tuple[str, int, str]]:
    """Every line where `name` appears as an *identifier* (never in a comment or a string).

    Returns (relative_path, 1-based line, stripped line text) tuples, de-duplicated per line.
    """
    base = base or root
    # "Session.send" is not an identifier; the reference is to the attribute name.
    name = name.split(".")[-1]
    targets = [root] if root.is_file() else list(iter_files(root))
    hits: list[tuple[str, int, str]] = []
    seen: set[tuple[str, int]] = set()
    for f in targets:
        if f.suffix != ".py":
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        data = text.encode("utf-8", "replace")
        tree = parser().parse(data)
        rel = f.relative_to(base).as_posix() if _is_under(f, base) else f.name
        lines = text.splitlines()
        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            if node.type == "identifier" and data[node.start_byte:node.end_byte].decode("utf-8", "replace") == name:
                line_no = node.start_point[0] + 1
                if (rel, line_no) not in seen:
                    seen.add((rel, line_no))
                    snippet = lines[line_no - 1].strip()[:200] if line_no <= len(lines) else ""
                    hits.append((rel, line_no, snippet))
                    if len(hits) >= max_results:
                        return sorted(hits, key=lambda h: (h[0], h[1]))
            stack.extend(node.children)
    return sorted(hits, key=lambda h: (h[0], h[1]))
