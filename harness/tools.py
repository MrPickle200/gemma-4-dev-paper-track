"""Tool implementations and their OpenAI function schemas.

B0 (IDEA.md §3, ablation row 1) is the *baseline*: deliberately the crude text tools a
weak model would get from an ordinary coding agent — text search, file reads, directory
listings, and a raw search/replace edit with no syntax check. M1/M2/M3 replace parts of
this toolbox (constrained JSON edit, syntax gate, tree-sitter navigation); keeping B0
honest is what makes the ablation meaningful.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", "build", "dist",
             ".mypy_cache", ".pytest_cache", ".tox", ".eggs"}
SKIP_EXTS = {".pyc", ".pyo", ".so", ".dll", ".pyd", ".png", ".jpg", ".jpeg", ".gif",
             ".zip", ".gz", ".whl", ".pdf", ".ico", ".npz", ".npy"}
MAX_SCAN_BYTES = 2_000_000


@dataclass
class ToolResult:
    ok: bool
    output: str = ""
    error: str | None = None
    meta: dict = field(default_factory=dict)


def _fn_schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def _iter_files(root: Path):
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


class ToolBox:
    def __init__(self, workdir: Path, max_output_chars: int = 8000, repo_name: str = "",
                 line_numbers: bool = False):
        self.workdir = Path(workdir).resolve()
        self.max_output_chars = max_output_chars
        self.repo_name = repo_name                    # e.g. "psf/requests"
        self.repo_leaf = repo_name.split("/")[-1]     # e.g. "requests"
        # B0 is specified as "cat" (IDEA.md §3), and cat does not print line numbers.
        # Printing them makes small models copy the "314: " prefixes into search_block,
        # which would manufacture malformed edits and corrupt the baseline.
        self.line_numbers = line_numbers
        # counters feeding the metrics in IDEA.md §4
        self.stats = {
            "tool_calls": 0,
            "tool_errors": 0,
            "edits_attempted": 0,
            "edits_malformed": 0,      # search_block missing -> metric #2 (Malformed Edit Rate)
            "edits_ambiguous": 0,      # search_block matched more than once
            "malformed_arguments": 0,  # model emitted un-parseable tool arguments -> metric #2
            "repeated_identical_calls": 0,  # stuck-in-a-loop turns -> metric #4
        }
        self._last_call_sig: tuple | None = None

    # ---------------------------------------------------------------- schemas
    def specs(self, variant: str = "B0") -> list[dict]:
        return [
            _fn_schema(
                "search_text",
                "Search the repository for a Python regular expression and return matching "
                "'path:line: text' entries. This is the only way to locate code. The pattern is "
                "a regex only - it cannot express 'in some file'; to restrict the search pass "
                "'path' instead (e.g. path='requests/sessions.py').",
                {
                    "pattern": {"type": "string", "description": "Python regular expression"},
                    "path": {"type": "string", "description": "Directory or file to search, relative to the repo root"},
                    "max_results": {"type": "integer", "description": "Cap on returned matches (default 40)"},
                },
                ["pattern"],
            ),
            _fn_schema(
                "read_file",
                "Read a text file exactly as `cat` would, optionally restricted to a line range. "
                "The output has no line numbers, so any text you copy from it is already safe to "
                "use as `search_block`.",
                {
                    "file": {"type": "string", "description": "Path relative to the repo root"},
                    "start_line": {"type": "integer", "description": "1-based first line (default 1)"},
                    "end_line": {"type": "integer", "description": "1-based last line, inclusive"},
                },
                ["file"],
            ),
            _fn_schema(
                "list_dir",
                "List files and directories under a path, two levels deep.",
                {"path": {"type": "string", "description": "Directory relative to the repo root (default '.')"}},
                [],
            ),
            _fn_schema(
                "apply_edit",
                "Replace the first exact occurrence of search_block with replace_block in a file. "
                "search_block must be copied verbatim from the file, including indentation, and must "
                "be unique enough to identify the site. Nothing is written if search_block is not found.",
                {
                    "file": {"type": "string", "description": "Path relative to the repo root"},
                    "search_block": {"type": "string", "description": "Exact existing text, whitespace preserved"},
                    "replace_block": {"type": "string", "description": "Replacement text"},
                },
                ["file", "search_block", "replace_block"],
            ),
            _fn_schema(
                "finish",
                "Call this when the fix is complete. Stops the run.",
                {"summary": {"type": "string", "description": "What you changed and why"}},
                ["summary"],
            ),
        ]

    # --------------------------------------------------------------- dispatch
    def execute(self, name: str, args: dict) -> ToolResult:
        self.stats["tool_calls"] += 1
        handler = {
            "search_text": self._search_text,
            "read_file": self._read_file,
            "list_dir": self._list_dir,
            "apply_edit": self._apply_edit,
            "finish": self._finish,
        }.get(name)
        if handler is None:
            self.stats["tool_errors"] += 1
            return ToolResult(ok=False, error=f"unknown tool '{name}'")
        try:
            res = handler(**args)
        except TypeError as e:
            self.stats["tool_errors"] += 1
            return ToolResult(ok=False, error=f"bad arguments for {name}: {e}")
        if not res.ok:
            self.stats["tool_errors"] += 1
        return self._truncate(res)

    def _truncate(self, res: ToolResult) -> ToolResult:
        if len(res.output) <= self.max_output_chars:
            return res
        res.meta["truncated"] = True
        res.meta["full_chars"] = len(res.output)
        res.output = (
            res.output[: self.max_output_chars]
            + f"\n... [truncated: {len(res.output)} chars total, showing {self.max_output_chars}]"
        )
        return res

    def _resolve(self, rel: str) -> Path:
        """Resolve a model-supplied path against the repo root.

        Small models routinely prefix paths with the repo name (e.g.
        'psf/requests/sessions.py' or 'requests/sessions.py') even though the worktree
        root already IS the repo. Accept those forms rather than burning a turn on it.
        """
        rel = (rel or "").lstrip("./\\").replace("\\", "/")
        candidates = [rel]
        for prefix in (f"{self.repo_name}/", f"{self.repo_leaf}/"):
            if self.repo_name and rel.startswith(prefix):
                candidates.append(rel[len(prefix):])
        for cand in candidates:
            p = (self.workdir / cand).resolve()
            if str(p).startswith(str(self.workdir)) and p.exists():
                return p
        p = (self.workdir / candidates[-1]).resolve()
        if not str(p).startswith(str(self.workdir)):
            raise ValueError(f"path escapes the repository: {rel}")
        return p

    def _path_hint(self) -> str:
        return (f"Paths are relative to the repository root, so use e.g. "
                f"'{self.repo_leaf}/sessions.py', not '{self.repo_name}/sessions.py'.")

    # ----------------------------------------------------------------- tools
    def _search_text(self, pattern: str, path: str = ".", max_results: int = 40) -> ToolResult:
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return ToolResult(ok=False, error=f"invalid regex: {e}")
        root = self._resolve(path)
        targets = [root] if root.is_file() else sorted(_iter_files(root))
        hits, scanned = [], 0
        for f in targets:
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            scanned += 1
            for i, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    rel = f.relative_to(self.workdir).as_posix()
                    hits.append(f"{rel}:{i}: {line.strip()[:200]}")
                    if len(hits) >= max_results:
                        break
            if len(hits) >= max_results:
                break
        meta = {"scanned_files": scanned, "hits": len(hits)}
        if not hits:
            return ToolResult(ok=True, output=f"No matches for {pattern!r} under {path}.", meta=meta)
        return ToolResult(ok=True, output="\n".join(hits), meta=meta)

    def _read_file(self, file: str, start_line: int = 1, end_line: int | None = None) -> ToolResult:
        p = self._resolve(file)
        if not p.is_file():
            return ToolResult(ok=False, error=f"not a file: {file}. {self._path_hint()}")
        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        start = max(1, int(start_line))
        end = len(lines) if end_line is None else min(len(lines), int(end_line))
        if start > end:
            return ToolResult(ok=False, error=f"start_line {start} > end_line {end}")
        body = "\n".join(
            f"{i:>5}: {lines[i - 1]}" if self.line_numbers else lines[i - 1]
            for i in range(start, end + 1)
        )
        return ToolResult(ok=True, output=body,
                          meta={"total_lines": len(lines), "start": start, "end": end})

    def _list_dir(self, path: str = ".") -> ToolResult:
        root = self._resolve(path)
        if not root.is_dir():
            return ToolResult(ok=False, error=f"not a directory: {path}")
        out, count = [], 0
        for p in sorted(root.rglob("*")):
            rel = p.relative_to(root)
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            if len(rel.parts) > 2:
                continue
            count += 1
            out.append(rel.as_posix() + ("/" if p.is_dir() else ""))
        return ToolResult(ok=True, output="\n".join(out) or "(empty)", meta={"entries": count})

    def _apply_edit(self, file: str, search_block: str, replace_block: str) -> ToolResult:
        self.stats["edits_attempted"] += 1
        p = self._resolve(file)
        if not p.is_file():
            self.stats["edits_malformed"] += 1
            return ToolResult(ok=False, error=f"not a file: {file}. {self._path_hint()}")
        if search_block == "":
            self.stats["edits_malformed"] += 1
            return ToolResult(ok=False, error="search_block is empty")

        text = p.read_text(encoding="utf-8", errors="ignore")
        n = text.count(search_block)
        if n == 0:
            self.stats["edits_malformed"] += 1
            return ToolResult(
                ok=False,
                error="search_block not found in the file - copy it verbatim, including "
                      "indentation, and re-issue the edit.",
            )
        if n > 1:
            self.stats["edits_ambiguous"] += 1
        # B0 semantics: raw replace, no syntax check before writing (M1 adds the gate).
        p.write_text(text.replace(search_block, replace_block, 1), encoding="utf-8")
        return ToolResult(
            ok=True,
            output=f"Edited {file}: replaced 1 occurrence ({n} match(es) found).",
            meta={"occurrences": n, "ambiguous": n > 1},
        )

    def _finish(self, summary: str) -> ToolResult:
        return ToolResult(ok=True, output="done", meta={"summary": summary, "finish": True})
