"""Tool implementations and their OpenAI function schemas, for every ablation variant.

B0 (IDEA.md §3, ablation row 1) is the *baseline*: deliberately the crude text tools a weak
model would get from an ordinary coding agent - text search, file reads, directory listings,
and a raw search/replace edit with no syntax check.

The other variants replace parts of that toolbox:
  M1 (+AST-Edit)  structured edit schema (`target_symbol`) + in-memory `ast.parse` syntax gate
  M2 (+AST-Nav)   tree-sitter navigation (`get_repo_map`/`read_symbol`/`find_references`)
                  instead of the bash search tools
  M3 (full)       M2 + `lock_targets`, which triggers the agent's context compaction

Keeping B0 honest is what makes the ablation meaningful, so the B0 code paths are unchanged.
"""

from __future__ import annotations

import ast
import inspect
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import astnav
from .astnav import SKIP_DIRS, SKIP_EXTS, iter_files


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


# Backwards-compatible alias: older scripts imported the private helper.
_iter_files = iter_files


class ToolBox:
    def __init__(self, workdir: Path, max_output_chars: int = 8000, repo_name: str = "",
                 line_numbers: bool = False, variant: str = "B0"):
        self.workdir = Path(workdir).resolve()
        self.max_output_chars = max_output_chars
        self.repo_name = repo_name                    # e.g. "psf/requests"
        self.repo_leaf = repo_name.split("/")[-1]     # e.g. "requests"
        # B0 is specified as "cat" (IDEA.md §3), and cat does not print line numbers.
        # Printing them makes small models copy the "314: " prefixes into search_block,
        # which would manufacture malformed edits and corrupt the baseline.
        self.line_numbers = line_numbers
        self.variant = variant
        # The variant is the single switch that turns module behaviour on/off.
        self.use_ast_nav = variant in ("M2", "M3")
        self.structured_edit = variant in ("M1", "M2", "M3")
        self.syntax_gate = variant in ("M1", "M2", "M3")
        self.allow_lock = variant == "M3"
        self.locked: dict | None = None               # set by lock_targets (M3)
        # counters feeding the metrics in IDEA.md §4
        self.stats = {
            "tool_calls": 0,
            "tool_errors": 0,
            "edits_attempted": 0,
            "edits_malformed": 0,      # search_block missing -> metric #2 (Malformed Edit Rate)
            "edits_ambiguous": 0,      # search_block matched more than once
            "edits_noop": 0,           # search_block == replace_block (model "edited" nothing)
            "edits_syntax_rejected": 0,  # M1 syntax gate refused the write -> metric #2
            "malformed_arguments": 0,  # model emitted un-parseable tool arguments -> metric #2
            "repeated_identical_calls": 0,  # stuck-in-a-loop turns -> metric #4
            "nav_calls": 0,            # M2/M3 tree-sitter navigation calls
            "lock_calls": 0,           # M3 lock_targets calls
            "symbols_locked": 0,       # M3 symbols handed to the patching phase
            "compactions": 0,          # M3 context compactions performed
        }
        self._last_call_sig: tuple | None = None

    # ---------------------------------------------------------------- schemas
    def specs(self, variant: str | None = None) -> list[dict]:
        variant = variant or self.variant
        tools: list[dict] = []
        if variant in ("M2", "M3"):
            tools += self._nav_specs()
        else:
            tools += self._bash_specs()
        if variant == "M3":
            tools.append(self._lock_spec())
        tools.append(self._edit_spec())
        tools.append(_fn_schema(
            "finish",
            "Call this when the fix is complete. Stops the run.",
            {"summary": {"type": "string", "description": "What you changed and why"}},
            ["summary"],
        ))
        return tools

    def _bash_specs(self) -> list[dict]:
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
        ]

    def _nav_specs(self) -> list[dict]:
        return [
            _fn_schema(
                "get_repo_map",
                "Return the skeleton of a file or directory: every class and function with its "
                "signature and docstring, with all bodies hidden. Use this instead of listing or "
                "reading whole files - it shows you the structure for very few tokens.",
                {
                    "path": {"type": "string", "description": "File or directory relative to the repo root (default '.')"},
                },
                [],
            ),
            _fn_schema(
                "read_symbol",
                "Load the full source of exactly one function or class, by name. A nested method "
                "may be named as 'Class.method' (e.g. 'Session.send'). The text has no line "
                "numbers, so it is safe to copy straight into `search_block`.",
                {
                    "file": {"type": "string", "description": "Path relative to the repo root"},
                    "symbol_name": {"type": "string", "description": "Name of the function or class, e.g. 'send' or 'Session.send'"},
                },
                ["file", "symbol_name"],
            ),
            _fn_schema(
                "find_references",
                "Find where a name is used across the repository. Only real code identifiers are "
                "returned - never matches inside comments or string literals. Returns 'path:line: text'.",
                {
                    "symbol_name": {"type": "string", "description": "Identifier to look for"},
                    "path": {"type": "string", "description": "Directory or file to search (default '.')"},
                    "max_results": {"type": "integer", "description": "Cap on returned matches (default 40)"},
                },
                ["symbol_name"],
            ),
        ]

    def _lock_spec(self) -> dict:
        return _fn_schema(
            "lock_targets",
            "End the localization phase. Declare the file(s), the symbol(s) you will change, and "
            "a one-sentence root-cause note. This clears your scratch search history and starts the "
            "patching phase with a clean context containing only the issue, your note and the locked "
            "symbols. Call this once, as soon as you know where the bug is.",
            {
                "files": {"type": "array", "items": {"type": "string"},
                          "description": "Relative paths of the file(s) you will edit"},
                "symbols": {"type": "array", "items": {"type": "string"},
                            "description": "The function(s)/class(es) to change, e.g. 'requests/sessions.py::Session.send'"},
                "root_cause_note": {"type": "string", "description": "One sentence: what is wrong and what to change"},
            },
            ["files", "symbols", "root_cause_note"],
        )

    def _edit_spec(self) -> dict:
        if not self.structured_edit:
            return _fn_schema(
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
            )
        return _fn_schema(
            "apply_edit",
            "Apply a search/replace edit to one file. search_block must be copied verbatim from the "
            "file, including indentation (never include line numbers). replace_block is the new text. "
            "The edit is parsed with ast.parse() before anything is written: if it would leave the "
            "file with a Python syntax error it is rejected, nothing is written, and you get the error "
            "back so you can fix it in the next turn.",
            {
                "file": {"type": "string", "description": "Path relative to the repo root"},
                "target_symbol": {"type": "string", "description": "Name of the function or class the edit belongs to"},
                "search_block": {"type": "string", "description": "Exact existing text, whitespace preserved"},
                "replace_block": {"type": "string", "description": "Replacement text"},
            },
            ["file", "target_symbol", "search_block", "replace_block"],
        )

    # --------------------------------------------------------------- dispatch
    def execute(self, name: str, args: dict) -> ToolResult:
        self.stats["tool_calls"] += 1
        handler = {
            "search_text": self._search_text,
            "read_file": self._read_file,
            "list_dir": self._list_dir,
            "get_repo_map": self._get_repo_map,
            "read_symbol": self._read_symbol,
            "find_references": self._find_references,
            "lock_targets": self._lock_targets,
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
            # Arguments that do not match the tool's schema are a *malformed call* (metric #2),
            # not a rejected edit - the call never reached the tool. And the model can only act on
            # the parameters it may use, so do not hand it our Python signature: E2B produced 83
            # such calls under M1's constrained decoding (template tokens inside the argument
            # string), and `ToolBox._search_text() got an unexpected keyword argument` told it
            # nothing. This counter is the shared target of metric #2 and #4; see PROGRESS.md.
            self.stats["malformed_arguments"] += 1
            expected = ", ".join(inspect.signature(handler).parameters)
            return ToolResult(ok=False, error=f"bad arguments for {name}: got "
                                              f"{sorted(args)} but the tool accepts "
                                              f"({expected})")
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

    # ------------------------------------------------------------ bash tools
    def _search_text(self, pattern: str, path: str = ".", max_results: int = 40) -> ToolResult:
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return ToolResult(ok=False, error=f"invalid regex: {e}")
        root = self._resolve(path)
        targets = [root] if root.is_file() else sorted(iter_files(root))
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

    # ------------------------------------------------------------- AST tools
    def _get_repo_map(self, path: str = ".") -> ToolResult:
        self.stats["nav_calls"] += 1
        root = self._resolve(path)
        if root.is_file() and root.suffix != ".py":
            return ToolResult(ok=False, error=f"not a Python file: {path}")
        text = astnav.repo_map(root, base=self.workdir)
        return ToolResult(ok=True, output=text,
                          meta={"bytes": len(text), "style": "skeleton"})

    def _read_symbol(self, file: str, symbol_name: str) -> ToolResult:
        self.stats["nav_calls"] += 1
        p = self._resolve(file)
        if not p.is_file():
            return ToolResult(ok=False, error=f"not a file: {file}. {self._path_hint()}")
        rel = p.relative_to(self.workdir).as_posix()
        found = astnav.find_symbol(p, symbol_name, rel)
        if not found:
            names = [s.qualified for s in astnav.symbols(p, rel)][:40]
            return ToolResult(
                ok=False,
                error=f"no symbol named '{symbol_name}' in {file}. Available symbols: {names}",
            )
        chunks = []
        for s in found:
            header = f"# {rel}:{s.start_line}-{s.end_line}  {s.signature}"
            chunks.append(header + "\n" + astnav.read_source(p, s))
        return ToolResult(ok=True, output="\n\n".join(chunks),
                          meta={"symbols": [s.name for s in found], "count": len(found)})

    def _find_references(self, symbol_name: str, path: str = ".", max_results: int = 40) -> ToolResult:
        self.stats["nav_calls"] += 1
        root = self._resolve(path)
        hits = astnav.find_references(root, symbol_name, base=self.workdir, max_results=max_results)
        meta = {"hits": len(hits), "style": "identifier-only"}
        if not hits:
            return ToolResult(ok=True, output=f"No code references to {symbol_name!r}.", meta=meta)
        return ToolResult(ok=True, output="\n".join(f"{r}:{ln}: {t}" for r, ln, t in hits), meta=meta)

    def _lock_targets(self, files: list, symbols: list, root_cause_note: str = "") -> ToolResult:
        self.stats["lock_calls"] += 1
        self.stats["symbols_locked"] += len(symbols or [])
        repeat = self.locked is not None
        self.locked = {
            "files": [str(f) for f in (files or [])],
            "symbols": [str(s) for s in (symbols or [])],
            "root_cause_note": root_cause_note or "",
        }
        if repeat:
            output = ("Targets updated. You are already in the patching phase - make the edit "
                      "with apply_edit now, or call finish if it is already done.")
        else:
            output = (f"Locked {len(self.locked['symbols'])} symbol(s) across "
                      f"{len(self.locked['files'])} file(s). Scratch history cleared - continue "
                      f"in the patch phase.")
        return ToolResult(ok=True, output=output, meta={"lock": True, "locked": self.locked})

    # ------------------------------------------------------------ edit tools
    def _apply_edit(self, file: str, search_block: str, replace_block: str,
                    target_symbol: str | None = None) -> ToolResult:
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

        # A no-op edit (identical search and replace) is a model failure that still "succeeds";
        # count it separately so the edit-rate metric is not inflated by do-nothing edits.
        noop = search_block == replace_block
        if noop:
            self.stats["edits_noop"] += 1

        new_text = text.replace(search_block, replace_block, 1)

        # Module 2 (IDEA.md §2): in-memory syntax gate. Nothing is written if the edit
        # would leave the file unparseable, so the repository never enters a broken state.
        if self.syntax_gate and p.suffix == ".py":
            err = _syntax_error(new_text)
            if err is not None:
                self.stats["edits_malformed"] += 1
                self.stats["edits_syntax_rejected"] += 1
                return ToolResult(
                    ok=False,
                    error=(f"syntax gate rejected the edit, nothing was written: {err}. "
                           f"Re-issue apply_edit with a corrected replace_block."),
                    meta={"syntax_error": err, "target_symbol": target_symbol, "wrote": False},
                )

        p.write_text(new_text, encoding="utf-8")
        return ToolResult(
            ok=True,
            output=f"Edited {file}: replaced 1 occurrence ({n} match(es) found).",
            meta={"occurrences": n, "ambiguous": n > 1, "target_symbol": target_symbol,
                  "noop": noop,
                  "syntax_checked": bool(self.syntax_gate and p.suffix == ".py")},
        )

    def _finish(self, summary: str) -> ToolResult:
        return ToolResult(ok=True, output="done", meta={"summary": summary, "finish": True})


def _syntax_error(text: str) -> str | None:
    """Return a short description if `text` is not valid Python, else None."""
    try:
        ast.parse(text)
    except SyntaxError as e:
        where = f"line {e.lineno}" + (f", col {e.offset}" if e.offset else "")
        return f"{e.msg} at {where}"
    except ValueError as e:            # e.g. source contains null bytes
        return str(e)
    return None
