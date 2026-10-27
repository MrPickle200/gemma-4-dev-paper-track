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
import difflib
import inspect
import json
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
            # `additionalProperties: false` is what makes llama.cpp's constrained decoder refuse
            # an invented key: without it the grammar accepts any object shape, which is how M1's
            # cells produced a single key `pattern="x"\npath` instead of two parameters.
            "parameters": {"type": "object", "properties": properties,
                           "required": required, "additionalProperties": False},
        },
    }


# Backwards-compatible alias: older scripts imported the private helper.
_iter_files = iter_files


class PathError(ValueError):
    """A model-supplied path that cannot be mapped to a real path inside the repository.

    Distinct from a plain ValueError so `ToolBox.execute` can answer it with the repository
    context (closest matches) instead of a Python traceback.
    """


def _norm_rel(raw: str) -> str:
    """Trim what a model wraps a path in, and use '/' separators on every platform."""
    return (raw or "").strip().replace("\\", "/").lstrip("/")


def _inside(root: Path, candidate: Path) -> Path | None:
    """`candidate` resolved under `root`, or None when it would escape the worktree."""
    try:
        p = candidate.resolve()
        p.relative_to(root)
        return p
    except (ValueError, OSError):
        return None


def _repo_files(root: Path) -> list[str]:
    """Every non-vendor file in the worktree as a repo-relative posix path, sorted."""
    out = []
    for p in iter_files(root):
        try:
            out.append(p.relative_to(root).as_posix())
        except ValueError:
            continue
    return sorted(out)


def resolve_repo_path(worktree_root: Path, raw_path: str) -> Path:
    """Resolve a model-supplied path against the worktree root (TASK.md Task 1).

    Small models prefix paths with the clone's directory name or the owner/repo pair -
    `psf/requests/sessions.py`, `sympy/sympy/core/matrix.py`, `requests/requests/sessions.py` -
    even though the worktree root already *is* the repository. Those 18 `OTHER_ERROR not a file`
    calls were pure noise, and worse, a mis-resolved path in M3 left `lock_targets` with nothing
    to load into Phase 2.

    Order: the literal path; then the same path with leading directory components dropped, one at
    a time; then a repo-wide search for a *unique* suffix match. Anything else raises `PathError`
    carrying the five closest paths (difflib), so the model can correct itself in the next turn.
    """
    root = Path(worktree_root).resolve()
    raw = _norm_rel(raw_path)
    if raw in ("", "."):
        return root

    direct = _inside(root, root / raw)
    if direct is not None and direct.exists():
        return direct

    parts = [p for p in raw.split("/") if p not in ("", ".")]
    for i in range(1, len(parts)):
        cand = "/".join(parts[i:])
        p = _inside(root, root / cand)
        if p is not None and p.exists():
            return p

    files = _repo_files(root)
    suffix_hits = [rel for rel in files if rel == raw or rel.endswith("/" + raw)]
    if len(suffix_hits) == 1:
        return root / suffix_hits[0]
    if len(suffix_hits) > 1:
        shown = ", ".join(f"'{rel}'" for rel in suffix_hits[:5])
        raise PathError(f"ambiguous path: {len(suffix_hits)} files end with '{raw}' ({shown}). "
                        f"Pass one of those full paths.")

    close = difflib.get_close_matches(raw, files, n=5, cutoff=0.5)
    if not close:
        leaf = parts[-1] if parts else raw
        by_name = {Path(rel).name: rel for rel in files}
        close = [by_name[m] for m in difflib.get_close_matches(leaf, list(by_name), n=5, cutoff=0.6)]
    hint = (" Closest matches: " + ", ".join(f"'{c}'" for c in close) + ".") if close else ""
    raise PathError(f"not a file: {raw_path}. No such path in the repository.{hint}")


# --------------------------------------------------------------- edit matching (Task 3)
def _lf_and_unescape(text: str) -> tuple[str, list[int], list[int]]:
    """Normalise CRLF and literal '\\r\\n' escapes to '\\n', keeping a map back to the original.

    E2B's search_block contained the two-character sequence backslash-r backslash-n (see
    E2B_DIAGNOSTICS_REPORT.md §5.1), i.e. an escaped newline where the file has a real one, so a
    byte-exact comparison said "not found" forever. The reverse map matters when the *file* holds
    a literal '\\r\\n' inside a string literal: folding it in a copy is fine for matching, but
    writing that copy back would corrupt the source, so writes go through the original string.
    """
    norm: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    i = 0
    n = len(text)
    while i < n:
        if text.startswith("\\r\\n", i):
            norm.append("\n"); starts.append(i); ends.append(i + 4); i += 4
        elif text.startswith("\r\n", i):
            norm.append("\n"); starts.append(i); ends.append(i + 2); i += 2
        elif text[i] == "\r":
            norm.append("\n"); starts.append(i); ends.append(i + 1); i += 1
        else:
            norm.append(text[i]); starts.append(i); ends.append(i + 1); i += 1
    return "".join(norm), starts, ends


def _sanitize_block(text: str) -> str:
    """The model's search/replace text, normalised the same way as the file copy."""
    return _lf_and_unescape(text)[0]


def _line_spans(text: str) -> list[tuple[int, int, str]]:
    """(start, end, line) per line, `end` including the terminating newline when present."""
    out: list[tuple[int, int, str]] = []
    start = 0
    for line in text.split("\n"):
        end = start + len(line)
        out.append((start, min(end + 1, len(text)), line))
        start = end + 1
    return out


def _lines(text: str) -> list[str]:
    """Lines without the empty tail produced by a trailing newline."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


@dataclass
class MatchPlan:
    start: int = -1          # character offsets in the normalised text
    end: int = -1
    tier: int = 0            # 1 exact, 2 trailing-whitespace, 3 indentation-flexible
    indent: int = 0          # tier 3 only: columns the file is indented beyond the search block
    occurrences: int = 0
    lines: tuple[int, int] = (0, 0)   # 1-based first/last matched line
    ambiguous: list[int] = field(default_factory=list)     # tier 3: 1-based start lines


def _find_region(text: str, search: str) -> MatchPlan:
    """Locate `search` in the normalised file text, in three progressively fuzzier tiers.

    Tier 1 is a byte-exact substring. Tier 2 compares `rstrip()`ed lines, because a model
    re-emitting a code block routinely drops trailing spaces. Tier 3 compares `strip()`ed
    non-blank lines and only accepts a *single* location: it re-derives the indentation offset so
    `replace_block` can be shifted to match, and that guess is unsafe unless the site is unique.
    """
    if not search:
        return MatchPlan()

    # ---- tier 1: exact
    first = text.find(search)
    if first >= 0:
        count = text.count(search)
        start_line = text.count("\n", 0, first) + 1
        end_line = start_line + search.count("\n") - (1 if search.endswith("\n") else 0)
        return MatchPlan(start=first, end=first + len(search), tier=1, occurrences=count,
                         lines=(start_line, max(start_line, end_line)))

    file_spans = _line_spans(text)
    file_lines = [s[2] for s in file_spans]
    search_lines = _lines(search)
    if not search_lines:
        return MatchPlan()

    # ---- tier 2: trailing whitespace
    fkeys = [ln.rstrip() for ln in file_lines]
    skeys = [ln.rstrip() for ln in search_lines]
    starts = [i for i in range(0, len(fkeys) - len(skeys) + 1)
              if fkeys[i:i + len(skeys)] == skeys]
    if starts:
        return _plan_from_lines(file_spans, starts, len(skeys), tier=2)

    # ---- tier 3: indentation / blank-line insensitive, unique match only
    fpair = [(i, ln.strip()) for i, ln in enumerate(file_lines) if ln.strip()]
    skeys = [ln.strip() for ln in search_lines if ln.strip()]
    if not skeys:
        return MatchPlan()
    hits = [k for k in range(0, len(fpair) - len(skeys) + 1)
            if [s for _, s in fpair[k:k + len(skeys)]] == skeys]
    if not hits:
        return MatchPlan()
    if len(hits) > 1:
        lines = [fpair[k][0] + 1 for k in hits]
        return MatchPlan(occurrences=len(hits), tier=3, ambiguous=lines)
    k = hits[0]
    first_line, last_line = fpair[k][0], fpair[k + len(skeys) - 1][0]
    search_first = next(ln for ln in search_lines if ln.strip())
    indent = _indent(file_lines[first_line]) - _indent(search_first)
    plan = _plan_from_lines(file_spans, [first_line], last_line - first_line + 1, tier=3)
    plan.indent = indent
    return plan


def _plan_from_lines(spans: list[tuple[int, int, str]], starts: list[int], n: int,
                     tier: int) -> MatchPlan:
    start = spans[starts[0]][0]
    end = spans[starts[0] + n - 1][1]
    return MatchPlan(start=start, end=end, tier=tier, occurrences=len(starts),
                     lines=(starts[0] + 1, starts[0] + n))


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _reindent(block: str, offset: int) -> str:
    """Shift every non-blank line of `block` by `offset` columns of indentation."""
    if offset == 0:
        return block
    out = []
    for line in block.split("\n"):
        if not line.strip():
            out.append(line)
        elif offset > 0:
            out.append(" " * offset + line)
        else:
            drop = 0
            while drop < -offset and drop < len(line) and line[drop] == " ":
                drop += 1
            out.append(line[drop:])
    return "\n".join(out)


def closest_region(text: str, search: str, max_show: int = 20) -> tuple[int, int, str, float]:
    """The file region most similar to `search` (1-based lines, text, similarity).

    Candidate windows are anchored on lines that equal the search block's first non-blank line -
    that is where the model thinks the code starts - and fall back to every window when no line
    does. Scoring every window equally picks a region one line too early whenever two windows
    score the same, which is exactly the off-by-one that sent E2B round in circles.
    """
    file_lines = _lines(text)
    skey = [ln.strip() for ln in search.split("\n") if ln.strip()]
    if not file_lines or not skey:
        return 0, 0, "", 0.0
    k = len(skey)
    keys = [ln.strip() for ln in file_lines]
    anchors = [i for i, key in enumerate(keys) if key == skey[0]]
    starts = [i for i in anchors if i + k <= len(keys)] or list(range(max(1, len(keys) - k + 1)))
    best = (0.0, 0, 0)
    for i in starts:
        window = [x for x in keys[i:i + k] if x]
        if not window:
            continue
        r = difflib.SequenceMatcher(None, window, skey).ratio()
        if r > best[0]:
            best = (r, i, i + k)
    ratio, a, b = best
    shown = file_lines[a:min(b, a + max_show)]
    return a + 1, min(b, a + max_show), "\n".join(shown), round(ratio, 2)


# ------------------------------------------------------------- symbol references (Task 4)
def split_symbol_ref(symbol_name: str) -> tuple[str | None, str]:
    """Split 'requests/sessions.py::Session.request' into ('requests/sessions.py', 'Session.request').

    Small models pass the file along inside the symbol far more often than not, and a bare
    'file.py::name' would otherwise be treated as a symbol name and never resolve.
    """
    s = (symbol_name or "").strip()
    if "::" in s:
        f, _, nm = s.partition("::")
        f = f.strip()
        if f.lower().endswith(".py") or "/" in f or "\\" in f:
            return f, (nm.strip() or s)
    return None, s


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
        self.use_ast_nav = variant in ("M2", "M3", "M4")
        self.structured_edit = variant in ("M1", "M2", "M3", "M4")
        self.syntax_gate = variant in ("M1", "M2", "M3", "M4")
        self.allow_lock = variant in ("M3", "M4")
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
            "arguments_repaired": 0,   # glued keys recovered by the regex fallback (Task 2.3)
            "repeated_identical_calls": 0,  # stuck-in-a-loop turns -> metric #4
            "loop_blocked": 0,         # identical repeats the loop-breaker refused to re-execute
            "nav_calls": 0,            # M2/M3 tree-sitter navigation calls
            "lock_calls": 0,           # M3 lock_targets calls
            "lock_rejected": 0,        # M3 locks refused by the pre-compaction validation gate
            "symbols_locked": 0,       # M3 symbols handed to the patching phase
            "compactions": 0,          # M3 context compactions performed
        }
        self._last_call_sig: tuple | None = None

    # ---------------------------------------------------------------- schemas
    def specs(self, variant: str | None = None) -> list[dict]:
        variant = variant or self.variant
        tools: list[dict] = []
        if variant in ("M2", "M3", "M4"):
            tools += self._nav_specs()
        else:
            tools += self._bash_specs()
        if variant in ("M3", "M4"):
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
    # `pattern="def x"\npath` - two parameters glued into one key by the model. Gemma 4's own
    # template tokens (`<|"|>`, `<channel|>`) sometimes leak into the string as well, so the
    # value is unescaped as JSON when it looks quoted.
    _GLUED_KEY = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(.*?)\s*\n\s*([A-Za-z_]\w*)\s*$", re.S)

    def _repair_arguments(self, name: str, args: dict, handler) -> tuple[dict | None, str | None]:
        """Recover a tool call whose keys were glued together, or say what the keys should be.

        llama.cpp's constrained decoder will now refuse this shape (additionalProperties=false),
        but a fallback keeps old server builds usable and names the failure in the model's words
        instead of leaking a Python signature.
        """
        expected = list(inspect.signature(handler).parameters)
        glued = [k for k in args if "=" in k and "\n" in k]
        if not glued:
            return args, None

        repaired = {k: v for k, v in args.items() if k not in glued}
        ok = True
        for key in glued:
            m = self._GLUED_KEY.match(key)
            if not m:
                ok = False
                continue
            left, value, right = m.group(1), m.group(2).strip(), m.group(3)
            if value[:1] == value[-1:] == '"':
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    value = value[1:-1]
            if left not in expected or right not in expected:
                ok = False
                continue
            repaired[left] = value
            repaired[right] = args[key]

        if ok:
            return repaired, None
        names = "', '".join(expected)
        return None, (f"Invalid JSON keys. Expected separate keys '{names}' for {name}, "
                      f"not one key containing '=' and a newline. Send one JSON object whose "
                      f"keys are exactly those parameter names.")

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

        original_args = args
        args, repair_err = self._repair_arguments(name, args, handler)
        if repair_err is not None:
            self.stats["tool_errors"] += 1
            self.stats["malformed_arguments"] += 1
            return ToolResult(ok=False, error=repair_err, meta={"args_raw": original_args})
        repaired = args is not original_args
        if repaired:
            self.stats["arguments_repaired"] += 1

        try:
            res = handler(**args)
        except PathError as e:
            # A path that does not resolve is a model-supplied path problem, not a crash: hand
            # back the repository's closest matches (Task 1) so the next turn can be correct.
            self.stats["tool_errors"] += 1
            return ToolResult(ok=False, error=f"{e} {self._path_hint()}")
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
        if repaired:
            res.meta["arguments_repaired"] = {"from": original_args, "to": args}
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
        """Map a model-supplied path to a real path in the worktree (Task 1).

        Raises `PathError` (a ValueError) when no unique target exists; `execute` turns that into
        a tool result that names the closest paths in the repository.
        """
        return resolve_repo_path(self.workdir, rel)

    def _path_hint(self) -> str:
        hint = (f"Paths are relative to the repository root, e.g. "
                f"'{self.repo_leaf}/sessions.py'.")
        if self.repo_name:
            hint += (f" A redundant owner prefix ('{self.repo_name}/sessions.py') is accepted "
                     f"as well.")
        return hint

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
        file_part, name = split_symbol_ref(symbol_name)
        p = None
        if file_part:
            try:
                p = self._resolve(file_part)
            except PathError:
                p = None            # the embedded path is wrong; trust the `file` argument
        if p is None:
            p = self._resolve(file)
        rel = p.relative_to(self.workdir).as_posix()
        if not p.is_file():
            raise PathError(f"not a file: {file_part or file}")

        found = astnav.find_symbol(p, name, rel)
        if not found:
            return self._symbol_miss(symbol_name, name, rel, astnav.symbols(p, rel))
        if len(found) > 1 and "." not in name:
            quals = [s.qualified for s in found]
            return ToolResult(
                ok=False,
                error=(f"Symbol '{symbol_name}' is ambiguous in '{rel}': {len(found)} definitions "
                       f"match {quals}. Pass the qualified name, e.g. '{quals[0]}'."),
                meta={"candidates": quals},
            )
        chunks = []
        for s in found:
            header = f"# {rel}:{s.start_line}-{s.end_line}  {s.signature}"
            chunks.append(header + "\n" + astnav.read_source(p, s))
        return ToolResult(ok=True, output="\n\n".join(chunks),
                          meta={"symbols": [s.qualified for s in found], "count": len(found),
                                "resolved_file": rel})

    def _symbol_miss(self, query: str, name: str, rel: str,
                     syms: list) -> ToolResult:
        """A miss that lists what the file *does* define (Task 4.2), never a bare error."""
        available = [s.qualified for s in syms]
        close = difflib.get_close_matches(name.split(".")[-1], available, n=10, cutoff=0.6)
        if not close:
            close = difflib.get_close_matches(name.split(".")[-1], [s.name for s in syms],
                                              n=10, cutoff=0.6)
        msg = (f"Symbol '{query}' not found in '{rel}'. Available symbols in this file: "
               f"{available[:40] if available else '[]'}.")
        if close:
            msg += f" Closest matches in this file: {close}."
        if not available:
            msg += " This file defines no classes or functions."
        return ToolResult(ok=False, error=msg,
                          meta={"available_symbols": available[:40], "closest": close})

    def _find_references(self, symbol_name: str, path: str = ".", max_results: int = 40) -> ToolResult:
        self.stats["nav_calls"] += 1
        _, name = split_symbol_ref(symbol_name)
        root = self._resolve(path)
        hits = astnav.find_references(root, name, base=self.workdir, max_results=max_results)
        meta = {"hits": len(hits), "style": "identifier-only", "symbol": name}
        if not hits:
            return ToolResult(ok=True, output=f"No code references to {name!r}.", meta=meta)
        return ToolResult(ok=True, output="\n".join(f"{r}:{ln}: {t}" for r, ln, t in hits), meta=meta)

    def _lock_targets(self, files: list, symbols: list, root_cause_note: str = "") -> ToolResult:
        """Validate the lock, then let the agent compact (TASK.md Task 5.1).

        A lock that names a path the repository does not have, or a symbol that yields no source,
        used to be accepted silently: Phase 2 then started with a context that held no code, the
        model re-ran Phase-1 navigation, and a third of M3's tasks lost the phase transition. The
        gate refuses such a lock, leaves Phase 1 intact and answers with the symbols the file
        really defines.
        """
        self.stats["lock_calls"] += 1
        files = [str(f) for f in (files or [])]
        symbols = [str(s) for s in (symbols or [])]
        problems: list[str] = []
        suggestions: list[str] = []

        norm_files: list[str] = []
        for f in files:
            try:
                p = self._resolve(f)
            except PathError as e:
                problems.append(f"file {e}")
                continue
            if not p.is_file():
                problems.append(f"file '{f}' is not a file")
                continue
            rel = p.relative_to(self.workdir).as_posix()
            if rel not in norm_files:
                norm_files.append(rel)

        norm_syms: list[str] = []
        for s in symbols:
            file_part, name = split_symbol_ref(s)
            candidates = [file_part] if file_part else list(norm_files)
            hit = False
            for rel_in in candidates:
                try:
                    p = self._resolve(rel_in)
                except PathError:
                    continue
                if not p.is_file():
                    continue
                rel = p.relative_to(self.workdir).as_posix()
                found = astnav.find_symbol(p, name, rel)
                if not found:
                    continue
                src = "".join(astnav.read_source(p, sym) for sym in found)
                if not src.strip():
                    continue
                hit = True
                entry = f"{rel}::{name}"
                if entry not in norm_syms:
                    norm_syms.append(entry)
                break
            if not hit:
                pool: list[str] = []
                for rel_in in candidates:
                    try:
                        p = self._resolve(rel_in)
                    except PathError:
                        continue
                    if p.is_file():
                        rel = p.relative_to(self.workdir).as_posix()
                        pool += [sy.qualified for sy in astnav.symbols(p, rel)]
                where = file_part or (", ".join(norm_files) or "the repository")
                problems.append(f"symbol '{s}' has no source in {where}")
                suggestions += pool[:10]

        if not norm_files and not norm_syms:
            problems.append("no file and no symbol resolved")
        if problems:
            self.stats["lock_rejected"] += 1
            hint = f" Available symbols: {suggestions[:10]}." if suggestions else ""
            return ToolResult(
                ok=False,
                error=("Cannot start the patching phase - the locked targets do not resolve. "
                       + "; ".join(problems)
                       + hint
                       + " You are still in the localization phase: call lock_targets again with "
                         "paths relative to the repository root and symbol names that exist in "
                         "the named file."),
                meta={"lock_rejected": True, "suggestions": suggestions[:10]},
            )

        repeat = self.locked is not None
        self.stats["symbols_locked"] += len(norm_syms)
        self.locked = {
            "files": norm_files,
            "symbols": norm_syms,
            "root_cause_note": root_cause_note or "",
            "whole_files": not norm_syms,     # no symbol named -> Phase 2 gets whole files
        }
        if repeat:
            output = ("Targets updated. You are already in the patching phase - make the edit "
                      "with apply_edit now, or call finish if it is already done.")
        else:
            output = (f"Locked {len(norm_syms)} symbol(s) across {len(norm_files)} file(s). "
                      f"Scratch history cleared - continue in the patch phase.")
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

        original = p.read_bytes().decode("utf-8", errors="ignore")
        newline = "\r\n" if "\r\n" in original else "\n"
        text, starts, ends = _lf_and_unescape(original)
        search = _sanitize_block(search_block)
        replace = _sanitize_block(replace_block)
        if not search.strip():
            self.stats["edits_malformed"] += 1
            return ToolResult(ok=False, error="search_block is empty after normalising newlines")

        plan = _find_region(text, search)
        if plan.occurrences == 0:
            self.stats["edits_malformed"] += 1
            return ToolResult(
                ok=False,
                error=_mismatch_message(text, search),
                meta={"tier": 0, "target_symbol": target_symbol, "wrote": False},
            )
        if plan.tier == 3 and plan.occurrences > 1:
            self.stats["edits_malformed"] += 1
            return ToolResult(
                ok=False,
                error=("search_block matches more than one place once indentation is ignored "
                       f"(candidate start lines: {plan.ambiguous[:10]}). Include the surrounding "
                       "code so the site is unique, and copy its indentation exactly."),
                meta={"tier": 3, "ambiguous_lines": plan.ambiguous, "wrote": False},
            )
        if plan.occurrences > 1:
            self.stats["edits_ambiguous"] += 1

        # A no-op edit (identical search and replace) is a model failure that still "succeeds";
        # count it separately so the edit-rate metric is not inflated by do-nothing edits.
        noop = search == replace
        if noop:
            self.stats["edits_noop"] += 1

        replacement = _reindent(replace, plan.indent)
        if text[plan.start:plan.end].endswith("\n") and not replacement.endswith("\n"):
            replacement += "\n"

        # Map the normalised offsets back onto the original bytes so a literal '\r\n' escape
        # elsewhere in the file is written back unchanged.
        orig_start = starts[plan.start] if plan.start < len(starts) else len(original)
        orig_end = ends[plan.end - 1] if plan.end > plan.start else orig_start
        new_text = (original[:orig_start]
                    + replacement.replace("\n", newline)
                    + original[orig_end:])

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
                    meta={"syntax_error": err, "target_symbol": target_symbol, "wrote": False,
                          "tier": plan.tier},
                )

        p.write_bytes(new_text.encode("utf-8"))
        tier_note = {1: "exact", 2: "trailing-whitespace-insensitive",
                     3: f"indentation-flexible, indentation offset {plan.indent:+d}"}[plan.tier]
        return ToolResult(
            ok=True,
            output=(f"Edited {file}: replaced 1 occurrence in lines {plan.lines[0]}-"
                    f"{plan.lines[1]} ({plan.occurrences} match(es) found; matched "
                    f"{tier_note})."),
            meta={"occurrences": plan.occurrences, "ambiguous": plan.occurrences > 1,
                  "target_symbol": target_symbol, "noop": noop, "tier": plan.tier,
                  "indent_offset": plan.indent, "lines": list(plan.lines),
                  "syntax_checked": bool(self.syntax_gate and p.suffix == ".py")},
        )

    def _finish(self, summary: str) -> ToolResult:
        return ToolResult(ok=True, output="done", meta={"summary": summary, "finish": True})


def _mismatch_message(text: str, search: str) -> str:
    """A failed edit's answer: what was tried, and the closest region in the file.

    "search_block not found" is kept as the leading words on purpose - it is the string the
    diagnostics taxonomy classifies as SEARCH_BLOCK_MISMATCH, and old traces use it too.
    """
    first, last, snippet, ratio = closest_region(text, search)
    msg = ("search_block not found in the file - copy it verbatim, including indentation, and "
           "re-issue the edit. Matching is tried at three levels (exact, trailing-whitespace-"
           "insensitive, indentation-flexible) after normalising CRLF and literal '\\r\\n' "
           "escapes.")
    if snippet and ratio >= 0.35:
        numbered = "\n".join(f"{i:>5}| {line}"
                             for i, line in enumerate(snippet.split("\n"), first))
        msg += (f"\nThe closest region is lines {first}-{last} (similarity {ratio:.2f}). Line "
                f"numbers are for reference only - never copy them into search_block:\n{numbered}")
    return msg


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
