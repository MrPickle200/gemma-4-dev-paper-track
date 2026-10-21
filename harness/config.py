"""Run configuration and log schema constants."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
EXPERIMENTS = LOGS / "experiments"
RUNS = LOGS / "runs"                 # default location, unchanged for the committed E2B grid
PATCHES = LOGS / "patches"
WORKTREES = ROOT / "worktrees"
REPOS = ROOT / "repos"

# The tag of the experiment this process writes into, or None for the default logs/runs.
# Set through `set_experiment()` (scripts expose it as --experiment) so one batch = one directory.
EXPERIMENT: str | None = None

VARIANTS = ("B0", "M1", "M2", "M3")


def set_experiment(tag: str | None) -> Path:
    """Write this run's traces and patches under `logs/experiments/<tag>/` and return that dir.

    Every batch used to land in the flat `logs/runs/`, so a smoke test, a regression re-run and
    the full grid were only distinguishable by timestamp inside the file names. An experiment tag
    keeps its evidence together (runs, patches, manifest, tables, diagnostics report) and lets the
    aggregate scripts select it with a single `--experiment` instead of a hand-written glob.
    `tag=None` restores the default paths, which is what the committed E2B grid uses.
    """
    global EXPERIMENT, RUNS, PATCHES
    tag = (tag or "").strip().strip("/\\")
    EXPERIMENT = tag or None
    base = (EXPERIMENTS / tag) if tag else LOGS
    RUNS = base / "runs"
    PATCHES = base / "patches"
    RUNS.mkdir(parents=True, exist_ok=True)
    PATCHES.mkdir(parents=True, exist_ok=True)
    return base

# A run that ended because the *backend* failed (transport/HTTP/JSON) or the harness raised is not an
# experimental outcome: such cells must be re-run rather than skipped as "done" or averaged in.
INFRA_STOP_PREFIXES = ("llm_error", "harness_error")

# ...but `llm_error` conflates two unlike things. A transport error means the backend broke; a
# model-side error means the *model* produced output the server could not use. The second kind is an
# experimental result - context bloat is metric #3, and unparsable tool-call JSON is exactly what
# Module 1 exists to change - so it must be named and scored. Excluding it would both delete that
# evidence and re-run the cell forever, because the overflow recurs on every attempt.
MODEL_SIDE_ERRORS = (
    ("Context size has been exceeded", "context_exceeded"),        # metric #3 / Module 3
    ("Failed to parse tool call arguments", "toolcall_rejected"),   # Module 1's target
)

# The context each run is *designed* to have (any model's window here, `-Ctx 32768`).
MIN_RUN_CTX = 32768


def model_side_stop_reason(error: str | None) -> str | None:
    """The precise stop reason for a *model-side* failure, or None if this is not one."""
    txt = str(error or "")
    return next((label for marker, label in MODEL_SIDE_ERRORS if marker in txt), None)


def turn_error(recs: list[dict]) -> str:
    """The first failed turn's error message in a trace ('' when there is none).

    An `llm_error` cannot be classified from `run_end` alone: the message lives on the turn.
    """
    return next((r.get("error") or "" for r in recs
                 if r.get("type") == "turn" and r.get("error")), "")


def ctx_is_adequate(ctx_per_run: int | None) -> bool:
    """True when every concurrent run was guaranteed the whole designed context window.

    llama-server keeps ONE KV pool (`kv_unified = true` in its banner) that all slots draw from,
    so `-c 32768` with 4 slots gives each run ~8192 tokens, not 32768. The pool then decides when
    a long run dies - which is precisely the behaviour metric #3 is meant to measure - so such a
    trace must be re-run against a server sized `-c >= MIN_RUN_CTX * workers`, never scored.
    `None` (a trace from before this field existed) counts as inadequate: those servers were all
    launched `-c 32768`.
    """
    return ctx_per_run is not None and ctx_per_run >= MIN_RUN_CTX


def refined_stop_reason(stop_reason: str | None, error: str | None = None,
                        ctx_per_run: int | None = None) -> str:
    """The stop reason to *score*: `llm_error` refined by the failing message and by how much
    context the run actually had."""
    sr = str(stop_reason or "")
    if not sr.startswith(INFRA_STOP_PREFIXES):
        return sr
    label = model_side_stop_reason(error)
    if label is None:
        return sr                                    # the backend/proxy really did break
    if label == "context_exceeded" and not ctx_is_adequate(ctx_per_run):
        return sr                                    # a shared-pool artefact, not an outcome
    return label


def is_infra_failure(stop_reason: str | None, error: str | None = None,
                     ctx_per_run: int | None = None) -> bool:
    """True when a run ended because the backend or the harness broke - or because the backend
    starved it of context. Re-run those, never score them.

    Pass the failing turn's message (`turn_error(recs)`) and the context the run actually had
    (`ctx_per_run`, logged in `run_start`), or an `llm_error` is assumed to be infrastructure.
    """
    return refined_stop_reason(stop_reason, error, ctx_per_run).startswith(INFRA_STOP_PREFIXES)


def is_proxy_capped(cfg: dict) -> bool:
    """True for a trace that ran **buffered through a proxy** (a tunnel).

    Such a run is complete, but not safe to pool with uncapped ones: the proxy's read timeout
    (Cloudflare 524 at 125 s) kills any turn that stays silent that long, so the runs that
    survive are biased towards short turns - exactly the rambling Module 3 measures. Traces
    written before `stream` existed count as capped. See AGENTS.md and PROGRESS.md.
    """
    return bool(cfg.get("api_key_set")) and cfg.get("stream") is not True


@dataclass
class RunConfig:
    """Everything that determines one run, and that must therefore be logged."""

    instance_id: str
    model: str = "gemma-4-E2B"
    variant: str = "B0"
    base_url: str = "http://127.0.0.1:8080"          # OpenAI-compatible root
    temperature: float = 0.0                          # IDEA.md §3
    max_tokens: int = 16384                           # IDEA.md §3
    max_turns: int = 15                               # IDEA.md §3
    thinking: str = "off"                             # "auto" | "on" | "off"
    max_tool_output_chars: int = 8000                 # truncate tool results
    line_numbers: bool = False                        # B0 = plain `cat`; see tools.ToolBox
    stream: bool = True                               # SSE; required behind a proxied tunnel, see llm.py
    seed: int = 0
    # Tokens each concurrent run was guaranteed by the backend (server pool / workers), or None
    # for a trace made before this was recorded. Logged so `context_exceeded` can be told apart
    # from a shared-pool artefact; see ctx_is_adequate.
    ctx_per_run: int | None = None
    api_key: str | None = None                        # sent as Authorization: Bearer; see LLMClient
    run_id: str = ""
    workdir: Path | None = None
    patch_path: Path | None = None
    log_path: Path | None = None

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if self.thinking not in ("auto", "on", "off"):
            raise ValueError("thinking must be auto|on|off")
        if not self.run_id:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            self.run_id = f"{self.instance_id}__{self.model}__{self.variant}__{stamp}"
        RUNS.mkdir(parents=True, exist_ok=True)
        PATCHES.mkdir(parents=True, exist_ok=True)
        if self.log_path is None:
            self.log_path = RUNS / f"{self.run_id}.jsonl"
        if self.patch_path is None:
            self.patch_path = PATCHES / f"{self.run_id}.patch"

    @property
    def chat_template_kwargs(self) -> dict | None:
        """Gemma 4 thinking control. 'auto' sends nothing and lets the template decide
        (which, with tools present, means thinking ON)."""
        if self.thinking == "auto":
            return None
        return {"enable_thinking": self.thinking == "on"}

    def as_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "model": self.model,
            "variant": self.variant,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_turns": self.max_turns,
            "thinking": self.thinking,
            "max_tool_output_chars": self.max_tool_output_chars,
            "line_numbers": self.line_numbers,
            "stream": self.stream,
            "seed": self.seed,
            "ctx_per_run": self.ctx_per_run,
            "api_key_set": self.api_key is not None,
            "experiment": EXPERIMENT,
        }
