"""Run configuration and log schema constants."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
RUNS = LOGS / "runs"
PATCHES = LOGS / "patches"
WORKTREES = ROOT / "worktrees"
REPOS = ROOT / "repos"

VARIANTS = ("B0", "M1", "M2", "M3")

# A run that ended because the *backend* failed (transport/HTTP/JSON) or the harness raised is not an
# experimental outcome: such cells must be re-run rather than skipped as "done" or averaged in.
INFRA_STOP_PREFIXES = ("llm_error", "harness_error")


def is_infra_failure(stop_reason: str | None) -> bool:
    return str(stop_reason or "").startswith(INFRA_STOP_PREFIXES)


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
    seed: int = 0
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
            "seed": self.seed,
            "api_key_set": self.api_key is not None,
        }
