"""Thin client for an OpenAI-compatible /v1/chat/completions endpoint.

Uses `requests` rather than the openai SDK on purpose: the llama.cpp server returns
fields the SDK would drop (`timings`) and fields we must log separately
(`reasoning_content`). The wire format is still plain OpenAI, so pointing `base_url`
at a vLLM or SGLang server is the only change needed to swap backends.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import requests


@dataclass
class Completion:
    content: str = ""
    reasoning_content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    finish_reason: str = ""
    timings: dict = field(default_factory=dict)
    wall_s: float = 0.0
    error: str | None = None


class LLMClient:
    def __init__(self, base_url: str, model: str, temperature: float = 0.0,
                 max_tokens: int = 16384, thinking: str = "off", timeout_s: int = 1800):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.timeout_s = timeout_s

    def health(self) -> dict:
        r = requests.get(f"{self.base_url}/health", timeout=15)
        r.raise_for_status()
        return r.json()

    def served_model(self) -> str:
        r = requests.get(f"{self.base_url}/models", timeout=30)
        r.raise_for_status()
        return r.json()["data"][0]["id"]

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             response_format: dict | None = None, max_tokens: int | None = None) -> Completion:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,       # server default is 1.0 - always send ours
            "max_tokens": max_tokens or self.max_tokens,
        }
        if tools:
            payload["tools"] = tools
        if response_format:
            payload["response_format"] = response_format
        if self.thinking != "auto":
            payload["chat_template_kwargs"] = {"enable_thinking": self.thinking == "on"}

        t0 = time.time()
        try:
            r = requests.post(f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout_s)
        except requests.RequestException as e:
            return Completion(wall_s=time.time() - t0, error=f"transport: {e}")
        wall = time.time() - t0

        if r.status_code != 200:
            return Completion(wall_s=wall, error=f"http {r.status_code}: {r.text[:400]}")

        try:
            body = r.json()
        except json.JSONDecodeError as e:
            return Completion(wall_s=wall, error=f"bad json from server: {e}")

        choice = body.get("choices", [{}])[0]
        msg = choice.get("message", {}) or {}
        usage = body.get("usage", {}) or {}
        return Completion(
            content=msg.get("content") or "",
            reasoning_content=msg.get("reasoning_content") or "",
            tool_calls=msg.get("tool_calls") or [],
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            finish_reason=choice.get("finish_reason", ""),
            timings=body.get("timings", {}) or {},
            wall_s=wall,
        )
