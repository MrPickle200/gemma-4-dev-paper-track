"""Thin client for an OpenAI-compatible /v1/chat/completions endpoint.

Uses `requests` rather than the openai SDK on purpose: the llama.cpp server returns
fields the SDK would drop (`timings`) and fields we must log separately
(`reasoning_content`). The wire format is still plain OpenAI, so pointing `base_url`
at a vLLM or SGLang server is the only change needed to swap backends.

Requests are **streamed** (`stream=True`, `stream_options.include_usage`). This is not a
UI nicety: a non-streaming response carries no byte at all until the turn has finished
generating, so any turn longer than a proxy's read timeout is killed in flight. Behind a
Cloudflare quick tunnel that is 125 s (error 524), which E2B's `thinking on` turns exceed
often enough to lose a third of a grid. Streamed, the response headers and the first
delta arrive in ~0.2 s and the connection keeps dribbling tokens, so the read timer never
expires. The folded result is byte-identical to the non-streaming one (see
tests/test_llm_stream.py); `--no-stream` restores the old behaviour.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field

import requests

from .config import MIN_RUN_CTX


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


def accumulate_stream(resp) -> Completion:
    """Fold a llama-server SSE stream into a Completion.

    Takes anything with `iter_lines(decode_unicode=True)`, so a stub response can drive it
    in tests. llama-server emits OpenAI-style deltas: `content` and `reasoning_content`
    arrive as text fragments, tool calls arrive as `tool_calls[i].function.arguments`
    fragments that must be concatenated per `index` (a call is *not* re-sent whole), and
    `usage`/`timings` ride on the final chunk.

    A stream that ends without a finish_reason or the [DONE] sentinel is *truncated* - the
    proxy dropped it, or the server died mid-generation - and is returned as an error, never
    as a short answer. Otherwise half a turn could be mistaken for a model outcome.
    """
    content: list[str] = []
    reasoning: list[str] = []
    calls: dict[int, dict] = {}
    usage: dict = {}
    timings: dict = {}
    finish = ""
    saw_done = False
    try:
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue                      # keep-alive comment or event separator
            data = raw[5:].strip()
            if data == "[DONE]":
                saw_done = True
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue                      # ignore anything that is not a JSON event
            if chunk.get("error"):
                return Completion(error=f"sse error: {str(chunk['error'])[:400]}")
            if chunk.get("usage"):
                usage = chunk["usage"]
            if chunk.get("timings"):
                timings = chunk["timings"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    content.append(delta["content"])
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                for tc in delta.get("tool_calls") or []:
                    slot = calls.setdefault(tc.get("index", 0),
                                            {"id": None, "type": None, "name": None, "args": []})
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    if tc.get("type"):
                        slot["type"] = tc["type"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["args"].append(fn["arguments"])
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    except requests.RequestException as e:
        return Completion(error=f"transport (stream): {e}")

    if not saw_done and not finish:
        return Completion(error="truncated stream: no finish_reason and no [DONE]")

    return Completion(
        content="".join(content),
        reasoning_content="".join(reasoning),
        tool_calls=[{"id": c["id"], "type": c["type"] or "function",
                     "function": {"name": c["name"] or "", "arguments": "".join(c["args"])}}
                    for _, c in sorted(calls.items())],
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        cached_tokens=(usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
        finish_reason=finish,
        timings=timings,
    )


def verify_run_context(base_url: str, workers: int, api_key: str | None = None,
                       ctx: int = MIN_RUN_CTX, timeout: float = 900.0) -> int | None:
    """Ask the server to serve `workers` concurrent runs of a full `ctx`-token prompt.

    Reading `/props` is not enough to know what a run really gets: llama-server advertises
    `n_ctx = 32768` both when it keeps one *shared* KV pool (which then serves each of four runs
    only ~8k) and when it holds four caches of its own - measured on b11696, both configurations
    exist and the shared one silently ended long runs mid-decode. So measure the capability we
    actually depend on, concurrently (serially would not expose sharing). Returns `ctx` if every
    one of the `workers` requests was served, else None: the batch must not be trusted.
    """
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    # ~14 tokens per repetition for this filler; stay just under `ctx` so the prompt-length
    # check cannot reject us for overshooting and hide a genuine capacity failure.
    filler = "The quick brown fox jumps over the lazy dog while reviewing a patch. "
    content = filler * max(1, int(ctx * 0.95 / 14.0))
    out: dict[int, tuple[bool, int, str]] = {}

    def one(i: int) -> None:
        try:
            r = requests.post(f"{base_url.rstrip('/')}/chat/completions", timeout=timeout,
                              headers=headers,
                              json={"model": "probe", "messages": [{"role": "user", "content": content}],
                                    "max_tokens": 1, "temperature": 0.0, "stream": False})
            if r.status_code == 200:
                pt = int(r.json().get("usage", {}).get("prompt_tokens", 0))
                out[i] = (pt >= ctx * 0.9, pt, "")
            else:
                out[i] = (False, 0, r.text[:120])
        except requests.RequestException as e:
            out[i] = (False, 0, f"{type(e).__name__}: {e}"[:120])

    threads = [threading.Thread(target=one, args=(i,)) for i in range(max(1, workers))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ok = all(v[0] for v in out.values())
    served = sum(v[1] for v in out.values())
    detail = "" if ok else " first failure: " + next((v[2] for v in out.values() if not v[0]), "")
    print(f"  context probe: {workers} concurrent request(s), {served} tokens served together "
          f"-> {'PASS' if ok else 'FAIL'}{detail}")
    return ctx if ok else None


class LLMClient:
    def __init__(self, base_url: str, model: str, temperature: float = 0.0,
                 max_tokens: int = 16384, thinking: str = "off", timeout_s: int = 1800,
                 api_key: str | None = None, stream: bool = True):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.timeout_s = timeout_s
        self.api_key = api_key
        self.stream = stream

    @property
    def _headers(self) -> dict:
        """llama-server `--api-key` checks a bearer token; empty when talking to plain localhost."""
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def health(self) -> dict:
        r = requests.get(f"{self.base_url}/health", timeout=15, headers=self._headers)
        r.raise_for_status()
        return r.json()

    def ctx_pool(self) -> int | None:
        """The context the server advertises - the *shared* budget, not a per-slot one.

        llama-server keeps one unified KV pool that every slot draws from (`kv_unified = true` in
        its banner), so this single number is divided among all concurrent runs. None when the
        server does not report it; see `config.ctx_is_adequate`.
        """
        try:
            r = requests.get(f"{self.base_url}/props", timeout=15, headers=self._headers)
            r.raise_for_status()
            n = r.json().get("default_generation_settings", {}).get("n_ctx")
            return int(n) if n else None
        except (requests.RequestException, ValueError, TypeError):
            return None

    def served_model(self) -> str:
        r = requests.get(f"{self.base_url}/models", timeout=30, headers=self._headers)
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
        if self.stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}   # usage rides the last chunk

        t0 = time.time()
        try:
            r = requests.post(f"{self.base_url}/chat/completions", json=payload,
                              timeout=self.timeout_s, headers=self._headers, stream=self.stream)
        except requests.RequestException as e:
            return Completion(wall_s=time.time() - t0, error=f"transport: {e}")
        wall = time.time() - t0

        if r.status_code != 200:
            return Completion(wall_s=wall, error=f"http {r.status_code}: {r.text[:400]}")

        if self.stream:
            try:
                comp = accumulate_stream(r)
            finally:
                r.close()
            comp.wall_s = time.time() - t0
            return comp

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
