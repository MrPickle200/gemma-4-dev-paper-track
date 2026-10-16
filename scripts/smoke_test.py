"""BƯỚC 1 acceptance test for the llama.cpp backend.

Verifies, in order:
  1. the OpenAI-compatible endpoint is alive and reports the served model
  2. plain generation with temperature=0.0
  3. thinking mode is controllable and is reported in `reasoning_content`
     (it defaults to OFF in the Gemma 4 template, but activates when tools are
     present or the first message has role system/developer)
  4. JSON-schema constrained decoding - the mechanism Module 2 AST-Edit relies on
  5. native function calling - the mechanism the agent loop relies on

Run with the server up:  .\.venv\Scripts\python.exe scripts\smoke_test.py
"""

from __future__ import annotations

import json
import sys
import time

import requests

# Windows consoles default to cp1252 and cannot print the Vietnamese step names.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

BASE = "http://127.0.0.1:8080"

EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "file": {"type": "string"},
        "target_symbol": {"type": "string"},
        "search_block": {"type": "string"},
        "replace_block": {"type": "string"},
    },
    "required": ["file", "target_symbol", "search_block", "replace_block"],
    "additionalProperties": False,
}

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "get_repo_map",
        "description": "Return the symbol skeleton (classes, functions, signatures) of a file or directory.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "File or directory path"}},
            "required": ["path"],
        },
    },
}


def wait_for_server(timeout_s: int = 300) -> None:
    print(f"[0/5] waiting for {BASE}/health ...")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            if requests.get(f"{BASE}/health", timeout=5).status_code == 200:
                print(f"      ready after {time.time() - t0:.1f}s")
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise SystemExit(f"server did not become healthy within {timeout_s}s")


def chat(model: str, messages: list, max_tokens: int = 1024, **extra) -> dict:
    payload = {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": max_tokens}
    payload.update(extra)
    t0 = time.time()
    r = requests.post(f"{BASE}/v1/chat/completions", json=payload, timeout=900)
    r.raise_for_status()
    out = r.json()
    out["_wall_s"] = time.time() - t0
    return out


def summarize(tag: str, out: dict) -> None:
    msg = out["choices"][0]["message"]
    tps = out.get("timings", {}).get("predicted_per_second")
    print(f"      {tag}: {out['_wall_s']:.1f}s wall, "
          f"{out['usage']['completion_tokens']} completion tokens"
          + (f", {tps:.1f} tok/s" if tps else ""))
    print(f"      content={ (msg.get('content') or '')[:120]!r}")
    if msg.get("reasoning_content"):
        print(f"      reasoning_content present ({len(msg['reasoning_content'])} chars)")


def main() -> int:
    failures = []
    wait_for_server()

    print("[1/5] /v1/models")
    model = requests.get(f"{BASE}/v1/models", timeout=30).json()["data"][0]["id"]
    print(f"      serving: {model}")

    print("[2/5] plain completion, temperature=0.0")
    out = chat(model, [{"role": "user", "content": "Reply with exactly: HARNESS OK"}], max_tokens=256)
    summarize("plain", out)
    if "HARNESS OK" not in (out["choices"][0]["message"].get("content") or ""):
        failures.append("plain completion did not return the expected string")

    print("[3/5] thinking control")
    on = chat(model, [{"role": "user", "content": "What is 17 * 23? Answer with the number."}],
              max_tokens=1024, chat_template_kwargs={"enable_thinking": True})
    off = chat(model, [{"role": "user", "content": "What is 17 * 23? Answer with the number."}],
               max_tokens=1024, chat_template_kwargs={"enable_thinking": False})
    r_on = len(on["choices"][0]["message"].get("reasoning_content") or "")
    r_off = len(off["choices"][0]["message"].get("reasoning_content") or "")
    t_on, t_off = on["usage"]["completion_tokens"], off["usage"]["completion_tokens"]
    print(f"      thinking=True : {t_on} tokens, {r_on} reasoning chars")
    print(f"      thinking=False: {t_off} tokens, {r_off} reasoning chars")
    if not (t_on > t_off):
        failures.append("enable_thinking did not change token usage")

    print("[4/5] constrained JSON decoding (Module 2 schema)")
    out = chat(model, [
        {"role": "user", "content":
            "In file calc.py the function add() currently returns 'a - b'. "
            "Produce a search/replace edit that makes it return 'a + b'."},
    ], max_tokens=1024,
        response_format={"type": "json_schema", "json_schema": {"name": "edit", "schema": EDIT_SCHEMA}},
        chat_template_kwargs={"enable_thinking": False})
    raw = out["choices"][0]["message"].get("content") or ""
    try:
        obj = json.loads(raw)
        missing = set(EDIT_SCHEMA["required"]) - obj.keys()
        print(f"      schema keys: {sorted(obj)}")
        if missing:
            failures.append(f"schema output missing keys: {missing}")
    except json.JSONDecodeError as e:
        print(f"      raw: {raw[:300]}")
        failures.append(f"constrained output was not valid JSON: {e}")

    print("[5/5] native function calling")
    out = chat(model, [
        {"role": "user", "content": "Show me the symbols in harness/agent.py. Use your tool."},
    ], max_tokens=1024, tools=[BASH_TOOL], chat_template_kwargs={"enable_thinking": False})
    calls = out["choices"][0]["message"].get("tool_calls")
    if calls:
        fn = calls[0]["function"]
        print(f"      tool_call: {fn['name']}({fn['arguments']})")
    else:
        failures.append("model did not emit a tool_call")

    print()
    if failures:
        for f in failures:
            print(f"FAILED: {f}")
        return 1
    print("BƯỚC 1 ACCEPTANCE: PASS (endpoint, thinking control, constrained JSON, tool calls)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
