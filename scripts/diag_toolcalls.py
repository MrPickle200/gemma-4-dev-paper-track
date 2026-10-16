"""Diagnose Gemma 4 native tool-calling through llama.cpp.

Question: when the model emits a tool call, does llama.cpp convert Gemma 4's native
argument syntax into valid JSON arguments? We probe with a trivially simple call first
(one short string argument); if that is already malformed, the server-side conversion is
broken and would contaminate every ablation result.

Usage:  .\\.venv\\Scripts\\python.exe scripts\\diag_toolcalls.py
"""

from __future__ import annotations

import json
import sys

import requests

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

BASE = "http://127.0.0.1:8080"

TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_text",
        "description": "Search the repository for a regular expression.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Python regular expression"},
                "path": {"type": "string", "description": "Directory to search"},
            },
            "required": ["pattern"],
        },
    },
}]

PROBES = [
    ("single short string arg", "Search the repo for the literal text ZZZNOTFOUNDZZZ. Use the tool now."),
    ("two short string args", "Search the repo for ZZZNOTFOUNDZZZ but only inside the tests directory."),
    ("arg with punctuation", "Search the repo for the exact regex 'method = method.upper()'. Use the tool now."),
]


def raw_call(user: str) -> dict:
    r = requests.post(f"{BASE}/v1/chat/completions", json={
        "messages": [{"role": "user", "content": user}],
        "temperature": 0.0,
        "max_tokens": 512,
        "tools": TOOLS,
        "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=600)
    r.raise_for_status()
    return r.json()


def main() -> int:
    bad = 0
    for label, user in PROBES:
        body = raw_call(user)
        msg = body["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        print(f"--- {label}")
        if not calls:
            print(f"    NO TOOL CALL; content={ (msg.get('content') or '')[:200]!r}")
            bad += 1
            continue
        for c in calls:
            args_raw = c["function"]["arguments"]
            try:
                parsed = json.loads(args_raw)
                # does the parsed value look sane, or has text leaked into a key?
                sanity = all(isinstance(k, str) and "\n" not in k for k in parsed)
                print(f"    JSON OK   : {parsed}   (keys sane: {sanity})")
                if not sanity:
                    bad += 1
                    print("    ^^ text leaked into a key name - conversion is lossy")
            except json.JSONDecodeError as e:
                bad += 1
                print(f"    JSON BROKEN: {args_raw!r}  ({e})")
    print(f"\n{len(PROBES) - bad}/{len(PROBES)} probes produced clean JSON arguments")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
