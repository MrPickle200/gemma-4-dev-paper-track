"""Pre-flight check of an OpenAI-compatible llama-server endpoint (local or tunnelled).

Run this after starting a backend (scripts/serve.ps1, or the Colab notebook) and before
run_batch.py, so a dead tunnel or a mis-sized context is caught in seconds instead of after a
batch of failures.

  .\\.venv\\Scripts\\python.exe scripts\\check_endpoint.py
  .\\.venv\\Scripts\\python.exe scripts\\check_endpoint.py --base-url https://x.trycloudflare.com --api-key <key>

Exit code is 0 only if health, auth, and a tool call all work.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests                                          # noqa: E402

from harness.llm import LLMClient                         # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--model", default="gemma-4-E2B")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    hdr = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    ok = True

    print(f"endpoint {base}")
    try:
        r = requests.get(f"{base}/health", timeout=30)
        print(f"  /health        {r.status_code} {r.text[:80]}")
        ok &= r.status_code == 200
    except requests.RequestException as e:
        print(f"  /health        FAILED {type(e).__name__}: {e}")
        return 1

    try:
        props = requests.get(f"{base}/props", headers=hdr, timeout=30).json()
        slots = props.get("total_slots")
        n_ctx = (props.get("default_generation_settings") or {}).get("n_ctx")
        print(f"  model_path     {props.get('model_path')}")
        print(f"  build_info     {props.get('build_info')}")
        print(f"  total_slots    {slots}")
        print(f"  n_ctx / slot   {n_ctx}")
        if n_ctx and n_ctx < 32768:
            print(f"  WARNING: per-slot ctx {n_ctx} < 32768 (laptop parity); long prompts will truncate")
    except (requests.RequestException, ValueError) as e:
        print(f"  /props         skipped ({type(e).__name__})")

    with_key = requests.get(f"{base}/models", headers=hdr, timeout=30).status_code
    no_key = requests.get(f"{base}/models", timeout=30).status_code
    print(f"  /models        with key {with_key}, without {no_key}"
          + ("  (auth enforced)" if no_key == 401 else "  (WARNING: no auth)"))
    ok &= with_key == 200

    client = LLMClient(base, args.model, thinking="on", timeout_s=900, api_key=args.api_key)
    tools = [{"type": "function", "function": {
        "name": "add", "description": "Add two integers",
        "parameters": {"type": "object", "properties": {
            "a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}}]
    t0 = time.time()
    c = client.chat([{"role": "user", "content": "Use the add tool to compute 2 + 3."}], tools=tools)
    wall = time.time() - t0
    calls = c.tool_calls or []
    name = (calls[0].get("function", {}).get("name") if calls else None)
    print(f"  tool call      {wall:.1f}s finish={c.finish_reason} name={name} "
          f"reasoning_chars={len(c.reasoning_content)} error={c.error}")
    ok &= bool(calls) and name == "add" and not c.error

    print("RESULT:", "OK" if ok else "PROBLEM")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
