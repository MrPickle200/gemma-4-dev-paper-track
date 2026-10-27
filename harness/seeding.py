"""M4 zero-turn AST-semantic seeding.

Uses an embedding model (Ollama: embeddinggemma, 270m text+code) to rank
repository AST symbols against a task's problem statement at Turn 0, then
injects the top-K candidates into the initial user message. No extra
tools are exposed to the LLM.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def extract_identifiers(text: str) -> List[str]:
    out = set()
    for m in re.finditer(r"`([^`]+)`", text):
        out.add(m.group(1))
    for m in re.finditer(r"\b([A-Z][a-zA-Z0-9_]*|[a-z]+_[a-z0-9_]+)\b", text):
        out.add(m.group(1))
    return sorted(out)


def ollama_embed(model: str = "embeddinggemma-2:270m", texts: List[str] = None,
                 base_url: str = "http://localhost:11434", batch_size: int = 256) -> np.ndarray:
    """L2-normalised embeddings via Ollama's batched /api/embed endpoint."""
    import urllib.request

    import urllib.error

    texts = list(texts or [])

    def embed_chunk(chunk):
        data = json.dumps({"model": model, "input": chunk}).encode()
        req = urllib.request.Request(f"{base_url}/api/embed", data=data,
                                    headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.load(r)["embeddings"]
        except urllib.error.HTTPError as e:
            if len(chunk) == 1:
                print(f"embed failed for one text ({e.code}); using zero vector")
                # infer dimension from a trivial call
                return [[0.0] * 768]
            mid = len(chunk) // 2
            return embed_chunk(chunk[:mid]) + embed_chunk(chunk[mid:])

    vecs: list = []
    for start in range(0, len(texts), batch_size):
        vecs.extend(embed_chunk(texts[start:start + batch_size]))
    arr = np.asarray(vecs, dtype=np.float32)
    n = np.linalg.norm(arr, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return arr / n


def load_cache(path: Path) -> Optional[Dict[str, Any]]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def save_cache(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def format_block(candidates: List[Dict[str, Any]]) -> str:
    lines = [
        "=== AUTO-RETRIEVED CANDIDATE SYMBOLS (Turn-0 AST-Semantic Seeding) ===",
        "The following existing symbols in the repository were automatically ranked as most relevant to the issue description. Use exact IDs from this list when calling `read_symbol` or `lock_targets`:",
        "",
    ]
    for i, c in enumerate(candidates, 1):
        lines.append(f"{i}. `{c['symbol_id']}`")
        lines.append(f"   Signature: {c.get('signature', '')}")
        if c.get("summary"):
            lines.append(f"   Summary: {c['summary']}")
        lines.append("")
    lines.append("======================================================================")
    return "\n".join(lines)
