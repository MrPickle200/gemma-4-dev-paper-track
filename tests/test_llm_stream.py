"""Unit tests for the SSE path in harness/llm.py.

Streaming is what keeps a grid alive behind a proxy with a read timeout (Cloudflare 524 at
125 s), so the folding has to be exactly right: tool-call arguments arrive in fragments and
must be concatenated per index, and a stream cut in flight must be reported as an error
rather than as a short answer. No server needed. Run with:

  .\\.venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.llm import LLMClient, accumulate_stream      # noqa: E402


class StubStream:
    """The minimum surface accumulate_stream needs, plus a way to fail mid-stream."""

    def __init__(self, lines, exc=None):
        self._lines = lines
        self._exc = exc

    def iter_lines(self, decode_unicode=False):
        for line in self._lines:
            yield line
        if self._exc:
            raise self._exc


def sse(payload: str) -> str:
    return f"data: {payload}"


def test_accumulates_content_reasoning_and_fragmented_tool_calls():
    """The real shape llama-server sends: reasoning, then a tool call split across deltas."""
    lines = [
        ": keep-alive",
        "",
        sse('{"choices":[{"index":0,"finish_reason":null,"delta":{"role":"assistant","content":null}}]}'),
        sse('{"choices":[{"index":0,"finish_reason":null,"delta":{"reasoning_content":"Let me "}}]}'),
        sse('{"choices":[{"index":0,"finish_reason":null,"delta":{"reasoning_content":"read it."}}]}'),
        sse('{"choices":[{"index":0,"finish_reason":null,"delta":{"tool_calls":[{"index":0,'
            '"id":"call_a","type":"function","function":{"name":"read_file","arguments":"{\\"pa"}}]}}]}'),
        sse('{"choices":[{"index":0,"finish_reason":null,"delta":{"tool_calls":[{"index":0,'
            '"function":{"arguments":"th\\": \\"setup"}}]}}]}'),
        sse('{"choices":[{"index":0,"finish_reason":null,"delta":{"tool_calls":[{"index":0,'
            '"function":{"arguments":".py\\"}"}}]}}]}'),
        sse('{"choices":[{"index":0,"finish_reason":"tool_calls","delta":{}}]}'),
        sse('{"choices":[],"usage":{"prompt_tokens":93,"completion_tokens":129,'
            '"prompt_tokens_details":{"cached_tokens":88}},'
            '"timings":{"prompt_ms":486.0,"predicted_ms":1859.0}}'),
        "data: [DONE]",
    ]
    comp = accumulate_stream(StubStream(lines))

    assert comp.error is None
    assert comp.reasoning_content == "Let me read it."
    assert comp.content == ""
    assert comp.finish_reason == "tool_calls"
    assert comp.prompt_tokens == 93 and comp.completion_tokens == 129 and comp.cached_tokens == 88
    assert comp.timings == {"prompt_ms": 486.0, "predicted_ms": 1859.0}
    assert comp.tool_calls == [{
        "id": "call_a", "type": "function",
        "function": {"name": "read_file", "arguments": '{"path": "setup.py"}'},
    }]


def test_multiple_tool_calls_are_ordered_by_index():
    """Parallel calls interleave in the stream; they must not be reordered or merged."""
    lines = [
        sse('{"choices":[{"index":0,"finish_reason":null,"delta":{"tool_calls":[{"index":1,'
            '"id":"b","type":"function","function":{"name":"search_text","arguments":"{\\"p"}}]}}]}'),
        sse('{"choices":[{"index":0,"finish_reason":null,"delta":{"tool_calls":[{"index":0,'
            '"id":"a","type":"function","function":{"name":"read_file","arguments":"{\\"x"}}]}}]}'),
        sse('{"choices":[{"index":0,"finish_reason":null,"delta":{"tool_calls":[{"index":1,'
            '"function":{"arguments":"at\\": \\"y\\"}"}},{"index":0,'
            '"function":{"arguments":"\\": 1}"}}]}}]}'),
        sse('{"choices":[{"index":0,"finish_reason":"tool_calls","delta":{}}]}'),
        "data: [DONE]",
    ]
    comp = accumulate_stream(StubStream(lines))

    assert [c["id"] for c in comp.tool_calls] == ["a", "b"]
    assert comp.tool_calls[0]["function"]["arguments"] == '{"x": 1}'
    assert comp.tool_calls[1]["function"]["arguments"] == '{"pat": "y"}'


def test_cut_stream_is_an_error_not_a_short_answer():
    """No finish_reason and no [DONE] means the proxy dropped us mid-turn."""
    lines = [
        sse('{"choices":[{"index":0,"finish_reason":null,"delta":{"content":"I was saying"}}]}'),
    ]
    comp = accumulate_stream(StubStream(lines))
    assert comp.error is not None and comp.error.startswith("truncated stream")
    assert comp.content == ""          # nothing is offered up as a result


def test_transport_failure_mid_stream_is_an_error():
    lines = [sse('{"choices":[{"index":0,"finish_reason":null,"delta":{"content":"partial"}}]}')]
    comp = accumulate_stream(StubStream(lines, exc=requests.exceptions.ChunkedEncodingError("gone")))
    assert comp.error is not None and comp.error.startswith("transport (stream)")


def test_error_event_in_stream_is_an_error():
    comp = accumulate_stream(StubStream([sse('{"error":{"message":"context exceeded"}}')]))
    assert comp.error is not None and comp.error.startswith("sse error")


def test_finish_reason_without_done_sentinel_still_counts_as_complete():
    """[DONE] is what llama-server sends, but a finish_reason is equally conclusive."""
    lines = [
        sse('{"choices":[{"index":0,"finish_reason":null,"delta":{"content":"done"}}]}'),
        sse('{"choices":[{"index":0,"finish_reason":"stop","delta":{}}]}'),
    ]
    comp = accumulate_stream(StubStream(lines))
    assert comp.error is None and comp.content == "done" and comp.finish_reason == "stop"


class _FakeResponse:
    def __init__(self, body=None, lines=None, status=200):
        self.status_code = status
        self._body = body
        self._lines = lines or []

    def iter_lines(self, decode_unicode=False):
        yield from self._lines

    def json(self):
        return self._body

    @property
    def text(self):
        return ""

    def close(self):
        pass


def test_chat_requests_streaming_and_asks_for_usage(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None, headers=None, stream=None):
        captured.update(url=url, json=json, stream=stream)
        return _FakeResponse(lines=[
            sse('{"choices":[{"index":0,"finish_reason":null,"delta":{"content":"hi"}}]}'),
            sse('{"choices":[{"index":0,"finish_reason":"stop","delta":{}}]}'),
            "data: [DONE]",
        ])

    monkeypatch.setattr("harness.llm.requests.post", fake_post)
    client = LLMClient(base_url="http://127.0.0.1:8080", model="m", thinking="on")
    comp = client.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])

    assert captured["stream"] is True
    assert captured["json"]["stream"] is True
    assert captured["json"]["stream_options"] == {"include_usage": True}
    assert captured["url"].endswith("/chat/completions")
    assert comp.content == "hi" and comp.finish_reason == "stop"


def test_chat_no_stream_restores_the_buffered_request(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None, headers=None, stream=None):
        captured.update(json=json, stream=stream)
        return _FakeResponse(body={
            "choices": [{"message": {"content": "hi", "reasoning_content": "r"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        })

    monkeypatch.setattr("harness.llm.requests.post", fake_post)
    client = LLMClient(base_url="http://127.0.0.1:8080", model="m", thinking="on", stream=False)
    comp = client.chat([{"role": "user", "content": "hi"}])

    assert "stream" not in captured["json"]
    assert captured["json"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert comp.content == "hi" and comp.reasoning_content == "r"
