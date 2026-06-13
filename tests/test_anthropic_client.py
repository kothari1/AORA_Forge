"""De-risk the REAL Anthropic code path without a key.

We can't make live calls in CI, but we can verify the structured-output parsing,
the validate-and-retry loop, cost telemetry, and the offline fallback by injecting
a fake Anthropic client whose ``messages.create`` returns canned responses. This
catches bugs that would otherwise only surface the first time a real key is set.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from aora_forge.llm.client import AnthropicLLMClient, ModelTier


class _Demo(BaseModel):
    label: str
    score: float


class _Block:
    def __init__(self, type: str, **kw: Any) -> None:
        self.type = type
        for k, v in kw.items():
            setattr(self, k, v)


class _Usage:
    def __init__(self, in_tok: int, out_tok: int, cc: int = 0, cr: int = 0) -> None:
        self.input_tokens = in_tok
        self.output_tokens = out_tok
        self.cache_creation_input_tokens = cc
        self.cache_read_input_tokens = cr


class _Resp:
    def __init__(self, content: list[Any], usage: _Usage) -> None:
        self.content = content
        self.usage = usage


class _FakeMessages:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def create(self, **kwargs: Any) -> Any:
        resp = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        if isinstance(resp, Exception):
            raise resp
        return resp


class _FakeClient:
    def __init__(self, responses: list[Any]) -> None:
        self.messages = _FakeMessages(responses)


def _client_with(responses: list[Any]) -> AnthropicLLMClient:
    c = AnthropicLLMClient(api_key="test-key-not-used")  # no network at construction
    c._client = _FakeClient(responses)  # type: ignore[assignment]
    return c


def _fallback() -> _Demo:
    return _Demo(label="fallback", score=0.0)


def test_structured_happy_path() -> None:
    resp = _Resp(
        content=[_Block("tool_use", input={"label": "ok", "score": 0.9})],
        usage=_Usage(120, 40, cc=10, cr=5),
    )
    client = _client_with([resp])
    obj, usage = client.complete_structured(
        system="s", user="u", schema=_Demo, offline_fallback=_fallback, task="t"
    )
    assert obj.label == "ok" and obj.score == 0.9
    assert usage.mocked is False
    assert usage.input_tokens == 120 and usage.output_tokens == 40
    # opus pricing: 120*5 + 40*25 + 10*5*1.25 + 5*5*0.1 per 1e6
    assert usage.cost_usd > 0


def test_structured_retries_then_succeeds() -> None:
    bad = _Resp(content=[_Block("text", text="oops no tool")], usage=_Usage(10, 5))
    good = _Resp(
        content=[_Block("tool_use", input={"label": "fixed", "score": 0.5})], usage=_Usage(10, 5)
    )
    client = _client_with([bad, good])
    obj, usage = client.complete_structured(
        system="s", user="u", schema=_Demo, offline_fallback=_fallback, task="t"
    )
    assert obj.label == "fixed"
    assert client._client.messages.calls == 2  # type: ignore[attr-defined]


def test_structured_validation_retry_on_bad_input() -> None:
    # first tool input is missing a required field -> ValidationError -> retry
    bad = _Resp(content=[_Block("tool_use", input={"label": "x"})], usage=_Usage(10, 5))
    good = _Resp(
        content=[_Block("tool_use", input={"label": "x", "score": 0.7})], usage=_Usage(10, 5)
    )
    client = _client_with([bad, good])
    obj, _ = client.complete_structured(
        system="s", user="u", schema=_Demo, offline_fallback=_fallback, task="t"
    )
    assert obj.score == 0.7


def test_structured_falls_back_after_exhausted_retries() -> None:
    bad = _Resp(content=[_Block("text", text="never a tool")], usage=_Usage(1, 1))
    client = _client_with([bad, bad, bad])
    obj, _ = client.complete_structured(
        system="s",
        user="u",
        schema=_Demo,
        offline_fallback=_fallback,
        task="t",
        max_validation_retries=3,
    )
    assert obj.label == "fallback"  # offline fallback used


def test_text_completion_path() -> None:
    resp = _Resp(content=[_Block("text", text="  hello world  ")], usage=_Usage(5, 3))
    client = _client_with([resp])
    text, usage = client.complete_text(
        system="s",
        user="u",
        offline_fallback=lambda: "fb",
        model_tier=ModelTier.WORKER,
        task="t",
    )
    assert text == "hello world"
    assert usage.mocked is False
