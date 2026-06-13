"""De-risk the OpenAI and Vertex code paths without keys/credentials.

Like ``test_anthropic_client.py`` but for the other two providers: bypass the
credential-requiring constructors with ``__new__`` and inject a fake underlying
client, then verify structured-output parsing, validate-and-retry, the offline
fallback, cost telemetry, and text completion. CI-safe (no SDKs/creds required).
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from aora_forge.llm.base import ModelTier
from aora_forge.llm.config import DEFAULT_MODELS, Provider
from aora_forge.llm.openai_client import OpenAILLMClient
from aora_forge.llm.vertex_client import VertexGeminiLLMClient


class _Demo(BaseModel):
    label: str
    score: float


def _fallback() -> _Demo:
    return _Demo(label="fallback", score=0.0)


# --------------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------------- #


class _Fn:
    def __init__(self, args: str) -> None:
        self.arguments = args


class _TC:
    def __init__(self, args: str) -> None:
        self.function = _Fn(args)


class _OAMsg:
    def __init__(self, tool_calls: Any = None, content: Any = None) -> None:
        self.tool_calls = tool_calls
        self.content = content


class _OAChoice:
    def __init__(self, msg: _OAMsg) -> None:
        self.message = msg


class _OAUsage:
    def __init__(self, p: int, c: int) -> None:
        self.prompt_tokens = p
        self.completion_tokens = c
        self.prompt_tokens_details = None


class _OAResp:
    def __init__(self, choice: _OAChoice, usage: _OAUsage) -> None:
        self.choices = [choice]
        self.usage = usage


class _OACompletions:
    def __init__(self, responses: list[Any]) -> None:
        self._r = list(responses)
        self.calls = 0

    def create(self, **kw: Any) -> Any:
        r = self._r[min(self.calls, len(self._r) - 1)]
        self.calls += 1
        return r


class _FakeOpenAI:
    def __init__(self, responses: list[Any]) -> None:
        self.chat = type("C", (), {"completions": _OACompletions(responses)})()


def _openai_with(responses: list[Any]) -> OpenAILLMClient:
    c = OpenAILLMClient.__new__(OpenAILLMClient)
    c._client = _FakeOpenAI(responses)  # type: ignore[attr-defined]
    c._models = DEFAULT_MODELS[Provider.OPENAI]  # type: ignore[attr-defined]
    return c


def _oa_tool_resp(args: dict, p: int = 50, c: int = 20) -> _OAResp:
    return _OAResp(_OAChoice(_OAMsg(tool_calls=[_TC(json.dumps(args))])), _OAUsage(p, c))


def test_openai_structured_happy_path() -> None:
    client = _openai_with([_oa_tool_resp({"label": "ok", "score": 0.9})])
    obj, usage = client.complete_structured(
        system="s", user="u", schema=_Demo, offline_fallback=_fallback, task="t"
    )
    assert obj.label == "ok" and obj.score == 0.9
    assert usage.mocked is False and usage.input_tokens == 50 and usage.cost_usd > 0


def test_openai_validation_retry() -> None:
    client = _openai_with(
        [_oa_tool_resp({"label": "x"}), _oa_tool_resp({"label": "x", "score": 0.7})]
    )
    obj, _ = client.complete_structured(
        system="s", user="u", schema=_Demo, offline_fallback=_fallback, task="t"
    )
    assert obj.score == 0.7


def test_openai_text_path() -> None:
    resp = _OAResp(_OAChoice(_OAMsg(content="  hi there  ")), _OAUsage(3, 2))
    client = _openai_with([resp])
    text, usage = client.complete_text(
        system="s", user="u", offline_fallback=lambda: "fb", model_tier=ModelTier.WORKER, task="t"
    )
    assert text == "hi there" and usage.mocked is False


# --------------------------------------------------------------------------- #
# Vertex Gemini
# --------------------------------------------------------------------------- #


class _UM:
    def __init__(self, p: int, c: int, cached: int = 0) -> None:
        self.prompt_token_count = p
        self.candidates_token_count = c
        self.cached_content_token_count = cached


class _GResp:
    def __init__(self, parsed: Any = None, text: str = "", usage: _UM | None = None) -> None:
        self.parsed = parsed
        self.text = text
        self.usage_metadata = usage or _UM(0, 0)


class _GModels:
    def __init__(self, responses: list[Any]) -> None:
        self._r = list(responses)
        self.calls = 0

    def generate_content(self, **kw: Any) -> Any:
        r = self._r[min(self.calls, len(self._r) - 1)]
        self.calls += 1
        return r


class _FakeGenai:
    def __init__(self, responses: list[Any]) -> None:
        self.models = _GModels(responses)


def _vertex_with(responses: list[Any]) -> VertexGeminiLLMClient:
    c = VertexGeminiLLMClient.__new__(VertexGeminiLLMClient)
    c._client = _FakeGenai(responses)  # type: ignore[attr-defined]
    c._models = DEFAULT_MODELS[Provider.VERTEX]  # type: ignore[attr-defined]
    return c


def test_vertex_uses_sdk_parsed_instance() -> None:
    # Gemini's response_schema gives a validated instance on resp.parsed
    client = _vertex_with(
        [_GResp(parsed=_Demo(label="ok", score=0.8), usage=_UM(40, 12, cached=4))]
    )
    obj, usage = client.complete_structured(
        system="s", user="u", schema=_Demo, offline_fallback=_fallback, task="t"
    )
    assert obj.label == "ok" and obj.score == 0.8
    assert usage.input_tokens == 40 and usage.cache_read_input_tokens == 4 and usage.cost_usd > 0


def test_vertex_falls_back_to_raw_text() -> None:
    # parsed is None -> parse resp.text
    client = _vertex_with(
        [_GResp(parsed=None, text='{"label": "fromtext", "score": 0.3}', usage=_UM(10, 5))]
    )
    obj, _ = client.complete_structured(
        system="s", user="u", schema=_Demo, offline_fallback=_fallback, task="t"
    )
    assert obj.label == "fromtext"


def test_vertex_retries_then_fallback() -> None:
    bad = _GResp(parsed=None, text="not json", usage=_UM(1, 1))
    client = _vertex_with([bad, bad, bad])
    obj, _ = client.complete_structured(
        system="s",
        user="u",
        schema=_Demo,
        offline_fallback=_fallback,
        task="t",
        max_validation_retries=3,
    )
    assert obj.label == "fallback"


def test_vertex_text_path() -> None:
    client = _vertex_with([_GResp(text="  pong  ", usage=_UM(2, 1))])
    text, usage = client.complete_text(
        system="s", user="u", offline_fallback=lambda: "fb", task="t"
    )
    assert text == "pong" and usage.mocked is False
