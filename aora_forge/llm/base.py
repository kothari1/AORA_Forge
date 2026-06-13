"""Shared LLM primitives: the ``LLMClient`` interface, the deterministic
``MockLLMClient``, model tiers, pricing, and the structured-output schema helper.

Provider-specific clients (Anthropic, OpenAI, Vertex Gemini) all build on this so
the pipelines depend only on the interface, never on a provider.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel

from aora_forge.schemas import LLMUsage
from aora_forge.utils.logging import get_logger

log = get_logger("llm.base")

M = TypeVar("M", bound=BaseModel)


class ModelTier(str, Enum):
    """Which model a call should use — expressed as *intent*, resolved to a
    concrete model id per provider in ``llm.config``."""

    PLANNER = "planner"  # high-quality reasoning: clustering, spec generation, authoring
    WORKER = "worker"  # cheap inner loop: validation judging, small helpers


# Pricing (USD per 1M tokens) across providers. June-2026 estimates; the cost
# figures are advisory telemetry, not billing. Unknown models cost 0 (logged).
PRICING: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-4-8": (5.0, 25.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    # OpenAI (approximate)
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.4, 1.6),
    # Vertex Gemini (approximate)
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.30, 2.5),
    "gemini-2.0-flash-001": (0.15, 0.6),
}
_CACHE_WRITE_MULT = 1.25  # Anthropic cache creation ~1.25x input rate
_CACHE_READ_MULT = 0.10  # Anthropic / Gemini cache read ~0.1x input rate


def cost_usd(
    model: str, in_tok: int, out_tok: int, cache_create: int = 0, cache_read: int = 0
) -> float:
    in_rate, out_rate = PRICING.get(model, (0.0, 0.0))
    if model not in PRICING:
        log.debug("no pricing for model %s; cost reported as 0", model)
    return (
        in_tok * in_rate
        + out_tok * out_rate
        + cache_create * in_rate * _CACHE_WRITE_MULT
        + cache_read * in_rate * _CACHE_READ_MULT
    ) / 1_000_000.0


def pydantic_tool_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """JSON schema for a Pydantic model, shaped for a tool/function ``input_schema``.

    Pydantic v2 emits a valid object schema with ``$defs``/``$ref``; we pass it
    through largely as-is, since the validate-and-retry loop is the real
    correctness guarantee. ``$defs`` are inlined for providers (OpenAI) that don't
    accept them.
    """
    js = schema.model_json_schema()
    js.setdefault("type", "object")
    return _inline_defs(js)


def _inline_defs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve local ``$ref``/``$defs`` into an inlined schema (best-effort).

    Some providers' structured-output validators reject ``$ref``. We inline the
    definitions so the schema is self-contained. Falls back to the original on any
    unexpected shape.
    """
    defs = schema.get("$defs") or schema.get("definitions") or {}
    if not defs:
        return schema

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node and isinstance(node["$ref"], str):
                name = node["$ref"].split("/")[-1]
                target = defs.get(name)
                if target is not None:
                    merged = {k: v for k, v in node.items() if k != "$ref"}
                    return resolve({**target, **merged})
            return {k: resolve(v) for k, v in node.items() if k not in ("$defs", "definitions")}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    out = resolve(schema)
    return out if isinstance(out, dict) else schema


class LLMClient:
    """Interface for structured + text completion with usage telemetry.

    Every call carries an ``offline_fallback`` thunk that computes a valid result
    deterministically from the same inputs, so real clients can degrade gracefully
    and ``MockLLMClient`` *is* the fallback.
    """

    name: str = "base"

    def complete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[M],
        offline_fallback: Callable[[], M],
        model_tier: ModelTier = ModelTier.PLANNER,
        task: str = "structured",
        max_tokens: int = 4096,
        max_validation_retries: int = 3,
        cache_system: bool = True,
    ) -> tuple[M, LLMUsage]:
        raise NotImplementedError

    def complete_text(
        self,
        *,
        system: str,
        user: str,
        offline_fallback: Callable[[], str],
        model_tier: ModelTier = ModelTier.PLANNER,
        task: str = "text",
        max_tokens: int = 2048,
        cache_system: bool = True,
    ) -> tuple[str, LLMUsage]:
        raise NotImplementedError


class MockLLMClient(LLMClient):
    """Returns the caller's ``offline_fallback`` result, validated through the
    schema. Loud about being a mock (``usage.mocked == True``); deterministic."""

    name = "mock"

    def complete_structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[M],
        offline_fallback: Callable[[], M],
        model_tier: ModelTier = ModelTier.PLANNER,
        task: str = "structured",
        max_tokens: int = 4096,
        max_validation_retries: int = 3,
        cache_system: bool = True,
    ) -> tuple[M, LLMUsage]:
        result = offline_fallback()
        validated = schema.model_validate(result.model_dump())
        log.debug("MOCK structured [%s] -> %s", task, type(validated).__name__)
        return validated, LLMUsage(model="mock", mocked=True)

    def complete_text(
        self,
        *,
        system: str,
        user: str,
        offline_fallback: Callable[[], str],
        model_tier: ModelTier = ModelTier.PLANNER,
        task: str = "text",
        max_tokens: int = 2048,
        cache_system: bool = True,
    ) -> tuple[str, LLMUsage]:
        text = offline_fallback()
        log.debug("MOCK text [%s] -> %d chars", task, len(text))
        return text, LLMUsage(model="mock", mocked=True)
