"""Anthropic SDK wrapper with structured output, prompt caching, retry, and
per-call cost telemetry — plus a deterministic offline fallback.

Why two clients
---------------
The pipelines must run end-to-end whether or not an ``ANTHROPIC_API_KEY`` is
present (overnight CI, the grad student's laptop, a headless box). So every
structured/text call takes an ``offline_fallback`` thunk that computes a valid
result deterministically from the same inputs. ``AnthropicLLMClient`` ignores it
on the happy path (using it only as a last resort if the API hard-fails);
``MockLLMClient`` *is* the fallback. ``get_llm_client`` picks the right one.

Structured output
-----------------
We force a single tool call whose ``input_schema`` is the target Pydantic model's
JSON schema, then validate the tool input with Pydantic and retry up to 3 times
feeding the ``ValidationError`` back — the exact idiom LEAD uses for its
``OrchestratorOutput`` discriminated union (system report §6.3). We deliberately
do *not* set ``strict`` mode, so expressive Pydantic constraints (``min_length``
etc.) stay in the model for internal validation while the retry loop absorbs any
schema drift.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from aora_forge.schemas import LLMUsage
from aora_forge.utils.logging import get_logger

log = get_logger("llm.client")

M = TypeVar("M", bound=BaseModel)


class ModelTier(str, Enum):
    """Which model a call should use. Kept abstract so callers express *intent*
    (planning vs. cheap inner loop), not a concrete model id."""

    PLANNER = "planner"  # high-quality reasoning: clustering, spec generation, prompt authoring
    WORKER = "worker"  # cheap inner loop: validation judging, small helpers


# Model ids and pricing (USD per 1M tokens) — from the claude-api skill, June 2026.
MODEL_BY_TIER: dict[ModelTier, str] = {
    ModelTier.PLANNER: "claude-opus-4-8",
    ModelTier.WORKER: "claude-haiku-4-5",
}
_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
}
_CACHE_WRITE_MULT = 1.25  # cache creation costs ~1.25x the input rate
_CACHE_READ_MULT = 0.10  # cache reads cost ~0.1x the input rate


def _cost_usd(model: str, in_tok: int, out_tok: int, cache_create: int, cache_read: int) -> float:
    in_rate, out_rate = _PRICING.get(model, (0.0, 0.0))
    return (
        in_tok * in_rate
        + out_tok * out_rate
        + cache_create * in_rate * _CACHE_WRITE_MULT
        + cache_read * in_rate * _CACHE_READ_MULT
    ) / 1_000_000.0


def pydantic_tool_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """JSON schema for a Pydantic model, shaped for an Anthropic tool ``input_schema``.

    Pydantic v2 already emits a valid object schema with ``$defs``/``$ref`` (which
    the tool API accepts). We pass it through largely as-is — the retry loop plus
    Pydantic validation is the real correctness guarantee, so we don't need a
    fully strict schema here.
    """
    js = schema.model_json_schema()
    js.setdefault("type", "object")
    return js


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #


class LLMClient:
    """Interface for structured + text completion with usage telemetry."""

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


# --------------------------------------------------------------------------- #
# Deterministic mock — used offline and in unit tests.
# --------------------------------------------------------------------------- #


class MockLLMClient(LLMClient):
    """Returns the caller's ``offline_fallback`` result, validated through the
    schema. Loud about being a mock (``usage.mocked == True``). Deterministic, so
    tests can assert on it.
    """

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
        # Round-trip through the schema so the mock path enforces the same contract.
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


# --------------------------------------------------------------------------- #
# Real Anthropic client.
# --------------------------------------------------------------------------- #


class AnthropicLLMClient(LLMClient):
    """Real Claude calls via the Anthropic SDK.

    * Structured output via a forced single-tool ``tool_use`` + Pydantic-validate
      + retry (3 attempts, error fed back).
    * Prompt caching on the system block (system prompts are stable and large
      enough to clear the 4096-token Opus/Haiku minimum when reused).
    * Transient errors (rate limit, overloaded, 5xx) are retried by the SDK
      (``max_retries``); we add the validation retry on top.
    """

    name = "anthropic"

    def __init__(self, api_key: str | None = None, max_retries: int = 4) -> None:
        import anthropic  # imported lazily so the package imports without the SDK

        # Typed as Any: we build request params as plain dicts (valid TypedDicts at
        # runtime) and read response blocks dynamically, which the SDK's strict
        # overloads would otherwise reject. Validation correctness is enforced by
        # the Pydantic round-trip + retry loop, not by static request typing.
        self._anthropic: Any = anthropic
        self._client: Any = anthropic.Anthropic(api_key=api_key, max_retries=max_retries)

    def _usage_from_response(self, model: str, resp: Any) -> LLMUsage:
        u = resp.usage
        in_tok = getattr(u, "input_tokens", 0) or 0
        out_tok = getattr(u, "output_tokens", 0) or 0
        cc = getattr(u, "cache_creation_input_tokens", 0) or 0
        cr = getattr(u, "cache_read_input_tokens", 0) or 0
        return LLMUsage(
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_creation_input_tokens=cc,
            cache_read_input_tokens=cr,
            cost_usd=_cost_usd(model, in_tok, out_tok, cc, cr),
            mocked=False,
        )

    def _system_blocks(self, system: str, cache: bool) -> list[dict[str, Any]]:
        block: dict[str, Any] = {"type": "text", "text": system}
        if cache:
            block["cache_control"] = {"type": "ephemeral"}
        return [block]

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
        model = MODEL_BY_TIER[model_tier]
        tool_name = f"emit_{schema.__name__.lower()}"
        tool = {
            "name": tool_name,
            "description": f"Return a well-formed {schema.__name__} as the tool input.",
            "input_schema": pydantic_tool_schema(schema),
        }
        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        total = LLMUsage(model=model, mocked=False)
        last_err: str | None = None

        for attempt in range(1, max_validation_retries + 1):
            try:
                resp = self._client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    system=self._system_blocks(system, cache_system),
                    messages=messages,
                    tools=[tool],
                    tool_choice={"type": "tool", "name": tool_name},
                )
            except Exception as exc:  # noqa: BLE001 — SDK already retried transient errors
                log.warning("[%s] API error on attempt %d: %s", task, attempt, exc)
                if attempt == max_validation_retries:
                    log.warning("[%s] falling back to offline result after API failure", task)
                    return offline_fallback(), LLMUsage(model="mock", mocked=True)
                continue

            total = total + self._usage_from_response(model, resp)
            tool_input = _first_tool_use_input(resp)
            if tool_input is None:
                last_err = "model returned no tool_use block"
                messages += _retry_messages(resp, last_err)
                continue
            try:
                obj = schema.model_validate(tool_input)
                return obj, total
            except ValidationError as ve:
                last_err = str(ve)
                log.info("[%s] validation retry %d: %s", task, attempt, last_err.splitlines()[0])
                messages += _retry_messages(resp, f"ValidationError: {ve}")

        log.warning("[%s] exhausted retries (%s); using offline fallback", task, last_err)
        return offline_fallback(), total

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
        model = MODEL_BY_TIER[model_tier]
        try:
            resp = self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=self._system_blocks(system, cache_system),
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("[%s] API error: %s; using offline fallback", task, exc)
            return offline_fallback(), LLMUsage(model="mock", mocked=True)
        text = "".join(
            getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text"
        )
        return text.strip(), self._usage_from_response(model, resp)


def _first_tool_use_input(resp: Any) -> dict[str, Any] | None:
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use":
            inp = block.input
            return inp if isinstance(inp, dict) else json.loads(inp)
    return None


def _retry_messages(resp: Any, error: str) -> list[dict[str, Any]]:
    """Append the assistant turn and a user correction so the model can fix itself."""
    assistant_content = [
        {"type": b.type, **({k: v for k, v in _block_dump(b).items() if k != "type"})}
        for b in resp.content
    ]
    return [
        {"role": "assistant", "content": assistant_content},
        {
            "role": "user",
            "content": (
                f"Your previous tool call did not validate. {error}\n"
                "Call the tool again with corrected input that satisfies the schema."
            ),
        },
    ]


def _block_dump(block: Any) -> dict[str, Any]:
    """Best-effort dict view of a response content block for echoing back."""
    if hasattr(block, "model_dump"):
        return block.model_dump()
    return {"type": getattr(block, "type", "text"), "text": getattr(block, "text", "")}


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def get_llm_client(force_mock: bool = False) -> LLMClient:
    """Return an ``AnthropicLLMClient`` when a key is reachable, else a
    ``MockLLMClient``. The caller never has to branch on backend availability.
    """
    if force_mock:
        log.info("Using MockLLMClient (forced).")
        return MockLLMClient()
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if not api_key:
        log.info("No ANTHROPIC_API_KEY in env — using deterministic MockLLMClient.")
        return MockLLMClient()
    try:
        client = AnthropicLLMClient(api_key=api_key)
        log.info("Using AnthropicLLMClient (real Claude API).")
        return client
    except Exception as exc:  # noqa: BLE001 — SDK missing / construction failed
        log.warning("Anthropic SDK unavailable (%s) — using MockLLMClient.", exc)
        return MockLLMClient()
