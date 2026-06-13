"""Anthropic (Claude API) client: forced-tool structured output, prompt caching,
cost telemetry, and a 3-attempt validate-and-retry loop (LEAD's idiom)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from aora_forge.llm.base import (
    LLMClient,
    M,
    ModelTier,
    cost_usd,
    log,
    pydantic_tool_schema,
)
from aora_forge.llm.config import DEFAULT_MODELS, Provider
from aora_forge.schemas import LLMUsage


class AnthropicLLMClient(LLMClient):
    """Real Claude calls via the Anthropic SDK."""

    name = "anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        models: dict[ModelTier, str] | None = None,
        max_retries: int = 4,
    ) -> None:
        import anthropic  # lazy import so the package imports without the SDK

        # Typed Any: requests are plain dicts (valid TypedDicts at runtime) and
        # responses are read dynamically; correctness comes from the
        # Pydantic-validate-and-retry loop, not static request typing.
        self._anthropic: Any = anthropic
        self._client: Any = anthropic.Anthropic(api_key=api_key, max_retries=max_retries)
        self._models = models or DEFAULT_MODELS[Provider.ANTHROPIC]

    def _usage(self, model: str, resp: Any) -> LLMUsage:
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
            cost_usd=cost_usd(model, in_tok, out_tok, cc, cr),
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
        model = self._models[model_tier]
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
                log.warning("[anthropic:%s] API error attempt %d: %s", task, attempt, exc)
                if attempt == max_validation_retries:
                    return offline_fallback(), LLMUsage(model="mock", mocked=True)
                continue

            total = total + self._usage(model, resp)
            tool_input = _first_tool_use_input(resp)
            if tool_input is None:
                messages += _retry_messages(resp, "model returned no tool_use block")
                continue
            try:
                return schema.model_validate(tool_input), total
            except ValidationError as ve:
                last_err = str(ve)
                log.info("[anthropic:%s] validation retry %d", task, attempt)
                messages += _retry_messages(resp, f"ValidationError: {ve}")

        log.warning("[anthropic:%s] exhausted retries (%s); offline fallback", task, last_err)
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
        model = self._models[model_tier]
        try:
            resp = self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=self._system_blocks(system, cache_system),
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("[anthropic:%s] API error: %s; offline fallback", task, exc)
            return offline_fallback(), LLMUsage(model="mock", mocked=True)
        text = "".join(
            getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text"
        )
        return text.strip(), self._usage(model, resp)


def _first_tool_use_input(resp: Any) -> dict[str, Any] | None:
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use":
            inp = block.input
            return inp if isinstance(inp, dict) else json.loads(inp)
    return None


def _retry_messages(resp: Any, error: str) -> list[dict[str, Any]]:
    assistant_content = [
        {"type": b.type, **{k: v for k, v in _block_dump(b).items() if k != "type"}}
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
    if hasattr(block, "model_dump"):
        return block.model_dump()
    return {"type": getattr(block, "type", "text"), "text": getattr(block, "text", "")}
