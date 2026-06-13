"""OpenAI client: forced function-calling for structured output, validate-and-retry,
cost telemetry. Same ``LLMClient`` interface as the Anthropic and Vertex clients."""

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


class OpenAILLMClient(LLMClient):
    """Real OpenAI calls via the official SDK."""

    name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        models: dict[ModelTier, str] | None = None,
        max_retries: int = 4,
    ) -> None:
        import openai  # lazy import

        self._openai: Any = openai
        self._client: Any = openai.OpenAI(api_key=api_key, max_retries=max_retries)
        self._models = models or DEFAULT_MODELS[Provider.OPENAI]

    def _usage(self, model: str, resp: Any) -> LLMUsage:
        u = getattr(resp, "usage", None)
        in_tok = getattr(u, "prompt_tokens", 0) or 0
        out_tok = getattr(u, "completion_tokens", 0) or 0
        cached = 0
        details = getattr(u, "prompt_tokens_details", None)
        if details is not None:
            cached = getattr(details, "cached_tokens", 0) or 0
        return LLMUsage(
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_read_input_tokens=cached,
            cost_usd=cost_usd(model, in_tok, out_tok, 0, cached),
            mocked=False,
        )

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
        fn_name = f"emit_{schema.__name__.lower()}"
        tool = {
            "type": "function",
            "function": {
                "name": fn_name,
                "description": f"Return a well-formed {schema.__name__}.",
                "parameters": pydantic_tool_schema(schema),
            },
        }
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        total = LLMUsage(model=model, mocked=False)
        last_err: str | None = None

        for attempt in range(1, max_validation_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=[tool],
                    tool_choice={"type": "function", "function": {"name": fn_name}},
                    max_tokens=max_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("[openai:%s] API error attempt %d: %s", task, attempt, exc)
                if attempt == max_validation_retries:
                    return offline_fallback(), LLMUsage(model="mock", mocked=True)
                continue

            total = total + self._usage(model, resp)
            args = _first_function_args(resp)
            if args is None:
                # re-ask without an unmatched assistant tool_call (keeps messages valid)
                messages.append({"role": "user", "content": "Return the function call now."})
                continue
            try:
                return schema.model_validate(args), total
            except ValidationError as ve:
                last_err = str(ve)
                log.info("[openai:%s] validation retry %d", task, attempt)
                messages.append(
                    {
                        "role": "user",
                        "content": f"Your previous arguments were invalid: {ve}. "
                        "Call the function again with corrected arguments.",
                    }
                )

        log.warning("[openai:%s] exhausted retries (%s); offline fallback", task, last_err)
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
            resp = self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("[openai:%s] API error: %s; offline fallback", task, exc)
            return offline_fallback(), LLMUsage(model="mock", mocked=True)
        text = resp.choices[0].message.content or ""
        return text.strip(), self._usage(model, resp)


def _first_function_args(resp: Any) -> dict[str, Any] | None:
    try:
        tool_calls = resp.choices[0].message.tool_calls
    except (AttributeError, IndexError):
        return None
    if not tool_calls:
        return None
    raw = tool_calls[0].function.arguments
    try:
        return raw if isinstance(raw, dict) else json.loads(raw)
    except json.JSONDecodeError:
        return None
