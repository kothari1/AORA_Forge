"""Facade over the multi-provider LLM layer.

Keeps the import surface stable (`from aora_forge.llm.client import LLMClient,
ModelTier, MockLLMClient, AnthropicLLMClient, get_llm_client`) while the actual
clients live in per-provider modules. ``get_llm_client`` selects Anthropic,
OpenAI, Vertex Gemini, or the deterministic mock.
"""

from __future__ import annotations

import os

from aora_forge.llm.anthropic_client import AnthropicLLMClient
from aora_forge.llm.base import (
    PRICING,
    LLMClient,
    MockLLMClient,
    ModelTier,
    cost_usd,
    pydantic_tool_schema,
)
from aora_forge.llm.config import (
    DEFAULT_MODELS,
    Provider,
    gcp_config,
    model_map,
    resolve_provider,
)
from aora_forge.llm.openai_client import OpenAILLMClient
from aora_forge.llm.vertex_client import VertexGeminiLLMClient
from aora_forge.utils.logging import get_logger

log = get_logger("llm.client")

# Backward-compatible alias (Anthropic tier->model map).
MODEL_BY_TIER = DEFAULT_MODELS[Provider.ANTHROPIC]

__all__ = [
    "LLMClient",
    "MockLLMClient",
    "AnthropicLLMClient",
    "OpenAILLMClient",
    "VertexGeminiLLMClient",
    "ModelTier",
    "Provider",
    "get_llm_client",
    "resolve_provider",
    "pydantic_tool_schema",
    "cost_usd",
    "PRICING",
    "MODEL_BY_TIER",
]


def get_llm_client(provider: Provider | str | None = None, force_mock: bool = False) -> LLMClient:
    """Return a client for the requested (or auto-resolved) provider.

    Resolution: explicit ``provider`` arg > ``AORA_FORGE_PROVIDER`` env > first
    reachable credential (Anthropic, OpenAI, Vertex) > deterministic mock. Any
    construction failure (missing SDK, bad credential) degrades to the mock, so the
    pipeline always runs.
    """
    if force_mock:
        log.info("Using MockLLMClient (forced).")
        return MockLLMClient()

    if provider is None:
        prov = resolve_provider()
    elif isinstance(provider, Provider):
        prov = provider
    else:
        prov = resolve_provider() if not provider else Provider(str(provider).lower())

    if prov is Provider.MOCK:
        log.info("No reachable LLM credential — using deterministic MockLLMClient.")
        return MockLLMClient()

    try:
        if prov is Provider.ANTHROPIC:
            key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
            client: LLMClient = AnthropicLLMClient(api_key=key, models=model_map(prov))
        elif prov is Provider.OPENAI:
            client = OpenAILLMClient(
                api_key=os.environ.get("OPENAI_API_KEY"), models=model_map(prov)
            )
        elif prov is Provider.VERTEX:
            client = VertexGeminiLLMClient(config=gcp_config(), models=model_map(prov))
        else:  # pragma: no cover - exhaustive
            return MockLLMClient()
    except Exception as exc:  # noqa: BLE001 — SDK missing / construction failed
        log.warning("Provider %s unavailable (%s) — using MockLLMClient.", prov.value, exc)
        return MockLLMClient()

    models = model_map(prov)
    log.info(
        "Using %s (planner=%s, worker=%s).",
        client.name,
        models[ModelTier.PLANNER],
        models[ModelTier.WORKER],
    )
    return client
