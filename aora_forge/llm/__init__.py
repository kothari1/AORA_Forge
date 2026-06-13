"""Multi-provider LLM layer (Anthropic, OpenAI, Vertex Gemini) — Anthropic SDK,
OpenAI SDK, and google-genai (Vertex) behind one ``LLMClient`` interface.

``get_llm_client`` picks the provider from ``AORA_FORGE_PROVIDER`` or the first
reachable credential, falling back to a deterministic ``MockLLMClient`` so every
pipeline call runs with or without a key. Each call supplies an ``offline_fallback``
that produces meaningful, domain-aware output under the mock.
"""

from aora_forge.llm.base import LLMClient, MockLLMClient, ModelTier
from aora_forge.llm.client import (
    AnthropicLLMClient,
    OpenAILLMClient,
    Provider,
    VertexGeminiLLMClient,
    get_llm_client,
    resolve_provider,
)

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
]
