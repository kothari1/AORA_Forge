"""LLM access layer (Anthropic SDK only).

``client`` provides a small ``LLMClient`` interface with two implementations —
a real ``AnthropicLLMClient`` and a deterministic ``MockLLMClient`` — plus a
``get_llm_client`` factory that selects between them based on whether an API key
is reachable. Every pipeline call supplies an ``offline_fallback`` so the mock
(and the demo, when no key is present) produces meaningful, domain-aware output.
"""

from aora_forge.llm.client import (
    AnthropicLLMClient,
    LLMClient,
    MockLLMClient,
    ModelTier,
    get_llm_client,
)

__all__ = [
    "LLMClient",
    "AnthropicLLMClient",
    "MockLLMClient",
    "ModelTier",
    "get_llm_client",
]
