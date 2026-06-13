"""Provider selection and per-provider model configuration.

AORA-Forge speaks to three providers through one interface: Anthropic (Claude
API), OpenAI, and Google Gemini via Vertex AI (service-account auth). Which one is
used is resolved here — from ``AORA_FORGE_PROVIDER`` if set, else by which
credential is reachable, else the deterministic mock.

Per-provider model ids (per ``ModelTier``) have sensible defaults and can be
overridden with ``AORA_FORGE_<PROVIDER>_<TIER>_MODEL`` env vars, so a model rename
never needs a code change.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from aora_forge.llm.base import ModelTier
from aora_forge.utils.logging import get_logger

log = get_logger("llm.config")


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    VERTEX = "vertex"  # Gemini via Vertex AI (service-account)
    MOCK = "mock"


# Default model ids per provider and tier.
DEFAULT_MODELS: dict[Provider, dict[ModelTier, str]] = {
    Provider.ANTHROPIC: {
        ModelTier.PLANNER: "claude-opus-4-8",
        ModelTier.WORKER: "claude-haiku-4-5",
    },
    Provider.OPENAI: {
        ModelTier.PLANNER: "gpt-4o",
        ModelTier.WORKER: "gpt-4o-mini",
    },
    Provider.VERTEX: {
        ModelTier.PLANNER: "gemini-2.5-pro",
        ModelTier.WORKER: "gemini-2.5-flash",
    },
}

# Default location for the GCP service-account key shipped with the repo (gitignored).
_DEFAULT_SA_PATH = Path(__file__).resolve().parents[2] / ".secrets" / "gcp-lead-sa.json"


def model_map(provider: Provider) -> dict[ModelTier, str]:
    """Resolve the {tier -> model id} map for a provider, honouring env overrides."""
    base = dict(DEFAULT_MODELS.get(provider, {}))
    for tier in (ModelTier.PLANNER, ModelTier.WORKER):
        env_key = f"AORA_FORGE_{provider.value.upper()}_{tier.value.upper()}_MODEL"
        override = os.environ.get(env_key)
        if override:
            base[tier] = override
    return base


@dataclass
class GCPConfig:
    """Resolved GCP/Vertex settings."""

    sa_path: str | None
    project: str | None
    location: str

    @property
    def available(self) -> bool:
        return bool(self.sa_path and Path(self.sa_path).exists())


def gcp_config() -> GCPConfig:
    """Resolve Vertex settings from env + the bundled service-account file."""
    sa_path = (
        os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        or os.environ.get("AORA_FORGE_GCP_SA")
        or (str(_DEFAULT_SA_PATH) if _DEFAULT_SA_PATH.exists() else None)
    )
    project = os.environ.get("AORA_FORGE_GCP_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project and sa_path and Path(sa_path).exists():
        try:
            project = json.loads(Path(sa_path).read_text()).get("project_id")
        except Exception:  # noqa: BLE001
            project = None
    location = os.environ.get("AORA_FORGE_GCP_LOCATION", "us-central1")
    return GCPConfig(sa_path=sa_path, project=project, location=location)


def _has_anthropic() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def _has_openai() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def resolve_provider() -> Provider:
    """Pick the active provider.

    1. ``AORA_FORGE_PROVIDER`` (anthropic | openai | vertex | gemini | mock) wins.
    2. Else the first reachable credential, in order: Anthropic, OpenAI, Vertex.
    3. Else the deterministic mock.
    """
    forced = os.environ.get("AORA_FORGE_PROVIDER", "").strip().lower()
    if forced:
        alias = {
            "gemini": Provider.VERTEX,
            "vertexai": Provider.VERTEX,
            "claude": Provider.ANTHROPIC,
        }
        try:
            return alias.get(forced, Provider(forced))
        except ValueError:
            log.warning("unknown AORA_FORGE_PROVIDER=%r; falling back to auto-detect", forced)

    if _has_anthropic():
        return Provider.ANTHROPIC
    if _has_openai():
        return Provider.OPENAI
    if gcp_config().available:
        return Provider.VERTEX
    return Provider.MOCK
