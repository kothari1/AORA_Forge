"""Google Gemini via Vertex AI (service-account auth).

Structured output uses Gemini's native ``response_schema`` (a Pydantic model) with
``response_mime_type='application/json'`` — the cleanest of the three providers:
the SDK returns a validated instance on ``response.parsed``. We still keep a
parse-and-retry fallback for robustness.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from aora_forge.llm.base import LLMClient, M, ModelTier, cost_usd, log
from aora_forge.llm.config import DEFAULT_MODELS, GCPConfig, Provider, gcp_config
from aora_forge.schemas import LLMUsage

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


class VertexGeminiLLMClient(LLMClient):
    """Real Gemini calls through Vertex AI with a GCP service account."""

    name = "vertex"

    def __init__(
        self,
        config: GCPConfig | None = None,
        models: dict[ModelTier, str] | None = None,
    ) -> None:
        from google import genai  # lazy import
        from google.oauth2 import service_account

        cfg = config or gcp_config()
        if not cfg.available:
            raise RuntimeError(
                "Vertex requires a GCP service-account JSON. Set GOOGLE_APPLICATION_CREDENTIALS "
                "or AORA_FORGE_GCP_SA, or place .secrets/gcp-lead-sa.json."
            )
        creds = service_account.Credentials.from_service_account_file(cfg.sa_path, scopes=_SCOPES)
        self._genai: Any = genai
        self._client: Any = genai.Client(
            vertexai=True, project=cfg.project, location=cfg.location, credentials=creds
        )
        self._models = models or DEFAULT_MODELS[Provider.VERTEX]
        self._cfg = cfg

    def _usage(self, model: str, resp: Any) -> LLMUsage:
        um = getattr(resp, "usage_metadata", None)
        in_tok = getattr(um, "prompt_token_count", 0) or 0
        out_tok = getattr(um, "candidates_token_count", 0) or 0
        cached = getattr(um, "cached_content_token_count", 0) or 0
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
        total = LLMUsage(model=model, mocked=False)
        contents = user
        last_err: str | None = None

        for attempt in range(1, max_validation_retries + 1):
            cfg: dict[str, Any] = {
                "system_instruction": system,
                "response_mime_type": "application/json",
                "response_schema": schema,
                "max_output_tokens": max_tokens,
            }
            try:
                resp = self._client.models.generate_content(
                    model=model, contents=contents, config=cfg
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("[vertex:%s] API error attempt %d: %s", task, attempt, exc)
                if attempt == max_validation_retries:
                    return offline_fallback(), LLMUsage(model="mock", mocked=True)
                continue

            total = total + self._usage(model, resp)
            # 1) the SDK already validated against the Pydantic schema
            parsed = getattr(resp, "parsed", None)
            if isinstance(parsed, schema):
                return parsed, total
            # 2) fall back to parsing the raw JSON text
            raw = getattr(resp, "text", None)
            if raw:
                try:
                    return schema.model_validate(json.loads(raw)), total
                except (ValidationError, json.JSONDecodeError) as ve:
                    last_err = str(ve)
            else:
                last_err = "empty response"
            log.info("[vertex:%s] validation retry %d", task, attempt)
            contents = (
                f"{user}\n\nYour previous JSON was invalid: {last_err}. "
                "Return corrected JSON that satisfies the schema."
            )

        log.warning("[vertex:%s] exhausted retries (%s); offline fallback", task, last_err)
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
            resp = self._client.models.generate_content(
                model=model,
                contents=user,
                config={"system_instruction": system, "max_output_tokens": max_tokens},
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("[vertex:%s] API error: %s; offline fallback", task, exc)
            return offline_fallback(), LLMUsage(model="mock", mocked=True)
        return (getattr(resp, "text", "") or "").strip(), self._usage(model, resp)
