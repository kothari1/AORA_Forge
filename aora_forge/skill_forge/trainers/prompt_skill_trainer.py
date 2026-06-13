"""Prompt-skill trainer — the simplest, most reliable skill type.

A "prompt skill" is a specialised executor system prompt that the executor adopts
when the scene matches a known failure theme. The trainer: (1) generates the
prompt via Claude (Opus-tier), then (2) validates it by asking a cheap judge
(Haiku-tier) whether an executor following it would avoid each held-out failure
scenario from the cluster. Both steps have deterministic offline fallbacks.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, Field

from aora_forge.llm.client import LLMClient, ModelTier
from aora_forge.llm.prompts import PROMPT_SKILL_SYSTEM, PROMPT_VALIDATION_SYSTEM
from aora_forge.schemas import (
    FailureCluster,
    FailureRecord,
    Skill,
    SkillSpec,
    SkillType,
    SkillValidation,
)
from aora_forge.skill_forge.trainer_base import Trainer
from aora_forge.utils.logging import RunTelemetry, get_logger

log = get_logger("skill_forge.prompt_trainer")

# Tool names the executor already has — referenced by the offline prompt template.
_TOOLS = "observe, move, turn, look_around, capture_image, check_target_depth, done"


class _Verdict(BaseModel):
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    reason: str = ""


def _offline_prompt(spec: SkillSpec, cluster: FailureCluster) -> str:
    """A real, specialised prompt template (used when no API key is present)."""
    return (
        f"SPECIALISED STRATEGY — {cluster.title}\n"
        f"The agent has repeatedly failed in this situation: {cluster.hypothesized_root_cause} "
        f"When the current scene matches this theme, adopt the following discipline.\n\n"
        f"1. Treat arrival as UNCONFIRMED until verified. Before calling done(), you MUST call "
        f"check_target_depth with a tight bounding box drawn around the *visible surface of the "
        f"target only* (not surrounding clutter), and confirm the returned P25 depth is within "
        f"the arrival threshold. If the status is 'uncertain', move closer and re-check rather "
        f"than declaring done().\n"
        f"2. Cross-reference the RGB and depth views: the target's pixels in RGB must map to the "
        f"near (close) colours in the depth image. If a nearer surface sits between you and the "
        f"target, it is an obstacle — reposition, do not declare arrival.\n"
        f"3. If you have scanned (look_around) more than twice without net spatial progress, stop "
        f"scanning and commit to moving toward the most promising direction; re-scan only after "
        f"covering distance.\n"
        f"4. If two objects match the query, prefer the one whose appearance best matches every "
        f"attribute in the instruction; capture_image both before committing if unsure.\n\n"
        f"Available tools (do not invent others): {_TOOLS}.\n"
        f"This strategy is embodiment-blind: it applies whether you translate continuously or in "
        f"discrete steps."
    )


def _verdict_fallback(prompt_text: str, record: FailureRecord) -> Callable[[], _Verdict]:
    """Bind the offline verdict for one record (a named closure mypy can type)."""
    return lambda: _offline_verdict(prompt_text, record)


def _offline_verdict(prompt_text: str, record: FailureRecord) -> _Verdict:
    """Heuristic offline judge: passes if the prompt gives concrete, on-theme guidance."""
    low = prompt_text.lower()
    has_verify = any(k in low for k in ("check_target_depth", "verify", "confirm", "unconfirmed"))
    has_tool = any(t in low for t in ("look_around", "move", "capture_image", "done"))
    substantive = len(prompt_text) > 200
    passed = bool(has_verify and has_tool and substantive)
    score = 0.85 if passed else (0.5 if substantive else 0.2)
    return _Verdict(
        passed=passed,
        score=score,
        reason="prompt gives concrete verify-before-done guidance"
        if passed
        else "prompt too vague",
    )


class PromptSkillTrainer(Trainer):
    skill_type = SkillType.PROMPT

    def train(
        self,
        spec: SkillSpec,
        cluster: FailureCluster,
        records: list[FailureRecord],
        client: LLMClient,
        *,
        telemetry: RunTelemetry | None = None,
    ) -> Skill:
        # 1) generate the prompt skill
        user = (
            f"SkillSpec:\n{spec.model_dump_json(indent=2)}\n\n"
            f"Failure cluster theme: {cluster.title}\n"
            f"Root cause: {cluster.hypothesized_root_cause}\n\n"
            f"Write the specialised executor prompt now."
        )
        prompt_text, gen_usage = client.complete_text(
            system=PROMPT_SKILL_SYSTEM,
            user=user,
            offline_fallback=lambda: _offline_prompt(spec, cluster),
            model_tier=ModelTier.PLANNER,
            task="train_prompt_skill",
            max_tokens=1024,
        )
        if telemetry is not None:
            telemetry.record("train_prompt_skill", gen_usage)

        # 2) validate against held-out scenarios (the cluster's member failures)
        held_out = records[:6]  # bound validation cost
        per_case: list[dict] = []
        passes = 0
        for r in held_out:
            verdict, v_usage = client.complete_structured(
                system=PROMPT_VALIDATION_SYSTEM,
                user=(
                    f"PROMPT SKILL:\n{prompt_text}\n\n"
                    f"HELD-OUT FAILURE SCENARIO:\n"
                    f"  embodiment: {r.embodiment.value}\n"
                    f"  instruction: {r.task_instruction}\n"
                    f"  what went wrong: {r.narrative}\n"
                    f"  failure mode: {r.failure_mode.value}\n\n"
                    f"Would an executor following the prompt skill avoid this failure?"
                ),
                schema=_Verdict,
                offline_fallback=_verdict_fallback(prompt_text, r),
                model_tier=ModelTier.WORKER,
                task="validate_prompt_skill",
                max_tokens=512,
            )
            if telemetry is not None:
                telemetry.record("validate_prompt_skill", v_usage)
            per_case.append(
                {
                    "record_id": r.record_id,
                    "passed": verdict.passed,
                    "score": verdict.score,
                    "reason": verdict.reason,
                }
            )
            passes += int(verdict.passed)

        n = len(held_out)
        score = (sum(c["score"] for c in per_case) / n) if n else 0.0
        validation = SkillValidation(
            passed=(n > 0 and passes >= max(1, n // 2)),
            score=round(score, 3),
            n_cases_total=n,
            n_cases_passed=passes,
            notes=f"{passes}/{n} held-out scenarios judged avoidable.",
            per_case=per_case,
        )

        ctx = self.aggregate_scene_context(records)
        skill = Skill(
            skill_id=spec.skill_name,
            skill_name=spec.skill_name,
            skill_type=SkillType.PROMPT,
            spec=spec,
            artifact_kind="prompt",
            artifact_ref="artifact.prompt",
            artifact_inline=prompt_text,
            validation=validation,
            scene_graph_context=ctx,
            provenance={
                "cluster_id": cluster.cluster_id,
                "record_ids": [r.record_id for r in records],
            },
        )
        log.info(
            "forged prompt skill '%s': %s (score=%.2f)",
            skill.skill_name,
            "PASS" if validation.passed else "WEAK",
            validation.score,
        )
        return skill
