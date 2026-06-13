"""FailureCluster -> SkillSpec via Claude (the C2 → C3 handoff).

The spec is the contract the trainer fulfils and the library indexes. The offline
fallback builds a valid, embodiment-blind spec deterministically from the cluster
so the pipeline runs without a key.
"""

from __future__ import annotations

import re

from aora_forge.llm.client import LLMClient, ModelTier
from aora_forge.llm.prompts import SPEC_GENERATOR_SYSTEM
from aora_forge.schemas import (
    FailureCluster,
    FailureRecord,
    ReconstructionSpec,
    SkillIO,
    SkillSpec,
    SkillType,
)
from aora_forge.utils.logging import RunTelemetry, get_logger

log = get_logger("skill_forge.spec_generator")


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s or "skill"


def _render_cluster(cluster: FailureCluster, records: list[FailureRecord]) -> str:
    examples = "\n".join(f"  - {r.short()}" for r in records[:5])
    return (
        f"Failure cluster to turn into a SkillSpec:\n"
        f"  cluster_id: {cluster.cluster_id}\n"
        f"  title: {cluster.title}\n"
        f"  description: {cluster.description}\n"
        f"  hypothesized_root_cause: {cluster.hypothesized_root_cause}\n"
        f"  suggested_skill_type: {cluster.suggested_skill_type.value}\n"
        f"  embodiments_involved: {[e.value for e in cluster.embodiments_involved]}\n"
        f"  failure_modes_involved: {[m.value for m in cluster.failure_modes_involved]}\n"
        f"  example member failures:\n{examples}\n\n"
        f"Produce the SkillSpec via the tool. Make target_embodiments cover every "
        f"embodiment in the cluster; keep the skill embodiment-blind."
    )


def _default_io(skill_type: SkillType) -> tuple[list[SkillIO], list[SkillIO]]:
    if skill_type is SkillType.CODE:
        return (
            [
                SkillIO(
                    name="target_bbox",
                    type="bbox",
                    description="normalised [x1,y1,x2,y2] of the target",
                ),
                SkillIO(
                    name="depth_p25_m", type="float", description="P25 depth in the bbox (metres)"
                ),
            ],
            [
                SkillIO(
                    name="accept",
                    type="bool",
                    description="whether the arrival/condition should be accepted",
                )
            ],
        )
    if skill_type is SkillType.CLASSIFIER:
        return (
            [
                SkillIO(
                    name="clip_feature",
                    type="list[float]",
                    description="image-region CLIP embedding",
                )
            ],
            [
                SkillIO(
                    name="is_target",
                    type="bool",
                    description="whether the region is the queried target",
                ),
                SkillIO(name="confidence", type="float", description="probability in [0,1]"),
            ],
        )
    # prompt
    return (
        [
            SkillIO(name="scene_context", type="str", description="rendered scene-graph context"),
            SkillIO(name="observation", type="image_b64", description="current RGB observation"),
        ],
        [
            SkillIO(
                name="strategy_prompt", type="str", description="specialised executor instruction"
            )
        ],
    )


def _offline_spec(cluster: FailureCluster, records: list[FailureRecord]) -> SkillSpec:
    st = cluster.suggested_skill_type
    skill_name = f"{_slug(cluster.title)}_{st.value}"
    inputs, outputs = _default_io(st)
    needs_recon = st in (SkillType.CLASSIFIER, SkillType.POLICY)
    frame_refs = [r.representative_frame_ref for r in records if r.representative_frame_ref]
    return SkillSpec(
        spec_id=f"spec_{cluster.cluster_id}",
        skill_name=skill_name,
        skill_type=st,
        description=f"Skill addressing '{cluster.title}': {cluster.hypothesized_root_cause}",
        source_cluster_id=cluster.cluster_id,
        target_embodiments=cluster.embodiments_involved or [],
        inputs=inputs,
        outputs=outputs,
        success_criterion=(
            f"Resolves the '{cluster.title}' failure on held-out scenarios: "
            f"the agent no longer exhibits {', '.join(m.value for m in cluster.failure_modes_involved)}."
        ),
        training_data_needs=(
            f"Held-out failure scenarios from cluster {cluster.cluster_id} "
            f"({cluster.size} members)."
        ),
        integration_point=(
            f"executor_prompt augmentation when the scene matches '{cluster.title}'"
            if st is SkillType.PROMPT
            else f"direct_tool '{skill_name}' the planner may call before done()"
        ),
        rationale=(
            f"A {st.value} skill is the most reliable fix for this root cause "
            f"({cluster.hypothesized_root_cause})."
        ),
        reconstruction=ReconstructionSpec(
            needed=needs_recon,
            source_frame_refs=frame_refs,
            method="stub",
            notes=(
                "Train inside a 3DGS reconstruction of the failure scene (C3)."
                if needs_recon
                else "Pure prompt/logic skill; no reconstruction needed."
            ),
        ),
    )


def generate_spec(
    cluster: FailureCluster,
    records: list[FailureRecord],
    client: LLMClient,
    *,
    telemetry: RunTelemetry | None = None,
) -> SkillSpec:
    """Generate a ``SkillSpec`` for one failure cluster."""
    members = [r for r in records if r.record_id in set(cluster.member_record_ids)]
    spec, usage = client.complete_structured(
        system=SPEC_GENERATOR_SYSTEM,
        user=_render_cluster(cluster, members),
        schema=SkillSpec,
        offline_fallback=lambda: _offline_spec(cluster, members),
        model_tier=ModelTier.PLANNER,
        task="generate_spec",
        max_tokens=4096,
    )
    if telemetry is not None:
        telemetry.record("generate_spec", usage)
    # Guarantee provenance fields the LLM might omit.
    if not spec.source_cluster_id:
        spec.source_cluster_id = cluster.cluster_id
    if not spec.target_embodiments:
        spec.target_embodiments = cluster.embodiments_involved
    log.info(
        "spec '%s' (%s) for cluster %s [%s]",
        spec.skill_name,
        spec.skill_type.value,
        cluster.cluster_id,
        "mock" if usage.mocked else usage.model,
    )
    return spec
