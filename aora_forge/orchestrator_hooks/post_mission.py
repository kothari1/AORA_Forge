"""The after-mission growth loop.

This is the whole AORA-Forge thesis in one function: take the failures a mission
logged, cluster them into themes (C2), forge a validated skill for each theme
(possibly inside a 3DGS reconstruction, C3), and store it — for both embodiments
through the same code path (C1). Both ``scripts/demo_full_loop.py`` and any live
post-mission hook call ``grow_from_failures``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aora_forge.failures.clusterer import cluster_failures
from aora_forge.llm.client import LLMClient, get_llm_client
from aora_forge.schemas import FailureCluster, FailureRecord, Skill
from aora_forge.skill_forge.spec_generator import generate_spec
from aora_forge.skill_forge.trainer_base import get_trainer
from aora_forge.skill_library.retriever import get_embedder
from aora_forge.skill_library.store import SkillStore
from aora_forge.utils.logging import RunTelemetry, get_logger

log = get_logger("orchestrator_hooks.post_mission")


@dataclass
class GrowthSummary:
    """What one growth run produced."""

    n_failures: int
    clusters: list[FailureCluster] = field(default_factory=list)
    skills: list[Skill] = field(default_factory=list)
    telemetry: RunTelemetry | None = None

    @property
    def n_skills_validated(self) -> int:
        return sum(1 for s in self.skills if s.validation.passed)


def grow_from_failures(
    records: list[FailureRecord],
    client: LLMClient,
    store: SkillStore,
    *,
    telemetry: RunTelemetry | None = None,
    max_skills: int | None = None,
    embed_on_save: bool = True,
) -> GrowthSummary:
    """Cluster failures, forge a skill per cluster, and store each one."""
    clusters = cluster_failures(records, client, telemetry=telemetry)
    if max_skills is not None:
        clusters = clusters[:max_skills]

    embedder = get_embedder() if embed_on_save else None
    by_id = {r.record_id: r for r in records}
    skills: list[Skill] = []
    for cluster in clusters:
        # Each skill is forged from *its own cluster's* failures — not the whole
        # pile — so its scene-graph key reflects the theme it fixes (essential for
        # discriminative retrieval).
        members = [by_id[m] for m in cluster.member_record_ids if m in by_id]
        spec = generate_spec(cluster, members, client, telemetry=telemetry)
        trainer = get_trainer(spec.skill_type)
        skill = trainer.train(spec, cluster, members, client, telemetry=telemetry)
        embedding = None
        if embedder is not None and skill.scene_graph_context is not None:
            text = (
                f"{skill.skill_name}. {spec.description}. {skill.scene_graph_context.summary_text}"
            )
            embedding = embedder.encode(text)
        store.save(skill, embedding=embedding)
        skills.append(skill)

    summary = GrowthSummary(
        n_failures=len(records), clusters=clusters, skills=skills, telemetry=telemetry
    )
    log.info(
        "growth complete: %d failures -> %d clusters -> %d skills (%d validated)",
        summary.n_failures,
        len(clusters),
        len(skills),
        summary.n_skills_validated,
    )
    return summary


class PostMissionHook:
    """Wraps ``grow_from_failures`` against a fixed store for live integration."""

    def __init__(self, store: SkillStore) -> None:
        self.store = store

    def process(
        self,
        records: list[FailureRecord],
        client: LLMClient | None = None,
        *,
        telemetry: RunTelemetry | None = None,
        max_skills: int | None = None,
    ) -> GrowthSummary:
        client = client or get_llm_client()
        return grow_from_failures(
            records, client, self.store, telemetry=telemetry, max_skills=max_skills
        )
