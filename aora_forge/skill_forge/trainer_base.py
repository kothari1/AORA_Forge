"""The ``Trainer`` ABC and a factory mapping ``SkillType`` to a concrete trainer.

A trainer takes a ``SkillSpec`` plus the failures that motivated it and returns a
validated ``Skill``. Each ABC has at least one concrete implementation (prompt,
code, classifier) — no empty stubs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from aora_forge.llm.client import LLMClient
from aora_forge.scene_graph.builder import build_from_failure
from aora_forge.schemas import (
    FailureCluster,
    FailureRecord,
    SceneGraphContext,
    Skill,
    SkillSpec,
    SkillType,
)
from aora_forge.utils.logging import RunTelemetry


class Trainer(ABC):
    """Turn a SkillSpec + its failures into a validated Skill."""

    skill_type: SkillType

    @abstractmethod
    def train(
        self,
        spec: SkillSpec,
        cluster: FailureCluster,
        records: list[FailureRecord],
        client: LLMClient,
        *,
        telemetry: RunTelemetry | None = None,
    ) -> Skill:
        """Forge and validate the skill."""

    # ---- shared helpers -------------------------------------------------- #

    @staticmethod
    def aggregate_scene_context(records: list[FailureRecord]) -> SceneGraphContext | None:
        """Build the skill's retrieval key from the scenes it was born to fix.

        Rather than a raw union (which over-includes one-off contaminating objects
        from unrelated failures that happened to land in the cluster), we keep the
        *characteristic* objects — those recurring across the cluster's failures.
        That denoising is what makes object-overlap retrieval discriminative: a
        small-target skill's key is dominated by small targets, not by a stray
        sofa that slipped in.
        """
        from collections import Counter

        from aora_forge.scene_graph.builder import build_from_labels

        if not records:
            return None
        label_counts: Counter[str] = Counter()
        env_counts: Counter[str] = Counter()
        emb_counts: Counter = Counter()
        for r in records:
            rc = build_from_failure(r)
            for node in rc.nodes:
                label_counts[node.label] += 1
            if r.environment:
                env_counts[r.environment] += 1
            emb_counts[r.embodiment] += 1

        recurring = [lab for lab, c in label_counts.items() if c >= 2]
        labels = (
            [lab for lab, _ in label_counts.most_common(8)]
            if not recurring
            else [lab for lab, _ in label_counts.most_common(10) if lab in recurring]
        )
        embodiment = emb_counts.most_common(1)[0][0] if emb_counts else None
        environment = env_counts.most_common(1)[0][0] if env_counts else None
        return build_from_labels(labels, embodiment=embodiment, environment=environment)


def get_trainer(skill_type: SkillType) -> Trainer:
    """Return a concrete trainer for the given skill type.

    Imported lazily to avoid a circular import (trainers import this module).
    """
    from aora_forge.skill_forge.trainers.classifier_trainer import ClassifierSkillTrainer
    from aora_forge.skill_forge.trainers.code_skill_trainer import CodeSkillTrainer
    from aora_forge.skill_forge.trainers.prompt_skill_trainer import PromptSkillTrainer

    if skill_type is SkillType.PROMPT:
        return PromptSkillTrainer()
    if skill_type is SkillType.CODE:
        return CodeSkillTrainer()
    if skill_type is SkillType.CLASSIFIER:
        return ClassifierSkillTrainer()
    # POLICY (3DGS-trained closed-loop) is future work; fall back to the most
    # reliable trainer so the pipeline still produces a usable skill.
    return PromptSkillTrainer()
