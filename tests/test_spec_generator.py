"""Skill-spec generation with the mocked LLM."""

from __future__ import annotations

from aora_forge.failures.clusterer import cluster_failures
from aora_forge.llm.client import MockLLMClient
from aora_forge.schemas import SkillType
from aora_forge.skill_forge.spec_generator import generate_spec
from aora_forge.utils.synthetic_data import generate_synthetic_failures


def test_generate_spec_contract() -> None:
    records = generate_synthetic_failures(40, seed=3)
    client = MockLLMClient()
    clusters = cluster_failures(records, client)
    cluster = clusters[0]
    spec = generate_spec(cluster, records, client)

    assert spec.skill_name and spec.skill_name == spec.skill_name.lower()
    assert " " not in spec.skill_name  # snake_case identifier
    assert spec.source_cluster_id == cluster.cluster_id
    # embodiment-blind: spec covers the cluster's embodiments
    assert set(spec.target_embodiments) == set(cluster.embodiments_involved)
    assert spec.success_criterion
    assert spec.integration_point


def test_classifier_spec_requests_reconstruction() -> None:
    """A look-alike (classifier) cluster should request a 3DGS reconstruction (C3)."""
    records = generate_synthetic_failures(50, seed=4)
    client = MockLLMClient()
    clusters = cluster_failures(records, client)
    target = next((c for c in clusters if c.suggested_skill_type is SkillType.CLASSIFIER), None)
    assert target is not None, "expected a classifier-suggesting cluster"
    spec = generate_spec(target, records, client)
    assert spec.skill_type is SkillType.CLASSIFIER
    assert spec.reconstruction.needed is True
