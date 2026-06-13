"""Failure clustering with the mocked (deterministic) LLM."""

from __future__ import annotations

from aora_forge.failures.clusterer import cluster_failures
from aora_forge.llm.client import MockLLMClient
from aora_forge.utils.synthetic_data import generate_synthetic_failures


def test_cluster_failures_basic() -> None:
    records = generate_synthetic_failures(40, seed=1)
    clusters = cluster_failures(records, MockLLMClient())

    assert clusters, "expected at least one cluster"
    # sorted by priority descending
    priorities = [c.priority for c in clusters]
    assert priorities == sorted(priorities, reverse=True)

    valid_ids = {r.record_id for r in records}
    seen: set[str] = set()
    for c in clusters:
        assert c.member_record_ids, "cluster must have members"
        for m in c.member_record_ids:
            assert m in valid_ids, "member id must reference a real record"
        seen.update(c.member_record_ids)
        assert c.embodiments_involved, "embodiments must be filled"
        assert c.failure_modes_involved, "failure modes must be filled"
        assert 1 <= len(c.representative_record_ids) <= 3

    # every record lands in some cluster (offline path partitions by theme)
    assert seen == valid_ids


def test_cluster_failures_empty() -> None:
    assert cluster_failures([], MockLLMClient()) == []


def test_cluster_spans_both_embodiments() -> None:
    records = generate_synthetic_failures(50, seed=2)
    clusters = cluster_failures(records, MockLLMClient())
    # at least one theme should involve both embodiments (cross-embodiment, C1)
    assert any(len(c.embodiments_involved) == 2 for c in clusters)
