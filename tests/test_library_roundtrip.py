"""On-disk skill store: save -> reload round-trip, index, atomic embedding."""

from __future__ import annotations

import numpy as np

from aora_forge.scene_graph.builder import build_from_labels
from aora_forge.schemas import (
    Embodiment,
    Skill,
    SkillIO,
    SkillSpec,
    SkillType,
    SkillValidation,
)
from aora_forge.skill_library.store import SkillStore


def _skill(name: str = "arrival_verifier", labels: list[str] | None = None) -> Skill:
    spec = SkillSpec(
        spec_id=f"spec_{name}",
        skill_name=name,
        skill_type=SkillType.CODE,
        description="verify arrival before done()",
        source_cluster_id="cl0",
        target_embodiments=[Embodiment.DRONE_FIGS, Embodiment.GROUND_HABITAT],
        inputs=[SkillIO(name="depth", type="float", description="P25 depth")],
        outputs=[SkillIO(name="accept", type="bool", description="accept arrival")],
        success_criterion="reject far arrivals",
        integration_point="direct_tool",
    )
    ctx = build_from_labels(labels or ["green clock"], embodiment=Embodiment.DRONE_FIGS)
    return Skill(
        skill_id=name,
        skill_name=name,
        skill_type=SkillType.CODE,
        spec=spec,
        artifact_kind="python",
        artifact_ref="artifact.py",
        artifact_inline="def f():\n    return True\n",
        validation=SkillValidation(passed=True, score=1.0, n_cases_total=2, n_cases_passed=2),
        scene_graph_context=ctx,
    )


def test_save_load_roundtrip(tmp_path) -> None:
    store = SkillStore(tmp_path)
    skill = _skill()
    store.save(skill, embedding=np.ones(8, dtype=np.float32))

    loaded = store.load("arrival_verifier")
    assert loaded.skill_name == skill.skill_name
    assert loaded.spec.success_criterion == "reject far arrivals"
    assert loaded.artifact_inline == skill.artifact_inline

    # artifact + context files written
    sdir = tmp_path / "arrival_verifier"
    assert (sdir / "artifact.py").exists()
    assert (sdir / "scene_graph_context.json").exists()
    assert (sdir / "embedding.npy").exists()

    emb = store.load_embedding("arrival_verifier")
    assert emb is not None and emb.shape == (8,)


def test_index_upsert_and_versioning(tmp_path) -> None:
    store = SkillStore(tmp_path)
    store.save(_skill("s_a"))
    store.save(_skill("s_b"))
    assert len(store) == 2
    entries = {e.skill_id for e in store.entries()}
    assert entries == {"s_a", "s_b"}

    # re-saving the same id bumps the version, not the count
    store.save(_skill("s_a"))
    assert len(store) == 2
    assert store.load("s_a").version == 2


def test_entries_carry_scene_labels(tmp_path) -> None:
    store = SkillStore(tmp_path)
    store.save(_skill("s_clock", labels=["green clock", "table"]))
    entry = next(e for e in store.entries() if e.skill_id == "s_clock")
    assert "green clock" in entry.scene_object_labels
    assert entry.validation_score == 1.0
