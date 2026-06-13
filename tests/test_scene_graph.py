"""Scene graph: building, summary rendering, overlap, and merge."""

from __future__ import annotations

from aora_forge.scene_graph.builder import (
    build_from_failure,
    build_from_labels,
    extract_object_labels,
)
from aora_forge.scene_graph.context import context_overlap, ensure_summary, render_summary
from aora_forge.scene_graph.update import merge_contexts
from aora_forge.schemas import (
    Embodiment,
    FailureMode,
    FailureRecord,
)


def test_build_from_labels_and_summary() -> None:
    ctx = build_from_labels(
        ["green clock", "table"],
        embodiment=Embodiment.DRONE_FIGS,
        environment="flightroom_ssv_exp",
        relations=[("green clock", "on", "table")],
    )
    assert ctx.object_labels() == ["green clock", "table"]
    assert len(ctx.relations) == 1
    assert ctx.summary_text  # filled by ensure_summary
    summary = render_summary(ctx)
    assert "green clock" in summary and "on" in summary and "drone_figs" in summary


def test_extract_object_labels() -> None:
    labels = extract_object_labels("the agent declared done at the green clock near a wall")
    assert any("clock" in lab for lab in labels)
    assert any("wall" in lab for lab in labels)


def test_build_from_failure_uses_target() -> None:
    rec = FailureRecord(
        record_id="r1",
        embodiment=Embodiment.GROUND_HABITAT,
        environment="hm3d:00800",
        task_instruction="find the sofa",
        target_query="sofa",
        failure_mode=FailureMode.WRONG_ROOM,
        narrative="searched the wrong room for the sofa, never crossed the doorway",
    )
    ctx = build_from_failure(rec)
    assert "sofa" in ctx.object_labels()
    assert ctx.embodiment is Embodiment.GROUND_HABITAT


def test_context_overlap() -> None:
    a = build_from_labels(["green clock", "table"])
    b = build_from_labels(["green clock", "chair"])
    c = build_from_labels(["wall", "doorway"])
    assert context_overlap(a, b) > 0.0  # share "green clock"
    assert context_overlap(a, c) == 0.0  # disjoint
    # empty -> 0
    from aora_forge.schemas import SceneGraphContext

    assert context_overlap(a, SceneGraphContext()) == 0.0


def test_merge_contexts_dedup() -> None:
    a = build_from_labels(["green clock", "table"], embodiment=Embodiment.DRONE_FIGS)
    b = build_from_labels(["table", "chair"])
    merged = merge_contexts(a, b)
    labels = set(merged.object_labels())
    assert labels == {"green clock", "table", "chair"}  # table deduped
    assert merged.embodiment is Embodiment.DRONE_FIGS
    assert merged.summary_text  # re-rendered


def test_ensure_summary_idempotent() -> None:
    from aora_forge.schemas import SceneGraphContext, SceneGraphNode

    ctx = SceneGraphContext(nodes=[SceneGraphNode(node_id="n0", label="bed")])
    ensure_summary(ctx)
    first = ctx.summary_text
    ensure_summary(ctx)  # should not change
    assert ctx.summary_text == first
    assert "bed" in first
