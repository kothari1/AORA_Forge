"""End-to-end smoke test of the full growth loop on synthetic data (mocked LLM).

This is the pytest counterpart of ``scripts/demo_full_loop.py``: it must stay
green so the architecture's data flow is regression-protected.
"""

from __future__ import annotations

from aora_forge.llm.client import MockLLMClient
from aora_forge.orchestrator_hooks.post_mission import grow_from_failures
from aora_forge.orchestrator_hooks.tool_registry import OrchestratorToolRegistry
from aora_forge.scene_graph.builder import build_from_labels
from aora_forge.schemas import Embodiment, SkillType
from aora_forge.skill_library.store import SkillStore
from aora_forge.utils.logging import RunTelemetry
from aora_forge.utils.synthetic_data import generate_synthetic_failures


def test_full_growth_loop(tmp_path) -> None:
    records = generate_synthetic_failures(30, seed=7)
    client = MockLLMClient()
    telemetry = RunTelemetry()
    store = SkillStore(tmp_path / "library")

    summary = grow_from_failures(records, client, store, telemetry=telemetry)

    # produced skills, all validated in the deterministic path
    assert summary.skills, "expected forged skills"
    assert summary.n_skills_validated == len(summary.skills)
    # at least one of each reliable skill type appears across themes
    types = {s.skill_type for s in summary.skills}
    assert SkillType.PROMPT in types
    # every skill persisted and reloadable
    assert len(store) == len(summary.skills)
    for s in summary.skills:
        reloaded = store.load(s.skill_id)
        assert reloaded.validation.passed

    # code skills carry a runnable artifact; classifier skills a learned head
    for s in summary.skills:
        if s.skill_type is SkillType.CODE:
            assert "def " in (s.artifact_inline or "")
        if s.skill_type is SkillType.CLASSIFIER:
            assert s.validation.score >= 0.8  # the numpy MLP separates the synthetic classes

    # telemetry recorded the stages and is flagged as mock
    assert telemetry.all_mocked
    assert telemetry.stage_calls.get("cluster_failures", 0) >= 1
    assert telemetry.stage_calls.get("generate_spec", 0) >= 1

    # retrieval renders tools that plug into the planner
    registry = OrchestratorToolRegistry(store)
    ctx = build_from_labels(["green clock"], embodiment=Embodiment.DRONE_FIGS)
    tools = registry.tools_for_context(ctx, top_k=3)
    assert tools, "expected retrievable tools"
    anthropic_tools = registry.as_anthropic_tools(tools)
    assert all("input_schema" in t for t in anthropic_tools)
    lead_regs = registry.as_lead_registrations(tools)
    assert all(r["integration"] in ("direct_tool", "executor_prompt") for r in lead_regs)
