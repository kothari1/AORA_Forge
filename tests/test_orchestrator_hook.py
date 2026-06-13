"""The orchestrator hook: the planner's tool surface strictly grows after growth."""

from __future__ import annotations

from aora_forge.llm.client import MockLLMClient
from aora_forge.orchestrator_hooks.post_mission import grow_from_failures
from aora_forge.orchestrator_hooks.stub_planner import (
    LEAD_BASE_DIRECT_TOOLS,
    StubLEADPlanner,
)
from aora_forge.orchestrator_hooks.tool_registry import OrchestratorToolRegistry
from aora_forge.scene_graph.builder import build_from_labels
from aora_forge.schemas import Embodiment
from aora_forge.skill_library.store import SkillStore
from aora_forge.utils.synthetic_data import generate_synthetic_failures


def test_planner_gains_tools_after_growth(tmp_path) -> None:
    records = generate_synthetic_failures(40, seed=0)
    store = SkillStore(tmp_path / "library")
    grow_from_failures(records, MockLLMClient(), store)
    registry = OrchestratorToolRegistry(store)

    planner = StubLEADPlanner()
    before = planner.baseline_view()
    assert before.direct_tools == LEAD_BASE_DIRECT_TOOLS
    assert before.executor_prompt_augmentations == []

    ctx = build_from_labels(
        ["green clock", "table"], embodiment=Embodiment.DRONE_FIGS, environment="flightroom_ssv_exp"
    )
    after = planner.view_with_growth(ctx, registry, top_k=3)

    # the planner retains every base tool ...
    for t in LEAD_BASE_DIRECT_TOOLS:
        assert t in after.direct_tools
    # ... and gains at least one grown capability (direct tool or prompt augmentation)
    added = StubLEADPlanner.new_capabilities(before, after)
    assert added["new_direct_tools"] or added["new_executor_prompts"], (
        "growth should add at least one capability for a clock scene"
    )
    # every grown tool renders as a valid Anthropic tool spec
    for tool in after.grown_tools:
        spec = tool.to_anthropic_tool()
        assert spec["name"] and spec["input_schema"]["type"] == "object"
