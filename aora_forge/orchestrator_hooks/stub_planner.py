"""A stub LEAD/AORA planner — just enough to *prove the integration point*.

This does NOT run LEAD (no NavEnv, no VLM). It models the planner's tool surface:
the fixed LEAD action vocabulary (``OrchestratorOutput`` union) plus the base
executor tool catalog the planner can dispatch as a ``direct_tool``. AORA-Forge's
``OrchestratorToolRegistry`` injects grown skills into that surface — code/classifier
skills as new ``direct_tool`` entries, prompt skills as executor-prompt
augmentations — and this stub lets us show, concretely, that *the planner now sees a
tool that did not exist before growth*.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aora_forge.orchestrator_hooks.tool_registry import OrchestratorToolRegistry
from aora_forge.schemas import OrchestratorTool, SceneGraphContext

# LEAD's planner action vocabulary (OrchestratorOutput discriminated union, §6.3).
LEAD_PLANNER_ACTIONS = ["dispatch_nav", "direct_tool", "capture_image", "report_done", "abort"]
# The base executor tool catalog a `direct_tool` action can name (system report §3.2).
LEAD_BASE_DIRECT_TOOLS = [
    "observe",
    "move",
    "turn",
    "look_around",
    "capture_image",
    "check_target_depth",
    "done",
]


@dataclass
class PlannerView:
    """What the planner can see/do for one scene."""

    actions: list[str]
    direct_tools: list[str]  # names callable via the direct_tool action
    executor_prompt_augmentations: list[str]  # grown prompt-skill names in effect
    grown_tools: list[OrchestratorTool] = field(default_factory=list)


class StubLEADPlanner:
    """Models the LEAD planner's tool surface and accepts grown-skill injection."""

    def __init__(self) -> None:
        self.actions = list(LEAD_PLANNER_ACTIONS)
        self.base_direct_tools = list(LEAD_BASE_DIRECT_TOOLS)

    def baseline_view(self) -> PlannerView:
        """The planner before any growth: fixed actions + base tools, no augmentations."""
        return PlannerView(
            actions=list(self.actions),
            direct_tools=list(self.base_direct_tools),
            executor_prompt_augmentations=[],
        )

    def view_with_growth(
        self,
        scene_context: SceneGraphContext,
        registry: OrchestratorToolRegistry,
        *,
        top_k: int = 3,
    ) -> PlannerView:
        """The planner after AORA-Forge injects scene-relevant grown skills."""
        grown = registry.tools_for_context(scene_context, top_k=top_k)
        direct = list(self.base_direct_tools)
        augmentations: list[str] = []
        for t in grown:
            if t.integration == "direct_tool" and t.name not in direct:
                direct.append(t.name)
            elif t.integration == "executor_prompt":
                augmentations.append(t.name)
        return PlannerView(
            actions=list(self.actions),
            direct_tools=direct,
            executor_prompt_augmentations=augmentations,
            grown_tools=grown,
        )

    @staticmethod
    def new_capabilities(before: PlannerView, after: PlannerView) -> dict[str, list[str]]:
        """The capabilities growth added — what the planner can now do that it couldn't."""
        return {
            "new_direct_tools": [t for t in after.direct_tools if t not in before.direct_tools],
            "new_executor_prompts": list(after.executor_prompt_augmentations),
        }
