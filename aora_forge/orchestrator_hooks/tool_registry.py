"""Inject grown skills as tools into a LEAD/AORA planner, conditioned on the
current scene context.

The planner asks "given where I am now, what grown capabilities apply?" — the
registry answers by retrieving scene-relevant skills from the library and
rendering them as tool specs that plug into LEAD's ``OrchestratorOutput`` action
vocabulary.
"""

from __future__ import annotations

from typing import Any

from aora_forge.schemas import OrchestratorTool, SceneGraphContext
from aora_forge.skill_library.registry import skill_to_tool
from aora_forge.skill_library.retriever import SkillRetriever
from aora_forge.skill_library.store import SkillStore
from aora_forge.utils.logging import get_logger

log = get_logger("orchestrator_hooks.tool_registry")


class OrchestratorToolRegistry:
    """Serves scene-conditioned grown skills as planner tools."""

    def __init__(self, store: SkillStore, retriever: SkillRetriever | None = None) -> None:
        self.store = store
        self.retriever = retriever or SkillRetriever(store)
        self.retriever.build_index()

    def tools_for_context(
        self,
        scene_context: SceneGraphContext | str,
        *,
        top_k: int = 3,
        scene_graph_conditioned: bool = True,
    ) -> list[OrchestratorTool]:
        """Retrieve the top-k scene-relevant grown skills, as orchestrator tools."""
        hits = self.retriever.retrieve(
            scene_context,
            top_k=top_k,
            scene_graph_conditioned=scene_graph_conditioned,
        )
        tools: list[OrchestratorTool] = []
        for entry, score in hits:
            skill = self.store.load(entry.skill_id)
            tool = skill_to_tool(skill)
            tools.append(tool)
            log.debug("registered tool '%s' (score=%.3f)", tool.name, score)
        return tools

    @staticmethod
    def as_anthropic_tools(tools: list[OrchestratorTool]) -> list[dict[str, Any]]:
        """Render as Anthropic Messages API tool definitions."""
        return [t.to_anthropic_tool() for t in tools]

    @staticmethod
    def as_lead_registrations(tools: list[OrchestratorTool]) -> list[dict[str, Any]]:
        """Render as LEAD planner registrations (against OrchestratorOutput)."""
        return [t.to_lead_registration() for t in tools]
