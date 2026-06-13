"""Hooks that connect AORA-Forge back to a LEAD/AORA orchestrator.

``tool_registry`` : given the current scene context, retrieve relevant grown
                    skills and render them as tools the planner can see.
``post_mission``  : the after-mission growth loop — collect failures, cluster,
                    forge, store. Reused by ``scripts/`` and any live integration.
"""

from aora_forge.orchestrator_hooks.post_mission import (
    GrowthSummary,
    PostMissionHook,
    grow_from_failures,
)
from aora_forge.orchestrator_hooks.tool_registry import OrchestratorToolRegistry

__all__ = [
    "grow_from_failures",
    "GrowthSummary",
    "PostMissionHook",
    "OrchestratorToolRegistry",
]
