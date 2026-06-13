"""AORA-Forge — an embodiment-blind orchestrator that grows new robot skills
from real deployment failures.

The package is organised around the failure → skill pipeline:

    failures/        collect and cluster real deployment failures into themes
    skill_forge/     turn a failure cluster into a trained, validated Skill
    skill_library/   persist, retrieve, and register grown skills as tools
    scene_graph/     sparse scene-graph context that conditions retrieval
    orchestrator_hooks/  inject grown skills back into a LEAD/AORA planner
    embodiments/     the Embodiment interface that keeps the pipeline blind to
                     whether it is serving a drone (FiGS) or a ground robot (HM3D)
    llm/             Anthropic SDK wrapper (+ deterministic mock for offline runs)

`aora_forge.schemas` is the single source of truth for every data structure that
crosses a component boundary.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Aditya Kothari"

from aora_forge import schemas  # noqa: F401  (re-export for `from aora_forge import schemas`)

__all__ = ["schemas", "__version__", "__author__"]
