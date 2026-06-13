"""Sparse scene graphs — the structured context that conditions skill retrieval.

The transfer AORA-Forge makes (and that SayPlan/ConceptGraphs/CuriousBot leave
open, per ``docs/01_literature_synthesis.md`` §1.5): use the current scene's
sparse graph as the *retrieval key* over a library of grown skills, not just as a
planning or exploration substrate.

``context``  : render a graph to its embedding key + overlap metrics.
``builder``  : build a sparse graph from observations / failure records (stub).
``update``   : grow/merge a graph from subtask outcomes.
"""

from aora_forge.scene_graph.builder import build_from_failure, build_from_labels
from aora_forge.scene_graph.context import context_overlap, ensure_summary, render_summary
from aora_forge.scene_graph.update import merge_contexts

__all__ = [
    "render_summary",
    "ensure_summary",
    "context_overlap",
    "build_from_labels",
    "build_from_failure",
    "merge_contexts",
]
