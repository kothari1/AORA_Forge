"""Grow/merge scene graphs from subtask outcomes.

The persistent-memory direction the survey (§1.5) flags as open: a graph that
grows from successful subtask completions. Tonight this provides the merge
primitive (union of nodes/relations, dedup by label) the retriever and the
post-mission hook use; the full incremental online builder is future work.
"""

from __future__ import annotations

from aora_forge.scene_graph.context import ensure_summary
from aora_forge.schemas import SceneGraphContext, SceneGraphNode


def merge_contexts(base: SceneGraphContext, incoming: SceneGraphContext) -> SceneGraphContext:
    """Union two contexts, deduplicating nodes by (lowercased) label.

    Keeps ``base``'s embodiment/environment unless missing. Re-renders the
    summary so the merged context has a fresh embedding key.
    """
    by_label: dict[str, SceneGraphNode] = {n.label.lower(): n for n in base.nodes}
    for n in incoming.nodes:
        key = n.label.lower()
        if key not in by_label:
            by_label[key] = n
        else:
            # merge attributes, keep higher confidence
            existing = by_label[key]
            existing.attributes.update(n.attributes)
            existing.confidence = max(existing.confidence, n.confidence)

    merged = SceneGraphContext(
        embodiment=base.embodiment or incoming.embodiment,
        environment=base.environment or incoming.environment,
        nodes=list(by_label.values()),
        relations=base.relations + incoming.relations,
    )
    merged.summary_text = ""  # force re-render
    return ensure_summary(merged)
