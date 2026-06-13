"""Render a ``SceneGraphContext`` to its embedding key and compute overlap.

``summary_text`` is the single deterministic string the retriever embeds. Keeping
the rendering here (rather than re-deriving it ad hoc at each call site) is what
makes embedding-based retrieval reproducible: the same graph always yields the
same key.
"""

from __future__ import annotations

from aora_forge.schemas import SceneGraphContext


def render_summary(ctx: SceneGraphContext) -> str:
    """Deterministic natural-language rendering of a scene graph (the embedding key)."""
    parts: list[str] = []
    if ctx.embodiment is not None:
        parts.append(f"Embodiment: {ctx.embodiment.value}.")
    if ctx.environment:
        parts.append(f"Environment: {ctx.environment}.")
    if ctx.nodes:
        node_strs = []
        # sort for determinism
        for n in sorted(ctx.nodes, key=lambda x: x.node_id):
            attrs = ", ".join(f"{k}={v}" for k, v in sorted(n.attributes.items()))
            node_strs.append(f"{n.label}" + (f" ({attrs})" if attrs else ""))
        parts.append("Objects: " + "; ".join(node_strs) + ".")
    if ctx.relations:
        labels = {n.node_id: n.label for n in ctx.nodes}
        rel_strs = []
        for r in sorted(ctx.relations, key=lambda x: (x.subject_id, x.predicate, x.object_id)):
            subj = labels.get(r.subject_id, r.subject_id)
            obj = labels.get(r.object_id, r.object_id)
            rel_strs.append(f"{subj} {r.predicate} {obj}")
        parts.append("Relations: " + "; ".join(rel_strs) + ".")
    return " ".join(parts).strip()


def ensure_summary(ctx: SceneGraphContext) -> SceneGraphContext:
    """Fill ``summary_text`` in place if empty; return the same context."""
    if not ctx.summary_text:
        ctx.summary_text = render_summary(ctx)
    return ctx


def context_overlap(a: SceneGraphContext, b: SceneGraphContext) -> float:
    """Jaccard overlap over object labels — the coarse pre-filter for
    scene-graph-conditioned retrieval. Returns 0.0 when either side is empty.
    """
    la = {n.label.lower() for n in a.nodes}
    lb = {n.label.lower() for n in b.nodes}
    if not la or not lb:
        return 0.0
    return len(la & lb) / len(la | lb)
