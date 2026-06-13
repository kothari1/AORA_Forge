"""Build a sparse scene graph from observations or a failure record.

Tonight's builder is intentionally simple and *honest about being sparse*: real
open-vocabulary graph construction (ConceptGraphs-style multi-view fusion from
RGB-D) is future work and would import the perception stack. Here we build a
useful sparse graph from the structured signal we already have — the target
query, any objects named in the failure narrative, and the embodiment/environment
tags — which is exactly the retrieval key the library needs.
"""

from __future__ import annotations

import re

from aora_forge.scene_graph.context import ensure_summary
from aora_forge.schemas import (
    Embodiment,
    FailureRecord,
    SceneGraphContext,
    SceneGraphNode,
    SceneGraphRelation,
)

# A small open-vocabulary lexicon of object words we expect in LEAD/AORA scenes.
# Used to pull object mentions out of a free-text narrative. Not exhaustive — it
# is a sparse-graph seed, not a perception system.
_OBJECT_WORDS = {
    "clock",
    "mannequin",
    "chair",
    "sofa",
    "couch",
    "bed",
    "table",
    "plant",
    "tv",
    "monitor",
    "television",
    "extinguisher",
    "drill",
    "leafblower",
    "door",
    "window",
    "wall",
    "box",
    "shelf",
    "lamp",
    "cabinet",
    "sink",
    "toilet",
    "refrigerator",
    "fridge",
    "counter",
    "stairs",
    "picture",
}


def _node_id(label: str, idx: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return f"n{idx}_{slug}" or f"n{idx}"


def build_from_labels(
    labels: list[str],
    *,
    embodiment: Embodiment | None = None,
    environment: str | None = None,
    relations: list[tuple[str, str, str]] | None = None,
) -> SceneGraphContext:
    """Build a context from a list of object labels (+ optional (subj, pred, obj) relations)."""
    nodes: list[SceneGraphNode] = []
    label_to_id: dict[str, str] = {}
    for i, lab in enumerate(labels):
        nid = _node_id(lab, i)
        label_to_id[lab.lower()] = nid
        nodes.append(SceneGraphNode(node_id=nid, label=lab))
    rels: list[SceneGraphRelation] = []
    for subj, pred, obj in relations or []:
        s = label_to_id.get(subj.lower())
        o = label_to_id.get(obj.lower())
        if s and o:
            rels.append(SceneGraphRelation(subject_id=s, predicate=pred, object_id=o))
    ctx = SceneGraphContext(
        embodiment=embodiment, environment=environment, nodes=nodes, relations=rels
    )
    return ensure_summary(ctx)


def extract_object_labels(text: str) -> list[str]:
    """Pull a sparse set of object phrases out of free text (target query / narrative)."""
    found: list[str] = []
    lowered = text.lower()
    for word in _OBJECT_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            # try to grab a one-word colour/adjective prefix for a richer label
            m = re.search(rf"(\w+\s+)?{re.escape(word)}", lowered)
            phrase = m.group(0).strip() if m else word
            if phrase not in found:
                found.append(phrase)
    return found


def build_from_failure(record: FailureRecord) -> SceneGraphContext:
    """Build a sparse scene context for a failure record.

    Starts from any ``scene_context`` already attached (e.g. logged by the env),
    otherwise seeds nodes from the target query plus object mentions in the
    narrative and instruction. The target object is always the first node.
    """
    if record.scene_context is not None and record.scene_context.nodes:
        return ensure_summary(record.scene_context)

    labels: list[str] = [record.target_query]
    for extra in extract_object_labels(f"{record.task_instruction} {record.narrative}"):
        if extra.lower() != record.target_query.lower() and extra not in labels:
            labels.append(extra)
    return build_from_labels(labels, embodiment=record.embodiment, environment=record.environment)
