"""Retrieval: the hashing embedder, embedding-mode, and scene-graph conditioning."""

from __future__ import annotations

import numpy as np

from aora_forge.llm.client import MockLLMClient
from aora_forge.orchestrator_hooks.post_mission import grow_from_failures
from aora_forge.scene_graph.builder import build_from_labels
from aora_forge.schemas import Embodiment
from aora_forge.skill_library.retriever import HashingEmbedder, SkillRetriever
from aora_forge.skill_library.store import SkillStore
from aora_forge.utils.synthetic_data import generate_synthetic_failures


def test_hashing_embedder_similarity() -> None:
    emb = HashingEmbedder(dim=128)
    a = emb.encode("green clock on the table")
    b = emb.encode("a green clock near a table")
    c = emb.encode("repeated collisions into a wall")
    # normalised
    assert abs(np.linalg.norm(a) - 1.0) < 1e-6
    # shared vocabulary => higher cosine than unrelated text
    assert float(a @ b) > float(a @ c)


def _populated_store(tmp_path, n: int = 40, seed: int = 5) -> SkillStore:
    records = generate_synthetic_failures(n, seed=seed)
    store = SkillStore(tmp_path)
    grow_from_failures(records, MockLLMClient(), store)
    return store


def test_retrieve_embedding_mode(tmp_path) -> None:
    store = _populated_store(tmp_path)
    retr = SkillRetriever(store)
    retr.build_index()
    hits = retr.retrieve("small target premature arrival clock", top_k=3)
    assert hits, "expected retrieval hits"
    assert len(hits) <= 3
    # scores are floats, descending
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_scene_graph_conditioned_surfaces_relevant_skill(tmp_path) -> None:
    store = _populated_store(tmp_path)
    retr = SkillRetriever(store)
    retr.build_index()
    # a clock scene should surface a skill whose forged-in context contains a clock
    q = build_from_labels(["green clock", "table"], embodiment=Embodiment.DRONE_FIGS)
    hits = retr.retrieve(q, top_k=3, scene_graph_conditioned=True)
    assert hits, "expected scene-conditioned hits"
    top_ids = {e.skill_id for e, _ in hits}
    # the top results should be among skills whose scene context mentions a clock
    clock_skills = {
        e.skill_id for e in store.entries() if any("clock" in lab for lab in e.scene_object_labels)
    }
    assert top_ids & clock_skills, "a clock-relevant skill should be retrieved"
