"""Two-mode skill retrieval.

* **Embedding-only** — encode the current context, return the top-k skills by
  cosine similarity.
* **Scene-graph-conditioned** — first filter to skills whose forged-in scene graph
  overlaps the current scene (object-label Jaccard), then rank the survivors by
  embedding similarity. This is the operation §1.5 of the synthesis flags as the
  open transfer: the sparse scene graph as a *retrieval key over grown skills*.

Embeddings come from ``sentence-transformers`` when available **and** explicitly
opted into (``AORA_FORGE_USE_ST=1``); otherwise a deterministic NumPy hashing
embedder is used so retrieval is reproducible and fully offline (no model
download). Shared vocabulary still drives similarity, which is what the demo and
tests need.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Protocol

import numpy as np

from aora_forge.scene_graph.context import context_overlap, ensure_summary
from aora_forge.schemas import SceneGraphContext, SkillLibraryEntry
from aora_forge.skill_library.store import SkillStore
from aora_forge.utils.logging import get_logger

log = get_logger("skill_library.retriever")

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dim: int

    def encode(self, text: str) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder (no dependencies, offline).

    Each token is hashed to a bucket with a sign; the vector is L2-normalised.
    Cosine similarity of two texts grows with shared vocabulary — adequate for
    retrieval over short skill descriptions + scene summaries.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float64)
        for tok in _TOKEN.findall(text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 8) & 1 else -1.0
        n = np.linalg.norm(v)
        return (v / n) if n > 0 else v


class SentenceTransformerEmbedder:  # pragma: no cover - exercised only when opted in
    """Wraps a sentence-transformers model (opt-in via AORA_FORGE_USE_ST=1)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, text: str) -> np.ndarray:
        vec = self._model.encode([text], normalize_embeddings=True)[0]
        return np.asarray(vec, dtype=np.float64)


def get_embedder() -> Embedder:
    """Return the active embedder. Hashing by default; sentence-transformers only
    when ``AORA_FORGE_USE_ST=1`` (avoids surprise model downloads overnight)."""
    if os.environ.get("AORA_FORGE_USE_ST") == "1":
        try:
            emb = SentenceTransformerEmbedder()
            log.info("Using SentenceTransformerEmbedder (dim=%d).", emb.dim)
            return emb
        except Exception as exc:  # noqa: BLE001
            log.warning("sentence-transformers unavailable (%s); using HashingEmbedder.", exc)
    return HashingEmbedder()


class SkillRetriever:
    """Builds an in-memory embedding index over the store and serves retrieval."""

    def __init__(self, store: SkillStore, embedder: Embedder | None = None) -> None:
        self.store = store
        self.embedder = embedder or get_embedder()
        self._entries: list[SkillLibraryEntry] = []
        self._matrix: np.ndarray | None = None
        self._contexts: dict[str, SceneGraphContext] = {}

    def _index_text(self, entry: SkillLibraryEntry, ctx: SceneGraphContext | None) -> str:
        scene = (
            ctx.summary_text if ctx and ctx.summary_text else " ".join(entry.scene_object_labels)
        )
        return f"{entry.skill_name}. {entry.description}. {scene}"

    def build_index(self, *, persist_embeddings: bool = True) -> int:
        """(Re)build the embedding matrix from everything in the store."""
        self._entries = self.store.entries()
        vectors = []
        self._contexts = {}
        for e in self._entries:
            skill = self.store.load(e.skill_id)
            ctx = skill.scene_graph_context
            if ctx is not None:
                ensure_summary(ctx)
                self._contexts[e.skill_id] = ctx
            cached = self.store.load_embedding(e.skill_id)
            if cached is not None and cached.shape[0] == self.embedder.dim:
                vec = cached.astype(np.float64)
            else:
                vec = self.embedder.encode(self._index_text(e, ctx))
                if persist_embeddings:
                    self.store.write_embedding(e.skill_id, vec)
            vectors.append(vec)
        self._matrix = np.vstack(vectors) if vectors else None
        log.info("retriever index built over %d skill(s)", len(self._entries))
        return len(self._entries)

    def _cosine(self, q: np.ndarray) -> np.ndarray:
        assert self._matrix is not None
        qn = q / (np.linalg.norm(q) or 1.0)
        return self._matrix @ qn  # rows already normalised

    def retrieve(
        self,
        query: str | SceneGraphContext,
        *,
        top_k: int = 3,
        scene_graph_conditioned: bool = False,
        min_overlap: float = 0.0,
    ) -> list[tuple[SkillLibraryEntry, float]]:
        """Return up to ``top_k`` (entry, score) pairs, highest score first.

        ``query`` may be free text or a ``SceneGraphContext``. With
        ``scene_graph_conditioned=True`` and a context query, skills are first
        filtered by object-label overlap, then ranked by embedding similarity.
        """
        if self._matrix is None:
            self.build_index()
        if self._matrix is None:  # still empty
            return []

        if isinstance(query, SceneGraphContext):
            ensure_summary(query)
            q_text = query.summary_text
            q_ctx: SceneGraphContext | None = query
        else:
            q_text, q_ctx = query, None

        q_vec = self.embedder.encode(q_text)
        sims = self._cosine(q_vec)

        if scene_graph_conditioned and q_ctx is not None:
            # Object-overlap is the primary signal (which skills are *about* the
            # objects present), embedding similarity the tiebreak. This makes a
            # "clock" scene surface clock-relevant skills even when descriptions
            # share boilerplate vocabulary.
            overlaps = [
                context_overlap(
                    self._contexts.get(self._entries[i].skill_id, SceneGraphContext()), q_ctx
                )
                for i in range(len(self._entries))
            ]
            with_overlap = [i for i in range(len(self._entries)) if overlaps[i] > min_overlap]
            if with_overlap:
                ranked = sorted(with_overlap, key=lambda i: (-overlaps[i], -float(sims[i])))[:top_k]
                return [(self._entries[i], round(overlaps[i], 3)) for i in ranked]
            # nothing overlaps -> fall back to pure embedding ranking

        ranked = sorted(range(len(self._entries)), key=lambda i: -float(sims[i]))[:top_k]
        return [(self._entries[i], float(sims[i])) for i in ranked]
