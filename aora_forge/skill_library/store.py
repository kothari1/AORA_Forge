"""On-disk skill library.

Layout (one directory per skill):

    <root>/
      index.jsonl                     # one SkillLibraryEntry per line
      <skill_id>/
        skill.json                    # the full Skill model
        spec.json                     # the SkillSpec (also embedded in skill.json; kept for diffing)
        artifact.{prompt|py|json}     # the trained artifact bytes
        scene_graph_context.json      # the retrieval-key context
        embedding.npy                 # cached retrieval embedding (optional)

All writes are atomic (write to a temp file, then ``os.replace``), so a crashed
run never leaves a half-written skill the index points at.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from aora_forge.schemas import Skill, SkillLibraryEntry
from aora_forge.utils.logging import get_logger

log = get_logger("skill_library.store")

_ARTIFACT_NAME = {
    "prompt": "artifact.prompt",
    "python": "artifact.py",
    "classifier": "artifact.json",
}


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


class SkillStore:
    """Read/write the on-disk skill library rooted at ``root``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.jsonl"

    # ---- write ----------------------------------------------------------- #

    def _entry_for(self, skill: Skill) -> SkillLibraryEntry:
        ctx = skill.scene_graph_context
        return SkillLibraryEntry(
            skill_id=skill.skill_id,
            skill_name=skill.skill_name,
            skill_type=skill.skill_type,
            description=skill.spec.description,
            target_embodiments=skill.spec.target_embodiments,
            validation_score=skill.validation.score,
            scene_object_labels=[n.label for n in ctx.nodes] if ctx else [],
            created_at=skill.created_at,
            tags=[skill.skill_type.value, *(["validated"] if skill.validation.passed else [])],
        )

    def save(self, skill: Skill, embedding: np.ndarray | None = None) -> Path:
        """Persist a skill (atomically) and update the index."""
        # version bump if a skill with this id already exists
        sdir = self.root / skill.skill_id
        if sdir.exists():
            try:
                existing = self.load(skill.skill_id)
                skill.version = existing.version + 1
            except Exception:  # noqa: BLE001
                pass
        sdir.mkdir(parents=True, exist_ok=True)

        _atomic_write_text(sdir / "skill.json", skill.model_dump_json(indent=2))
        _atomic_write_text(sdir / "spec.json", skill.spec.model_dump_json(indent=2))
        if skill.artifact_inline is not None:
            artifact_name = _ARTIFACT_NAME.get(skill.artifact_kind, skill.artifact_ref)
            _atomic_write_text(sdir / artifact_name, skill.artifact_inline)
        if skill.scene_graph_context is not None:
            _atomic_write_text(
                sdir / "scene_graph_context.json",
                skill.scene_graph_context.model_dump_json(indent=2),
            )
        if embedding is not None:
            self.write_embedding(skill.skill_id, embedding)

        self._upsert_index(self._entry_for(skill))
        log.info("saved skill '%s' (v%d) -> %s", skill.skill_id, skill.version, sdir)
        return sdir

    def write_embedding(self, skill_id: str, embedding: np.ndarray) -> None:
        path = self.root / skill_id / "embedding.npy"
        # np.save appends ".npy" unless the name already ends in it, so the temp
        # name must end in ".npy" too or os.replace will miss the written file.
        tmp = path.with_name("embedding.tmp.npy")
        np.save(tmp, embedding.astype(np.float32))
        os.replace(tmp, path)

    def _upsert_index(self, entry: SkillLibraryEntry) -> None:
        entries = {e.skill_id: e for e in self.entries()}
        entries[entry.skill_id] = entry
        lines = [e.model_dump_json() for e in entries.values()]
        _atomic_write_text(self.index_path, "\n".join(lines) + ("\n" if lines else ""))

    # ---- read ------------------------------------------------------------ #

    def entries(self) -> list[SkillLibraryEntry]:
        if not self.index_path.exists():
            return []
        out: list[SkillLibraryEntry] = []
        for line in self.index_path.read_text().splitlines():
            line = line.strip()
            if line:
                out.append(SkillLibraryEntry.model_validate_json(line))
        return out

    def load(self, skill_id: str) -> Skill:
        path = self.root / skill_id / "skill.json"
        return Skill.model_validate_json(path.read_text())

    def load_embedding(self, skill_id: str) -> np.ndarray | None:
        path = self.root / skill_id / "embedding.npy"
        if path.exists():
            return np.load(path)
        return None

    def skills(self) -> list[Skill]:
        return [self.load(e.skill_id) for e in self.entries()]

    def __len__(self) -> int:
        return len(self.entries())
