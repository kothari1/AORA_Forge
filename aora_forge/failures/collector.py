"""Read ``FailureRecord``s from LEAD/AORA deployment logs.

The grounding for contribution C2: the curriculum comes from *these*, not from an
LLM's imagination. AORA_v1 writes an ``episodes.jsonl`` (one row per episode, with
a ``failure_mode`` field drawn from its closed enum) plus per-episode ``nav.jsonl``
/ ``nav.csv``. LEAD writes ``orchestrator.jsonl`` / ``nav.jsonl``. This collector
ingests the common shape and is forgiving about field names so a record logged by
either system maps cleanly onto our unified ``FailureRecord``.

It also round-trips our own records to/from JSONL, which the synthetic generator
and the demo use.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from aora_forge.failures.taxonomy import normalize_failure_mode
from aora_forge.schemas import (
    Embodiment,
    FailureMode,
    FailureObservations,
    FailureRecord,
)
from aora_forge.utils.logging import get_logger

log = get_logger("failures.collector")

# Field-name aliases accepted from heterogeneous log formats.
_INSTRUCTION_KEYS = ("task_instruction", "instruction", "task", "nl_instruction")
_QUERY_KEYS = ("target_query", "query", "goal", "object_category", "goal_object")
_ENV_KEYS = ("environment", "scene_id", "scene", "env_id")
_MODE_KEYS = ("failure_mode", "failure", "mode")
_NARRATIVE_KEYS = ("narrative", "reason", "summary", "explanation")


def _first(d: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def _obs_from_dict(d: dict[str, Any]) -> FailureObservations:
    """Pull known numeric signals out of a heterogeneous record dict."""
    raw = d.get("observations") or d.get("signals") or {}
    if not isinstance(raw, dict):
        raw = {}
    # also accept top-level signal fields
    merged = {**raw}
    for k in (
        "steps_used",
        "max_steps",
        "vlm_claimed_success",
        "done_accepted",
        "done_rejected_count",
        "dist_to_goal_final",
        "collided_step_fraction",
        "scan_count",
        "geodesic_progress_final_window",
        "target_ever_visible",
    ):
        if k in d:
            merged.setdefault(k, d[k])
    # tolerate only-known keys
    known = FailureObservations.model_fields.keys()
    return FailureObservations(**{k: v for k, v in merged.items() if k in known})


def record_from_dict(
    d: dict[str, Any], *, default_embodiment: Embodiment | None = None
) -> FailureRecord:
    """Build a ``FailureRecord`` from a heterogeneous log/episode dict."""
    emb_raw = d.get("embodiment")
    if emb_raw is not None:
        embodiment = Embodiment(emb_raw) if not isinstance(emb_raw, Embodiment) else emb_raw
    elif default_embodiment is not None:
        embodiment = default_embodiment
    else:
        embodiment = Embodiment.GROUND_HABITAT  # AORA's default; logs usually carry it

    mode_raw = _first(d, _MODE_KEYS, "timeout_other")
    mode: FailureMode = (
        mode_raw if isinstance(mode_raw, FailureMode) else normalize_failure_mode(str(mode_raw))
    )
    rec_id = str(
        d.get("record_id")
        or d.get("episode_id")
        or d.get("id")
        or f"rec_{abs(hash(json.dumps(d, sort_keys=True, default=str))) % 10_000_000}"
    )
    return FailureRecord(
        record_id=rec_id,
        embodiment=embodiment,
        environment=str(_first(d, _ENV_KEYS, "unknown")),
        task_instruction=str(_first(d, _INSTRUCTION_KEYS, "")),
        target_query=str(_first(d, _QUERY_KEYS, "")),
        failure_mode=mode,
        narrative=str(
            _first(d, _NARRATIVE_KEYS, f"{mode.value} on '{_first(d, _QUERY_KEYS, '')}'")
        ),
        observations=_obs_from_dict(d),
        episode_id=d.get("episode_id"),
        provenance=dict(d["provenance"]) if isinstance(d.get("provenance"), dict) else {},
    )


def collect_from_jsonl(
    path: str | Path, *, default_embodiment: Embodiment | None = None
) -> list[FailureRecord]:
    """Load failure records from a JSONL file (one record/episode per line).

    Rows whose ``failure_mode`` normalises to ``NONE`` (successes) are dropped —
    the curriculum is made of failures.
    """
    p = Path(path)
    records: list[FailureRecord] = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                log.warning("skipping unparseable line in %s", p)
                continue
            rec = record_from_dict(d, default_embodiment=default_embodiment)
            if rec.failure_mode is not FailureMode.NONE:
                records.append(rec)
    log.info("collected %d failure record(s) from %s", len(records), p)
    return records


def write_jsonl(records: Iterable[FailureRecord], path: str | Path) -> Path:
    """Persist records as JSONL (used by the synthetic generator and the demo)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")
    return p
