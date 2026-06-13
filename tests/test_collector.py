"""Failure collector: heterogeneous-dict parsing, JSONL round-trip, success filtering."""

from __future__ import annotations

from aora_forge.failures.collector import collect_from_jsonl, record_from_dict, write_jsonl
from aora_forge.schemas import Embodiment, FailureMode
from aora_forge.utils.synthetic_data import generate_synthetic_failures


def test_record_from_dict_aliases() -> None:
    # AORA-style episode row using alias field names
    d = {
        "episode_id": "ep_7_chair",
        "scene_id": "hm3d:00800-TEEsavR23oF",
        "instruction": "find the chair",
        "object_category": "chair",
        "failure": "hallucinated_done",
        "vlm_claimed_success": True,
        "dist_to_goal_final": 2.2,
    }
    rec = record_from_dict(d, default_embodiment=Embodiment.GROUND_HABITAT)
    assert rec.embodiment is Embodiment.GROUND_HABITAT
    assert rec.environment == "hm3d:00800-TEEsavR23oF"
    assert rec.target_query == "chair"
    assert rec.failure_mode is FailureMode.HALLUCINATED_DONE
    assert rec.observations.vlm_claimed_success is True
    assert rec.observations.dist_to_goal_final == 2.2


def test_jsonl_roundtrip_and_success_filter(tmp_path) -> None:
    records = generate_synthetic_failures(20, seed=11)
    path = write_jsonl(records, tmp_path / "f.jsonl")
    loaded = collect_from_jsonl(path)
    # all synthetic records are failures (not NONE) -> none filtered
    assert len(loaded) == len(records)
    assert {r.record_id for r in loaded} == {r.record_id for r in records}


def test_success_rows_are_dropped(tmp_path) -> None:
    p = tmp_path / "mixed.jsonl"
    p.write_text(
        '{"record_id": "ok1", "failure_mode": "NONE", "query": "chair"}\n'
        '{"record_id": "bad1", "failure_mode": "claimed_not_reached", "query": "clock"}\n'
    )
    loaded = collect_from_jsonl(p, default_embodiment=Embodiment.DRONE_FIGS)
    assert [r.record_id for r in loaded] == ["bad1"]


def test_unparseable_line_skipped(tmp_path) -> None:
    p = tmp_path / "broken.jsonl"
    p.write_text(
        'not json at all\n{"record_id": "good", "failure_mode": "scan_loop", "query": "bed"}\n'
    )
    loaded = collect_from_jsonl(p, default_embodiment=Embodiment.GROUND_HABITAT)
    assert [r.record_id for r in loaded] == ["good"]
