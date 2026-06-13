"""Schema construction, serialization round-trips, and taxonomy behaviour."""

from __future__ import annotations

from aora_forge.failures.taxonomy import (
    classify_from_signals,
    normalize_failure_mode,
    suggest_skill_type,
)
from aora_forge.schemas import (
    Embodiment,
    FailureCluster,
    FailureMode,
    FailureObservations,
    FailureRecord,
    LLMUsage,
    OrchestratorTool,
    SkillIO,
    SkillType,
    ToolProvenance,
)


def _record(rid: str = "r1", mode: FailureMode = FailureMode.CLAIMED_NOT_REACHED) -> FailureRecord:
    return FailureRecord(
        record_id=rid,
        embodiment=Embodiment.DRONE_FIGS,
        environment="flightroom_ssv_exp",
        task_instruction="go to the green clock",
        target_query="green clock",
        failure_mode=mode,
        narrative="claimed done at 2.1 m from a small clock in clutter",
    )


def test_failure_record_roundtrip() -> None:
    r = _record()
    again = FailureRecord.model_validate_json(r.model_dump_json())
    assert again == r
    assert "green clock" in r.short()
    assert r.failure_mode is FailureMode.CLAIMED_NOT_REACHED


def test_failure_cluster_size() -> None:
    c = FailureCluster(
        cluster_id="cl0",
        title="small target FPR",
        description="x",
        hypothesized_root_cause="y",
        suggested_skill_type=SkillType.PROMPT,
        member_record_ids=["r1", "r2", "r3"],
    )
    assert c.size == 3


def test_normalize_failure_mode_aliases() -> None:
    assert normalize_failure_mode("claimed_not_reached") is FailureMode.CLAIMED_NOT_REACHED
    assert normalize_failure_mode("HALLUCINATED_DONE") is FailureMode.HALLUCINATED_DONE
    assert normalize_failure_mode("timeout") is FailureMode.TIMEOUT_OTHER
    # unknown -> catch-all, never raises
    assert normalize_failure_mode("totally_unknown_mode") is FailureMode.TIMEOUT_OTHER


def test_classify_from_signals() -> None:
    # accepted done but far -> claimed_not_reached
    obs = FailureObservations(done_accepted=True, dist_to_goal_final=2.3)
    assert classify_from_signals(obs) is FailureMode.CLAIMED_NOT_REACHED
    # accepted done and close -> success
    obs2 = FailureObservations(done_accepted=True, dist_to_goal_final=1.0)
    assert classify_from_signals(obs2) is FailureMode.NONE
    # high collision fraction -> stuck
    obs3 = FailureObservations(collided_step_fraction=0.5)
    assert classify_from_signals(obs3) is FailureMode.STUCK_AGAINST_WALL


def test_suggest_skill_type() -> None:
    assert suggest_skill_type(FailureMode.TARGET_MISIDENTIFICATION) is SkillType.CLASSIFIER
    assert suggest_skill_type(FailureMode.DONE_GATE_LOOP) is SkillType.CODE
    assert suggest_skill_type(FailureMode.WRONG_ROOM) is SkillType.PROMPT


def test_orchestrator_tool_renders() -> None:
    tool = OrchestratorTool(
        name="arrival_verifier",
        description="verify arrival",
        input_schema={"type": "object", "properties": {}},
        skill_type=SkillType.CODE,
        embodiments=[Embodiment.DRONE_FIGS, Embodiment.GROUND_HABITAT],
        integration="direct_tool",
        provenance=ToolProvenance(skill_id="s1"),
    )
    a = tool.to_anthropic_tool()
    assert a["name"] == "arrival_verifier" and "input_schema" in a
    lead = tool.to_lead_registration()
    assert lead["integration"] == "direct_tool"
    assert lead["embodiments"] == ["drone_figs", "ground_habitat"]


def test_llm_usage_add() -> None:
    a = LLMUsage(model="claude-opus-4-8", input_tokens=100, output_tokens=50, cost_usd=0.001)
    b = LLMUsage(model="claude-opus-4-8", input_tokens=10, output_tokens=5, cost_usd=0.0001)
    c = a + b
    assert c.input_tokens == 110 and c.output_tokens == 55
    assert abs(c.cost_usd - 0.0011) < 1e-9
    assert c.mocked is False


def test_skill_io_defaults() -> None:
    io = SkillIO(name="depth", type="float", description="depth in m")
    assert io.required is True
