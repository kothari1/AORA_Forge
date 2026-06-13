"""The unified failure taxonomy: classifier, LEAD↔AORA alias table, and the
mapping from failure mode to a *theme hint* and a *default skill type*.

The ``FailureMode`` enum itself is defined in ``aora_forge.schemas`` (so the
schemas have no upward dependency). This module owns the *behaviour*: how to
normalise the various names LEAD and AORA emit into that enum, how to classify a
failure from numeric signals (a deterministic port of AORA_v1's
``failure_modes.classify``), and how a failure mode suggests what *kind* of skill
would fix it.
"""

from __future__ import annotations

from aora_forge.schemas import FailureMode, FailureObservations, SkillType

# --------------------------------------------------------------------------- #
# LEAD/AORA name reconciliation. LEAD's report (§18) uses lowercase strings like
# "claimed_not_reached"/"timeout"; AORA_v1's failure_modes.py uses UPPER_SNAKE.
# Both map onto the unified FailureMode enum.
# --------------------------------------------------------------------------- #

_ALIASES: dict[str, FailureMode] = {
    # LEAD report names
    "claimed_not_reached": FailureMode.CLAIMED_NOT_REACHED,
    "timeout": FailureMode.TIMEOUT_OTHER,
    "oob": FailureMode.OOB,
    "wrong_target": FailureMode.TARGET_MISIDENTIFICATION,
    "target_misidentification": FailureMode.TARGET_MISIDENTIFICATION,
    # AORA_v1 failure_modes.py names
    "none": FailureMode.NONE,
    "hallucinated_done": FailureMode.HALLUCINATED_DONE,
    "done_gate_loop": FailureMode.DONE_GATE_LOOP,
    "wrong_room": FailureMode.WRONG_ROOM,
    "instance_confusion": FailureMode.INSTANCE_CONFUSION,
    "scan_loop": FailureMode.SCAN_LOOP,
    "stuck_against_wall": FailureMode.STUCK_AGAINST_WALL,
    "timeout_no_progress": FailureMode.TIMEOUT_NO_PROGRESS,
    "target_not_visible": FailureMode.TARGET_NOT_VISIBLE,
    "planner_budget_exhausted": FailureMode.PLANNER_BUDGET_EXHAUSTED,
    "planner_abort": FailureMode.PLANNER_ABORT,
    "llm_error": FailureMode.LLM_ERROR,
    "timeout_other": FailureMode.TIMEOUT_OTHER,
}


def normalize_failure_mode(raw: str) -> FailureMode:
    """Map any LEAD/AORA failure-mode string onto the unified enum.

    Falls back to ``TIMEOUT_OTHER`` for unrecognised strings (the same catch-all
    AORA_v1 uses), rather than raising — log ingestion should be forgiving.
    """
    key = raw.strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    # Already a canonical enum value?
    try:
        return FailureMode(raw.strip().upper())
    except ValueError:
        return FailureMode.TIMEOUT_OTHER


# --------------------------------------------------------------------------- #
# Deterministic classifier from numeric signals. Mirrors AORA_v1's
# failure_modes.classify precedence (terminal causes first, then behavioural
# signatures) but over our FailureObservations.
# --------------------------------------------------------------------------- #

# Thresholds, kept identical to AORA_v1/STRESS_TEST_PLAN where they overlap.
DONE_GATE_LOOP_MIN_REJECTIONS = 3
STUCK_COLLIDED_STEP_FRACTION = 0.30
SCAN_LOOP_MIN_SCANS = 3
NO_PROGRESS_MIN_FRACTION = 0.20


def classify_from_signals(obs: FailureObservations) -> FailureMode:
    """Classify a failure from extracted episode signals.

    This is the deterministic baseline; the LLM clusterer reasons over narratives,
    not these scalars. Keeping a pure classifier means the same record can be
    grouped two ways (signature vs. theme) and the two compared in evaluation.
    """
    # success
    if obs.done_accepted and obs.dist_to_goal_final is not None and obs.dist_to_goal_final <= 1.6:
        return FailureMode.NONE

    # terminal causes first
    if obs.done_accepted:
        # accepted done() but not actually close → the headline false-positive mode
        return FailureMode.CLAIMED_NOT_REACHED
    if obs.done_rejected_count >= DONE_GATE_LOOP_MIN_REJECTIONS:
        return FailureMode.DONE_GATE_LOOP

    # behavioural signatures
    if obs.collided_step_fraction >= STUCK_COLLIDED_STEP_FRACTION:
        return FailureMode.STUCK_AGAINST_WALL
    if obs.scan_count >= SCAN_LOOP_MIN_SCANS and (
        obs.geodesic_progress_final_window is not None
        and obs.geodesic_progress_final_window < NO_PROGRESS_MIN_FRACTION
    ):
        return FailureMode.SCAN_LOOP
    if obs.target_ever_visible is False:
        return FailureMode.TARGET_NOT_VISIBLE
    if (
        obs.geodesic_progress_final_window is not None
        and obs.geodesic_progress_final_window < NO_PROGRESS_MIN_FRACTION
    ):
        return FailureMode.TIMEOUT_NO_PROGRESS

    return FailureMode.TIMEOUT_OTHER


# --------------------------------------------------------------------------- #
# Failure mode → theme hint and default skill type. Used to seed the clusterer
# and the spec generator (and as the deterministic offline fallback).
# --------------------------------------------------------------------------- #

# A short natural-language theme each failure mode tends to belong to. The LLM
# clusterer may merge/split these; this is a prior, not a constraint.
FAILURE_THEME_HINTS: dict[FailureMode, str] = {
    FailureMode.CLAIMED_NOT_REACHED: "premature success on small or cluttered targets",
    FailureMode.HALLUCINATED_DONE: "premature success on small or cluttered targets",
    FailureMode.DONE_GATE_LOOP: "arrival-gate thrashing near the target",
    FailureMode.TARGET_MISIDENTIFICATION: "target disambiguation among look-alikes",
    FailureMode.INSTANCE_CONFUSION: "target disambiguation among look-alikes",
    FailureMode.WRONG_ROOM: "multi-room goal localisation",
    FailureMode.TARGET_NOT_VISIBLE: "search under occlusion / out-of-view targets",
    FailureMode.SCAN_LOOP: "unproductive scanning without progress",
    FailureMode.STUCK_AGAINST_WALL: "collision recovery in clutter",
    FailureMode.TIMEOUT_NO_PROGRESS: "long-horizon exploration stalls",
    FailureMode.OOB: "boundary / geofence handling",
    FailureMode.PLANNER_BUDGET_EXHAUSTED: "subtask budgeting on long missions",
    FailureMode.PLANNER_ABORT: "early give-up on solvable goals",
    FailureMode.TIMEOUT_OTHER: "generic timeout",
    FailureMode.LLM_ERROR: "transient backend errors",
    FailureMode.NONE: "no failure",
}

# Which skill *type* most naturally addresses a failure mode.
#   - PROMPT: a specialised executor system prompt (most reliable; reasoning/strategy fixes)
#   - CODE:   a verified Python helper (geometric / deterministic checks)
#   - CLASSIFIER: a learned head (perceptual disambiguation)
_DEFAULT_SKILL_TYPE: dict[FailureMode, SkillType] = {
    FailureMode.CLAIMED_NOT_REACHED: SkillType.PROMPT,
    FailureMode.HALLUCINATED_DONE: SkillType.PROMPT,
    FailureMode.DONE_GATE_LOOP: SkillType.CODE,
    FailureMode.TARGET_MISIDENTIFICATION: SkillType.CLASSIFIER,
    FailureMode.INSTANCE_CONFUSION: SkillType.CLASSIFIER,
    FailureMode.WRONG_ROOM: SkillType.PROMPT,
    FailureMode.TARGET_NOT_VISIBLE: SkillType.PROMPT,
    FailureMode.SCAN_LOOP: SkillType.CODE,
    FailureMode.STUCK_AGAINST_WALL: SkillType.CODE,
    FailureMode.TIMEOUT_NO_PROGRESS: SkillType.PROMPT,
    FailureMode.OOB: SkillType.CODE,
    FailureMode.PLANNER_BUDGET_EXHAUSTED: SkillType.PROMPT,
    FailureMode.PLANNER_ABORT: SkillType.PROMPT,
    FailureMode.TIMEOUT_OTHER: SkillType.PROMPT,
    FailureMode.LLM_ERROR: SkillType.CODE,
    FailureMode.NONE: SkillType.PROMPT,
}


def suggest_skill_type(mode: FailureMode) -> SkillType:
    """Default skill type for a failure mode (prior for the spec generator)."""
    return _DEFAULT_SKILL_TYPE.get(mode, SkillType.PROMPT)
