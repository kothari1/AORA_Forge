"""Generate realistic synthetic ``FailureRecord``s for the demo and tests.

Grounded in LEAD's *actual* failure modes (``lead-agent_system_report_v2.3.2.md``
§6.6 false positives, §18 eval) and AORA's HM3D stress hypotheses: a mix of
``CLAIMED_NOT_REACHED`` / ``DONE_GATE_LOOP`` / ``TARGET_MISIDENTIFICATION`` /
``TARGET_NOT_VISIBLE`` / ``SCAN_LOOP`` / ``STUCK_AGAINST_WALL`` /
``TIMEOUT_NO_PROGRESS`` / ``OOB`` across five themes and **both** embodiments.

Target classes are *correlated* with themes (small/cluttered objects fail by
premature success; room-scale objects fail by multi-room mislocalisation;
look-alikes fail by misidentification), with ~20% cross-contamination for
realism. That correlation is what makes scene-graph-conditioned retrieval
meaningful — a "small clock" scene surfaces the small-target skill — while the
noise keeps it from being trivially separable.

Deterministic given a seed, so the demo and tests are reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from aora_forge.schemas import (
    Embodiment,
    FailureMode,
    FailureObservations,
    FailureRecord,
)

_DRONE_ENV = "flightroom_ssv_exp"
_GROUND_ENVS = ["hm3d:00800-TEEsavR23oF", "hm3d:00802-wcojb4TFT35", "hm3d:00813-svBbv1Pavdk"]

# All targets per embodiment (used for the ~20% cross-contamination draw).
_ALL_DRONE = [
    "green clock",
    "human mannequin",
    "yellow cordless drill",
    "red fire extinguisher",
    "green leafblower",
]
_ALL_GROUND = ["chair", "sofa", "bed", "potted plant", "tv monitor", "toilet"]


@dataclass
class _ThemeTemplate:
    theme: str
    modes: list[FailureMode]
    drone_targets: list[str]
    ground_targets: list[str]
    narratives: list[str] = field(default_factory=list)


_THEMES: list[_ThemeTemplate] = [
    _ThemeTemplate(
        theme="small cluttered target",
        modes=[
            FailureMode.CLAIMED_NOT_REACHED,
            FailureMode.DONE_GATE_LOOP,
            FailureMode.HALLUCINATED_DONE,
        ],
        drone_targets=["green clock", "yellow cordless drill", "red fire extinguisher"],
        ground_targets=["tv monitor", "potted plant"],
        narratives=[
            "Declared done() while {q} was still ~{d:.1f} m away; an obstacle at ~1 m satisfied the centre-depth gate.",
            "check_target_depth sampled foreground clutter, not the {q}; the auto gate rejected done() {k} times in a loop.",
            "VLM claimed arrival at the {q} but the ground-truth centroid was {d:.1f} m; the target is small and in a busy region.",
        ],
    ),
    _ThemeTemplate(
        theme="multi-room goal",
        modes=[
            FailureMode.WRONG_ROOM,
            FailureMode.TIMEOUT_NO_PROGRESS,
            FailureMode.TARGET_NOT_VISIBLE,
        ],
        drone_targets=["human mannequin"],
        ground_targets=["sofa", "bed", "toilet"],
        narratives=[
            "Searched the current room for the {q} but it was in an adjacent room; never crossed the doorway.",
            "Spent the whole step budget exploring without net geodesic progress toward the {q}.",
            "The {q} was never in view - it sits in a region the agent did not reach before timing out.",
        ],
    ),
    _ThemeTemplate(
        theme="look-alike disambiguation",
        modes=[FailureMode.TARGET_MISIDENTIFICATION, FailureMode.INSTANCE_CONFUSION],
        drone_targets=["green clock", "green leafblower"],
        ground_targets=["chair", "potted plant"],
        narratives=[
            "Two objects matched '{q}'; the agent approached the nearer/larger one, which was the wrong instance.",
            "Could not tell the queried {q} from a visually similar distractor and committed to the distractor.",
        ],
    ),
    _ThemeTemplate(
        theme="occlusion and scanning",
        modes=[FailureMode.SCAN_LOOP, FailureMode.TARGET_NOT_VISIBLE],
        drone_targets=["yellow cordless drill"],
        ground_targets=["chair", "bed"],
        narratives=[
            "look_around fired {k} times from nearly the same pose; the {q} was occluded and never resolved.",
            "The {q} was behind furniture; repeated scans without moving failed to reveal it.",
        ],
    ),
    _ThemeTemplate(
        theme="collision and boundary",
        modes=[FailureMode.STUCK_AGAINST_WALL, FailureMode.OOB],
        drone_targets=["red fire extinguisher", "human mannequin"],
        ground_targets=["sofa", "bed"],
        narratives=[
            "Repeated forward actions into a wall while approaching the {q}; {pct:.0%} of steps collided.",
            "Drifted out of bounds while repositioning toward the {q}; the run terminated OOB.",
        ],
    ),
]


def _pick_target(theme: _ThemeTemplate, embodiment: Embodiment, rng: random.Random) -> str:
    """Pick a target biased toward the theme (~80%), else any target (~20% noise)."""
    preferred = theme.drone_targets if embodiment is Embodiment.DRONE_FIGS else theme.ground_targets
    pool_all = _ALL_DRONE if embodiment is Embodiment.DRONE_FIGS else _ALL_GROUND
    if preferred and rng.random() < 0.8:
        return rng.choice(preferred)
    return rng.choice(pool_all)


def _observations_for(mode: FailureMode, rng: random.Random) -> FailureObservations:
    """Build signal values consistent with the failure mode."""
    obs = FailureObservations(steps_used=rng.randint(25, 140), max_steps=150)
    if mode in (FailureMode.CLAIMED_NOT_REACHED, FailureMode.HALLUCINATED_DONE):
        obs.done_accepted = True
        obs.vlm_claimed_success = True
        obs.dist_to_goal_final = round(rng.uniform(1.9, 2.6), 2)
    elif mode is FailureMode.DONE_GATE_LOOP:
        obs.done_rejected_count = rng.randint(3, 7)
        obs.dist_to_goal_final = round(rng.uniform(1.6, 2.4), 2)
    elif mode is FailureMode.STUCK_AGAINST_WALL:
        obs.collided_step_fraction = round(rng.uniform(0.32, 0.6), 2)
    elif mode is FailureMode.SCAN_LOOP:
        obs.scan_count = rng.randint(3, 6)
        obs.geodesic_progress_final_window = round(rng.uniform(0.0, 0.15), 3)
    elif mode is FailureMode.TARGET_NOT_VISIBLE:
        obs.target_ever_visible = False
    elif mode is FailureMode.TIMEOUT_NO_PROGRESS:
        obs.geodesic_progress_final_window = round(rng.uniform(0.0, 0.18), 3)
    elif mode is FailureMode.OOB:
        obs.dist_to_goal_final = round(rng.uniform(3.0, 6.0), 2)
    return obs


def generate_synthetic_failures(n: int = 50, seed: int = 0) -> list[FailureRecord]:
    """Generate ``n`` synthetic failure records across themes and both embodiments."""
    rng = random.Random(seed)
    records: list[FailureRecord] = []
    for i in range(n):
        embodiment = rng.choice([Embodiment.DRONE_FIGS, Embodiment.GROUND_HABITAT])
        env = _DRONE_ENV if embodiment is Embodiment.DRONE_FIGS else rng.choice(_GROUND_ENVS)
        theme = rng.choice(_THEMES)
        target = _pick_target(theme, embodiment, rng)

        mode = rng.choice(theme.modes)
        # OOB only for the drone; ground is navmesh-constrained -> swap to STUCK.
        if mode is FailureMode.OOB and embodiment is Embodiment.GROUND_HABITAT:
            mode = FailureMode.STUCK_AGAINST_WALL
        if mode is FailureMode.STUCK_AGAINST_WALL and embodiment is Embodiment.DRONE_FIGS:
            mode = FailureMode.OOB

        obs = _observations_for(mode, rng)
        narrative = rng.choice(theme.narratives).format(
            q=target,
            d=(obs.dist_to_goal_final or rng.uniform(2.0, 4.0)),
            k=(obs.done_rejected_count or obs.scan_count or 4),
            pct=(obs.collided_step_fraction or 0.4),
        )
        instruction = rng.choice(
            [
                f"navigate to the {target}",
                f"find the {target}",
                f"go to the {target} and stop in front of it",
            ]
        )
        records.append(
            FailureRecord(
                record_id=f"syn_{i:03d}",
                embodiment=embodiment,
                environment=env,
                task_instruction=instruction,
                target_query=target,
                failure_mode=mode,
                narrative=narrative,
                observations=obs,
                representative_frame_ref=f"frame://{env}/{target.replace(' ', '_')}/{i}.jpg",
                episode_id=f"ep_{i}_{target.split()[0]}",
                provenance={"source": "synthetic", "theme": theme.theme, "seed": seed},
            )
        )
    return records
