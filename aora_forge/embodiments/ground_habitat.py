"""Ground-robot embodiment — AORA_v1's discrete agent in Habitat HM3D.

Interface-only tonight: action space, tool surface, and frame conventions are
sourced from ``AORA_v1/ARCHITECTURE.md`` §4 (the LEAD→Habitat embodiment-diff
table). The growth pipeline reads this ``spec``; the heavy ``habitat_sim`` import
stays in the AORA_v1 repo. AORA-Forge's outputs (grown skills, tool registrations)
are what AORA_v1 will eventually consume.
"""

from __future__ import annotations

from typing import Any

from aora_forge.embodiments.base import (
    Embodiment,
    EmbodimentObservation,
    EmbodimentSpec,
)
from aora_forge.schemas import Embodiment as EmbodimentTag
from aora_forge.schemas import SceneGraphContext


class GroundHabitatEmbodiment(Embodiment):
    """The AORA ground agent: discrete ObjectNav actions, navmesh-constrained."""

    tag = EmbodimentTag.GROUND_HABITAT

    def spec(self) -> EmbodimentSpec:
        return EmbodimentSpec(
            tag=self.tag,
            display_name="Ground robot (Habitat HM3D ObjectNav)",
            control_mode="discrete",
            action_space={
                "move": "forward_m only -> N x MOVE_FORWARD (0.25 m each); no strafe/vertical",
                "turn": "yaw_deg quantised to 30 deg -> TURN_LEFT/TURN_RIGHT; achieved yaw reported",
                "look_around": "render 6 frames at 60 deg spacing, offscreen (no step cost)",
                "capture_image": "save the current RGB frame as evidence",
                "check_target_depth": "P25 depth in a normalised bbox around the target",
                "done": "declare arrival (gated by an auto centre-depth check at 1.8 m)",
            },
            tool_surface=[
                "observe",
                "move",
                "turn",
                "look_around",
                "capture_image",
                "check_target_depth",
                "done",
            ],
            coordinate_notes=(
                "Frame: right-handed, Y-up, camera looks down -Z. Planner-facing positions "
                "are Habitat-native [x, y, z] (y=height); yaw_deg is heading about +Y, "
                "0 deg = -Z (Habitat forward), CCW positive. Collisions are hard "
                "(allow_sliding=False); OOB is impossible by construction (navmesh)."
            ),
            notes=(
                "Discrete ObjectNav action space (comparable to VLFM / the ObjectNav table). "
                "No baked CLIP heatmap; per-frame zero-shot detector hints are future work. "
                "STOP within 1.0 m geodesic of the goal object is the ObjectNav success criterion."
            ),
        )

    def normalize_observation(self, raw: dict[str, Any]) -> EmbodimentObservation:
        return EmbodimentObservation(
            step=int(raw.get("step", 0)),
            position=raw.get("position"),
            heading_deg=raw.get("yaw_deg"),
            rgb_ref=raw.get("rgb_ref"),
            depth_ref=raw.get("depth_ref"),
            status=raw.get("status", "navigating"),
            extras={"collided": raw.get("collided", False)},
        )

    def empty_scene_context(self) -> SceneGraphContext:
        return SceneGraphContext(embodiment=self.tag)
