"""Drone embodiment — LEAD's continuous flying camera in a FiGS/GemSplat scene.

Interface-only tonight: this exposes the action space, tool surface, and frame
conventions the growth pipeline needs (sourced from
``AORA_v1/lead-agent_system_report_v2.3.2.md`` §6, §7, §17), without importing
``Semantic_HSM``/FiGS (the heavy native deps live in the LEAD repo). When the
live environment is wired in, only ``reset``/``step``/``normalize_observation``
gain real bodies; the ``spec`` is already accurate.
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


class DroneFiGSEmbodiment(Embodiment):
    """The LEAD drone: continuous single-integrator control, max 2.0 m/s."""

    tag = EmbodimentTag.DRONE_FIGS

    def spec(self) -> EmbodimentSpec:
        return EmbodimentSpec(
            tag=self.tag,
            display_name="Drone (FiGS / GemSplat flight room)",
            control_mode="continuous",
            action_space={
                "move": "translate by (forward_m, right_m, up_m) — single integrator, max 2.0 m/s",
                "turn": "rotate by yaw_deg (continuous)",
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
                "Frame: +X=East, +Y=North, +Z=DOWN (gravity +Z). Yaw 0 deg faces +X, "
                "positive yaw is CCW viewed from above. up_m>0 raises the drone "
                "(world Z decreases). State xcr[10] = [x,y,z,vx,vy,vz,qw,qx,qy,qz]."
            ),
            notes=(
                "Free-flying camera: no collisions, out-of-bounds terminates (check_bounds). "
                "Semantic signal: baked per-Gaussian CLIP heatmap (free per frame)."
            ),
        )

    def normalize_observation(self, raw: dict[str, Any]) -> EmbodimentObservation:
        return EmbodimentObservation(
            step=int(raw.get("step", 0)),
            position=raw.get("position"),
            heading_deg=raw.get("yaw_deg"),
            rgb_ref=raw.get("rgb_ref"),
            depth_ref=raw.get("depth_ref"),
            status=raw.get("status", "flying"),
            extras={"semantic_max": raw.get("semantic_max")},
        )

    def empty_scene_context(self) -> SceneGraphContext:
        return SceneGraphContext(embodiment=self.tag)
