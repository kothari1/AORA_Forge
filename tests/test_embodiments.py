"""Embodiments (C1 keystone): accurate specs, normalised observations, and that
the interface keeps the pipeline blind to which robot it serves."""

from __future__ import annotations

import pytest

from aora_forge.embodiments import DroneFiGSEmbodiment, GroundHabitatEmbodiment
from aora_forge.embodiments.base import Embodiment as EmbodimentABC
from aora_forge.schemas import Embodiment


def test_drone_spec_is_continuous() -> None:
    spec = DroneFiGSEmbodiment().spec()
    assert spec.tag is Embodiment.DRONE_FIGS
    assert spec.control_mode == "continuous"
    assert "check_target_depth" in spec.tool_surface
    assert "+Z=DOWN" in spec.coordinate_notes


def test_ground_spec_is_discrete() -> None:
    spec = GroundHabitatEmbodiment().spec()
    assert spec.tag is Embodiment.GROUND_HABITAT
    assert spec.control_mode == "discrete"
    assert "Y-up" in spec.coordinate_notes
    # both embodiments expose the SAME tool surface (semantically identical) -> C1
    assert spec.tool_surface == DroneFiGSEmbodiment().spec().tool_surface


def test_normalize_observation() -> None:
    drone = DroneFiGSEmbodiment()
    obs = drone.normalize_observation({"step": 5, "position": [1, 2, 3], "yaw_deg": 90})
    assert obs.step == 5 and obs.position == [1, 2, 3] and obs.heading_deg == 90

    ground = GroundHabitatEmbodiment()
    gobs = ground.normalize_observation({"step": 2, "collided": True})
    assert gobs.extras["collided"] is True


def test_empty_scene_context_tagged() -> None:
    assert DroneFiGSEmbodiment().empty_scene_context().embodiment is Embodiment.DRONE_FIGS
    assert GroundHabitatEmbodiment().empty_scene_context().embodiment is Embodiment.GROUND_HABITAT


def test_step_reset_are_stub_boundaries() -> None:
    # reset/step require the live env (LEAD NavEnv / AORA HabitatEnv) -> explicit NotImplementedError
    for emb in (DroneFiGSEmbodiment(), GroundHabitatEmbodiment()):
        with pytest.raises(NotImplementedError):
            emb.reset()
        with pytest.raises(NotImplementedError):
            emb.step({"move": 1.0})


def test_both_are_embodiment_subclasses() -> None:
    assert issubclass(DroneFiGSEmbodiment, EmbodimentABC)
    assert issubclass(GroundHabitatEmbodiment, EmbodimentABC)
