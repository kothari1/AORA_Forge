"""The Embodiment interface — the keystone of contribution C1.

The whole point of AORA-Forge is that the skill-growth pipeline never branches on
which robot a failure came from. ``Embodiment`` is the *only* place embodiment
specifics are allowed to live, and even here they are confined to declaring an
action space and translating observations — never to the growth logic.
"""

from aora_forge.embodiments.base import Embodiment as EmbodimentBase
from aora_forge.embodiments.base import EmbodimentObservation, EmbodimentSpec
from aora_forge.embodiments.drone_figs import DroneFiGSEmbodiment
from aora_forge.embodiments.ground_habitat import GroundHabitatEmbodiment

__all__ = [
    "EmbodimentBase",
    "EmbodimentSpec",
    "EmbodimentObservation",
    "DroneFiGSEmbodiment",
    "GroundHabitatEmbodiment",
]
