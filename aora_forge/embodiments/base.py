"""The ``Embodiment`` abstract base class.

This is the *one* interface that lets a single skill-growth mechanism serve a
drone and a ground robot (C1). The contract is deliberately thin: an embodiment
declares who it is, what its action space looks like, and how to translate a raw
environment observation into the embodiment-neutral form the rest of the pipeline
consumes. The growth code (clustering, spec generation, training, retrieval)
takes ``EmbodimentSpec``/``EmbodimentObservation`` and never imports a concrete
embodiment — that is what keeps it blind.

Tonight the two concrete embodiments are interface-only stubs that point at the
real environments (LEAD's ``NavEnv`` and AORA_v1's ``HabitatEnv``); they expose
the metadata the pipeline needs (action space, tool surface, coordinate notes)
without importing the heavy native deps those environments require.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from aora_forge.schemas import Embodiment as EmbodimentTag
from aora_forge.schemas import SceneGraphContext


@dataclass
class EmbodimentSpec:
    """The embodiment-neutral description the growth pipeline reads."""

    tag: EmbodimentTag
    display_name: str
    control_mode: str  # "continuous" | "discrete"
    action_space: dict[str, str]  # tool/action name -> human description
    tool_surface: list[str]  # tool names the executor sees
    coordinate_notes: str  # frame conventions, for prompts/specs
    notes: str = ""


@dataclass
class EmbodimentObservation:
    """A single observation, normalised across embodiments.

    The pipeline never sees raw RGB/depth tensors — it sees this. Concrete
    embodiments fill ``rgb_ref``/``depth_ref`` with paths/handles; the actual
    pixel data stays in the environment process.
    """

    step: int
    position: list[float] | None = None
    heading_deg: float | None = None
    rgb_ref: str | None = None
    depth_ref: str | None = None
    status: str = "navigating"
    extras: dict[str, Any] = field(default_factory=dict)


class Embodiment(ABC):
    """Abstract embodiment. Concrete subclasses wrap a real environment."""

    #: The enum tag carried through every record/skill produced from this embodiment.
    tag: EmbodimentTag

    @abstractmethod
    def spec(self) -> EmbodimentSpec:
        """Return the embodiment-neutral description for the growth pipeline."""

    @abstractmethod
    def normalize_observation(self, raw: dict[str, Any]) -> EmbodimentObservation:
        """Translate a raw environment observation dict into the neutral form."""

    @abstractmethod
    def empty_scene_context(self) -> SceneGraphContext:
        """A scene context seeded with this embodiment's tag (no nodes yet)."""

    # Optional capability hooks; concrete envs override when available tonight
    # they raise NotImplementedError to make the stub boundary explicit.

    def reset(self) -> EmbodimentObservation:  # pragma: no cover - stub boundary
        raise NotImplementedError(
            f"{type(self).__name__}.reset requires the live environment "
            "(LEAD NavEnv / AORA HabitatEnv); not loaded in AORA-Forge."
        )

    def step(self, action: dict[str, Any]) -> EmbodimentObservation:  # pragma: no cover
        raise NotImplementedError(
            f"{type(self).__name__}.step requires the live environment; not loaded here."
        )
