"""The 3DGS reconstruction interface — contribution **C3**, stubbed tonight.

AORA-Forge's bet is that a real deployment-failure scene, reconstructed in 3D
Gaussian Splatting, is the right substrate to train a specialized closed-loop
skill against (vs. ReaDy-Go's synthetic-avatar augmentation, RoboSplat's
manipulation demo editing, or DrEureka's domain-randomized physics sim — see
``docs/01_literature_synthesis.md`` §1.3).

Tonight everything *around* reconstruction is real; reconstruction itself is a
believable stub so downstream code (the classifier/policy trainers, the renders a
closed-loop skill would consume) is exercisable. The ``Reconstructor`` interface
is the contract a real implementation (gsplat / nerfacto / an external service)
must satisfy.

### What a real implementation must do
1. ``reconstruct(frame_refs, ...)`` — take the RGB frames captured around a
   failure (``FailureRecord.representative_frame_ref`` and neighbours), run SfM +
   3DGS training, and return a ``ReconstructionHandle`` pointing at the trained
   splat (plus quality metrics: PSNR/SSIM, Gaussian count, bounds).
   *Active view selection (GauSS-MI, arXiv:2504.21067) and robust aerial GS
   (DroneSplat, arXiv:2503.16964) are the right front-ends here.*
2. ``render(handle, pose)`` — render an RGB (and depth) observation from an
   arbitrary camera pose inside the reconstruction, so a skill can be trained
   closed-loop by rolling out against rendered observations.
3. ``free(handle)`` — release GPU memory.

The stub honours (1) and (2) with canned outputs and records enough metadata that
the swap to a real reconstructor changes only this file.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from aora_forge.schemas import Embodiment
from aora_forge.utils.logging import get_logger

log = get_logger("skill_forge.reconstruction")


@dataclass
class ReconstructionHandle:
    """A handle to a (real or stubbed) 3DGS reconstruction of a failure scene."""

    reconstruction_id: str
    scene_id: str
    embodiment: Embodiment | None
    method: str  # "stub" | "gsplat" | "nerfacto" | "external"
    num_input_frames: int
    gaussian_count: int = 0
    psnr_db: float | None = None  # render quality vs. held-out frames
    is_stub: bool = True
    metadata: dict[str, str] = field(default_factory=dict)


class Reconstructor(ABC):
    """Interface a 3DGS reconstruction backend must satisfy."""

    @abstractmethod
    def reconstruct(
        self,
        frame_refs: list[str],
        *,
        scene_id: str,
        embodiment: Embodiment | None = None,
    ) -> ReconstructionHandle:
        """Build a reconstruction from frames captured around a failure."""

    @abstractmethod
    def render(self, handle: ReconstructionHandle, pose: list[float]) -> str:
        """Render an observation from a pose; returns an rgb_ref (path/uri)."""

    def free(self, handle: ReconstructionHandle) -> None:  # pragma: no cover - noop for stub
        """Release any resources held by the reconstruction."""
        return None


class StubReconstructor(Reconstructor):
    """A deterministic, dependency-free stub.

    Returns a canned ``ReconstructionHandle`` with plausible metadata so the
    classifier/policy trainers and any closed-loop rollout code can run end-to-end
    tonight. Clearly marked ``is_stub=True`` everywhere it surfaces.
    """

    def reconstruct(
        self,
        frame_refs: list[str],
        *,
        scene_id: str,
        embodiment: Embodiment | None = None,
    ) -> ReconstructionHandle:
        digest = hashlib.sha1((scene_id + "|".join(sorted(frame_refs))).encode()).hexdigest()[:10]
        n = max(len(frame_refs), 1)
        handle = ReconstructionHandle(
            reconstruction_id=f"recon_{digest}",
            scene_id=scene_id,
            embodiment=embodiment,
            method="stub",
            num_input_frames=len(frame_refs),
            gaussian_count=50_000 + 1000 * n,  # canned but pose-stable
            psnr_db=28.5,  # plausible indoor-GS PSNR
            is_stub=True,
            metadata={"note": "stub reconstruction; swap StubReconstructor for gsplat backend"},
        )
        log.info(
            "STUB reconstructed scene '%s' from %d frame(s) -> %s (%d gaussians)",
            scene_id,
            len(frame_refs),
            handle.reconstruction_id,
            handle.gaussian_count,
        )
        return handle

    def render(self, handle: ReconstructionHandle, pose: list[float]) -> str:
        pose_slug = "_".join(f"{x:.2f}" for x in pose)
        return f"stub://{handle.reconstruction_id}/render?pose={pose_slug}"


_DEFAULT = StubReconstructor()


def get_reconstructor() -> Reconstructor:
    """Return the active reconstructor (the stub tonight)."""
    return _DEFAULT
