"""Single source of truth for every data structure that crosses a component
boundary in AORA-Forge.

Design notes
------------
* **Pydantic v2.** Every model validates on construction. The clustering and
  spec-generation pipelines parse raw LLM output straight into these models with
  a retry-on-``ValidationError`` loop — the same idiom LEAD uses for its
  ``OrchestratorOutput`` discriminated union (see
  ``lead-agent_system_report_v2.3.2.md`` §6.3). Porting that idiom is deliberate:
  it is the cheapest reliability mechanism we have.

* **Embodiment-blindness.** Nothing in these schemas is drone-specific or
  ground-robot-specific. ``Embodiment`` is an enum tag carried *as data*; the
  growth mechanism never branches on it. That is contribution **C1**.

* **Failure-driven, not LLM-imagined.** ``FailureRecord`` is grounded in the
  *actual* failure modes LEAD/AORA emit (``failures/taxonomy.py``), not in tasks
  an LLM dreamed up. That is contribution **C2**.

* **3DGS as the forge substrate.** ``SkillSpec.reconstruction`` and
  ``Skill.reconstruction_ref`` carry the (stubbed-tonight) handle to a 3D
  Gaussian-Splatting reconstruction of the failure scene. That is contribution
  **C3**; see ``skill_forge/reconstruction.py``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "0.1.0"


def utc_now() -> datetime:
    """Timezone-aware UTC timestamp. Centralised so tests can monkeypatch it."""
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Core enums (the embodiment tag + the failure taxonomy + skill kinds).
# `failures/taxonomy.py` imports FailureMode from here and adds the classifier
# logic and theme mapping — the enum itself lives here so the schemas have no
# upward dependency.
# --------------------------------------------------------------------------- #


class Embodiment(str, Enum):
    """The robot embodiments AORA-Forge serves. The pipeline is *blind* to which
    one a given record came from — this tag exists only so that retrieval and
    evaluation can be sliced per embodiment, never so the growth code can branch.
    """

    DRONE_FIGS = "drone_figs"  # LEAD's continuous drone in a FiGS/GemSplat scene
    GROUND_HABITAT = "ground_habitat"  # AORA_v1's discrete ground agent in HM3D


class FailureMode(str, Enum):
    """Unified failure taxonomy.

    Superset of LEAD's reported modes (``lead-agent_system_report`` §18:
    ``claimed_not_reached``, ``timeout``, OOB, target misidentification) and
    AORA_v1's ``aora/utils/failure_modes.py`` enum. The two vocabularies are
    reconciled here so a failure logged by either system maps onto one closed
    set. See ``failures/taxonomy.py`` for the deterministic classifier and the
    LEAD↔AORA alias table.
    """

    NONE = "NONE"  # success / not a failure
    # ---- false-positive arrivals (the headline LEAD FPR failure) ----
    CLAIMED_NOT_REACHED = "CLAIMED_NOT_REACHED"  # VLM said done(), GT distance too large
    HALLUCINATED_DONE = (
        "HALLUCINATED_DONE"  # AORA_v1 name for the same family (done accepted, wrong)
    )
    DONE_GATE_LOOP = "DONE_GATE_LOOP"  # repeatedly rejected by the depth gate
    # ---- target grounding ----
    TARGET_MISIDENTIFICATION = "TARGET_MISIDENTIFICATION"  # went to the wrong object
    INSTANCE_CONFUSION = "INSTANCE_CONFUSION"  # two instances match the query
    TARGET_NOT_VISIBLE = "TARGET_NOT_VISIBLE"  # target never entered view frustum
    WRONG_ROOM = "WRONG_ROOM"  # searched the wrong region
    # ---- exploration / motion pathologies ----
    SCAN_LOOP = "SCAN_LOOP"  # look_around loop with no displacement
    STUCK_AGAINST_WALL = "STUCK_AGAINST_WALL"  # repeated collisions (ground only)
    TIMEOUT_NO_PROGRESS = "TIMEOUT_NO_PROGRESS"  # step budget spent, no geodesic progress
    OOB = "OOB"  # out of bounds (drone only)
    # ---- orchestrator-level ----
    PLANNER_BUDGET_EXHAUSTED = "PLANNER_BUDGET_EXHAUSTED"
    PLANNER_ABORT = "PLANNER_ABORT"
    # ---- catch-alls ----
    TIMEOUT_OTHER = "TIMEOUT_OTHER"  # timed out, no signature matched
    LLM_ERROR = "LLM_ERROR"


class SkillType(str, Enum):
    """The kinds of skill AORA-Forge can forge.

    ``PROMPT`` and ``CODE`` are fully implemented tonight (most reliable first).
    ``CLASSIFIER`` is the tier-3 stretch (tiny CLIP head). ``POLICY`` is reserved
    for the 3DGS-trained closed-loop skills that plug in when reconstruction is
    real — the interface is here; the trainer is a documented stub.
    """

    PROMPT = "prompt"  # a specialised executor system prompt
    CODE = "code"  # a Voyager-style verified Python function
    CLASSIFIER = "classifier"  # a small learned head (e.g. CLIP-feature MLP)
    POLICY = "policy"  # a closed-loop policy trained in a 3DGS reconstruction (future)


# --------------------------------------------------------------------------- #
# Scene graph — the sparse structured context that conditions retrieval.
# --------------------------------------------------------------------------- #


class SceneGraphNode(BaseModel):
    """One object node in a sparse scene graph."""

    node_id: str
    label: str  # open-vocab object class, e.g. "green clock"
    attributes: dict[str, str] = Field(default_factory=dict)  # colour, size_class, material, ...
    position: list[float] | None = None  # [x, y, z] in the embodiment's frame, if known
    confidence: float = 1.0
    last_seen_step: int | None = None


class SceneGraphRelation(BaseModel):
    """A directed relation between two nodes, e.g. (clock, on, table)."""

    subject_id: str
    predicate: str  # on, near, left_of, inside, occluded_by, ...
    object_id: str
    confidence: float = 1.0


class SceneGraphContext(BaseModel):
    """The scene-graph context attached to a failure or queried at retrieval time.

    ``summary_text`` is the embedding key: a deterministic natural-language
    rendering of nodes + relations that ``skill_library/retriever.py`` encodes.
    Keeping the key explicit (rather than re-deriving it ad hoc) is what makes
    retrieval reproducible.
    """

    embodiment: Embodiment | None = None
    environment: str | None = None  # scene id, e.g. "flightroom_ssv_exp" or "hm3d:00800"
    nodes: list[SceneGraphNode] = Field(default_factory=list)
    relations: list[SceneGraphRelation] = Field(default_factory=list)
    summary_text: str = ""  # embedding key; filled by scene_graph/context.py

    def object_labels(self) -> list[str]:
        return [n.label for n in self.nodes]


# --------------------------------------------------------------------------- #
# Failure records — the grounded, real-deployment input to the whole pipeline.
# --------------------------------------------------------------------------- #


class FailureObservations(BaseModel):
    """Numeric signals extracted from a single failed episode's logs
    (``nav.jsonl`` / ``nav.csv`` / ``orchestrator.jsonl``). Mirrors the signal
    set AORA_v1's ``failure_modes.extract_signals`` produces, so a real episode
    record drops straight in.
    """

    steps_used: int | None = None
    max_steps: int | None = None
    vlm_claimed_success: bool | None = None
    done_accepted: bool | None = None
    done_rejected_count: int = 0
    dist_to_goal_final: float | None = None  # metres to GT centroid at episode end
    collided_step_fraction: float = 0.0
    scan_count: int = 0
    geodesic_progress_final_window: float | None = None  # fraction in [0, 1]
    target_ever_visible: bool | None = None


class FailureRecord(BaseModel):
    """One real deployment failure. The atomic unit of contribution **C2**.

    A ``FailureRecord`` is produced by ``failures/collector.py`` from the logs a
    LEAD/AORA mission emits. It deliberately keeps the *narrative* (a short
    natural-language account, typically lifted from the planner's
    ``thought_process``) alongside the structured signals, because the clusterer
    reasons over the narrative while evaluation reasons over the signals.
    """

    model_config = ConfigDict(extra="forbid")

    record_id: str
    embodiment: Embodiment
    environment: str  # scene id
    task_instruction: str  # the NL mission, e.g. "go to the green clock"
    target_query: str  # the object sought, e.g. "green clock"
    failure_mode: FailureMode
    narrative: str = Field(
        ...,
        description="Short natural-language account of what went wrong (1-3 sentences).",
    )
    observations: FailureObservations = Field(default_factory=FailureObservations)
    scene_context: SceneGraphContext | None = None
    representative_frame_ref: str | None = None  # path/uri to an RGB frame (for 3DGS later)
    episode_id: str | None = None
    timestamp: datetime = Field(default_factory=utc_now)
    provenance: dict[str, Any] = Field(
        default_factory=dict,
        description="Where this came from: log paths, line offsets, run id.",
    )

    def short(self) -> str:
        """One-line rendering used inside clustering prompts."""
        return (
            f"[{self.record_id}] {self.embodiment.value} / {self.environment} / "
            f"{self.failure_mode.value}: '{self.task_instruction}' → {self.narrative}"
        )


# --------------------------------------------------------------------------- #
# Failure clusters — themes discovered over the failure set (C2).
# --------------------------------------------------------------------------- #


class FailureCluster(BaseModel):
    """A theme discovered over the failure set by ``failures/clusterer.py``.

    Clusters are the *curriculum*: each one names a recurring way the deployed
    agent fails, and becomes the brief for one grown skill. Because the members
    are real ``FailureRecord``s, the curriculum is failure-driven (C2), not
    LLM-imagined.
    """

    model_config = ConfigDict(extra="forbid")

    cluster_id: str
    title: str  # short human-readable theme name
    description: str  # what unites these failures
    hypothesized_root_cause: str
    suggested_skill_type: SkillType
    member_record_ids: list[str] = Field(default_factory=list)
    representative_record_ids: list[str] = Field(default_factory=list)
    embodiments_involved: list[Embodiment] = Field(default_factory=list)
    failure_modes_involved: list[FailureMode] = Field(default_factory=list)
    priority: float = Field(
        default=0.0,
        description="Ranking score, typically size × severity; higher = forge first.",
    )

    @property
    def size(self) -> int:
        return len(self.member_record_ids)


# --------------------------------------------------------------------------- #
# Skill specs and skills — the output of the forge (C3 substrate handle lives here).
# --------------------------------------------------------------------------- #


class SkillIO(BaseModel):
    """One input or output port of a skill, as a typed, documented field."""

    name: str
    type: str = Field(..., description='Logical type, e.g. "str", "float", "image_b64", "bbox".')
    description: str
    required: bool = True


class ReconstructionSpec(BaseModel):
    """Handle to the 3DGS reconstruction that will serve as a skill's training
    substrate (contribution **C3**).

    Tonight this is *populated but stubbed*: ``skill_forge/reconstruction.py``
    returns a canned ``ReconstructionHandle`` so downstream code is exercisable.
    When real reconstruction lands, only that module changes.
    """

    needed: bool = False
    source_frame_refs: list[str] = Field(default_factory=list)  # RGB frames from the failure scene
    reconstruction_id: str | None = None  # set once reconstructed
    method: Literal["stub", "gsplat", "nerfacto", "external"] = "stub"
    notes: str = ""


class SkillSpec(BaseModel):
    """The contract for a skill, produced from a ``FailureCluster`` by
    ``skill_forge/spec_generator.py``.

    A ``SkillSpec`` is embodiment-blind by construction: ``target_embodiments``
    usually lists *both* embodiments, asserting the same grown skill should serve
    drone and ground robot alike (C1). The trainer reads this contract; the
    library indexes it; the orchestrator hook turns it into a tool.
    """

    model_config = ConfigDict(extra="forbid")

    spec_id: str
    skill_name: str = Field(..., description="snake_case identifier, unique in the library.")
    skill_type: SkillType
    description: str
    source_cluster_id: str
    target_embodiments: list[Embodiment]
    inputs: list[SkillIO] = Field(default_factory=list)
    outputs: list[SkillIO] = Field(default_factory=list)
    success_criterion: str = Field(
        ...,
        description="Natural-language pass condition the trainer validates against.",
    )
    training_data_needs: str = ""
    integration_point: str = Field(
        ...,
        description="How the planner would invoke this: a direct_tool name, an "
        "executor-prompt augmentation, or a dispatch strategy hint.",
    )
    rationale: str = ""
    reconstruction: ReconstructionSpec = Field(default_factory=ReconstructionSpec)


class SkillValidation(BaseModel):
    """The trainer's verdict on a forged skill."""

    passed: bool
    score: float = 0.0  # in [0, 1]
    n_cases_total: int = 0
    n_cases_passed: int = 0
    notes: str = ""
    per_case: list[dict[str, Any]] = Field(default_factory=list)


class Skill(BaseModel):
    """A forged, validated skill plus everything needed to store, retrieve, and
    register it. The artifact bytes (prompt text / Python source / weights) live
    on disk next to this metadata; ``artifact_ref`` points at them.
    """

    model_config = ConfigDict(extra="forbid")

    skill_id: str
    skill_name: str
    skill_type: SkillType
    spec: SkillSpec
    artifact_kind: Literal["prompt", "python", "classifier"]
    artifact_ref: str = Field(..., description="Relative path within the skill's library dir.")
    artifact_inline: str | None = Field(
        default=None,
        description="The artifact text inlined (prompt or code), when small enough to embed.",
    )
    validation: SkillValidation
    scene_graph_context: SceneGraphContext | None = None
    reconstruction_ref: str | None = None  # ReconstructionHandle id, when C3 is real
    embedding_ref: str | None = None  # relative path to embedding.npy
    version: int = 1
    created_at: datetime = Field(default_factory=utc_now)
    provenance: dict[str, Any] = Field(default_factory=dict)


class SkillLibraryEntry(BaseModel):
    """The compact index record persisted one-per-line in the library's
    ``index.jsonl``. Holds just enough to retrieve and rank without loading every
    skill's full ``Skill`` blob.
    """

    model_config = ConfigDict(extra="forbid")

    skill_id: str
    skill_name: str
    skill_type: SkillType
    description: str
    target_embodiments: list[Embodiment]
    validation_score: float
    scene_object_labels: list[str] = Field(
        default_factory=list,
        description="Object labels from the scene the skill was forged in; a coarse retrieval filter.",
    )
    created_at: datetime
    tags: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Orchestrator tool — the bridge back into a LEAD/AORA planner.
# --------------------------------------------------------------------------- #


class ToolProvenance(BaseModel):
    skill_id: str
    cluster_id: str | None = None
    record_ids: list[str] = Field(default_factory=list)


class OrchestratorTool(BaseModel):
    """A grown skill rendered as a tool the LEAD/AORA planner can see.

    ``integration`` says *how* the skill plugs into LEAD's ``OrchestratorOutput``
    discriminated union (``lead-agent_system_report`` §6.3):

    * ``direct_tool``       — a new named ``direct_tool`` the planner may call.
    * ``executor_prompt``   — a specialised system prompt the executor adopts for
                              the next ``dispatch_nav`` when the scene matches.
    * ``dispatch_strategy`` — a strategy hint attached to a ``dispatch_nav``.

    ``to_anthropic_tool`` emits the standard ``{name, description, input_schema}``
    tool shape; ``to_lead_registration`` emits the LEAD-facing record.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    input_schema: dict[str, Any]
    skill_type: SkillType
    embodiments: list[Embodiment]
    integration: Literal["direct_tool", "executor_prompt", "dispatch_strategy"]
    provenance: ToolProvenance

    def to_anthropic_tool(self) -> dict[str, Any]:
        """Render as an Anthropic Messages API tool definition."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def to_lead_registration(self) -> dict[str, Any]:
        """Render as a record LEAD's planner can register against its
        ``OrchestratorOutput`` action vocabulary.
        """
        return {
            "tool_name": self.name,
            "integration": self.integration,
            "description": self.description,
            "input_schema": self.input_schema,
            "embodiments": [e.value for e in self.embodiments],
            "provenance": self.provenance.model_dump(),
        }


# --------------------------------------------------------------------------- #
# LLM telemetry — used by llm/client.py for cost accounting.
# --------------------------------------------------------------------------- #


class LLMUsage(BaseModel):
    """Per-call token + cost telemetry."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd: float = 0.0
    mocked: bool = False

    def __add__(self, other: LLMUsage) -> LLMUsage:
        return LLMUsage(
            model=self.model if self.model == other.model else "mixed",
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens
            + other.cache_creation_input_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            mocked=self.mocked and other.mocked,
        )


# Re-export the public surface.
__all__ = [
    "SCHEMA_VERSION",
    "utc_now",
    "Embodiment",
    "FailureMode",
    "SkillType",
    "SceneGraphNode",
    "SceneGraphRelation",
    "SceneGraphContext",
    "FailureObservations",
    "FailureRecord",
    "FailureCluster",
    "SkillIO",
    "ReconstructionSpec",
    "SkillSpec",
    "SkillValidation",
    "Skill",
    "SkillLibraryEntry",
    "ToolProvenance",
    "OrchestratorTool",
    "LLMUsage",
]
