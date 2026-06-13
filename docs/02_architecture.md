# AORA-Forge — Architecture

*Aditya Kothari · Stanford MSL · June 2026*

This document describes the system as built. Every component named here exists in
`aora_forge/`; every arrow carries a Pydantic model from `aora_forge/schemas.py`
(the single source of truth). The 3DGS reconstruction backend is the one piece
stubbed tonight (`skill_forge/reconstruction.py`); everything around it is real.

## 1. Component diagram

```
   real deployment logs                         current scene (objects present)
 (LEAD nav.jsonl / orchestrator.jsonl,                       │
  AORA episodes.jsonl)                                       │
        │                                                    │
        ▼                                                    ▼
 ┌──────────────────┐   FailureRecord[]   ┌──────────────────────────────────┐
 │ FailureCollector │────────────────────▶│  SceneGraphBuilder / Context      │
 │ failures/        │                     │  scene_graph/                     │
 │   collector.py   │                     │  (sparse graph = retrieval key)   │
 │   taxonomy.py    │                     └──────────────────────────────────┘
 └──────────────────┘                                        │
        │ FailureRecord[]                                     │ SceneGraphContext
        ▼                                                     │
 ┌──────────────────┐   FailureCluster[]                      │
 │ FailureClusterer │  (themes; C2)                           │
 │ failures/        │                                         │
 │   clusterer.py   │── LLM (Opus) ──┐                        │
 └──────────────────┘                │                        │
        │ FailureCluster             │                        │
        ▼                            │                        │
 ┌──────────────────┐   SkillSpec    │                        │
 │ SkillSpecGenerator│── LLM (Opus) ─┤                        │
 │ skill_forge/      │               │                        │
 │   spec_generator  │               │                        │
 └──────────────────┘               │                        │
        │ SkillSpec                  │                        │
        ▼                            │                        │
 ┌──────────────────────────────┐   │  ┌──────────────────┐   │
 │ SkillForge (trainer_base +   │   │  │ Reconstructor    │   │
 │ trainers/)                   │◀──┼──│ skill_forge/     │   │
 │  prompt_skill_trainer (LLM)  │   │  │ reconstruction.py│   │
 │  code_skill_trainer  (LLM+   │   │  │  (3DGS; STUB)    │   │  C3
 │     execution verify)        │   │  └──────────────────┘   │
 │  classifier_trainer (NumPy   │   │                         │
 │     MLP head)                │   │                         │
 └──────────────────────────────┘   │                         │
        │ Skill (+ SkillValidation)  │                         │
        ▼                            │                         │
 ┌──────────────────┐   on disk      │                         │
 │ SkillLibrary     │  (atomic)      │                         │
 │ skill_library/   │                │                         │
 │   store.py       │                │                         │
 └──────────────────┘                │                         │
        │ SkillLibraryEntry[]         │                        │
        ▼                            │                         │
 ┌──────────────────┐                │                         │
 │ SkillRetriever   │◀───────────────┼─────────────────────────┘
 │ skill_library/   │  embedding + scene-graph-conditioned retrieval
 │   retriever.py   │
 └──────────────────┘
        │ Skill[]
        ▼
 ┌──────────────────┐   OrchestratorTool[]   ┌───────────────────────────────┐
 │ SkillRegistry +  │───────────────────────▶│  LEAD / AORA planner          │
 │ ToolRegistry     │  to_anthropic_tool()   │  (OrchestratorOutput union:   │
 │ skill_library/   │  to_lead_registration()│   dispatch_nav | direct_tool  │
 │ orchestrator_    │                        │   | capture_image | ...)      │
 │   hooks/         │                        └───────────────────────────────┘
 └──────────────────┘
```

The whole left column is wrapped by `orchestrator_hooks/post_mission.py::grow_from_failures`
— the one function that *is* the thesis: failures in, validated skills out, same
code path for both embodiments.

## 2. Components

| Component | Module | Responsibility | In | Out | Depends on |
|---|---|---|---|---|---|
| **FailureCollector** | `failures/collector.py` | Parse heterogeneous LEAD/AORA logs into `FailureRecord`s; round-trip JSONL | log dicts | `FailureRecord[]` | `taxonomy` |
| **FailureTaxonomy** | `failures/taxonomy.py` | Unify LEAD↔AORA failure-mode names; deterministic classifier; mode→theme/skill priors | signals / strings | `FailureMode`, `SkillType` | schemas |
| **FailureClusterer** | `failures/clusterer.py` | Cluster failures into themes (LLM over narratives; deterministic offline fallback) | `FailureRecord[]` | `FailureCluster[]` | `llm`, `taxonomy` |
| **SkillSpecGenerator** | `skill_forge/spec_generator.py` | Turn a cluster into a SkillSpec contract | `FailureCluster` | `SkillSpec` | `llm` |
| **SkillForge** | `skill_forge/trainer_base.py` + `trainers/` | Forge + validate a skill of the spec's type | `SkillSpec`, members | `Skill` | `llm`, `reconstruction`, `scene_graph` |
| **Reconstructor** | `skill_forge/reconstruction.py` | 3DGS reconstruction of a failure scene (**stub**) | frame refs | `ReconstructionHandle` | — |
| **SkillLibrary** | `skill_library/store.py` | Atomic on-disk persistence + index | `Skill` | `SkillLibraryEntry[]` | numpy |
| **SkillRetriever** | `skill_library/retriever.py` | Embedding + scene-graph-conditioned retrieval | context | `(entry, score)[]` | store, embedder |
| **SkillRegistry** | `skill_library/registry.py` | Skill → `OrchestratorTool` (planner tool spec) | `Skill` | `OrchestratorTool` | schemas |
| **SceneGraph** | `scene_graph/*` | Build/merge sparse graphs; render embedding key; overlap | observations / records | `SceneGraphContext` | schemas |
| **OrchestratorToolRegistry** | `orchestrator_hooks/tool_registry.py` | Scene-conditioned tool injection into the planner | `SceneGraphContext` | `OrchestratorTool[]` | retriever, registry |
| **PostMissionHook** | `orchestrator_hooks/post_mission.py` | The end-to-end growth loop | `FailureRecord[]` | `GrowthSummary` | all of the above |
| **Embodiment** | `embodiments/*` | The C1 keystone interface (drone / ground) | raw obs | `EmbodimentObservation`, `EmbodimentSpec` | schemas |
| **LLMClient** | `llm/client.py` | Anthropic SDK wrapper + deterministic mock | system+user | `(model, LLMUsage)` | anthropic |

## 3. Data flow — a failure becomes a tool

1. A deployed LEAD/AORA mission logs failures. `FailureCollector.collect_from_jsonl`
   reads them; `FailureTaxonomy.normalize_failure_mode` reconciles the names.
2. `FailureClusterer.cluster_failures` groups them into `FailureCluster` themes
   (Claude Opus reasons over the *narratives*; the deterministic fallback groups by
   the mode→theme prior). **This is the curriculum (C2).**
3. For each cluster, `SkillSpecGenerator.generate_spec` produces a `SkillSpec` — an
   embodiment-blind contract whose `target_embodiments` lists every embodiment the
   cluster touched. Perceptual/closed-loop specs set `reconstruction.needed=True`.
4. `get_trainer(spec.skill_type).train(...)` forges the skill:
   - **prompt**: Opus writes a specialised executor prompt; a Haiku judge validates
     it against held-out failure scenarios.
   - **code**: Opus writes a Python function; it is *executed* against typed probes
     and retried up to 3× on failure (a skill that doesn't run isn't stored).
   - **classifier**: a 2-layer NumPy MLP head trains on (tonight, synthetic;
     later, reconstruction-rendered) features; if the spec asks, a (stub)
     `ReconstructionHandle` is obtained first — **this is where C3 plugs in.**
5. `SkillLibrary.store.save` writes the skill atomically (one directory: `skill.json`,
   `spec.json`, the artifact, `scene_graph_context.json`, `embedding.npy`) and updates
   `index.jsonl`. The skill's retrieval key is the *frequency-denoised* union of the
   scenes it was forged from.
6. At mission time, the planner asks `OrchestratorToolRegistry.tools_for_context`
   with the current `SceneGraphContext`. `SkillRetriever` filters by object overlap
   then ranks by embedding similarity; `SkillRegistry.skill_to_tool` renders each hit
   as an `OrchestratorTool` that plugs into LEAD's `OrchestratorOutput` union — a new
   `direct_tool` (code/classifier) or an `executor_prompt` augmentation (prompt).

## 4. The Embodiment interface (C1)

`embodiments/base.py::Embodiment` is the **only** place embodiment specifics may
live. The growth pipeline consumes `EmbodimentSpec`/`EmbodimentObservation` and never
imports a concrete embodiment — that discipline is what makes the cross-embodiment
claim clean (and mirrors AORA's planner-prompt embodiment-leak rule, `ARCHITECTURE.md`
§5 in the AORA_v1 repo).

```python
class Embodiment(ABC):
    tag: schemas.Embodiment                      # DRONE_FIGS | GROUND_HABITAT
    def spec(self) -> EmbodimentSpec: ...        # control mode, action space, tool surface, frame
    def normalize_observation(self, raw) -> EmbodimentObservation: ...
    def empty_scene_context(self) -> SceneGraphContext: ...
    # reset()/step() raise NotImplementedError until the live env is wired in
```

The two concretes (`DroneFiGSEmbodiment`, `GroundHabitatEmbodiment`) declare *accurate*
specs sourced from the LEAD report (§6, §7, §17) and AORA `ARCHITECTURE.md` §4 — the
continuous drone (max 2 m/s, +Z-down frame, free-flight + OOB) vs. the discrete ground
agent (0.25 m forward / 30° turns, Y-up frame, collisions + navmesh). Their `spec()`
is what a real environment will satisfy; only `reset`/`step`/`normalize_observation`
gain live bodies later.

**Why this delivers C1:** swap the concrete embodiment, and the clusterer, spec
generator, trainers, store, and retriever run byte-identically. Nothing downstream of
the tag branches on it. The test `test_cluster_spans_both_embodiments` asserts a single
theme spans both robots.

## 5. Schema definitions (Pydantic v2 — `aora_forge/schemas.py`)

The authoritative definitions are in code; this is the reference map.

| Model | Key fields | Purpose |
|---|---|---|
| `Embodiment` (enum) | `DRONE_FIGS`, `GROUND_HABITAT` | the C1 tag carried as data |
| `FailureMode` (enum) | unified LEAD+AORA taxonomy (16 values) | closed failure vocabulary |
| `SkillType` (enum) | `PROMPT`, `CODE`, `CLASSIFIER`, `POLICY` | kinds of skill forged |
| `FailureObservations` | steps, dist_to_goal, done_rejected_count, collided_fraction, scan_count, geodesic_progress, target_ever_visible | numeric signals from logs |
| `FailureRecord` | id, embodiment, environment, task_instruction, target_query, failure_mode, **narrative**, observations, scene_context, representative_frame_ref, provenance | one real failure (C2 atom) |
| `FailureCluster` | id, title, description, hypothesized_root_cause, suggested_skill_type, member/representative ids, embodiments_involved, failure_modes_involved, priority | a theme (curriculum unit) |
| `SkillIO` | name, type, description, required | a typed skill port |
| `ReconstructionSpec` | needed, source_frame_refs, reconstruction_id, method | C3 substrate handle |
| `SkillSpec` | id, skill_name, skill_type, description, source_cluster_id, **target_embodiments**, inputs, outputs, success_criterion, integration_point, reconstruction | the skill contract |
| `SkillValidation` | passed, score, n_cases_total/passed, notes, per_case | the trainer's verdict |
| `Skill` | id, name, type, spec, artifact_kind, artifact_ref/inline, validation, scene_graph_context, reconstruction_ref, embedding_ref, version, provenance | a forged, validated skill |
| `SkillLibraryEntry` | id, name, type, description, target_embodiments, validation_score, scene_object_labels, created_at, tags | the compact index record |
| `SceneGraphNode` / `Relation` / `Context` | nodes (label, attrs, position), relations (subj, predicate, obj), summary_text | sparse retrieval key |
| `OrchestratorTool` | name, description, input_schema, skill_type, embodiments, integration, provenance + `to_anthropic_tool()` / `to_lead_registration()` | the planner-facing tool |
| `LLMUsage` | model, input/output/cache tokens, cost_usd, mocked | per-call telemetry |

## 6. LLM layer

All LLM calls go through `llm/client.py`. `ModelTier.PLANNER → claude-opus-4-8`
(clustering, spec generation, prompt/code authoring); `ModelTier.WORKER →
claude-haiku-4-5` (validation judging). Structured output is a forced single-tool
`tool_use` whose `input_schema` is the target Pydantic model's JSON schema, validated
and retried up to 3× — LEAD's `OrchestratorOutput` idiom. System prompts are
prompt-cached. Every call carries an `offline_fallback`, so `MockLLMClient` (and the
real client on a hard API failure) returns a valid, domain-aware result and the entire
pipeline runs without a key. `get_llm_client()` selects the backend.
