# AORA-Forge

**An embodiment-blind orchestrator that grows new robot skills from real deployment
failures via 3DGS real-to-sim reconstruction.**

AORA-Forge is the skill-growth engine behind LEAD/AORA. A deployed agent (a drone in
a FiGS/GemSplat scene, or a ground robot in Habitat HM3D) logs where it *actually*
fails. AORA-Forge clusters those failures into themes, turns each theme into a skill
specification, forges and validates a new skill, stores it, and registers it back into
the orchestrator's tool list — so the next mission has a capability the last one
lacked. The same mechanism runs for both embodiments; that is the point.

> Research target: **ICRA 2027.** This repository is the framework into which 3D
> Gaussian Splatting reconstruction plugs later. Tonight the reconstruction interface
> is stubbed and *everything around it is real and runnable*.

## The three claims

| | Claim | Where it lives |
|---|---|---|
| **C1** | **Embodiment-blind self-improvement** — one growth mechanism serves drone *and* ground robot. | `embodiments/`, `schemas.Embodiment` (a data tag the pipeline never branches on) |
| **C2** | **Failure-driven curriculum** — tasks come from real clustered deployment failures, not LLM-imagined curricula. | `failures/` |
| **C3** | **3DGS as the skill-forge substrate** — failure scenes are reconstructed in 3DGS and skills trained inside. | `skill_forge/reconstruction.py` (interface; stubbed tonight) |

See `docs/01_literature_synthesis.md` for the cited positioning against Voyager,
RoboGen, AURA, ReaDy-Go, RoboSplat, the cross-embodiment line (RT-X, π₀, Yang'24), and
the scene-graph thread.

## Pipeline

```
 real deployment logs (LEAD nav.jsonl / orchestrator.jsonl)
        │
        ▼
 FailureCollector ──▶ FailureClusterer ──▶ SkillSpecGenerator
   (failures/)          (LLM, themes)        (LLM, contract)
                                                    │
                                                    ▼
                          SkillForge ──────────────┤  (+ 3DGS reconstruction stub)
                          ├─ prompt_skill_trainer   │
                          ├─ code_skill_trainer     │
                          └─ classifier_trainer     │
                                                    ▼
                              SkillLibrary  ◀── store / retrieve / register
                                    │
                                    ▼
              SceneGraphRetriever ──▶ OrchestratorTool ──▶ LEAD/AORA planner
```

Every arrow carries a Pydantic model from `aora_forge/schemas.py` — the single source
of truth.

## Setup

```bash
# Option A — conda (recommended)
conda env create -f environment.yml
conda activate aora_forge
pip install -e .

# Option B — pip into an existing env (works without conda)
pip install -e .            # core: pydantic, anthropic, numpy
pip install -e ".[embed]"   # optional: sentence-transformers for semantic retrieval
```

`import aora_forge` should now work. The embedding retriever degrades gracefully to a
deterministic hashing embedding if `sentence-transformers` (or its model download) is
unavailable, so tests and the demo run fully offline.

## Usage

```bash
# End-to-end on synthetic failures: cluster → spec → train → store → retrieve.
# Uses the real Claude API if ANTHROPIC_API_KEY is set; otherwise runs a clearly
# flagged deterministic mock so the whole pipeline still executes offline.
python scripts/demo_full_loop.py

# Individual stages
python scripts/analyze_failures.py   # logs → clusters
python scripts/grow_skill.py         # cluster → spec → trained skill in the library
python scripts/eval_skill.py         # evaluate a stored skill against a holdout
```

## LLM backend

LLM calls go through `aora_forge/llm/client.py` (Anthropic SDK only). Planner-tier and
spec-generation calls default to **Claude Opus 4.8** (`claude-opus-4-8`); cheap
inner-loop calls (clustering helpers, validation judging) default to **Claude Haiku
4.5** (`claude-haiku-4-5`). Prompt caching is applied to system prompts and large
context; per-call token/cost telemetry is logged.

If `ANTHROPIC_API_KEY` is not set, the client falls back to a **deterministic
`MockLLMClient`** that produces structurally valid, theme-aware responses so the
architecture is provable end-to-end without a key. The mock is loud about being a mock
(every `LLMUsage.mocked == True`). See `STATUS.md` for tonight's run.

## Status (what's real vs. stubbed)

This is an overnight MVP build. `STATUS.md` is the authoritative, continuously-updated
snapshot. In brief: schemas, failure clustering, spec generation, prompt/code skill
training, the on-disk skill library, embedding + scene-graph retrieval, the
orchestrator tool registry, and the end-to-end demo are **implemented**; 3DGS
reconstruction, real HM3D/FiGS environments, and the tiny CLIP-head classifier trainer
are **interface-only or bounded stretch** (clearly marked in code and in `STATUS.md`).

## Layout

```
aora_forge/
  schemas.py            single source of truth (Pydantic v2)
  embodiments/          the Embodiment interface (C1)
  failures/             collector · taxonomy · clusterer (C2)
  skill_forge/          spec_generator · trainers/ · reconstruction (C3 stub)
  skill_library/        store · retriever · registry
  scene_graph/          builder · context · update
  orchestrator_hooks/   tool_registry · post_mission
  llm/                  client (Anthropic + deterministic mock) · prompts
  utils/                logging · synthetic_data
docs/                   literature synthesis + architecture + ICRA narrative + risks
scripts/                analyze_failures · grow_skill · demo_full_loop · eval_skill
tests/                  schema / clusterer / spec / library / retriever / demo
```

## Author

Aditya Kothari — Stanford MSL (`kothari1@stanford.edu`).

## License

MIT.
