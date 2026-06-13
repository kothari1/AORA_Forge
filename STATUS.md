# STATUS — read me first

*Overnight build of AORA-Forge, finished ~early morning. Author: Adi Kothari.*

## TL;DR

The full AORA-Forge MVP framework is **built, tested, type-checked, and pushed** to
<https://github.com/kothari1/AORA_Forge>. The end-to-end growth loop runs:
**real-shaped failures → cluster into themes → generate a SkillSpec → forge a
validated skill → store → retrieve as a planner tool**, with the *same code path*
serving the drone and the ground robot (C1), the curriculum driven by failures (C2),
and a 3DGS reconstruction handle on the substrate (C3, stubbed tonight). 42 tests
pass; ruff and mypy are clean; CI is wired.

**The one thing that needs you:** there was **no `ANTHROPIC_API_KEY` in the
environment overnight**, so the demo ran against a deterministic mock. The real
Anthropic code path is implemented and unit-tested with a fake client — it will work
the moment you set the key. See **#1** below and `BLOCKERS.md`.

## What to read first (in order)

1. **This file**, then `BLOCKERS.md` (what needs your input).
2. `docs/01_literature_synthesis.md` — the cited positioning (includes 3 corrections
   to the prompt's citations: AURA is 2506.02507/June-2025, GS-Planner is 2405.10142,
   SAGE/2512.17102 is a *digital* agent not manipulation).
3. `docs/02_architecture.md` — the system as built (every component → real module).
4. `docs/03_icra_narrative.md`, `docs/04_mvp_milestones.md`, `docs/05_risks.md` — pitch,
   12-week plan, risk register.
5. The code: start at `aora_forge/orchestrator_hooks/post_mission.py::grow_from_failures`
   (the whole thesis in one function), then `aora_forge/schemas.py`.

## What works (verified tonight — reproduce these)

```bash
cd /home/admin/projects/AORA_Forge
pip install -e .                       # already done in base conda env
python -c "import aora_forge"          # ✓ imports
pytest tests/ -q                       # ✓ 42 passed
ruff check aora_forge scripts tests    # ✓ clean
mypy aora_forge                        # ✓ clean (36 files)
python scripts/demo_full_loop.py --mock        # ✓ 50 failures → 9 validated skills → retrieval
python scripts/orchestrator_hook_demo.py --mock # ✓ planner gains grown tools (Tier-3 stretch A)
```

Concretely working, end to end:
- **Schemas** (`schemas.py`) — single source of truth; unified LEAD↔AORA failure taxonomy.
- **Failure ingestion** — collector reads the real AORA `episodes.jsonl`/`nav.csv` shape
  (forgiving aliases); deterministic classifier ports AORA_v1's `failure_modes.classify`.
- **Clustering (C2)** — LLM over narratives, deterministic offline fallback; cross-embodiment
  themes confirmed.
- **Spec generation** — embodiment-blind contracts; classifier specs request a 3DGS recon.
- **Three real trainers** — prompt (LLM-written + judged), **code (LLM-written, verified by
  executing typed probes, retry-on-failure)**, **classifier (a genuine 2-layer NumPy MLP head,
  no torch/GPU needed)**.
- **Skill library** — atomic on-disk store (one dir/skill + `index.jsonl`); versioning.
- **Retrieval** — scene-graph-conditioned (object-overlap primary, embedding tiebreak) over
  frequency-denoised keys; offline hashing embedder so it runs without sentence-transformers.
- **Orchestrator hooks** — skills → `OrchestratorTool` specs; the stub LEAD planner *gains
  tools that did not exist before growth* (the integration-point proof).
- **LLM layer** — real Anthropic client (forced-tool structured output, prompt caching, cost
  telemetry, 3-attempt validate-and-retry = LEAD's idiom) + deterministic mock; the real path
  is unit-tested with a fake client (`tests/test_anthropic_client.py`).

## What's stubbed / interface-only (by design tonight)

- **3DGS reconstruction (`skill_forge/reconstruction.py`)** — `StubReconstructor` returns canned
  `ReconstructionHandle`s with plausible metadata so downstream code runs. The `Reconstructor`
  ABC documents exactly what a real gsplat/nerfacto backend must implement. **This is C3's
  swap-in point; only this file changes.**
- **Embodiments (`embodiments/`)** — `spec()` (action space, frame, tool surface) is *accurate*
  and used by the pipeline; `reset()`/`step()` raise `NotImplementedError` (they need LEAD's
  `NavEnv` / AORA's `HabitatEnv`, which live in the other repos).
- **Scene-graph builder** — sparse graph from target query + narrative object mentions; real
  open-vocab RGB-D fusion (ConceptGraphs-style) is future work.
- **Failures are synthetic tonight** — grounded in LEAD's real modes, but generated, not logged.
  The collector already ingests the real log format; Week 1 wires it to AORA_v1's outputs.

## What's broken / needs your judgement

Nothing is broken. Items needing *your* input are in `BLOCKERS.md` — chiefly: set
`ANTHROPIC_API_KEY` and re-run the demo for the real-API proof; supply real failure
logs; and a few design decisions (which embodiment to demo first, sentence-transformers
opt-in).

## #1 — the real-API demo

Tonight: no key in env → demo ran in **MOCK** mode (deterministic, $0). The pipeline is
*identical* with a key; `get_llm_client()` auto-selects the backend. To get the real-API
proof:

```bash
export ANTHROPIC_API_KEY=sk-ant-...        # your key
python scripts/demo_full_loop.py           # now hits Opus 4.8 + Haiku 4.5
```

I did **not** hijack the Claude Code OAuth token for automated API calls (wrong tool for
the job, fragile, and you said the key would be in env). The real path is proven by
`tests/test_anthropic_client.py` (5 tests injecting a fake Anthropic response).

## Repo state

- Pushed to `main` at <https://github.com/kothari1/AORA_Forge> (public).
- `git log --oneline`: phased, single-author (Aditya Kothari), no external attribution.
- 42 tests green · ruff clean · mypy clean · CI workflow added (`.github/workflows/ci.yml`).
- ~40 Python modules in `aora_forge/`, 5 scripts, 10 test files, 6 docs (01–05 + transcript).

## What this run cost (rough estimate)

- **Background research:** 5 literature agents, ~**84K** subagent tokens total (logged from
  task notifications: 14.8K + 13.3K + 20.2K + 18.0K + 17.5K).
- **Main build session:** exact token count isn't available to me; order of magnitude is a
  few million cached-input tokens (large doc reads: LEAD report, lit review, the claude-api
  skill) and a few hundred K output tokens across ~40 files written.
- **The demo itself:** **$0** tonight (deterministic mock). A real-API demo run is estimated
  **~$0.50–0.80** at Opus 4.8 / Haiku 4.5 rates (1 Opus cluster call + ~9 Opus spec/train
  calls + ~24 Haiku validation calls), less with prompt caching.

## Suggested first moves (from `docs/04_mvp_milestones.md`, Week 1)

1. `export ANTHROPIC_API_KEY=...` and run `python scripts/demo_full_loop.py` — see the real
   clusters/specs/skills and the true token cost.
2. Point `failures/collector.py` at AORA_v1's `outputs/.../episodes.jsonl` to grow skills
   from *real* failures (`python scripts/grow_skill.py --in <real_failures.jsonl>`).
3. Read `docs/01` §4 (subsumption risks) — ReaDy-Go and AURA follow-ups are the threats;
   the plan front-loads the cross-embodiment result to counter them.
