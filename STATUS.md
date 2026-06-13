# STATUS — read me first

*Overnight build of AORA-Forge, finished ~early morning. Author: Adi Kothari.*

## TL;DR

The full AORA-Forge MVP framework is **built, tested, type-checked, and pushed** to
<https://github.com/kothari1/AORA_Forge>. The end-to-end growth loop runs:
**real-shaped failures → cluster into themes → generate a SkillSpec → forge a
validated skill → store → retrieve as a planner tool**, with the *same code path*
serving the drone and the ground robot (C1), the curriculum driven by failures (C2),
and a 3DGS reconstruction handle on the substrate (C3, stubbed). 53 tests
pass; ruff and mypy are clean; CI is wired.

**The LLM stack now runs on all three providers you asked for — Claude (Anthropic
API), OpenAI, and Gemini (via Vertex AI)** — behind one `LLMClient` interface, picked
by `AORA_FORGE_PROVIDER` or the first reachable credential, else a deterministic mock.
**The demo has been verified END TO END against live Vertex Gemini** using your
`gcp-lead` service account (`gemini-2.5-pro`/`flash`): 20 failures → 3 clusters → 3
skills, ~$0.06, 205 s. The Anthropic and OpenAI paths are implemented and unit-tested
with injected fakes; they run the moment their keys are present. (Your SA key is stored
**only** in the gitignored `.secrets/gcp-lead-sa.json` — never committed.)

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
pytest tests/ -q                       # ✓ 53 passed
ruff check aora_forge scripts tests    # ✓ clean
mypy aora_forge                        # ✓ clean (41 files)
python scripts/demo_full_loop.py --mock                       # ✓ mock: 50 failures → 9 skills
AORA_FORGE_PROVIDER=vertex python scripts/demo_full_loop.py   # ✓ LIVE Gemini (your gcp-lead SA)
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
- **LLM layer (3 providers)** — Anthropic / OpenAI / Vertex Gemini behind one `LLMClient`
  (forced-tool or `response_schema` structured output, prompt caching where supported, cost
  telemetry, 3-attempt validate-and-retry = LEAD's idiom) + deterministic mock. **Vertex is
  proven live**; all three paths are unit-tested with injected fakes.

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

## #1 — running the real-API demo (all three providers)

```bash
pip install -e ".[providers]"                 # adds openai + google-genai (one-time)

# Gemini via Vertex — works now with your gcp-lead SA in .secrets/ (no extra env):
AORA_FORGE_PROVIDER=vertex python scripts/demo_full_loop.py

# Claude:
AORA_FORGE_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-ant-... python scripts/demo_full_loop.py
# OpenAI:
AORA_FORGE_PROVIDER=openai OPENAI_API_KEY=sk-... python scripts/demo_full_loop.py
```

With no `AORA_FORGE_PROVIDER` set, the provider is auto-detected (Anthropic → OpenAI →
Vertex by credential, else mock). The Vertex path is **proven live**; the Anthropic and
OpenAI paths are proven by `tests/test_anthropic_client.py` + `tests/test_providers.py`
(injected fake responses — structured parsing, validate-and-retry, fallback, telemetry).
I did **not** repurpose the Claude Code OAuth token for automated calls.

## Repo state

- Pushed to `main` at <https://github.com/kothari1/AORA_Forge> (public).
- `git log --oneline`: phased, single-author (Aditya Kothari), no external attribution.
- 53 tests green · ruff clean · mypy clean · CI workflow added (`.github/workflows/ci.yml`).
- ~45 Python modules in `aora_forge/`, 5 scripts, 11 test files, 6 docs (01–05 + transcript).

## What this run cost (rough estimate)

- **Background research:** 5 literature agents, ~**84K** subagent tokens total (logged from
  task notifications: 14.8K + 13.3K + 20.2K + 18.0K + 17.5K).
- **Main build session:** exact token count isn't available to me; order of magnitude is a
  few million cached-input tokens (large doc reads: LEAD report, lit review, the claude-api
  skill) and a few hundred K output tokens across ~40 files written.
- **The demo, measured live on Vertex Gemini:** **$0.056** for a 20-failure → 3-skill run
  (18 calls: 1 cluster + 3 spec + 3 code-train + 2 prompt-train + 9 validation; 15.5K input +
  4.4K output tokens; 205 s wall-clock on `gemini-2.5-pro`/`flash`). A full 50-failure run is
  proportionally larger. Claude (Opus 4.8 / Haiku 4.5) would be ~3–5× this at list rates.
  Mock mode is **$0**.

## Suggested first moves (from `docs/04_mvp_milestones.md`, Week 1)

1. `export ANTHROPIC_API_KEY=...` and run `python scripts/demo_full_loop.py` — see the real
   clusters/specs/skills and the true token cost.
2. Point `failures/collector.py` at AORA_v1's `outputs/.../episodes.jsonl` to grow skills
   from *real* failures (`python scripts/grow_skill.py --in <real_failures.jsonl>`).
3. Read `docs/01` §4 (subsumption risks) — ReaDy-Go and AURA follow-ups are the threats;
   the plan front-loads the cross-embodiment result to counter them.
