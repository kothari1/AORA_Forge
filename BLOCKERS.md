# BLOCKERS — things that need Adi's input or judgement

None of these stopped the build; the framework is complete and tested. These are the
items that need *you* — a credential, a decision, or real data — before the next steps.

## 1. LLM providers — Vertex works; Anthropic/OpenAI need keys (not blockers)

**Resolved:** the stack now runs on all three providers (Claude API, OpenAI, Gemini via
Vertex AI) behind one interface. The **Vertex path is verified live** with your `gcp-lead`
service account and is the default on jugg (no `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` are set,
so auto-detect picks Vertex). The Anthropic and OpenAI paths are implemented and unit-tested
with injected fakes; they activate the moment their keys are present:

```bash
AORA_FORGE_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-ant-... python scripts/demo_full_loop.py
AORA_FORGE_PROVIDER=openai    OPENAI_API_KEY=sk-...        python scripts/demo_full_loop.py
```

- **Credential security:** your GCP service-account JSON is stored **only** in
  `.secrets/gcp-lead-sa.json`, which is gitignored and was **never committed** (verified). The
  private key is not echoed anywhere in the repo. Note: the key is now also in this chat
  transcript — consider rotating it if that's a concern, since transcripts persist.
- **Model defaults** (override via `AORA_FORGE_<PROVIDER>_<TIER>_MODEL`): Vertex
  `gemini-2.5-pro`/`gemini-2.5-flash`; Anthropic `claude-opus-4-8`/`claude-haiku-4-5`; OpenAI
  `gpt-4o`/`gpt-4o-mini`. **Decision for you:** confirm the OpenAI model ids you want (I guessed
  `gpt-4o`/`gpt-4o-mini`; pricing in `llm/base.py` is approximate).
- **Known finding from the live run:** the code-skill trainer's execution-verifier is brittle
  to LLM-authored function signatures (one Gemini-written code skill verified to 0 probes and
  was correctly marked *weak* — the gate working as designed). Prompt skills (the reliable type
  by design) validated well. Worth hardening `code_skill_trainer._verify` to be more tolerant of
  varied signatures before relying on code skills in the paper.

## 2. No real failure corpus yet

- **What's needed:** the curriculum (C2) should come from *real* LEAD/AORA deployment failures.
  Tonight the pipeline runs on synthetic failures (grounded in LEAD's real modes, but generated).
- **Why it's not done:** AORA_v1's HM3D runs are still largely `executor_only`; a rich failure
  corpus across both embodiments isn't produced yet.
- **Action:** once AORA_v1 emits `episodes.jsonl` with `failure_mode` fields, run
  `python scripts/grow_skill.py --in <path/to/episodes.jsonl>` — the collector already ingests
  that format (forgiving field aliases). Decision for you: how many real failures before you
  trust the first growth round? (Week 1 in `docs/04` suggests ≥100.)

## 3. Design decisions for you

- **Which embodiment to demo first?** `docs/04` front-loads the drone (FiGS, where LEAD already
  has eval infra) then the ground robot (HM3D). Confirm or flip.
- **Embedding backend.** Retrieval defaults to a deterministic NumPy hashing embedder (offline,
  reproducible). For semantically stronger retrieval, set `AORA_FORGE_USE_ST=1` to use
  `sentence-transformers` (installed; downloads `all-MiniLM-L6-v2` on first use — needs
  internet). I left it opt-in to avoid a surprise download overnight. Your call for the paper.
- **Skill-type policy.** The mode→skill-type prior (`failures/taxonomy.py`) routes
  misidentification→classifier, gate-loops→code, strategy→prompt. Tune if you disagree with any
  mapping.

## 4. Stubbed-by-design (not blockers, but flagged so you know)

- **3DGS reconstruction** is a stub (`skill_forge/reconstruction.py`); the `Reconstructor` ABC
  documents what a real gsplat/nerfacto backend must do. This is intentional per the overnight
  scope ("stub the reconstruction interface; implement everything around it"). The heavy native
  deps (Habitat/NeRFStudio/gsplat) were deliberately *not* installed — the parallel AORA_v1 run
  owns Habitat.
- **Embodiment `reset`/`step`** raise `NotImplementedError` — they need LEAD's `NavEnv` / AORA's
  `HabitatEnv`. The `spec()` (action space, frame, tools) is accurate and is what the pipeline
  actually consumes.

## 5. Minor / nice-to-have

- The deterministic offline clusterer produces ~8–9 themes (one per failure-mode theme hint);
  a real Opus run will merge to fewer, cleaner clusters (the prompt asks for 3–6). Worth eyeballing
  the real clusters once you have a key.
- Retrieval quality is bounded by the hashing embedder offline; the scene-graph object-overlap
  conditioning carries most of the signal. Re-check precision with sentence-transformers + real
  scenes (see §3).
