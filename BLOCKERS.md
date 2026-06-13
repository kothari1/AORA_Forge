# BLOCKERS — things that need Adi's input or judgement

None of these stopped the build; the framework is complete and tested. These are the
items that need *you* — a credential, a decision, or real data — before the next steps.

## 1. No `ANTHROPIC_API_KEY` in the overnight environment  ⟵ highest priority

- **What happened:** there was no `ANTHROPIC_API_KEY` (nor `ANTHROPIC_AUTH_TOKEN`) in the
  shell env, and the `anthropic` SDK was not installed (I installed it). The parallel AORA_v1
  run appears to have hit the same wall — its latest batch is `..._executor_only`.
- **Impact:** the demo and tests ran against the deterministic `MockLLMClient`. The real
  Anthropic path is fully implemented and unit-tested (`tests/test_anthropic_client.py`),
  but no *live* call was made tonight.
- **What I did NOT do (on purpose):** I did not repurpose the Claude Code OAuth token in
  `~/.claude/.credentials.json` for automated API calls — it's scoped for Claude Code, using
  it programmatically is fragile and arguably out of intent, and you said the key would be in
  env. If you *want* me to wire OAuth-token auth as a fallback, that's a one-line change in
  `AnthropicLLMClient.__init__` (the SDK takes `auth_token=` + the `oauth-2025-04-20` beta
  header) — but confirm first.
- **Action for you:** `export ANTHROPIC_API_KEY=sk-ant-...` then
  `python scripts/demo_full_loop.py`. That produces the real-API proof and the true token cost.

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
