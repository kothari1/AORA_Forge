# AORA-Forge — 12-Week MVP Plan (to the ICRA 2027 Abstract)

*Aditya Kothari · Stanford MSL · June 2026*

**Target:** ICRA 2027 abstract, assumed ~**September 2026**. **Resources:** one
RTX 4090 (jugg); ~**3 grad-student hours/day**; the parallel AORA_v1 build supplies the
real HM3D failure logs. **Constraint reality:** the 4090 caps what can be trained;
photoreal 3DGS reconstruction and any closed-loop skill training are the GPU-heavy steps
and must be scheduled, not assumed.

Each week names a deliverable, a **kill switch** (what to do if it fails), and feeds the
next. The framework built tonight is Week-0 complete: schemas, pipelines, trainers, store,
retrieval, demo, tests, CI.

| Week | Deliverable | Kill switch (fallback if it fails) |
|---|---|---|
| **1** | Wire the real collector to AORA_v1's `episodes.jsonl` + `nav.csv`; produce a real failure corpus (≥100 records across both embodiments). | If AORA_v1 hasn't generated enough real failures, run the existing executor-only HM3D batches harder; worst case, keep synthetic for the pipeline and flag the corpus as semi-synthetic. |
| **2** | Run real clustering with Claude Opus; human-audit theme coherence on the real corpus; lock the theme set for the first growth round. | If LLM clusters are incoherent, fall back to the deterministic mode→theme classifier (already built) and report it as the C2 baseline. |
| **3** | Forge prompt + code skills for the top 3–4 real themes; validate; install into a *stub* orchestrator hook; confirm the LEAD planner *sees* new tools. | If forging is unreliable, restrict to prompt skills only (most reliable) for the first milestone. |
| **4** | **Real-to-sim 3DGS: first reconstruction.** Reconstruct one drone failure scene from FiGS captures into a usable splat; render novel views. Replace `StubReconstructor` for this one scene. | If reconstruction quality is too low, use a pre-built FiGS/GemSplat scene as the substrate and defer in-the-wild reconstruction (DroneSplat-class) to later. |
| **5** | First **closed-loop skill trained in the reconstruction** (a perceptual disambiguation or arrival-verification skill) on the 4090; bound to ≤20 min/train. | If closed-loop training doesn't converge in budget, ship the NumPy/CLIP-head classifier skill (already built) trained on reconstruction-rendered features instead. |
| **6** | Drone evaluation harness: install grown skills into LEAD; measure held-out FPR/SSR/TSR vs. the no-growth baseline on `flightroom_ssv_exp`. | If the LEAD integration is fragile, evaluate offline from logs (LEAD's eval is already log-based) rather than live. |
| **7** | Repeat 4–6 for the **ground robot** in HM3D: reconstruct one HM3D failure scene, forge a skill, evaluate ObjectNav SR/SPL. **This closes C1.** | If HM3D reconstruction is hard, reuse the HM3D mesh-rendered observations as a proxy substrate and note the deviation. |
| **8** | Cross-embodiment result: one *shared* grown skill (e.g. arrival-verification) installed on **both** embodiments; report both deltas. | If no single skill transfers cleanly, report per-embodiment grown skills from the shared mechanism (still C1) and discuss transfer as future work. |
| **9** | C3 substrate ablation: 3DGS vs. generic-synthetic vs. domain-randomized for one perceptual skill. | If the ablation is inconclusive, narrow to 3DGS-vs-generic (drop the DR arm) to get a clean two-way comparison. |
| **10** | Scale: grow the full skill set (N≈8–12) across both embodiments; build the headline evaluation table (2×2×3). | If breadth is too costly, report the strongest 4–6 skills with full rigor rather than many shallow ones (log what was dropped). |
| **11** | Baselines: imagined-curriculum (AURA-style) and imagined-scene (RoboGen-style) arms on the same failure set; complete the table. | If full baseline reimplementation is infeasible, use a prompt-only "LLM invents the curriculum" ablation as the imagined baseline and state the simplification. |
| **12** | Write the abstract + figure 1 (the identical code path forking at the `Embodiment` tag) + the evaluation table; internal review; submit. | If results are thin, submit the C1+C2 story (mechanism + failure-driven curriculum, both demonstrated) and stage C3 photoreal results for the full paper. |

## The three single points of failure

1. **3DGS reconstruction quality (Weeks 4, 7).** If reconstructions of real failure
   scenes are too low-fidelity to train a useful skill, C3's headline weakens. *Early
   indicator:* PSNR/SSIM of held-out renders below a usable threshold by Week 4.
   *De-risk:* GauSS-MI active view selection; pre-built FiGS scenes as a fallback
   substrate; route prompt/code skills (which don't need photoreal) around it.
2. **Training a genuinely useful specialised skill on the 4090 in budget (Week 5).** The
   ≤20-min/train constraint is real on one GPU. *Early indicator:* a closed-loop skill not
   converging by Week 5. *De-risk:* the dependency-free NumPy/CLIP-head classifier (built
   tonight) is the always-available floor; prompt skills need no training at all.
3. **The evaluation harness producing trustworthy deployed deltas (Weeks 6–8).** Validation
   scores are not deployment success; the paper needs *deployed* held-out reductions.
   *Early indicator:* inability to install grown skills into LEAD/AORA and measure a clean
   before/after by Week 6. *De-risk:* both LEAD and AORA evaluate offline from logs, so the
   before/after can be measured without live re-integration if needed.

## Sequencing logic

The plan front-loads the **highest-risk, highest-defensibility** deliverable — the
cross-embodiment result (C1) and the first real 3DGS reconstruction (C3) — per the lit
synthesis (§4): each contribution alone is one follow-up paper from a competitor, so the
conjunction must be locked early. Weeks 1–3 reuse tonight's framework verbatim against
real logs; the GPU-bound reconstruction/training (Weeks 4–5, 7) is where the schedule is
tightest and the kill switches matter most.
