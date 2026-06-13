# AORA-Forge — ICRA 2027 Narrative

*Aditya Kothari · Stanford MSL · June 2026*

## Title

**AORA-Forge: Growing Cross-Embodiment Robot Skills from Real Deployment Failures
via 3D Gaussian-Splat Reconstruction**

## Elevator pitch (5 sentences)

Deployed language-guided robots fail in recurring, characteristic ways — declaring
arrival at a small target they never reached, mistaking a look-alike, looping a scan
without progress — and today those failures are discarded. AORA-Forge turns them into
curriculum: it clusters real logged failures into themes, reconstructs the failure
scenes in 3D Gaussian Splatting, and forges a specialised, validated skill for each
theme that the orchestrator can call on the next mission. Crucially, the same growth
mechanism produces skills for two morphologically distinct embodiments — a drone in a
Gaussian-splat flight simulator and a ground robot in Habitat HM3D — because the
cross-embodiment generality lives in an *embodiment-blind orchestration layer*, not in
a shared low-level policy or action space. We demonstrate that failure-driven,
reconstruction-grounded skill growth measurably reduces the deployed agent's
characteristic failure rates on held-out missions across both embodiments. AORA-Forge
is, to our knowledge, the first system to combine failure-derived curriculum, 3DGS
reconstruction as the training substrate, and embodiment-blind cross-embodiment skill
growth in one self-improvement loop.

## Contributions (each tied to a specific evaluation)

**C1 — Embodiment-blind self-improvement.**
*Claim:* one growth mechanism produces skills for the drone and the ground robot.
*Evaluation:* run `grow_from_failures` on each embodiment's failure logs through the
identical code path; report the fraction of grown skills whose `target_embodiments`
span both, and the per-embodiment held-out failure-rate reduction. *Ablation:* swap the
concrete `Embodiment` and confirm zero changes downstream (the code path is byte-identical;
asserted by `test_cluster_spans_both_embodiments`). *Delta vs. prior art:* unlike Yang et
al. 2024 (arXiv:2402.19432), generality requires **no shared goal-reaching interface and
no shared policy** (see `01_literature_synthesis.md` §3, C1).

**C2 — Failure-driven curriculum.**
*Claim:* the curriculum is real clustered deployment failures, not LLM-imagined tasks.
*Evaluation:* cluster N real LEAD/AORA failure logs into themes; measure (a) theme
coherence (human + LLM-judge agreement that members share a fixable cause), (b)
coverage (fraction of logged failures assigned to an actionable theme), and (c) the
held-out reduction in each clustered failure mode after the corresponding skill is
installed. *Delta vs. prior art:* AURA (arXiv:2506.02507) and RoboGen (arXiv:2311.01455)
both source curriculum from imagination; we source it from logs.

**C3 — 3DGS as the skill-forge substrate.**
*Claim:* reconstructing the failure scene in 3DGS and training the specialised skill
inside it beats training on a generic or domain-randomized substrate.
*Evaluation:* for perceptual/closed-loop skills, compare held-out success when the skill
is trained (i) in a 3DGS reconstruction of the failure scene vs. (ii) on generic
synthetic data vs. (iii) in a domain-randomized physics sim (the DrEureka substrate).
*Delta vs. prior art:* ReaDy-Go (arXiv:2602.11575) trains nav policies in a GS sim but
is single-embodiment, environment-specific, and synthetically populated — not
failure-conditioned (§1.3, §3, C3).

## Story arc — what the reviewer thinks, in order

1. *"Self-improving skill libraries — haven't I seen this? Voyager, SkillRL, AURA."*
   → Yes, and we cite them as the mechanism we build on; our novelty is **substrate and
   source**, stated up front.
2. *"Cross-embodiment is just co-training one policy."* → Not here — there is **no shared
   policy**; the invariant is the growth process. Figure 1 shows the identical code path
   forking only at the `Embodiment` tag.
3. *"3DGS for policy training exists (ReaDy-Go)."* → Right, and we show exactly what
   ReaDy-Go does *not* do (single embodiment, authored not failure-conditioned), and that
   our failure→reconstruct→specialise loop clears all three of C1/C2/C3 together.
4. *"Does it actually work?"* → The evaluation table: failure-rate reductions on held-out
   missions for both embodiments, plus the C3 substrate ablation.
5. *"Is the curriculum real?"* → Yes; it is mined from a real LEAD/AORA deployment, not
   prompted. That grounding is the paper's defensibility.

## Evaluation matrix

Headline table: **2 embodiments × 2 environments × 3 baselines**, with the number of
grown skills (N) and the failure-rate reduction per failure mode.

| Embodiment | Environment | Baseline (no growth) | + RoboGen-style imagined skills | + AORA-style imagined curriculum | **+ AORA-Forge (ours)** |
|---|---|---|---|---|---|
| Drone | FiGS `flightroom_ssv_exp` | FPR / SSR / TSR | … | … | **FPR↓, SSR↑** |
| Drone | FiGS 2nd scene | … | … | … | **…** |
| Ground | HM3D scene A | ObjectNav SR / SPL | … | … | **SR↑, SPL↑** |
| Ground | HM3D scene B | … | … | … | **…** |

Per-cell metrics: drone — False-Positive Rate (claimed-not-reached), Spatial Success
Rate (≤1.6 m), Task Success Rate; ground — ObjectNav Success Rate, SPL, and the
per-failure-mode rate (HALLUCINATED_DONE, SCAN_LOOP, etc.). N (skills grown) reported
per embodiment. The three baselines isolate the *source* of curriculum (none / imagined
sim-scene / imagined prompt-curriculum) against ours (real failures + 3DGS).

## Ablation table outline

| Ablation | Removes | Hypothesis (what should degrade) |
|---|---|---|
| No clustering (one skill per failure) | C2 theme structure | over-specific skills; poor held-out transfer |
| Imagined curriculum (LLM proposes tasks) | C2 grounding | lower coverage of *actual* failures |
| Generic substrate (no 3DGS) | C3 | perceptual skills generalise worse to the real scene |
| Domain-randomized sim substrate | C3 photoreal | sim-appearance gap; weaker on the held-out real scene |
| Single-embodiment growth (drone only) | C1 | no cross-embodiment skills; ground robot unaided |
| No scene-graph conditioning (embedding only) | retrieval key | wrong skill retrieved; smaller benefit |
| No execution-verification on code skills | trainer rigor | unrunnable/incorrect skills enter the library |

## Threats to validity (what reviewers will attack)

- **"This is RoboGen/AURA with extra steps."** Pre-empt by foregrounding that our
  curriculum source (logged failures) and substrate (failure-scene 3DGS) are exactly the
  two axes those systems do not touch, and by reporting the imagined-curriculum baseline.
- **Synthetic-failure dependence.** Tonight's pipeline runs on synthetic failures; the
  paper must run on *real* LEAD/AORA logs. The collector already ingests the real log
  format — the dependency is producing enough real failures, which the AORA_v1 HM3D runs
  generate.
- **3DGS reconstruction quality from drone footage** may be too low to train useful skills
  (DroneSplat-class problem). Mitigation: active view selection (GauSS-MI) and the option
  to fall back to a generic substrate for skills that don't need photoreal grounding.
- **Cross-embodiment is "easy" because both tasks are navigation.** Honest framing: the
  claim is about the *orchestration layer* being embodiment-blind, demonstrated on two
  genuinely different control regimes (continuous flight vs. discrete navmesh); a third
  embodiment (e.g. a manipulator skill) would strengthen it and is future work.
- **Skill validation is not deployment success.** Our `SkillValidation` is a proxy; the
  paper's headline numbers must be *deployed* held-out failure-rate reductions, not
  validation scores.
