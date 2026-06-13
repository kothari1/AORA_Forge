# AORA-Forge — Risk Register

*Aditya Kothari · Stanford MSL · June 2026*

Top 10 risks, ranked by impact. L = likelihood, I = impact (both H/M/L).

| # | Risk | L | I | Mitigation | Early indicator |
|---|---|---|---|---|---|
| 1 | **A ReaDy-Go follow-up scoops C3** (adds a 2nd embodiment or failure-driven reconstruction to GS real-to-sim). | M | H | Lead with the C1∧C2∧C3 conjunction (they have none combined); lock the cross-embodiment result early (Week 8). Frame 3DGS as *failure-conditioned closed-loop training*, never as "we used Gaussian Splatting". | A new arXiv from the SNU group / GS-nav community adding embodiments or failure mining. |
| 2 | **"This is just RoboGen/AURA with extra steps"** reviewer rejection. | M | H | Foreground that our curriculum *source* (logged failures) and *substrate* (failure-scene 3DGS) are exactly the axes those systems don't touch; run the imagined-curriculum baseline so the delta is measured, not asserted. | Reviewer pre-prints / internal reviewers raising it; weak separation in the baseline arm. |
| 3 | **3DGS reconstruction from drone footage is too low quality** to train a useful skill (DroneSplat-class problem). | M | H | GauSS-MI active view selection; pre-built FiGS/GemSplat scenes as fallback substrate; route prompt/code skills (no photoreal need) around reconstruction. | Held-out render PSNR/SSIM below usable threshold by Week 4. |
| 4 | **No real failure corpus in time** (AORA_v1 doesn't generate enough real HM3D failures). | M | H | The collector already ingests the real log format; run executor-only HM3D batches harder; worst case proceed semi-synthetic and label it. | AORA_v1 still executor-only / sparse failures by Week 1. |
| 5 | **Cross-embodiment dismissed as trivial** because both tasks are navigation. | M | M | Honest framing: the claim is an embodiment-blind *orchestration layer*, demonstrated on two genuinely different control regimes; add a third (manipulator) embodiment as future work to strengthen. | Reviewers asking "why not one shared policy?". |
| 6 | **Closed-loop skill won't train on one 4090 in budget** (≤20 min). | M | M | The dependency-free NumPy/CLIP-head classifier (built tonight) is the always-available floor; prompt skills need no training; bound and abort non-converging trains. | A closed-loop train not converging by Week 5. |
| 7 | **Skill validation ≠ deployment success** — grown skills validate but don't move deployed metrics. | M | H | Headline numbers must be *deployed* held-out failure-rate reductions, not validation scores; evaluate offline from logs (LEAD/AORA already do) for clean before/after. | Validation-up but deployed-metrics-flat in Week 6 pilots. |
| 8 | **A frontier lab ships an embodiment-agnostic skill-growing orchestrator** (Gemini-Robotics-class), eroding C1's novelty. | L | H | C2+C3 are the durable moat — a frontier lab won't anchor on one student's logged drone failures + a Stanford-flight-room 3DGS; grounding is the defensibility. | A major-lab release of cross-embodiment skill growth. |
| 9 | **Retrieval picks the wrong grown skill** (scene graph + embedding insufficient to disambiguate failure mode from objects alone). | M | M | Scene-graph conditioning + frequency-denoised keys are in place; add failure-mode/context tags to the retrieval key; evaluate retrieval precision as an ablation. | Low retrieval precision in the no-conditioning ablation. |
| 10 | **3DGS / NeRF eval substrate becomes commodity** (S2E's NavBench-GS etc.), making "we used 3DGS" read as incremental. | M | M | Never claim 3DGS itself as novel; claim the failure→reconstruct→specialise loop. The substrate ablation (3DGS vs generic vs DR) is what carries C3. | 3DGS sim treated as standard in concurrent submissions. |

## Cross-cutting note

The honest bottom line from `01_literature_synthesis.md` §4: **no single paper subsumes
AORA-Forge, but each contribution alone is one follow-up away from a competitor.** The
risk register therefore concentrates mitigation on the *conjunction* (C1∧C2∧C3) and on
the *grounding* (real logged failures from a real deployment) — the two things hardest
for a competitor to replicate quickly and the project's primary defensibility.
