# AORA-Forge — Literature Synthesis for the ICRA 2027 Pitch

*Aditya Kothari · Stanford MSL · June 2026*

> **Scope and relationship to prior survey.** This document deliberately does **not**
> re-tread `AORA_v1/LEAD_Literature_Review.md`, which already maps the
> *planner-over-skills / VLM-execution / mapless-navigation / agentic-orchestration*
> threads that LEAD lives in. It extends that survey along the three axes
> AORA-Forge actually contends on — **self-improving skill libraries**,
> **LLM-driven reward/data generation**, **real-to-sim 3DGS for policy training**,
> and **cross-embodiment** — plus the **scene-graph** thread that conditions
> retrieval. Where the LEAD survey's §12 gap analysis names items 1, 2, 3, 5 as
> "where LEAD is genuinely under-explored," this document turns three of those into
> defensible, cited contributions (C1, C2, C3).
>
> Every quoted claim below is verbatim from the cited paper's arXiv **abstract
> page** (not the PDF), pulled June 2026. Three of the prompt's working citations
> were corrected against the live record; the corrections are flagged inline and in
> §5.

AORA-Forge's thesis, stated once so the rest of the document can refer to it:

> **AORA-Forge: an embodiment-blind orchestrator that grows new skills from real
> deployment failures via 3DGS real-to-sim reconstruction, demonstrated across two
> embodiments (a drone in FiGS and a ground robot in HM3D ObjectNav).**
>
> - **C1 — Embodiment-blind self-improvement.** The *same* growth mechanism produces
>   skills for the drone *and* the ground robot.
> - **C2 — Failure-driven curriculum.** Tasks come from *real clustered deployment
>   failures*, not LLM-imagined curricula.
> - **C3 — 3DGS as the skill-forge substrate.** Failure scenes are reconstructed in
>   3D Gaussian Splatting and specialized skills are trained inside the
>   reconstruction.

---

## 1. The four threads

For each paper the operative question is the one the user asked: *what does this
work do that AORA-Forge does **not** need to redo, and what does it leave open?*

### 1.1 Skill libraries and LLM-driven task/reward/data generation

**Voyager** (Wang et al., arXiv:2305.16291). The canonical "ever-growing skill
library" agent. It "consists of three key components: 1) an automatic curriculum
that maximizes exploration, 2) an ever-growing skill library of executable code for
storing and retrieving complex behaviors, and 3) a new iterative prompting mechanism
that incorporates environment feedback, execution errors, and self-verification for
program improvement," and "interacts with GPT-4 via blackbox queries, which bypasses
the need for model parameter fine-tuning." *Don't redo:* the
store-as-retrievable-code + iterative-error-refinement loop — AORA-Forge ports this
idiom directly (`code_skill_trainer`). *Leaves open:* Voyager lives entirely in
Minecraft with ground-truth symbolic world state; its "failures" are in-game code
errors, its single embodiment is virtual, and it has **no perceptual substrate** — no
notion of reconstructing a failure scene. C2 (real failures) and C3 (3DGS substrate)
are exactly its blind spots.

**RoboGen** (Wang et al., arXiv:2311.01455, ICML 2024). The generative-simulation
data engine: "the agent first proposes interesting tasks and skills to develop, and
then generates corresponding simulation environments by populating pertinent objects
and assets with proper spatial configurations," yielding "an endless stream of skill
demonstrations." *Don't redo:* the propose→generate→learn scaffolding. *Leaves open:*
the tasks are **LLM-imagined** and the scenes **procedurally populated** — the polar
opposite of AORA-Forge's "reconstruct the *actual* scene where deployment *actually*
failed." RoboGen is the cleanest foil for C2 and C3 simultaneously: "this is RoboGen
with real failures and real reconstructions instead of imagination."

**Eureka** (Ma et al., arXiv:2310.12931, ICLR 2024) and **DrEureka** (Ma et al.,
arXiv:2406.01967, RSS 2024). Eureka "perform[s] evolutionary optimization over reward
code," outperforming human experts "on 83% of the tasks" across "29 open-source RL
environments that include 10 distinct robot morphologies." DrEureka extends this to
sim-to-real: it "automatically constructs suitable reward functions and domain
randomization distributions to support real-world transfer." *Don't redo:*
LLM-as-reward-designer and LLM-authored domain randomization. *Leaves open:* both are
**simulation-anchored** — Eureka writes rewards over simulator state variables (no
perception at all), DrEureka presupposes "only the physics simulation for the target
task" and never feeds *real failures* back into the loop. The "10 distinct
morphologies" are independent single-task RL setups, **not** a transferable
cross-embodiment library, and include **no aerial embodiment**. C1, C2, C3 untouched.

### 1.2 Recent self-improving agents (the closest competing claims, late-2025→2026)

**SAGE** ("Reinforcement Learning for Self-Improving Agent with Skill Library," Wang
et al., arXiv:2512.17102, Dec 2025). *Correction:* the prompt labeled this
"manipulation-focused"; the abstract is in fact a **digital software-agent** system
evaluated on **AppWorld**, not manipulation — worth getting right before a reviewer
does. SAGE introduces "Skill Augmented GRPO for self-Evolution," whose "Sequential
Rollout" runs "a chain of similar tasks" so that "skills generated from previous tasks
accumulate in the library and become available for subsequent tasks," reporting "8.9%
higher Scenario Goal Completion while requiring 26% fewer interaction steps." *Leaves
open:* no embodiment, no actuation, curriculum from a **benchmark task-chain** not from
failures. AORA-Forge beats it on every physical axis it never touches — and should
borrow its *quantitative rigor* (report an analogous efficiency story in physical
metrics).

**SkillRL** ("Evolving Agents via Recursive Skill-Augmented RL," Xia et al.,
arXiv:2602.08234, Feb 2026). The strongest competitor *on the learning-loop axis*. It
builds "a hierarchical skill library SkillBank" via "an experience-based distillation
mechanism," "an adaptive retrieval strategy for general and task-specific heuristics,
and a recursive evolution mechanism that allows the skill library to co-evolve with
the agent's policy during reinforcement learning," reaching SOTA on ALFWorld/WebShop
"outperforming strong baselines over 15.3%." *Implication for our framing:* we
**cannot** claim "hierarchical skills that co-evolve with the policy" as novel — they
have it. Our delta is purely **substrate and source**: SkillRL distills from clean
text/simulated trajectories on a single digital embodiment; AORA-Forge distills from
**real clustered failure modes** trained against **3DGS reconstructions** across
**two physical embodiments**.

**Agent Skills survey** (Xu & Yan, arXiv:2602.12430, Agent Skills '26 Workshop). A
**survey**, not a system: "agent skills — composable packages of instructions, code,
and resources that agents load on demand — enable dynamic capability extension without
retraining," organized along architecture / acquisition / deployment / security, and
noting "26.1% of community-contributed skills contain vulnerabilities." *Use, don't
beat:* cite it as the taxonomy AORA-Forge extends from disembodied CUA skills into
**physically grounded robot behaviors**, and adopt its security/governance framing as
the motivation for a *provenance + validation gate* on grown skills (we have one:
`SkillValidation`).

**AURA** ("Autonomous Upskilling with Retrieval-Augmented Agents," Zhu et al.,
arXiv:2506.02507, CoRL 2025 GenPriors workshop). *Correction:* the prompt placed this
in Dec 2025 / Feb 2026; the live record is **June 2025** (arXiv:2506.02507), and v1
carried the alternate title "Agentic Upskilling via Reinforced Abstractions." This is
the **closest robotics competitor** and must be differentiated carefully. AURA is "a
schema-validated curriculum reinforcement learning (RL) framework that leverages Large
Language Models (LLMs) as autonomous designers of multi-stage curricula," which
"transforms user prompts into YAML workflows that encode full reward functions, domain
randomization strategies, and training configurations," with "a retrieval-augmented
feedback loop … based on prior training results stored in a vector database." *Leaves
open — three concrete deltas:* (1) **failure source** — AURA's curriculum is
LLM-imagined *from user prompts* and refined on *sim reward curves*; ours is induced
from *real clustered deployment failures*; (2) **embodiment breadth** — AURA is a
single custom humanoid; ours spans drone + ground robot; (3) **training substrate** —
AURA uses a conventional randomized physics sim; ours uses **3DGS photoreal
reconstructions of the actual failure scenes**. Crucially, AURA *also* uses
retrieval-augmented vector-DB upskilling, so our novelty must be stated precisely as
"retrieval over **real failure clusters**, reconstructed in **3DGS**, transferred
across **heterogeneous embodiments**" — never merely "LLM-driven autonomous
upskilling," which AURA already owns.

### 1.3 Real-to-sim and 3DGS for policy training

**ReaDy-Go** ("Real-to-Sim Dynamic 3D Gaussian Splatting Simulation…," Yoo et al.,
arXiv:2602.11575, Feb 2026). **The nearest prior art** — characterize it exactly.
ReaDy-Go "synthesizes photorealistic dynamic scenarios in target environments by
augmenting a reconstructed static GS scene with dynamic human GS obstacles, and trains
navigation policies using the generated datasets," noting prior GS work "considered
only static scenes or non-photorealistic human obstacles." The precise contour:
**single embodiment** (a ground mobile robot), **environment-specific** (policies per
target environment, framed as "Environment-Specific Visual Navigation"), obstacles are
**synthetic animated human avatars from 2D trajectories** — *not* captured failures —
and there is **no failure-mining loop** deciding *what* to reconstruct. *Leaves open
for us:* (a) **no drone / GS-flight-sim** and no cross-embodiment story; (b) scene
difficulty is *authored* not *failure-conditioned*; (c) it trains one navigation
behavior per environment, not a library of *specialized closed-loop skills* for
specific failure modes. AORA-Forge's differentiators are exactly the three things
ReaDy-Go does not do: failure-conditioned reconstruction, specialized-skill training,
and dual embodiment. **This is the paper most likely to be cited against us — and the
one our C2+C3+C1 combination most cleanly clears.**

**RoboSplat** (Yang et al., arXiv:2504.13175, RSS 2025). "Generates diverse, visually
realistic demonstrations by directly manipulating 3D Gaussians," reconstructing "the
scene through 3D Gaussian Splatting (3DGS), … edit[ing] the reconstructed scene, and
augment[ing] data across six types of generalization," reaching "87.8% in one-shot
settings." *Leaves open:* manipulation only; **open-loop demonstration generation**,
not closed-loop rollout inside the reconstruction; single edited scene; no failure
trigger; no navigation, drone, or ground robot. AORA-Forge trains *closed-loop* skills
*against real failure scenes* — RoboSplat augments appearance/pose of demos.

**GauSS-MI** (Xie et al., arXiv:2504.21067) and **GS-Planner** (Jin et al.,
arXiv:2405.10142, IROS 2024 — *corrected ID*; the prompt gave no ID). Both are
**active-reconstruction** methods, not policy learners. GauSS-MI "formulate[s] a
criterion … for real-time assessment of visual mutual information from novel
viewpoints, facilitating the selection of next best view." GS-Planner is "a planning
framework for active high-fidelity reconstruction using 3D Gaussian Splatting," using
"quadrotor as the robotic platform" with "a safety constraint with 3DGS to generate
executable trajectories." *Leaves open:* neither trains a skill/policy — they decide
*where to look* (GauSS-MI) or fly a drone *to reconstruct* via classical planning
(GS-Planner). **They are candidate *components* of AORA-Forge's reconstruction
front-end** (active view selection to densify a failure scene), not competitors to the
forge.

**DroneSplat** (Tang et al., arXiv:2503.16964). "Robust 3D reconstruction from
in-the-wild drone imagery," removing "dynamic distractors" and "support[ing]
high-quality rendering under limited view constraints." *Leaves open:* the drone is a
camera, not an agent — no control, no closed loop, no skill. **Useful upstream** for
building a GS flight-sim from messy aerial captures; not a competitor.

### 1.4 Cross-embodiment

The recurring pattern across this whole thread: *cross-embodiment is located at the
single-shared-policy-weights level*, and the most morphologically diverse result buys
that diversity by collapsing every task into one shared action interface.

**Open X-Embodiment / RT-X** (O'Neill et al., arXiv:2310.08864, ICRA 2024). A model
"trained on this data, which we call RT-X, exhibits positive transfer and improves the
capabilities of multiple robots." But the 22 embodiments are all **manipulators**, and
transfer rests on a **shared Cartesian action space** plus **co-training one network**.
**One-shot pretraining, no skill-growth loop.**

**π₀** (Black et al., arXiv:2410.24164 — *corrected/confirmed ID*; RSS 2025). A
flow-matching VLA "trained on a large and diverse dataset from multiple dexterous robot
platforms, including single-arm robots, dual-arm robots, and mobile manipulators." It
has the closest thing to an orchestrator — "follow language instructions … from a
high-level VLM policy" — but that hierarchy still drives **one shared low-level VLA on
manipulation embodiments**, and skill growth is **per-skill supervised fine-tuning**.

**"Pushing the Limits of Cross-Embodiment Learning"** (Yang et al.,
arXiv:2402.19432). The strongest "one checkpoint on a drone *and* a ground robot"
result: "We train a single goal-conditioned policy that is capable of controlling
robotic arms, quadcopters, quadrupeds, and mobile bases," and "deploy our policy …
on a mobile manipulator … in a zero-shot manner." **But** the unifying trick is a
**shared goal-reaching interface** — everything is reframed as one goal-conditioned
task — and cross-embodiment is again **co-training one shared policy**. This is the
paper to cite as the cross-embodiment frontier, and to differentiate from precisely:
their generality *requires collapsing tasks to goal-reaching and sharing a low-level
policy*; **AORA-Forge requires neither a shared policy nor a shared action space** —
its cross-embodiment claim lives in the *embodiment-blind orchestration / skill-growth
layer*.

**From Seeing to Experiencing (S2E)** (He et al., arXiv:2507.22028). A navigation
foundation model with "RL post-training," introducing **NavBench-GS**, "a photorealistic
3D Gaussian-Splatting evaluation benchmark with physical interactions." It claims
generalization "across diverse environments and embodiments" but **does not enumerate
the embodiments** in the abstract — a weak cross-embodiment citation, but a useful
datapoint that *3DGS is becoming the standard navigation-eval substrate* (good for our
external validity). **H-Zero** (Lin et al., arXiv:2512.00971) pretrains "a
generalizable humanoid base policy" with "zero-shot and few-shot transfer to novel
humanoid robots" — but it is explicitly **cross-*humanoid***, i.e. transfer *depends on
shared morphology*, the opposite of embodiment-blindness.

### 1.5 Scene graphs as structured retrieval context

**SayPlan** (Rana et al., arXiv:2307.06135, CoRL 2023) shows an LLM can "conduct a
'semantic search' for task-relevant subgraphs from a smaller, collapsed representation
of the full graph." **ConceptGraphs** (Gu et al., arXiv:2309.16650) builds an
open-vocabulary 3D scene graph "by leveraging 2D foundation models and fusing their
output to 3D by multi-view association," queryable by "abstract natural-language
prompts." **CuriousBot** (Wang et al., arXiv:2501.13338, RA-L) grows "a 3D relational
object graph that encodes diverse object relations and enables exploration through
active interaction." *Don't redo:* the construction and collapse/semantic-search
machinery. *Leaves open — the precise transfer AORA-Forge makes:* all three use the
graph for *planning* or *exploration*; **none uses the sparse graph as a retrieval key
over a library of grown skills**. AORA-Forge indexes each skill by the graph-context it
was forged in, and retrieves by current-scene-graph overlap (`SceneGraphRetriever`).

---

## 2. The unclaimed territory

Read the threads together and a single white space appears, bounded on four sides:

1. **Skill-library / self-improvement (Voyager, SAGE, SkillRL, AURA, Agent-Skills)**
   gives us the *mechanism* — grow, store, retrieve, co-evolve skills — but anchors it
   to **clean simulators, text benchmarks, or LLM-imagined curricula**, and (except
   AURA) to **disembodied or single-embodiment** agents. None mines **real deployment
   failures**.
2. **LLM reward/data generation (RoboGen, Eureka, DrEureka)** gives us *automatic
   supervision*, but **in simulation, from imagined or pre-authored tasks**, with **no
   perceptual reconstruction** of where things actually broke.
3. **Real-to-sim 3DGS (ReaDy-Go, RoboSplat, GauSS-MI, GS-Planner, DroneSplat)** gives
   us the *photoreal substrate*, but **never failure-conditioned, never closed-loop +
   multi-skill + multi-embodiment all at once.** ReaDy-Go trains nav policies in a GS
   sim but is single-embodiment, environment-specific, and synthetically populated.
4. **Cross-embodiment (RT-X, π₀, Yang'24, S2E, H-Zero)** gives us *transfer*, but
   **only at the shared-policy-weights level**, requiring a shared action space or
   shared morphology.

The unclaimed cell is the intersection: **a self-improvement loop whose curriculum is
real clustered failures (C2), whose training substrate is a 3DGS reconstruction of the
failure scene (C3), and whose cross-embodiment generality lives in an embodiment-blind
*orchestration* layer rather than a shared low-level policy (C1).** No surveyed system
occupies more than two of those three cells; the nearest (AURA, ReaDy-Go) each occupy
exactly one-and-a-half.

A second, sharper way to state the white space: every cross-embodiment result above
makes robots *similar* (shared action space / shared morphology / shared goal
interface) so one policy can span them. AORA-Forge instead lets robots stay *different*
and makes the **skill-growth process** the invariant. That is a genuinely different
bet, and it is unoccupied.

---

## 3. Direct competitor map (per contribution)

### C1 — Embodiment-blind self-improvement

| Closest system | What it does | Delta vs. AORA-Forge |
|---|---|---|
| Yang et al. 2024 (arXiv:2402.19432) | One goal-conditioned policy on arm+quadcopter+quadruped+mobile base | Cross-embodiment via **shared goal-reaching action interface + co-trained single policy**. AORA-Forge needs **no shared policy and no shared action space**; the *growth mechanism* is the invariant. |
| RT-X / π₀ (2310.08864 / 2410.24164) | Co-trained generalist on 22 arms / dexterous platforms | Single morphology class; one-shot pretraining, **no skill-growth loop**. |
| H-Zero (2512.00971) | Cross-humanoid locomotion pretraining | Transfer **requires shared humanoid morphology** — antithetical to embodiment-blindness. |

### C2 — Failure-driven curriculum

| Closest system | What it does | Delta vs. AORA-Forge |
|---|---|---|
| AURA (2506.02507) | LLM designs curricula from **user prompts**, refined on sim reward curves via vector-DB retrieval | Curriculum is **imagined**, not failure-derived; single humanoid; physics-sim substrate. AORA-Forge's curriculum is **clustered real failures**. |
| RoboGen (2311.01455) | LLM **proposes** tasks + generates sim scenes | "Endless stream" of **imagined** tasks/scenes — the explicit antithesis of failure-grounding. |
| Voyager (2305.16291) | Automatic curriculum from in-game exploration | "Failures" are **in-game code errors** with ground-truth state, not real deployment failures. |

### C3 — 3DGS as the skill-forge substrate

| Closest system | What it does | Delta vs. AORA-Forge |
|---|---|---|
| ReaDy-Go (2602.11575) | Trains nav policies in a static-GS scene + **synthetic** human GS obstacles | **Single embodiment**, environment-specific (not failure-specific), obstacles **authored not captured**, **one policy not a skill library**. AORA-Forge: failure-conditioned reconstruction → specialized closed-loop skills → two embodiments. |
| RoboSplat (2504.13175) | Edits Gaussians to synthesize **manipulation demos** | **Open-loop** demo augmentation, manipulation only, no failure trigger, no closed-loop rollout in the reconstruction. |
| GauSS-MI / GS-Planner (2504.21067 / 2405.10142) | Active 3DGS reconstruction (NBV / quadrotor coverage planning) | **Reconstruction only — no skill learned.** Candidate *front-end components*, not competitors. |

---

## 4. Risks to the contribution (papers that could subsume us in 3–6 months)

Ranked by how much each would hurt.

1. **A ReaDy-Go follow-up that adds (a) a second embodiment or (b) failure-driven
   reconstruction.** ReaDy-Go (Feb 2026) is the single most dangerous neighbor: it
   already trains nav policies in a GS real-to-sim. The SNU group could plausibly add a
   drone or a "reconstruct-where-it-failed" loop within two cycles. **Mitigation:** lead
   with C1+C2+C3 *together* (they have none of the three combined), and get the
   *cross-embodiment* result — the hardest for them to replicate quickly — locked early.
2. **An AURA follow-up that swaps the LLM-imagined curriculum for logged failures.**
   AURA already has autonomous upskilling + retrieval + sim training on a real robot.
   Re-pointing its curriculum source at deployment logs is a small conceptual step.
   **Mitigation:** own the *failure-clustering-as-curriculum* framing explicitly and
   pair it with the 3DGS substrate (AURA uses domain-randomized physics sim, not
   photoreal reconstruction).
3. **A SkillRL/SAGE-style learning loop ported to robotics.** The digital
   self-improvement papers (arXiv:2602.08234, 2512.17102) have more rigorous learning
   loops than anything in robotics; a robotics port is inevitable. **Mitigation:** we do
   not compete on the learning loop — we compete on *substrate (3DGS) and source (real
   failures) and breadth (two embodiments)*; cite SkillRL's loop as prior art we build
   *on top of*.
4. **A Gemini-Robotics / π-class lab releasing an embodiment-agnostic orchestrator.**
   The frontier labs (per the LEAD survey §3.6 on Gemini Robotics 1.5) are converging on
   planner-dispatches-executor. If one ships an *orchestrator that grows skills across
   embodiments*, C1's novelty erodes. **Mitigation:** C2+C3 are the durable moat — a
   frontier lab is unlikely to anchor on *one PhD student's logged drone failures* and
   *3DGS reconstruction of a Stanford flight room*; our grounding is our defensibility.
5. **"3DGS as a navigation benchmark" becoming standard (S2E's NavBench-GS,
   arXiv:2507.22028) such that 3DGS-as-substrate reads as incremental.** If 3DGS sim is
   commodity by submission time, C3 must be framed as *failure-conditioned, closed-loop
   skill training in the reconstruction*, not "we used Gaussian Splatting." **Mitigation:
   never claim 3DGS itself as novel** — claim the failure→reconstruct→specialize loop.

Honest bottom line: **no single surveyed paper subsumes AORA-Forge, but each
contribution taken alone is one follow-up paper away from a competitor.** The defensible
position is the *conjunction* C1∧C2∧C3 and the *grounding* (real logged failures from a
real LEAD/AORA deployment). The strategic implication for the milestone plan (see
`04_mvp_milestones.md`): the cross-embodiment demonstration (C1) is both the highest-risk
and the highest-defensibility deliverable, and should be de-risked first.

---

## 5. Bibliography (with verified arXiv IDs)

Corrections to the prompt's working citations are marked **[corrected]**.

*Skill libraries / self-improving agents*
- Voyager — Wang et al., 2023. arXiv:2305.16291.
- RoboGen — Wang et al., ICML 2024. arXiv:2311.01455.
- Eureka — Ma et al., ICLR 2024. arXiv:2310.12931.
- DrEureka — Ma et al., RSS 2024. arXiv:2406.01967.
- SAGE ("RL for Self-Improving Agent with Skill Library") — Wang et al., Dec 2025. arXiv:2512.17102. *(Digital/AppWorld agent, **not** manipulation — prompt label corrected.)*
- SkillRL — Xia et al., Feb 2026. arXiv:2602.08234.
- Agent Skills for LLMs (survey) — Xu & Yan, Agent Skills '26 Workshop. arXiv:2602.12430.
- AURA — Zhu et al., CoRL 2025 GenPriors workshop. **[corrected]** arXiv:2506.02507 (June 2025; prompt said Feb 2026 / 2602.*).

*Real-to-sim / 3DGS for policy training*
- ReaDy-Go — Yoo et al., Feb 2026. arXiv:2602.11575.
- RoboSplat — Yang et al., RSS 2025. arXiv:2504.13175.
- GauSS-MI — Xie et al., 2025. arXiv:2504.21067.
- DroneSplat — Tang et al., 2025. arXiv:2503.16964.
- GS-Planner — Jin et al., IROS 2024. **[corrected]** arXiv:2405.10142 (prompt gave no ID).

*Cross-embodiment*
- Open X-Embodiment / RT-X — O'Neill et al., ICRA 2024. arXiv:2310.08864.
- π₀ — Black et al., RSS 2025. **[confirmed]** arXiv:2410.24164.
- Pushing the Limits of Cross-Embodiment Learning — Yang et al., 2024. arXiv:2402.19432.
- From Seeing to Experiencing (S2E, NavBench-GS) — He et al., 2025. arXiv:2507.22028.
- H-Zero — Lin et al., Dec 2025. arXiv:2512.00971.

*Scene graphs / structured memory*
- SayPlan — Rana et al., CoRL 2023. arXiv:2307.06135.
- ConceptGraphs — Gu et al., 2023. arXiv:2309.16650.
- CuriousBot — Wang et al., RA-L (2025). arXiv:2501.13338.

*Internal source documents (not to be re-derived)*
- `AORA_v1/lead-agent_system_report_v2.3.2.md` — LEAD v2.5 system report (architecture, failure modes, eval).
- `AORA_v1/LEAD_Literature_Review.md` — LEAD survey (§12 gap analysis items 1, 2, 3, 5 map to C1/C2/C3).
- `AORA_v1/ARCHITECTURE.md` — AORA (Habitat port; the embodiment-blind-planner ablation and HM3D consumer of our outputs).
