#!/usr/bin/env python3
"""End-to-end demonstration of the AORA-Forge growth loop.

    synthetic failures -> cluster -> spec -> train+validate -> store -> retrieve

Runs against the **real Claude API** when ``ANTHROPIC_API_KEY`` is set, otherwise
against a deterministic mock (clearly flagged). This script is the proof that the
architecture is sound: a pile of real-shaped deployment failures becomes a set of
validated, retrievable skills registered as planner tools — across both
embodiments through one code path.

    python scripts/demo_full_loop.py                 # auto: real if key, else mock
    python scripts/demo_full_loop.py --mock          # force the offline mock
    python scripts/demo_full_loop.py --n 50 --seed 0 # control the synthetic set
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aora_forge.failures.collector import write_jsonl  # noqa: E402
from aora_forge.llm.client import get_llm_client  # noqa: E402
from aora_forge.orchestrator_hooks.post_mission import grow_from_failures  # noqa: E402
from aora_forge.orchestrator_hooks.tool_registry import OrchestratorToolRegistry  # noqa: E402
from aora_forge.scene_graph.builder import build_from_labels  # noqa: E402
from aora_forge.schemas import Embodiment  # noqa: E402
from aora_forge.utils.logging import RunTelemetry  # noqa: E402
from aora_forge.utils.synthetic_data import generate_synthetic_failures  # noqa: E402

# A few "current scene" queries to demonstrate scene-conditioned retrieval.
_RETRIEVAL_PROBES = [
    (["green clock", "table"], Embodiment.DRONE_FIGS, "flightroom_ssv_exp"),
    (["chair", "sofa"], Embodiment.GROUND_HABITAT, "hm3d:00800-TEEsavR23oF"),
    (["potted plant"], Embodiment.GROUND_HABITAT, "hm3d:00813-svBbv1Pavdk"),
]


def _rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=50, help="number of synthetic failures")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--store", default="demo_skill_store", help="on-disk skill library dir")
    ap.add_argument("--mock", action="store_true", help="force the offline mock LLM")
    ap.add_argument("--max-skills", type=int, default=None, help="cap number of skills forged")
    args = ap.parse_args()

    _rule("AORA-Forge — end-to-end growth demo")
    client = get_llm_client(force_mock=args.mock)
    telemetry = RunTelemetry()

    # 1) Generate (or, in production, collect) the deployment failures.
    records = generate_synthetic_failures(args.n, seed=args.seed)
    out_dir = Path(args.store)
    failures_path = write_jsonl(records, out_dir / "synthetic_failures.jsonl")
    by_emb = {e.value: sum(1 for r in records if r.embodiment is e) for e in Embodiment}
    print(f"\nGenerated {len(records)} synthetic failures -> {failures_path}")
    print(f"  by embodiment : {by_emb}")
    print(f"  failure modes : {sorted({r.failure_mode.value for r in records})}")

    # 2-5) Grow: cluster -> spec -> train+validate -> store.
    _rule("Growing skills from failures")
    from aora_forge.skill_library.store import SkillStore

    store = SkillStore(out_dir / "library")
    summary = grow_from_failures(
        records, client, store, telemetry=telemetry, max_skills=args.max_skills
    )

    _rule("Forged skills")
    print(
        f"{len(summary.clusters)} clusters -> {len(summary.skills)} skills "
        f"({summary.n_skills_validated} validated)\n"
    )
    for s in summary.skills:
        v = s.validation
        cross = ",".join(e.value for e in s.spec.target_embodiments)
        flag = "PASS" if v.passed else "weak"
        print(f"  [{flag}] {s.skill_name}")
        print(
            f"         type={s.skill_type.value:<10} score={v.score:<5} "
            f"cases={v.n_cases_passed}/{v.n_cases_total} embodiments=[{cross}]"
        )
        if s.reconstruction_ref:
            print(f"         3DGS substrate: {s.reconstruction_ref} (stub)")

    # 6) Retrieve: scene-conditioned skill lookup, rendered as planner tools.
    _rule("Scene-conditioned retrieval (skills -> planner tools)")
    registry = OrchestratorToolRegistry(store)
    for labels, emb, env in _RETRIEVAL_PROBES:
        ctx = build_from_labels(labels, embodiment=emb, environment=env)
        tools = registry.tools_for_context(ctx, top_k=3, scene_graph_conditioned=True)
        print(f"\n  scene {labels} on {emb.value}:")
        if not tools:
            print("    (no skills retrieved)")
        for t in tools:
            print(f"    -> {t.name}  [{t.integration}]")

    # 7) Summary: time, tokens, cost.
    _rule("Run summary — what this cost")
    for line in telemetry.summary_lines():
        print("  " + line)
    if telemetry.all_mocked:
        print("\n  NOTE: ran in MOCK mode (no ANTHROPIC_API_KEY). Set the key and re-run")
        print("        for the real-API proof; the pipeline is identical either way.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
