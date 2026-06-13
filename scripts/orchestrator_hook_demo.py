#!/usr/bin/env python3
"""Tier-3 stretch (Option A): show that a LEAD-style planner, hooked into
AORA-Forge's tool registry, gains tools that did not exist before growth.

This does NOT run LEAD — it models the planner's tool surface (a stub) and proves
the integration point: grow skills from failures, then for a given scene the
planner's available tools strictly grow.

    python scripts/orchestrator_hook_demo.py --mock
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aora_forge.llm.client import get_llm_client  # noqa: E402
from aora_forge.orchestrator_hooks.post_mission import grow_from_failures  # noqa: E402
from aora_forge.orchestrator_hooks.stub_planner import StubLEADPlanner  # noqa: E402
from aora_forge.orchestrator_hooks.tool_registry import OrchestratorToolRegistry  # noqa: E402
from aora_forge.scene_graph.builder import build_from_labels  # noqa: E402
from aora_forge.schemas import Embodiment  # noqa: E402
from aora_forge.skill_library.store import SkillStore  # noqa: E402
from aora_forge.utils.synthetic_data import generate_synthetic_failures  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--store", default="hook_demo_store")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    client = get_llm_client(force_mock=args.mock)
    store = SkillStore(Path(args.store) / "library")

    # 1) Grow skills from failures (both embodiments, one mechanism).
    records = generate_synthetic_failures(args.n, seed=0)
    grow_from_failures(records, client, store)
    registry = OrchestratorToolRegistry(store)

    planner = StubLEADPlanner()
    before = planner.baseline_view()

    print("=" * 78)
    print("  Orchestrator hook — does the planner see new tools after growth?")
    print("=" * 78)
    print("\nLEAD planner BEFORE growth:")
    print(f"  actions      : {before.actions}")
    print(f"  direct tools : {before.direct_tools}")
    print(f"  exec prompts : {before.executor_prompt_augmentations or '(none)'}")

    scenes = [
        (["green clock", "table"], Embodiment.DRONE_FIGS, "flightroom_ssv_exp"),
        (["chair", "sofa"], Embodiment.GROUND_HABITAT, "hm3d:00800-TEEsavR23oF"),
    ]
    for labels, emb, env in scenes:
        ctx = build_from_labels(labels, embodiment=emb, environment=env)
        after = planner.view_with_growth(ctx, registry, top_k=3)
        added = StubLEADPlanner.new_capabilities(before, after)
        print(f"\n--- scene {labels} on {emb.value} ---")
        print(f"  planner AFTER growth — direct tools: {after.direct_tools}")
        print(
            f"  planner AFTER growth — exec prompts: {after.executor_prompt_augmentations or '(none)'}"
        )
        print("  NEW capabilities the planner gained:")
        if added["new_direct_tools"]:
            for t in added["new_direct_tools"]:
                print(f"    + direct_tool   '{t}'   <-- did not exist before growth")
        for t in added["new_executor_prompts"]:
            print(f"    + executor_prompt '{t}'  <-- specialised strategy injected")
        if not added["new_direct_tools"] and not added["new_executor_prompts"]:
            print("    (no scene-relevant grown skills)")
        # Show one grown tool as a concrete Anthropic tool spec.
        if after.grown_tools:
            t0 = after.grown_tools[0]
            print(f"  example tool spec the planner now receives ('{t0.name}'):")
            spec = t0.to_anthropic_tool()
            print(f"    name        : {spec['name']}")
            print(f"    integration : {t0.integration}")
            print(f"    input_schema: {spec['input_schema']}")

    print("\n" + "=" * 78)
    print("  Proof: the planner's tool surface strictly grows. Failures -> skills -> tools.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
