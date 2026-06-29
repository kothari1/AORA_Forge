#!/usr/bin/env python3
"""Run a real A/B: does a grown skill improve AORA_v1's ObjectNav performance?

Drives AORA_v1 (in its aora_v1 conda env) over a fixed minival episode set —
baseline vs. with the grown executor-prompt skill prepended — and reports
SSR / SPL / FPR / failure-mode deltas. Saves a JSON the dashboard can display.

    # default: the arrival-verifier skill grown from real HALLUCINATED_DONE failures
    python scripts/run_ab.py --store real_skill_store/library \
        --skill visual_target_arrival_verifier --limit 6

Each episode makes real Vertex calls + GPU rendering; bound it with --limit and
--max-usd. Both runs use identical args, so the episode set is identical.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aora_forge.evaluation.ab_runner import run_ab  # noqa: E402
from aora_forge.skill_library.store import SkillStore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", default="real_skill_store/library", help="skill library dir")
    ap.add_argument("--skill", default="visual_target_arrival_verifier", help="skill_id to inject")
    ap.add_argument("--category", default=None, help="restrict to one ObjectNav category")
    ap.add_argument("--limit", type=int, default=6, help="episodes per run (both runs identical)")
    ap.add_argument("--max-usd", type=float, default=30.0, help="AORA_v1 cost-ledger cap")
    ap.add_argument("--mode", default="executor_only", choices=["executor_only", "lead"])
    ap.add_argument(
        "--live-dir", default="/tmp/aora_live", help="stream frames here during the run"
    )
    ap.add_argument("--out", default="ab_result.json", help="where to save the result JSON")
    args = ap.parse_args()

    skill = SkillStore(args.store).load(args.skill)
    print(f"Injecting grown skill '{skill.skill_name}' ({skill.skill_type.value})")
    print(f"  → grown from: {skill.provenance.get('record_ids', [])[:3]} ...")
    print(
        f"Running {args.limit} episodes baseline vs with-skill ({args.mode})... (this takes a while)\n"
    )

    result = run_ab(
        skill,
        category=args.category,
        limit=args.limit,
        max_usd=args.max_usd,
        mode=args.mode,
        live_dir=args.live_dir,
    )

    print("\n" + "=" * 66)
    print(f"  A/B RESULT — does '{result.skill_name}' help?")
    print("=" * 66)
    print(f"  {'metric':<28}{'baseline':>10}{'+skill':>10}{'Δ':>10}")
    for row in result.table_rows():
        print(f"  {row[0]:<28}{row[1]:>10}{row[2]:>10}{row[3]:>10}")
    print(f"\n  baseline failure modes : {result.baseline.failure_modes}")
    print(f"  +skill   failure modes : {result.with_skill.failure_modes}")
    print(f"\n  {result.note}")

    Path(args.out).write_text(json.dumps(result.to_json(), indent=2))
    print(f"\n  saved → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
