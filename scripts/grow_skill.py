#!/usr/bin/env python3
"""Grow skills from a failures JSONL and store them in the skill library.

    python scripts/grow_skill.py --in failures.jsonl --store ./library
    python scripts/grow_skill.py --synthetic 50 --store ./library --max-skills 4
    python scripts/grow_skill.py --synthetic 50 --cluster-title "small"   # one theme

This is the single-stage counterpart to ``demo_full_loop.py``: it runs the growth
loop and persists skills, without the retrieval demonstration.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aora_forge.failures.collector import collect_from_jsonl  # noqa: E402
from aora_forge.llm.client import get_llm_client  # noqa: E402
from aora_forge.orchestrator_hooks.post_mission import grow_from_failures  # noqa: E402
from aora_forge.skill_library.store import SkillStore  # noqa: E402
from aora_forge.utils.logging import RunTelemetry  # noqa: E402
from aora_forge.utils.synthetic_data import generate_synthetic_failures  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="infile", default=None)
    ap.add_argument("--synthetic", type=int, default=None)
    ap.add_argument("--store", default="library")
    ap.add_argument("--max-skills", type=int, default=None)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    if args.infile:
        records = collect_from_jsonl(args.infile)
    elif args.synthetic:
        records = generate_synthetic_failures(args.synthetic, seed=0)
    else:
        ap.error("provide --in <jsonl> or --synthetic <N>")

    client = get_llm_client(force_mock=args.mock)
    tel = RunTelemetry()
    store = SkillStore(args.store)
    summary = grow_from_failures(records, client, store, telemetry=tel, max_skills=args.max_skills)

    print(
        f"Forged {len(summary.skills)} skills into {args.store} "
        f"({summary.n_skills_validated} validated):"
    )
    for s in summary.skills:
        print(f"  {s.skill_name}  [{s.skill_type.value}]  score={s.validation.score}")
    for line in tel.summary_lines():
        print("  " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
