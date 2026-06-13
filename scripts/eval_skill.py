#!/usr/bin/env python3
"""Evaluate a stored skill.

For prompt/code/classifier skills this re-reports the stored validation and, for
code skills, re-runs the verification probes against the live artifact (a cheap
regression check that the stored function still passes). A holdout JSONL of
failures can be supplied to re-judge a prompt skill against unseen scenarios.

    python scripts/eval_skill.py --store ./library --skill <skill_id>
    python scripts/eval_skill.py --store ./library --skill <skill_id> --holdout new_failures.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aora_forge.failures.collector import collect_from_jsonl  # noqa: E402
from aora_forge.schemas import SkillType  # noqa: E402
from aora_forge.skill_library.store import SkillStore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", default="library")
    ap.add_argument("--skill", required=True, help="skill_id to evaluate")
    ap.add_argument(
        "--holdout", default=None, help="optional failures JSONL to re-judge a prompt skill"
    )
    args = ap.parse_args()

    store = SkillStore(args.store)
    skill = store.load(args.skill)
    v = skill.validation
    print(f"Skill: {skill.skill_id}  (type={skill.skill_type.value}, v{skill.version})")
    print(f"  description : {skill.spec.description}")
    print(
        f"  stored validation: passed={v.passed} score={v.score} "
        f"cases={v.n_cases_passed}/{v.n_cases_total}"
    )
    print(f"  notes       : {v.notes}")

    if skill.skill_type is SkillType.CODE and skill.artifact_inline:
        from aora_forge.skill_forge.trainers.code_skill_trainer import _extract_function, _verify

        code, fn = _extract_function(skill.artifact_inline)
        passed, per_case, err = _verify(code, fn, skill.spec)
        print(f"  live re-verify: {'PASS' if passed else 'FAIL'} ({err or 'all probes passed'})")
        for c in per_case:
            print(f"    - {c}")

    if args.holdout and skill.skill_type is SkillType.PROMPT and skill.artifact_inline:
        from aora_forge.skill_forge.trainers.prompt_skill_trainer import _offline_verdict

        holdout = collect_from_jsonl(args.holdout)
        passes = sum(int(_offline_verdict(skill.artifact_inline, r).passed) for r in holdout)
        print(
            f"  holdout re-judge (offline heuristic): {passes}/{len(holdout)} scenarios avoidable"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
