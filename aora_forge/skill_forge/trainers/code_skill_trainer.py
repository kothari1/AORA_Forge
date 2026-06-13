"""Code-skill trainer — Voyager-style: ask Claude to write a Python function whose
signature matches the spec, then *verify it by running it* against generated test
probes, retrying up to 3 times with the error fed back.

This is the same retry-on-failure idiom as LEAD's Pydantic validation loop, ported
to executable verification: a skill that does not run is not stored.
"""

from __future__ import annotations

import ast
import re
from typing import Any

from aora_forge.llm.client import LLMClient, ModelTier
from aora_forge.llm.prompts import CODE_SKILL_SYSTEM
from aora_forge.schemas import (
    FailureCluster,
    FailureRecord,
    Skill,
    SkillIO,
    SkillSpec,
    SkillType,
    SkillValidation,
)
from aora_forge.skill_forge.trainer_base import Trainer
from aora_forge.utils.logging import RunTelemetry, get_logger

log = get_logger("skill_forge.code_trainer")

_MAX_ATTEMPTS = 3


def _sample_arg(io: SkillIO) -> Any:
    """A representative argument value for a typed input port."""
    t = io.type.lower()
    if t in ("float", "number"):
        return 1.2
    if t in ("int", "integer"):
        return 3
    if t in ("bool", "boolean"):
        return True
    if t == "bbox":
        return [0.2, 0.2, 0.6, 0.6]
    if t.startswith("list"):
        return [0.1] * 8
    if t in ("image_b64", "str", "string"):
        return ""
    return None


def _offline_code(spec: SkillSpec) -> str:
    """A real, correct function for the default arrival-gate spec (offline path)."""
    name = spec.skill_name
    has_depth = any("depth" in i.name.lower() for i in spec.inputs)
    has_bbox = any(i.type.lower() == "bbox" for i in spec.inputs)
    if has_depth and has_bbox:
        return (
            "```python\n"
            f"def {name}(target_bbox, depth_p25_m, threshold_m=1.6):\n"
            '    """Accept an arrival only when the target bbox is plausible and the\n'
            "    P25 depth within it is within the arrival threshold. Prevents the\n"
            "    premature-success failure mode (claimed arrival at distance).\n\n"
            "    Args:\n"
            "        target_bbox: normalised [x1, y1, x2, y2] around the target surface.\n"
            "        depth_p25_m: 25th-percentile depth in the bbox, in metres (or None).\n"
            "        threshold_m: arrival distance threshold in metres.\n"
            "    Returns:\n"
            "        bool: whether arrival should be accepted.\n"
            '    """\n'
            "    if depth_p25_m is None:\n"
            "        return False\n"
            "    try:\n"
            "        x1, y1, x2, y2 = target_bbox\n"
            "    except (TypeError, ValueError):\n"
            "        return False\n"
            "    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)\n"
            "    if area <= 0.0:\n"
            "        return False\n"
            "    return bool(depth_p25_m <= threshold_m)\n"
            "```"
        )
    # generic safe fallback: echo a boolean decision based on the first numeric input
    return (
        "```python\n"
        f"def {name}(*args, **kwargs):\n"
        '    """Conservative default skill: returns False (no-accept) unless given a\n'
        "    numeric first argument within a unit threshold. Replace with a real\n"
        '    implementation when the spec is concrete."""\n'
        "    if args and isinstance(args[0], (int, float)):\n"
        "        return bool(args[0] <= 1.0)\n"
        "    return False\n"
        "```"
    )


def _extract_function(source_block: str) -> tuple[str, str]:
    """Pull the function source and its name out of an LLM response."""
    m = re.search(r"```(?:python)?\s*(.*?)```", source_block, re.DOTALL)
    code = (m.group(1) if m else source_block).strip()
    tree = ast.parse(code)  # raises SyntaxError -> caught by caller for retry
    func_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    if not func_names:
        raise ValueError("no function definition found in generated code")
    return code, func_names[0]


def _verify(code: str, func_name: str, spec: SkillSpec) -> tuple[bool, list[dict], str]:
    """Exec the code in an isolated namespace and run typed probes.

    Returns (passed, per_case, error). A skill must (a) import/exec cleanly,
    (b) run on representative typed args, and (c) for the depth-gate pattern,
    accept a near target and reject a far one.
    """
    ns: dict[str, Any] = {}
    try:
        exec(compile(code, f"<skill:{func_name}>", "exec"), ns)  # noqa: S102 - sandboxed namespace
    except Exception as exc:  # noqa: BLE001
        return False, [], f"exec failed: {exc}"
    fn = ns.get(func_name)
    if not callable(fn):
        return False, [], f"{func_name} is not callable after exec"

    per_case: list[dict] = []
    args = [_sample_arg(io) for io in spec.inputs] or [1.0]

    # Probe 1: runs on representative args.
    try:
        out = fn(*args)
        per_case.append({"probe": "runs_on_typed_args", "passed": True, "result": repr(out)[:60]})
    except Exception as exc:  # noqa: BLE001
        return (
            False,
            [{"probe": "runs_on_typed_args", "passed": False, "error": str(exc)}],
            str(exc),
        )

    # Probe 2 (depth-gate pattern): near accepts, far rejects.
    has_depth = any("depth" in io.name.lower() for io in spec.inputs)
    has_bbox = any(io.type.lower() == "bbox" for io in spec.inputs)
    if has_depth and has_bbox:
        try:
            near = fn([0.2, 0.2, 0.6, 0.6], 1.0)
            far = fn([0.2, 0.2, 0.6, 0.6], 5.0)
            ok = bool(near) and not bool(far)
            per_case.append(
                {
                    "probe": "near_accepts_far_rejects",
                    "passed": ok,
                    "near": bool(near),
                    "far": bool(far),
                }
            )
        except Exception as exc:  # noqa: BLE001
            per_case.append(
                {"probe": "near_accepts_far_rejects", "passed": False, "error": str(exc)}
            )

    passed = all(c.get("passed", False) for c in per_case)
    return passed, per_case, "" if passed else "one or more probes failed"


class CodeSkillTrainer(Trainer):
    skill_type = SkillType.CODE

    def train(
        self,
        spec: SkillSpec,
        cluster: FailureCluster,
        records: list[FailureRecord],
        client: LLMClient,
        *,
        telemetry: RunTelemetry | None = None,
    ) -> Skill:
        user_base = (
            f"SkillSpec:\n{spec.model_dump_json(indent=2)}\n\n"
            f"Write one Python function named '{spec.skill_name}' matching these inputs/outputs."
        )
        code = ""
        func_name = spec.skill_name
        per_case: list[dict] = []
        error = ""
        passed = False

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            user = (
                user_base
                if attempt == 1
                else (
                    f"{user_base}\n\nYour previous attempt failed verification: {error}\n"
                    "Return a corrected function."
                )
            )
            raw, usage = client.complete_text(
                system=CODE_SKILL_SYSTEM,
                user=user,
                offline_fallback=lambda: _offline_code(spec),
                model_tier=ModelTier.PLANNER,
                task="train_code_skill",
                max_tokens=1500,
            )
            if telemetry is not None:
                telemetry.record("train_code_skill", usage)
            try:
                code, func_name = _extract_function(raw)
            except (SyntaxError, ValueError) as exc:
                error = f"parse error: {exc}"
                log.info("code skill attempt %d: %s", attempt, error)
                continue
            passed, per_case, error = _verify(code, func_name, spec)
            if passed:
                break
            log.info("code skill attempt %d: verify failed (%s)", attempt, error)

        n = len(per_case)
        n_pass = sum(1 for c in per_case if c.get("passed"))
        validation = SkillValidation(
            passed=passed,
            score=round(n_pass / n, 3) if n else 0.0,
            n_cases_total=n,
            n_cases_passed=n_pass,
            notes=error or "all verification probes passed",
            per_case=per_case,
        )
        ctx = self.aggregate_scene_context(records)
        skill = Skill(
            skill_id=spec.skill_name,
            skill_name=spec.skill_name,
            skill_type=SkillType.CODE,
            spec=spec,
            artifact_kind="python",
            artifact_ref="artifact.py",
            artifact_inline=code,
            validation=validation,
            scene_graph_context=ctx,
            provenance={"cluster_id": cluster.cluster_id, "function_name": func_name},
        )
        log.info(
            "forged code skill '%s': %s (%d/%d probes)",
            skill.skill_name,
            "PASS" if passed else "FAIL",
            n_pass,
            n,
        )
        return skill
