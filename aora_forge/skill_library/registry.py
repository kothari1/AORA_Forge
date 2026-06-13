"""Convert a stored ``Skill`` into an ``OrchestratorTool`` the LEAD/AORA planner
can see and call.

This is the bridge back into LEAD's ``OrchestratorOutput`` discriminated union
(system report §6.3): a grown skill becomes either a new ``direct_tool`` the
planner may call (code/classifier skills) or an ``executor_prompt`` augmentation
the executor adopts for the next ``dispatch_nav`` (prompt skills).
"""

from __future__ import annotations

from typing import Any

from aora_forge.schemas import (
    OrchestratorTool,
    Skill,
    SkillIO,
    SkillType,
    ToolProvenance,
)

# Logical SkillIO type -> JSON-schema fragment.
_JSON_TYPE: dict[str, dict[str, Any]] = {
    "float": {"type": "number"},
    "number": {"type": "number"},
    "int": {"type": "integer"},
    "integer": {"type": "integer"},
    "bool": {"type": "boolean"},
    "boolean": {"type": "boolean"},
    "str": {"type": "string"},
    "string": {"type": "string"},
    "image_b64": {"type": "string", "description": "base64-encoded image"},
    "bbox": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
    "list[float]": {"type": "array", "items": {"type": "number"}},
    "list[str]": {"type": "array", "items": {"type": "string"}},
}


def _io_to_property(io: SkillIO) -> dict[str, Any]:
    frag = dict(_JSON_TYPE.get(io.type.lower(), {"type": "string"}))
    if io.description and "description" not in frag:
        frag["description"] = io.description
    return frag


def _input_schema(inputs: list[SkillIO]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {io.name: _io_to_property(io) for io in inputs},
        "required": [io.name for io in inputs if io.required],
    }


def skill_to_tool(skill: Skill) -> OrchestratorTool:
    """Render a grown skill as an orchestrator tool spec."""
    integration = "executor_prompt" if skill.skill_type is SkillType.PROMPT else "direct_tool"
    desc = skill.spec.description
    if skill.skill_type is SkillType.PROMPT:
        desc += " (Adopt as an executor strategy when the scene matches this theme.)"
    else:
        desc += f" Call before done() to {skill.spec.success_criterion[:80]}".rstrip()
    return OrchestratorTool(
        name=skill.skill_name,
        description=desc,
        input_schema=_input_schema(skill.spec.inputs),
        skill_type=skill.skill_type,
        embodiments=skill.spec.target_embodiments,
        integration=integration,  # type: ignore[arg-type]
        provenance=ToolProvenance(
            skill_id=skill.skill_id,
            cluster_id=str(skill.provenance.get("cluster_id"))
            if skill.provenance.get("cluster_id")
            else None,
            record_ids=list(skill.provenance.get("record_ids", [])),
        ),
    )


def register_skills(skills: list[Skill]) -> list[OrchestratorTool]:
    """Render a batch of skills as orchestrator tools."""
    return [skill_to_tool(s) for s in skills]
