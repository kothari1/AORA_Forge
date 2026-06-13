"""LLM-driven clustering of failures into themed ``FailureCluster``s — the heart
of contribution C2.

The LLM reasons over the *narratives* (what went wrong, in words) to discover
themes a single skill could fix; the deterministic offline fallback groups by the
failure-mode → theme prior from ``taxonomy``. Either way the output is the same
contract, so the rest of the pipeline doesn't care which ran.
"""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, Field

from aora_forge.failures.taxonomy import FAILURE_THEME_HINTS, suggest_skill_type
from aora_forge.llm.client import LLMClient, ModelTier
from aora_forge.llm.prompts import CLUSTER_FAILURES_SYSTEM
from aora_forge.schemas import (
    FailureCluster,
    FailureRecord,
)
from aora_forge.utils.logging import RunTelemetry, get_logger

log = get_logger("failures.clusterer")


class _ClusterList(BaseModel):
    """Call-local wrapper so the structured-output tool can return many clusters."""

    clusters: list[FailureCluster] = Field(default_factory=list)


def _render_records(records: list[FailureRecord]) -> str:
    lines = ["Here are the deployment failures to cluster:\n"]
    for r in records:
        obs = r.observations
        sig = []
        if obs.dist_to_goal_final is not None:
            sig.append(f"dist={obs.dist_to_goal_final:.1f}m")
        if obs.vlm_claimed_success is not None:
            sig.append(f"claimed_success={obs.vlm_claimed_success}")
        if obs.steps_used is not None:
            sig.append(f"steps={obs.steps_used}")
        sigtxt = f" [{', '.join(sig)}]" if sig else ""
        lines.append(
            f"- id={r.record_id} | {r.embodiment.value} | {r.environment} | "
            f"mode={r.failure_mode.value} | query='{r.target_query}'{sigtxt}\n"
            f"    instruction: {r.task_instruction}\n"
            f"    narrative: {r.narrative}"
        )
    lines.append("\nCluster them into coherent, fixable themes and return via the tool.")
    return "\n".join(lines)


def _offline_cluster(records: list[FailureRecord]) -> _ClusterList:
    """Deterministic clustering by the failure-mode → theme prior.

    Groups records whose failure modes share a theme hint, producing one cluster
    per distinct theme. This is the offline/mock path; with a real key the LLM
    clusters over narratives instead.
    """
    by_theme: dict[str, list[FailureRecord]] = {}
    for r in records:
        theme = FAILURE_THEME_HINTS.get(r.failure_mode, "generic failure")
        by_theme.setdefault(theme, []).append(r)

    clusters: list[FailureCluster] = []
    for i, (theme, members) in enumerate(sorted(by_theme.items(), key=lambda kv: -len(kv[1]))):
        modes = [m.failure_mode for m in members]
        embs = sorted({m.embodiment for m in members}, key=lambda e: e.value)
        skill_votes = Counter(suggest_skill_type(m.failure_mode) for m in members)
        suggested = skill_votes.most_common(1)[0][0]
        clusters.append(
            FailureCluster(
                cluster_id=f"cl{i:02d}",
                title=theme,
                description=(
                    f"{len(members)} failures sharing the theme '{theme}' across "
                    f"{len(embs)} embodiment(s)."
                ),
                hypothesized_root_cause=(
                    f"Recurring {Counter(m.value for m in modes).most_common(1)[0][0]} "
                    f"under conditions characteristic of: {theme}."
                ),
                suggested_skill_type=suggested,
                member_record_ids=[m.record_id for m in members],
                representative_record_ids=[m.record_id for m in members[:3]],
                embodiments_involved=embs,
                failure_modes_involved=sorted(set(modes), key=lambda m: m.value),
                priority=float(len(members)),
            )
        )
    return _ClusterList(clusters=clusters)


def _postprocess(result: _ClusterList, records: list[FailureRecord]) -> list[FailureCluster]:
    """Fill in derived fields the LLM may have left empty and drop dangling ids."""
    valid_ids = {r.record_id for r in records}
    by_id = {r.record_id for r in records}
    mode_by_id = {r.record_id: r.failure_mode for r in records}
    emb_by_id = {r.record_id: r.embodiment for r in records}

    out: list[FailureCluster] = []
    for i, cl in enumerate(result.clusters):
        members = [m for m in cl.member_record_ids if m in valid_ids]
        if not members:
            continue
        if not cl.cluster_id:
            cl.cluster_id = f"cl{i:02d}"
        if not cl.representative_record_ids:
            cl.representative_record_ids = members[:3]
        else:
            cl.representative_record_ids = [m for m in cl.representative_record_ids if m in by_id][
                :3
            ] or members[:3]
        if not cl.embodiments_involved:
            cl.embodiments_involved = sorted({emb_by_id[m] for m in members}, key=lambda e: e.value)
        if not cl.failure_modes_involved:
            cl.failure_modes_involved = sorted(
                {mode_by_id[m] for m in members}, key=lambda m: m.value
            )
        cl.member_record_ids = members
        if cl.priority <= 0:
            cl.priority = float(len(members))
        out.append(cl)
    out.sort(key=lambda c: -c.priority)
    return out


def cluster_failures(
    records: list[FailureRecord],
    client: LLMClient,
    *,
    telemetry: RunTelemetry | None = None,
) -> list[FailureCluster]:
    """Cluster failures into themes. Returns clusters sorted by priority (desc)."""
    if not records:
        return []
    result, usage = client.complete_structured(
        system=CLUSTER_FAILURES_SYSTEM,
        user=_render_records(records),
        schema=_ClusterList,
        offline_fallback=lambda: _offline_cluster(records),
        model_tier=ModelTier.PLANNER,
        task="cluster_failures",
        max_tokens=8192,
    )
    if telemetry is not None:
        telemetry.record("cluster_failures", usage)
    clusters = _postprocess(result, records)
    log.info(
        "clustered %d failures into %d theme(s) [%s]",
        len(records),
        len(clusters),
        "mock" if usage.mocked else usage.model,
    )
    return clusters
