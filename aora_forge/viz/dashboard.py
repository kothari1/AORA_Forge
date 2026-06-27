"""A Gradio dashboard to *see* AORA-Forge: real failures -> clusters -> forged
skills -> retrieval as planner tools, plus the robot-side visuals (trajectory
plots and captured failure frames) that AORA_v1 produces.

Launch with ``python scripts/dashboard.py`` (or ``make dashboard``). Bind to
localhost by default; view via NoMachine's browser on jugg or an SSH/Tailscale
port-forward (``ssh -L 7861:localhost:7861 jugg``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless server: render figures to images, never a window
import matplotlib.pyplot as plt  # noqa: E402

from aora_forge.failures.collector import collect_from_aora_outputs  # noqa: E402
from aora_forge.llm.client import get_llm_client, resolve_provider  # noqa: E402
from aora_forge.orchestrator_hooks.post_mission import grow_from_failures  # noqa: E402
from aora_forge.orchestrator_hooks.stub_planner import StubLEADPlanner  # noqa: E402
from aora_forge.orchestrator_hooks.tool_registry import OrchestratorToolRegistry  # noqa: E402
from aora_forge.scene_graph.builder import build_from_labels  # noqa: E402
from aora_forge.schemas import Embodiment, FailureRecord  # noqa: E402
from aora_forge.skill_library.store import SkillStore  # noqa: E402
from aora_forge.utils.logging import RunTelemetry, get_logger  # noqa: E402
from aora_forge.utils.synthetic_data import generate_synthetic_failures  # noqa: E402

log = get_logger("viz.dashboard")

_AORA_OUTPUTS = "/home/admin/projects/AORA_v1/outputs"
_STORE_DIR = Path(__file__).resolve().parents[2] / "dashboard_store" / "library"


# --------------------------------------------------------------------------- #
# Helpers (pure-ish; return data the Gradio callbacks hand to widgets)
# --------------------------------------------------------------------------- #


def _short_scene(s: str) -> str:
    # hm3d paths are long; keep the informative tail
    return s.split("/")[-1].replace(".basis.glb", "") if "/" in s else s


def _failures_table(records: list[FailureRecord]) -> list[list[str]]:
    rows = []
    for r in records:
        rows.append(
            [
                r.record_id.split("::")[-1],
                r.embodiment.value,
                _short_scene(r.environment),
                r.failure_mode.value,
                r.target_query,
                (r.narrative[:90] + "…") if len(r.narrative) > 90 else r.narrative,
            ]
        )
    return rows


def _mode_figure(records: list[FailureRecord]) -> Any:
    from collections import Counter

    counts = Counter(r.failure_mode.value for r in records)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    if counts:
        labels = [k for k, _ in counts.most_common()]
        vals = [counts[k] for k in labels]
        ax.barh(labels[::-1], vals[::-1], color="#c0563a")
        ax.set_xlabel("count")
        ax.set_title(f"Failure modes ({len(records)} failures)")
    fig.tight_layout()
    return fig


def load_failures(source: str, aora_path: str, n: int, seed: int) -> tuple:
    if source.startswith("Real"):
        records = collect_from_aora_outputs(aora_path or _AORA_OUTPUTS)
        note = f"Loaded **{len(records)} real failures** from `{aora_path or _AORA_OUTPUTS}`."
        if not records:
            note += " — none found; is the path right / has AORA_v1 produced runs?"
    else:
        records = generate_synthetic_failures(int(n), seed=int(seed))
        note = f"Generated **{len(records)} synthetic failures** (seed {seed}) — grounded in LEAD's real modes, but fabricated."
    return records, _failures_table(records), _mode_figure(records), note


def grow(records: list[FailureRecord], provider: str, max_skills: int) -> tuple:
    if not records:
        return None, [], [], "Load failures first.", []
    client = get_llm_client(
        force_mock=(provider == "mock"), provider=None if provider in ("mock", "auto") else provider
    )
    tel = RunTelemetry()
    store = SkillStore(_STORE_DIR)
    cap = int(max_skills) if max_skills and int(max_skills) > 0 else None
    summary = grow_from_failures(records, client, store, telemetry=tel, max_skills=cap)

    cluster_rows = [
        [
            c.cluster_id,
            c.title,
            c.size,
            c.suggested_skill_type.value,
            ", ".join(e.value for e in c.embodiments_involved),
        ]
        for c in summary.clusters
    ]
    skill_rows = [
        [
            s.skill_name,
            s.skill_type.value,
            "✓" if s.validation.passed else "weak",
            f"{s.validation.score:.2f}",
            ", ".join(e.value for e in s.spec.target_embodiments),
        ]
        for s in summary.skills
    ]
    cost_md = "### Run cost\n```\n" + "\n".join(tel.summary_lines()) + "\n```"
    skill_names = [s.skill_name for s in summary.skills]
    return str(_STORE_DIR), cluster_rows, skill_rows, cost_md, skill_names


def show_artifact(store_path: str, skill_name: str) -> str:
    if not store_path or not skill_name:
        return ""
    try:
        skill = SkillStore(store_path).load(skill_name)
    except Exception as exc:  # noqa: BLE001
        return f"(could not load {skill_name}: {exc})"
    head = (
        f"## {skill.skill_name}  ·  {skill.skill_type.value}\n"
        f"**validation:** {'PASS' if skill.validation.passed else 'weak'} "
        f"(score {skill.validation.score}, {skill.validation.n_cases_passed}/{skill.validation.n_cases_total} cases)\n\n"
        f"**spec:** {skill.spec.description}\n\n"
        f"**success criterion:** {skill.spec.success_criterion}\n\n"
        f"**forged from failures:** {', '.join(str(r) for r in skill.provenance.get('record_ids', []))[:200]}\n\n"
        f"---\n### artifact ({skill.artifact_kind})\n"
    )
    body = skill.artifact_inline or "(stored on disk)"
    lang = "python" if skill.artifact_kind == "python" else ""
    return head + f"```{lang}\n{body[:4000]}\n```"


def retrieve(store_path: str, objects: str, embodiment: str) -> tuple:
    if not store_path:
        return [], "Grow skills first.", ""
    labels = [o.strip() for o in objects.split(",") if o.strip()]
    emb = Embodiment(embodiment)
    ctx = build_from_labels(labels, embodiment=emb)
    registry = OrchestratorToolRegistry(SkillStore(store_path))
    planner = StubLEADPlanner()
    before = planner.baseline_view()
    after = planner.view_with_growth(ctx, registry, top_k=3)
    added = StubLEADPlanner.new_capabilities(before, after)

    rows = [
        [t.name, t.integration, ", ".join(e.value for e in t.embodiments)]
        for t in after.grown_tools
    ]
    md = "### Planner gains (vs. before growth)\n"
    if added["new_direct_tools"]:
        md += "\n".join(
            f"- **+ direct_tool** `{t}` — did not exist before" for t in added["new_direct_tools"]
        )
        md += "\n"
    if added["new_executor_prompts"]:
        md += "\n".join(f"- **+ executor_prompt** `{t}`" for t in added["new_executor_prompts"])
    if not added["new_direct_tools"] and not added["new_executor_prompts"]:
        md += "_(no scene-relevant grown skills)_"
    tool_json = (
        json.dumps(after.grown_tools[0].to_anthropic_tool(), indent=2) if after.grown_tools else ""
    )
    return rows, md, tool_json


def scan_robot_artifacts(aora_path: str) -> list[tuple[str, str]]:
    """Find the real robot-side visuals AORA_v1 produced: trajectory plots and
    captured frames, captioned with their episode + (where available) failure."""
    root = Path(aora_path or _AORA_OUTPUTS)
    # map run_dir -> failure_mode from episodes.jsonl
    mode_by_dir: dict[str, str] = {}
    for ep in root.glob("**/episodes.jsonl"):
        for line in ep.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("run_dir"):
                mode_by_dir[str(d["run_dir"])] = d.get("failure_mode", "?")
    items: list[tuple[str, str]] = []
    for cap in sorted(root.glob("**/captures/*.jpg"))[:24]:
        ep_dir = str(cap.parent.parent)
        mode = mode_by_dir.get(ep_dir, "")
        items.append((str(cap), f"{cap.parent.parent.name} · {mode} · {cap.stem}"))
    for png in sorted(root.glob("**/path.png"))[:24]:
        mode = mode_by_dir.get(str(png.parent), "")
        items.append((str(png), f"{png.parent.name} · {mode} · trajectory"))
    return items


# --------------------------------------------------------------------------- #
# Gradio UI
# --------------------------------------------------------------------------- #


def build_dashboard() -> Any:
    import gradio as gr

    default_provider = resolve_provider().value
    provider_choices = ["mock", "vertex", "anthropic", "openai"]

    with gr.Blocks(title="AORA-Forge") as demo:
        gr.Markdown(
            "# AORA-Forge — skill growth from real deployment failures\n"
            "Failures → cluster → forge a validated skill → retrieve as a planner tool, "
            "the same way for a drone and a ground robot. Tabs follow the pipeline; the last "
            "shows the **real** robot-side visuals from AORA_v1 runs."
        )
        records_state = gr.State([])
        store_state = gr.State("")

        with gr.Tab("1 · Failures (the curriculum)"):
            with gr.Row():
                source = gr.Radio(
                    ["Real (AORA_v1 runs)", "Synthetic"],
                    value="Real (AORA_v1 runs)",
                    label="data source",
                )
                aora_path = gr.Textbox(_AORA_OUTPUTS, label="AORA_v1 outputs path")
            with gr.Row():
                n = gr.Slider(10, 80, value=40, step=5, label="N (synthetic)")
                seed = gr.Slider(0, 20, value=0, step=1, label="seed (synthetic)")
            load_btn = gr.Button("Load failures", variant="primary")
            load_note = gr.Markdown()
            mode_plot = gr.Plot(label="failure-mode distribution")
            fail_table = gr.Dataframe(
                headers=["id", "embodiment", "scene", "failure_mode", "target", "narrative"],
                label="failures",
                wrap=True,
            )
            load_btn.click(
                load_failures,
                [source, aora_path, n, seed],
                [records_state, fail_table, mode_plot, load_note],
            )

        with gr.Tab("2 · Grow skills"):
            with gr.Row():
                provider = gr.Dropdown(
                    provider_choices,
                    value="mock",
                    label="LLM provider",
                    info=f"auto-detected = {default_provider}. 'mock' is free/offline; others make real API calls.",
                )
                max_skills = gr.Slider(
                    0, 12, value=4, step=1, label="max skills (0 = all clusters)"
                )
            grow_btn = gr.Button("Grow skills from the loaded failures", variant="primary")
            cost_md = gr.Markdown()
            with gr.Row():
                cluster_table = gr.Dataframe(
                    headers=["cluster", "title", "size", "suggested type", "embodiments"],
                    label="clusters",
                )
                skill_table = gr.Dataframe(
                    headers=["skill", "type", "validated", "score", "embodiments"],
                    label="forged skills",
                )
            skill_pick = gr.Dropdown([], label="inspect a skill's artifact")
            artifact_md = gr.Markdown()
            grow_btn.click(
                grow,
                [records_state, provider, max_skills],
                [store_state, cluster_table, skill_table, cost_md, skill_pick],
            )
            skill_pick.change(show_artifact, [store_state, skill_pick], artifact_md)

        with gr.Tab("3 · Retrieve & inject"):
            gr.Markdown(
                "Given a scene, which grown skills get retrieved and injected as planner tools?"
            )
            with gr.Row():
                objects = gr.Textbox("bed, chair", label="objects in the scene (comma-separated)")
                emb_pick = gr.Dropdown(
                    [e.value for e in Embodiment],
                    value=Embodiment.GROUND_HABITAT.value,
                    label="embodiment",
                )
            retr_btn = gr.Button("Retrieve relevant skills", variant="primary")
            retr_md = gr.Markdown()
            retr_table = gr.Dataframe(
                headers=["tool", "integration", "embodiments"], label="retrieved as planner tools"
            )
            tool_json = gr.Code(language="json", label="example tool spec the planner receives")
            retr_btn.click(
                retrieve, [store_state, objects, emb_pick], [retr_table, retr_md, tool_json]
            )

        with gr.Tab("4 · Robot view (real)"):
            gr.Markdown(
                "The **real** ground robot in Habitat HM3D, from AORA_v1 runs: trajectory plots "
                "(`path.png`) and the frames the agent captured as it failed (e.g. *giving_up_no_bed*). "
                "Live agent-cam / 3DGS scene rendering is the LEAD/AORA side (where headless-vs-windowed "
                "is a real toggle); AORA-Forge will render **3DGS reconstructions** here once C3 is built."
            )
            scan_path = gr.Textbox(_AORA_OUTPUTS, label="AORA_v1 outputs path")
            scan_btn = gr.Button("Scan for robot visuals", variant="primary")
            gallery = gr.Gallery(
                label="trajectories & captured failure frames", columns=4, height=560
            )
            scan_btn.click(scan_robot_artifacts, [scan_path], gallery)

    return demo
