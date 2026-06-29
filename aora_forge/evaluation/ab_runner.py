"""Drive AORA_v1 baseline-vs-grown-skill and compare the real metrics.

AORA_v1 lives in its own conda env (``aora_v1``, py3.9, habitat-sim) and cannot be
imported from AORA-Forge's env, so we drive it via ``conda run -n aora_v1 python
scripts/run_objectnav.py`` over a fixed episode set: once plain (baseline), once
with ``--skill-prompt-file`` (the grown executor-prompt skill prepended to the
executor system prompt — the clean hook we added to AORA_v1). Both runs use the
minival config (episodes on the downloaded 00800/00802 scenes), identical args, so
the episode set is identical and the only difference is the injected skill.

Then we parse each batch's ``episodes.jsonl`` and report deltas:
- **SSR** success rate, **SPL**, **FPR** (vlm_claimed_success but not actually
  successful — the false-positive "done" our arrival-verifier skill targets), and
  the per-failure-mode histogram.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from aora_forge.schemas import Skill
from aora_forge.utils.logging import get_logger

log = get_logger("evaluation.ab_runner")

AORA_V1_ROOT = Path(os.environ.get("AORA_V1_ROOT", "/home/admin/projects/AORA_v1"))
AORA_V1_ENV = os.environ.get("AORA_V1_ENV", "aora_v1")
MINIVAL_CONFIG = "aora/configs/objectnav_hm3d_minival_ab.yaml"


@dataclass
class BatchMetrics:
    """Aggregate metrics over one batch's episodes.jsonl."""

    label: str
    n: int = 0
    ssr: float = 0.0  # success rate
    spl: float = 0.0  # mean SPL (0 on failures)
    fpr: float = 0.0  # claimed success but failed (false-positive arrival)
    mean_steps: float = 0.0
    failure_modes: dict[str, int] = field(default_factory=dict)
    out_dir: str = ""

    @classmethod
    def from_episodes(cls, jsonl: Path, label: str) -> BatchMetrics:
        recs = []
        if jsonl.exists():
            for line in jsonl.read_text().splitlines():
                line = line.strip()
                if line:
                    with contextlib.suppress(json.JSONDecodeError):
                        recs.append(json.loads(line))
        n = len(recs)
        if n == 0:
            return cls(label=label, out_dir=str(jsonl.parent))
        succ = sum(1 for r in recs if r.get("success"))
        spls = [r.get("spl") for r in recs if isinstance(r.get("spl"), (int, float))]
        fpr = sum(1 for r in recs if r.get("vlm_claimed_success") and not r.get("success"))
        steps = [r.get("steps") for r in recs if isinstance(r.get("steps"), (int, float))]
        return cls(
            label=label,
            n=n,
            ssr=round(succ / n, 3),
            spl=round(sum(spls) / n, 3) if spls else 0.0,
            fpr=round(fpr / n, 3),
            mean_steps=round(sum(steps) / len(steps), 1) if steps else 0.0,
            failure_modes=dict(Counter(r.get("failure_mode", "?") for r in recs)),
            out_dir=str(jsonl.parent),
        )


@dataclass
class ABResult:
    skill_name: str
    baseline: BatchMetrics
    with_skill: BatchMetrics
    episodes: int
    category: str | None
    note: str = ""

    def deltas(self) -> dict[str, float]:
        b, s = self.baseline, self.with_skill
        return {
            "ssr": round(s.ssr - b.ssr, 3),
            "spl": round(s.spl - b.spl, 3),
            "fpr": round(s.fpr - b.fpr, 3),
            "mean_steps": round(s.mean_steps - b.mean_steps, 1),
        }

    def table_rows(self) -> list[list[str]]:
        d = self.deltas()
        return [
            ["episodes", str(self.baseline.n), str(self.with_skill.n), ""],
            [
                "success rate (SSR)",
                f"{self.baseline.ssr:.3f}",
                f"{self.with_skill.ssr:.3f}",
                f"{d['ssr']:+.3f}",
            ],
            ["SPL", f"{self.baseline.spl:.3f}", f"{self.with_skill.spl:.3f}", f"{d['spl']:+.3f}"],
            [
                "false-positive done (FPR)",
                f"{self.baseline.fpr:.3f}",
                f"{self.with_skill.fpr:.3f}",
                f"{d['fpr']:+.3f}",
            ],
            [
                "mean steps",
                f"{self.baseline.mean_steps:.1f}",
                f"{self.with_skill.mean_steps:.1f}",
                f"{d['mean_steps']:+.1f}",
            ],
        ]

    def to_json(self) -> dict:
        return {
            "skill_name": self.skill_name,
            "episodes": self.episodes,
            "category": self.category,
            "baseline": asdict(self.baseline),
            "with_skill": asdict(self.with_skill),
            "deltas": self.deltas(),
            "note": self.note,
        }


def _run_batch(
    *,
    label: str,
    skill_prompt_file: str | None,
    category: str | None,
    limit: int,
    max_usd: float,
    mode: str,
    live_dir: str | None,
    timeout_s: int,
) -> Path:
    """Run one AORA_v1 batch via subprocess in the aora_v1 env; return episodes.jsonl path."""
    cmd = [
        "conda",
        "run",
        "-n",
        AORA_V1_ENV,
        "python",
        "scripts/run_objectnav.py",
        "--config",
        MINIVAL_CONFIG,
        "--mode",
        mode,
        "--limit",
        str(limit),
        "--no-video",
        "--max-usd",
        str(max_usd),
        "-q",
    ]
    if category:
        cmd += ["--category", category]
    if skill_prompt_file:
        cmd += ["--skill-prompt-file", skill_prompt_file]
    if live_dir:
        cmd += ["--live-dir", live_dir]

    log.info(
        "[%s] running AORA_v1: limit=%d category=%s skill=%s",
        label,
        limit,
        category,
        bool(skill_prompt_file),
    )
    proc = subprocess.run(
        cmd, cwd=str(AORA_V1_ROOT), capture_output=True, text=True, timeout=timeout_s
    )
    out = proc.stdout + "\n" + proc.stderr
    if proc.returncode != 0:
        log.warning("[%s] AORA_v1 returned %d", label, proc.returncode)
    # parse "Batch output: <dir>" from stdout
    m = re.search(r"Batch output:\s*(\S+)", out)
    if m:
        batch_dir = Path(m.group(1))
    else:
        # fallback: newest batch dir under the minival output root
        root = AORA_V1_ROOT / "outputs" / "objectnav_minival"
        batches = sorted(root.glob("batch_*"), key=lambda p: p.stat().st_mtime)
        batch_dir = batches[-1] if batches else root
    log.info("[%s] batch dir: %s", label, batch_dir)
    return batch_dir / "episodes.jsonl"


def run_ab(
    skill: Skill | str,
    *,
    category: str | None = None,
    limit: int = 6,
    max_usd: float = 30.0,
    mode: str = "executor_only",
    live_dir: str | None = None,
    timeout_s: int = 3600,
) -> ABResult:
    """Run baseline vs. with-skill over a fixed minival episode set and compare.

    ``skill`` may be a ``Skill`` (its prompt artifact is used) or a raw prompt
    string. ``category`` restricts to one ObjectNav category (else the first
    ``limit`` episodes across categories — deterministic, identical for both runs).
    """
    if isinstance(skill, Skill):
        skill_name = skill.skill_name
        prompt_text = skill.artifact_inline or ""
    else:
        skill_name = "custom_prompt"
        prompt_text = skill
    if not prompt_text.strip():
        raise ValueError("empty skill prompt")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
        tf.write(prompt_text)
        skill_file = tf.name

    baseline_jsonl = _run_batch(
        label="baseline",
        skill_prompt_file=None,
        category=category,
        limit=limit,
        max_usd=max_usd,
        mode=mode,
        live_dir=live_dir,
        timeout_s=timeout_s,
    )
    skill_jsonl = _run_batch(
        label="with_skill",
        skill_prompt_file=skill_file,
        category=category,
        limit=limit,
        max_usd=max_usd,
        mode=mode,
        live_dir=live_dir,
        timeout_s=timeout_s,
    )

    baseline = BatchMetrics.from_episodes(baseline_jsonl, "baseline")
    with_skill = BatchMetrics.from_episodes(skill_jsonl, f"+{skill_name}")
    result = ABResult(
        skill_name=skill_name,
        baseline=baseline,
        with_skill=with_skill,
        episodes=baseline.n,
        category=category,
        note=f"executor-prompt skill '{skill_name}' prepended to AORA_v1 executor; "
        f"minival/{mode}; identical {limit}-episode set.",
    )
    log.info(
        "A/B done: SSR %.3f->%.3f, FPR %.3f->%.3f",
        baseline.ssr,
        with_skill.ssr,
        baseline.fpr,
        with_skill.fpr,
    )
    return result
