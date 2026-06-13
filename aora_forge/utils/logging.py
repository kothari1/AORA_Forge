"""Lightweight logging + run-level telemetry.

Two things live here:

* ``get_logger`` — a thin wrapper over the stdlib logger with a consistent format,
  so every module logs the same way without re-configuring handlers.
* ``RunTelemetry`` — an accumulator for LLM usage + wall-clock + per-stage counts,
  used by ``scripts/demo_full_loop.py`` to print the "what this run cost" summary
  the morning report wants.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field

from aora_forge.schemas import LLMUsage

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring the root handler once."""
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s", "%H:%M:%S")
        )
        root = logging.getLogger("aora_forge")
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(f"aora_forge.{name}")


@dataclass
class RunTelemetry:
    """Accumulates LLM usage and stage timings across a run.

    Designed to be passed down through the pipeline so every LLM call's
    ``LLMUsage`` lands in one place, then summarised at the end.
    """

    total_usage: LLMUsage = field(default_factory=lambda: LLMUsage(model="none", mocked=True))
    stage_calls: dict[str, int] = field(default_factory=dict)
    stage_usd: dict[str, float] = field(default_factory=dict)
    _t0: float = field(default_factory=time.monotonic)

    def record(self, stage: str, usage: LLMUsage) -> None:
        self.total_usage = self.total_usage + usage
        self.stage_calls[stage] = self.stage_calls.get(stage, 0) + 1
        self.stage_usd[stage] = self.stage_usd.get(stage, 0.0) + usage.cost_usd

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self._t0

    @property
    def all_mocked(self) -> bool:
        return self.total_usage.mocked

    def summary_lines(self) -> list[str]:
        """Human-readable summary block for the run report."""
        u = self.total_usage
        mode = "MOCK (deterministic, no key)" if self.all_mocked else "REAL LLM API"
        lines = [
            f"LLM backend          : {mode}",
            f"Wall-clock           : {self.elapsed_s:6.1f} s",
            f"Total input tokens   : {u.input_tokens:,}",
            f"  cache read         : {u.cache_read_input_tokens:,}",
            f"  cache creation     : {u.cache_creation_input_tokens:,}",
            f"Total output tokens  : {u.output_tokens:,}",
            f"Estimated cost (USD) : ${u.cost_usd:,.4f}",
            "Calls by stage:",
        ]
        for stage in sorted(self.stage_calls):
            lines.append(
                f"  {stage:<28} {self.stage_calls[stage]:>3} call(s)  "
                f"${self.stage_usd.get(stage, 0.0):.4f}"
            )
        return lines
