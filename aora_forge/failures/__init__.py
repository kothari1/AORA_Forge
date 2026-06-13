"""Failure ingestion and clustering — contribution C2.

``taxonomy``  : the unified FailureMode classifier + theme/skill mapping.
``collector`` : read FailureRecords from LEAD/AORA logs (nav.jsonl, orchestrator.jsonl).
``clusterer`` : LLM-driven clustering of failures into themed FailureClusters.
"""

from aora_forge.failures.taxonomy import (
    FAILURE_THEME_HINTS,
    classify_from_signals,
    normalize_failure_mode,
    suggest_skill_type,
)

__all__ = [
    "classify_from_signals",
    "normalize_failure_mode",
    "suggest_skill_type",
    "FAILURE_THEME_HINTS",
]
