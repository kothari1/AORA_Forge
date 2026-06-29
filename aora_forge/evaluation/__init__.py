"""A/B evaluation: does injecting a grown skill improve AORA_v1's real performance?

``ab_runner`` drives AORA_v1 (in its own ``aora_v1`` conda env, via subprocess)
twice over a fixed episode set — baseline vs. with a grown executor-prompt skill
prepended — then compares SSR / SPL / FPR / failure-mode deltas. This is the
empirical test of contribution C2: skills grown from real failures should reduce
those same failures.
"""

from aora_forge.evaluation.ab_runner import ABResult, BatchMetrics, run_ab

__all__ = ["run_ab", "ABResult", "BatchMetrics"]
