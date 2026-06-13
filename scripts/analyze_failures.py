#!/usr/bin/env python3
"""Read failure logs (or synthetic failures), cluster them, and write the
resulting ``FailureCluster``s to JSONL.

    python scripts/analyze_failures.py --in failures.jsonl --out clusters.jsonl
    python scripts/analyze_failures.py --synthetic 50          # no logs needed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aora_forge.failures.clusterer import cluster_failures  # noqa: E402
from aora_forge.failures.collector import collect_from_jsonl  # noqa: E402
from aora_forge.llm.client import get_llm_client  # noqa: E402
from aora_forge.utils.logging import RunTelemetry  # noqa: E402
from aora_forge.utils.synthetic_data import generate_synthetic_failures  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="infile", default=None, help="failures JSONL")
    ap.add_argument("--synthetic", type=int, default=None, help="use N synthetic failures instead")
    ap.add_argument("--out", default="clusters.jsonl", help="output clusters JSONL")
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
    clusters = cluster_failures(records, client, telemetry=tel)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for c in clusters:
            f.write(c.model_dump_json() + "\n")

    print(f"{len(records)} failures -> {len(clusters)} clusters -> {out}")
    for c in clusters:
        print(f"  {c.cluster_id}: {c.title}  (n={c.size}, type={c.suggested_skill_type.value})")
    for line in tel.summary_lines():
        print("  " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
