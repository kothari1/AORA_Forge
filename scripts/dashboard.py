#!/usr/bin/env python3
"""Launch the AORA-Forge visual dashboard.

    python scripts/dashboard.py                       # localhost:7861
    python scripts/dashboard.py --host 0.0.0.0        # reachable over Tailscale/LAN
    ssh -L 7861:localhost:7861 jugg                   # then open http://localhost:7861

On jugg you can also just open http://localhost:7861 in a browser inside NoMachine.
Defaults to localhost-only (safer). Requires:  pip install -e ".[viz]"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aora_forge.viz.dashboard import build_dashboard  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--host", default="127.0.0.1", help="bind address (use 0.0.0.0 for Tailscale/LAN)"
    )
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--share", action="store_true", help="create a public Gradio share link")
    args = ap.parse_args()

    print(f"Launching AORA-Forge dashboard on http://{args.host}:{args.port}")
    print("  • inside NoMachine on jugg: open that URL in a browser")
    print(
        f"  • from your laptop: ssh -L {args.port}:localhost:{args.port} jugg, then http://localhost:{args.port}"
    )
    demo = build_dashboard()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
