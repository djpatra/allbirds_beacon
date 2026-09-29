#!/usr/bin/env python3
"""Entry point: run all 6 merchant queries (N=5) and persist answers."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from merchant_rag.cli import main

if __name__ == "__main__":
    # default to --all if no args
    if len(sys.argv) == 1:
        sys.argv.extend([
            "--all",
            "--n", "5",
            "--out", str(ROOT / "demand" / "merchant_answers.json"),
        ])
    raise SystemExit(main())
