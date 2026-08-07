#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlr.learned_laplacian.canonical_report import finalize_canonical_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize the canonical Sofa50 report.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(
            "runs/learned_laplacian/sofa50_50mesh_2000epoch_absolute_h2_confidence"
        ),
    )
    args = parser.parse_args()
    summary = finalize_canonical_report(args.run_dir)
    print(json.dumps(summary["conclusions"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
