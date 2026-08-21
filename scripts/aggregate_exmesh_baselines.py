#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlr.baselines.exmesh_suite import aggregate_results, load_suite_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate auditable ExMesh-protocol per-scene status files."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    payload = aggregate_results(load_suite_config(args.config), args.output_root)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
