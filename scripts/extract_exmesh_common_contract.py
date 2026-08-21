#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlr.baselines.exmesh_suite import (  # noqa: E402
    extract_common_contract,
    load_suite_config,
    save_common_contract,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract the official ExMesh DTU observation/camera/evaluation contract."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dtu-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_suite_config(args.config)
    contract = extract_common_contract(config, args.dtu_root)
    output = save_common_contract(contract, args.output)
    print(json.dumps({"output": str(output), "contract_audit": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
