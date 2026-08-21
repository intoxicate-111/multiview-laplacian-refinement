#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlr.baselines.exmesh_suite import prepare_nds_scene  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Adapt an audited ExMesh scene to the official NDS input layout."
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--scene-id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--image-mode", choices=("symlink", "copy"), default="symlink"
    )
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    manifest = prepare_nds_scene(
        contract,
        args.scene_id,
        args.output_dir,
        image_mode=args.image_mode,
    )
    print(
        json.dumps(
            {
                "output": str(args.output_dir / "adapter_manifest.json"),
                "scene_id": manifest["scene_id"],
                "num_views": manifest["num_views"],
                "image_bytes_unchanged": manifest["image_bytes_unchanged"],
                "contract_audit": manifest["contract_audit"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
