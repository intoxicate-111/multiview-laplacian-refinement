#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlr.learned_laplacian.multi_dataset import (
    PreparedMeshDataset,
    validate_disjoint_splits,
)
from mlr.learned_laplacian.multi_trainer import train_multi_object


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train one shared learned-Laplacian model over variable-topology meshes."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument(
        "--input-mode",
        choices=["coarse_only", "multiview_only", "coarse_plus_multiview"],
    )
    parser.add_argument("--zero-images", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    train_dataset = PreparedMeshDataset.from_manifest(args.manifest, "train")
    validation_dataset = PreparedMeshDataset.from_manifest(args.manifest, "validation")
    validate_disjoint_splits(train_dataset, validation_dataset)
    result = train_multi_object(
        train_dataset,
        validation_dataset,
        config,
        output_dir=args.output_dir,
        device_override=args.device,
        input_mode_override=args.input_mode,
        zero_images=args.zero_images,
    )
    summary = {
        "train_meshes": len(train_dataset),
        "validation_meshes": len(validation_dataset),
        "best_epoch": result.best_epoch,
        "best_selection_loss": result.best_selection_loss,
        "final_train_loss": result.final_train_loss,
        "final_validation_loss": result.final_validation_loss,
        "optimizer_steps": result.optimizer_steps,
        "target_mode": result.target_mode,
        "device": result.device,
        "runtime_seconds": result.runtime_seconds,
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
