#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
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
    _write_run_metadata(args.output_dir, args.manifest, config)
    train_dataset = PreparedMeshDataset.from_manifest(args.manifest, "train")
    validation_dataset = PreparedMeshDataset.from_manifest(args.manifest, "validation")
    loading_start = time.perf_counter()
    validate_disjoint_splits(train_dataset, validation_dataset)
    train_samples = tuple(train_dataset)
    validation_samples = tuple(validation_dataset)
    loading_seconds = time.perf_counter() - loading_start
    print(
        f"loaded {len(train_samples)} training meshes and "
        f"{len(validation_samples)} validation meshes in {loading_seconds:.2f} seconds",
        flush=True,
    )
    result = train_multi_object(
        train_samples,
        validation_samples,
        config,
        output_dir=args.output_dir,
        device_override=args.device,
        input_mode_override=args.input_mode,
        zero_images=args.zero_images,
        initial_loading_seconds=loading_seconds,
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
        "initial_loading_seconds": result.initial_loading_seconds,
        "static_preparation_seconds": result.static_preparation_seconds,
        "device_cache_seconds": result.device_cache_seconds,
        "mean_epoch_train_seconds": result.mean_epoch_train_seconds,
        "mean_validation_seconds": result.mean_validation_seconds,
        "peak_gpu_memory_mb": result.peak_gpu_memory_mb,
        "initial_learning_rate": result.initial_learning_rate,
        "final_learning_rate": result.final_learning_rate,
        "lr_scheduler_type": result.lr_scheduler_type,
        "lr_reduction_count": result.lr_reduction_count,
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(summary, indent=2))
    return 0


def _write_run_metadata(output_dir: Path, manifest_path: Path, config: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("samples"), list):
        raise ValueError("Manifest must be an object containing a 'samples' list.")
    portable_manifest = dict(manifest)
    portable_samples = []
    for item in manifest["samples"]:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("Each manifest sample must be an object with a path.")
        sample = dict(item)
        path = Path(sample["path"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        sample["path"] = str(path.resolve())
        portable_samples.append(sample)
    portable_manifest["samples"] = portable_samples
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "experiment_config": config,
                "manifest_path": "dataset_manifest.json",
                "source_manifest": str(manifest_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(portable_manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
