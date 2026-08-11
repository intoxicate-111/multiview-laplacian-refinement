#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from mlr.learned_laplacian.multi_dataset import (
    PreparedMeshDataset,
    validate_disjoint_splits,
)
from mlr.learned_laplacian.multi_trainer import train_multi_object


FREQUENCIES = (0, 2, 4, 6)


def summarize_completed_arms(output_root: Path) -> list[dict[str, object]]:
    summaries = []
    for num_frequencies in FREQUENCIES:
        path = output_root / f"k{num_frequencies}" / "ablation_summary.json"
        if path.is_file():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "ablation_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    return summaries


def experiment_config(
    base: dict[str, object],
    *,
    num_frequencies: int,
    max_optimizer_steps: int,
    num_workers: int | None = None,
) -> dict[str, object]:
    config = copy.deepcopy(base)
    config["seed"] = 7

    image_encoder = config.setdefault("image_encoder", {})
    image_encoder["feature_dim"] = 32
    image_encoder["first_stride"] = 2
    image_encoder["second_stride"] = 1

    model = config.setdefault("model", {})
    model["hidden_dim"] = 128
    model["num_graph_layers"] = 3
    model.pop("oracle_residual_expert", None)
    model["position_encoding"] = {
        "num_frequencies": int(num_frequencies),
        "include_input": True,
    }

    config.setdefault("query_training", {})["apply_to_validation"] = False
    config.setdefault("training", {})["vertex_sampling"] = {"mode": "full"}

    multi = config.setdefault("multi_object_training", {})
    multi["epochs"] = max(int(multi.get("epochs", 1)), max_optimizer_steps)
    multi["max_optimizer_steps"] = int(max_optimizer_steps)
    multi["checkpoint_every_epochs"] = 0
    multi["checkpoint_epochs"] = []

    if num_workers is not None:
        data_loading = config.setdefault("data_loading", {})
        data_loading["num_workers"] = int(num_workers)
        data_loading["persistent_workers"] = num_workers > 0
        data_loading["pin_memory"] = num_workers > 0

    config["position_encoding_ablation"] = {
        "arm": f"k{num_frequencies}",
        "num_frequencies": int(num_frequencies),
        "include_input": True,
        "k0_definition": "raw_normalized_xyz",
        "capacity": "C1",
        "image_feature_dim": 32,
        "graph_hidden_dim": 128,
        "graph_layers": 3,
        "feature_resolution": "F1",
        "image_first_stride": 2,
        "image_second_stride": 1,
        "input_image_size": 960,
        "views_per_sample": 14,
        "max_optimizer_steps": int(max_optimizer_steps),
        "seed": 7,
    }
    return config


def run_arm(
    manifest_path: Path,
    base_config_path: Path,
    output_root: Path,
    *,
    num_frequencies: int,
    max_optimizer_steps: int,
    device: str,
    num_workers: int | None,
) -> dict[str, object]:
    arm_dir = output_root / f"k{num_frequencies}"
    if arm_dir.exists() and any(arm_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {arm_dir}")
    arm_dir.mkdir(parents=True, exist_ok=True)

    base = json.loads(base_config_path.read_text(encoding="utf-8"))
    config = experiment_config(
        base,
        num_frequencies=num_frequencies,
        max_optimizer_steps=max_optimizer_steps,
        num_workers=num_workers,
    )
    (arm_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    train_dataset = PreparedMeshDataset.from_manifest(manifest_path, "train")
    validation_dataset = PreparedMeshDataset.from_manifest(manifest_path, "validation")
    validate_disjoint_splits(train_dataset, validation_dataset)
    result = train_multi_object(
        train_dataset,
        validation_dataset,
        config,
        output_dir=arm_dir,
        device_override=device,
    )
    summary: dict[str, object] = {
        "arm": f"k{num_frequencies}",
        "num_frequencies": num_frequencies,
        "optimizer_steps": result.optimizer_steps,
        "completed_epochs": result.completed_epochs,
        "best_epoch": result.best_epoch,
        "best_validation_loss": result.best_selection_loss,
        "final_train_loss": result.final_train_loss,
        "final_validation_loss": result.final_validation_loss,
        "runtime_seconds": result.runtime_seconds,
        "peak_gpu_memory_mb": result.peak_gpu_memory_mb,
    }
    (arm_dir / "ablation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Sofa50 C1F1 Fourier-frequency ablation."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--frequencies", nargs="+", type=int, choices=FREQUENCIES, default=FREQUENCIES
    )
    parser.add_argument("--max-optimizer-steps", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--skip-root-summary", action="store_true")
    args = parser.parse_args()

    if args.summarize_only:
        print(json.dumps(summarize_completed_arms(args.output_root.resolve()), indent=2))
        return 0
    if args.max_optimizer_steps < 1:
        parser.error("--max-optimizer-steps must be positive")
    if args.num_workers is not None and args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    summaries = []
    for num_frequencies in args.frequencies:
        summaries.append(
            run_arm(
                args.manifest.resolve(),
                args.base_config.resolve(),
                args.output_root.resolve(),
                num_frequencies=num_frequencies,
                max_optimizer_steps=args.max_optimizer_steps,
                device=args.device,
                num_workers=args.num_workers,
            )
        )
    if not args.skip_root_summary:
        summaries = summarize_completed_arms(args.output_root.resolve())
    print(json.dumps(summaries, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
