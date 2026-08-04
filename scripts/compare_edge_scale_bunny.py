#!/usr/bin/env python3
"""Build a concise raw-vs-edge-normalized Bunny comparison report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlr.learned_laplacian.dataset import load_prepared_sample
from mlr.learned_laplacian.losses import laplacian_prediction_metrics
from mlr.learned_laplacian.target_scaling import normalize_laplacian_by_edge_scale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--normalized-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    args = parser.parse_args()

    sample = load_prepared_sample(args.sample)
    h = sample["local_edge_length"]
    report = {
        "sample": str(args.sample),
        "epsilon": args.epsilon,
        "isolated_vertices": int((h == 0).sum().item()),
        "comparison": {},
    }
    for mode in ("coarse_only", "coarse_plus_multiview"):
        raw = _load_run(args.raw_root / mode, h, args.epsilon)
        normalized = _load_run(args.normalized_root / mode, h, args.epsilon)
        report["comparison"][mode] = {
            "raw_target_training": raw,
            "edge_scale_normalized_target_training": normalized,
            "normalized_over_raw_ratios": {
                "raw_laplacian_mse": _ratio(
                    normalized["laplacian_prediction_raw"]["mse"],
                    raw["laplacian_prediction_raw"]["mse"],
                ),
                "point_to_surface_mean": _ratio(
                    normalized["geometry"]["point_to_surface_mean"],
                    raw["geometry"]["point_to_surface_mean"],
                ),
                "target_position_rmse": _ratio(
                    normalized["geometry"]["target_position_rmse"],
                    raw["geometry"]["target_position_rmse"],
                ),
                "bbox_diagonal_ratio": _ratio(
                    normalized["geometry"]["bbox_diagonal_ratio_to_coarse"],
                    raw["geometry"]["bbox_diagonal_ratio_to_coarse"],
                ),
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def _load_run(run_dir: Path, h: torch.Tensor, epsilon: float) -> dict:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    raw_target = torch.from_numpy(np.load(run_dir / "delta_target.npy"))
    raw_prediction = torch.from_numpy(np.load(run_dir / "delta_pred.npy"))
    normalized_target = normalize_laplacian_by_edge_scale(raw_target, h, eps=epsilon)
    normalized_prediction = normalize_laplacian_by_edge_scale(raw_prediction, h, eps=epsilon)
    normalized_metrics = laplacian_prediction_metrics(normalized_prediction, normalized_target)
    predicted = metrics["geometry"]["predicted"]
    return {
        "target_mode": metrics.get("target_mode", "raw_laplacian"),
        "training": metrics["training"],
        "laplacian_prediction_raw": metrics.get(
            "laplacian_prediction_raw", metrics["laplacian_prediction"]
        ),
        "laplacian_prediction_normalized": metrics.get(
            "laplacian_prediction_normalized", normalized_metrics
        ),
        "geometry": {
            "point_to_surface_mean": predicted["point_to_surface_mean"],
            "point_to_surface_engine": predicted["point_to_surface_engine"],
            "target_position_rmse": predicted["target_position_rmse"],
            "chamfer": predicted["chamfer"],
            "normal_consistency": predicted["normal_consistency"],
            "bbox_diagonal_ratio_to_coarse": predicted["bbox_diagonal_ratio_to_coarse"],
            "collapsed_or_exploded": predicted["collapsed_or_exploded"],
        },
        "predicted_improves_over_coarse": metrics["predicted_improves_over_coarse"],
    }


def _ratio(value: float, reference: float) -> float:
    return float(value) / max(float(reference), 1e-300)


if __name__ == "__main__":
    main()
