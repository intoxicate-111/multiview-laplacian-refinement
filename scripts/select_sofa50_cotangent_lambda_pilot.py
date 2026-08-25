#!/usr/bin/env python3
from __future__ import annotations

"""Select Cotangent lambda from matched short-run validation Chamfer only."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any


GRID = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0)


def _tag(value: float) -> str:
    return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", required=True, type=Path)
    parser.add_argument("--uniform-config", required=True, type=Path)
    parser.add_argument("--selected-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    for regularization in GRID:
        run = args.pilot_root / f"lambda_{_tag(regularization)}"
        metrics_path = run / "metrics.json"
        metrics = _read(metrics_path)
        validation = metrics["per_object_metrics"]["validation"]
        chamfers = [
            float(item["hybrid_chamfer"])
            for item in validation.values()
            if item.get("hybrid_chamfer") is not None
        ]
        if len(chamfers) != 50:
            raise RuntimeError(
                f"{metrics_path} contains {len(chamfers)} validation Chamfers, expected 50."
            )
        rows.append(
            {
                "lambda": regularization,
                "tag": _tag(regularization),
                "validation_mean_final_hybrid_chamfer": sum(chamfers) / len(chamfers),
                "validation_median_final_hybrid_chamfer": sorted(chamfers)[len(chamfers) // 2],
                "optimizer_steps": int(metrics["optimizer_steps"]),
                "best_training_selection_chamfer": float(metrics["best_selection_loss"]),
                "checkpoint_sha256_file": str(run / "checkpoint_best.pt"),
            }
        )
    selected = min(rows, key=lambda row: row["validation_mean_final_hybrid_chamfer"])
    uniform = _read(args.uniform_config.resolve())
    settings = uniform["training"]["hybrid_single_geometry_loss"]
    settings["operator"] = "symmetric_cotangent_stiffness"
    settings["cotangent_relative_area_epsilon"] = 1e-12
    settings["lambda"] = selected["lambda"]
    uniform["multi_object_training"]["gradient_accumulation_meshes"] = 1
    uniform["multi_object_training"]["max_optimizer_steps"] = 20000
    uniform["multi_object_training"]["checkpoint_optimizer_steps"] = [
        5000,
        10000,
        15000,
        20000,
    ]
    uniform["method"] = "sofa50_v2_cotangent_hybrid_single_loss"
    uniform["recovery"]["operator"] = (
        "symmetric_cotangent_stiffness_no_mass_normalization"
    )
    uniform["recovery"]["lambda"] = selected["lambda"]
    metadata = uniform["experiment_metadata"]
    metadata.update(
        {
            "experiment": "Sofa50_v2_uniform_vs_cotangent_single_loss_hybrid",
            "arm": f"cotangent_single_loss_hybrid_lambda{selected['tag']}",
            "operator": "symmetric_cotangent_stiffness",
            "operator_geometry_source": "input_coarse_mesh_vertices_and_faces",
            "mass_normalization": False,
            "negative_cotangent_weights_retained": True,
            "cotangent_relative_area_epsilon": 1e-12,
            "fixed_lambda_source": "validation_only_short_matched_pilot",
            "distributed_world_size": 8,
            "gradient_accumulation_meshes_per_rank": 1,
            "effective_global_batch_meshes": 8,
            "training_gpu_model": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "pilot": False,
            "pilot_optimizer_steps": rows[0]["optimizer_steps"],
            "validation_pilot_curve": rows,
        }
    )
    uniform["target_definition"] = (
        "latent_delta_pred_unsupervised;cotangent_operator_built_from_input_geometry"
    )
    uniform["target_semantics"] = (
        "both_output_branches_are_latent_and_receive_only_final_geometry_gradient"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "validation_lambda_curve.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    selection = {
        "contract_audit": True,
        "selection_split": "validation",
        "selection_metric": "mean final V_H unified-v2 Chamfer",
        "test_or_ood_used": False,
        "candidate_grid": list(GRID),
        "pilot_curve": rows,
        "selected_lambda": selected["lambda"],
        "selected_tag": selected["tag"],
        "selected_validation_chamfer": selected[
            "validation_mean_final_hybrid_chamfer"
        ],
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.selected_config.parent.mkdir(parents=True, exist_ok=True)
    args.selected_config.write_text(
        json.dumps(uniform, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(selection, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
