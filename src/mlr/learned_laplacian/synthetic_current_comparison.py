from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mlr.io import load_mesh

from .canonical_experiment import _exact_query_sample, _load_device_item, _topology_change
from .canonical_pipeline import canonical_current_graph_recovery_inputs
from .diagnostics import _amp_settings, _loss_kwargs
from .evaluation import reconstruct_and_evaluate
from .losses import laplacian_prediction_metrics, weighted_robust_laplacian_loss
from .multi_dataset import PreparedMeshDataset, validate_disjoint_splits
from .multi_trainer import _build_model
from .trainer import load_checkpoint


CONDITIONS = ("correct_rgb", "zero_rgb")
MODELS = ("A", "B")


def run_synthetic_current_comparison(
    manifest_path: str | Path,
    a_checkpoint: str | Path,
    a_config_path: str | Path,
    b_checkpoint: str | Path,
    b_config_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but torch.cuda.is_available() is false.")
    datasets = {
        split: PreparedMeshDataset.from_manifest(manifest_path, split)
        for split in ("train", "validation", "test")
    }
    validate_disjoint_splits(*datasets.values())
    dataset = datasets["test"]
    if len(dataset) != 25:
        raise ValueError(f"Expected 25 synthetic-current test variants, found {len(dataset)}.")
    object_ids = [
        str(dataset.load_static(index).get("metadata", {}).get("object_id", ""))
        for index in range(len(dataset))
    ]
    if len(set(object_ids)) != 5 or any(object_ids.count(value) != 5 for value in set(object_ids)):
        raise ValueError("Test split must contain five objects with five variants each.")

    model_specs = {
        "A": {
            "checkpoint": Path(a_checkpoint).resolve(),
            "config_path": Path(a_config_path).resolve(),
            "training_formulation": "GT-query",
            "training_status": "existing / frozen",
            "standard": "existing canonical C2 / F2 / 14 VIEWS",
            "training_rerun": "NO",
            "native_metrics": str(Path(a_config_path).resolve().with_name("metrics.json")),
        },
        "B": {
            "checkpoint": Path(b_checkpoint).resolve(),
            "config_path": Path(b_config_path).resolve(),
            "training_formulation": "current-query",
            "training_status": "newly trained",
            "standard": "C2 / F2 / 14 VIEWS",
            "training_rerun": "YES",
            "native_metrics": str(Path(b_checkpoint).resolve().with_name("metrics.json")),
        },
    }
    for spec in model_specs.values():
        spec["config"] = _read_json(spec["config_path"])
        model = _build_model(spec["config"], None, False).to(resolved_device)
        spec["checkpoint_payload"] = load_checkpoint(
            spec["checkpoint"], model, map_location=resolved_device
        )
        spec["training_seed"] = int(spec["config"].get("seed", -1))
        spec["training_budget_optimizer_steps"] = int(
            spec["config"].get("multi_object_training", {}).get(
                "max_optimizer_steps", -1
            )
        )
        spec["checkpoint_epoch"] = int(
            spec["checkpoint_payload"].get("epoch", -1)
        )
        model.eval()
        spec["model"] = model
        spec["amp_enabled"], spec["amp_dtype"] = _amp_settings(
            spec["config"], resolved_device
        )

    preparation_config = dict(model_specs["B"]["config"])
    per_variant: list[dict[str, Any]] = []
    aggregate_arrays: dict[tuple[str, str], dict[str, list[torch.Tensor]]] = {
        (model_name, condition): {"prediction": [], "target": []}
        for model_name in MODELS
        for condition in CONDITIONS
    }
    for index in range(len(dataset)):
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        metadata = dict(static.get("metadata", {}))
        object_id = str(metadata["object_id"])
        variant_index = int(metadata["variant_index"])
        prepared = _load_device_item(dataset, index, preparation_config, resolved_device)
        base = _exact_query_sample(prepared.sample, resolved_device)
        target = prepared.training_target.float()
        valid = prepared.sample["valid_scale_mask"].to(dtype=torch.bool)
        for model_name in MODELS:
            spec = model_specs[model_name]
            condition_outputs: dict[str, dict[str, Any]] = {}
            for condition in CONDITIONS:
                conditioned = dict(base)
                if condition == "zero_rgb":
                    conditioned["images"] = torch.zeros_like(base["images"])
                with torch.no_grad(), torch.autocast(
                    device_type=resolved_device.type,
                    dtype=spec["amp_dtype"],
                    enabled=spec["amp_enabled"],
                ):
                    output = spec["model"](conditioned)
                prediction = output.delta_hat_prediction.float()
                metrics = laplacian_prediction_metrics(prediction, target, valid_mask=valid)
                valid_prediction = prediction[valid].detach().cpu()
                valid_target = target[valid].detach().cpu()
                aggregate_arrays[(model_name, condition)]["prediction"].append(valid_prediction)
                aggregate_arrays[(model_name, condition)]["target"].append(valid_target)
                loss = weighted_robust_laplacian_loss(
                    valid_prediction,
                    valid_target,
                    torch.ones(len(valid_prediction)),
                    **_loss_kwargs(preparation_config),
                )
                condition_outputs[condition] = {
                    "prediction": prediction.detach().cpu(),
                    "confidence": (
                        None
                        if output.confidence_prediction is None
                        else output.confidence_prediction.float().detach().cpu()
                    ),
                    "metrics": metrics,
                    "loss": float(loss),
                }
            correct = condition_outputs["correct_rgb"]
            zero = condition_outputs["zero_rgb"]
            if correct["confidence"] is None:
                raise RuntimeError(f"Experiment {model_name} checkpoint has no confidence output.")
            recovery = _reconstruct_one(
                static,
                correct["prediction"],
                correct["confidence"],
                output_dir / "reconstruction" / model_name / sample_id,
                preparation_config,
            )
            row = {
                "experiment": model_name,
                "sample_id": sample_id,
                "object_id": object_id,
                "variant_index": variant_index,
                "vertex_count": int(valid.sum().item()),
                "correct_rgb_loss": correct["loss"],
                "zero_rgb_loss": zero["loss"],
                "correct_zero_loss_gap": zero["loss"] - correct["loss"],
                "normalized_mse": correct["metrics"]["mse"],
                "vector_l2": correct["metrics"]["vector_endpoint_error"],
                "global_cosine": correct["metrics"]["global_cosine"],
                "high_10_percent_cosine": correct["metrics"]["top_10_percent_cosine"],
                "prediction_target_norm_ratio": correct["metrics"][
                    "prediction_to_target_norm_ratio"
                ],
                "zero_rgb_normalized_mse": zero["metrics"]["mse"],
                "zero_rgb_global_cosine": zero["metrics"]["global_cosine"],
                "correct_zero_cosine_gap": correct["metrics"]["global_cosine"]
                - zero["metrics"]["global_cosine"],
                **recovery,
            }
            per_variant.append(row)
            print(
                f"{model_name} {sample_id}: mse={row['normalized_mse']:.8g} "
                f"chamfer={row['reconstruction_chamfer']:.8g} "
                f"improved={row['improved_over_initial']}",
                flush=True,
            )
        del prepared, base, target
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()

    aggregate = _aggregate(preparation_config, aggregate_arrays, per_variant)
    per_object = _aggregate_per_object(per_variant)
    summary = {
        "comparison_type": "existing_canonical_baseline_vs_new_formulation_not_strict_paired_training",
        "manifest": str(manifest_path),
        "test_domain": "same synthetic C-query test split",
        "test_samples": len(dataset),
        "test_objects": len(set(object_ids)),
        "target": "delta_target_hat=(L_current@P_proxy)/(h_current^2+1e-12)",
        "experiment_setup": {
            model_name: {
                key: value
                for key, value in spec.items()
                if key
                in {
                    "checkpoint",
                    "config_path",
                    "training_formulation",
                    "training_status",
                    "standard",
                    "training_rerun",
                    "training_seed",
                    "training_budget_optimizer_steps",
                    "checkpoint_epoch",
                    "native_metrics",
                }
            }
            for model_name, spec in model_specs.items()
        },
        "aggregate": aggregate,
        "per_object": per_object,
        "per_variant": per_variant,
    }
    _write_json(output_dir / "comparison.json", summary)
    _write_csv(output_dir / "per_variant.csv", per_variant)
    _write_csv(output_dir / "per_object.csv", per_object)
    (output_dir / "comparison.md").write_text(
        _markdown_report(summary), encoding="utf-8"
    )
    return summary


def _reconstruct_one(
    static: Mapping[str, Any],
    delta_hat_prediction: torch.Tensor,
    confidence: torch.Tensor,
    output_dir: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    epsilon = float(config.get("target_scaling", {}).get("epsilon", 1e-12))
    visibility = static["visibility_backface_and_occlusion"]
    inputs = canonical_current_graph_recovery_inputs(
        static["vertices"],
        static["faces"],
        delta_hat_prediction,
        visibility,
        confidence,
        epsilon=epsilon,
    )
    recovery_config = dict(config.get("recovery", {}))
    recovery_config.update(
        {
            "dense_vertex_limit": 5000,
            "chamfer_samples": 3000,
            "metric_seed": 7,
            "evaluate_oracle": False,
        }
    )
    metrics = reconstruct_and_evaluate(
        static,
        inputs.delta_pred_raw,
        output_dir,
        recovery_config,
        normalized_prediction=inputs.delta_hat_prediction,
        edge_scale_epsilon=epsilon,
        laplacian_weight=inputs.weight,
        unseen_anchor_weight=float(recovery_config.get("unseen_anchor_weight", 0.0)),
        evaluate_laplacian_prediction=True,
        evaluate_initial_geometry=True,
        solver_confidence=np.ones(len(inputs.delta_pred_raw), dtype=np.float64),
    )
    recovered = load_mesh(output_dir / "predicted_refined.obj")
    initial = static["vertices"].detach().cpu().numpy()
    faces = static["faces"].detach().cpu().numpy()
    topology = _topology_change(initial, recovered.vertices, faces)
    initial_geometry = metrics["geometry"]["coarse"]
    geometry = metrics["geometry"]["predicted"]
    initial_chamfer = float(initial_geometry["chamfer"])
    reconstruction_chamfer = float(geometry["chamfer"])
    return {
        "initial_chamfer": initial_chamfer,
        "reconstruction_chamfer": reconstruction_chamfer,
        "initial_point_to_surface": float(
            initial_geometry["point_to_surface_bidirectional_mean"]
        ),
        "reconstruction_point_to_surface": float(
            geometry["point_to_surface_bidirectional_mean"]
        ),
        "initial_normal_consistency": float(initial_geometry["normal_consistency"]),
        "reconstruction_normal_consistency": float(geometry["normal_consistency"]),
        "introduced_flipped_faces": int(topology["introduced_flips"]),
        "new_degenerate_faces": int(topology["new_degeneracies"]),
        "improved_over_initial": bool(reconstruction_chamfer < initial_chamfer),
        "mean_confidence": float(inputs.confidence_prediction.mean()),
        "visible_vertex_ratio": float(inputs.visible.float().mean()),
    }


def _aggregate(
    evaluation_config: Mapping[str, Any],
    arrays: Mapping[tuple[str, str], Mapping[str, Sequence[torch.Tensor]]],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result = {}
    for model_name in MODELS:
        condition_values = {}
        for condition in CONDITIONS:
            prediction = torch.cat(list(arrays[(model_name, condition)]["prediction"]), dim=0)
            target = torch.cat(list(arrays[(model_name, condition)]["target"]), dim=0)
            metrics = laplacian_prediction_metrics(prediction, target)
            loss = weighted_robust_laplacian_loss(
                prediction,
                target,
                torch.ones(len(prediction)),
                **_loss_kwargs(evaluation_config),
            )
            condition_values[condition] = {
                "normalized_mse": metrics["mse"],
                "vector_l2": metrics["vector_endpoint_error"],
                "global_cosine": metrics["global_cosine"],
                "high_10_percent_cosine": metrics["top_10_percent_cosine"],
                "prediction_target_norm_ratio": metrics[
                    "prediction_to_target_norm_ratio"
                ],
                "loss": float(loss),
            }
        selected = [row for row in rows if row["experiment"] == model_name]
        correct = condition_values["correct_rgb"]
        zero = condition_values["zero_rgb"]
        result[model_name] = {
            **correct,
            "zero_rgb_loss": zero["loss"],
            "correct_zero_loss_gap": zero["loss"] - correct["loss"],
            "correct_zero_cosine_gap": correct["global_cosine"]
            - zero["global_cosine"],
            "initial_chamfer": _mean(selected, "initial_chamfer"),
            "reconstruction_chamfer": _mean(selected, "reconstruction_chamfer"),
            "reconstruction_point_to_surface": _mean(
                selected, "reconstruction_point_to_surface"
            ),
            "reconstruction_normal_consistency": _mean(
                selected, "reconstruction_normal_consistency"
            ),
            "introduced_flipped_faces": int(
                sum(int(row["introduced_flipped_faces"]) for row in selected)
            ),
            "new_degenerate_faces": int(
                sum(int(row["new_degenerate_faces"]) for row in selected)
            ),
            "improved_over_initial": int(
                sum(bool(row["improved_over_initial"]) for row in selected)
            ),
            "sample_count": len(selected),
        }
    return result


def _aggregate_per_object(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["experiment"]), str(row["object_id"]))].append(row)
    output = []
    for (experiment, object_id), selected in sorted(grouped.items()):
        output.append(
            {
                "experiment": experiment,
                "object_id": object_id,
                "variant_count": len(selected),
                "normalized_mse": _mean(selected, "normalized_mse"),
                "vector_l2": _mean(selected, "vector_l2"),
                "global_cosine": _mean(selected, "global_cosine"),
                "high_10_percent_cosine": _mean(selected, "high_10_percent_cosine"),
                "prediction_target_norm_ratio": _mean(
                    selected, "prediction_target_norm_ratio"
                ),
                "correct_rgb_loss": _mean(selected, "correct_rgb_loss"),
                "zero_rgb_loss": _mean(selected, "zero_rgb_loss"),
                "correct_zero_loss_gap": _mean(selected, "correct_zero_loss_gap"),
                "initial_chamfer": _mean(selected, "initial_chamfer"),
                "reconstruction_chamfer": _mean(selected, "reconstruction_chamfer"),
                "reconstruction_point_to_surface": _mean(
                    selected, "reconstruction_point_to_surface"
                ),
                "reconstruction_normal_consistency": _mean(
                    selected, "reconstruction_normal_consistency"
                ),
                "introduced_flipped_faces": int(
                    sum(int(row["introduced_flipped_faces"]) for row in selected)
                ),
                "improved_over_initial": int(
                    sum(bool(row["improved_over_initial"]) for row in selected)
                ),
            }
        )
    return output


def _markdown_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Existing GT-query baseline vs synthetic current-query formulation",
        "",
        "This is an existing canonical baseline versus a new formulation experiment. It is not a strict paired-training ablation.",
        "",
        "| Field | Experiment A | Experiment B |",
        "|---|---|---|",
        "| Training formulation | GT-query | current-query |",
        "| Training status | existing / frozen | newly trained |",
        "| Standard | existing canonical C2 / F2 / 14 VIEWS | C2 / F2 / 14 VIEWS |",
        "| Training rerun | NO | YES |",
        "| Test query | same synthetic C | same synthetic C |",
        "| Target | L_current @ P_proxy | L_current @ P_proxy |",
        "",
        "| Metric | Experiment A | Experiment B |",
        "|---|---:|---:|",
    ]
    labels = (
        ("normalized MSE", "normalized_mse"),
        ("vector L2", "vector_l2"),
        ("global cosine", "global_cosine"),
        ("High-10% cosine", "high_10_percent_cosine"),
        ("prediction / target norm", "prediction_target_norm_ratio"),
        ("correct RGB loss", "loss"),
        ("zero RGB loss", "zero_rgb_loss"),
        ("correct-zero loss gap", "correct_zero_loss_gap"),
        ("reconstruction Chamfer", "reconstruction_chamfer"),
        ("point-to-surface", "reconstruction_point_to_surface"),
        ("normal consistency", "reconstruction_normal_consistency"),
        ("introduced flipped faces", "introduced_flipped_faces"),
        ("improved over initial", "improved_over_initial"),
    )
    for label, key in labels:
        lines.append(
            f"| {label} | {_format(summary['aggregate']['A'][key])} | "
            f"{_format(summary['aggregate']['B'][key])} |"
        )
    lines.extend(
        [
            "",
            f"Shared test samples: `{summary['test_samples']}` variants from `{summary['test_objects']}` held-out objects.",
            "",
            "Per-object metrics are in `per_object.csv`; per-variant metrics are in `per_variant.csv`.",
            "",
        ]
    )
    return "\n".join(lines)


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else math.nan


def _format(value: Any) -> str:
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.8g}"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
