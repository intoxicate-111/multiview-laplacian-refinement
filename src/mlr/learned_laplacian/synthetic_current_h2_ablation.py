from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from mlr.data import Mesh
from mlr.io import load_mesh

from .canonical_experiment import _exact_query_sample, _load_device_item
from .canonical_pipeline import canonical_current_graph_recovery_inputs
from .diagnostics import _amp_settings
from .evaluation import reconstruct_and_evaluate
from .losses import robust_laplacian_error_per_vertex
from .multi_dataset import PreparedMeshDataset, validate_disjoint_splits
from .multi_trainer import _build_model
from .synthetic_current_comparison import _topology_change
from .synthetic_current_topk_recovery import (
    _descending_residual_ranking,
    _selection_audit,
    _topk_indices,
    _vertex_surface_distances,
)
from .target_scaling import (
    EDGE_SCALE_NORMALIZED_LAPLACIAN,
    RAW_LAPLACIAN,
    normalize_laplacian_by_edge_scale,
    prediction_to_raw_laplacian,
)
from .trainer import _seed_everything, load_checkpoint


ARMS = (
    "A_canonical_h2_normalized",
    "B_direct_raw_laplacian",
    "C_normalized_output_raw_loss",
)
PERCENTAGES = (0, 1, 10, 20, 50, 100)
SMALL_H_GROUPS = (
    ("smallest_1_percent", 0, 1),
    ("percentile_1_10", 1, 10),
    ("percentile_10_25", 10, 25),
    ("percentile_25_50", 25, 50),
    ("percentile_50_100", 50, 100),
)
RAW_METRIC_FIELDS = (
    "raw_epe",
    "raw_top_1_percent_epe",
    "raw_top_10_percent_epe",
    "raw_top_20_percent_epe",
    "raw_top_50_percent_epe",
    "raw_global_cosine",
    "prediction_to_target_raw_norm_ratio",
    "raw_residual_rms",
    "raw_residual_maximum",
    "recovery_weighted_raw_residual_rms",
)
GEOMETRY_FIELDS = (
    "initial_chamfer",
    "reconstruction_chamfer",
    "initial_point_to_surface",
    "reconstruction_point_to_surface",
    "initial_normal_consistency",
    "reconstruction_normal_consistency",
)


def run_h2_normalization_ablation(
    manifest_path: str | Path,
    arm_a_run: str | Path,
    arm_b_run: str | Path,
    arm_c_run: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    manifest = Path(manifest_path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        raise ValueError("The controlled ablation evaluator requires a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but torch.cuda.is_available() is false.")

    datasets = {
        split: PreparedMeshDataset.from_manifest(manifest, split)
        for split in ("train", "validation", "test")
    }
    validate_disjoint_splits(*datasets.values())
    expected_counts = {"train": 200, "validation": 25, "test": 25}
    actual_counts = {name: len(dataset) for name, dataset in datasets.items()}
    if actual_counts != expected_counts:
        raise ValueError(f"Unexpected split counts: {actual_counts}.")

    specs = _load_arm_specs(
        {
            ARMS[0]: Path(arm_a_run).resolve(),
            ARMS[1]: Path(arm_b_run).resolve(),
            ARMS[2]: Path(arm_c_run).resolve(),
        },
        resolved_device,
    )
    preflight = _preflight_audit(manifest, datasets, specs)
    if not preflight["passed"]:
        _write_json(output / "contract_audit.json", preflight)
        raise RuntimeError("H2 ablation preflight contract audit failed.")

    prediction_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    selection_checks: list[dict[str, Any]] = []
    target_formula_checks: list[dict[str, Any]] = []
    roundtrip_checks: list[dict[str, Any]] = []
    small_h_arrays: dict[str, dict[str, list[np.ndarray]]] = {
        split: defaultdict(list) for split in ("validation", "test")
    }
    small_h_path = output / "small_h_per_vertex.csv"
    small_h_fields = (
        "split",
        "sample_id",
        "object_id",
        "variant_index",
        "vertex_index",
        "valid_scale",
        "h_current",
        "h_current_squared",
        "raw_residual",
        "normalized_residual",
        "normalized_huber_loss",
        "target_weighted_normalized_huber_loss",
        "confidence",
        "visibility_count",
        "recovery_weight",
        "recovery_weighted_raw_residual",
        "recovered_vertex_to_gt_surface_distance",
    )
    with small_h_path.open("w", encoding="utf-8", newline="") as handle:
        small_h_writer = csv.DictWriter(handle, fieldnames=small_h_fields)
        small_h_writer.writeheader()
        for split in ("validation", "test"):
            dataset = datasets[split]
            for index in range(len(dataset)):
                static = dataset.load_static(index)
                sample_id = str(static["sample_id"])
                metadata = dict(static.get("metadata", {}))
                _validate_sample_contract(sample_id, metadata)
                formula = _target_formula_audit(static)
                formula["split"] = split
                target_formula_checks.append(formula)
                inferred = {
                    arm: _infer_one(dataset, index, spec, resolved_device)
                    for arm, spec in specs.items()
                }
                for arm, values in inferred.items():
                    metrics = _raw_metrics(
                        values["prediction_raw"],
                        values["target_raw"],
                        values["recovery_weight"],
                        values["valid"],
                    )
                    prediction_rows.append(
                        {
                            "split": split,
                            "arm": arm,
                            "sample_id": sample_id,
                            "object_id": metadata.get("object_id"),
                            "variant_index": metadata.get("variant_index"),
                            "vertex_count": int(values["prediction_raw"].shape[0]),
                            "valid_vertex_count": int(values["valid"].sum().item()),
                            "mean_confidence": float(values["confidence"].mean().item()),
                            "visible_vertex_fraction": float(
                                (values["visibility_count"] > 0).float().mean().item()
                            ),
                            **metrics,
                        }
                    )
                    roundtrip_checks.append(
                        {
                            "split": split,
                            "arm": arm,
                            "sample_id": sample_id,
                            "max_abs_output_to_raw_roundtrip_error": float(
                                values["roundtrip_error"]
                            ),
                        }
                    )

                recovered_a: np.ndarray
                if split == "validation":
                    values = inferred[ARMS[0]]
                    recovery_dir = output / "small_h_validation_recovery" / sample_id
                    _, recovered_a = _recover_raw_one(
                        static,
                        values["prediction_raw"],
                        values["prediction_normalized"],
                        values["confidence"],
                        recovery_dir,
                        specs[ARMS[0]]["config"],
                    )
                    print(
                        f"small_h validation {sample_id} recovered",
                        flush=True,
                    )
                else:
                    recovered_a = np.empty((0, 3), dtype=np.float64)
                    for arm, values in inferred.items():
                        raw_residual = torch.linalg.vector_norm(
                            values["prediction_raw"] - values["target_raw"], dim=-1
                        )
                        ranking = _descending_residual_ranking(raw_residual.numpy())
                        selections = {
                            percentage: _topk_indices(
                                ranking, len(ranking), percentage
                            )
                            for percentage in PERCENTAGES
                        }
                        selection_checks.append(
                            _selection_audit(
                                sample_id, arm, selections, raw_residual
                            )
                        )
                        for percentage in PERCENTAGES:
                            indices = selections[percentage]
                            hybrid_raw = values["prediction_raw"].clone()
                            hybrid_normalized = values["prediction_normalized"].clone()
                            if len(indices):
                                selected = torch.from_numpy(indices)
                                hybrid_raw[selected] = values["target_raw"][selected]
                                hybrid_normalized[selected] = values[
                                    "target_normalized"
                                ][selected]
                            recovery_dir = (
                                output
                                / "reconstruction"
                                / arm
                                / f"replace_{percentage:03d}pct"
                                / sample_id
                            )
                            recovery, recovered_vertices = _recover_raw_one(
                                static,
                                hybrid_raw,
                                hybrid_normalized,
                                values["confidence"],
                                recovery_dir,
                                specs[arm]["config"],
                            )
                            remaining = raw_residual.clone()
                            if len(indices):
                                remaining[torch.from_numpy(indices)] = 0.0
                            baseline_energy = float(
                                raw_residual.double().square().sum().item()
                            )
                            remaining_energy = float(
                                remaining.double().square().sum().item()
                            )
                            recovery_rows.append(
                                {
                                    "arm": arm,
                                    "replacement_percent": percentage,
                                    "sample_id": sample_id,
                                    "object_id": metadata.get("object_id"),
                                    "variant_index": metadata.get("variant_index"),
                                    "vertex_count": len(ranking),
                                    "replaced_vertex_count": len(indices),
                                    "actual_replacement_percent": (
                                        100.0 * len(indices) / len(ranking)
                                    ),
                                    "raw_residual_energy_replaced_fraction": (
                                        (baseline_energy - remaining_energy)
                                        / baseline_energy
                                        if baseline_energy
                                        else 1.0
                                    ),
                                    **recovery,
                                }
                            )
                            print(
                                f"{arm} {sample_id} replace={percentage}% "
                                f"chamfer={recovery['reconstruction_chamfer']:.9g} "
                                f"improved={recovery['improved_over_initial']}",
                                flush=True,
                            )
                            if arm == ARMS[0] and percentage == 0:
                                recovered_a = recovered_vertices
                    if recovered_a.size == 0:
                        raise RuntimeError(f"Missing Arm A baseline recovery for {sample_id}.")

                gt_mesh = Mesh(
                    torch.as_tensor(static["gt_vertices"]).cpu().numpy(),
                    torch.as_tensor(static["gt_faces"]).cpu().numpy(),
                ).ensure_normals()
                recovered_surface_distance = _vertex_surface_distances(
                    recovered_a, gt_mesh
                )
                _record_small_h_diagnostics(
                    split,
                    static,
                    metadata,
                    inferred[ARMS[0]],
                    recovered_surface_distance,
                    small_h_writer,
                    small_h_arrays[split],
                    specs[ARMS[0]]["config"],
                )
                del inferred
                torch.cuda.empty_cache()

    prediction_aggregate = _aggregate_prediction(prediction_rows)
    recovery_aggregate = _aggregate_recovery(recovery_rows)
    recovery_per_object = _aggregate_recovery_by_object(recovery_rows)
    small_h_summary = _aggregate_small_h(small_h_arrays)
    audit = _final_audit(
        preflight,
        target_formula_checks,
        roundtrip_checks,
        selection_checks,
        prediction_rows,
        recovery_rows,
    )
    _write_json(output / "contract_audit.json", audit)
    if not audit["passed"]:
        raise RuntimeError("H2 ablation final contract audit failed.")

    decision = _decision(recovery_aggregate, prediction_aggregate)
    summary = {
        "experiment": "C2F2 28-View Current-Graph Normalization Three-Arm Ablation",
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "device": str(resolved_device),
        "arms": {
            arm: {
                "run_dir": str(spec["run_dir"]),
                "checkpoint": str(spec["checkpoint"]),
                "checkpoint_sha256": spec["checkpoint_sha256"],
                "optimizer_steps": spec["optimizer_steps"],
                "target_mode": spec["target_mode"],
                "prediction_loss_space": spec["prediction_loss_space"],
                "native_best_validation_loss": spec["native_metrics"].get(
                    "best_selection_loss"
                ),
                "native_final_validation_loss": spec["native_metrics"].get(
                    "final_validation_loss"
                ),
                "runtime_seconds": spec["native_metrics"].get("runtime_seconds"),
            }
            for arm, spec in specs.items()
        },
        "contract": {
            "model": "C2F2",
            "views": 28,
            "optimizer_steps": 20_000,
            "local_query_jitter": False,
            "target_raw": "L_current@P_proxy",
            "topk_ranking": "descending raw solver-input residual magnitude",
            "topk_percentages": list(PERCENTAGES),
            "native_validation_losses_are_not_cross_arm_comparable": True,
        },
        "contract_audit": audit,
        "prediction_aggregate": prediction_aggregate,
        "recovery_aggregate": recovery_aggregate,
        "recovery_per_object": recovery_per_object,
        "small_h_group_summary": small_h_summary,
        "decision": decision,
        "outputs": {
            "prediction_per_sample": str(output / "prediction_per_sample.csv"),
            "prediction_aggregate": str(output / "prediction_aggregate.csv"),
            "topk_per_sample": str(output / "topk_oracle_replacement_per_sample.csv"),
            "topk_aggregate": str(output / "topk_oracle_replacement_aggregate.csv"),
            "recovery_per_object": str(output / "recovery_per_object.csv"),
            "small_h_per_vertex": str(small_h_path),
            "small_h_group_summary": str(output / "small_h_group_summary.csv"),
            "report": str(output / "REPORT.md"),
        },
    }
    _write_json(output / "h2_normalization_ablation_summary.json", summary)
    _write_csv(output / "prediction_per_sample.csv", prediction_rows)
    _write_csv(output / "prediction_aggregate.csv", prediction_aggregate)
    _write_csv(output / "topk_oracle_replacement_per_sample.csv", recovery_rows)
    _write_csv(output / "topk_oracle_replacement_aggregate.csv", recovery_aggregate)
    _write_csv(output / "recovery_per_object.csv", recovery_per_object)
    _write_csv(output / "small_h_group_summary.csv", small_h_summary)
    (output / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _load_arm_specs(
    run_dirs: Mapping[str, Path], device: torch.device
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        run_dir = run_dirs[arm]
        config = _run_config(run_dir)
        checkpoint = run_dir / "checkpoint_latest.pt"
        metrics_path = run_dir / "metrics.json"
        if not checkpoint.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(f"Incomplete run directory for {arm}: {run_dir}")
        model = _build_model(config, None, False).to(device)
        payload = load_checkpoint(checkpoint, model, map_location=device)
        model.eval()
        amp_enabled, amp_dtype = _amp_settings(config, device)
        native_metrics = _read_json(metrics_path)
        result[arm] = {
            "run_dir": run_dir,
            "checkpoint": checkpoint,
            "checkpoint_sha256": _sha256(checkpoint),
            "config": config,
            "model": model,
            "amp_enabled": amp_enabled,
            "amp_dtype": amp_dtype,
            "optimizer_steps": int(payload.get("optimizer_steps", -1)),
            "target_mode": str(config.get("target_mode")),
            "prediction_loss_space": str(
                config.get("training", {}).get(
                    "prediction_loss_space", "output_representation"
                )
            ),
            "native_metrics": native_metrics,
        }
    return result


def _run_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_config.json"
    value = _read_json(path)
    config = value.get("experiment_config", value)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid experiment config in {path}.")
    return config


def _infer_one(
    dataset: PreparedMeshDataset,
    index: int,
    spec: Mapping[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor | float]:
    config = spec["config"]
    prepared = _load_device_item(dataset, index, config, device)
    conditioned = _exact_query_sample(prepared.sample, device)
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=spec["amp_dtype"],
        enabled=bool(spec["amp_enabled"]),
    ):
        model_output = spec["model"](conditioned)
    if model_output.confidence_prediction is None:
        raise RuntimeError("All ablation arms require confidence prediction.")
    prediction_output = model_output.predicted_laplacian.float().detach().cpu()
    confidence = model_output.confidence_prediction.float().detach().cpu()
    h = prepared.sample["local_edge_length"].float().detach().cpu()
    valid = prepared.sample["valid_scale_mask"].bool().detach().cpu()
    epsilon = float(config.get("target_scaling", {}).get("epsilon", 1e-12))
    target_raw = prepared.raw_target.float().detach().cpu()
    target_normalized = normalize_laplacian_by_edge_scale(
        target_raw, h, eps=epsilon, valid_scale_mask=valid
    )
    target_mode = str(config.get("target_mode"))
    prediction_raw = prediction_to_raw_laplacian(
        prediction_output,
        h,
        input_representation=target_mode,
        eps=epsilon,
    )
    if target_mode == EDGE_SCALE_NORMALIZED_LAPLACIAN:
        prediction_normalized = prediction_output
    elif target_mode == RAW_LAPLACIAN:
        prediction_normalized = normalize_laplacian_by_edge_scale(
            prediction_raw, h, eps=epsilon, valid_scale_mask=valid
        )
    else:
        raise ValueError(f"Unsupported target_mode {target_mode!r}.")
    visibility = prepared.sample["visibility"].detach().cpu()
    canonical = canonical_current_graph_recovery_inputs(
        prepared.sample["vertices"].detach().cpu(),
        prepared.sample["faces"].detach().cpu(),
        prediction_normalized,
        visibility,
        confidence,
        epsilon=epsilon,
    )
    roundtrip_error = torch.max(
        torch.abs(canonical.delta_pred_raw.cpu() - prediction_raw)
    ).item()
    return {
        "prediction_output": prediction_output,
        "prediction_raw": prediction_raw,
        "prediction_normalized": prediction_normalized,
        "target_raw": target_raw,
        "target_normalized": target_normalized,
        "confidence": confidence,
        "h": h,
        "valid": valid,
        "visibility_count": visibility.to(torch.int64).sum(dim=0),
        "recovery_weight": canonical.weight.detach().cpu(),
        "target_confidence": prepared.sample["target_confidence"].float().detach().cpu(),
        "roundtrip_error": float(roundtrip_error),
    }


def _raw_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    recovery_weight: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, float]:
    prediction = prediction[valid].double()
    target = target[valid].double()
    weight = recovery_weight[valid].double().clamp_min(0.0)
    residual = torch.linalg.vector_norm(prediction - target, dim=-1)
    order = torch.argsort(residual, descending=True, stable=True)

    def top(fraction: float) -> float:
        count = max(1, int(math.ceil(fraction * len(residual))))
        return float(residual[order[:count]].mean().item())

    cosine = F.cosine_similarity(
        prediction.reshape(1, -1), target.reshape(1, -1), dim=-1, eps=1e-12
    )
    target_norm = torch.linalg.vector_norm(target)
    weighted_rms = torch.sqrt(
        (weight * residual.square()).sum() / weight.sum().clamp_min(1e-12)
    )
    return {
        "raw_epe": float(residual.mean().item()),
        "raw_top_1_percent_epe": top(0.01),
        "raw_top_10_percent_epe": top(0.10),
        "raw_top_20_percent_epe": top(0.20),
        "raw_top_50_percent_epe": top(0.50),
        "raw_global_cosine": float(cosine.item()),
        "prediction_to_target_raw_norm_ratio": float(
            (torch.linalg.vector_norm(prediction) / target_norm.clamp_min(1e-12)).item()
        ),
        "raw_residual_rms": float(torch.sqrt(residual.square().mean()).item()),
        "raw_residual_maximum": float(residual.max().item()),
        "recovery_weighted_raw_residual_rms": float(weighted_rms.item()),
    }


def _recover_raw_one(
    static: Mapping[str, Any],
    prediction_raw: torch.Tensor,
    prediction_normalized: torch.Tensor,
    confidence: torch.Tensor,
    output_dir: Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    epsilon = float(config.get("target_scaling", {}).get("epsilon", 1e-12))
    canonical = canonical_current_graph_recovery_inputs(
        static["vertices"],
        static["faces"],
        prediction_normalized,
        static["visibility_backface_and_occlusion"],
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
        prediction_raw,
        output_dir,
        recovery_config,
        normalized_prediction=prediction_normalized,
        edge_scale_epsilon=epsilon,
        laplacian_weight=canonical.weight,
        unseen_anchor_weight=float(recovery_config.get("unseen_anchor_weight", 0.0)),
        evaluate_laplacian_prediction=True,
        evaluate_initial_geometry=True,
        solver_confidence=np.ones(len(prediction_raw), dtype=np.float64),
    )
    recovered = load_mesh(output_dir / "predicted_refined.obj")
    initial = torch.as_tensor(static["vertices"]).cpu().numpy()
    faces = torch.as_tensor(static["faces"]).cpu().numpy()
    topology = _topology_change(initial, recovered.vertices, faces)
    coarse = metrics["geometry"]["coarse"]
    refined = metrics["geometry"]["predicted"]
    initial_chamfer = float(coarse["chamfer"])
    refined_chamfer = float(refined["chamfer"])
    return (
        {
            "initial_chamfer": initial_chamfer,
            "reconstruction_chamfer": refined_chamfer,
            "initial_point_to_surface": float(
                coarse["point_to_surface_bidirectional_mean"]
            ),
            "reconstruction_point_to_surface": float(
                refined["point_to_surface_bidirectional_mean"]
            ),
            "initial_normal_consistency": float(coarse["normal_consistency"]),
            "reconstruction_normal_consistency": float(
                refined["normal_consistency"]
            ),
            "introduced_flipped_faces": int(topology["introduced_flips"]),
            "new_degenerate_faces": int(topology["new_degeneracies"]),
            "improved_over_initial": bool(refined_chamfer < initial_chamfer),
            "mean_confidence": float(confidence.mean().item()),
            "visible_vertex_fraction": float(canonical.visible.float().mean().item()),
        },
        recovered.vertices,
    )


def _record_small_h_diagnostics(
    split: str,
    static: Mapping[str, Any],
    metadata: Mapping[str, Any],
    values: Mapping[str, torch.Tensor | float],
    recovered_distance: np.ndarray,
    writer: csv.DictWriter,
    aggregate: dict[str, list[np.ndarray]],
    config: Mapping[str, Any],
) -> None:
    prediction_normalized = torch.as_tensor(values["prediction_normalized"])
    target_normalized = torch.as_tensor(values["target_normalized"])
    prediction_raw = torch.as_tensor(values["prediction_raw"])
    target_raw = torch.as_tensor(values["target_raw"])
    h = torch.as_tensor(values["h"])
    valid = torch.as_tensor(values["valid"]).bool()
    confidence = torch.as_tensor(values["confidence"])
    visibility_count = torch.as_tensor(values["visibility_count"])
    recovery_weight = torch.as_tensor(values["recovery_weight"])
    target_weight = torch.as_tensor(values["target_confidence"])
    huber_delta = float(config.get("training", {}).get("huber_delta", 0.01))
    normalized_loss = robust_laplacian_error_per_vertex(
        prediction_normalized,
        target_normalized,
        loss_type="huber",
        huber_delta=huber_delta,
    )
    normalized_residual = torch.linalg.vector_norm(
        prediction_normalized - target_normalized, dim=-1
    )
    raw_residual = torch.linalg.vector_norm(prediction_raw - target_raw, dim=-1)
    weighted_loss = target_weight * normalized_loss
    weighted_raw = recovery_weight.clamp_min(0.0).sqrt() * raw_residual
    arrays = {
        "h_current": h.numpy(),
        "normalized_residual": normalized_residual.numpy(),
        "raw_residual": raw_residual.numpy(),
        "weighted_normalized_loss": weighted_loss.numpy(),
        "weighted_raw_residual": weighted_raw.numpy(),
        "recovered_distance": np.asarray(recovered_distance),
        "valid": valid.numpy(),
    }
    for name, array in arrays.items():
        aggregate[name].append(array)
    sample_id = str(static["sample_id"])
    for vertex in range(len(h)):
        writer.writerow(
            {
                "split": split,
                "sample_id": sample_id,
                "object_id": metadata.get("object_id"),
                "variant_index": metadata.get("variant_index"),
                "vertex_index": vertex,
                "valid_scale": bool(valid[vertex]),
                "h_current": float(h[vertex]),
                "h_current_squared": float(h[vertex].square()),
                "raw_residual": float(raw_residual[vertex]),
                "normalized_residual": float(normalized_residual[vertex]),
                "normalized_huber_loss": float(normalized_loss[vertex]),
                "target_weighted_normalized_huber_loss": float(weighted_loss[vertex]),
                "confidence": float(confidence[vertex]),
                "visibility_count": int(visibility_count[vertex]),
                "recovery_weight": float(recovery_weight[vertex]),
                "recovery_weighted_raw_residual": float(weighted_raw[vertex]),
                "recovered_vertex_to_gt_surface_distance": float(
                    recovered_distance[vertex]
                ),
            }
        )


def _aggregate_small_h(
    values: Mapping[str, Mapping[str, Sequence[np.ndarray]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        arrays = {
            name: np.concatenate(list(chunks))
            for name, chunks in values[split].items()
        }
        valid_indices = np.flatnonzero(arrays["valid"].astype(bool))
        order = valid_indices[
            np.argsort(arrays["h_current"][valid_indices], kind="stable")
        ]
        count = len(order)
        total_loss = float(arrays["weighted_normalized_loss"][order].sum())
        for group, lower, upper in SMALL_H_GROUPS:
            start = 0 if lower == 0 else int(math.ceil(lower * count / 100.0))
            stop = count if upper == 100 else int(math.ceil(upper * count / 100.0))
            selected = order[start:stop]
            rows.append(
                {
                    "split": split,
                    "h_percentile_group": group,
                    "vertex_count": len(selected),
                    "vertex_fraction": len(selected) / count,
                    "h_minimum": float(arrays["h_current"][selected].min()),
                    "h_maximum": float(arrays["h_current"][selected].max()),
                    "normalized_loss_contribution_fraction": float(
                        arrays["weighted_normalized_loss"][selected].sum()
                        / max(total_loss, 1e-30)
                    ),
                    **_mean_median(
                        arrays["normalized_residual"][selected],
                        "normalized_residual",
                    ),
                    **_mean_median(
                        arrays["raw_residual"][selected], "raw_residual"
                    ),
                    **_mean_median(
                        arrays["weighted_raw_residual"][selected],
                        "recovery_weighted_raw_residual",
                    ),
                    **_mean_median(
                        arrays["recovered_distance"][selected],
                        "recovered_vertex_to_gt_surface_distance",
                    ),
                }
            )
    return rows


def _mean_median(values: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
    }


def _aggregate_prediction(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for split in ("validation", "test"):
        for arm in ARMS:
            selected = [
                row for row in rows if row["split"] == split and row["arm"] == arm
            ]
            if len(selected) != 25:
                raise RuntimeError(f"Expected 25 prediction rows for {split}/{arm}.")
            output.append(
                {
                    "split": split,
                    "arm": arm,
                    "sample_count": len(selected),
                    **{field: _mean(selected, field) for field in RAW_METRIC_FIELDS},
                    "mean_confidence": _mean(selected, "mean_confidence"),
                    "visible_vertex_fraction": _mean(
                        selected, "visible_vertex_fraction"
                    ),
                }
            )
    return output


def _aggregate_recovery(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for arm in ARMS:
        arm_rows: dict[int, dict[str, Any]] = {}
        for percentage in PERCENTAGES:
            selected = [
                row
                for row in rows
                if row["arm"] == arm and row["replacement_percent"] == percentage
            ]
            if len(selected) != 25:
                raise RuntimeError(f"Expected 25 recovery rows for {arm}/{percentage}%.")
            current = {
                "arm": arm,
                "replacement_percent": percentage,
                "sample_count": len(selected),
                "mean_actual_replacement_percent": _mean(
                    selected, "actual_replacement_percent"
                ),
                "mean_raw_residual_energy_replaced_fraction": _mean(
                    selected, "raw_residual_energy_replaced_fraction"
                ),
                **{field: _mean(selected, field) for field in GEOMETRY_FIELDS},
                "introduced_flipped_faces": int(
                    sum(int(row["introduced_flipped_faces"]) for row in selected)
                ),
                "new_degenerate_faces": int(
                    sum(int(row["new_degenerate_faces"]) for row in selected)
                ),
                "improved_over_initial": int(
                    sum(bool(row["improved_over_initial"]) for row in selected)
                ),
            }
            arm_rows[percentage] = current
        baseline = arm_rows[0]
        oracle = arm_rows[100]
        for percentage in PERCENTAGES:
            current = arm_rows[percentage]
            current["chamfer_oracle_gap_closed_fraction"] = _gap_closed(
                baseline["reconstruction_chamfer"],
                current["reconstruction_chamfer"],
                oracle["reconstruction_chamfer"],
            )
            current["p2s_oracle_gap_closed_fraction"] = _gap_closed(
                baseline["reconstruction_point_to_surface"],
                current["reconstruction_point_to_surface"],
                oracle["reconstruction_point_to_surface"],
            )
            output.append(current)
    return output


def _aggregate_recovery_by_object(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["arm"]), int(row["replacement_percent"]), str(row["object_id"]))].append(row)
    output = []
    for (arm, percentage, object_id), selected in sorted(grouped.items()):
        output.append(
            {
                "arm": arm,
                "replacement_percent": percentage,
                "object_id": object_id,
                "sample_count": len(selected),
                **{field: _mean(selected, field) for field in GEOMETRY_FIELDS},
                "introduced_flipped_faces": int(
                    sum(int(row["introduced_flipped_faces"]) for row in selected)
                ),
                "improved_over_initial": int(
                    sum(bool(row["improved_over_initial"]) for row in selected)
                ),
            }
        )
    return output


def _preflight_audit(
    manifest: Path,
    datasets: Mapping[str, PreparedMeshDataset],
    specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    controlled = {
        arm: _controlled_config(spec["config"]) for arm, spec in specs.items()
    }
    initial_hashes = {
        arm: _initial_state_hash(spec["config"]) for arm, spec in specs.items()
    }
    split_ids = {
        split: list(dataset.sample_ids) for split, dataset in datasets.items()
    }
    run_split_ids = {
        arm: _manifest_split_ids(spec["run_dir"] / "dataset_manifest.json")
        for arm, spec in specs.items()
    }
    run_manifests_match = all(
        ids == split_ids for ids in run_split_ids.values()
    )
    arm_semantics = {
        ARMS[0]: (
            specs[ARMS[0]]["target_mode"] == EDGE_SCALE_NORMALIZED_LAPLACIAN
            and specs[ARMS[0]]["prediction_loss_space"] == "output_representation"
        ),
        ARMS[1]: (
            specs[ARMS[1]]["target_mode"] == RAW_LAPLACIAN
            and specs[ARMS[1]]["prediction_loss_space"] == "output_representation"
        ),
        ARMS[2]: (
            specs[ARMS[2]]["target_mode"] == EDGE_SCALE_NORMALIZED_LAPLACIAN
            and specs[ARMS[2]]["prediction_loss_space"] == RAW_LAPLACIAN
        ),
    }
    all_configs_equal = len(
        {json.dumps(value, sort_keys=True) for value in controlled.values()}
    ) == 1
    optimizer_steps = {
        arm: int(spec["optimizer_steps"]) for arm, spec in specs.items()
    }
    no_jitter = {
        arm: not bool(spec["config"].get("local_query_jitter", {}).get("enabled", False))
        for arm, spec in specs.items()
    }
    passed = bool(
        manifest.is_file()
        and all_configs_equal
        and len(set(initial_hashes.values())) == 1
        and all(value == 20_000 for value in optimizer_steps.values())
        and all(no_jitter.values())
        and all(arm_semantics.values())
        and run_manifests_match
        and {name: len(ids) for name, ids in split_ids.items()}
        == {"train": 200, "validation": 25, "test": 25}
    )
    return {
        "passed": passed,
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "split_counts": {name: len(ids) for name, ids in split_ids.items()},
        "split_sample_ids": split_ids,
        "run_manifest_sample_ids_match": run_manifests_match,
        "controlled_configs_equal": all_configs_equal,
        "initial_state_hashes": initial_hashes,
        "initial_states_equal": len(set(initial_hashes.values())) == 1,
        "optimizer_steps": optimizer_steps,
        "no_local_query_jitter": no_jitter,
        "arm_semantics": arm_semantics,
    }


def _final_audit(
    preflight: Mapping[str, Any],
    formula: Sequence[Mapping[str, Any]],
    roundtrips: Sequence[Mapping[str, Any]],
    selections: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    recovery_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    max_formula = max(
        float(row["current_graph_proxy_raw_target_max_abs_error"]) for row in formula
    )
    max_roundtrip = max(
        float(row["max_abs_output_to_raw_roundtrip_error"]) for row in roundtrips
    )
    selection_passed = all(bool(row["passed"]) for row in selections)
    counts = {
        "prediction_rows": len(prediction_rows),
        "recovery_rows": len(recovery_rows),
        "formula_checks": len(formula),
        "roundtrip_checks": len(roundtrips),
        "selection_checks": len(selections),
    }
    counts_match = counts == {
        "prediction_rows": 150,
        "recovery_rows": 450,
        "formula_checks": 50,
        "roundtrip_checks": 150,
        "selection_checks": 75,
    }
    passed = bool(
        preflight["passed"]
        and max_formula <= 1e-7
        and max_roundtrip <= 1e-6
        and selection_passed
        and counts_match
    )
    return {
        **dict(preflight),
        "passed": passed,
        "counts": counts,
        "counts_match": counts_match,
        "maximum_current_graph_proxy_raw_target_error": max_formula,
        "maximum_output_to_raw_roundtrip_error": max_roundtrip,
        "selection_checks_passed": selection_passed,
        "selection_checks": list(selections),
    }


def _controlled_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    result.pop("method", None)
    result.pop("target_mode", None)
    result.pop("target_definition", None)
    result.get("training", {}).pop("prediction_loss_space", None)
    result.get("recovery", {}).pop("denormalization", None)
    result.pop("experiment_metadata", None)
    return result


def _initial_state_hash(config: Mapping[str, Any]) -> str:
    _seed_everything(int(config.get("seed", 7)))
    model = _build_model(config, None, False)
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _manifest_split_ids(path: Path) -> dict[str, list[str]]:
    value = _read_json(path)
    samples = value.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"Manifest has no samples list: {path}")
    result = {split: [] for split in ("train", "validation", "test")}
    for item in samples:
        if not isinstance(item, Mapping):
            raise ValueError(f"Manifest contains a non-object sample: {path}")
        split = str(item.get("split"))
        if split in result:
            result[split].append(str(item.get("sample_id")))
    return result


def _validate_sample_contract(sample_id: str, metadata: Mapping[str, Any]) -> None:
    if metadata.get("proxy_definition") != "P_proxy=source_gt_vertices_with_exact_same_topology":
        raise RuntimeError(f"Unexpected proxy contract for {sample_id}.")
    if metadata.get("target_constructor") != "delta_target=L_current@P_proxy":
        raise RuntimeError(f"Unexpected target contract for {sample_id}.")


def _target_formula_audit(static: Mapping[str, Any]) -> dict[str, Any]:
    from .canonical_pipeline import current_uniform_laplacian_raw

    saved = torch.as_tensor(static["raw_laplacian_target"]).float().cpu()
    recomputed = torch.as_tensor(
        current_uniform_laplacian_raw(static["gt_vertices"], static["faces"]),
        dtype=saved.dtype,
    )
    return {
        "sample_id": str(static["sample_id"]),
        "current_graph_proxy_raw_target_max_abs_error": float(
            torch.max(torch.abs(saved - recomputed)).item()
        ),
    }


def _decision(
    recovery: Sequence[Mapping[str, Any]],
    prediction: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    baseline = {
        str(row["arm"]): row
        for row in recovery
        if int(row["replacement_percent"]) == 0
    }
    test_prediction = {
        str(row["arm"]): row
        for row in prediction
        if row["split"] == "test"
    }
    pairwise = {}
    for left, right in ((ARMS[1], ARMS[0]), (ARMS[2], ARMS[0]), (ARMS[1], ARMS[2])):
        pairwise[f"{left}_minus_{right}"] = {
            "refined_chamfer": float(
                baseline[left]["reconstruction_chamfer"]
                - baseline[right]["reconstruction_chamfer"]
            ),
            "improved_over_initial_count": int(
                baseline[left]["improved_over_initial"]
                - baseline[right]["improved_over_initial"]
            ),
            "recovery_weighted_raw_residual_rms": float(
                test_prediction[left]["recovery_weighted_raw_residual_rms"]
                - test_prediction[right]["recovery_weighted_raw_residual_rms"]
            ),
            "raw_top_1_percent_epe": float(
                test_prediction[left]["raw_top_1_percent_epe"]
                - test_prediction[right]["raw_top_1_percent_epe"]
            ),
            "raw_top_10_percent_epe": float(
                test_prediction[left]["raw_top_10_percent_epe"]
                - test_prediction[right]["raw_top_10_percent_epe"]
            ),
        }
    best_chamfer = min(
        ARMS, key=lambda arm: float(baseline[arm]["reconstruction_chamfer"])
    )
    best_improved = max(
        ARMS, key=lambda arm: int(baseline[arm]["improved_over_initial"])
    )
    equivalence_relative_chamfer = 0.01
    equivalence_improved_count = 1

    def approximately_equal(left: str, right: str) -> bool:
        left_chamfer = float(baseline[left]["reconstruction_chamfer"])
        right_chamfer = float(baseline[right]["reconstruction_chamfer"])
        relative = abs(left_chamfer - right_chamfer) / min(
            left_chamfer, right_chamfer
        )
        count_difference = abs(
            int(baseline[left]["improved_over_initial"])
            - int(baseline[right]["improved_over_initial"])
        )
        return bool(
            relative <= equivalence_relative_chamfer
            and count_difference <= equivalence_improved_count
        )

    def dominates(left: str, right: str) -> bool:
        return bool(
            float(baseline[left]["reconstruction_chamfer"])
            < float(baseline[right]["reconstruction_chamfer"])
            and int(baseline[left]["improved_over_initial"])
            >= int(baseline[right]["improved_over_initial"])
        )

    patterns = {
        "B_approximately_C_and_both_dominate_A": bool(
            approximately_equal(ARMS[1], ARMS[2])
            and dominates(ARMS[1], ARMS[0])
            and dominates(ARMS[2], ARMS[0])
        ),
        "B_dominates_C_approximately_A": bool(
            dominates(ARMS[1], ARMS[2])
            and approximately_equal(ARMS[2], ARMS[0])
        ),
        "A_dominates_B_and_C": bool(
            dominates(ARMS[0], ARMS[1]) and dominates(ARMS[0], ARMS[2])
        ),
        "B_dominates_A_and_C_is_between": bool(
            dominates(ARMS[1], ARMS[2]) and dominates(ARMS[2], ARMS[0])
        ),
    }
    matched = [name for name, value in patterns.items() if value]
    first_90 = {}
    for arm in ARMS:
        first_90[arm] = next(
            (
                int(row["replacement_percent"])
                for row in recovery
                if row["arm"] == arm
                and float(row["chamfer_oracle_gap_closed_fraction"]) >= 0.90
            ),
            None,
        )
    return {
        "primary_best_refined_chamfer_arm": best_chamfer,
        "primary_best_improved_count_arm": best_improved,
        "pairwise_differences": pairwise,
        "primary_pattern_equivalence_tolerance": {
            "relative_refined_chamfer": equivalence_relative_chamfer,
            "improved_over_initial_count": equivalence_improved_count,
        },
        "primary_pattern_checks": patterns,
        "matched_primary_pattern": matched[0] if len(matched) == 1 else None,
        "first_replacement_percent_closing_90pct_chamfer_oracle_gap": first_90,
        "interpretation_rule_requires_joint_primary_and_secondary_review": True,
    }


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# C2F2 28-View Current-Graph Normalization Three-Arm Ablation",
        "",
        "## Contract",
        "",
        f"- Manifest SHA-256: `{summary['manifest_sha256']}`.",
        "- All arms use the same C2F2 architecture, 28 views, split IDs, seed, initialization, optimizer, scheduler, batching and 20,000-step budget.",
        "- Local query jitter is disabled.",
        "- Raw target: `L_current@P_proxy`.",
        "- Top-k ranking: raw solver-input residual magnitude.",
        f"- Contract audit: `{summary['contract_audit']['passed']}`.",
        "- Native validation losses are retained with their loss-space labels and are not compared across arms.",
        "",
        "## Native training records",
        "",
        "| Arm | Output target | Loss space | Best native validation | Final native validation | Runtime seconds |",
        "|---|---|---|---:|---:|---:|",
    ]
    for arm in ARMS:
        values = summary["arms"][arm]
        lines.append(
            f"| {arm} | {values['target_mode']} | {values['prediction_loss_space']} | "
            f"{_f(values['native_best_validation_loss'])} | "
            f"{_f(values['native_final_validation_loss'])} | {_f(values['runtime_seconds'])} |"
        )
    lines.extend(
        [
            "",
            "## Unified test raw-space prediction",
            "",
            "| Arm | Raw EPE | Top 1% | Top 10% | Top 20% | Top 50% | Raw cosine | Raw norm ratio | Raw RMS | Raw max | Recovery-weighted RMS |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["prediction_aggregate"]:
        if row["split"] != "test":
            continue
        lines.append(
            f"| {row['arm']} | {_f(row['raw_epe'])} | {_f(row['raw_top_1_percent_epe'])} | "
            f"{_f(row['raw_top_10_percent_epe'])} | {_f(row['raw_top_20_percent_epe'])} | "
            f"{_f(row['raw_top_50_percent_epe'])} | {_f(row['raw_global_cosine'])} | "
            f"{_f(row['prediction_to_target_raw_norm_ratio'])} | {_f(row['raw_residual_rms'])} | "
            f"{_f(row['raw_residual_maximum'])} | {_f(row['recovery_weighted_raw_residual_rms'])} |"
        )
    lines.extend(
        [
            "",
            "## Test recovery and Top-k exact-target replacement",
            "",
            "| Arm | Replacement | Refined Chamfer | Refined P2S | Normal | Flips | Improved/25 | Chamfer gap closed |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["recovery_aggregate"]:
        lines.append(
            f"| {row['arm']} | {row['replacement_percent']}% | "
            f"{_f(row['reconstruction_chamfer'])} | {_f(row['reconstruction_point_to_surface'])} | "
            f"{_f(row['reconstruction_normal_consistency'])} | {row['introduced_flipped_faces']} | "
            f"{row['improved_over_initial']}/25 | {_f(row['chamfer_oracle_gap_closed_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "## Arm A small-h diagnostic",
            "",
            "| Split | h group | Vertices | Fraction | Normalized-loss contribution | Normalized residual mean | Raw residual mean | Weighted raw residual mean | Recovered surface distance mean |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["small_h_group_summary"]:
        lines.append(
            f"| {row['split']} | {row['h_percentile_group']} | {row['vertex_count']} | "
            f"{_f(row['vertex_fraction'])} | {_f(row['normalized_loss_contribution_fraction'])} | "
            f"{_f(row['normalized_residual_mean'])} | {_f(row['raw_residual_mean'])} | "
            f"{_f(row['recovery_weighted_raw_residual_mean'])} | "
            f"{_f(row['recovered_vertex_to_gt_surface_distance_mean'])} |"
        )
    lines.extend(["", "## Decision fields", "", "```json", json.dumps(summary["decision"], indent=2), "```", ""])
    return "\n".join(lines)


def _gap_closed(baseline: float, current: float, oracle: float) -> float:
    denominator = float(baseline) - float(oracle)
    return (float(baseline) - float(current)) / denominator if denominator else 1.0


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def _f(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.9g}"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
