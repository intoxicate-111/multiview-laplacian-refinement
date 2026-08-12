from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mlr.data import Mesh
from mlr.gt_laplacian import closest_points_on_mesh
from mlr.io import load_mesh

from .canonical_experiment import _exact_query_sample, _load_device_item, _write_heatmap_ply
from .canonical_pipeline import canonical_current_graph_recovery_inputs
from .diagnostics import _amp_settings
from .multi_dataset import PreparedMeshDataset, validate_disjoint_splits
from .multi_trainer import _build_model
from .synthetic_current_oracle_recovery import (
    EXPECTED_MANIFEST_SHA256,
    _read_json,
    _write_csv,
    _write_json,
)
from .synthetic_current_comparison import _reconstruct_one
from .trainer import load_checkpoint


CHECKPOINTS = ("current_query_20k", "current_query_50k")
PERCENTAGES = (0, 1, 10, 20, 50, 100)
GROUPS = (
    ("top_0_1_percent", 0, 1),
    ("percentile_1_10", 1, 10),
    ("percentile_10_20", 10, 20),
    ("percentile_20_50", 20, 50),
    ("bottom_50_percent", 50, 100),
)
REPRESENTATIVE_IDS = (
    "43bd0910-1dd1-4b1e-9ba2-e9801e6b5761__v00",
    "43bd0910-1dd1-4b1e-9ba2-e9801e6b5761__v04",
)


def run_topk_recovery_comparison(
    oracle_summary_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    oracle_summary_path = Path(oracle_summary_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = _read_json(oracle_summary_path)
    manifest = Path(str(reference["manifest"])).resolve()
    if _sha256(manifest) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("Manifest SHA-256 does not match the fixed synthetic-current contract.")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but torch.cuda.is_available() is false.")

    datasets = {
        split: PreparedMeshDataset.from_manifest(manifest, split)
        for split in ("train", "validation", "test")
    }
    validate_disjoint_splits(*datasets.values())
    dataset = datasets["test"]
    if len(dataset) != 25:
        raise ValueError(f"Expected 25 test variants, found {len(dataset)}.")
    object_ids = [
        str(dataset.load_static(index)["metadata"]["object_id"])
        for index in range(len(dataset))
    ]
    if len(set(object_ids)) != 5:
        raise ValueError("Expected five held-out objects.")

    model_specs = _load_models(reference, resolved_device)
    rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    selection_checks: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        metadata = dict(static["metadata"])
        if metadata.get("target_constructor") != "delta_target=L_current@P_proxy":
            raise RuntimeError(f"Unexpected target contract for {sample_id}.")
        gt_mesh = Mesh(
            static["gt_vertices"].detach().cpu().numpy(),
            static["gt_faces"].detach().cpu().numpy(),
        ).ensure_normals()
        initial_vertices = static["vertices"].detach().cpu().numpy()
        initial_surface_distance = _vertex_surface_distances(initial_vertices, gt_mesh)
        for checkpoint_name in CHECKPOINTS:
            spec = model_specs[checkpoint_name]
            prepared = _load_device_item(
                dataset, index, spec["config"], resolved_device
            )
            conditioned = _exact_query_sample(prepared.sample, resolved_device)
            with torch.no_grad(), torch.autocast(
                device_type=resolved_device.type,
                dtype=spec["amp_dtype"],
                enabled=spec["amp_enabled"],
            ):
                output = spec["model"](conditioned)
            if output.confidence_prediction is None:
                raise RuntimeError(f"{checkpoint_name} has no confidence prediction.")
            prediction_hat = output.delta_hat_prediction.float().detach().cpu()
            confidence = output.confidence_prediction.float().detach().cpu()
            target_hat = prepared.training_target.float().detach().cpu()
            valid = prepared.sample["valid_scale_mask"].bool().detach().cpu()
            baseline_inputs = canonical_current_graph_recovery_inputs(
                static["vertices"],
                static["faces"],
                prediction_hat,
                static["visibility_backface_and_occlusion"],
                confidence,
                epsilon=spec["epsilon"],
            )
            target_inputs = canonical_current_graph_recovery_inputs(
                static["vertices"],
                static["faces"],
                target_hat,
                static["visibility_backface_and_occlusion"],
                confidence,
                epsilon=spec["epsilon"],
            )
            raw_residual = torch.linalg.vector_norm(
                baseline_inputs.delta_pred_raw.cpu()
                - target_inputs.delta_pred_raw.cpu(),
                dim=-1,
            )
            normalized_residual = torch.linalg.vector_norm(
                prediction_hat - target_hat, dim=-1
            )
            ranking = _descending_residual_ranking(raw_residual.numpy())
            selections = {
                percentage: _topk_indices(ranking, len(ranking), percentage)
                for percentage in PERCENTAGES
            }
            selection_checks.append(
                _selection_audit(sample_id, checkpoint_name, selections, raw_residual)
            )
            group_masks = _residual_group_masks(ranking, len(ranking))
            if sample_id in REPRESENTATIVE_IDS:
                _write_representative_inputs(
                    output_dir,
                    sample_id,
                    checkpoint_name,
                    static,
                    raw_residual,
                    normalized_residual,
                    ranking,
                    selections,
                )

            baseline_energy = float(raw_residual.double().square().sum().item())
            representative_arrays: dict[str, np.ndarray] = {}
            for percentage in PERCENTAGES:
                indices = selections[percentage]
                hybrid_hat = prediction_hat.clone()
                if len(indices):
                    hybrid_hat[torch.from_numpy(indices)] = target_hat[
                        torch.from_numpy(indices)
                    ]
                recovery_dir = (
                    output_dir
                    / "reconstruction"
                    / checkpoint_name
                    / f"replace_{percentage:03d}pct"
                    / sample_id
                )
                recovery = _reconstruct_one(
                    static, hybrid_hat, confidence, recovery_dir, spec["config"]
                )
                recovered = load_mesh(recovery_dir / "predicted_refined.obj")
                surface_distance = _vertex_surface_distances(
                    recovered.vertices, gt_mesh
                )
                remaining = raw_residual.clone()
                if len(indices):
                    remaining[torch.from_numpy(indices)] = 0.0
                normalized_remaining = normalized_residual.clone()
                if len(indices):
                    normalized_remaining[torch.from_numpy(indices)] = 0.0
                replaced_energy = baseline_energy - float(
                    remaining.double().square().sum().item()
                )
                row = {
                    "checkpoint": checkpoint_name,
                    "replacement_percent": percentage,
                    "sample_id": sample_id,
                    "object_id": str(metadata["object_id"]),
                    "variant_index": int(metadata["variant_index"]),
                    "vertex_count": len(ranking),
                    "valid_vertex_count": int(valid.sum().item()),
                    "replaced_vertex_count": int(len(indices)),
                    "actual_replacement_percent": 100.0 * len(indices) / len(ranking),
                    "raw_residual_energy_replaced_fraction": (
                        replaced_energy / baseline_energy if baseline_energy else 1.0
                    ),
                    "remaining_raw_residual_mean": float(remaining.mean().item()),
                    "remaining_raw_residual_rms": float(
                        torch.sqrt(remaining.double().square().mean()).item()
                    ),
                    "remaining_normalized_residual_mean": float(
                        normalized_remaining.mean().item()
                    ),
                    "raw_residual_vs_recovered_surface_distance_spearman": (
                        _spearman(raw_residual.numpy(), surface_distance)
                    ),
                    **recovery,
                }
                row.update(_geometry_change(row))
                rows.append(row)
                group_rows.extend(
                    _group_geometry_rows(
                        row,
                        group_masks,
                        raw_residual.numpy(),
                        normalized_residual.numpy(),
                        initial_surface_distance,
                        surface_distance,
                    )
                )
                if sample_id in REPRESENTATIVE_IDS:
                    representative_arrays[
                        f"replace_{percentage:03d}pct_surface_distance"
                    ] = surface_distance
                print(
                    f"{checkpoint_name} {sample_id} replace={percentage}% "
                    f"chamfer={row['reconstruction_chamfer']:.8g} "
                    f"improved={row['improved_over_initial']}",
                    flush=True,
                )
            if sample_id in REPRESENTATIVE_IDS:
                npz_path = (
                    output_dir
                    / "visualizations"
                    / checkpoint_name
                    / f"{sample_id}_topk_diagnostics.npz"
                )
                with np.load(npz_path) as existing:
                    base_arrays = {key: existing[key] for key in existing.files}
                np.savez_compressed(npz_path, **base_arrays, **representative_arrays)
            del prepared, conditioned, output
            if resolved_device.type == "cuda":
                torch.cuda.empty_cache()

    aggregate = _aggregate(rows)
    group_aggregate = _aggregate_groups(group_rows)
    audit = _contract_audit(reference, aggregate, selection_checks)
    if not audit["passed"]:
        _write_json(output_dir / "topk_contract_audit.json", audit)
        raise RuntimeError("Top-k baseline/exact-target contract audit failed.")
    decision = _decision(aggregate, group_aggregate)
    summary = {
        "experiment": "Sofa50 Synthetic Current-query Top-k Raw-residual Oracle Replacement",
        "device": device,
        "manifest": str(manifest),
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "test_samples": 25,
        "test_objects": 5,
        "checkpoints": {
            name: {
                "checkpoint": str(model_specs[name]["checkpoint"]),
                "checkpoint_sha256": _sha256(model_specs[name]["checkpoint"]),
                "config": str(model_specs[name]["config_path"]),
                "checkpoint_epoch": int(model_specs[name]["checkpoint_epoch"]),
            }
            for name in CHECKPOINTS
        },
        "selection_contract": {
            "ranking_variable": "vertex-wise solver-input raw residual L2",
            "formula": "||delta_pred_raw[i]-delta_target_raw[i]||_2",
            "tie_break": "ascending vertex index",
            "positive_topk_count": "ceil(percentage/100 * vertex_count)",
            "percentages": list(PERCENTAGES),
            "nested_selection": True,
            "normalized_residual_used_for_selection": False,
        },
        "recovery_contract": (
            "checkpoint-native confidence times fixed visibility; only selected "
            "delta_pred_hat vertices are replaced by delta_target_hat"
        ),
        "contract_audit": audit,
        "aggregate": aggregate,
        "residual_group_aggregate": group_aggregate,
        "decision": decision,
        "outputs": {
            "per_sample_csv": str(
                output_dir / "topk_oracle_replacement_per_sample.csv"
            ),
            "residual_group_csv": str(
                output_dir / "topk_residual_group_geometry.csv"
            ),
            "aggregate_csv": str(
                output_dir / "topk_oracle_replacement_aggregate.csv"
            ),
            "report": str(output_dir / "topk_oracle_replacement_report.md"),
        },
    }
    _write_json(output_dir / "topk_oracle_replacement_summary.json", summary)
    _write_json(output_dir / "topk_contract_audit.json", audit)
    _write_csv(output_dir / "topk_oracle_replacement_per_sample.csv", rows)
    _write_csv(output_dir / "topk_residual_group_geometry.csv", group_rows)
    _write_csv(output_dir / "topk_oracle_replacement_aggregate.csv", aggregate)
    _write_csv(output_dir / "topk_residual_group_aggregate.csv", group_aggregate)
    (output_dir / "topk_oracle_replacement_report.md").write_text(
        _report(summary), encoding="utf-8"
    )
    return summary


def _load_models(reference: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in CHECKPOINTS:
        metadata = reference["checkpoints"][f"{name}_pred"]
        checkpoint = Path(str(metadata["checkpoint"])).resolve()
        config_path = Path(str(metadata["config_path"])).resolve()
        if not checkpoint.is_file() or not config_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint/config for {name}.")
        config = _read_json(config_path)
        model = _build_model(config, None, False).to(device)
        payload = load_checkpoint(checkpoint, model, map_location=device)
        model.eval()
        amp_enabled, amp_dtype = _amp_settings(config, device)
        result[name] = {
            "checkpoint": checkpoint,
            "config_path": config_path,
            "config": config,
            "model": model,
            "checkpoint_epoch": int(payload.get("epoch", -1)),
            "amp_enabled": amp_enabled,
            "amp_dtype": amp_dtype,
            "epsilon": float(config.get("target_scaling", {}).get("epsilon", 1e-12)),
        }
    return result


def _descending_residual_ranking(residual: np.ndarray) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    if not len(residual) or not np.isfinite(residual).all():
        raise ValueError("Raw residual must be a non-empty finite vector.")
    indices = np.arange(len(residual), dtype=np.int64)
    return np.lexsort((indices, -residual)).astype(np.int64)


def _topk_count(vertex_count: int, percentage: int) -> int:
    if vertex_count < 1 or percentage not in PERCENTAGES:
        raise ValueError("Invalid vertex count or replacement percentage.")
    if percentage == 0:
        return 0
    if percentage == 100:
        return vertex_count
    return min(vertex_count, int(math.ceil(vertex_count * percentage / 100.0)))


def _topk_indices(ranking: np.ndarray, vertex_count: int, percentage: int) -> np.ndarray:
    return np.asarray(ranking[: _topk_count(vertex_count, percentage)], dtype=np.int64)


def _residual_group_masks(ranking: np.ndarray, vertex_count: int) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name, lower, upper in GROUPS:
        start = 0 if lower == 0 else _topk_count(vertex_count, lower)
        stop = _topk_count(vertex_count, upper)
        mask = np.zeros(vertex_count, dtype=bool)
        mask[ranking[start:stop]] = True
        result[name] = mask
    covered = np.sum(np.stack(list(result.values())), axis=0)
    if not np.array_equal(covered, np.ones(vertex_count, dtype=np.int64)):
        raise RuntimeError("Residual percentile groups must partition all vertices exactly once.")
    return result


def _selection_audit(
    sample_id: str,
    checkpoint: str,
    selections: Mapping[int, np.ndarray],
    residual: torch.Tensor,
) -> dict[str, Any]:
    nested = all(
        set(selections[left].tolist()).issubset(set(selections[right].tolist()))
        for left, right in zip(PERCENTAGES[:-1], PERCENTAGES[1:])
    )
    counts_match = all(
        len(selections[percentage]) == _topk_count(len(residual), percentage)
        for percentage in PERCENTAGES
    )
    exact_endpoints = len(selections[0]) == 0 and len(selections[100]) == len(residual)
    topk_by_raw_residual = True
    residual_array = residual.detach().double().cpu().numpy()
    all_indices = np.arange(len(residual_array), dtype=np.int64)
    for percentage in PERCENTAGES[1:-1]:
        selected = selections[percentage]
        unselected = np.setdiff1d(all_indices, selected, assume_unique=True)
        topk_by_raw_residual = topk_by_raw_residual and bool(
            residual_array[selected].min() >= residual_array[unselected].max()
        )
    return {
        "sample_id": sample_id,
        "checkpoint": checkpoint,
        "nested": nested,
        "counts_match": counts_match,
        "exact_endpoints": exact_endpoints,
        "topk_by_raw_solver_input_residual": topk_by_raw_residual,
        "passed": nested and counts_match and exact_endpoints and topk_by_raw_residual,
    }


def _vertex_surface_distances(vertices: np.ndarray, gt_mesh: Mesh) -> np.ndarray:
    result = closest_points_on_mesh(
        np.asarray(vertices, dtype=np.float64), gt_mesh.vertices, gt_mesh.faces
    )
    return np.asarray(result.distances, dtype=np.float64)


def _group_geometry_rows(
    row: Mapping[str, Any],
    masks: Mapping[str, np.ndarray],
    raw_residual: np.ndarray,
    normalized_residual: np.ndarray,
    initial_distance: np.ndarray,
    recovered_distance: np.ndarray,
) -> list[dict[str, Any]]:
    output = []
    for group_name, mask in masks.items():
        output.append(
            {
                "checkpoint": row["checkpoint"],
                "replacement_percent": row["replacement_percent"],
                "sample_id": row["sample_id"],
                "object_id": row["object_id"],
                "residual_percentile_group": group_name,
                "vertex_count": int(mask.sum()),
                "baseline_raw_residual_mean": float(raw_residual[mask].mean()),
                "baseline_raw_residual_rms": float(
                    np.sqrt(np.mean(np.square(raw_residual[mask])))
                ),
                "baseline_normalized_residual_mean": float(
                    normalized_residual[mask].mean()
                ),
                "initial_vertex_to_gt_surface_mean": float(initial_distance[mask].mean()),
                "recovered_vertex_to_gt_surface_mean": float(
                    recovered_distance[mask].mean()
                ),
                "recovered_vertex_to_gt_surface_median": float(
                    np.median(recovered_distance[mask])
                ),
                "recovered_vertex_to_gt_surface_p95": float(
                    np.quantile(recovered_distance[mask], 0.95)
                ),
                "recovered_vertex_to_gt_surface_maximum": float(
                    recovered_distance[mask].max()
                ),
            }
        )
    return output


def _write_representative_inputs(
    output_dir: Path,
    sample_id: str,
    checkpoint: str,
    static: Mapping[str, Any],
    raw_residual: torch.Tensor,
    normalized_residual: torch.Tensor,
    ranking: np.ndarray,
    selections: Mapping[int, np.ndarray],
) -> None:
    directory = output_dir / "visualizations" / checkpoint
    vertices = static["vertices"].detach().cpu().numpy()
    faces = static["faces"].detach().cpu().numpy()
    rank_score = np.empty(len(ranking), dtype=np.float64)
    rank_score[ranking] = np.linspace(1.0, 0.0, len(ranking), endpoint=True)
    _write_heatmap_ply(
        directory / f"{sample_id}_raw_solver_input_residual.ply",
        vertices,
        faces,
        raw_residual.numpy(),
    )
    _write_heatmap_ply(
        directory / f"{sample_id}_normalized_residual_control.ply",
        vertices,
        faces,
        normalized_residual.numpy(),
    )
    _write_heatmap_ply(
        directory / f"{sample_id}_raw_residual_rank.ply",
        vertices,
        faces,
        rank_score,
    )
    npz_path = directory / f"{sample_id}_topk_diagnostics.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "raw_solver_input_residual": raw_residual.numpy(),
        "normalized_residual_control": normalized_residual.numpy(),
        "descending_raw_residual_ranking": ranking,
    }
    for percentage, indices in selections.items():
        mask = np.zeros(len(ranking), dtype=bool)
        mask[indices] = True
        arrays[f"replace_{percentage:03d}pct_mask"] = mask
    np.savez_compressed(npz_path, **arrays)


def _geometry_change(row: Mapping[str, Any]) -> dict[str, float]:
    initial = float(row["initial_chamfer"])
    refined = float(row["reconstruction_chamfer"])
    return {
        "absolute_chamfer_change": refined - initial,
        "percent_chamfer_change": 100.0 * (refined - initial) / initial,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["checkpoint"]), int(row["replacement_percent"]))].append(row)
    result = []
    for checkpoint in CHECKPOINTS:
        checkpoint_rows: dict[int, dict[str, Any]] = {}
        for percentage in PERCENTAGES:
            selected = grouped[(checkpoint, percentage)]
            if len(selected) != 25:
                raise RuntimeError(f"Expected 25 rows for {checkpoint} at {percentage}%.")
            current = {
                "checkpoint": checkpoint,
                "replacement_percent": percentage,
                "sample_count": len(selected),
                "mean_actual_replacement_percent": _mean(
                    selected, "actual_replacement_percent"
                ),
                "mean_raw_residual_energy_replaced_fraction": _mean(
                    selected, "raw_residual_energy_replaced_fraction"
                ),
                "initial_chamfer": _mean(selected, "initial_chamfer"),
                "refined_chamfer": _mean(selected, "reconstruction_chamfer"),
                "initial_point_to_surface": _mean(
                    selected, "initial_point_to_surface"
                ),
                "refined_point_to_surface": _mean(
                    selected, "reconstruction_point_to_surface"
                ),
                "initial_normal_consistency": _mean(
                    selected, "initial_normal_consistency"
                ),
                "refined_normal_consistency": _mean(
                    selected, "reconstruction_normal_consistency"
                ),
                "introduced_flipped_faces": int(
                    sum(int(row["introduced_flipped_faces"]) for row in selected)
                ),
                "improved_over_initial": int(
                    sum(bool(row["improved_over_initial"]) for row in selected)
                ),
                "mean_raw_residual_vs_recovered_surface_distance_spearman": _mean(
                    selected,
                    "raw_residual_vs_recovered_surface_distance_spearman",
                ),
            }
            checkpoint_rows[percentage] = current
        baseline = checkpoint_rows[0]
        oracle = checkpoint_rows[100]
        for percentage in PERCENTAGES:
            current = checkpoint_rows[percentage]
            current["chamfer_change_from_baseline"] = float(
                current["refined_chamfer"] - baseline["refined_chamfer"]
            )
            current["p2s_change_from_baseline"] = float(
                current["refined_point_to_surface"]
                - baseline["refined_point_to_surface"]
            )
            current["chamfer_oracle_gap_closed_fraction"] = _gap_closed(
                float(baseline["refined_chamfer"]),
                float(current["refined_chamfer"]),
                float(oracle["refined_chamfer"]),
            )
            current["p2s_oracle_gap_closed_fraction"] = _gap_closed(
                float(baseline["refined_point_to_surface"]),
                float(current["refined_point_to_surface"]),
                float(oracle["refined_point_to_surface"]),
            )
            result.append(current)
    return result


def _aggregate_groups(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["checkpoint"]),
                int(row["replacement_percent"]),
                str(row["residual_percentile_group"]),
            )
        ].append(row)
    output = []
    for checkpoint in CHECKPOINTS:
        for percentage in PERCENTAGES:
            for group_name, _, _ in GROUPS:
                selected = grouped[(checkpoint, percentage, group_name)]
                output.append(
                    {
                        "checkpoint": checkpoint,
                        "replacement_percent": percentage,
                        "residual_percentile_group": group_name,
                        "sample_count": len(selected),
                        "baseline_raw_residual_mean": _weighted_group_mean(
                            selected, "baseline_raw_residual_mean"
                        ),
                        "baseline_normalized_residual_mean": _weighted_group_mean(
                            selected, "baseline_normalized_residual_mean"
                        ),
                        "initial_vertex_to_gt_surface_mean": _weighted_group_mean(
                            selected, "initial_vertex_to_gt_surface_mean"
                        ),
                        "recovered_vertex_to_gt_surface_mean": _weighted_group_mean(
                            selected, "recovered_vertex_to_gt_surface_mean"
                        ),
                    }
                )
    return output


def _contract_audit(
    reference: Mapping[str, Any],
    aggregate: Sequence[Mapping[str, Any]],
    selection_checks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    index = {
        (str(row["checkpoint"]), int(row["replacement_percent"])): row
        for row in aggregate
    }
    metric_mapping = {
        "refined_chamfer": "reconstruction_chamfer",
        "refined_point_to_surface": "reconstruction_point_to_surface",
        "refined_normal_consistency": "reconstruction_normal_consistency",
    }
    endpoint_checks: dict[str, Any] = {}
    passed = all(bool(row["passed"]) for row in selection_checks)
    for checkpoint in CHECKPOINTS:
        learned_reference = reference["aggregate"][f"{checkpoint}_pred"]
        baseline = index[(checkpoint, 0)]
        checks = {}
        for actual_key, reference_key in metric_mapping.items():
            actual = float(baseline[actual_key])
            expected = float(learned_reference[reference_key])
            match = math.isclose(actual, expected, rel_tol=2e-3, abs_tol=2e-6)
            checks[actual_key] = {
                "reference": expected,
                "rerun": actual,
                "match": match,
            }
            passed = passed and match
        flip_difference = abs(
            int(baseline["introduced_flipped_faces"])
            - int(learned_reference["introduced_flipped_faces"])
        )
        improved_match = int(baseline["improved_over_initial"]) == int(
            learned_reference["improved_over_initial"]
        )
        checks["introduced_flipped_faces"] = {
            "absolute_difference": flip_difference,
            "match": flip_difference <= 25,
        }
        checks["improved_over_initial"] = {"match": improved_match}
        passed = passed and flip_difference <= 25 and improved_match
        endpoint_checks[f"{checkpoint}_0pct"] = checks
    oracle_reference = reference["aggregate"]["current_graph_exact_target_oracle"]
    exact_50 = index[("current_query_50k", 100)]
    exact_checks = {}
    for actual_key, reference_key in metric_mapping.items():
        actual = float(exact_50[actual_key])
        expected = float(oracle_reference[reference_key])
        match = math.isclose(actual, expected, rel_tol=2e-3, abs_tol=2e-6)
        exact_checks[actual_key] = {
            "reference": expected,
            "rerun": actual,
            "match": match,
        }
        passed = passed and match
    endpoint_checks["current_query_50k_100pct_vs_prior_oracle"] = exact_checks
    return {
        "passed": passed,
        "manifest_match": reference["manifest_sha256"] == EXPECTED_MANIFEST_SHA256,
        "selection_checks_passed": all(bool(row["passed"]) for row in selection_checks),
        "selection_checks": list(selection_checks),
        "endpoint_checks": endpoint_checks,
        "geometry_float_relative_tolerance": 2e-3,
        "geometry_float_absolute_tolerance": 2e-6,
        "flip_absolute_tolerance": 25,
    }


def _decision(
    aggregate: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    index = {
        (str(row["checkpoint"]), int(row["replacement_percent"])): row
        for row in aggregate
    }
    output: dict[str, Any] = {}
    for checkpoint in CHECKPOINTS:
        rows = [index[(checkpoint, percentage)] for percentage in PERCENTAGES]
        first_mean_below_initial = next(
            (
                int(row["replacement_percent"])
                for row in rows
                if float(row["refined_chamfer"]) < float(row["initial_chamfer"])
            ),
            None,
        )
        first_90 = next(
            (
                int(row["replacement_percent"])
                for row in rows
                if float(row["chamfer_oracle_gap_closed_fraction"]) >= 0.90
            ),
            None,
        )
        baseline_groups = {
            str(row["residual_percentile_group"]): row
            for row in groups
            if row["checkpoint"] == checkpoint and row["replacement_percent"] == 0
        }
        top = float(
            baseline_groups["top_0_1_percent"][
                "recovered_vertex_to_gt_surface_mean"
            ]
        )
        bottom = float(
            baseline_groups["bottom_50_percent"][
                "recovered_vertex_to_gt_surface_mean"
            ]
        )
        output[checkpoint] = {
            "first_replacement_percent_with_mean_chamfer_below_initial": (
                first_mean_below_initial
            ),
            "first_replacement_percent_closing_at_least_90pct_chamfer_oracle_gap": (
                first_90
            ),
            "top1_raw_residual_group_baseline_surface_distance": top,
            "bottom50_raw_residual_group_baseline_surface_distance": bottom,
            "top1_to_bottom50_surface_distance_ratio": top / bottom if bottom else math.inf,
            "one_percent_chamfer_gap_closed_fraction": float(
                index[(checkpoint, 1)]["chamfer_oracle_gap_closed_fraction"]
            ),
            "ten_percent_chamfer_gap_closed_fraction": float(
                index[(checkpoint, 10)]["chamfer_oracle_gap_closed_fraction"]
            ),
        }
    return output


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Sofa50 Synthetic-current Top-k Raw-residual Oracle Replacement",
        "",
        "## Contract",
        "",
        f"- Manifest SHA-256: `{summary['manifest_sha256']}`.",
        "- Ranking variable: `||delta_pred_raw[i]-delta_target_raw[i]||_2`.",
        "- Normalized residual is not used for selection.",
        "- Each checkpoint retains its own confidence and the fixed visibility/recovery contract.",
        "- Replacement sets are nested and ties use ascending vertex index.",
        f"- Contract audit: `{summary['contract_audit']['passed']}`.",
        "",
        "## Aggregate comparison",
        "",
        "| Checkpoint | Replacement | Raw residual energy replaced | Refined Chamfer | Chamfer gap closed | Refined P2S | P2S gap closed | Normal | Flips | Improved/25 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["aggregate"]:
        lines.append(
            f"| {row['checkpoint']} | {row['replacement_percent']}% | "
            f"{_f(row['mean_raw_residual_energy_replaced_fraction'])} | "
            f"{_f(row['refined_chamfer'])} | "
            f"{_f(row['chamfer_oracle_gap_closed_fraction'])} | "
            f"{_f(row['refined_point_to_surface'])} | "
            f"{_f(row['p2s_oracle_gap_closed_fraction'])} | "
            f"{_f(row['refined_normal_consistency'])} | "
            f"{row['introduced_flipped_faces']} | {row['improved_over_initial']}/25 |"
        )
    lines.extend(
        [
            "",
            "## Raw-residual percentile groups at baseline recovery",
            "",
            "| Checkpoint | Group | Raw residual mean | Normalized residual mean | Initial surface distance | Recovered surface distance |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["residual_group_aggregate"]:
        if int(row["replacement_percent"]) != 0:
            continue
        lines.append(
            f"| {row['checkpoint']} | {row['residual_percentile_group']} | "
            f"{_f(row['baseline_raw_residual_mean'])} | "
            f"{_f(row['baseline_normalized_residual_mean'])} | "
            f"{_f(row['initial_vertex_to_gt_surface_mean'])} | "
            f"{_f(row['recovered_vertex_to_gt_surface_mean'])} |"
        )
    lines.extend(["", "## Decision fields", ""])
    for checkpoint, values in summary["decision"].items():
        lines.extend(
            [
                f"### {checkpoint}",
                "",
                f"- First percentage with mean Chamfer below initial: `{values['first_replacement_percent_with_mean_chamfer_below_initial']}`.",
                f"- First percentage closing at least 90% of the Chamfer oracle gap: `{values['first_replacement_percent_closing_at_least_90pct_chamfer_oracle_gap']}`.",
                f"- 1% Chamfer oracle-gap closure: `{_f(values['one_percent_chamfer_gap_closed_fraction'])}`.",
                f"- 10% Chamfer oracle-gap closure: `{_f(values['ten_percent_chamfer_gap_closed_fraction'])}`.",
                f"- Baseline top-1%/bottom-50% recovered surface-distance ratio: `{_f(values['top1_to_bottom50_surface_distance_ratio'])}`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Outputs",
            "",
            "- `topk_oracle_replacement_summary.json`",
            "- `topk_oracle_replacement_per_sample.csv`",
            "- `topk_oracle_replacement_aggregate.csv`",
            "- `topk_residual_group_geometry.csv`",
            "- `topk_residual_group_aggregate.csv`",
            "- `visualizations/current_query_20k/`",
            "- `visualizations/current_query_50k/`",
            "",
        ]
    )
    return "\n".join(lines)


def print_terminal_summary(summary: Mapping[str, Any], output_dir: str | Path) -> None:
    print("checkpoint\treplace\tchamfer\tp2s\tflips\timproved")
    for row in summary["aggregate"]:
        print(
            f"{row['checkpoint']}\t{row['replacement_percent']}%\t"
            f"{_f(row['refined_chamfer'])}\t{_f(row['refined_point_to_surface'])}\t"
            f"{row['introduced_flipped_faces']}\t{row['improved_over_initial']}/25"
        )
    print(f"report\t{Path(output_dir).resolve() / 'topk_oracle_replacement_report.md'}")


def _gap_closed(baseline: float, current: float, oracle: float) -> float:
    denominator = baseline - oracle
    return (baseline - current) / denominator if abs(denominator) > 1e-15 else 1.0


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    from scipy.stats import spearmanr

    statistic = spearmanr(left, right).statistic
    return float(statistic) if np.isfinite(statistic) else 0.0


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _weighted_group_mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    weights = np.asarray([int(row["vertex_count"]) for row in rows], dtype=np.float64)
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return float(np.sum(weights * values) / np.sum(weights))


def _f(value: Any) -> str:
    return f"{float(value):.9g}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
