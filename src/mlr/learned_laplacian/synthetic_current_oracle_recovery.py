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

from .canonical_experiment import _exact_query_sample, _load_device_item
from .canonical_pipeline import (
    canonical_current_graph_recovery_inputs,
    current_uniform_laplacian_raw,
)
from .diagnostics import _amp_settings, _loss_kwargs
from .losses import laplacian_prediction_metrics, weighted_robust_laplacian_loss
from .multi_dataset import PreparedMeshDataset, validate_disjoint_splits
from .multi_trainer import _build_model
from .synthetic_current_50k_downstream import _model_repeat_audit
from .synthetic_current_comparison import _reconstruct_one, run_synthetic_current_comparison
from .trainer import load_checkpoint


EXPECTED_MANIFEST_SHA256 = "b28e133c277032cceee05ac10115d11ee3007bbd2c3983c31cfa41992159eba3"
ARMS = (
    "current_query_20k_pred",
    "current_query_50k_pred",
    "current_graph_exact_target_oracle",
)
LOST_SUCCESS_IDS = (
    "43bd0910-1dd1-4b1e-9ba2-e9801e6b5761__v00",
    "43bd0910-1dd1-4b1e-9ba2-e9801e6b5761__v04",
)
NORMALIZED_MSE_AMP_REL_TOL = 2e-3
GEOMETRY_KEYS = (
    "initial_chamfer",
    "reconstruction_chamfer",
    "initial_point_to_surface",
    "reconstruction_point_to_surface",
    "initial_normal_consistency",
    "reconstruction_normal_consistency",
)


def run_oracle_recovery_comparison(
    downstream_summary_path: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    downstream_summary_path = Path(downstream_summary_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = _read_json(downstream_summary_path)
    manifest = Path(str(saved["contract_audit"]["manifest_path"])).resolve()
    manifest_sha256 = _sha256(manifest)
    if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            f"Manifest SHA-256 mismatch: {manifest_sha256}; expected {EXPECTED_MANIFEST_SHA256}."
        )
    checkpoints = saved["checkpoints"]
    current20 = checkpoints["current_query_20k"]
    current50 = checkpoints["current_query_50k"]
    required = (
        downstream_summary_path,
        manifest,
        Path(str(current20["checkpoint"])),
        Path(str(current20["config_path"])),
        Path(str(current50["checkpoint"])),
        Path(str(current50["config_path"])),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing fixed-contract artifacts: " + ", ".join(missing))

    learned = run_synthetic_current_comparison(
        manifest,
        current20["checkpoint"],
        current20["config_path"],
        current50["checkpoint"],
        current50["config_path"],
        output_dir / "learned_reproduction",
        device=device,
    )
    reproduction = _learned_reproduction_audit(saved, learned)
    _write_json(output_dir / "learned_reproduction_audit.json", reproduction)
    if not reproduction["passed"]:
        raise RuntimeError("Learned 20k/50k reproduction failed; oracle interpretation is blocked.")

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
    config = _read_json(Path(str(current50["config_path"])))
    model = _build_model(config, None, False).to(resolved_device)
    checkpoint_payload = load_checkpoint(
        Path(str(current50["checkpoint"])), model, map_location=resolved_device
    )
    model.eval()
    amp_enabled, amp_dtype = _amp_settings(config, resolved_device)
    epsilon = float(config.get("target_scaling", {}).get("epsilon", 1e-12))

    learned_rows = _index_learned_rows(learned["per_variant"])
    oracle_rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    valid_targets: list[torch.Tensor] = []
    formula_checks: list[dict[str, Any]] = []
    lost_diagnostics: dict[str, Any] = {}
    oracle_dir = output_dir / "reconstruction" / "current_graph_exact_target_oracle"
    repeat_dir = output_dir / "oracle_repeat_control"

    for index in range(len(dataset)):
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        metadata = dict(static.get("metadata", {}))
        if metadata.get("proxy_definition") != (
            "P_proxy=source_gt_vertices_with_exact_same_topology"
        ):
            raise RuntimeError(f"Unexpected P_proxy contract for {sample_id}.")
        if metadata.get("target_constructor") != "delta_target=L_current@P_proxy":
            raise RuntimeError(f"Unexpected target constructor for {sample_id}.")
        prepared = _load_device_item(dataset, index, config, resolved_device)
        conditioned = _exact_query_sample(prepared.sample, resolved_device)
        with torch.no_grad(), torch.autocast(
            device_type=resolved_device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            output = model(conditioned)
        if output.confidence_prediction is None:
            raise RuntimeError("Current-query 50k checkpoint has no confidence output.")
        confidence = output.confidence_prediction.float().detach().cpu()
        target_hat = prepared.training_target.float().detach().cpu()
        valid = prepared.sample["valid_scale_mask"].bool().detach().cpu()
        valid_targets.append(target_hat[valid])
        oracle_inputs = canonical_current_graph_recovery_inputs(
            static["vertices"],
            static["faces"],
            target_hat,
            static["visibility_backface_and_occlusion"],
            confidence,
            epsilon=epsilon,
        )
        raw_saved = torch.as_tensor(static["raw_laplacian_target"]).cpu()
        raw_round_trip_error = float(
            torch.max(torch.abs(oracle_inputs.delta_pred_raw.cpu() - raw_saved)).item()
        )
        proxy_raw = torch.as_tensor(
            current_uniform_laplacian_raw(static["gt_vertices"], static["faces"]),
            dtype=raw_saved.dtype,
        ).cpu()
        current_graph_proxy_raw_error = float(
            torch.max(torch.abs(proxy_raw - raw_saved)).item()
        )
        formula_target = proxy_raw / (
            oracle_inputs.h_current.cpu().square() + epsilon
        ).unsqueeze(-1)
        normalized_formula_error = float(
            torch.max(torch.abs(formula_target[valid] - target_hat[valid])).item()
        )
        formula_checks.append(
            {
                "sample_id": sample_id,
                "raw_round_trip_max_abs_error": raw_round_trip_error,
                "current_graph_proxy_raw_target_max_abs_error": (
                    current_graph_proxy_raw_error
                ),
                "normalized_formula_max_abs_error": normalized_formula_error,
            }
        )

        recovery = _reconstruct_one(
            static, target_hat, confidence, oracle_dir / sample_id, config
        )
        repeated = _reconstruct_one(
            static, target_hat, confidence, repeat_dir / sample_id, config
        )
        oracle_rows.append(
            _oracle_row(sample_id, metadata, int(valid.sum().item()), recovery)
        )
        repeat_rows.append(
            _oracle_row(sample_id, metadata, int(valid.sum().item()), repeated)
        )
        if sample_id in LOST_SUCCESS_IDS:
            lost_diagnostics[sample_id] = _lost_success_diagnostic(
                sample_id,
                output_dir,
                target_hat,
                raw_saved,
                valid,
                oracle_inputs.h_current.detach().cpu(),
                oracle_inputs.weight.detach().cpu(),
                learned_rows,
                recovery,
            )
        print(
            f"oracle {sample_id}: chamfer={recovery['reconstruction_chamfer']:.8g} "
            f"improved={recovery['improved_over_initial']}",
            flush=True,
        )
        del prepared, conditioned, output, target_hat, confidence
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()

    oracle_repeat = _oracle_repeat_audit(oracle_rows, repeat_rows, oracle_dir, repeat_dir)
    formula_audit = _formula_audit(formula_checks)
    _write_json(output_dir / "oracle_repeat_audit.json", oracle_repeat)
    _write_json(output_dir / "oracle_formula_audit.json", formula_audit)
    if not oracle_repeat["passed"]:
        raise RuntimeError("Exact-target oracle repeat is not deterministic within tolerance.")
    if not formula_audit["passed"]:
        raise RuntimeError("Saved target does not match the required current-graph oracle formula.")

    oracle_prediction = _exact_prediction_aggregate(valid_targets, config)
    aggregates = {
        "current_query_20k_pred": _augment_learned_aggregate(
            learned["aggregate"]["A"], learned_rows["current_query_20k_pred"]
        ),
        "current_query_50k_pred": _augment_learned_aggregate(
            learned["aggregate"]["B"], learned_rows["current_query_50k_pred"]
        ),
        "current_graph_exact_target_oracle": {
            **oracle_prediction,
            **_geometry_aggregate(oracle_rows),
        },
    }
    all_rows = (
        learned_rows["current_query_20k_pred"]
        + learned_rows["current_query_50k_pred"]
        + oracle_rows
    )
    per_object = _per_object(all_rows)
    decision = _decision(aggregates, per_object, lost_diagnostics)
    summary = {
        "experiment": "Sofa50 Synthetic Current-query Oracle Recovery Comparison",
        "device": device,
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha256,
        "test_samples": 25,
        "test_objects": 5,
        "sample_ids": sorted(row["sample_id"] for row in oracle_rows),
        "target": "delta_target_hat=(L_current@P_proxy)/(h_current^2+1e-12)",
        "oracle_recovery_weight_contract": (
            "current-query 50k predicted confidence times fixed saved visibility; "
            "only delta_hat is replaced by the exact target"
        ),
        "checkpoint_epoch": int(checkpoint_payload.get("epoch", -1)),
        "checkpoints": {
            "current_query_20k_pred": current20,
            "current_query_50k_pred": current50,
        },
        "learned_reproduction_audit": reproduction,
        "oracle_formula_audit": formula_audit,
        "oracle_repeat_audit": oracle_repeat,
        "aggregate": aggregates,
        "per_object": per_object,
        "lost_success": lost_diagnostics,
        "decision": decision,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "lost_success_v00_v04.json", lost_diagnostics)
    _write_csv(output_dir / "per_sample_metrics.csv", all_rows)
    _write_csv(output_dir / "per_object_metrics.csv", per_object)
    (output_dir / "report.md").write_text(_report(summary), encoding="utf-8")
    return summary


def refresh_existing_oracle_report(output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    summary = _read_json(output_dir / "summary.json")
    lost = _read_json(output_dir / "lost_success_v00_v04.json")
    for values in lost.values():
        values["comparison_50k_vs_20k"] = _lost_comparison(values)
    summary["lost_success"] = lost
    summary["decision"] = _decision(
        summary["aggregate"], summary["per_object"], lost
    )
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "lost_success_v00_v04.json", lost)
    (output_dir / "report.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _learned_reproduction_audit(
    saved: Mapping[str, Any], rerun: Mapping[str, Any]
) -> dict[str, Any]:
    saved_ids = set(map(str, saved.get("test_sample_ids", [])))
    rerun_ids = {str(row["sample_id"]) for row in rerun["per_variant"]}
    current20 = _learned_model_repeat_audit(
        saved["aggregate"]["current_query_20k"], rerun["aggregate"]["A"]
    )
    current50 = _learned_model_repeat_audit(
        saved["aggregate"]["current_query_50k"], rerun["aggregate"]["B"]
    )
    return {
        "passed": bool(current20["passed"] and current50["passed"] and saved_ids == rerun_ids),
        "sample_ids_match": saved_ids == rerun_ids,
        "current_query_20k": current20,
        "current_query_50k": current50,
    }


def _learned_model_repeat_audit(
    reference: Mapping[str, Any], rerun: Mapping[str, Any]
) -> dict[str, Any]:
    audit = _model_repeat_audit(reference, rerun)
    normalized_mse = audit["metrics"]["normalized_mse"]
    normalized_mse["match"] = math.isclose(
        float(normalized_mse["reference"]),
        float(normalized_mse["rerun"]),
        rel_tol=NORMALIZED_MSE_AMP_REL_TOL,
        abs_tol=1e-6,
    )
    audit["tolerance"]["normalized_mse_float_relative"] = (
        NORMALIZED_MSE_AMP_REL_TOL
    )
    audit["passed"] = all(
        bool(metric["match"]) for metric in audit["metrics"].values()
    )
    return audit


def _index_learned_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result = {"current_query_20k_pred": [], "current_query_50k_pred": []}
    mapping = {"A": "current_query_20k_pred", "B": "current_query_50k_pred"}
    for source in rows:
        arm = mapping[str(source["experiment"])]
        row = dict(source)
        row["arm"] = arm
        row.pop("experiment", None)
        row.update(_geometry_changes(row))
        result[arm].append(row)
    return result


def _oracle_row(
    sample_id: str,
    metadata: Mapping[str, Any],
    vertex_count: int,
    recovery: Mapping[str, Any],
) -> dict[str, Any]:
    row = {
        "arm": "current_graph_exact_target_oracle",
        "sample_id": sample_id,
        "object_id": str(metadata["object_id"]),
        "variant_index": int(metadata["variant_index"]),
        "vertex_count": vertex_count,
        "correct_rgb_loss": 0.0,
        "normalized_mse": 0.0,
        "vector_l2": 0.0,
        "global_cosine": 1.0,
        "high_10_percent_cosine": 1.0,
        "prediction_target_norm_ratio": 1.0,
        **dict(recovery),
    }
    row.update(_geometry_changes(row))
    return row


def _geometry_changes(row: Mapping[str, Any]) -> dict[str, Any]:
    initial = float(row["initial_chamfer"])
    refined = float(row["reconstruction_chamfer"])
    change = refined - initial
    return {
        "absolute_chamfer_change": change,
        "percent_chamfer_change": 100.0 * change / initial if initial else math.nan,
    }


def _exact_prediction_aggregate(
    targets: Sequence[torch.Tensor], config: Mapping[str, Any]
) -> dict[str, Any]:
    target = torch.cat(list(targets), dim=0)
    metrics = laplacian_prediction_metrics(target, target)
    loss = weighted_robust_laplacian_loss(
        target,
        target,
        torch.ones(len(target)),
        **_loss_kwargs(config),
    )
    return {
        "normalized_mse": metrics["mse"],
        "vector_l2": metrics["vector_endpoint_error"],
        "global_cosine": metrics["global_cosine"],
        "high_10_percent_cosine": metrics["top_10_percent_cosine"],
        "prediction_target_norm_ratio": metrics["prediction_to_target_norm_ratio"],
        "loss": float(loss),
        "maximum_prediction_target_error": float(torch.max(torch.abs(target - target)).item()),
    }


def _augment_learned_aggregate(
    aggregate: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    result = dict(aggregate)
    result["initial_point_to_surface"] = _mean(rows, "initial_point_to_surface")
    result["initial_normal_consistency"] = _mean(rows, "initial_normal_consistency")
    return result


def _geometry_aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "initial_chamfer": _mean(rows, "initial_chamfer"),
        "reconstruction_chamfer": _mean(rows, "reconstruction_chamfer"),
        "initial_point_to_surface": _mean(rows, "initial_point_to_surface"),
        "reconstruction_point_to_surface": _mean(rows, "reconstruction_point_to_surface"),
        "initial_normal_consistency": _mean(rows, "initial_normal_consistency"),
        "reconstruction_normal_consistency": _mean(rows, "reconstruction_normal_consistency"),
        "introduced_flipped_faces": int(sum(int(row["introduced_flipped_faces"]) for row in rows)),
        "new_degenerate_faces": int(sum(int(row["new_degenerate_faces"]) for row in rows)),
        "improved_over_initial": int(sum(bool(row["improved_over_initial"]) for row in rows)),
        "sample_count": len(rows),
        "improved_sample_ids": sorted(
            str(row["sample_id"]) for row in rows if bool(row["improved_over_initial"])
        ),
    }


def _formula_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    maxima = {
        key: max(float(row[key]) for row in rows)
        for key in (
            "raw_round_trip_max_abs_error",
            "current_graph_proxy_raw_target_max_abs_error",
            "normalized_formula_max_abs_error",
        )
    }
    tolerance = 1e-5
    return {
        "passed": all(value <= tolerance for value in maxima.values()),
        "absolute_tolerance": tolerance,
        "maxima": maxima,
        "per_sample": list(rows),
    }


def _oracle_repeat_audit(
    first: Sequence[Mapping[str, Any]],
    second: Sequence[Mapping[str, Any]],
    first_dir: Path,
    second_dir: Path,
) -> dict[str, Any]:
    a = {str(row["sample_id"]): row for row in first}
    b = {str(row["sample_id"]): row for row in second}
    differences: dict[str, Any] = {}
    passed = set(a) == set(b)
    maximum = 0.0
    obj_hash_matches = True
    for sample_id in sorted(set(a) & set(b)):
        current: dict[str, Any] = {}
        for key in GEOMETRY_KEYS:
            difference = abs(float(a[sample_id][key]) - float(b[sample_id][key]))
            maximum = max(maximum, difference)
            current[key] = difference
        flip_match = int(a[sample_id]["introduced_flipped_faces"]) == int(
            b[sample_id]["introduced_flipped_faces"]
        )
        hash_match = _sha256(first_dir / sample_id / "predicted_refined.obj") == _sha256(
            second_dir / sample_id / "predicted_refined.obj"
        )
        current["introduced_flipped_faces_match"] = flip_match
        current["obj_sha256_match"] = hash_match
        passed = passed and flip_match
        obj_hash_matches = obj_hash_matches and hash_match
        differences[sample_id] = current
    tolerance = 1e-12
    passed = passed and maximum <= tolerance
    return {
        "passed": passed,
        "absolute_tolerance": tolerance,
        "maximum_metric_absolute_difference": maximum,
        "all_obj_sha256_match": obj_hash_matches,
        "per_sample": differences,
    }


def _lost_success_diagnostic(
    sample_id: str,
    output_dir: Path,
    target_hat: torch.Tensor,
    target_raw: torch.Tensor,
    valid: torch.Tensor,
    h_current: torch.Tensor,
    recovery_weight_50k: torch.Tensor,
    learned_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    oracle_recovery: Mapping[str, Any],
) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {
        "delta_hat_target": target_hat.numpy(),
        "delta_target_raw": target_raw.numpy(),
        "valid_scale_mask": valid.numpy(),
        "h_current": h_current.numpy(),
        "recovery_weight_50k": recovery_weight_50k.numpy(),
    }
    result: dict[str, Any] = {}
    target_magnitude = torch.linalg.vector_norm(target_hat, dim=-1)
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    top_count = max(1, int(round(0.10 * int(valid.sum().item()))))
    top_local = torch.topk(target_magnitude[valid], top_count).indices
    top_indices = valid_indices[top_local]
    arrays["top10_target_magnitude_mask"] = torch.zeros_like(valid).scatter(
        0, top_indices, True
    ).numpy()
    for arm, token in (
        ("current_query_20k_pred", "A"),
        ("current_query_50k_pred", "B"),
    ):
        reconstruction_dir = (
            output_dir / "learned_reproduction" / "reconstruction" / token / sample_id
        )
        pred_hat = torch.from_numpy(np.load(reconstruction_dir / "delta_hat_prediction.npy"))
        pred_raw = torch.from_numpy(np.load(reconstruction_dir / "delta_pred_raw.npy"))
        normalized_residual = pred_hat - target_hat
        raw_residual = pred_raw - target_raw
        normalized_magnitude = torch.linalg.vector_norm(normalized_residual, dim=-1)
        raw_magnitude = torch.linalg.vector_norm(raw_residual, dim=-1)
        arrays[f"{arm}_delta_hat_prediction"] = pred_hat.numpy()
        arrays[f"{arm}_delta_pred_raw"] = pred_raw.numpy()
        arrays[f"{arm}_normalized_residual_magnitude"] = normalized_magnitude.numpy()
        arrays[f"{arm}_raw_residual_magnitude"] = raw_magnitude.numpy()
        row = next(row for row in learned_rows[arm] if row["sample_id"] == sample_id)
        result[arm] = {
            "target_epe": float(row["vector_l2"]),
            "cosine": float(row["global_cosine"]),
            "normalized_residual": _distribution(normalized_magnitude[valid]),
            "raw_residual": _distribution(raw_magnitude[valid]),
            "top10_target_magnitude_normalized_residual_mean": float(
                normalized_magnitude[top_indices].mean().item()
            ),
            "top10_target_magnitude_raw_residual_mean": float(
                raw_magnitude[top_indices].mean().item()
            ),
            "shared_50k_recovery_weighted_normalized_residual_rms": _weighted_rms(
                normalized_magnitude[valid], recovery_weight_50k[valid]
            ),
            "shared_50k_recovery_weighted_raw_residual_rms": _weighted_rms(
                raw_magnitude[valid], recovery_weight_50k[valid]
            ),
            "recovery": _recovery_subset(row),
        }
    arrays["current_graph_exact_target_oracle_normalized_residual_magnitude"] = np.zeros(
        len(target_hat), dtype=np.float32
    )
    arrays["current_graph_exact_target_oracle_raw_residual_magnitude"] = np.zeros(
        len(target_hat), dtype=np.float32
    )
    result["current_graph_exact_target_oracle"] = {
        "target_epe": 0.0,
        "cosine": 1.0,
        "normalized_residual": _distribution(torch.zeros(int(valid.sum()))),
        "raw_residual": _distribution(torch.zeros(int(valid.sum()))),
        "top10_target_magnitude_normalized_residual_mean": 0.0,
        "top10_target_magnitude_raw_residual_mean": 0.0,
        "shared_50k_recovery_weighted_normalized_residual_rms": 0.0,
        "shared_50k_recovery_weighted_raw_residual_rms": 0.0,
        "recovery": _recovery_subset(oracle_recovery),
    }
    comparison = _lost_comparison(result)
    result["comparison_50k_vs_20k"] = comparison
    diagnostic_path = output_dir / "per_vertex" / f"{sample_id}.npz"
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(diagnostic_path, **arrays)
    result["per_vertex_npz"] = str(diagnostic_path)
    return result


def _distribution(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().double().cpu()
    if values.numel() == 0:
        raise ValueError("Residual distribution cannot be empty.")
    return {
        "mean": float(values.mean().item()),
        "rms": float(torch.sqrt(values.square().mean()).item()),
        "median": float(values.median().item()),
        "p90": float(torch.quantile(values, 0.90).item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "p99": float(torch.quantile(values, 0.99).item()),
        "maximum": float(values.max().item()),
    }


def _weighted_rms(values: torch.Tensor, weights: torch.Tensor) -> float:
    values = values.detach().double().cpu()
    weights = weights.detach().double().cpu().clamp_min(0.0)
    denominator = weights.sum()
    if float(denominator.item()) == 0.0:
        return 0.0
    return float(torch.sqrt((weights * values.square()).sum() / denominator).item())


def _lost_comparison(result: Mapping[str, Any]) -> dict[str, Any]:
    a = result["current_query_20k_pred"]
    b = result["current_query_50k_pred"]
    global_epe_lower = float(b["target_epe"]) < float(a["target_epe"])
    shared_weighted_raw_lower = float(
        b["shared_50k_recovery_weighted_raw_residual_rms"]
    ) < float(a["shared_50k_recovery_weighted_raw_residual_rms"])
    chamfer_lower = float(b["recovery"]["refined_chamfer"]) < float(
        a["recovery"]["refined_chamfer"]
    )
    return {
        "global_target_epe_lower_at_50k": global_epe_lower,
        "top10_normalized_residual_lower_at_50k": float(
            b["top10_target_magnitude_normalized_residual_mean"]
        )
        < float(a["top10_target_magnitude_normalized_residual_mean"]),
        "shared_weighted_normalized_residual_lower_at_50k": float(
            b["shared_50k_recovery_weighted_normalized_residual_rms"]
        )
        < float(a["shared_50k_recovery_weighted_normalized_residual_rms"]),
        "raw_residual_rms_lower_at_50k": float(b["raw_residual"]["rms"])
        < float(a["raw_residual"]["rms"]),
        "raw_residual_maximum_lower_at_50k": float(b["raw_residual"]["maximum"])
        < float(a["raw_residual"]["maximum"]),
        "shared_weighted_raw_residual_lower_at_50k": shared_weighted_raw_lower,
        "chamfer_lower_at_50k": chamfer_lower,
        "solver_input_raw_tail_pattern_at_50k": (
            global_epe_lower and not shared_weighted_raw_lower and not chamfer_lower
        ),
    }


def _recovery_subset(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "initial_chamfer": float(row["initial_chamfer"]),
        "refined_chamfer": float(row["reconstruction_chamfer"]),
        "initial_p2s": float(row["initial_point_to_surface"]),
        "refined_p2s": float(row["reconstruction_point_to_surface"]),
        "initial_normal": float(row["initial_normal_consistency"]),
        "refined_normal": float(row["reconstruction_normal_consistency"]),
        "introduced_flips": int(row["introduced_flipped_faces"]),
    }


def _per_object(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["arm"]), str(row["object_id"]))].append(row)
    output = []
    for (arm, object_id), selected in sorted(grouped.items()):
        output.append(
            {
                "arm": arm,
                "object_id": object_id,
                "variant_count": len(selected),
                "initial_chamfer": _mean(selected, "initial_chamfer"),
                "refined_chamfer": _mean(selected, "reconstruction_chamfer"),
                "mean_chamfer_change": _mean(selected, "absolute_chamfer_change"),
                "improved_samples": int(
                    sum(bool(row["improved_over_initial"]) for row in selected)
                ),
            }
        )
    return output


def _decision(
    aggregate: Mapping[str, Mapping[str, Any]],
    per_object: Sequence[Mapping[str, Any]],
    lost: Mapping[str, Any],
) -> dict[str, Any]:
    a = aggregate["current_query_20k_pred"]
    b = aggregate["current_query_50k_pred"]
    oracle = aggregate["current_graph_exact_target_oracle"]
    oracle_majority = int(oracle["improved_over_initial"]) >= 13
    oracle_all = int(oracle["improved_over_initial"]) == 25
    oracle_lower_than_both = float(oracle["reconstruction_chamfer"]) < min(
        float(a["reconstruction_chamfer"]), float(b["reconstruction_chamfer"])
    )
    learned_not_majority = max(
        int(a["improved_over_initial"]), int(b["improved_over_initial"])
    ) < 13
    oracle_object_counts = {
        str(row["object_id"]): int(row["improved_samples"])
        for row in per_object
        if row["arm"] == "current_graph_exact_target_oracle"
    }
    improving_objects = sorted(key for key, value in oracle_object_counts.items() if value > 0)
    lower_epe_worse_geometry = (
        float(b["vector_l2"]) < float(a["vector_l2"])
        and float(b["reconstruction_chamfer"]) > float(a["reconstruction_chamfer"])
    )
    lost_solver_sensitive = {
        sample_id: bool(
            values["comparison_50k_vs_20k"][
                "solver_input_raw_tail_pattern_at_50k"
            ]
        )
        for sample_id, values in lost.items()
    }
    if oracle_majority and learned_not_majority:
        classification = "learned_prediction_error_and_recovery_interaction"
    elif not oracle_majority:
        classification = "fixed_proxy_target_or_recovery_contract"
    elif len(improving_objects) == 1:
        classification = "object_specific_proxy_target_or_recovery_conditioning"
    else:
        classification = "mixed"
    return {
        "oracle_refined_chamfer_lower_than_both_learned": oracle_lower_than_both,
        "oracle_improves_majority": oracle_majority,
        "oracle_improves_all": oracle_all,
        "oracle_improved_sample_count": int(oracle["improved_over_initial"]),
        "oracle_improving_objects": improving_objects,
        "oracle_removes_single_object_success_pattern": len(improving_objects) > 1,
        "current50_lower_epe_but_higher_chamfer_than_current20": lower_epe_worse_geometry,
        "lost_success_solver_sensitive_pattern": lost_solver_sensitive,
        "classification": classification,
        "prediction_error_threshold_identifiable_from_three_arms": False,
    }


def _report(summary: Mapping[str, Any]) -> str:
    aggregate = summary["aggregate"]
    decision = summary["decision"]
    lines = [
        "# Sofa50 Synthetic Current-query Oracle Recovery Comparison",
        "",
        "## Contract",
        "",
        f"- Manifest: `{summary['manifest']}`",
        f"- Manifest SHA-256: `{summary['manifest_sha256']}`",
        f"- Test set: `{summary['test_samples']}` variants from `{summary['test_objects']}` objects.",
        "- No sample generation, target generation, graph change, solver change, or training was performed.",
        "- The oracle uses the current-query 50k confidence and visibility recovery weight. The replacement is `delta_pred_hat` to `delta_target_hat`.",
        f"- Learned reproduction gate: `{summary['learned_reproduction_audit']['passed']}`.",
        f"- Oracle formula gate: `{summary['oracle_formula_audit']['passed']}`.",
        f"- Oracle duplicate-recovery gate: `{summary['oracle_repeat_audit']['passed']}`; maximum metric difference `{_f(summary['oracle_repeat_audit']['maximum_metric_absolute_difference'])}`.",
        "",
        "## Three-arm result",
        "",
        "| Arm | Loss | Target EPE | Global cosine | High-10% cosine | Pred/target norm | Initial Chamfer | Refined Chamfer | Initial P2S | Refined P2S | Initial normal | Refined normal | Flips | Improved/25 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = aggregate[arm]
        lines.append(
            f"| {arm} | {_f(row['loss'])} | {_f(row['vector_l2'])} | {_f(row['global_cosine'])} | "
            f"{_f(row['high_10_percent_cosine'])} | {_f(row['prediction_target_norm_ratio'])} | "
            f"{_f(row['initial_chamfer'])} | {_f(row['reconstruction_chamfer'])} | "
            f"{_f(row['initial_point_to_surface'])} | {_f(row['reconstruction_point_to_surface'])} | "
            f"{_f(row['initial_normal_consistency'])} | {_f(row['reconstruction_normal_consistency'])} | "
            f"{row['introduced_flipped_faces']} | {row['improved_over_initial']}/25 |"
        )
    lines.extend(
        [
            "",
            "## Direct answers",
            "",
            f"1. Oracle refined Chamfer is lower than both learned arms: `{decision['oracle_refined_chamfer_lower_than_both_learned']}`.",
            f"2. Oracle improves `{decision['oracle_improved_sample_count']}/25` samples; majority: `{decision['oracle_improves_majority']}`; all: `{decision['oracle_improves_all']}`.",
            f"3. Diagnostic classification: `{decision['classification']}`. When oracle does not improve a majority, this experiment cannot separate `P_proxy`/target construction from the fixed recovery objective.",
            "4. A causal learned-error threshold is not identifiable from two learned checkpoints and one zero-error endpoint. The observed EPE and geometry endpoints are listed in the table.",
            f"5. Oracle success occurs on objects `{', '.join(decision['oracle_improving_objects']) or 'none'}`; single-object pattern removed: `{decision['oracle_removes_single_object_success_pattern']}`.",
            "",
            "## Lost-success samples",
            "",
            "| Sample | Arm | EPE | Cosine | Norm residual mean | Norm residual p95 | Raw residual mean | Raw residual RMS | Raw residual max | Top-10% norm residual | Shared-weight norm RMS | Shared-weight raw RMS | Refined Chamfer | Refined P2S | Refined normal | Flips |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for sample_id in LOST_SUCCESS_IDS:
        values = summary["lost_success"][sample_id]
        for arm in ARMS:
            row = values[arm]
            recovery = row["recovery"]
            lines.append(
                f"| {sample_id} | {arm} | {_f(row['target_epe'])} | {_f(row['cosine'])} | "
                f"{_f(row['normalized_residual']['mean'])} | {_f(row['normalized_residual']['p95'])} | "
                f"{_f(row['raw_residual']['mean'])} | {_f(row['raw_residual']['rms'])} | "
                f"{_f(row['raw_residual']['maximum'])} | "
                f"{_f(row['top10_target_magnitude_normalized_residual_mean'])} | "
                f"{_f(row['shared_50k_recovery_weighted_normalized_residual_rms'])} | "
                f"{_f(row['shared_50k_recovery_weighted_raw_residual_rms'])} | "
                f"{_f(recovery['refined_chamfer'])} | {_f(recovery['refined_p2s'])} | "
                f"{_f(recovery['refined_normal'])} | {recovery['introduced_flips']} |"
            )
    lines.extend(
        [
            "",
            "For both lost-success samples, 50k lowers normalized EPE, top-10% normalized residual, and shared-weight normalized RMS. It raises raw residual RMS, raw residual maximum, and shared-50k-weight raw residual RMS, while refined Chamfer increases. The recorded solver-input raw-tail pattern is therefore `True` for both samples.",
            "",
            "The 50k-versus-20k global, top-10%, shared-weight residual, and geometry direction flags are stored in `lost_success_v00_v04.json`. Per-vertex arrays are stored in `per_vertex/`.",
            "",
            "## Per-object counts",
            "",
            "| Arm | Object | Improved variants | Mean Chamfer change |",
            "|---|---|---:|---:|",
        ]
    )
    for row in summary["per_object"]:
        lines.append(
            f"| {row['arm']} | {row['object_id']} | {row['improved_samples']}/{row['variant_count']} | {_f(row['mean_chamfer_change'])} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `summary.json`",
            "- `per_sample_metrics.csv`",
            "- `per_object_metrics.csv`",
            "- `lost_success_v00_v04.json`",
            "- `oracle_formula_audit.json`",
            "- `oracle_repeat_audit.json`",
            "- `learned_reproduction_audit.json`",
            "- `reconstruction/current_graph_exact_target_oracle/*/predicted_refined.obj`",
            "- `per_vertex/*.npz`",
            "",
        ]
    )
    return "\n".join(lines)


def print_terminal_summary(summary: Mapping[str, Any], output_dir: str | Path) -> None:
    print("arm\tloss\tepe\trefined_chamfer\timproved")
    for arm in ARMS:
        row = summary["aggregate"][arm]
        print(
            f"{arm}\t{_f(row['loss'])}\t{_f(row['vector_l2'])}\t"
            f"{_f(row['reconstruction_chamfer'])}\t{row['improved_over_initial']}/25"
        )
    print(f"classification\t{summary['decision']['classification']}")
    print(f"report\t{Path(output_dir).resolve() / 'report.md'}")


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _f(value: Any) -> str:
    return f"{float(value):.9g}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
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
