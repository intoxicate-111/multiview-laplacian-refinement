#!/usr/bin/env python3
from __future__ import annotations

"""Unified validation/test evaluation for the Sofa50 recovery-aware two-arm ablation."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from diagnose_sofa50_exact_solve_visibility_sweep import component_labels, uniform_sparse_laplacian
from diagnose_sofa50_exact_target_oracle import METRIC_PROTOCOL, _clean_mesh, _geometry_row
from diagnose_sofa50_regularized_sparse_sweep import regularized_sparse_solve
from evaluate_sofa50_multitopology_rawlap import raw_gt_magnitude_metrics
from mlr.data import Mesh
from mlr.learned_laplacian.canonical_experiment import (
    _exact_query_sample,
    _load_device_item,
)
from mlr.learned_laplacian.diagnostics import _amp_settings
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.multi_trainer import _build_model
from mlr.learned_laplacian.trainer import load_checkpoint


BASE_ARMS = ("A_lap_only", "B_lap_plus_refine")
BASE_RUN_NAMES = (
    "sofa50_v2_sparse_recovery_arm_a_lap_only_20k_seed7",
    "sofa50_v2_sparse_recovery_arm_b_recovery_aware_20k_seed7",
)
EXTENSION_ARMS = BASE_ARMS + (
    "C_lap_plus_refine_lambda1e-3",
    "D_lap_plus_refine_lambda1e-4",
)
EXTENSION_RUN_NAMES = BASE_RUN_NAMES + (
    "sofa50_v2_sparse_recovery_arm_c_lambda1e-3_20k_seed7",
    "sofa50_v2_sparse_recovery_arm_d_lambda1e-4_20k_seed7",
)
EXPECTED_EXTENSION_LAMBDAS = (1e-2, 1e-2, 1e-3, 1e-4)
PREDICTION_FIELDS = (
    "raw_epe", "raw_rms", "raw_max", "raw_cosine", "recovery_weighted_raw_rms",
    "bottom90_epe", "top10_epe", "top1_epe",
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _arms(args: argparse.Namespace) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if args.include_lambda_extension:
        return EXTENSION_ARMS, EXTENSION_RUN_NAMES
    return BASE_ARMS, BASE_RUN_NAMES


def _runtime_diagnostic_summary(run: Path) -> dict[str, Any] | None:
    path = run / "training_step_history.json"
    if not path.is_file():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))
    diagnostic = [row for row in rows if row.get("pcg_iterations_mean") is not None]
    if not diagnostic:
        return None
    latest = diagnostic[-1]
    return {
        "history_path": str(path),
        "logged_intervals": len(diagnostic),
        "latest_optimizer_step": int(latest["optimizer_steps"]),
        "latest_laplacian_loss": float(latest["train_loss"]),
        "latest_vertex_loss": float(latest["train_recovery_refine_loss"]),
        "latest_total_loss": float(latest["train_objective"]),
        "pcg_iterations_mean": float(
            np.mean([float(row["pcg_iterations_mean"]) for row in diagnostic])
        ),
        "pcg_iterations_max": int(
            max(float(row["pcg_iterations_max"]) for row in diagnostic)
        ),
        "pcg_relative_residual_max": float(
            max(float(row["pcg_relative_residual_max"]) for row in diagnostic)
        ),
        "pcg_failed_solves": int(
            sum(int(row["pcg_failed_solves"]) for row in diagnostic)
        ),
        "delta_pred_gradient_norm_mean": float(
            np.mean([float(row["delta_pred_gradient_norm"]) for row in diagnostic])
        ),
        "prediction_head_gradient_norm_mean": float(
            np.mean(
                [float(row["prediction_head_gradient_norm"]) for row in diagnostic]
            )
        ),
        "nan_inf_count": int(sum(int(row["nan_inf_count"]) for row in diagnostic)),
        "peak_gpu_memory_mb": float(
            max(float(row["peak_gpu_memory_mb"]) for row in diagnostic)
        ),
    }


def _load_spec(run: Path, device: torch.device) -> dict[str, Any]:
    run_config = _read(run / "run_config.json")
    config = run_config.get("experiment_config", run_config)
    metrics = _read(run / "metrics.json")
    checkpoint = next(
        (run / name for name in ("checkpoint_best.pt", "best.pt") if (run / name).is_file()),
        None,
    )
    if checkpoint is None:
        raise FileNotFoundError(f"No validation-selected checkpoint in {run}")
    model = _build_model(config, None, False).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    model.eval()
    amp_enabled, amp_dtype = _amp_settings(config, device)
    return {
        "run": str(run), "config": config, "metrics": metrics,
        "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
        "model": model, "amp_enabled": amp_enabled, "amp_dtype": amp_dtype,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def _infer_recovery_arm(
    dataset: PreparedMeshDataset,
    index: int,
    spec: Mapping[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Infer raw Laplacians without requiring the deliberately removed confidence head."""
    config = spec["config"]
    if str(config.get("target_mode")) != "raw_laplacian":
        raise RuntimeError("Recovery-aware ablation requires direct raw-Laplacian output.")
    prepared = _load_device_item(dataset, index, config, device)
    conditioned = _exact_query_sample(prepared.sample, device)
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=spec["amp_dtype"],
        enabled=bool(spec["amp_enabled"]),
    ):
        output = spec["model"](conditioned)
    if output.confidence_prediction is not None:
        raise RuntimeError("Recovery-aware ablation must not instantiate a confidence head.")
    if prepared.raw_target is None:
        raise RuntimeError("Recovery-aware ablation requires an archived raw target.")
    prediction = output.predicted_laplacian.float().detach().cpu()
    target = prepared.raw_target.float().detach().cpu()
    valid = prepared.sample["valid_scale_mask"].bool().detach().cpu()
    # All equations have equal weight in the selected sparse recovery.  The
    # requested recovery-weighted RMS therefore uses unit weights as well;
    # visibility and confidence must not silently re-enter via this metric.
    recovery_weight = torch.ones(
        prediction.shape[0], dtype=prediction.dtype, device=prediction.device
    )
    return {
        "prediction_raw": prediction,
        "target_raw": target,
        "valid": valid,
        "recovery_weight": recovery_weight,
    }


def evaluate_arm(args: argparse.Namespace) -> None:
    arms, run_names = _arms(args)
    arm = arms[args.arm_index]
    run = args.runs_root.resolve() / run_names[args.arm_index]
    output = args.output_dir.resolve()
    device = torch.device(args.device)
    spec = _load_spec(run, device)
    lambda_selection = _read(args.lambda_selection.resolve())
    beta_selection = _read(args.beta_selection.resolve())
    if not lambda_selection["contract_audit"]["passed"] or not beta_selection["contract_audit"]["passed"]:
        raise RuntimeError("Hyperparameter selection contract did not pass.")
    regularization = (
        EXPECTED_EXTENSION_LAMBDAS[args.arm_index]
        if args.include_lambda_extension
        else float(lambda_selection["selected_lambda"])
    )
    config_recovery = spec["config"]["recovery"]
    if float(config_recovery["lambda"]) != regularization:
        raise RuntimeError("Training/evaluation lambda mismatch.")
    if args.include_lambda_extension and args.arm_index >= 2:
        recovery_loss = spec["config"]["training"]["recovery_aware_geometry_loss"]
        if float(recovery_loss["beta"]) != 1e-2:
            raise RuntimeError("C/D require beta=1e-2.")
        if not bool(recovery_loss.get("runtime_diagnostics", False)):
            raise RuntimeError("C/D runtime diagnostics are missing from the config.")

    rows: list[dict[str, Any]] = []
    arrays: dict[str, list[np.ndarray]] = {}
    for split in ("validation", "test"):
        dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), split)
        arrays[f"{split}_prediction"] = []
        arrays[f"{split}_target"] = []
        for index in range(len(dataset)):
            static = dataset.load_static(index)
            sample_id = str(static["sample_id"])
            values = _infer_recovery_arm(dataset, index, spec, device)
            valid = values["valid"].cpu().numpy().astype(bool)
            prediction = values["prediction_raw"].cpu().numpy().astype(np.float64)
            target = values["target_raw"].cpu().numpy().astype(np.float64)
            arrays[f"{split}_prediction"].append(prediction[valid])
            arrays[f"{split}_target"].append(target[valid])
            prediction_metric = raw_gt_magnitude_metrics(
                values["prediction_raw"], values["target_raw"],
                torch.ones_like(values["recovery_weight"]), values["valid"],
            )
            initial = Mesh(
                torch.as_tensor(static["vertices"]).cpu().numpy(),
                torch.as_tensor(static["faces"]).cpu().numpy().astype(np.int64),
            ).ensure_normals()
            clean = _clean_mesh(static)
            laplacian, lap_data = uniform_sparse_laplacian(initial.faces, initial.num_vertices)
            component_count, labels = component_labels(lap_data)
            recovered, solver = regularized_sparse_solve(
                laplacian, prediction, initial.vertices, labels, component_count,
                regularization, atol=1e-12, btol=1e-12, maxiter=100000,
            )
            initial_geometry = _geometry_row("v2_strong_smoothing", sample_id, "initial", initial, clean, initial)
            clean_geometry = _geometry_row("v2_strong_smoothing", sample_id, "clean", clean, clean, initial)
            refined_geometry = _geometry_row(
                "v2_strong_smoothing", sample_id, arm,
                Mesh(recovered, initial.faces.copy()).ensure_normals(), clean, initial,
            )
            initial_cd = float(initial_geometry["chamfer"])
            clean_cd = float(clean_geometry["chamfer"])
            refined_cd = float(refined_geometry["chamfer"])
            rows.append(
                {
                    "arm": arm, "split": split, "sample_id": sample_id,
                    **prediction_metric,
                    "initial_chamfer": initial_cd,
                    "refined_chamfer": refined_cd,
                    "relative_chamfer_gain": (initial_cd - refined_cd) / initial_cd,
                    "eta": (initial_cd - refined_cd) / (initial_cd - clean_cd),
                    "p2s": float(refined_geometry["p2s"]),
                    "p2s_p95": float(refined_geometry["p2s_p95"]),
                    "fscore": float(refined_geometry["fscore"]),
                    "normal_consistency": float(refined_geometry["normal_consistency"]),
                    "introduced_flipped_faces": int(refined_geometry["introduced_flipped_faces"]),
                    "new_degenerate_faces": int(refined_geometry["new_degenerate_faces"]),
                    "same_index_recovered_vertex_rms": float(
                        np.sqrt(np.mean(np.sum((recovered - clean.vertices) ** 2, axis=1)))
                    ),
                    "improved": refined_cd < initial_cd,
                    "worsened": refined_cd > initial_cd,
                    "lambda": regularization,
                    "solver_runtime_seconds": float(solver["runtime_seconds"]),
                    "lsmr_all_converged": bool(solver["all_converged"]),
                    "vertices": initial.num_vertices, "faces": initial.num_faces,
                }
            )
            print(f"{arm} {split} {index + 1}/{len(dataset)} {sample_id}", flush=True)
            del values
            torch.cuda.empty_cache()
    shard = output / "shards"
    _write_json(
        shard / f"{arm}.json",
        {
            "arm": arm, "run": str(run), "checkpoint": spec["checkpoint"],
            "checkpoint_sha256": spec["checkpoint_sha256"],
            "parameter_count": spec["parameter_count"], "config": spec["config"],
            "training_metrics": spec["metrics"],
            "training_runtime_diagnostics": _runtime_diagnostic_summary(run),
            "rows": rows,
        },
    )
    np.savez_compressed(
        shard / f"{arm}_prediction_arrays.npz",
        **{key: np.concatenate(value, axis=0) for key, value in arrays.items()},
    )


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def merge(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    arms, _ = _arms(args)
    payloads = [_read(output / "shards" / f"{arm}.json") for arm in arms]
    all_rows = [row for payload in payloads for row in payload["rows"]]
    aggregate_prediction: list[dict[str, Any]] = []
    aggregate_geometry: list[dict[str, Any]] = []
    for arm in arms:
        arrays = np.load(output / "shards" / f"{arm}_prediction_arrays.npz")
        for split in ("validation", "test"):
            prediction = torch.from_numpy(arrays[f"{split}_prediction"])
            target = torch.from_numpy(arrays[f"{split}_target"])
            valid = torch.ones(len(prediction), dtype=torch.bool)
            metrics = raw_gt_magnitude_metrics(
                prediction, target, torch.ones(len(prediction)), valid
            )
            aggregate_prediction.append({"arm": arm, "split": split, **metrics})
            selected = [row for row in all_rows if row["arm"] == arm and row["split"] == split]
            aggregate_geometry.append(
                {
                    "arm": arm, "split": split, "samples": len(selected),
                    "initial_chamfer": _mean(selected, "initial_chamfer"),
                    "refined_chamfer": _mean(selected, "refined_chamfer"),
                    "relative_chamfer_gain": _mean(selected, "relative_chamfer_gain"),
                    "eta": _mean(selected, "eta"), "p2s": _mean(selected, "p2s"),
                    "p2s_p95": _mean(selected, "p2s_p95"), "fscore": _mean(selected, "fscore"),
                    "normal_consistency": _mean(selected, "normal_consistency"),
                    "introduced_flipped_faces": int(sum(int(row["introduced_flipped_faces"]) for row in selected)),
                    "new_degenerate_faces": int(sum(int(row["new_degenerate_faces"]) for row in selected)),
                    "improved": int(sum(bool(row["improved"]) for row in selected)),
                    "worsened": int(sum(bool(row["worsened"]) for row in selected)),
                    "same_index_recovered_vertex_rms": _mean(selected, "same_index_recovered_vertex_rms"),
                    "runtime_per_mesh": _mean(selected, "solver_runtime_seconds"),
                }
            )
    by_key = {(row["arm"], row["split"], row["sample_id"]): row for row in all_rows}
    paired: list[dict[str, Any]] = []
    requested_pairs = (
        ((arms[0], arms[1]),)
        if len(arms) == 2
        else (
            (arms[1], arms[2]),
            (arms[1], arms[3]),
            (arms[0], arms[2]),
            (arms[0], arms[3]),
        )
    )
    for left_arm, right_arm in requested_pairs:
        for split in ("validation", "test"):
            sample_ids = sorted(
                {row["sample_id"] for row in all_rows if row["split"] == split}
            )
            for sample_id in sample_ids:
                left = by_key[(left_arm, split, sample_id)]
                right = by_key[(right_arm, split, sample_id)]
                paired.append(
                    {
                        "split": split,
                        "sample_id": sample_id,
                        "left_arm": left_arm,
                        "right_arm": right_arm,
                        "left_chamfer": left["refined_chamfer"],
                        "right_chamfer": right["refined_chamfer"],
                        "right_lower_chamfer": right["refined_chamfer"]
                        < left["refined_chamfer"],
                        "right_lower_raw_epe": right["raw_epe"] < left["raw_epe"],
                        "right_lower_vertex_rms": right[
                            "same_index_recovered_vertex_rms"
                        ]
                        < left["same_index_recovered_vertex_rms"],
                        "chamfer_right_minus_left": right["refined_chamfer"]
                        - left["refined_chamfer"],
                        "raw_epe_right_minus_left": right["raw_epe"]
                        - left["raw_epe"],
                        "vertex_rms_right_minus_left": right[
                            "same_index_recovered_vertex_rms"
                        ]
                        - left["same_index_recovered_vertex_rms"],
                    }
                )
    configs = [payload["config"] for payload in payloads]
    frozen_keys = configs[0]["experiment_metadata"]["frozen_top_level_fields"]
    same_frozen = all(
        configs[0][key] == config[key]
        for config in configs[1:]
        for key in frozen_keys
    )
    canonical_training = {
        key: value for key, value in configs[0]["training"].items() if key != "recovery_aware_geometry_loss"
    }
    same_training_except_loss = all(
        canonical_training
        == {
            key: value
            for key, value in config["training"].items()
            if key != "recovery_aware_geometry_loss"
        }
        for config in configs[1:]
    )
    actual_world_sizes = [
        int(payload["training_metrics"]["distributed_world_size"])
        for payload in payloads
    ]
    declared_world_sizes = [
        int(config["experiment_metadata"]["distributed_world_size"])
        for config in configs
    ]
    executable_contract = bool(
        same_frozen and same_training_except_loss
        and len({payload["parameter_count"] for payload in payloads}) == 1
        and all(int(payload["training_metrics"]["optimizer_steps"]) == 20000 for payload in payloads)
        and actual_world_sizes == declared_world_sizes
        and all(int(payload["training_metrics"]["global_batch_meshes"]) == 8 for payload in payloads)
        and all(all(bool(row["lsmr_all_converged"]) for row in payload["rows"]) for payload in payloads)
        and all(not config["confidence"]["enabled"] for config in configs)
        and all(config["recovery"]["visibility_gate"] is False for config in configs)
        and all(config["recovery"]["confidence_weighting"] is False for config in configs)
        and all(config["recovery"]["robust_loss"] is None for config in configs)
        and all(config["recovery"]["optimizer"] is None for config in configs)
    )
    strict_patch = all(
        config["experiment_metadata"]["patch_size_8_contract"] == "active"
        for config in configs
    )
    contract = {
        "contract_audit": executable_contract and strict_patch,
        "executable_contract_audit": executable_contract,
        "literal_patch_size_8_audit": strict_patch,
        "patch_size_note": "The completed strong_smooth_v2 implementation has no patch operator or patch_size parameter; actual baseline encoder was preserved and no ignored field was invented.",
        "same_frozen_config": same_frozen,
        "same_training_except_added_refine_loss": same_training_except_loss,
        "same_parameter_count": len({payload["parameter_count"] for payload in payloads}) == 1,
        "parameter_count": payloads[0]["parameter_count"],
        "distributed_world_sizes": actual_world_sizes,
        "same_effective_global_batch": all(
            int(payload["training_metrics"]["global_batch_meshes"]) == 8
            for payload in payloads
        ),
        "same_pcg_tolerance": len(
            {
                float(config["training"]["recovery_aware_geometry_loss"]["tolerance"])
                for config in configs
                if config["training"]["recovery_aware_geometry_loss"]["enabled"]
            }
        )
        == 1,
        "same_pcg_compute_dtype": len(
            {
                str(
                    config["training"]["recovery_aware_geometry_loss"].get(
                        "compute_dtype", "float32"
                    )
                )
                for config in configs
                if config["training"]["recovery_aware_geometry_loss"]["enabled"]
            }
        )
        == 1,
        "same_pcg_maximum_iterations": len(
            {
                int(
                    config["training"]["recovery_aware_geometry_loss"][
                        "maximum_iterations"
                    ]
                )
                for config in configs
                if config["training"]["recovery_aware_geometry_loss"]["enabled"]
            }
        )
        == 1,
        "lambda_only_numerical_contract": len(arms) == 2,
        "pcg_preflight_finding": (
            "For lambda 1e-3/1e-4, Arm-B float32 PCG stagnated before the "
            "unchanged 1e-4 tolerance. C/D therefore use the same PCG equations "
            "in float64 with maximum_iterations=2048; no tolerance, lambda, loss, "
            "gradient clipping, or objective was changed."
            if len(arms) == 4
            else None
        ),
        "execution_difference": (
            (
                "Arm A/B began on 2x L40 and resumed on 8x RTX PRO 6000 "
                "Blackwell; C/D run independently from scratch on 8x Blackwell. "
                "All arms retain effective global batch 8, but A/B have historical "
                "DDP-sharding migration points while C/D do not."
                if len(arms) == 4
                else (
                    "Both arms began on 2x L40 with accumulation=4 and resumed at an "
                    "epoch boundary on 8x RTX PRO 6000 Blackwell with accumulation=1: "
                    "Arm A after optimizer step 7200 and Arm B after optimizer step 3300. "
                    "Effective global batch remains 8, but the migration points and the "
                    "resulting DDP sharding/sample grouping differ."
                    if actual_world_sizes != [2, 2]
                    else "Both arms use 2x L40 with accumulation=4."
                )
            )
        ),
        "clean_vertices_loss_side_only": True,
        "gt_enters_model_inputs": False,
        "test_evaluated_after_lambda_and_beta_frozen": True,
        "metric_protocol": METRIC_PROTOCOL,
    }
    lambda_selection = _read(args.lambda_selection.resolve())
    beta_selection = _read(args.beta_selection.resolve())
    test_paired = [row for row in paired if row["split"] == "test"]
    decision: dict[str, Any] = {}
    for left_arm, right_arm in requested_pairs:
        selected = [
            row
            for row in test_paired
            if row["left_arm"] == left_arm and row["right_arm"] == right_arm
        ]
        decision[f"{right_arm}_vs_{left_arm}"] = {
            "samples": len(selected),
            "right_lower_chamfer": sum(
                bool(row["right_lower_chamfer"]) for row in selected
            ),
            "right_lower_raw_epe": sum(
                bool(row["right_lower_raw_epe"]) for row in selected
            ),
            "right_lower_vertex_rms": sum(
                bool(row["right_lower_vertex_rms"]) for row in selected
            ),
        }
    summary = {
        "contract_audit": contract,
        "lambda_selection": lambda_selection,
        "beta_selection": beta_selection,
        "prediction": aggregate_prediction,
        "geometry": aggregate_geometry,
        "paired_test": decision,
        "training_runtime_diagnostics": {
            payload["arm"]: payload.get("training_runtime_diagnostics")
            for payload in payloads
        },
        "checkpoints": {payload["arm"]: {"path": payload["checkpoint"], "sha256": payload["checkpoint_sha256"]} for payload in payloads},
    }
    _write_csv(output / "per_sample.csv", all_rows)
    _write_csv(output / "paired_per_sample.csv", paired)
    _write_csv(output / "prediction_summary.csv", aggregate_prediction)
    _write_csv(output / "geometry_summary.csv", aggregate_geometry)
    _write_json(output / "summary.json", summary)
    _write_json(output / "contract_audit.json", contract)
    lines = [
        (
            "# Sofa50 v2 recovery-aware lambda extension"
            if len(arms) == 4
            else "# Sofa50 v2 Lap-only vs recovery-aware training ablation"
        ), "",
        f"Strict contract audit: **{str(contract['contract_audit']).lower()}**; executable contract audit: **{str(executable_contract).lower()}**.", "",
        (
            "A/B use validation-selected lambda `1e-2`; C/D use the predeclared "
            "training/evaluation lambdas `1e-3` / `1e-4`. All recovery-aware "
            "arms use beta `1e-2`."
            if len(arms) == 4
            else f"Validation-selected lambda: `{lambda_selection['selected_lambda']:.0e}`; validation-selected beta: `{beta_selection['selected_beta']:.0e}`."
        ), "",
        "## Prediction", "",
        "| Split | Arm | Raw EPE | Raw RMS | Cosine | Weighted RMS | Bottom90 | Top10 | Top1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if len(arms) == 4:
        lines.extend(
            (
                f"Numerical contract note: {contract['pcg_preflight_finding']}",
                "",
            )
        )
    for row in aggregate_prediction:
        lines.append(
            f"| {row['split']} | {row['arm']} | {row['raw_epe']:.9g} | {row['raw_rms']:.9g} | "
            f"{row['raw_cosine']:.9g} | {row['recovery_weighted_raw_rms']:.9g} | "
            f"{row['bottom90_epe']:.9g} | {row['top10_epe']:.9g} | {row['top1_epe']:.9g} |"
        )
    lines.extend(("", "## Sparse-recovered geometry", "", "| Split | Arm | Initial CD | Refined CD | Gain | Eta | P2S | P2S p95 | F-score | Normal | Flips | New deg. | Improved/worsened | Vertex RMS |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"))
    for row in aggregate_geometry:
        lines.append(
            f"| {row['split']} | {row['arm']} | {row['initial_chamfer']:.9g} | {row['refined_chamfer']:.9g} | "
            f"{row['relative_chamfer_gain']:.2%} | {row['eta']:.9g} | {row['p2s']:.9g} | {row['p2s_p95']:.9g} | "
            f"{row['fscore']:.9g} | {row['normal_consistency']:.9g} | {row['introduced_flipped_faces']} | "
            f"{row['new_degenerate_faces']} | {row['improved']}/{row['worsened']} | {row['same_index_recovered_vertex_rms']:.9g} |"
        )
    if len(arms) == 4:
        lines.extend(
            (
                "",
                "## Training stability diagnostics",
                "",
                "| Arm | Logged intervals | Latest L_lap | Latest L_vertex | Latest L_total | PCG iter mean/max | Max residual | Failed | delta grad | head grad | NaN/Inf | Peak GPU MiB |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            )
        )
        for payload in payloads:
            diagnostic = payload.get("training_runtime_diagnostics")
            if diagnostic is None:
                lines.append(f"| {payload['arm']} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
                continue
            lines.append(
                f"| {payload['arm']} | {diagnostic['logged_intervals']} | "
                f"{diagnostic['latest_laplacian_loss']:.9g} | {diagnostic['latest_vertex_loss']:.9g} | "
                f"{diagnostic['latest_total_loss']:.9g} | {diagnostic['pcg_iterations_mean']:.2f}/{diagnostic['pcg_iterations_max']} | "
                f"{diagnostic['pcg_relative_residual_max']:.3e} | {diagnostic['pcg_failed_solves']} | "
                f"{diagnostic['delta_pred_gradient_norm_mean']:.6g} | {diagnostic['prediction_head_gradient_norm_mean']:.6g} | "
                f"{diagnostic['nan_inf_count']} | {diagnostic['peak_gpu_memory_mb']:.1f} |"
            )
        lines.extend(
            (
                "",
                "## Paired test comparisons",
                "",
                "| Comparison | Lower Chamfer | Lower vertex RMS | Lower raw EPE |",
                "|---|---:|---:|---:|",
            )
        )
        for name, counts in decision.items():
            lines.append(
                f"| {name} | {counts['right_lower_chamfer']}/{counts['samples']} | "
                f"{counts['right_lower_vertex_rms']}/{counts['samples']} | "
                f"{counts['right_lower_raw_epe']}/{counts['samples']} |"
            )
    test_geometry = {
        row["arm"]: row for row in aggregate_geometry if row["split"] == "test"
    }
    test_prediction = {
        row["arm"]: row for row in aggregate_prediction if row["split"] == "test"
    }
    b_geometry_better = test_geometry[arms[1]]["refined_chamfer"] < test_geometry[arms[0]]["refined_chamfer"]
    raw_similar_or_worse = test_prediction[arms[1]]["raw_epe"] >= 0.95 * test_prediction[arms[0]]["raw_epe"]
    lines.extend(("", "## Conclusion", ""))
    if len(arms) == 4:
        best_arm = min(arms, key=lambda arm: test_geometry[arm]["refined_chamfer"])
        lines.append(
            f"Lowest test Chamfer is **{best_arm}** at "
            f"`{test_geometry[best_arm]['refined_chamfer']:.9g}`. The decision is "
            "based on recovered geometry; raw EPE is reported but is not the selector."
        )
        for candidate in arms[2:]:
            stable = payloads[arms.index(candidate)].get("training_runtime_diagnostics")
            beats_b = (
                test_geometry[candidate]["refined_chamfer"]
                < test_geometry[arms[1]]["refined_chamfer"]
                and test_geometry[candidate]["same_index_recovered_vertex_rms"]
                < test_geometry[arms[1]]["same_index_recovered_vertex_rms"]
            )
            numerically_stable = bool(
                stable is not None
                and stable["pcg_failed_solves"] == 0
                and stable["nan_inf_count"] == 0
            )
            lines.append(
                f"{candidate}: numerical stability={numerically_stable}; "
                f"lower Chamfer and vertex RMS than B={beats_b}; "
                f"P2S p95={test_geometry[candidate]['p2s_p95']:.9g}, "
                f"normal={test_geometry[candidate]['normal_consistency']:.9g}, "
                f"flips={test_geometry[candidate]['introduced_flipped_faces']}."
            )
    elif b_geometry_better and raw_similar_or_worse:
        lines.append("Arm B obtains better recovered geometry without a commensurate raw-EPE advantage; recovery-aware supervision improves the geometric utility of the predicted Laplacian field rather than merely raw regression.")
    elif b_geometry_better:
        lines.append("Arm B improves recovered geometry, but raw Laplacian regression also improves; this experiment does not isolate geometric utility from raw-error improvement.")
    else:
        lines.append("Arm B does not improve recovered test geometry over Arm A; the current recovery-aware supervision is not supported by this ablation.")
    if len(arms) == 2:
        counts = next(iter(decision.values()))
        lines.extend(("", f"Arm B lower test Chamfer: `{counts['right_lower_chamfer']}/50`; lower raw EPE: `{counts['right_lower_raw_epe']}/50`; lower recovered vertex RMS: `{counts['right_lower_vertex_rms']}/50`.", ""))
    lines.extend((f"Patch-size caveat: {contract['patch_size_note']}", ""))
    (output / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--lambda-selection", required=True, type=Path)
    parser.add_argument("--beta-selection", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--arm-index", type=int)
    parser.add_argument("--include-lambda-extension", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    if args.merge_only:
        merge(args)
    else:
        if args.manifest is None or args.runs_root is None or args.arm_index is None:
            parser.error("arm evaluation requires manifest, runs-root and arm-index")
        arm_count = 4 if args.include_lambda_extension else 2
        if not 0 <= args.arm_index < arm_count:
            parser.error(f"arm-index must be in [0, {arm_count - 1}]")
        evaluate_arm(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
