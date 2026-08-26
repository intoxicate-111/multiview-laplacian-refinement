#!/usr/bin/env python3
from __future__ import annotations

"""Frozen selected-checkpoint analysis for the S1 split-geometry hybrid."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from analyze_sofa50_joint_gradient_interference import _flatten, _metrics, _parameter_list
from analyze_sofa50_loss_mechanisms import (
    B_MAXITER, B_TOLERANCE, _correlation_rows, _gradient_rows, _huber, _mse_vertices,
    _positional_same_state_row, _recover, _same_state_rows,
)
from diagnose_sofa50_exact_solve_visibility_sweep import component_labels, uniform_sparse_laplacian
from diagnose_sofa50_exact_target_oracle import METRIC_PROTOCOL, _clean_mesh, _geometry_row
from diagnose_sofa50_frozen_hybrid_recovery import _component_metrics, _row, _spectral_row
from diagnose_sofa50_regularized_sparse_sweep import regularized_sparse_solve
from diagnose_sofa50_representation_b_vs_e import SPECTRAL_BANDS, SPECTRAL_PROTOCOL
from mlr.data import Mesh
from mlr.learned_laplacian.canonical_experiment import _exact_query_sample, _load_device_item
from mlr.learned_laplacian.diagnostics import _amp_settings
from mlr.learned_laplacian.differentiable_sparse_recovery import (
    differentiable_regularized_sparse_recovery_with_audit,
    uniform_laplacian_transpose_apply,
)
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.multi_trainer import _build_model
from mlr.learned_laplacian.trainer import load_checkpoint


LAMBDA_LAP = 1e-2
LAMBDA_HYBRID = 3e-2
MAXITER = 2048
TOLERANCE = 1e-8
METHODS = ("S1_Lap", "S1_Direct", "S1_Hybrid")


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _lap_semantics(split: str, sample_id: str, delta: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    error = np.linalg.norm(delta - target, axis=1)
    magnitude = np.linalg.norm(target, axis=1)
    order = np.argsort(magnitude, kind="stable")
    top10 = order[-max(1, int(np.ceil(0.10 * len(order)))):]
    top1 = order[-max(1, int(np.ceil(0.01 * len(order)))):]
    denominator = max(float(np.linalg.norm(delta) * np.linalg.norm(target)), 1e-30)
    return {
        "split": split, "sample_id": sample_id, "method": "S1_Lap",
        "raw_epe": float(error.mean()),
        "raw_rms": float(np.sqrt(np.mean(np.square(error)))),
        "raw_cosine": float(np.sum(delta * target) / denominator),
        "top10_epe": float(error[top10].mean()),
        "top1_epe": float(error[top1].mean()),
    }


def _direct_semantics(split: str, sample_id: str, direct: np.ndarray, clean: np.ndarray) -> dict[str, Any]:
    error = np.linalg.norm(direct - clean, axis=1)
    return {
        "split": split, "sample_id": sample_id, "method": "S1_Direct",
        "vertex_rms": float(np.sqrt(np.mean(np.square(error)))),
        "vertex_error_mean": float(error.mean()),
        "vertex_error_p95": float(np.quantile(error, 0.95)),
    }


def _shared_gradients(
    model: torch.nn.Module,
    output: Any,
    loss: torch.Tensor,
    delta: torch.Tensor,
    displacement: torch.Tensor,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray]]:
    g_delta, g_direct = torch.autograd.grad(loss, (delta, displacement), retain_graph=True)
    groups = model.split_geometry_parameter_groups()
    shared_params = list(groups["shared_frontend"])
    lap_param = torch.autograd.grad(delta, shared_params, grad_outputs=g_delta.detach(), retain_graph=True, allow_unused=True)
    direct_param = torch.autograd.grad(displacement, shared_params, grad_outputs=g_direct.detach(), retain_graph=True, allow_unused=True)
    image_params = _parameter_list((model.image_encoder,))
    image_ids = {id(parameter) for parameter in image_params}
    image_positions = [index for index, parameter in enumerate(shared_params) if id(parameter) in image_ids]
    def subset(values: Sequence[torch.Tensor | None], positions: Sequence[int]) -> torch.Tensor:
        return _flatten([values[index] for index in positions], [shared_params[index] for index in positions])
    vectors = {
        "shared_encoder_parameters": (subset(lap_param, image_positions), subset(direct_param, image_positions)),
        "full_shared_frontend_parameters": (_flatten(lap_param, shared_params), _flatten(direct_param, shared_params)),
    }
    projected = output.aggregated_image_features
    fork = output.vertex_features
    vectors["projected_image_field"] = (
        torch.autograd.grad(delta, projected, grad_outputs=g_delta.detach(), retain_graph=True)[0],
        torch.autograd.grad(displacement, projected, grad_outputs=g_direct.detach(), retain_graph=True)[0],
    )
    vectors["shared_vertex_feature_at_fork"] = (
        torch.autograd.grad(delta, fork, grad_outputs=g_delta.detach(), retain_graph=True)[0],
        torch.autograd.grad(displacement, fork, grad_outputs=g_direct.detach(), retain_graph=False)[0],
    )
    rows = [{"layer": layer, **_metrics(*pair)} for layer, pair in vectors.items()]
    return rows, {
        "lap_output_gradient_norm": float(torch.linalg.vector_norm(g_delta).detach().cpu()),
        "direct_output_gradient_norm": float(torch.linalg.vector_norm(g_direct).detach().cpu()),
        "all_finite": bool(torch.isfinite(g_delta).all() and torch.isfinite(g_direct).all() and all(torch.isfinite(value).all() for pair in vectors.values() for value in pair)),
    }, {
        "g_S1_delta": g_delta.detach().double().cpu().numpy(),
        "g_S1_direct": g_direct.detach().double().cpu().numpy(),
    }


def shard(args: argparse.Namespace) -> None:
    dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), args.split)
    run_payload = _read(args.run.resolve() / "run_config.json")
    config = run_payload.get("experiment_config", run_payload)
    device = torch.device(args.device)
    checkpoint = args.run.resolve() / "checkpoint_best.pt"
    model = _build_model(config, None, False).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    model.eval()
    if not model.split_geometry_towers_enabled or model.direct_predictor is None:
        raise RuntimeError("Selected checkpoint is not S1 split-geometry")
    amp_enabled, amp_dtype = _amp_settings(config, device)
    geometry_rows: list[dict[str, Any]] = []
    lap_rows: list[dict[str, Any]] = []
    direct_rows: list[dict[str, Any]] = []
    spectral_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    centered_rows: list[dict[str, Any]] = []
    rhs_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    gradient_audits: list[dict[str, Any]] = []
    output_gradient_spectral_rows: list[dict[str, Any]] = []
    output_correlation_rows: list[dict[str, Any]] = []
    same_state_rows: list[dict[str, Any]] = []
    positional_same_state_rows: list[dict[str, Any]] = []
    indices = [index for index in range(len(dataset)) if index % args.shard_count == args.shard_index]
    for progress, index in enumerate(indices, 1):
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        vertices = np.asarray(static["vertices"], dtype=np.float64)
        faces = np.asarray(static["faces"], dtype=np.int64)
        initial = Mesh(vertices, faces).ensure_normals()
        clean = _clean_mesh(static)
        prepared = _load_device_item(dataset, index, config, device)
        conditioned = _exact_query_sample(prepared.sample, device)
        gradient_enabled = True
        with torch.set_grad_enabled(gradient_enabled), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            output = model(conditioned)
        direct_prediction = output.direct_vertex_displacement_prediction
        if direct_prediction is None:
            raise RuntimeError("S1 direct branch missing")
        delta_tensor = output.predicted_laplacian.float()
        displacement_tensor = direct_prediction.float()
        delta = delta_tensor.detach().double().cpu().numpy()
        direct = vertices + displacement_tensor.detach().double().cpu().numpy()
        lap, lap_data = uniform_sparse_laplacian(faces, len(vertices))
        component_count, labels = component_labels(lap_data)
        lap_vertices, lap_audit = regularized_sparse_solve(
            lap, delta, vertices, labels, component_count, LAMBDA_LAP,
            atol=1e-12, btol=1e-12, maxiter=100000,
        )
        if not lap_audit["all_converged"]:
            raise RuntimeError(f"{sample_id}: standalone Lap solve failed")
        hybrid_tensor, hybrid_audit = differentiable_regularized_sparse_recovery_with_audit(
            delta_tensor.double(),
            prepared.sample["vertices"].double() + displacement_tensor.double(),
            prepared.sample["edge_index"],
            prepared.sample["vertex_degree"].double(),
            regularization=LAMBDA_HYBRID,
            maximum_iterations=MAXITER,
            tolerance=TOLERANCE,
        )
        hybrid = hybrid_tensor.detach().cpu().numpy()
        methods = {"S1_Lap": lap_vertices, "S1_Direct": direct, "S1_Hybrid": hybrid}
        initial_metric = _geometry_row(args.split, sample_id, "initial", initial, clean, initial)
        geometry_rows.append(_row(args.split, "Initial", sample_id, index, vertices, clean, initial, initial_metric))
        for method, method_vertices in methods.items():
            metric = _geometry_row(args.split, sample_id, method, Mesh(method_vertices, faces.copy()).ensure_normals(), clean, initial)
            solve = {"pcg_iterations": int(hybrid_audit.iterations), "pcg_relative_residual": float(hybrid_audit.relative_residual), "pcg_converged": bool(hybrid_audit.converged)} if method == "S1_Hybrid" else None
            geometry_rows.append(_row(args.split, method, sample_id, index, method_vertices, clean, initial, metric, solve, LAMBDA_HYBRID if method == "S1_Hybrid" else LAMBDA_LAP if method == "S1_Lap" else None))
        delta_gt = np.asarray(lap @ clean.vertices)
        lap_rows.append(_lap_semantics(args.split, sample_id, delta, delta_gt))
        direct_rows.append(_direct_semantics(args.split, sample_id, direct, clean.vertices))
        errors = {method: values - clean.vertices for method, values in methods.items()}
        spectral_rows.extend(_spectral_row(args.split, sample_id, faces, {f"{method}_error": error for method, error in errors.items()}, args.chebyshev_order))
        displacements = {method: values - vertices for method, values in methods.items()}
        sample_components, sample_centered = _component_metrics(args.split, sample_id, labels, displacements, clean.vertices - vertices)
        component_rows.extend(sample_components)
        centered_rows.extend(sample_centered)
        e_l = uniform_laplacian_transpose_apply(
            delta_tensor.detach().double() - torch.as_tensor(delta_gt, dtype=torch.float64, device=device),
            prepared.sample["edge_index"], prepared.sample["vertex_degree"].double(),
        ).cpu().numpy()
        e_d = LAMBDA_HYBRID * (direct - clean.vertices)
        rhs_rows.append({
            "split": args.split, "sample_id": sample_id,
            "lap_rhs_norm": float(np.linalg.norm(e_l)),
            "direct_rhs_norm": float(np.linalg.norm(e_d)),
            "combined_rhs_norm": float(np.linalg.norm(e_l + e_d)),
            "rhs_cosine": float(np.sum(e_l * e_d) / max(float(np.linalg.norm(e_l) * np.linalg.norm(e_d)), 1e-30)),
            "cancellation_ratio": float(np.linalg.norm(e_l + e_d) / max(float(np.linalg.norm(e_l) + np.linalg.norm(e_d)), 1e-30)),
        })
        if gradient_enabled:
            clean_tensor = prepared.clean_vertices
            if clean_tensor is None:
                raise RuntimeError("Missing loss-side clean vertices")
            loss = (hybrid_tensor - clean_tensor.double()).square().sum(dim=-1).mean()
            sample_gradient, audit, output_gradients = _shared_gradients(model, output, loss, delta_tensor, displacement_tensor)
            for row in sample_gradient:
                row.update({"split": args.split, "sample_id": sample_id, "sample_index": index})
            audit.update({"split": args.split, "sample_id": sample_id, "sample_index": index, "loss": float(loss.detach().cpu()), "pcg_iterations": int(hybrid_audit.iterations), "pcg_relative_residual": float(hybrid_audit.relative_residual)})
            gradient_rows.extend(sample_gradient)
            gradient_audits.append(audit)
            output_gradient_spectral_rows.extend(
                _gradient_rows(
                    args.split, sample_id, faces, output_gradients, args.chebyshev_order
                )
            )
            mapped_direct = np.asarray(lap @ direct)
            path_delta = {"g_S1_delta": delta, "g_S1_direct": mapped_direct}
            final_error = np.linalg.norm(hybrid - clean.vertices, axis=1)
            same_index_errors = {
                "g_S1_delta": np.linalg.norm(lap_vertices - clean.vertices, axis=1),
                "g_S1_direct": np.linalg.norm(direct - clean.vertices, axis=1),
            }
            features = {
                "raw_laplacian_error": {
                    path: np.linalg.norm(value - delta_gt, axis=1)
                    for path, value in path_delta.items()
                },
                "same_index_vertex_error": {
                    path: same_index_errors[path] for path in output_gradients
                },
                "gt_differential_magnitude": {
                    path: np.linalg.norm(delta_gt, axis=1) for path in output_gradients
                },
                "final_recovered_geometry_error": {
                    path: final_error for path in output_gradients
                },
            }
            output_correlation_rows.extend(
                _correlation_rows(
                    args.split, sample_id, output_gradients, features
                )
            )
            state_delta = delta_tensor.detach().float().requires_grad_(True)
            target_delta = torch.as_tensor(delta_gt, dtype=torch.float32, device=device)
            direct_loss = _huber(
                state_delta, target_delta,
                torch.ones((len(vertices),), dtype=torch.float32, device=device),
            )
            state_recovered = _recover(
                state_delta, prepared.sample["vertices"].float(),
                prepared.sample["edge_index"], prepared.sample["vertex_degree"].float(),
                regularization=LAMBDA_LAP, maximum_iterations=B_MAXITER,
                tolerance=B_TOLERANCE,
            )
            recovery_loss = _mse_vertices(state_recovered, clean_tensor.float())
            direct_state_gradient = torch.autograd.grad(direct_loss, state_delta, retain_graph=True)[0]
            recovery_state_gradient = torch.autograd.grad(recovery_loss, state_delta)[0]
            same_state_rows.extend(_same_state_rows(
                args.split, sample_id, faces,
                {"S1_state": (
                    direct_state_gradient.detach().cpu().numpy(),
                    recovery_state_gradient.detach().cpu().numpy(),
                )}, args.chebyshev_order,
            ))
            positional_same_state_rows.append(_positional_same_state_row(
                args.split, sample_id, "S1", delta_tensor.detach().double(),
                prepared.sample["vertices"].double() + displacement_tensor.detach().double(),
                clean_tensor.double(), prepared.sample["edge_index"],
                prepared.sample["vertex_degree"].double(),
            ))
        print(f"S1 {args.split} shard={args.shard_index} {progress}/{len(indices)} {sample_id}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    _write_json(args.output_dir / "shards" / f"s1_{args.split}_{args.shard_index:02d}.json", {
        "read_only": True, "split": args.split, "shard_index": args.shard_index, "shard_count": args.shard_count,
        "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
        "metric_protocol": METRIC_PROTOCOL, "spectral_protocol": SPECTRAL_PROTOCOL,
        "geometry_rows": geometry_rows, "lap_semantic_rows": lap_rows, "direct_semantic_rows": direct_rows,
        "spectral_rows": spectral_rows, "component_rows": component_rows, "centered_rows": centered_rows,
        "rhs_rows": rhs_rows, "gradient_rows": gradient_rows, "gradient_audits": gradient_audits,
        "output_gradient_spectral_rows": output_gradient_spectral_rows,
        "output_correlation_rows": output_correlation_rows,
        "same_state_rows": same_state_rows,
        "positional_same_state_rows": positional_same_state_rows,
    })


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def merge(args: argparse.Namespace) -> None:
    payloads = [_read(args.output_dir / "shards" / f"s1_{split}_{index:02d}.json") for split in ("validation", "test") for index in range(args.shard_count)]
    keys = ("geometry_rows", "lap_semantic_rows", "direct_semantic_rows", "spectral_rows", "component_rows", "centered_rows", "rhs_rows", "gradient_rows", "gradient_audits", "output_gradient_spectral_rows", "output_correlation_rows", "same_state_rows", "positional_same_state_rows")
    merged = {key: [row for payload in payloads for row in payload[key]] for key in keys}
    geometry_aggregate = []
    for split in ("validation", "test"):
        for method in ("Initial", *METHODS):
            selected = [row for row in merged["geometry_rows"] if row["split"] == split and row["arm"] == method]
            geometry_aggregate.append({
                "split": split, "method": method, "samples": len(selected),
                "initial_chamfer": _mean(selected, "initial_chamfer"), "chamfer": _mean(selected, "refined_chamfer"),
                "relative_gain": _mean(selected, "relative_chamfer_gain"), "vertex_rms": _mean(selected, "same_index_recovered_vertex_rms"),
                "p2s": _mean(selected, "p2s"), "p2s_p95": _mean(selected, "p2s_p95"), "fscore": _mean(selected, "fscore"),
                "normal": _mean(selected, "normal_consistency"), "flips": int(sum(row["introduced_flipped_faces"] for row in selected)),
                "flip_rate": float(sum(row["introduced_flipped_faces"] for row in selected) / sum(row["faces"] for row in selected)),
                "new_degenerates": int(sum(row["new_degenerate_faces"] for row in selected)),
                "improved": int(sum(row["improved"] for row in selected)), "worsened": int(sum(row["worsened"] for row in selected)),
            })
    semantic_aggregate = []
    for split in ("validation", "test"):
        lap = [row for row in merged["lap_semantic_rows"] if row["split"] == split]
        direct = [row for row in merged["direct_semantic_rows"] if row["split"] == split]
        semantic_aggregate.append({"split": split, "method": "S1_Lap", **{field: _mean(lap, field) for field in ("raw_epe", "raw_rms", "raw_cosine", "top10_epe", "top1_epe")}})
        semantic_aggregate.append({"split": split, "method": "S1_Direct", **{field: _mean(direct, field) for field in ("vertex_rms", "vertex_error_mean", "vertex_error_p95")}})
    spectral_aggregate = []
    for split in ("validation", "test"):
        for method in METHODS:
            selected = [row for row in merged["spectral_rows"] if row["split"] == split and row["signal"] == f"{method}_error"]
            spectral_aggregate.append({"split": split, "method": method, "samples": len(selected), "total_energy": float(sum(row["total_energy"] for row in selected)), **{f"{band}_energy": float(sum(row[f"{band}_energy"] for row in selected)) for band in SPECTRAL_BANDS}})
    component_aggregate = []
    for split in ("validation", "test"):
        for method in METHODS:
            comp = [row for row in merged["component_rows"] if row["split"] == split and row["arm"] == method]
            centered = [row for row in merged["centered_rows"] if row["split"] == split and row["arm"] == method]
            values = np.asarray([row["component_translation_error"] for row in comp])
            component_aggregate.append({"split": split, "method": method, "components": len(values), "translation_rms": float(np.sqrt(np.mean(np.square(values)))), "translation_mean": float(values.mean()), "centered_vrms": _mean(centered, "centered_vertex_rms")})
    rhs_aggregate = []
    for split in ("validation", "test"):
        selected = [row for row in merged["rhs_rows"] if row["split"] == split]
        rhs_aggregate.append({"split": split, "samples": len(selected), **{f"mean_{field}": _mean(selected, field) for field in ("lap_rhs_norm", "direct_rhs_norm", "combined_rhs_norm", "rhs_cosine", "cancellation_ratio")}, "median_rhs_cosine": float(np.median([row["rhs_cosine"] for row in selected])), "median_cancellation_ratio": float(np.median([row["cancellation_ratio"] for row in selected])), "p10_cancellation_ratio": float(np.quantile([row["cancellation_ratio"] for row in selected], 0.1)), "p90_cancellation_ratio": float(np.quantile([row["cancellation_ratio"] for row in selected], 0.9))})
    gradient_aggregate = []
    for split in ("validation", "test"):
        for layer in sorted({row["layer"] for row in merged["gradient_rows"]}):
            selected = [row for row in merged["gradient_rows"] if row["split"] == split and row["layer"] == layer]
            gradient_aggregate.append({"split": split, "layer": layer, "samples": len(selected), "mean_cosine": _mean(selected, "cosine"), "median_cosine": float(np.median([row["cosine"] for row in selected])), "mean_lap_norm": _mean(selected, "lap_norm"), "mean_direct_norm": _mean(selected, "direct_norm"), "median_norm_ratio": float(np.median([row["magnitude_ratio"] for row in selected])), "mean_alignment_ratio": _mean(selected, "alignment_ratio")})
    output_gradient_aggregate = []
    for split in ("validation", "test"):
        for path in ("g_S1_delta", "g_S1_direct"):
            selected = [row for row in merged["output_gradient_spectral_rows"] if row["split"] == split and row["path"] == path]
            output_gradient_aggregate.append({
                "split": split, "path": path, "samples": len(selected),
                "mean_gradient_norm": _mean(selected, "gradient_norm"),
                "total_energy": _mean(selected, "total_energy"),
                **{f"{band}_energy": _mean(selected, f"{band}_energy") for band in SPECTRAL_BANDS},
                **{f"mean_{band}_fraction": _mean(selected, f"{band}_fraction") for band in SPECTRAL_BANDS},
            })
    output_correlation_aggregate = []
    for split in ("validation", "test"):
        for path in ("g_S1_delta", "g_S1_direct"):
            for feature in sorted({row["feature"] for row in merged["output_correlation_rows"]}):
                selected = [row for row in merged["output_correlation_rows"] if row["split"] == split and row["path"] == path and row["feature"] == feature]
                output_correlation_aggregate.append({
                    "split": split, "path": path, "feature": feature, "samples": len(selected),
                    "mean_pearson": _mean(selected, "pearson"), "mean_spearman": _mean(selected, "spearman"),
                    "mean_top10_gradient_energy_fraction": _mean(selected, "top10_gradient_energy_fraction"),
                    "mean_top1_gradient_energy_fraction": _mean(selected, "top1_gradient_energy_fraction"),
                    "mean_feature_all_vertices": _mean(selected, "feature_mean_all_vertices"),
                    "mean_feature_on_top10_gradient_vertices": _mean(selected, "feature_mean_on_top10_gradient_vertices"),
                    "mean_feature_on_top1_gradient_vertices": _mean(selected, "feature_mean_on_top1_gradient_vertices"),
                })
    same_state_aggregate = []
    positional_same_state_aggregate = []
    for split in ("validation", "test"):
        selected = [row for row in merged["same_state_rows"] if row["split"] == split and row["state"] == "S1_state"]
        same_state_aggregate.append({
            "split": split, "state": "S1_state", "samples": len(selected),
            "mean_cosine": _mean(selected, "cosine"),
            "median_cosine": float(np.median([row["cosine"] for row in selected])),
            "mean_direct_norm": _mean(selected, "direct_norm"),
            "mean_recovery_norm": _mean(selected, "recovery_norm"),
            "mean_norm_ratio_recovery_over_direct": _mean(selected, "norm_ratio_recovery_over_direct"),
            **{f"{band}_fraction_{kind}": _mean(selected, f"{band}_fraction_{kind}") for band in SPECTRAL_BANDS for kind in ("direct", "recovery")},
        })
        selected_position = [row for row in merged["positional_same_state_rows"] if row["split"] == split and row["state"] == "S1"]
        positional_same_state_aggregate.append({
            "split": split, "state": "S1", "samples": len(selected_position),
            **{f"mean_{field}": _mean(selected_position, field) for field in (
                "cosine", "direct_norm", "hybrid_norm", "norm_ratio_hybrid_over_direct",
                "direct_vertex_gradient_mean", "direct_vertex_gradient_median", "direct_vertex_gradient_p95",
                "hybrid_vertex_gradient_mean", "hybrid_vertex_gradient_median", "hybrid_vertex_gradient_p95",
            )},
        })
    checks = {
        "all_read_only": all(payload["read_only"] for payload in payloads),
        "all_shards": len(payloads) == 2 * args.shard_count,
        "selected_checkpoint_identity": len({payload["checkpoint_sha256"] for payload in payloads}) == 1,
        "validation_test_50": all(next(row for row in geometry_aggregate if row["split"] == split and row["method"] == "S1_Hybrid")["samples"] == 50 for split in ("validation", "test")),
        "all_gradients_finite": all(row["all_finite"] for row in merged["gradient_audits"]),
        "all_counterfactuals_finite": all(
            np.isfinite(float(row[field]))
            for rows, fields in (
                (merged["same_state_rows"], ("cosine", "direct_norm", "recovery_norm")),
                (merged["positional_same_state_rows"], ("cosine", "direct_norm", "hybrid_norm")),
            )
            for row in rows for field in fields
        ),
        "all_pcg_converged": all(row.get("pcg_converged", True) for row in merged["geometry_rows"]),
    }
    hybrid_solve_rows = [row for row in merged["geometry_rows"] if row["arm"] == "S1_Hybrid"]
    summary = {"contract_audit": all(checks.values()), "contract_checks": checks, "checkpoint": payloads[0]["checkpoint"], "checkpoint_sha256": payloads[0]["checkpoint_sha256"], "metric_protocol": METRIC_PROTOCOL, "spectral_protocol": SPECTRAL_PROTOCOL, "solver_aggregate": {"solves": len(hybrid_solve_rows), "iterations_mean": _mean(hybrid_solve_rows, "pcg_iterations"), "iterations_max": int(max(row["pcg_iterations"] for row in hybrid_solve_rows)), "relative_residual_max": float(max(row["pcg_relative_residual"] for row in hybrid_solve_rows)), "failed": int(sum(not row["pcg_converged"] for row in hybrid_solve_rows))}, "geometry_aggregate": geometry_aggregate, "semantic_aggregate": semantic_aggregate, "spectral_aggregate": spectral_aggregate, "component_aggregate": component_aggregate, "rhs_aggregate": rhs_aggregate, "gradient_aggregate": gradient_aggregate, "output_gradient_aggregate": output_gradient_aggregate, "output_correlation_aggregate": output_correlation_aggregate, "same_state_aggregate": same_state_aggregate, "positional_same_state_aggregate": positional_same_state_aggregate}
    _write_json(args.output_dir / "s1_selected_summary.json", summary)
    for name, rows in merged.items():
        _write_csv(args.output_dir / f"{name}.csv", rows)
    for name, rows in (("geometry_aggregate", geometry_aggregate), ("semantic_aggregate", semantic_aggregate), ("spectral_aggregate", spectral_aggregate), ("component_aggregate", component_aggregate), ("rhs_aggregate", rhs_aggregate), ("gradient_aggregate", gradient_aggregate), ("output_gradient_aggregate", output_gradient_aggregate), ("output_correlation_aggregate", output_correlation_aggregate), ("same_state_aggregate", same_state_aggregate), ("positional_same_state_aggregate", positional_same_state_aggregate)):
        _write_csv(args.output_dir / f"{name}.csv", rows)
    print(json.dumps({"contract_audit": summary["contract_audit"], "checkpoint_sha256": summary["checkpoint_sha256"]}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("shard", "merge"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--shard-count", type=int, default=5)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--chebyshev-order", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.phase == "shard":
        if args.manifest is None or args.run is None:
            parser.error("--manifest and --run are required for shard")
        shard(args)
    else:
        merge(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
