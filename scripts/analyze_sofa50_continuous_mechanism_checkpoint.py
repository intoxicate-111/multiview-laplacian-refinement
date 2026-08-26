#!/usr/bin/env python3
from __future__ import annotations

"""Read-only branch/mechanism diagnostics for one continuous B+E checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from analyze_sofa50_frozen_vs_joint_mechanisms import (
    LAMBDA_B,
    _lap_semantic_row,
    _latent_row,
    _pcg,
    _position_semantic_row,
)
from diagnose_sofa50_exact_solve_visibility_sweep import (
    component_labels,
    uniform_sparse_laplacian,
)
from diagnose_sofa50_exact_target_oracle import _clean_mesh, _geometry_row
from diagnose_sofa50_frozen_hybrid_recovery import _component_metrics, _spectral_row
from diagnose_sofa50_regularized_sparse_sweep import regularized_sparse_solve
from diagnose_sofa50_representation_b_vs_e import SPECTRAL_BANDS, SPECTRAL_PROTOCOL
from mlr.data import Mesh
from mlr.learned_laplacian.canonical_experiment import _exact_query_sample, _load_device_item
from mlr.learned_laplacian.diagnostics import _amp_settings
from mlr.learned_laplacian.differentiable_sparse_recovery import (
    differentiable_regularized_sparse_recovery_with_audit,
)
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.multi_trainer import _build_model
from mlr.learned_laplacian.trainer import load_checkpoint
from mlr.learned_laplacian.two_branch_hybrid import TwoBranchPretrainedHybridModel


METHODS = ("Current_B", "Current_E", "Current_Hybrid")
LAMBDA_H = 3e-2
TOLERANCE = 1e-8
MAXIMUM_ITERATIONS = 2048


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean(rows: Iterable[Mapping[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in rows]
    return float(np.mean(values)) if values else float("nan")


def _parameter_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = sum(float(parameter.detach().double().square().sum().cpu()) for parameter in parameters)
    return float(np.sqrt(total))


def _parameter_drift(
    current: TwoBranchPretrainedHybridModel,
    initial: TwoBranchPretrainedHybridModel,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for branch in ("arm_b", "arm_e"):
        current_state = getattr(current, branch).state_dict()
        initial_state = getattr(initial, branch).state_dict()
        squared_difference = 0.0
        squared_initial = 0.0
        maximum = 0.0
        count = 0
        for name, value in current_state.items():
            if not value.is_floating_point():
                continue
            reference = initial_state[name]
            difference = value.detach().double().cpu() - reference.detach().double().cpu()
            squared_difference += float(difference.square().sum())
            squared_initial += float(reference.detach().double().cpu().square().sum())
            maximum = max(maximum, float(difference.abs().max()))
            count += difference.numel()
        norm = float(np.sqrt(squared_difference))
        result[branch] = {
            "parameters": count,
            "l2_drift": norm,
            "relative_l2_drift": norm / max(float(np.sqrt(squared_initial)), 1e-30),
            "rms_drift": float(np.sqrt(squared_difference / max(count, 1))),
            "maximum_coordinate_drift": maximum,
        }
    return result


def _gradient_row(
    model: TwoBranchPretrainedHybridModel,
    sample: Mapping[str, Any],
    clean_vertices: torch.Tensor,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
        output = model(sample)
    direct = output.direct_vertex_displacement_prediction
    if direct is None:
        raise RuntimeError("Continuous checkpoint omitted Arm E output")
    delta = output.predicted_laplacian.float()
    displacement = direct.float()
    recovered, audit = differentiable_regularized_sparse_recovery_with_audit(
        delta.double(),
        sample["vertices"].double() + displacement.double(),
        sample["edge_index"],
        sample["vertex_degree"].double(),
        regularization=LAMBDA_H,
        maximum_iterations=MAXIMUM_ITERATIONS,
        tolerance=TOLERANCE,
    )
    if not audit.converged:
        raise RuntimeError(f"gradient PCG failed: {audit}")
    loss = (recovered - clean_vertices.double()).square().sum(dim=-1).mean()
    latent_grads = torch.autograd.grad(
        loss, (delta, displacement), retain_graph=True, allow_unused=False
    )
    groups = model.branch_parameter_groups()
    group_names = tuple(groups)
    parameters = tuple(parameter for name in group_names for parameter in groups[name])
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=False, allow_unused=True
    )
    output_row: dict[str, Any] = {
        "loss": float(loss.detach().cpu()),
        "delta_gradient_norm": float(torch.linalg.vector_norm(latent_grads[0].detach().double()).cpu()),
        "direct_gradient_norm": float(torch.linalg.vector_norm(latent_grads[1].detach().double()).cpu()),
        "pcg_iterations": int(audit.iterations),
        "pcg_relative_residual": float(audit.relative_residual),
    }
    cursor = 0
    for name in group_names:
        group_gradients = gradients[cursor : cursor + len(groups[name])]
        cursor += len(groups[name])
        squared = sum(
            0.0 if gradient is None else float(gradient.detach().double().square().sum().cpu())
            for gradient in group_gradients
        )
        output_row[f"{name}_gradient_norm"] = float(np.sqrt(squared))
    output_row["b_total_gradient_norm"] = float(
        np.hypot(output_row["b_head_gradient_norm"], output_row["b_backbone_gradient_norm"])
    )
    output_row["e_total_gradient_norm"] = float(
        np.hypot(output_row["e_head_gradient_norm"], output_row["e_backbone_gradient_norm"])
    )
    output_row["b_to_e_gradient_norm_ratio"] = output_row["b_total_gradient_norm"] / max(
        output_row["e_total_gradient_norm"], 1e-30
    )
    return output_row


def _geometry(
    split: str,
    sample_id: str,
    index: int,
    method: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    clean: Mesh,
    initial: Mesh,
) -> dict[str, Any]:
    metric = _geometry_row(
        split,
        sample_id,
        method,
        Mesh(vertices, faces.copy()).ensure_normals(),
        clean,
        initial,
    )
    initial_metric = _geometry_row(split, sample_id, "initial", initial, clean, initial)
    initial_cd = float(initial_metric["chamfer"])
    refined_cd = float(metric["chamfer"])
    return {
        "split": split,
        "sample_id": sample_id,
        "sample_index": index,
        "method": method,
        "initial_chamfer": initial_cd,
        "chamfer": refined_cd,
        "relative_gain": (initial_cd - refined_cd) / initial_cd,
        "p2s": float(metric["p2s"]),
        "p2s_p95": float(metric["p2s_p95"]),
        "fscore": float(metric["fscore"]),
        "normal": float(metric["normal_consistency"]),
        "flips": int(metric["introduced_flipped_faces"]),
        "faces": int(len(faces)),
        "new_degenerates": int(metric["new_degenerate_faces"]),
        "vertex_rms": float(np.sqrt(np.mean(np.sum((vertices - clean.vertices) ** 2, axis=1)))),
        "improved": refined_cd < initial_cd,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--step0-checkpoint", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--indices", default="0,5,10,15,20,25,30,35,40,45")
    parser.add_argument("--chebyshev-order", type=int, default=128)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_payload = _read(args.run.resolve() / "run_config.json")
    config = run_payload.get("experiment_config", run_payload)
    device = torch.device(args.device)
    current_model = _build_model(config, None, False).to(device)
    initial_model = _build_model(config, None, False).to(device)
    if not isinstance(current_model, TwoBranchPretrainedHybridModel) or not isinstance(
        initial_model, TwoBranchPretrainedHybridModel
    ):
        raise RuntimeError("Expected two complete independent B/E networks")
    load_checkpoint(args.checkpoint.resolve(), current_model, map_location=device)
    load_checkpoint(args.step0_checkpoint.resolve(), initial_model, map_location=device)
    current_model.eval()
    initial_model.eval()
    parameter_drift = _parameter_drift(current_model, initial_model)
    dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), args.split)
    indices = list(range(len(dataset))) if args.indices == "all" else [
        int(value) for value in args.indices.split(",") if value.strip()
    ]
    if any(index < 0 or index >= len(dataset) for index in indices):
        raise ValueError(f"indices outside {args.split} dataset of size {len(dataset)}")
    amp_enabled, amp_dtype = _amp_settings(config, device)

    geometry_rows: list[dict[str, Any]] = []
    latent_rows: list[dict[str, Any]] = []
    lap_rows: list[dict[str, Any]] = []
    direct_rows: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    spectral_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    centered_rows: list[dict[str, Any]] = []

    for progress, index in enumerate(indices, start=1):
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        vertices = np.asarray(static["vertices"], dtype=np.float64)
        faces = np.asarray(static["faces"], dtype=np.int64)
        initial = Mesh(vertices, faces).ensure_normals()
        clean = _clean_mesh(static)
        prepared = _load_device_item(dataset, index, config, device)
        conditioned = _exact_query_sample(prepared.sample, device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            current_output = current_model(conditioned)
            initial_output = initial_model(conditioned)
        current_direct_prediction = current_output.direct_vertex_displacement_prediction
        initial_direct_prediction = initial_output.direct_vertex_displacement_prediction
        if current_direct_prediction is None or initial_direct_prediction is None:
            raise RuntimeError("Continuous B/E checkpoint omitted Arm E output")
        delta = current_output.predicted_laplacian.detach().double().cpu().numpy()
        displacement = current_direct_prediction.detach().double().cpu().numpy()
        delta0 = initial_output.predicted_laplacian.detach().double().cpu().numpy()
        displacement0 = initial_direct_prediction.detach().double().cpu().numpy()
        direct = vertices + displacement

        laplacian, lap_data = uniform_sparse_laplacian(faces, len(vertices))
        component_count, labels = component_labels(lap_data)
        b_vertices, b_audit = regularized_sparse_solve(
            laplacian,
            delta,
            vertices,
            labels,
            component_count,
            LAMBDA_B,
            atol=1e-12,
            btol=1e-12,
            maxiter=100000,
        )
        if not b_audit["all_converged"]:
            raise RuntimeError(f"{sample_id}: standalone Arm B solve failed")
        hybrid, hybrid_audit = _pcg(delta, direct, static, device)
        methods = {
            "Current_B": b_vertices,
            "Current_E": direct,
            "Current_Hybrid": hybrid,
        }
        for method, method_vertices in methods.items():
            geometry_rows.append(
                _geometry(args.split, sample_id, index, method, method_vertices, faces, clean, initial)
            )
        delta_gt = laplacian @ clean.vertices
        latent_rows.append(_latent_row(args.split, sample_id, "B_Direct", delta, laplacian @ direct))
        lap_rows.append(_lap_semantic_row(args.split, sample_id, "Current_B", delta, delta_gt))
        direct_rows.append(
            _position_semantic_row(args.split, sample_id, "Current_E", direct, clean.vertices)
        )
        drift_rows.append(
            {
                "sample_id": sample_id,
                "sample_index": index,
                "delta_b_rms_drift": float(np.sqrt(np.mean(np.square(delta - delta0)))),
                "delta_b_max_drift": float(np.max(np.abs(delta - delta0))),
                "v_direct_rms_drift": float(np.sqrt(np.mean(np.square(displacement - displacement0)))),
                "v_direct_max_drift": float(np.max(np.abs(displacement - displacement0))),
            }
        )
        errors = {method: values - clean.vertices for method, values in methods.items()}
        spectral_rows.extend(
            _spectral_row(
                args.split,
                sample_id,
                faces,
                {
                    **{f"{method}_error": error for method, error in errors.items()},
                    "representation_difference": delta - laplacian @ direct,
                },
                args.chebyshev_order,
            )
        )
        sample_components, sample_centered = _component_metrics(
            args.split,
            sample_id,
            labels,
            {method: values - vertices for method, values in methods.items()},
            clean.vertices - vertices,
        )
        component_rows.extend(sample_components)
        centered_rows.extend(sample_centered)
        clean_tensor = prepared.clean_vertices
        if clean_tensor is None:
            raise RuntimeError(f"{sample_id}: missing clean vertices")
        gradient = _gradient_row(
            current_model,
            conditioned,
            clean_tensor,
            amp_enabled,
            amp_dtype,
            device,
        )
        gradient_rows.append({"sample_id": sample_id, "sample_index": index, **gradient})
        print(
            f"continuous mechanism {args.label} {args.split} {progress}/{len(indices)} {sample_id}",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    geometry_aggregate = []
    for method in METHODS:
        selected = [row for row in geometry_rows if row["method"] == method]
        geometry_aggregate.append(
            {
                "method": method,
                "samples": len(selected),
                "chamfer": _mean(selected, "chamfer"),
                "relative_gain": _mean(selected, "relative_gain"),
                "vertex_rms": _mean(selected, "vertex_rms"),
                "p2s": _mean(selected, "p2s"),
                "p2s_p95": _mean(selected, "p2s_p95"),
                "fscore": _mean(selected, "fscore"),
                "normal": _mean(selected, "normal"),
                "flips": int(sum(int(row["flips"]) for row in selected)),
                "flip_rate": float(sum(int(row["flips"]) for row in selected) / sum(int(row["faces"]) for row in selected)),
                "new_degenerates": int(sum(int(row["new_degenerates"]) for row in selected)),
                "improved": int(sum(bool(row["improved"]) for row in selected)),
                "worsened": int(sum(not bool(row["improved"]) for row in selected)),
            }
        )
    spectral_aggregate = []
    for signal in sorted({str(row["signal"]) for row in spectral_rows}):
        selected = [row for row in spectral_rows if row["signal"] == signal]
        item: dict[str, Any] = {
            "signal": signal,
            "samples": len(selected),
            "total_energy": float(sum(float(row["total_energy"]) for row in selected)),
        }
        for band in SPECTRAL_BANDS:
            item[f"{band}_energy"] = float(sum(float(row[f"{band}_energy"]) for row in selected))
        spectral_aggregate.append(item)
    component_aggregate = []
    for method in METHODS:
        selected_components = [row for row in component_rows if row["arm"] == method]
        selected_centered = [row for row in centered_rows if row["arm"] == method]
        values = np.asarray(
            [float(row["component_translation_error"]) for row in selected_components],
            dtype=np.float64,
        )
        component_aggregate.append(
            {
                "method": method,
                "components": len(values),
                "component_translation_rms": float(np.sqrt(np.mean(np.square(values)))),
                "component_translation_mean": float(values.mean()),
                "centered_vertex_rms": _mean(selected_centered, "centered_vertex_rms"),
            }
        )
    gradient_fields = (
        "delta_gradient_norm",
        "direct_gradient_norm",
        "b_head_gradient_norm",
        "b_backbone_gradient_norm",
        "e_head_gradient_norm",
        "e_backbone_gradient_norm",
        "b_total_gradient_norm",
        "e_total_gradient_norm",
        "b_to_e_gradient_norm_ratio",
    )
    payload = {
        "read_only": True,
        "label": args.label,
        "split": args.split,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint.resolve()),
        "step0_checkpoint": str(args.step0_checkpoint.resolve()),
        "step0_checkpoint_sha256": _sha256(args.step0_checkpoint.resolve()),
        "samples": len(indices),
        "indices": indices,
        "two_complete_independent_networks": True,
        "parameter_count": sum(parameter.numel() for parameter in current_model.parameters()),
        "shared_parameter_storage": False,
        "parameter_drift": parameter_drift,
        "output_drift": {
            field: _mean(drift_rows, field)
            for field in (
                "delta_b_rms_drift",
                "delta_b_max_drift",
                "v_direct_rms_drift",
                "v_direct_max_drift",
            )
        },
        "gradient_aggregate": {field: _mean(gradient_rows, field) for field in gradient_fields},
        "gradient_all_finite_nonzero": all(
            np.isfinite(float(row[field])) and float(row[field]) > 0
            for row in gradient_rows
            for field in gradient_fields[:-1]
        ),
        "gradient_pcg_maximum_residual": max(float(row["pcg_relative_residual"]) for row in gradient_rows),
        "latent_aggregate": {
            field: _mean(latent_rows, field)
            for field in ("redundancy_rms", "relative_discrepancy", "cosine", "norm_ratio")
        },
        "lap_semantic_aggregate": {
            field: _mean(lap_rows, field)
            for field in ("raw_epe", "raw_rms", "raw_cosine", "top10_epe", "top1_epe")
        },
        "direct_semantic_aggregate": {
            field: _mean(direct_rows, field)
            for field in ("vertex_rms", "vertex_error_mean", "vertex_error_p95")
        },
        "geometry_aggregate": geometry_aggregate,
        "spectral_protocol": SPECTRAL_PROTOCOL,
        "spectral_aggregate": spectral_aggregate,
        "component_aggregate": component_aggregate,
        "hybrid_solver": {
            "lambda": LAMBDA_H,
            "tolerance": TOLERANCE,
            "maximum_iterations": MAXIMUM_ITERATIONS,
            "iterations_mean": _mean(gradient_rows, "pcg_iterations"),
            "relative_residual_max": max(float(row["pcg_relative_residual"]) for row in gradient_rows),
            "all_converged": all(float(row["pcg_relative_residual"]) <= 1.05e-8 for row in gradient_rows),
        },
        "geometry_rows": geometry_rows,
        "latent_rows": latent_rows,
        "lap_semantic_rows": lap_rows,
        "direct_semantic_rows": direct_rows,
        "drift_rows": drift_rows,
        "gradient_rows": gradient_rows,
        "spectral_rows": spectral_rows,
        "component_rows": component_rows,
        "centered_rows": centered_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "samples": len(indices)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
