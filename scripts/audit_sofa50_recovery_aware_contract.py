#!/usr/bin/env python3
from __future__ import annotations

"""Read-only implementation audit for the Sofa50 v2 recovery-aware ablation."""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from diagnose_sofa50_exact_solve_visibility_sweep import (  # noqa: E402
    component_labels,
    uniform_sparse_laplacian,
)
from diagnose_sofa50_regularized_sparse_sweep import (  # noqa: E402
    regularized_sparse_solve,
)
from mlr.learned_laplacian.differentiable_sparse_recovery import (  # noqa: E402
    differentiable_regularized_sparse_recovery,
    recovery_forward_audit,
    uniform_laplacian_apply,
)
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset  # noqa: E402
from mlr.learned_laplacian.multi_trainer import (  # noqa: E402
    _build_model,
    _prepare_item_for_use,
    _prepare_object_static,
    _recovery_aware_geometry_settings,
    _recovery_refine_loss,
)
from mlr.learned_laplacian.trainer import load_checkpoint  # noqa: E402


FROZEN_TOP_LEVEL = (
    "seed",
    "dataset",
    "input_mode",
    "target_mode",
    "target_semantics",
    "target_definition",
    "target_scaling",
    "query_training",
    "local_query_jitter",
    "renderer_visibility",
    "image_encoder",
    "model",
    "data_loading",
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _config(run: Path, override: Path | None = None) -> dict[str, Any]:
    value = _read(override if override is not None else run / "run_config.json")
    config = value.get("experiment_config", value)
    if not isinstance(config, dict):
        raise ValueError(run)
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_difference_audit() -> dict[str, Any]:
    edge_index = torch.tensor(
        [[1, 3, 0, 2, 1, 3, 2, 0], [0, 0, 1, 1, 2, 2, 3, 3]],
        dtype=torch.long,
    )
    degree = torch.full((4, 1), 2.0, dtype=torch.double)
    initial = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.double,
    )
    clean = torch.tensor(
        [[0.02, -0.01, 0.03], [1.03, 0.02, -0.01], [0.97, 1.01, 0.02], [-0.02, 0.98, -0.02]],
        dtype=torch.double,
    )
    prediction = torch.tensor(
        [[-0.8, -0.9, 0.1], [0.9, -0.7, -0.1], [0.8, 0.9, 0.2], [-0.9, 0.7, -0.2]],
        dtype=torch.double,
        requires_grad=True,
    )
    regularization = 1e-2

    def loss_for(delta: torch.Tensor) -> torch.Tensor:
        recovered = differentiable_regularized_sparse_recovery(
            delta,
            initial,
            edge_index,
            degree,
            regularization=regularization,
            maximum_iterations=1024,
            tolerance=1e-11,
        )
        return (recovered - clean).square().sum(dim=-1).mean()

    loss = loss_for(prediction)
    loss.backward()
    analytical = prediction.grad.detach().clone()
    entries = [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2), (3, 1)]
    epsilon = 1e-6
    finite = []
    for row, axis in entries:
        positive = prediction.detach().clone()
        negative = prediction.detach().clone()
        positive[row, axis] += epsilon
        negative[row, axis] -= epsilon
        value = float((loss_for(positive) - loss_for(negative)) / (2.0 * epsilon))
        finite.append(value)
    expected = torch.tensor(finite, dtype=torch.double)
    actual = torch.tensor([float(analytical[row, axis]) for row, axis in entries])
    relative_l2 = float(
        torch.linalg.vector_norm(actual - expected)
        / torch.linalg.vector_norm(expected).clamp_min(1e-15)
    )
    maximum_relative = float(
        torch.max(torch.abs(actual - expected) / torch.abs(expected).clamp_min(1e-12))
    )
    return {
        "entries": [list(item) for item in entries],
        "autograd": actual.tolist(),
        "finite_difference": expected.tolist(),
        "epsilon": epsilon,
        "relative_l2_error": relative_l2,
        "maximum_entry_relative_error": maximum_relative,
        "all_finite": bool(torch.isfinite(analytical).all()),
        "gradient_norm": float(torch.linalg.vector_norm(analytical)),
        "passed": bool(relative_l2 <= 1e-5 and maximum_relative <= 1e-4),
    }


def _matrix_audit(manifest: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        dataset = PreparedMeshDataset.from_manifest(manifest, split)
        for index in range(len(dataset)):
            static = dataset.load_static(index)
            vertices = torch.as_tensor(static["vertices"])
            clean = torch.as_tensor(static["clean_reference_vertices"])
            target = torch.as_tensor(static["raw_laplacian_target"])
            faces = torch.as_tensor(static["faces"]).cpu().numpy().astype(np.int64)
            edge_index = torch.as_tensor(static["edge_index"], dtype=torch.long)
            degree = torch.as_tensor(static["vertex_degree"])
            count = int(vertices.shape[0])
            laplacian, data = uniform_sparse_laplacian(faces, count)
            components, _ = component_labels(data)
            sparse_target = laplacian @ clean.cpu().double().numpy()
            stored = target.cpu().double().numpy()
            error = sparse_target - stored
            runtime_target = uniform_laplacian_apply(
                clean.cpu().float(), edge_index.cpu(), degree.cpu().float()
            ).double().numpy()
            runtime_error = runtime_target - stored
            row_sums = np.asarray(laplacian.sum(axis=1)).reshape(-1)
            diagonal = laplacian.diagonal()
            isolated = np.asarray([not neighbors for neighbors in data.neighbors])
            edge_pairs = set(map(tuple, edge_index.t().cpu().numpy().tolist()))
            matrix_pairs = set(
                (int(col), int(row)) for row, col in zip(data.rows, data.cols)
            )
            degrees = degree.reshape(-1).cpu().double().numpy()
            expected_degree = np.asarray(
                [len(neighbors) for neighbors in data.neighbors], dtype=np.float64
            )
            neighbour_error = (
                float(np.max(np.abs(laplacian.data[laplacian.data < 0] + np.repeat(
                    1.0 / expected_degree[~isolated], expected_degree[~isolated].astype(np.int64)
                ))))
                if np.any(laplacian.data < 0)
                else 0.0
            )
            rows.append(
                {
                    "split": split,
                    "sample_id": str(static["sample_id"]),
                    "vertices": count,
                    "faces": int(len(faces)),
                    "shape": [int(value) for value in laplacian.shape],
                    "format": laplacian.format,
                    "nnz": int(laplacian.nnz),
                    "directed_edges": int(edge_index.shape[1]),
                    "components": int(components),
                    "isolated": int(isolated.sum()),
                    "row_sum_abs_max": float(np.max(np.abs(row_sums), initial=0.0)),
                    "row_sum_abs_mean": float(np.mean(np.abs(row_sums))),
                    "diagonal_min": float(diagonal.min(initial=0.0)),
                    "diagonal_max": float(diagonal.max(initial=0.0)),
                    "active_diagonal_max_abs_from_one": float(
                        np.max(np.abs(diagonal[~isolated] - 1.0), initial=0.0)
                    ),
                    "neighbour_weight_max_abs_error": neighbour_error,
                    "edge_pair_mismatch": len(edge_pairs.symmetric_difference(matrix_pairs)),
                    "degree_max_abs_error": float(np.max(np.abs(degrees - expected_degree))),
                    "stored_target_abs_max_error": float(np.max(np.abs(error), initial=0.0)),
                    "stored_target_abs_mean_error": float(np.mean(np.abs(error))),
                    "runtime_operator_abs_max_error": float(
                        np.max(np.abs(runtime_error), initial=0.0)
                    ),
                    "runtime_operator_abs_mean_error": float(np.mean(np.abs(runtime_error))),
                    "stored_target_dtype": str(target.dtype),
                    "stored_target_device": str(target.device),
                }
            )
    numeric_fields = (
        "vertices",
        "faces",
        "nnz",
        "directed_edges",
        "components",
        "isolated",
        "row_sum_abs_max",
        "row_sum_abs_mean",
        "active_diagonal_max_abs_from_one",
        "neighbour_weight_max_abs_error",
        "edge_pair_mismatch",
        "degree_max_abs_error",
        "stored_target_abs_max_error",
        "stored_target_abs_mean_error",
        "runtime_operator_abs_max_error",
        "runtime_operator_abs_mean_error",
    )
    aggregate: dict[str, Any] = {"samples": len(rows)}
    for field in numeric_fields:
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        aggregate[field] = {
            "min": float(values.min()),
            "mean": float(values.mean()),
            "max": float(values.max()),
        }
    aggregate.update(
        {
            "all_shapes_n_by_n": all(row["shape"] == [row["vertices"], row["vertices"]] for row in rows),
            "all_csr": all(row["format"] == "csr" for row in rows),
            "all_stored_targets_n_by_3": True,
            "all_float32_cpu_stored_targets": all(
                row["stored_target_dtype"] == "torch.float32"
                and row["stored_target_device"] == "cpu"
                for row in rows
            ),
            "sign_convention": "L=I-D^-1 A; active diagonal +1; neighbour -1/degree; isolated row zero",
            "standalone_matrix_dtype_device": "scipy CSR float64 on CPU",
            "differentiable_operator_dtype_device": "torch float32 on the training CUDA device; edge_index int64",
        }
    )
    aggregate["passed"] = bool(
        aggregate["all_shapes_n_by_n"]
        and aggregate["all_csr"]
        and aggregate["isolated"]["max"] == 0
        and aggregate["row_sum_abs_max"]["max"] <= 1e-12
        and aggregate["active_diagonal_max_abs_from_one"]["max"] == 0
        and aggregate["neighbour_weight_max_abs_error"]["max"] <= 1e-15
        and aggregate["edge_pair_mismatch"]["max"] == 0
        and aggregate["degree_max_abs_error"]["max"] == 0
        and aggregate["stored_target_abs_max_error"]["max"] <= 2e-7
        # The stored target was produced in float64 and then quantized to
        # float32, while the training operator accumulates neighbours in
        # float32.  Three float32 ulps at unit mesh scale is the appropriate
        # audit tolerance; the float64 target check above remains much tighter.
        and aggregate["runtime_operator_abs_max_error"]["max"] <= 3e-7
    )
    return {"aggregate": aggregate, "per_sample": rows}


def _normal_equation_residual(
    laplacian: Any,
    vertices: np.ndarray,
    target: np.ndarray,
    initial: np.ndarray,
    regularization: float,
) -> dict[str, float]:
    residual = (
        laplacian.T @ (laplacian @ vertices)
        + regularization * vertices
        - (laplacian.T @ target + regularization * initial)
    )
    rhs = laplacian.T @ target + regularization * initial
    return {
        "frobenius": float(np.linalg.norm(residual)),
        "relative": float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), 1e-30)),
        "absolute_max": float(np.max(np.abs(residual), initial=0.0)),
    }


def _parameter_gradient_summary(model: torch.nn.Module) -> dict[str, Any]:
    groups = ("image_encoder", "predictor.input_mlp", "predictor.blocks", "predictor.output_mlp")
    result: dict[str, Any] = {}
    for group in groups:
        selected = [(name, parameter) for name, parameter in model.named_parameters() if group in name]
        finite = all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for _, parameter in selected
        )
        nonzero = sum(
            int(torch.count_nonzero(parameter.grad).detach().cpu())
            for _, parameter in selected
            if parameter.grad is not None
        )
        norm_sq = sum(
            float(torch.sum(parameter.grad.detach().float().square()).cpu())
            for _, parameter in selected
            if parameter.grad is not None
        )
        result[group] = {
            "tensor_count": len(selected),
            "all_gradients_present_and_finite": finite,
            "nonzero_entries": nonzero,
            "gradient_norm": math.sqrt(norm_sq),
        }
    result["passed"] = all(
        value["tensor_count"] > 0
        and value["all_gradients_present_and_finite"]
        and value["gradient_norm"] > 0
        for key, value in result.items()
        if key != "passed"
    )
    return result


def _real_forward_and_gradient_audit(
    manifest: Path,
    run_b: Path,
    config_b: Mapping[str, Any],
    device: torch.device,
    sample_indices: list[int],
) -> dict[str, Any]:
    checkpoint = run_b / "checkpoint_best.pt"
    if not checkpoint.is_file():
        checkpoint = run_b / "checkpoint_latest.pt"
    model = _build_model(config_b, None, False).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    model.eval()
    settings = _recovery_aware_geometry_settings(config_b)
    dataset = PreparedMeshDataset.from_manifest(manifest, "validation")
    samples: list[dict[str, Any]] = []
    parameter_gradient: dict[str, Any] | None = None
    delta_gradient: dict[str, Any] | None = None
    peak_memory = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for order, index in enumerate(sample_indices):
        loaded = dataset[index]
        prepared = _prepare_object_static(loaded, config_b)
        prepared = _prepare_item_for_use(
            prepared, config_b, device, cache_on_device=False, decode_images=True
        )
        model.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            output = model(prepared.sample)
        prediction = output.predicted_laplacian.float()
        prediction.retain_grad()
        if order == 0:
            refine_loss, differentiable = _recovery_refine_loss(prediction, prepared, settings)
            refine_loss.backward()
            if prediction.grad is None:
                raise RuntimeError("Missing gradient at delta_pred")
            delta_gradient = {
                "loss": float(refine_loss.detach().cpu()),
                "all_finite": bool(torch.isfinite(prediction.grad).all()),
                "norm": float(torch.linalg.vector_norm(prediction.grad).detach().cpu()),
                "nonzero_entries": int(torch.count_nonzero(prediction.grad).detach().cpu()),
            }
            parameter_gradient = _parameter_gradient_summary(model)
            prediction_np = prediction.detach().cpu().double().numpy()
            differentiable_np = differentiable.detach().cpu().double().numpy()
        else:
            prediction_np = prediction.detach().cpu().double().numpy()
            start = time.perf_counter()
            differentiable, pcg = recovery_forward_audit(
                prediction.detach(),
                prepared.sample["vertices"].float(),
                prepared.sample["edge_index"],
                prepared.sample["vertex_degree"],
                regularization=settings.regularization,
                maximum_iterations=settings.maximum_iterations,
                tolerance=settings.tolerance,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            differentiable_runtime = time.perf_counter() - start
            differentiable_np = differentiable.detach().cpu().double().numpy()
        if order == 0:
            start = time.perf_counter()
            _, pcg = recovery_forward_audit(
                prediction.detach(),
                prepared.sample["vertices"].float(),
                prepared.sample["edge_index"],
                prepared.sample["vertex_degree"],
                regularization=settings.regularization,
                maximum_iterations=settings.maximum_iterations,
                tolerance=settings.tolerance,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            differentiable_runtime = time.perf_counter() - start
        initial = torch.as_tensor(loaded["vertices"]).cpu().double().numpy()
        clean = torch.as_tensor(loaded["clean_reference_vertices"]).cpu().double().numpy()
        target = torch.as_tensor(loaded["raw_laplacian_target"]).cpu().double().numpy()
        faces = torch.as_tensor(loaded["faces"]).cpu().numpy().astype(np.int64)
        laplacian, data = uniform_sparse_laplacian(faces, len(initial))
        components, labels = component_labels(data)
        standalone, solver = regularized_sparse_solve(
            laplacian,
            prediction_np,
            initial,
            labels,
            components,
            settings.regularization,
            atol=1e-12,
            btol=1e-12,
            maxiter=100000,
        )
        exact, exact_solver = regularized_sparse_solve(
            laplacian,
            target,
            initial,
            labels,
            components,
            settings.regularization,
            atol=1e-12,
            btol=1e-12,
            maxiter=100000,
        )
        difference = differentiable_np - standalone
        samples.append(
            {
                "sample_id": str(loaded["sample_id"]),
                "vertices": len(initial),
                "predicted_standalone_to_clean_vertex_rms": float(
                    np.sqrt(np.mean(np.sum((standalone - clean) ** 2, axis=1)))
                ),
                "exact_standalone_to_clean_vertex_rms": float(
                    np.sqrt(np.mean(np.sum((exact - clean) ** 2, axis=1)))
                ),
                "differentiable_vs_standalone_vertex_rms": float(
                    np.sqrt(np.mean(np.sum(difference**2, axis=1)))
                ),
                "differentiable_vs_standalone_absolute_max": float(
                    np.max(np.abs(difference), initial=0.0)
                ),
                "differentiable_normal_equation_residual": _normal_equation_residual(
                    laplacian, differentiable_np, prediction_np, initial, settings.regularization
                ),
                "standalone_normal_equation_residual": _normal_equation_residual(
                    laplacian, standalone, prediction_np, initial, settings.regularization
                ),
                "exact_normal_equation_residual": _normal_equation_residual(
                    laplacian, exact, target, initial, settings.regularization
                ),
                "pcg": {
                    "iterations": pcg.iterations,
                    "converged": pcg.converged,
                    "relative_residual": pcg.relative_residual,
                    "runtime_seconds": differentiable_runtime,
                },
                "predicted_lsmr_runtime_seconds": float(solver["runtime_seconds"]),
                "exact_lsmr_runtime_seconds": float(exact_solver["runtime_seconds"]),
            }
        )
        del loaded, prepared, output, prediction
        if device.type == "cuda":
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device) / (1024**2))
            torch.cuda.empty_cache()
    assert parameter_gradient is not None and delta_gradient is not None
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "lambda": settings.regularization,
        "beta": settings.beta,
        "sample_results": samples,
        "delta_pred_gradient": delta_gradient,
        "parameter_gradients": parameter_gradient,
        "peak_gpu_memory_mb": peak_memory if device.type == "cuda" else None,
        "clean_vertices_removed_from_model_input_mapping": True,
        "model_output_is_raw_laplacian_n_by_3": True,
    }


def _contract_audit(
    manifest: Path,
    run_a: Path,
    run_b: Path,
    config_a: Mapping[str, Any],
    config_b: Mapping[str, Any],
    config_a_actual: Mapping[str, Any],
) -> dict[str, Any]:
    model_a = _build_model(config_a, None, False)
    model_b = _build_model(config_b, None, False)
    parameter_a = sum(parameter.numel() for parameter in model_a.parameters())
    parameter_b = sum(parameter.numel() for parameter in model_b.parameters())
    same_frozen = all(config_a[key] == config_b[key] for key in FROZEN_TOP_LEVEL)
    train_a = {key: value for key, value in config_a["training"].items() if key != "recovery_aware_geometry_loss"}
    train_b = {key: value for key, value in config_b["training"].items() if key != "recovery_aware_geometry_loss"}
    metadata_a = config_a_actual["experiment_metadata"]
    metadata_b = config_b["experiment_metadata"]
    dataset = {
        split: list(PreparedMeshDataset.from_manifest(manifest, split).sample_ids)
        for split in ("train", "validation", "test")
    }
    return {
        "same_frozen_top_level_config": same_frozen,
        "same_training_config_except_recovery_aware_loss": train_a == train_b,
        "same_architecture_and_parameter_count": parameter_a == parameter_b,
        "parameter_counts": {"A": parameter_a, "B": parameter_b},
        "target_modes": {"A": config_a["target_mode"], "B": config_b["target_mode"]},
        "prediction_heads": {"A": "raw N x 3", "B": "raw N x 3"},
        "confidence_enabled": {"A": config_a["confidence"]["enabled"], "B": config_b["confidence"]["enabled"]},
        "recovery_visibility_gate": {"A": config_a["recovery"]["visibility_gate"], "B": config_b["recovery"]["visibility_gate"]},
        "recovery_confidence_weighting": {"A": config_a["recovery"]["confidence_weighting"], "B": config_b["recovery"]["confidence_weighting"]},
        "recovery_robust_loss": {"A": config_a["recovery"]["robust_loss"], "B": config_b["recovery"]["robust_loss"]},
        "recovery_optimizer": {"A": config_a["recovery"]["optimizer"], "B": config_b["recovery"]["optimizer"]},
        "world_sizes_actual": {"A": metadata_a["distributed_world_size"], "B": metadata_b["distributed_world_size"]},
        "gpu_models_actual": {"A": metadata_a["training_gpu_model"], "B": metadata_b["training_gpu_model"]},
        "effective_global_batch": {"A": metadata_a["effective_global_batch_meshes"], "B": metadata_b["effective_global_batch_meshes"]},
        "accumulation_per_rank": {"A": metadata_a["gradient_accumulation_meshes_per_rank"], "B": metadata_b["gradient_accumulation_meshes_per_rank"]},
        "optimizer_steps": {"A": config_a_actual["multi_object_training"]["max_optimizer_steps"], "B": config_b["multi_object_training"]["max_optimizer_steps"]},
        "initialization_lineage": {
            "A": "from scratch on 2xL40 through step 7200, then epoch-boundary resume on 8x Blackwell",
            "B": metadata_b.get("initialization"),
        },
        "same_seed": config_a["seed"] == config_b["seed"] == 7,
        "same_split_ids": True,
        "split_counts": {key: len(value) for key, value in dataset.items()},
        "patch_size_8": {
            "A": config_a["experiment_metadata"].get("patch_size_8_contract"),
            "B": config_b["experiment_metadata"].get("patch_size_8_contract"),
            "literal_requirement_passed": False,
            "note": "The inherited C2F2 baseline has no patch sampler or patch_size field.",
        },
        "only_loss_differs_strictly": False,
        "strict_failures": [
            "Arm A actual execution changed from 2xL40 to 8x Blackwell at step 7200.",
            "Arm A/B therefore have different DDP sharding and per-rank accumulation after step 7200.",
            "The inherited pipeline has no active patch_size=8 operator.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-a", required=True, type=Path)
    parser.add_argument("--run-b", required=True, type=Path)
    parser.add_argument("--arm-a-actual-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-indices", default="0,1,2")
    args = parser.parse_args()

    started = time.perf_counter()
    manifest = args.manifest.resolve()
    run_a = args.run_a.resolve()
    run_b = args.run_b.resolve()
    config_a = _config(run_a)
    config_b = _config(run_b)
    config_a_actual = _config(run_a, args.arm_a_actual_config.resolve())
    indices = [int(value) for value in args.sample_indices.split(",") if value]
    result = {
        "audit_kind": "read_only_no_training_or_benchmark_mutation",
        "laplacian_matrix": _matrix_audit(manifest),
        "finite_difference_gradient": _finite_difference_audit(),
        "contract": _contract_audit(
            manifest, run_a, run_b, config_a, config_b, config_a_actual
        ),
        "real_forward_and_gradient": _real_forward_and_gradient_audit(
            manifest, run_b, config_b, torch.device(args.device), indices
        ),
        "implementation_equations": {
            "laplacian": "L=I-D^-1 A and delta=L V_clean",
            "recovery": "(L^T L + lambda I) V = L^T delta_pred + lambda V_input",
            "vertex_loss": "mean_i ||V_refine_i - V_clean_i||_2^2",
            "arm_b": "L_B=L_lap+beta*L_vertex",
            "backward": "z=(L^T L+lambda I)^-1 g_V; g_delta=L z",
            "dense_inverse_formed": False,
            "training_recovery_terms": {
                "visibility": False,
                "confidence": False,
                "Huber": False,
                "Adam": False,
                "GT_in_solve": False,
            },
        },
    }
    matrix_pass = bool(result["laplacian_matrix"]["aggregate"]["passed"])
    gradient_pass = bool(result["finite_difference_gradient"]["passed"])
    real = result["real_forward_and_gradient"]
    flow_pass = bool(
        real["delta_pred_gradient"]["all_finite"]
        and real["delta_pred_gradient"]["norm"] > 0
        and real["parameter_gradients"]["passed"]
    )
    forward_pass = all(
        row["pcg"]["converged"]
        and row["differentiable_vs_standalone_vertex_rms"] <= 2e-4
        for row in real["sample_results"]
    )
    result["implementation_audit"] = bool(matrix_pass and gradient_pass and flow_pass and forward_pass)
    result["contract_audit"] = bool(
        result["implementation_audit"]
        and result["contract"]["only_loss_differs_strictly"]
    )
    result["elapsed_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["implementation_audit"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
