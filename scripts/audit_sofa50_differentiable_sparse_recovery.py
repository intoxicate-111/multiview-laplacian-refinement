#!/usr/bin/env python3
from __future__ import annotations

"""Audit the differentiable recovery solve on real prepared Sofa50 meshes."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from diagnose_sofa50_exact_solve_visibility_sweep import (
    component_labels,
    uniform_sparse_laplacian,
)
from diagnose_sofa50_regularized_sparse_sweep import regularized_sparse_solve
from mlr.learned_laplacian.differentiable_sparse_recovery import (
    differentiable_regularized_sparse_recovery,
    recovery_forward_audit,
    uniform_laplacian_apply,
)
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--lambda-value", type=float, default=1e-2)
    parser.add_argument("--maximum-iterations", type=int, default=256)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--pcg-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), args.split)
    static = dataset.load_static(args.sample_index)
    initial = torch.as_tensor(static["vertices"], dtype=torch.float32)
    target = torch.as_tensor(static["raw_laplacian_target"], dtype=torch.float32)
    clean = torch.as_tensor(static["clean_reference_vertices"], dtype=torch.float32)
    edge_index = torch.as_tensor(static["edge_index"], dtype=torch.long)
    degree = torch.as_tensor(static["vertex_degree"], dtype=torch.float32)
    faces = torch.as_tensor(static["faces"], dtype=torch.long)

    # A deterministic, prediction-like perturbation makes the gradient audit
    # nontrivial without loading any checkpoint or test data.
    generator = torch.Generator().manual_seed(7)
    scale = max(float(torch.linalg.vector_norm(target, dim=1).mean()), 1e-6)
    prediction = target + 0.05 * scale * torch.randn(
        target.shape, generator=generator, dtype=target.dtype
    )

    device = torch.device(args.device)
    solve_dtype = torch.float64 if args.pcg_dtype == "float64" else torch.float32
    prediction_device = prediction.to(device=device, dtype=solve_dtype).requires_grad_(True)
    initial_device = initial.to(device=device, dtype=solve_dtype)
    edge_device = edge_index.to(device)
    degree_device = degree.to(device=device, dtype=solve_dtype)
    clean_device = clean.to(device=device, dtype=solve_dtype)

    recovered, pcg = recovery_forward_audit(
        prediction_device.detach(),
        initial_device,
        edge_device,
        degree_device,
        regularization=args.lambda_value,
        maximum_iterations=args.maximum_iterations,
        tolerance=args.tolerance,
    )
    differentiable = differentiable_regularized_sparse_recovery(
        prediction_device,
        initial_device,
        edge_device,
        degree_device,
        regularization=args.lambda_value,
        maximum_iterations=args.maximum_iterations,
        tolerance=args.tolerance,
    )
    refine_loss = (differentiable - clean_device).square().sum(dim=-1).mean()
    refine_loss.backward()

    laplacian, lap_data = uniform_sparse_laplacian(
        faces.cpu().numpy(), int(initial.shape[0])
    )
    component_count, labels = component_labels(lap_data)
    expected, lsmr = regularized_sparse_solve(
        laplacian,
        prediction.numpy().astype(np.float64),
        initial.numpy().astype(np.float64),
        labels,
        component_count,
        args.lambda_value,
        atol=1e-12,
        btol=1e-12,
        maxiter=100000,
    )
    recovered_cpu = recovered.cpu().numpy().astype(np.float64)
    difference = recovered_cpu - expected
    prepared_laplacian = uniform_laplacian_apply(
        clean_device, edge_device, degree_device
    ).cpu().numpy()
    sparse_laplacian = laplacian @ clean.numpy().astype(np.float64)
    operator_difference = prepared_laplacian.astype(np.float64) - sparse_laplacian
    gradient = prediction_device.grad
    if gradient is None:
        raise RuntimeError("Missing gradient with respect to predicted Laplacian.")

    forward_vertex_rms = float(
        np.sqrt(np.mean(np.sum(difference * difference, axis=1)))
    )
    reference_displacement_rms = float(
        np.sqrt(np.mean(np.sum((expected - initial.numpy()) ** 2, axis=1)))
    )
    result = {
        "sample_id": str(static["sample_id"]),
        "split": args.split,
        "vertices": int(initial.shape[0]),
        "faces": int(faces.shape[0]),
        "lambda": args.lambda_value,
        "pcg_dtype": args.pcg_dtype,
        "pcg": {
            "iterations": pcg.iterations,
            "converged": pcg.converged,
            "relative_residual": pcg.relative_residual,
        },
        "standalone_lsmr": lsmr,
        "forward_difference": {
            "vertex_rms": forward_vertex_rms,
            "absolute_max": float(np.max(np.abs(difference))),
            "relative_to_reference_displacement_rms": (
                forward_vertex_rms / max(reference_displacement_rms, 1e-12)
            ),
        },
        "operator_difference": {
            "rms": float(np.sqrt(np.mean(operator_difference * operator_difference))),
            "absolute_max": float(np.max(np.abs(operator_difference))),
        },
        "gradient": {
            "all_finite": bool(torch.isfinite(gradient).all()),
            "norm": float(torch.linalg.vector_norm(gradient).detach().cpu()),
            "nonzero_entries": int(torch.count_nonzero(gradient).detach().cpu()),
        },
        "refine_loss": float(refine_loss.detach().cpu()),
        "clean_vertices_loss_side_only": True,
        "gt_in_predictor_inputs": False,
    }
    result["passed"] = bool(
        result["pcg"]["converged"]
        and result["standalone_lsmr"]["all_converged"]
        and result["forward_difference"]["vertex_rms"] <= 2e-4
        and result["forward_difference"][
            "relative_to_reference_displacement_rms"
        ] <= 5e-3
        and result["operator_difference"]["absolute_max"] <= 2e-6
        and result["gradient"]["all_finite"]
        and result["gradient"]["norm"] > 0
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
