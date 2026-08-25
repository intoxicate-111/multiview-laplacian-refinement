#!/usr/bin/env python3
from __future__ import annotations

"""Fail-closed contract, gradient, and PCG/LSMR audit for the hybrid run."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.sparse import coo_matrix, eye, vstack
from scipy.sparse.linalg import lsmr

from mlr.data import Mesh
from mlr.learned_laplacian.differentiable_sparse_recovery import (
    differentiable_regularized_sparse_recovery,
    recovery_forward_audit,
    uniform_laplacian_apply,
    uniform_laplacian_transpose_apply,
)
from mlr.learned_laplacian.evaluation import evaluate_mesh_geometry
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.multi_trainer import _build_model


LAMBDA = 3e-2


def _cycle() -> tuple[torch.Tensor, torch.Tensor]:
    source = torch.tensor([1, 3, 0, 2, 1, 3, 2, 0], dtype=torch.long)
    destination = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
    return torch.stack((source, destination)), torch.full((4, 1), 2.0, dtype=torch.double)


def _gradient_audit() -> dict[str, Any]:
    edge, degree = _cycle()
    generator = torch.Generator().manual_seed(7)
    lap = torch.randn((4, 3), generator=generator, dtype=torch.double, requires_grad=True)
    direct = torch.randn((4, 3), generator=generator, dtype=torch.double, requires_grad=True)
    clean = torch.randn((4, 3), generator=generator, dtype=torch.double)

    def objective(lap_value: torch.Tensor, direct_value: torch.Tensor) -> torch.Tensor:
        recovered = differentiable_regularized_sparse_recovery(
            lap_value,
            direct_value,
            edge,
            degree,
            regularization=LAMBDA,
            maximum_iterations=256,
            tolerance=1e-11,
        )
        return (recovered - clean).square().sum(dim=-1).mean()

    objective(lap, direct).backward()
    assert lap.grad is not None and direct.grad is not None
    epsilon = 1e-6
    checks = []
    for branch, index in (("lap", (1, 2)), ("direct", (2, 0))):
        source = lap.detach() if branch == "lap" else direct.detach()
        plus, minus = source.clone(), source.clone()
        plus[index] += epsilon
        minus[index] -= epsilon
        if branch == "lap":
            finite = (objective(plus, direct.detach()) - objective(minus, direct.detach())) / (2 * epsilon)
            analytic = lap.grad[index]
        else:
            finite = (objective(lap.detach(), plus) - objective(lap.detach(), minus)) / (2 * epsilon)
            analytic = direct.grad[index]
        relative = abs(float(analytic - finite)) / max(abs(float(finite)), 1e-12)
        checks.append({"branch": branch, "analytic": float(analytic), "finite_difference": float(finite), "relative_error": relative})
    result = {
        "lap_gradient_norm": float(torch.linalg.vector_norm(lap.grad)),
        "direct_gradient_norm": float(torch.linalg.vector_norm(direct.grad)),
        "all_finite": bool(torch.isfinite(lap.grad).all() and torch.isfinite(direct.grad).all()),
        "maximum_checked_relative_error": max(item["relative_error"] for item in checks),
        "checks": checks,
        "expected_lap_dependency": "A^-1 L^T; backward L A^-1 g",
        "expected_direct_dependency": "lambda A^-1; backward lambda A^-1 g",
    }
    if not result["all_finite"] or min(result["lap_gradient_norm"], result["direct_gradient_norm"]) <= 0:
        raise RuntimeError("A hybrid branch has missing/non-finite gradient.")
    if result["maximum_checked_relative_error"] > 1e-5:
        raise RuntimeError(f"Hybrid finite-difference error is too large: {result}")
    return result


def _sparse_laplacian(static: dict[str, Any]):
    vertices = int(np.asarray(static["vertices"]).shape[0])
    edge = np.asarray(static["edge_index"], dtype=np.int64)
    degree = np.asarray(static["vertex_degree"], dtype=np.float64).reshape(-1)
    source, destination = edge
    rows = np.concatenate((np.arange(vertices), destination))
    columns = np.concatenate((np.arange(vertices), source))
    values = np.concatenate((np.ones(vertices), -1.0 / degree[destination]))
    return coo_matrix((values, (rows, columns)), shape=(vertices, vertices)).tocsr()


def _solver_audit(manifest: Path, count: int) -> list[dict[str, Any]]:
    dataset = PreparedMeshDataset.from_manifest(manifest, "validation")
    indices = sorted(set(np.linspace(0, len(dataset) - 1, count, dtype=int).tolist()))
    rows = []
    for index in indices:
        static = dataset.load_static(index)
        prediction = torch.as_tensor(static["raw_laplacian_target"], dtype=torch.double)
        direct = torch.as_tensor(static["vertices"], dtype=torch.double)
        edge = torch.as_tensor(static["edge_index"], dtype=torch.long)
        degree = torch.as_tensor(static["vertex_degree"], dtype=torch.double)
        pcg, audit = recovery_forward_audit(
            prediction,
            direct,
            edge,
            degree,
            regularization=LAMBDA,
            maximum_iterations=2048,
            tolerance=1e-8,
        )
        lap = _sparse_laplacian(static)
        system = vstack((lap, np.sqrt(LAMBDA) * eye(lap.shape[0])), format="csr")
        rhs = np.vstack((prediction.numpy(), np.sqrt(LAMBDA) * direct.numpy()))
        reference = np.column_stack([
            lsmr(system, rhs[:, axis], atol=1e-12, btol=1e-12, maxiter=100000)[0]
            for axis in range(3)
        ])
        pcg_np = pcg.numpy()
        lap_residual = np.linalg.norm(lap @ pcg_np - prediction.numpy())
        anchor_residual = np.linalg.norm(pcg_np - direct.numpy())
        normal_residual = np.linalg.norm(
            lap.T @ (lap @ pcg_np - prediction.numpy()) + LAMBDA * (pcg_np - direct.numpy())
        )
        pcg_mesh = Mesh(pcg_np, np.asarray(static["faces"], dtype=np.int64)).ensure_normals()
        reference_mesh = Mesh(reference, np.asarray(static["faces"], dtype=np.int64)).ensure_normals()
        clean = Mesh(
            np.asarray(static["clean_reference_vertices"], dtype=np.float64),
            np.asarray(static["clean_reference_faces"], dtype=np.int64),
        ).ensure_normals()
        pcg_cd = evaluate_mesh_geometry(pcg_mesh, clean, surface_samples=3000, seed=7)["chamfer"]
        reference_cd = evaluate_mesh_geometry(reference_mesh, clean, surface_samples=3000, seed=7)["chamfer"]
        difference = pcg_np - reference
        row = {
            "sample_id": str(static["sample_id"]),
            "vertices": int(pcg_np.shape[0]),
            "hybrid_objective": float(lap_residual**2 + LAMBDA * anchor_residual**2),
            "laplacian_residual": float(lap_residual),
            "direct_anchor_residual": float(anchor_residual),
            "normal_equation_residual": float(normal_residual),
            "relative_residual": float(audit.relative_residual),
            "pcg_iterations": int(audit.iterations),
            "pcg_converged": bool(audit.converged),
            "pcg_lsmr_vertex_rms": float(np.sqrt(np.mean(np.sum(difference**2, axis=1)))),
            "pcg_lsmr_max_coordinate": float(np.max(np.abs(difference))),
            "pcg_chamfer": float(pcg_cd),
            "lsmr_chamfer": float(reference_cd),
            "chamfer_difference": float(pcg_cd - reference_cd),
        }
        if not audit.converged or row["pcg_lsmr_vertex_rms"] > 1e-5 or abs(row["chamfer_difference"]) > 1e-5:
            raise RuntimeError(f"PCG/LSMR preflight failed: {row}")
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--representative-samples", type=int, default=3)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model = _build_model(config, None, False)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    added_parameters = sum(parameter.numel() for parameter in model.hybrid_direct_head.parameters())
    contract = {
        "dataset": config["dataset"]["name"] == "Sofa50MultiTopologyRawLap500_v2",
        "split": config["dataset"]["expected_split_counts"] == {"train": 400, "validation": 50, "test": 50},
        "lambda": config["training"]["hybrid_single_geometry_loss"]["lambda"] == LAMBDA,
        "only_final_geometry_loss": not config["training"]["recovery_aware_geometry_loss"]["enabled"],
        "confidence_disabled": not config["confidence"]["enabled"],
        "adaptive_lambda_disabled": not config["model"]["recovery_lambda_head"]["enabled"],
        "direct_head_enabled": model.hybrid_direct_head_enabled,
        "world_size_8": config["experiment_metadata"]["distributed_world_size"] == 8,
        "effective_batch_8": config["experiment_metadata"]["effective_global_batch_meshes"] == 8,
        "steps_20000": config["multi_object_training"]["max_optimizer_steps"] == 20000,
        "validation_chamfer_selection": config["experiment_metadata"]["checkpoint_selector"] == "validation_final_hybrid_unified_v2_chamfer_only",
        "from_scratch": config["experiment_metadata"]["initialization"] == "from_scratch",
    }
    payload = {
        "contract_audit": all(contract.values()),
        "contract_checks": contract,
        "architecture": {
            "total_parameters": total_parameters,
            "arm_b_parameters": 826115,
            "arm_e_parameters": 826115,
            "additional_vs_arm_b": total_parameters - 826115,
            "additional_vs_arm_e": total_parameters - 826115,
            "direct_head_parameters": added_parameters,
        },
        "gradient_audit": _gradient_audit(),
        "solver_audit": _solver_audit(args.manifest.resolve(), args.representative_samples),
        "solver_settings": {"lambda": LAMBDA, "dtype": "float64", "tolerance": 1e-8, "maximum_iterations": 2048},
    }
    if not payload["contract_audit"]:
        raise RuntimeError(f"Hybrid contract failed: {contract}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
