#!/usr/bin/env python3
from __future__ import annotations

"""Fail-closed contract, gradient, and PCG/LSMR audit for Cotangent Arm C."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.sparse import coo_matrix, eye, vstack
from scipy.sparse.linalg import lsmr

from mlr.data import Mesh
from mlr.learned_laplacian.cotangent_sparse_recovery import (
    build_symmetric_cotangent_stiffness,
    differentiable_cotangent_sparse_recovery,
    differentiable_cotangent_sparse_recovery_with_audit,
)
from mlr.learned_laplacian.evaluation import evaluate_mesh_geometry
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.multi_trainer import _build_model


def _matrix(
    vertices: torch.Tensor, faces: torch.Tensor, epsilon: float
):
    edges, weights, diagonal, construction = build_symmetric_cotangent_stiffness(
        vertices, faces, relative_area_epsilon=epsilon
    )
    edge = edges.numpy()
    count = len(vertices)
    rows = np.concatenate((np.arange(count), edge[0], edge[1]))
    columns = np.concatenate((np.arange(count), edge[1], edge[0]))
    values = np.concatenate((diagonal.numpy(), -weights.numpy(), -weights.numpy()))
    return (
        coo_matrix((values, (rows, columns)), shape=(count, count)).tocsr(),
        edges,
        weights,
        diagonal,
        construction,
    )


def _gradient_audit(regularization: float, epsilon: float) -> dict[str, Any]:
    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.2, 0.3, 1.0],
        ],
        dtype=torch.double,
    )
    faces = torch.tensor(
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=torch.long
    )
    _, edges, weights, diagonal, _ = _matrix(vertices, faces, epsilon)
    generator = torch.Generator().manual_seed(7)
    lap = torch.randn((4, 3), generator=generator, dtype=torch.double, requires_grad=True)
    direct = torch.randn((4, 3), generator=generator, dtype=torch.double, requires_grad=True)
    clean = torch.randn((4, 3), generator=generator, dtype=torch.double)

    def objective(lap_value: torch.Tensor, direct_value: torch.Tensor) -> torch.Tensor:
        recovered = differentiable_cotangent_sparse_recovery(
            lap_value,
            direct_value,
            edges,
            weights,
            diagonal,
            regularization=regularization,
            maximum_iterations=512,
            tolerance=1e-11,
        )
        return (recovered - clean).square().sum(dim=-1).mean()

    objective(lap, direct).backward()
    assert lap.grad is not None and direct.grad is not None
    checks = []
    finite_difference_epsilon = 1e-6
    for branch, index in (("delta_pred", (1, 2)), ("V_predict", (2, 0))):
        source = lap.detach() if branch == "delta_pred" else direct.detach()
        plus, minus = source.clone(), source.clone()
        plus[index] += finite_difference_epsilon
        minus[index] -= finite_difference_epsilon
        if branch == "delta_pred":
            finite = (objective(plus, direct.detach()) - objective(minus, direct.detach())) / (
                2 * finite_difference_epsilon
            )
            analytic = lap.grad[index]
        else:
            finite = (objective(lap.detach(), plus) - objective(lap.detach(), minus)) / (
                2 * finite_difference_epsilon
            )
            analytic = direct.grad[index]
        relative = abs(float(analytic - finite)) / max(abs(float(finite)), 1e-12)
        checks.append(
            {
                "branch": branch,
                "analytic": float(analytic),
                "finite_difference": float(finite),
                "relative_error": relative,
            }
        )
    payload = {
        "delta_pred_gradient_norm": float(torch.linalg.vector_norm(lap.grad)),
        "V_predict_gradient_norm": float(torch.linalg.vector_norm(direct.grad)),
        "all_finite": bool(torch.isfinite(lap.grad).all() and torch.isfinite(direct.grad).all()),
        "maximum_checked_relative_error": max(row["relative_error"] for row in checks),
        "checks": checks,
        "expected_delta_dependency": "A^-1 C^T; backward C A^-1 g because C=C^T",
        "expected_direct_dependency": "lambda A^-1; backward lambda A^-1 g",
    }
    if not payload["all_finite"] or payload["maximum_checked_relative_error"] > 1e-5:
        raise RuntimeError(f"Cotangent gradient audit failed: {payload}")
    return payload


def _solver_audit(
    manifest: Path,
    count: int,
    regularization: float,
    epsilon: float,
) -> list[dict[str, Any]]:
    dataset = PreparedMeshDataset.from_manifest(manifest, "validation")
    indices = sorted(set(np.linspace(0, len(dataset) - 1, count, dtype=int).tolist()))
    rows = []
    for index in indices:
        static = dataset.load_static(index)
        vertices = torch.as_tensor(static["vertices"], dtype=torch.double)
        faces = torch.as_tensor(static["faces"], dtype=torch.long)
        cotangent, edges, weights, diagonal, construction = _matrix(
            vertices, faces, epsilon
        )
        prediction = cotangent @ np.asarray(
            static["clean_reference_vertices"], dtype=np.float64
        )
        prediction_tensor = torch.from_numpy(prediction)
        pcg, audit = differentiable_cotangent_sparse_recovery_with_audit(
            prediction_tensor,
            vertices,
            edges,
            weights,
            diagonal,
            regularization=regularization,
            maximum_iterations=2048,
            tolerance=1e-8,
        )
        system = vstack(
            (cotangent, np.sqrt(regularization) * eye(cotangent.shape[0])),
            format="csr",
        )
        rhs = np.vstack((prediction, np.sqrt(regularization) * vertices.numpy()))
        reference = np.column_stack(
            [
                lsmr(system, rhs[:, axis], atol=1e-12, btol=1e-12, maxiter=100000)[0]
                for axis in range(3)
            ]
        )
        pcg_np = pcg.detach().numpy()
        laplacian_residual = np.linalg.norm(cotangent @ pcg_np - prediction)
        anchor_residual = np.linalg.norm(pcg_np - vertices.numpy())
        normal_residual = np.linalg.norm(
            cotangent.T @ (cotangent @ pcg_np - prediction)
            + regularization * (pcg_np - vertices.numpy())
        )
        clean = Mesh(
            np.asarray(static["clean_reference_vertices"], dtype=np.float64),
            np.asarray(static["faces"], dtype=np.int64),
        ).ensure_normals()
        pcg_cd = evaluate_mesh_geometry(
            Mesh(pcg_np, np.asarray(static["faces"], dtype=np.int64)).ensure_normals(),
            clean,
            surface_samples=3000,
            seed=7,
        )["chamfer"]
        reference_cd = evaluate_mesh_geometry(
            Mesh(reference, np.asarray(static["faces"], dtype=np.int64)).ensure_normals(),
            clean,
            surface_samples=3000,
            seed=7,
        )["chamfer"]
        difference = pcg_np - reference
        row = {
            "sample_id": str(static["sample_id"]),
            "vertices": int(len(vertices)),
            "protected_triangles": int(construction.protected_triangles),
            "negative_edge_weights": int(construction.negative_edge_weights),
            "hybrid_objective": float(
                laplacian_residual**2 + regularization * anchor_residual**2
            ),
            "laplacian_residual": float(laplacian_residual),
            "direct_anchor_residual": float(anchor_residual),
            "normal_equation_residual": float(normal_residual),
            "relative_residual": float(audit.relative_residual),
            "pcg_iterations": int(audit.iterations),
            "pcg_converged": bool(audit.converged),
            "pcg_lsmr_vertex_rms": float(
                np.sqrt(np.mean(np.sum(np.square(difference), axis=1)))
            ),
            "pcg_lsmr_max_coordinate": float(np.max(np.abs(difference))),
            "pcg_chamfer": float(pcg_cd),
            "lsmr_chamfer": float(reference_cd),
            "chamfer_difference": float(pcg_cd - reference_cd),
        }
        if (
            not audit.converged
            or row["pcg_lsmr_vertex_rms"] > 1e-5
            or abs(row["chamfer_difference"]) > 1e-5
        ):
            raise RuntimeError(f"Cotangent PCG/LSMR preflight failed: {row}")
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
    settings = config["training"]["hybrid_single_geometry_loss"]
    regularization = float(settings["lambda"])
    epsilon = float(settings["cotangent_relative_area_epsilon"])
    model = _build_model(config, None, False)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    contract = {
        "dataset": config["dataset"]["name"] == "Sofa50MultiTopologyRawLap500_v2",
        "split": config["dataset"]["expected_split_counts"]
        == {"train": 400, "validation": 50, "test": 50},
        "cotangent_operator": settings["operator"] == "symmetric_cotangent_stiffness",
        "mass_normalization_disabled": "no_mass_normalization"
        in config["recovery"]["operator"],
        "only_final_geometry_loss": not config["training"]["recovery_aware_geometry_loss"]["enabled"],
        "confidence_disabled": not config["confidence"]["enabled"],
        "adaptive_lambda_disabled": not config["model"]["recovery_lambda_head"]["enabled"],
        "direct_head_enabled": model.hybrid_direct_head_enabled,
        "world_size_8": config["experiment_metadata"]["distributed_world_size"] == 8,
        "effective_batch_8": config["experiment_metadata"]["effective_global_batch_meshes"] == 8,
        "steps_20000": config["multi_object_training"]["max_optimizer_steps"] == 20000,
        "validation_chamfer_selection": config["experiment_metadata"]["checkpoint_selector"]
        == "validation_final_hybrid_unified_v2_chamfer_only",
        "from_scratch": config["experiment_metadata"]["initialization"] == "from_scratch",
        "same_parameter_count_as_uniform": total_parameters == 892678,
    }
    payload = {
        "contract_audit": all(contract.values()),
        "contract_checks": contract,
        "architecture": {"total_parameters": total_parameters},
        "gradient_audit": _gradient_audit(regularization, epsilon),
        "solver_audit": _solver_audit(
            args.manifest.resolve(), args.representative_samples, regularization, epsilon
        ),
        "solver_settings": {
            "lambda": regularization,
            "dtype": "float64",
            "tolerance": float(settings["tolerance"]),
            "maximum_iterations": int(settings["maximum_iterations"]),
            "operator": "symmetric_cotangent_stiffness",
        },
    }
    if not payload["contract_audit"]:
        raise RuntimeError(f"Cotangent contract failed: {contract}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
