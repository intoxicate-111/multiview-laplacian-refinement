from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from mlr.coarse_lap_oracle import chamfer_distance, point_to_surface_stats
from mlr.data import Mesh
from mlr.io import save_mesh
from mlr.laplacian import unique_edges
from mlr.refinement import RefinementConfig, refine_mesh_with_laplacian

from .losses import laplacian_prediction_metrics


def reconstruct_and_evaluate(
    sample: Mapping[str, Any],
    predicted_laplacian: torch.Tensor | np.ndarray,
    output_dir: str | Path,
    reconstruction_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the existing non-differentiable solver for prediction and oracle evaluation."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vertices = _numpy(sample["vertices"])
    faces = _numpy(sample["faces"]).astype(np.int64)
    target = _numpy(sample["laplacian_target"])
    confidence = _numpy(sample["target_confidence"])
    prediction = _numpy(predicted_laplacian)
    if prediction.shape != vertices.shape:
        raise ValueError(f"predicted_laplacian must have shape {vertices.shape}, got {prediction.shape}.")
    if not np.isfinite(prediction).all():
        raise ValueError("predicted_laplacian contains NaN or infinite values.")

    coarse = Mesh(vertices.copy(), faces.copy()).ensure_normals()
    refinement = RefinementConfig(
        operator_type=str(reconstruction_config.get("operator_type", "uniform")),
        lambda_lap=float(reconstruction_config.get("lambda_lap", 1.0)),
        lambda_anchor=float(reconstruction_config.get("lambda_anchor", 0.01)),
        lambda_edge=float(reconstruction_config.get("lambda_edge", 0.0)),
        num_iters=int(reconstruction_config.get("num_iters", 500)),
        learning_rate=float(reconstruction_config.get("learning_rate", 0.01)),
        robust_loss=str(reconstruction_config.get("robust_loss", "huber")),
        huber_delta=float(reconstruction_config.get("huber_delta", 0.01)),
    )
    predicted_result = refine_mesh_with_laplacian(
        coarse,
        prediction,
        confidence=confidence,
        anchors=vertices,
        config=refinement,
    )
    oracle_result = refine_mesh_with_laplacian(
        coarse,
        target,
        confidence=confidence,
        anchors=vertices,
        config=refinement,
    )

    np.save(output_dir / "delta_target.npy", target)
    np.save(output_dir / "delta_pred.npy", prediction)
    save_mesh(coarse, output_dir / "coarse.obj")
    save_mesh(predicted_result.mesh, output_dir / "predicted_refined.obj")
    save_mesh(oracle_result.mesh, output_dir / "oracle_refined.obj")

    prediction_metrics = laplacian_prediction_metrics(
        torch.as_tensor(prediction), torch.as_tensor(target)
    )
    geometry = {
        "coarse": _mesh_quality_metrics(coarse, coarse),
        "predicted": _mesh_quality_metrics(predicted_result.mesh, coarse),
        "oracle": _mesh_quality_metrics(oracle_result.mesh, coarse),
    }
    target_positions = sample.get("target_positions")
    if target_positions is not None:
        target_positions_np = _numpy(target_positions)
        for name, mesh in (
            ("coarse", coarse),
            ("predicted", predicted_result.mesh),
            ("oracle", oracle_result.mesh),
        ):
            distances = np.linalg.norm(mesh.vertices - target_positions_np, axis=1)
            geometry[name]["target_position_rmse"] = float(np.sqrt(np.mean(distances**2)))
            geometry[name]["target_position_mae"] = float(np.mean(distances))

    if sample.get("gt_vertices") is not None and sample.get("gt_faces") is not None:
        gt_mesh = Mesh(
            _numpy(sample["gt_vertices"]),
            _numpy(sample["gt_faces"]).astype(np.int64),
        ).ensure_normals()
        chamfer_samples = int(reconstruction_config.get("chamfer_samples", 1000))
        for name, mesh in (
            ("coarse", coarse),
            ("predicted", predicted_result.mesh),
            ("oracle", oracle_result.mesh),
        ):
            surface = point_to_surface_stats(mesh.vertices, gt_mesh)
            geometry[name]["point_to_surface_mean"] = float(surface["mean"])
            geometry[name]["point_to_surface_max"] = float(surface["max"])
            geometry[name]["chamfer"] = float(
                chamfer_distance(mesh, gt_mesh, samples=chamfer_samples, seed=7)
            )

    coarse_metric = geometry["coarse"].get("point_to_surface_mean")
    predicted_metric = geometry["predicted"].get("point_to_surface_mean")
    if coarse_metric is None or predicted_metric is None:
        coarse_metric = geometry["coarse"].get("target_position_rmse")
        predicted_metric = geometry["predicted"].get("target_position_rmse")
    improves = bool(
        coarse_metric is not None and predicted_metric is not None and predicted_metric < coarse_metric
    )
    return {
        "laplacian_prediction": prediction_metrics,
        "geometry": geometry,
        "predicted_improves_over_coarse": improves,
        "reconstruction": {
            "predicted_final_loss": predicted_result.history[-1]["loss"],
            "oracle_final_loss": oracle_result.history[-1]["loss"],
            "all_finite": bool(
                np.isfinite(predicted_result.vertices).all() and np.isfinite(oracle_result.vertices).all()
            ),
        },
    }


def _mesh_quality_metrics(mesh: Mesh, reference: Mesh) -> dict[str, float | bool]:
    edges = unique_edges(mesh.faces)
    diagonal = float(np.linalg.norm(mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)))
    reference_diagonal = float(
        np.linalg.norm(reference.vertices.max(axis=0) - reference.vertices.min(axis=0))
    )
    if len(edges):
        lengths = np.linalg.norm(mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]], axis=1)
        mean_edge = float(lengths.mean())
    else:
        mean_edge = 0.0
    return {
        "all_finite": bool(np.isfinite(mesh.vertices).all()),
        "bbox_diagonal": diagonal,
        "bbox_diagonal_ratio_to_coarse": diagonal / max(reference_diagonal, 1e-12),
        "mean_edge_length": mean_edge,
    }


def _numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
