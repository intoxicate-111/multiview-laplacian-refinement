from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from mlr.coarse_lap_oracle import (
    build_uniform_laplacian_data,
    oracle_loss_and_grad,
    point_to_surface_stats as numpy_point_to_surface_stats,
)
from mlr.data import Mesh
from mlr.io import save_mesh
from mlr.laplacian import unique_edges
from mlr.refinement import RefinementConfig, RefinementResult, refine_mesh_with_laplacian

from .losses import laplacian_prediction_metrics
from .graph_layers import faces_to_edge_index
from .target_scaling import mean_incident_edge_length, normalize_laplacian_by_edge_scale


def reconstruct_and_evaluate(
    sample: Mapping[str, Any],
    predicted_laplacian: torch.Tensor | np.ndarray,
    output_dir: str | Path,
    reconstruction_config: Mapping[str, Any],
    normalized_prediction: torch.Tensor | np.ndarray | None = None,
    edge_scale_epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Run the existing non-differentiable solver for prediction and oracle evaluation."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vertices = _numpy(sample["vertices"])
    faces = _numpy(sample["faces"]).astype(np.int64)
    target = _numpy(sample.get("raw_laplacian_target", sample["laplacian_target"]))
    confidence = _numpy(sample["target_confidence"])
    prediction = _numpy(predicted_laplacian)
    if prediction.shape != vertices.shape:
        raise ValueError(f"predicted_laplacian must have shape {vertices.shape}, got {prediction.shape}.")
    if not np.isfinite(prediction).all():
        raise ValueError("predicted_laplacian contains NaN or infinite values.")
    if "local_edge_length" in sample:
        local_edge_length_t = torch.as_tensor(sample["local_edge_length"]).detach().cpu()
    else:
        # Keep the public evaluation entry point compatible with legacy samples,
        # including callers that do not pass through load_prepared_sample().
        vertices_t = torch.as_tensor(vertices)
        edge_index = faces_to_edge_index(torch.as_tensor(faces, dtype=torch.long))
        local_edge_length_t = mean_incident_edge_length(vertices_t, edge_index).cpu()
    normalized_target_t = normalize_laplacian_by_edge_scale(
        torch.as_tensor(target),
        local_edge_length_t,
        eps=edge_scale_epsilon,
        valid_scale_mask=torch.as_tensor(
            sample.get("valid_scale_mask", local_edge_length_t > 0), dtype=torch.bool
        ),
    )
    if normalized_prediction is None:
        normalized_prediction_t = normalize_laplacian_by_edge_scale(
            torch.as_tensor(prediction),
            local_edge_length_t,
            eps=edge_scale_epsilon,
            valid_scale_mask=torch.as_tensor(
                sample.get("valid_scale_mask", local_edge_length_t > 0), dtype=torch.bool
            ),
        )
    else:
        normalized_prediction_t = torch.as_tensor(normalized_prediction).detach().cpu()
    if tuple(normalized_prediction_t.shape) != tuple(normalized_target_t.shape):
        raise ValueError("normalized_prediction must have shape [N, 3].")

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
    dense_vertex_limit = int(reconstruction_config.get("dense_vertex_limit", 5000))
    predicted_result, solver_name = _reconstruct(
        coarse, prediction, confidence, refinement, dense_vertex_limit
    )
    oracle_result, oracle_solver_name = _reconstruct(
        coarse, target, confidence, refinement, dense_vertex_limit
    )

    np.save(output_dir / "delta_target.npy", target)
    np.save(output_dir / "delta_pred.npy", prediction)
    np.save(output_dir / "delta_hat_target.npy", normalized_target_t.numpy())
    np.save(output_dir / "delta_hat_pred.npy", normalized_prediction_t.numpy())
    np.save(output_dir / "local_edge_length.npy", local_edge_length_t.numpy())
    np.save(output_dir / "local_edge_scale.npy", local_edge_length_t.square().numpy())
    np.save(output_dir / "laplacian_error.npy", np.linalg.norm(prediction - target, axis=1))
    np.save(
        output_dir / "normalized_laplacian_error.npy",
        torch.linalg.vector_norm(normalized_prediction_t - normalized_target_t, dim=-1).numpy(),
    )
    save_mesh(coarse, output_dir / "coarse.obj")
    save_mesh(predicted_result.mesh, output_dir / "predicted_refined.obj")
    save_mesh(oracle_result.mesh, output_dir / "oracle_refined.obj")

    valid_scale_mask_t = torch.as_tensor(
        sample.get("valid_scale_mask", local_edge_length_t > 0), dtype=torch.bool
    )
    raw_prediction_metrics = laplacian_prediction_metrics(
        torch.as_tensor(prediction), torch.as_tensor(target), valid_mask=valid_scale_mask_t
    )
    normalized_prediction_metrics = laplacian_prediction_metrics(
        normalized_prediction_t, normalized_target_t, valid_mask=valid_scale_mask_t
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
        np.save(
            output_dir / "position_error.npy",
            np.linalg.norm(predicted_result.vertices - target_positions_np, axis=1),
        )

    if sample.get("gt_vertices") is not None and sample.get("gt_faces") is not None:
        gt_mesh = Mesh(
            _numpy(sample["gt_vertices"]),
            _numpy(sample["gt_faces"]).astype(np.int64),
        ).ensure_normals()
        chamfer_samples = int(reconstruction_config.get("chamfer_samples", 1000))
        metric_seed = int(reconstruction_config.get("metric_seed", 7))
        for name, mesh in (
            ("coarse", coarse),
            ("predicted", predicted_result.mesh),
            ("oracle", oracle_result.mesh),
        ):
            surface = _point_to_surface_stats(mesh.vertices, gt_mesh)
            geometry[name]["point_to_surface_mean"] = float(surface["mean"])
            geometry[name]["point_to_surface_median"] = float(surface["median"])
            geometry[name]["point_to_surface_max"] = float(surface["max"])
            geometry[name]["point_to_surface_engine"] = surface["engine"]
            geometry[name]["chamfer"] = float(
                _chamfer_distance(mesh, gt_mesh, samples=chamfer_samples, seed=metric_seed)
            )
            geometry[name]["normal_consistency"] = _normal_consistency(mesh, gt_mesh)

    coarse_metric = geometry["coarse"].get("point_to_surface_mean")
    predicted_metric = geometry["predicted"].get("point_to_surface_mean")
    if coarse_metric is None or predicted_metric is None:
        coarse_metric = geometry["coarse"].get("target_position_rmse")
        predicted_metric = geometry["predicted"].get("target_position_rmse")
    improves = bool(
        coarse_metric is not None and predicted_metric is not None and predicted_metric < coarse_metric
    )
    return {
        "laplacian_prediction": raw_prediction_metrics,
        "laplacian_prediction_raw": raw_prediction_metrics,
        "laplacian_prediction_normalized": normalized_prediction_metrics,
        "geometry": geometry,
        "predicted_improves_over_coarse": improves,
        "reconstruction": {
            "predicted_final_loss": predicted_result.history[-1]["loss"],
            "oracle_final_loss": oracle_result.history[-1]["loss"],
            "all_finite": bool(
                np.isfinite(predicted_result.vertices).all() and np.isfinite(oracle_result.vertices).all()
            ),
            "predicted_solver": solver_name,
            "oracle_solver": oracle_solver_name,
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
        "collapsed_or_exploded": bool(
            not np.isfinite(diagonal)
            or diagonal / max(reference_diagonal, 1e-12) < 0.25
            or diagonal / max(reference_diagonal, 1e-12) > 4.0
            or mean_edge <= 1e-12
        ),
    }


def _reconstruct(
    mesh: Mesh,
    delta_target: np.ndarray,
    confidence: np.ndarray,
    config: RefinementConfig,
    dense_vertex_limit: int,
) -> tuple[RefinementResult, str]:
    if mesh.num_vertices <= dense_vertex_limit or config.operator_type != "uniform":
        result = refine_mesh_with_laplacian(
            mesh,
            delta_target,
            confidence=confidence,
            anchors=mesh.vertices,
            config=config,
        )
        return result, "dense_refinement"
    if not np.allclose(confidence, 1.0):
        raise ValueError("Large sparse reconstruction currently requires uniform confidence.")
    return _refine_sparse_uniform(mesh, delta_target, config), "sparse_uniform_oracle_core"


def _refine_sparse_uniform(
    mesh: Mesh,
    delta_target: np.ndarray,
    config: RefinementConfig,
) -> RefinementResult:
    """Reuse the existing coarse-oracle sparse loss/gradient for large meshes."""

    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    anchors = vertices.copy()
    data = build_uniform_laplacian_data(mesh.faces, mesh.num_vertices)
    no_edges = np.zeros((0, 2), dtype=np.int64)
    no_lengths = np.zeros((0,), dtype=np.float64)
    m = np.zeros_like(vertices)
    v = np.zeros_like(vertices)
    history: list[dict[str, float]] = []
    for step in range(0, config.num_iters + 1):
        total, grad, parts = oracle_loss_and_grad(
            vertices,
            data,
            delta_target,
            anchors,
            anchors,
            no_edges,
            no_lengths,
            config.lambda_lap,
            config.lambda_anchor,
            0.0,
            0.0,
        )
        if step == 0 or step == config.num_iters or step % max(config.log_every, 1) == 0:
            history.append({"iter": float(step), "loss": float(total), **parts})
        if step == config.num_iters:
            break
        update_step = step + 1
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * (grad * grad)
        m_hat = m / (1.0 - 0.9**update_step)
        v_hat = v / (1.0 - 0.999**update_step)
        vertices -= config.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
    refined = mesh.with_vertices(vertices)
    return RefinementResult(mesh=refined, vertices=vertices, history=history, operator=None)


def _point_to_surface_stats(points: np.ndarray, surface_mesh: Mesh) -> dict[str, float | str]:
    if len(points) <= 5000 and surface_mesh.num_faces <= 10000:
        result = numpy_point_to_surface_stats(points, surface_mesh)
        return {**result, "engine": "numpy_exact"}
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError(
            "Large-mesh surface metrics require the optional bunny dependencies. "
            "Install with pip install -e '.[train,bunny]'."
        ) from exc
    surface = trimesh.Trimesh(
        vertices=surface_mesh.vertices,
        faces=surface_mesh.faces,
        process=False,
    )
    points = np.asarray(points)
    point_diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    surface_diagonal = float(
        np.linalg.norm(surface_mesh.vertices.max(axis=0) - surface_mesh.vertices.min(axis=0))
    )
    scale_ratio = point_diagonal / max(surface_diagonal, 1e-12)
    if scale_ratio < 0.25 or scale_ratio > 4.0:
        distances = _nearest_vertex_distances(points, surface_mesh.vertices)
        engine = "scipy_nearest_vertex_upper_bound_bbox_fallback"
    else:
        try:
            _, distances, _ = trimesh.proximity.closest_point(surface, points)
            engine = "trimesh_rtree_exact"
        except Exception as exc:
            # R-tree face candidates can exhaust memory for a severely exploded
            # prediction whose query boxes span most of the reference mesh. A
            # nearest reference vertex is a conservative (upper-bound) distance
            # and still lets failure diagnostics complete without hiding collapse.
            if not isinstance(exc, MemoryError) and exc.__class__.__name__ != "RTreeError":
                raise
            distances = _nearest_vertex_distances(points, surface_mesh.vertices)
            engine = "scipy_nearest_vertex_upper_bound_error_fallback"
    distances = np.asarray(distances, dtype=np.float64)
    return {
        "mean": float(np.mean(distances)),
        "rmse": float(np.sqrt(np.mean(distances * distances))),
        "median": float(np.median(distances)),
        "max": float(np.max(distances)),
        "engine": engine,
    }


def _nearest_vertex_distances(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree

    distances, _ = cKDTree(vertices).query(points, workers=-1)
    return np.asarray(distances, dtype=np.float64)


def _chamfer_distance(mesh: Mesh, gt_mesh: Mesh, samples: int, seed: int) -> float:
    mesh_points = _sample_vertices(mesh.vertices, samples, seed)
    gt_points = _sample_vertices(gt_mesh.vertices, samples, seed + 1)
    mesh_to_gt = _point_to_surface_stats(mesh_points, gt_mesh)["mean"]
    gt_to_mesh = _point_to_surface_stats(gt_points, mesh)["mean"]
    return 0.5 * (float(mesh_to_gt) + float(gt_to_mesh))


def _sample_vertices(vertices: np.ndarray, samples: int, seed: int) -> np.ndarray:
    if samples <= 0 or samples >= len(vertices):
        return vertices
    rng = np.random.default_rng(seed)
    return vertices[rng.choice(len(vertices), size=samples, replace=False)]


def _normal_consistency(mesh: Mesh, gt_mesh: Mesh) -> float:
    if mesh.num_vertices != gt_mesh.num_vertices or not np.array_equal(mesh.faces, gt_mesh.faces):
        return float("nan")
    mesh.ensure_normals()
    gt_mesh.ensure_normals()
    dots = np.einsum("ij,ij->i", mesh.normals, gt_mesh.normals)
    return float(np.mean(np.clip(dots, -1.0, 1.0)))


def _numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
