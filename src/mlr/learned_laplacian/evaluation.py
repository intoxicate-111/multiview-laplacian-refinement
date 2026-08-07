from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from mlr.coarse_lap_oracle import (
    apply_uniform_laplacian,
    apply_uniform_laplacian_transpose,
    build_uniform_laplacian_data,
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
    laplacian_weight: torch.Tensor | np.ndarray | None = None,
    unseen_anchor_weight: float = 0.0,
    evaluate_laplacian_prediction: bool = True,
    evaluate_initial_geometry: bool = True,
    solver_confidence: torch.Tensor | np.ndarray | None = None,
) -> dict[str, Any]:
    """Run the existing non-differentiable solver for prediction and oracle evaluation."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vertices = _numpy(sample["vertices"])
    faces = _numpy(sample["faces"]).astype(np.int64)
    target = _numpy(sample.get("raw_laplacian_target", sample["laplacian_target"]))
    confidence = _numpy(
        sample["target_confidence"] if solver_confidence is None else solver_confidence
    )
    if confidence.shape != (vertices.shape[0],):
        raise ValueError("solver_confidence must have shape [N].")
    prediction = _numpy(predicted_laplacian)
    recovery_weight = (
        np.ones(vertices.shape[0], dtype=np.float64)
        if laplacian_weight is None
        else _numpy(laplacian_weight).astype(np.float64).reshape(-1)
    )
    if recovery_weight.shape != (vertices.shape[0],):
        raise ValueError("laplacian_weight must have shape [N].")
    if not np.isfinite(recovery_weight).all() or np.any(recovery_weight < 0):
        raise ValueError("laplacian_weight must be finite and non-negative.")
    if unseen_anchor_weight < 0:
        raise ValueError("unseen_anchor_weight must be non-negative.")
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
    normalized_target_t = None
    if evaluate_laplacian_prediction:
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
    if tuple(normalized_prediction_t.shape) != tuple(vertices.shape):
        raise ValueError("normalized_prediction must have shape [N, 3].")

    coarse = Mesh(vertices.copy(), faces.copy()).ensure_normals()
    refinement = RefinementConfig(
        operator_type=str(reconstruction_config.get("operator_type", "uniform")),
        lambda_lap=float(reconstruction_config.get("lambda_lap", 1.0)),
        lambda_anchor=float(reconstruction_config.get("lambda_anchor", 0.01)),
        lambda_edge=float(reconstruction_config.get("lambda_edge", 0.0)),
        lambda_unseen_anchor=float(unseen_anchor_weight),
        num_iters=int(reconstruction_config.get("num_iters", 500)),
        learning_rate=float(reconstruction_config.get("learning_rate", 0.01)),
        robust_loss=str(reconstruction_config.get("robust_loss", "huber")),
        huber_delta=float(reconstruction_config.get("huber_delta", 0.01)),
    )
    dense_vertex_limit = int(reconstruction_config.get("dense_vertex_limit", 5000))
    predicted_result, solver_name = _reconstruct(
        coarse,
        prediction,
        confidence,
        refinement,
        dense_vertex_limit,
        laplacian_weight=recovery_weight,
    )
    evaluate_oracle = bool(reconstruction_config.get("evaluate_oracle", True))
    oracle_result = None
    oracle_solver_name = None
    if evaluate_oracle:
        oracle_result, oracle_solver_name = _reconstruct(
            coarse, target, confidence, refinement, dense_vertex_limit
        )

    np.save(output_dir / "delta_pred_raw.npy", prediction)
    np.save(
        output_dir / "delta_hat_prediction.npy", normalized_prediction_t.numpy()
    )
    np.save(output_dir / "h_current.npy", local_edge_length_t.numpy())
    np.save(output_dir / "h_current_squared.npy", local_edge_length_t.square().numpy())
    if bool(reconstruction_config.get("write_legacy_prediction_names", True)):
        np.save(output_dir / "delta_pred.npy", prediction)
        np.save(output_dir / "delta_hat_pred.npy", normalized_prediction_t.numpy())
        np.save(output_dir / "local_edge_length.npy", local_edge_length_t.numpy())
        np.save(output_dir / "local_edge_scale.npy", local_edge_length_t.square().numpy())
    if evaluate_laplacian_prediction:
        assert normalized_target_t is not None
        np.save(output_dir / "delta_target.npy", target)
        np.save(output_dir / "delta_hat_target.npy", normalized_target_t.numpy())
        np.save(output_dir / "laplacian_error.npy", np.linalg.norm(prediction - target, axis=1))
        np.save(
            output_dir / "normalized_laplacian_error.npy",
            torch.linalg.vector_norm(
                normalized_prediction_t - normalized_target_t, dim=-1
            ).numpy(),
        )
    save_mesh(coarse, output_dir / "coarse.obj")
    save_mesh(predicted_result.mesh, output_dir / "predicted_refined.obj")
    if oracle_result is not None:
        save_mesh(oracle_result.mesh, output_dir / "oracle_refined.obj")

    valid_scale_mask_t = torch.as_tensor(
        sample.get("valid_scale_mask", local_edge_length_t > 0), dtype=torch.bool
    )
    raw_prediction_metrics = None
    normalized_prediction_metrics = None
    if evaluate_laplacian_prediction:
        assert normalized_target_t is not None
        raw_prediction_metrics = laplacian_prediction_metrics(
            torch.as_tensor(prediction),
            torch.as_tensor(target),
            valid_mask=valid_scale_mask_t,
        )
        normalized_prediction_metrics = laplacian_prediction_metrics(
            normalized_prediction_t,
            normalized_target_t,
            valid_mask=valid_scale_mask_t,
        )
    geometry = {
        "coarse": _mesh_quality_metrics(coarse, coarse),
        "predicted": _mesh_quality_metrics(predicted_result.mesh, coarse),
    }
    if oracle_result is not None:
        geometry["oracle"] = _mesh_quality_metrics(oracle_result.mesh, coarse)
    target_positions = sample.get("target_positions")
    if target_positions is not None:
        target_positions_np = _numpy(target_positions)
        evaluated_meshes = [("predicted", predicted_result.mesh)]
        if evaluate_initial_geometry:
            evaluated_meshes.insert(0, ("coarse", coarse))
        if oracle_result is not None:
            evaluated_meshes.append(("oracle", oracle_result.mesh))
        for name, mesh in evaluated_meshes:
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
        evaluated_meshes = [("predicted", predicted_result.mesh)]
        if evaluate_initial_geometry:
            evaluated_meshes.insert(0, ("coarse", coarse))
        if oracle_result is not None:
            evaluated_meshes.append(("oracle", oracle_result.mesh))
        for name, mesh in evaluated_meshes:
            surface = _point_to_surface_stats(mesh.vertices, gt_mesh)
            reverse_surface = _point_to_surface_stats(gt_mesh.vertices, mesh)
            geometry[name]["point_to_surface_mean"] = float(surface["mean"])
            geometry[name]["point_to_surface_median"] = float(surface["median"])
            geometry[name]["point_to_surface_max"] = float(surface["max"])
            geometry[name]["point_to_surface_engine"] = surface["engine"]
            geometry[name]["point_to_surface_forward_mean"] = float(surface["mean"])
            geometry[name]["point_to_surface_reverse_mean"] = float(
                reverse_surface["mean"]
            )
            geometry[name]["point_to_surface_bidirectional_mean"] = float(
                0.5 * (surface["mean"] + reverse_surface["mean"])
            )
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
            "oracle_final_loss": (
                oracle_result.history[-1]["loss"] if oracle_result is not None else None
            ),
            "all_finite": bool(
                np.isfinite(predicted_result.vertices).all()
                and (
                    oracle_result is None
                    or np.isfinite(oracle_result.vertices).all()
                )
            ),
            "predicted_solver": solver_name,
            "oracle_solver": oracle_solver_name,
            "oracle_evaluated": evaluate_oracle,
            "visibility_weighted": laplacian_weight is not None,
            "visible_laplacian_equations": int(np.count_nonzero(recovery_weight > 0)),
            "zero_weight_laplacian_equations": int(
                np.count_nonzero(recovery_weight <= 0)
            ),
            "unseen_anchor_weight": float(unseen_anchor_weight),
            "laplacian_prediction_evaluated": bool(evaluate_laplacian_prediction),
            "initial_geometry_evaluated": bool(evaluate_initial_geometry),
            "predicted_final_terms": dict(predicted_result.history[-1]),
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
    *,
    laplacian_weight: np.ndarray | None = None,
) -> tuple[RefinementResult, str]:
    if mesh.num_vertices <= dense_vertex_limit or config.operator_type != "uniform":
        result = refine_mesh_with_laplacian(
            mesh,
            delta_target,
            confidence=confidence,
            anchors=mesh.vertices,
            config=config,
            laplacian_weight=laplacian_weight,
        )
        return result, "dense_refinement"
    if not np.allclose(confidence, 1.0):
        raise ValueError("Large sparse reconstruction currently requires uniform confidence.")
    return (
        _refine_sparse_uniform(mesh, delta_target, config, laplacian_weight),
        "sparse_uniform_oracle_core",
    )


def _refine_sparse_uniform(
    mesh: Mesh,
    delta_target: np.ndarray,
    config: RefinementConfig,
    laplacian_weight: np.ndarray | None = None,
) -> RefinementResult:
    """Reuse the existing coarse-oracle sparse loss/gradient for large meshes."""

    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    anchors = vertices.copy()
    data = build_uniform_laplacian_data(mesh.faces, mesh.num_vertices)
    m = np.zeros_like(vertices)
    v = np.zeros_like(vertices)
    history: list[dict[str, float]] = []
    weight = (
        np.ones(mesh.num_vertices, dtype=np.float64)
        if laplacian_weight is None
        else np.asarray(laplacian_weight, dtype=np.float64).reshape(mesh.num_vertices)
    )

    def loss_and_grad(current: np.ndarray):
        lap_residual = apply_uniform_laplacian(current, data) - delta_target
        weighted_lap_residual = lap_residual * np.sqrt(weight)[:, None]
        lap_loss = float(
            np.mean(np.sum(weighted_lap_residual * weighted_lap_residual, axis=1))
        )
        gradient = config.lambda_lap * (2.0 / mesh.num_vertices) * (
            apply_uniform_laplacian_transpose(lap_residual * weight[:, None], data)
        )
        anchor_residual = current - anchors
        anchor_loss = float(np.mean(np.sum(anchor_residual * anchor_residual, axis=1)))
        gradient += config.lambda_anchor * (2.0 / mesh.num_vertices) * anchor_residual
        unseen = (weight <= 0).astype(np.float64)[:, None]
        unseen_residual = anchor_residual * unseen
        unseen_loss = float(np.mean(np.sum(unseen_residual * unseen_residual, axis=1)))
        gradient += (
            config.lambda_unseen_anchor
            * (2.0 / mesh.num_vertices)
            * unseen_residual
        )
        total = (
            config.lambda_lap * lap_loss
            + config.lambda_anchor * anchor_loss
            + config.lambda_unseen_anchor * unseen_loss
        )
        parts = {
            "lap_loss": lap_loss,
            "weighted_lap_loss": float(config.lambda_lap * lap_loss),
            "anchor_loss": anchor_loss,
            "weighted_anchor_loss": float(config.lambda_anchor * anchor_loss),
            "unseen_anchor_loss": unseen_loss,
            "weighted_unseen_anchor_loss": float(
                config.lambda_unseen_anchor * unseen_loss
            ),
        }
        return total, gradient, parts

    for step in range(0, config.num_iters + 1):
        total, grad, parts = loss_and_grad(vertices)
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
    mesh.ensure_normals()
    gt_mesh.ensure_normals()
    if mesh.num_vertices == gt_mesh.num_vertices and np.array_equal(mesh.faces, gt_mesh.faces):
        dots = np.einsum("ij,ij->i", mesh.normals, gt_mesh.normals)
        return float(np.mean(np.abs(np.clip(dots, -1.0, 1.0))))
    # Different-topology coarse/expanded evaluation uses nearest-surface normals
    # in both directions. Absolute cosine avoids conflating a local winding flip
    # with geometric normal disagreement; winding is reported separately by the
    # renderer-orientation diagnostics.
    return 0.5 * (
        _directed_nearest_normal_consistency(mesh, gt_mesh)
        + _directed_nearest_normal_consistency(gt_mesh, mesh)
    )


def _directed_nearest_normal_consistency(query: Mesh, surface_mesh: Mesh) -> float:
    from scipy.spatial import cKDTree

    query.ensure_normals()
    surface_mesh.ensure_normals()
    try:
        import trimesh

        surface = trimesh.Trimesh(
            vertices=surface_mesh.vertices,
            faces=surface_mesh.faces,
            process=False,
        )
        _, _, face_indices = trimesh.proximity.closest_point(surface, query.vertices)
        triangles = surface_mesh.vertices[surface_mesh.faces]
        face_normals = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        face_normals /= np.maximum(
            np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-12
        )
        matched_normals = face_normals[np.asarray(face_indices, dtype=np.int64)]
    except Exception:
        _, nearest = cKDTree(surface_mesh.vertices).query(query.vertices, workers=-1)
        matched_normals = surface_mesh.normals[np.asarray(nearest, dtype=np.int64)]
    dots = np.einsum("ij,ij->i", query.normals, matched_normals)
    return float(np.mean(np.abs(np.clip(dots, -1.0, 1.0))))


def _numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
