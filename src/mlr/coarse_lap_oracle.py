from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data import Array, Mesh
from .gt_laplacian import closest_points_on_mesh
from .io import save_mesh
from .laplacian import build_laplacian, unique_edges, vertex_neighbors


@dataclass(frozen=True)
class CoarseGraphOracleConfig:
    operator_type: str = "uniform"
    num_iters: int = 3000
    learning_rate: float = 5e-3
    lambda_lap: float = 1.0
    lambda_anchor: float = 0.01
    lambda_pos: float = 0.1
    lambda_edge: float = 0.0
    normalized_eps: float = 1e-8
    log_every: int = 25
    print_every: int = 0
    chamfer_samples: int = 5000
    seed: int = 7


@dataclass
class CoarseGraphTargets:
    projected_vertices: Array
    delta_target: Array
    h: Array
    delta_hat_target: Array
    projection_distances: Array
    laplacian_matrix: Array
    laplacian_data: "UniformLaplacianData"


@dataclass(frozen=True)
class UniformLaplacianData:
    rows: Array
    cols: Array
    weights: Array
    neighbors: list[set[int]]
    num_vertices: int


@dataclass
class OptimizationResult:
    name: str
    mesh: Mesh
    vertices: Array
    history: list[dict[str, float]]
    metrics: dict[str, Any]


def run_coarse_graph_laplacian_oracles(
    coarse_mesh: Mesh,
    gt_mesh: Mesh,
    output_dir: str | Path,
    config: CoarseGraphOracleConfig | None = None,
    previous_refined_mesh: Mesh | None = None,
    previous_history: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or CoarseGraphOracleConfig()
    if config.operator_type != "uniform":
        raise ValueError("coarse-graph oracle currently supports only uniform Laplacian.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = prepare_coarse_graph_targets(coarse_mesh, gt_mesh, config)
    projected_mesh = Mesh(targets.projected_vertices, coarse_mesh.faces).ensure_normals()
    save_mesh(projected_mesh, output_dir / "projected_gt_on_coarse.obj")
    np.save(output_dir / "delta_target.npy", targets.delta_target)
    np.save(output_dir / "h.npy", targets.h)
    np.save(output_dir / "delta_hat_target.npy", targets.delta_hat_target)

    before_surface = point_to_surface_stats(coarse_mesh.vertices, gt_mesh)
    before_chamfer = chamfer_distance(
        coarse_mesh,
        gt_mesh,
        samples=config.chamfer_samples,
        seed=config.seed,
    )
    target_norm = np.linalg.norm(targets.delta_target, axis=1)
    delta_hat_norm = np.linalg.norm(targets.delta_hat_target, axis=1)
    raw_variants = {
        "lap_anchor": {"lambda_anchor": config.lambda_anchor, "lambda_pos": 0.0, "lambda_edge": 0.0},
        "lap_pos": {"lambda_anchor": config.lambda_anchor, "lambda_pos": config.lambda_pos, "lambda_edge": 0.0},
        "lap_pos_edge": {
            "lambda_anchor": config.lambda_anchor,
            "lambda_pos": config.lambda_pos,
            "lambda_edge": config.lambda_edge,
        },
    }

    raw_results: dict[str, OptimizationResult] = {}
    for name, weights in raw_variants.items():
        result = optimize_uniform_laplacian_oracle(
            coarse_mesh=coarse_mesh,
            gt_mesh=gt_mesh,
            laplacian_data=targets.laplacian_data,
            delta_target=targets.delta_target,
            position_target=targets.projected_vertices,
            before_surface=before_surface,
            before_chamfer=before_chamfer,
            config=config,
            name=f"raw_{name}",
            **weights,
        )
        raw_results[name] = result
        save_mesh(result.mesh, output_dir / f"refined_raw_{name}.obj")
        write_history(output_dir / f"loss_curve_raw_{name}", result.history)

    selected_raw_name = "lap_pos_edge"
    selected_raw = raw_results[selected_raw_name]
    save_mesh(selected_raw.mesh, output_dir / "refined_raw.obj")
    raw_log = {
        **base_log(coarse_mesh, gt_mesh, config, targets, before_surface, before_chamfer),
        **selected_raw.metrics,
        "selected_ablation": selected_raw_name,
        "mean_target_lap_norm": float(np.mean(target_norm)),
        "max_target_lap_norm": float(np.max(target_norm)),
        "ablations": {name: result.metrics for name, result in raw_results.items()},
    }
    write_json(output_dir / "log_raw.json", raw_log)

    h2 = targets.h[:, None] ** 2 + float(config.normalized_eps)
    recovered_delta = targets.delta_hat_target * h2
    recover_error = np.max(np.abs(recovered_delta - targets.delta_target))
    normalized_results: dict[str, OptimizationResult] = {}
    for mode_name, delta_for_refine in {
        "recover": recovered_delta,
        "predict": targets.delta_hat_target * h2,
    }.items():
        result = optimize_uniform_laplacian_oracle(
            coarse_mesh=coarse_mesh,
            gt_mesh=gt_mesh,
            laplacian_data=targets.laplacian_data,
            delta_target=delta_for_refine,
            position_target=targets.projected_vertices,
            before_surface=before_surface,
            before_chamfer=before_chamfer,
            config=config,
            name=f"normalized_{mode_name}",
            lambda_anchor=config.lambda_anchor,
            lambda_pos=config.lambda_pos,
            lambda_edge=config.lambda_edge,
        )
        normalized_results[mode_name] = result
        save_mesh(result.mesh, output_dir / f"refined_normalized_{mode_name}.obj")
        write_history(output_dir / f"loss_curve_normalized_{mode_name}", result.history)

    normalized_selected = normalized_results["predict"]
    save_mesh(normalized_selected.mesh, output_dir / "refined_normalized.obj")
    normalized_log = {
        **base_log(coarse_mesh, gt_mesh, config, targets, before_surface, before_chamfer),
        **normalized_selected.metrics,
        "mean_h": float(np.mean(targets.h)),
        "median_h": float(np.median(targets.h)),
        "min_h": float(np.min(targets.h)),
        "max_h": float(np.max(targets.h)),
        "mean_raw_delta_norm": float(np.mean(target_norm)),
        "mean_normalized_delta_norm": float(np.mean(delta_hat_norm)),
        "max_raw_delta_norm": float(np.max(target_norm)),
        "max_normalized_delta_norm": float(np.max(delta_hat_norm)),
        "max_normalize_recover_abs_error": float(recover_error),
        "modes": {name: result.metrics for name, result in normalized_results.items()},
    }
    write_json(output_dir / "log_normalized.json", normalized_log)

    comparison = comparison_summary(
        coarse_mesh=coarse_mesh,
        gt_mesh=gt_mesh,
        before_surface=before_surface,
        before_chamfer=before_chamfer,
        raw_result=selected_raw,
        normalized_result=normalized_selected,
        previous_refined_mesh=previous_refined_mesh,
        previous_history=previous_history,
        config=config,
    )
    comparison["raw_ablations"] = {name: result.metrics for name, result in raw_results.items()}
    comparison["normalized_modes"] = {name: result.metrics for name, result in normalized_results.items()}
    comparison["normalized_recover_vs_raw_final_lap_loss_abs_diff"] = float(
        abs(
            normalized_results["recover"].metrics["final_lap_loss"]
            - selected_raw.metrics["final_lap_loss"]
        )
    )
    comparison["normalized_predict_vs_recover_final_vertex_rmse"] = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (normalized_results["predict"].vertices - normalized_results["recover"].vertices) ** 2,
                    axis=1,
                )
            )
        )
    )
    write_json(output_dir / "comparison_summary.json", comparison)
    write_json(output_dir / "config.json", config.__dict__)
    return comparison


def prepare_coarse_graph_targets(
    coarse_mesh: Mesh,
    gt_mesh: Mesh,
    config: CoarseGraphOracleConfig | None = None,
) -> CoarseGraphTargets:
    config = config or CoarseGraphOracleConfig()
    closest = closest_points_on_mesh(coarse_mesh.vertices, gt_mesh.vertices, gt_mesh.faces)
    projected = closest.points
    laplacian = build_laplacian(coarse_mesh.vertices, coarse_mesh.faces, config.operator_type)
    laplacian_data = build_uniform_laplacian_data(coarse_mesh.faces, coarse_mesh.num_vertices)
    delta_target = laplacian.matrix @ projected
    h = local_vertex_scales(coarse_mesh.vertices, laplacian_data)
    delta_hat = delta_target / (h[:, None] ** 2 + float(config.normalized_eps))

    assert projected.shape == coarse_mesh.vertices.shape
    assert delta_target.shape == coarse_mesh.vertices.shape
    assert h.shape[0] == coarse_mesh.vertices.shape[0]
    assert laplacian.matrix.shape == (coarse_mesh.num_vertices, coarse_mesh.num_vertices)

    return CoarseGraphTargets(
        projected_vertices=projected,
        delta_target=delta_target,
        h=h,
        delta_hat_target=delta_hat,
        projection_distances=closest.distances,
        laplacian_matrix=laplacian.matrix,
        laplacian_data=laplacian_data,
    )


def build_uniform_laplacian_data(faces: Array, num_vertices: int) -> UniformLaplacianData:
    neighbors = vertex_neighbors(faces, num_vertices)
    rows: list[int] = []
    cols: list[int] = []
    weights: list[float] = []
    for idx, nbrs in enumerate(neighbors):
        if not nbrs:
            continue
        weight = 1.0 / len(nbrs)
        for nbr in sorted(nbrs):
            rows.append(idx)
            cols.append(nbr)
            weights.append(weight)
    return UniformLaplacianData(
        rows=np.asarray(rows, dtype=np.int64),
        cols=np.asarray(cols, dtype=np.int64),
        weights=np.asarray(weights, dtype=np.float64),
        neighbors=neighbors,
        num_vertices=num_vertices,
    )


def local_vertex_scales(vertices: Array, laplacian_data: UniformLaplacianData) -> Array:
    vertices = np.asarray(vertices, dtype=np.float64)
    h_sum = np.zeros(laplacian_data.num_vertices, dtype=np.float64)
    h_count = np.zeros(laplacian_data.num_vertices, dtype=np.float64)
    if len(laplacian_data.rows) > 0:
        lengths = np.linalg.norm(vertices[laplacian_data.rows] - vertices[laplacian_data.cols], axis=1)
        np.add.at(h_sum, laplacian_data.rows, lengths)
        np.add.at(h_count, laplacian_data.rows, 1.0)
    h = h_sum / np.maximum(h_count, 1.0)
    valid = h_count > 0
    if np.any(valid):
        h[~valid] = float(np.mean(h[valid]))
    else:
        h[:] = 1.0
    return np.maximum(h, 1e-12)


def optimize_uniform_laplacian_oracle(
    coarse_mesh: Mesh,
    gt_mesh: Mesh,
    laplacian_data: UniformLaplacianData,
    delta_target: Array,
    position_target: Array,
    before_surface: dict[str, float],
    before_chamfer: float,
    config: CoarseGraphOracleConfig,
    name: str,
    lambda_anchor: float,
    lambda_pos: float,
    lambda_edge: float,
) -> OptimizationResult:
    vertices = np.array(coarse_mesh.vertices, dtype=np.float64, copy=True)
    anchors = np.asarray(coarse_mesh.vertices, dtype=np.float64)
    position_target = np.asarray(position_target, dtype=np.float64)
    delta_target = np.asarray(delta_target, dtype=np.float64)
    edge_pairs = unique_edges(coarse_mesh.faces)
    target_edge_lengths = edge_lengths(vertices, edge_pairs)
    history: list[dict[str, float]] = []
    m = np.zeros_like(vertices)
    v = np.zeros_like(vertices)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    def record(step: int, total: float, parts: dict[str, float]) -> None:
        history.append({"iter": float(step), "total_loss": float(total), **parts})

    total, grad, parts = oracle_loss_and_grad(
        vertices,
        laplacian_data,
        delta_target,
        anchors,
        position_target,
        edge_pairs,
        target_edge_lengths,
        config.lambda_lap,
        lambda_anchor,
        lambda_pos,
        lambda_edge,
    )
    record(0, total, parts)

    for step in range(1, config.num_iters + 1):
        total, grad, parts = oracle_loss_and_grad(
            vertices,
            laplacian_data,
            delta_target,
            anchors,
            position_target,
            edge_pairs,
            target_edge_lengths,
            config.lambda_lap,
            lambda_anchor,
            lambda_pos,
            lambda_edge,
        )
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1**step)
        v_hat = v / (1.0 - beta2**step)
        vertices -= config.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        if step == config.num_iters or step % config.log_every == 0:
            total_after, _, parts_after = oracle_loss_and_grad(
                vertices,
                laplacian_data,
                delta_target,
                anchors,
                position_target,
                edge_pairs,
                target_edge_lengths,
                config.lambda_lap,
                lambda_anchor,
                lambda_pos,
                lambda_edge,
            )
            record(step, total_after, parts_after)
        if config.print_every > 0 and (step == 1 or step == config.num_iters or step % config.print_every == 0):
            print(f"{name} step={step} total={total:.8f} lap={parts['lap_loss']:.8f}", flush=True)

    mesh = coarse_mesh.with_vertices(vertices)
    after_surface = point_to_surface_stats(vertices, gt_mesh)
    after_chamfer = chamfer_distance(mesh, gt_mesh, samples=config.chamfer_samples, seed=config.seed + 17)
    displacement = np.linalg.norm(vertices - coarse_mesh.vertices, axis=1)
    metrics = {
        "name": name,
        "lambda_lap": float(config.lambda_lap),
        "lambda_anchor": float(lambda_anchor),
        "lambda_pos": float(lambda_pos),
        "lambda_edge": float(lambda_edge),
        "initial_lap_loss": float(history[0]["lap_loss"]),
        "final_lap_loss": float(history[-1]["lap_loss"]),
        "initial_total_loss": float(history[0]["total_loss"]),
        "final_total_loss": float(history[-1]["total_loss"]),
        "point_to_surface_distance_before": float(before_surface["mean"]),
        "point_to_surface_distance_after": float(after_surface["mean"]),
        "point_to_surface_distance_before_max": float(before_surface["max"]),
        "point_to_surface_distance_after_max": float(after_surface["max"]),
        "chamfer_before": float(before_chamfer),
        "chamfer_after": float(after_chamfer),
        "mean_refined_displacement": float(np.mean(displacement)),
        "max_refined_displacement": float(np.max(displacement)),
    }
    return OptimizationResult(name=name, mesh=mesh, vertices=vertices, history=history, metrics=metrics)


def oracle_loss_and_grad(
    vertices: Array,
    laplacian_data: UniformLaplacianData,
    delta_target: Array,
    anchors: Array,
    position_target: Array,
    edge_pairs: Array,
    target_edge_lengths: Array,
    lambda_lap: float,
    lambda_anchor: float,
    lambda_pos: float,
    lambda_edge: float,
) -> tuple[float, Array, dict[str, float]]:
    vertices = np.asarray(vertices, dtype=np.float64)
    num_vertices = vertices.shape[0]
    grad = np.zeros_like(vertices)
    parts: dict[str, float] = {}
    total = 0.0

    lap_residual = apply_uniform_laplacian(vertices, laplacian_data) - delta_target
    lap_loss = float(np.mean(np.sum(lap_residual * lap_residual, axis=1)))
    grad += lambda_lap * (2.0 / num_vertices) * apply_uniform_laplacian_transpose(
        lap_residual,
        laplacian_data,
    )
    total += lambda_lap * lap_loss
    parts["lap_loss"] = lap_loss
    parts["weighted_lap_loss"] = float(lambda_lap * lap_loss)

    anchor_residual = vertices - anchors
    anchor_loss = float(np.mean(np.sum(anchor_residual * anchor_residual, axis=1)))
    if lambda_anchor > 0:
        grad += lambda_anchor * (2.0 / num_vertices) * anchor_residual
        total += lambda_anchor * anchor_loss
    parts["anchor_loss"] = anchor_loss
    parts["weighted_anchor_loss"] = float(lambda_anchor * anchor_loss)

    pos_residual = vertices - position_target
    pos_loss = float(np.mean(np.sum(pos_residual * pos_residual, axis=1)))
    if lambda_pos > 0:
        grad += lambda_pos * (2.0 / num_vertices) * pos_residual
        total += lambda_pos * pos_loss
    parts["pos_loss"] = pos_loss
    parts["weighted_pos_loss"] = float(lambda_pos * pos_loss)

    edge_loss = 0.0
    if lambda_edge > 0 and len(edge_pairs) > 0:
        edge_loss, edge_grad = edge_loss_and_grad(vertices, edge_pairs, target_edge_lengths)
        grad += lambda_edge * edge_grad
        total += lambda_edge * edge_loss
    parts["edge_loss"] = float(edge_loss)
    parts["weighted_edge_loss"] = float(lambda_edge * edge_loss)
    return float(total), grad, parts


def apply_uniform_laplacian(vertices: Array, data: UniformLaplacianData) -> Array:
    vertices = np.asarray(vertices, dtype=np.float64)
    neighbor_mean = np.zeros_like(vertices)
    if len(data.rows) > 0:
        np.add.at(neighbor_mean, data.rows, data.weights[:, None] * vertices[data.cols])
    return vertices - neighbor_mean


def apply_uniform_laplacian_transpose(residual: Array, data: UniformLaplacianData) -> Array:
    residual = np.asarray(residual, dtype=np.float64)
    grad = np.array(residual, dtype=np.float64, copy=True)
    if len(data.rows) > 0:
        np.add.at(grad, data.cols, -data.weights[:, None] * residual[data.rows])
    return grad


def edge_lengths(vertices: Array, edge_pairs: Array) -> Array:
    if len(edge_pairs) == 0:
        return np.zeros(0, dtype=np.float64)
    return np.linalg.norm(vertices[edge_pairs[:, 0]] - vertices[edge_pairs[:, 1]], axis=1)


def edge_loss_and_grad(vertices: Array, edge_pairs: Array, target_lengths: Array) -> tuple[float, Array]:
    if len(edge_pairs) == 0:
        return 0.0, np.zeros_like(vertices)
    diff = vertices[edge_pairs[:, 0]] - vertices[edge_pairs[:, 1]]
    lengths = np.linalg.norm(diff, axis=1)
    residual = lengths - target_lengths
    loss = float(np.mean(residual * residual))
    direction = diff / np.maximum(lengths[:, None], 1e-12)
    edge_grad_per_edge = (2.0 / len(edge_pairs)) * residual[:, None] * direction
    grad = np.zeros_like(vertices)
    np.add.at(grad, edge_pairs[:, 0], edge_grad_per_edge)
    np.add.at(grad, edge_pairs[:, 1], -edge_grad_per_edge)
    return loss, grad


def point_to_surface_stats(points: Array, surface_mesh: Mesh) -> dict[str, float]:
    closest = closest_points_on_mesh(points, surface_mesh.vertices, surface_mesh.faces)
    distances = closest.distances
    return {
        "mean": float(np.mean(distances)),
        "rmse": float(np.sqrt(np.mean(distances * distances))),
        "median": float(np.median(distances)),
        "max": float(np.max(distances)),
    }


def chamfer_distance(mesh: Mesh, gt_mesh: Mesh, samples: int = 5000, seed: int = 7) -> float:
    coarse_to_gt = point_to_surface_stats(mesh.vertices, gt_mesh)["mean"]
    gt_vertices = gt_mesh.vertices
    if samples > 0 and samples < gt_mesh.num_vertices:
        rng = np.random.default_rng(seed)
        indices = rng.choice(gt_mesh.num_vertices, size=samples, replace=False)
        gt_vertices = gt_vertices[indices]
    gt_to_mesh = point_to_surface_stats(gt_vertices, mesh)["mean"]
    return 0.5 * (coarse_to_gt + gt_to_mesh)


def base_log(
    coarse_mesh: Mesh,
    gt_mesh: Mesh,
    config: CoarseGraphOracleConfig,
    targets: CoarseGraphTargets,
    before_surface: dict[str, float],
    before_chamfer: float,
) -> dict[str, Any]:
    return {
        "coarse_vertices": int(coarse_mesh.num_vertices),
        "coarse_faces": int(coarse_mesh.num_faces),
        "gt_vertices": int(gt_mesh.num_vertices),
        "gt_faces": int(gt_mesh.num_faces),
        "operator": config.operator_type,
        "mean_projection_distance": float(np.mean(targets.projection_distances)),
        "max_projection_distance": float(np.max(targets.projection_distances)),
        "point_to_surface_distance_before": float(before_surface["mean"]),
        "point_to_surface_distance_before_max": float(before_surface["max"]),
        "chamfer_before": float(before_chamfer),
        "chamfer_samples": int(config.chamfer_samples),
    }


def comparison_summary(
    coarse_mesh: Mesh,
    gt_mesh: Mesh,
    before_surface: dict[str, float],
    before_chamfer: float,
    raw_result: OptimizationResult,
    normalized_result: OptimizationResult,
    previous_refined_mesh: Mesh | None,
    previous_history: dict[str, Any] | None,
    config: CoarseGraphOracleConfig,
) -> dict[str, Any]:
    rows: dict[str, Any] = {
        "coarse_original": {
            "vertices": int(coarse_mesh.num_vertices),
            "faces": int(coarse_mesh.num_faces),
            "point_to_surface_distance": float(before_surface["mean"]),
            "point_to_surface_distance_max": float(before_surface["max"]),
            "chamfer": float(before_chamfer),
            "mean_displacement": 0.0,
            "max_displacement": 0.0,
        },
        "raw_oracle_refined": raw_result.metrics,
        "normalized_oracle_refined": normalized_result.metrics,
    }
    if previous_refined_mesh is not None:
        previous_surface = point_to_surface_stats(previous_refined_mesh.vertices, gt_mesh)
        previous_chamfer = chamfer_distance(
            previous_refined_mesh,
            gt_mesh,
            samples=config.chamfer_samples,
            seed=config.seed + 31,
        )
        displacement = np.linalg.norm(previous_refined_mesh.vertices - coarse_mesh.vertices, axis=1)
        previous = {
            "point_to_surface_distance": float(previous_surface["mean"]),
            "point_to_surface_distance_max": float(previous_surface["max"]),
            "chamfer": float(previous_chamfer),
            "mean_displacement": float(np.mean(displacement)),
            "max_displacement": float(np.max(displacement)),
        }
        if previous_history and "summary" in previous_history:
            previous["previous_highres_transfer_summary"] = previous_history["summary"]
        rows["previous_highres_gt_laplacian_transfer"] = previous
    return {
        "comparison": rows,
        "judgement": {
            "raw_improves_point_to_surface_over_coarse": bool(
                raw_result.metrics["point_to_surface_distance_after"] < before_surface["mean"]
            ),
            "normalized_matches_raw_lap_loss": bool(
                abs(normalized_result.metrics["final_lap_loss"] - raw_result.metrics["final_lap_loss"]) < 1e-8
            ),
        },
    }


def write_history(path_without_suffix: Path, history: list[dict[str, float]]) -> None:
    write_json(path_without_suffix.with_suffix(".json"), history)
    if not history:
        return
    with path_without_suffix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value
