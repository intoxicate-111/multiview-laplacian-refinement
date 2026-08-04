#!/usr/bin/env python3
"""Measure global-scale laws and Bunny cross-resolution target distributions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mlr.coarse_lap_oracle import apply_uniform_laplacian, build_uniform_laplacian_data
from mlr.data import Mesh
from mlr.io import load_mesh, save_mesh
from mlr.learned_laplacian.graph_layers import faces_to_edge_index
from mlr.learned_laplacian.target_scaling import (
    EDGE_SCALE_DEFINITION,
    EDGE_SCALE_SOURCE,
    edge_scale_statistics,
    mean_incident_edge_length,
    normalize_laplacian_by_edge_scale,
    vector_magnitude_statistics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine-mesh", type=Path, required=True)
    parser.add_argument("--coarse-mesh", type=Path)
    parser.add_argument("--coarse-face-count", type=int, default=7000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epsilon", type=float, default=1e-12)
    parser.add_argument("--scales", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epsilon <= 0:
        raise ValueError("--epsilon must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fine = load_mesh(args.fine_mesh)
    if args.coarse_mesh is None:
        coarse = _simplify_mesh(fine, args.coarse_face_count)
        save_mesh(coarse, args.output_dir / "simplified_coarse.obj")
        coarse_source = {
            "kind": "Open3D quadric simplification of the fine mesh",
            "requested_face_count": args.coarse_face_count,
        }
    else:
        coarse = load_mesh(args.coarse_mesh)
        coarse_source = {"kind": "mesh file", "path": str(args.coarse_mesh)}

    global_scaling = _global_scaling_diagnostic(
        fine.vertices, fine.faces, args.scales, args.epsilon
    )
    cross_resolution, arrays = _cross_resolution_diagnostic(
        fine.vertices,
        fine.faces,
        coarse.vertices,
        coarse.faces,
        args.epsilon,
    )
    report = {
        "definition": {
            "h_i": "mean length of unique undirected edges incident to vertex i",
            "scale_i": "h_i^2 (square of the mean, not mean squared edge length)",
            "normalized_target": "delta_hat_i = delta_i / (h_i^2 + epsilon)",
            "denormalization_for_solver": "delta_pred_i = delta_hat_pred_i * h_i^2",
            "edge_scale_definition": EDGE_SCALE_DEFINITION,
            "edge_scale_source": EDGE_SCALE_SOURCE,
            "epsilon": args.epsilon,
        },
        "global_scaling": global_scaling,
        "cross_resolution": {"coarse_source": coarse_source, **cross_resolution},
    }
    with (args.output_dir / "diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    np.savez_compressed(args.output_dir / "cross_resolution_arrays.npz", **arrays)
    print(json.dumps(report, indent=2, sort_keys=True))


def _global_scaling_diagnostic(
    vertices: np.ndarray,
    faces: np.ndarray,
    scales: list[float],
    epsilon: float,
) -> dict[str, Any]:
    if any(scale <= 0 for scale in scales):
        raise ValueError("all --scales values must be positive")
    data = build_uniform_laplacian_data(faces, len(vertices))
    base_h, base_delta, base_hat = _targets(vertices, faces, data, epsilon)
    active = base_h > 0
    base_all = _target_means(base_h, base_delta, base_hat, torch.ones_like(active))
    base_active = _target_means(base_h, base_delta, base_hat, active)
    cases: list[dict[str, Any]] = []
    for scale in scales:
        h, delta, delta_hat = _targets(vertices * scale, faces, data, epsilon)
        measured_all = _ratios(_target_means(h, delta, delta_hat, torch.ones_like(active)), base_all)
        measured_active = _ratios(_target_means(h, delta, delta_hat, active), base_active)
        expected = {
            "h_ratio": scale,
            "h2_ratio": scale**2,
            "delta_norm_ratio": scale,
            "delta_hat_norm_ratio": 1.0 / scale,
        }
        cases.append(
            {
                "scale": scale,
                "expected_ratios": expected,
                "measured_ratios_nonisolated": measured_active,
                "measured_ratios_all_vertices": measured_all,
                "maximum_absolute_ratio_error": max(
                    abs(measured_active[name] - expected[name]) for name in expected
                ),
            }
        )
    return {
        "vertex_count": len(vertices),
        "face_count": len(faces),
        "isolated_vertices": int((~active).sum()),
        "base_means_nonisolated": base_active,
        "base_means_all_vertices": base_all,
        "cases": cases,
        "interpretation": (
            "For a first-order uniform Laplacian, global coordinate scaling by a makes delta scale "
            "by a and h^2 by a^2, so delta_hat scales by 1/a. This target is therefore not "
            "globally scale invariant. Ratios are checked on nonisolated vertices. Isolated "
            "vertices have h=0 and are epsilon-dominated, so their normalized values instead "
            "scale with delta."
        ),
    }


def _cross_resolution_diagnostic(
    fine_vertices: np.ndarray,
    fine_faces: np.ndarray,
    coarse_vertices: np.ndarray,
    coarse_faces: np.ndarray,
    epsilon: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    fine_data = build_uniform_laplacian_data(fine_faces, len(fine_vertices))
    fine_h, fine_delta, fine_hat = _targets(fine_vertices, fine_faces, fine_data, epsilon)

    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError(
            "Cross-resolution projection requires the optional Bunny dependencies: "
            "pip install -e '.[train,bunny]'"
        ) from exc
    surface = trimesh.Trimesh(vertices=fine_vertices, faces=fine_faces, process=False)
    projected, distances, _ = trimesh.proximity.closest_point(surface, coarse_vertices)
    projected = np.asarray(projected, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    coarse_data = build_uniform_laplacian_data(coarse_faces, len(coarse_vertices))
    coarse_h, coarse_delta, coarse_hat = _targets(
        coarse_vertices, coarse_faces, coarse_data, epsilon, target_positions=projected
    )

    fine_active = fine_h > 0
    coarse_active = coarse_h > 0
    fine_raw_stats = vector_magnitude_statistics(fine_delta[fine_active])
    fine_hat_stats = vector_magnitude_statistics(fine_hat[fine_active])
    coarse_raw_stats = vector_magnitude_statistics(coarse_delta[coarse_active])
    coarse_hat_stats = vector_magnitude_statistics(coarse_hat[coarse_active])
    raw_discrepancy = _distribution_discrepancy(fine_raw_stats, coarse_raw_stats)
    normalized_discrepancy = _distribution_discrepancy(fine_hat_stats, coarse_hat_stats)
    comparison = {
        "fine": _resolution_report(fine_h, fine_delta, fine_hat),
        "coarse": _resolution_report(coarse_h, coarse_delta, coarse_hat),
        "coarse_to_fine_projection_distance": _scalar_statistics(distances),
        "distribution_log_ratio_discrepancy": {
            "definition": (
                "nonisolated vertices; mean absolute log(coarse/fine) over median, mean, "
                "and p95 magnitudes"
            ),
            "raw": raw_discrepancy,
            "normalized": normalized_discrepancy,
            "normalized_minus_raw": normalized_discrepancy - raw_discrepancy,
            "normalized_is_closer": normalized_discrepancy < raw_discrepancy,
        },
    }
    arrays = {
        "fine_h": fine_h.numpy(),
        "fine_delta_norm": torch.linalg.vector_norm(fine_delta, dim=-1).numpy(),
        "fine_delta_hat_norm": torch.linalg.vector_norm(fine_hat, dim=-1).numpy(),
        "coarse_h": coarse_h.numpy(),
        "coarse_delta_norm": torch.linalg.vector_norm(coarse_delta, dim=-1).numpy(),
        "coarse_delta_hat_norm": torch.linalg.vector_norm(coarse_hat, dim=-1).numpy(),
        "coarse_projection_distance": distances,
    }
    return comparison, arrays


def _targets(
    graph_vertices: np.ndarray,
    faces: np.ndarray,
    data: Any,
    epsilon: float,
    target_positions: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    graph_vertices_t = torch.as_tensor(graph_vertices, dtype=torch.float64)
    faces_t = torch.as_tensor(faces, dtype=torch.long)
    edge_index = faces_to_edge_index(faces_t, len(graph_vertices_t))
    h = mean_incident_edge_length(graph_vertices_t, edge_index)
    positions = graph_vertices if target_positions is None else target_positions
    delta = torch.from_numpy(apply_uniform_laplacian(positions, data))
    delta_hat = normalize_laplacian_by_edge_scale(delta, h, eps=epsilon)
    return h, delta, delta_hat


def _resolution_report(
    h: torch.Tensor, delta: torch.Tensor, delta_hat: torch.Tensor
) -> dict[str, Any]:
    h2 = h.square()
    raw_norm = torch.linalg.vector_norm(delta, dim=-1)
    hat_norm = torch.linalg.vector_norm(delta_hat, dim=-1)
    active = h > 0
    return {
        "vertex_count": len(h),
        "edge_scale": edge_scale_statistics(h),
        "raw_target_magnitude_all_vertices": vector_magnitude_statistics(delta),
        "normalized_target_magnitude_all_vertices": vector_magnitude_statistics(delta_hat),
        "raw_target_magnitude_nonisolated": vector_magnitude_statistics(delta[active]),
        "normalized_target_magnitude_nonisolated": vector_magnitude_statistics(delta_hat[active]),
        "correlations": {
            "scope": "nonisolated vertices",
            "pearson_log_h2_vs_log_raw_magnitude": _log_correlation(h2[active], raw_norm[active]),
            "pearson_log_h2_vs_log_normalized_magnitude": _log_correlation(
                h2[active], hat_norm[active]
            ),
        },
    }


def _target_means(
    h: torch.Tensor, delta: torch.Tensor, delta_hat: torch.Tensor, mask: torch.Tensor
) -> dict[str, float]:
    return {
        "h_mean": float(h[mask].mean()),
        "h2_mean": float(h[mask].square().mean()),
        "delta_norm_mean": float(torch.linalg.vector_norm(delta[mask], dim=-1).mean()),
        "delta_hat_norm_mean": float(torch.linalg.vector_norm(delta_hat[mask], dim=-1).mean()),
    }


def _ratios(values: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
    return {
        "h_ratio": _safe_ratio(values["h_mean"], reference["h_mean"]),
        "h2_ratio": _safe_ratio(values["h2_mean"], reference["h2_mean"]),
        "delta_norm_ratio": _safe_ratio(
            values["delta_norm_mean"], reference["delta_norm_mean"]
        ),
        "delta_hat_norm_ratio": _safe_ratio(
            values["delta_hat_norm_mean"], reference["delta_hat_norm_mean"]
        ),
    }


def _simplify_mesh(mesh: Mesh, face_count: int) -> Mesh:
    if face_count < 4 or face_count >= mesh.num_faces:
        raise ValueError("--coarse-face-count must be at least 4 and smaller than the fine mesh")
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "Generating a same-surface coarse mesh requires Open3D; alternatively pass "
            "--coarse-mesh."
        ) from exc
    source = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces)
    )
    simplified = source.simplify_quadric_decimation(target_number_of_triangles=face_count)
    return Mesh(
        np.asarray(simplified.vertices, dtype=np.float64),
        np.asarray(simplified.triangles, dtype=np.int64),
    ).ensure_normals()


def _distribution_discrepancy(a: dict[str, float], b: dict[str, float]) -> float:
    values = []
    for name in ("median", "mean", "p95"):
        values.append(abs(math.log(max(b[name], 1e-300) / max(a[name], 1e-300))))
    return float(np.mean(values))


def _log_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    x = torch.log(a.detach().double().cpu().clamp_min(1e-300)).numpy()
    y = torch.log(b.detach().double().cpu().clamp_min(1e-300)).numpy()
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _scalar_statistics(values: np.ndarray) -> dict[str, float]:
    return {
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
    }


def _safe_ratio(value: float, reference: float) -> float:
    return value / max(reference, 1e-300)


if __name__ == "__main__":
    main()
