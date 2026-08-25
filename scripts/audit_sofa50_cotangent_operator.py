#!/usr/bin/env python3
from __future__ import annotations

"""Audit Cotangent construction and Uniform/Cotangent operator scale on Sofa50."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import svds

from mlr.learned_laplacian.cotangent_sparse_recovery import (
    build_symmetric_cotangent_stiffness,
)
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _topology_edges(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    directed.sort(axis=1)
    unique, counts = np.unique(directed, axis=0, return_counts=True)
    return unique, counts


def _uniform_matrix(static: dict[str, Any]) -> csr_matrix:
    vertices = int(np.asarray(static["vertices"]).shape[0])
    edge = np.asarray(static["edge_index"], dtype=np.int64)
    degree = np.asarray(static["vertex_degree"], dtype=np.float64).reshape(-1)
    source, destination = edge
    rows = np.concatenate((np.arange(vertices), destination))
    columns = np.concatenate((np.arange(vertices), source))
    values = np.concatenate((np.ones(vertices), -1.0 / degree[destination]))
    return coo_matrix((values, (rows, columns)), shape=(vertices, vertices)).tocsr()


def _cotangent_matrix(
    vertices: torch.Tensor, faces: torch.Tensor, epsilon: float
) -> tuple[csr_matrix, np.ndarray, dict[str, Any]]:
    edges, weights, diagonal, audit = build_symmetric_cotangent_stiffness(
        vertices, faces, relative_area_epsilon=epsilon
    )
    edge = edges.numpy()
    weight = weights.numpy()
    count = len(vertices)
    rows = np.concatenate((np.arange(count), edge[0], edge[1]))
    columns = np.concatenate((np.arange(count), edge[1], edge[0]))
    values = np.concatenate((diagonal.numpy(), -weight, -weight))
    matrix = coo_matrix((values, (rows, columns)), shape=(count, count)).tocsr()
    return matrix, weight, audit.__dict__


def _operator_row(
    split: str,
    sample_id: str,
    name: str,
    matrix: csr_matrix,
    components: int,
    regularization: float,
) -> dict[str, Any]:
    difference = matrix - matrix.T
    frobenius = float(np.sqrt(np.square(matrix.data).sum()))
    symmetry = float(np.sqrt(np.square(difference.data).sum()) / max(frobenius, 1e-300))
    row_sum = np.asarray(matrix.sum(axis=1)).reshape(-1)
    large_values: np.ndarray | None = None
    try:
        large_values = np.sort(
            svds(
                matrix,
                k=min(6, matrix.shape[0] - 1),
                which="LM",
                return_singular_vectors=False,
            )
        )
        largest = float(large_values[-1])
    except Exception:
        vector = np.random.default_rng(7).normal(size=matrix.shape[1])
        vector /= np.linalg.norm(vector)
        for _ in range(80):
            vector = matrix.T @ (matrix @ vector)
            vector /= max(np.linalg.norm(vector), 1e-300)
        largest = float(np.linalg.norm(matrix @ vector))
    diagonal = matrix.diagonal()
    nonzero_singular_floor = None
    small_values: np.ndarray | None = None
    if matrix.shape[0] <= 15000:
        try:
            values = np.sort(
                svds(
                    matrix,
                    k=min(max(components + 3, 4), matrix.shape[0] - 1),
                    which="SM",
                    return_singular_vectors=False,
                )
            )
            small_values = values
            positive = values[values > max(largest * 1e-10, 1e-12)]
            if positive.size:
                nonzero_singular_floor = float(positive[0])
        except Exception:
            pass
    return {
        "split": split,
        "sample_id": sample_id,
        "operator": name,
        "vertices": int(matrix.shape[0]),
        "nnz": int(matrix.nnz),
        "symmetry_relative_frobenius": symmetry,
        "constant_nullspace_max_abs": float(np.max(np.abs(row_sum), initial=0.0)),
        "frobenius_norm": frobenius,
        "operator_norm_estimate": largest,
        "smallest_estimated_nonzero_singular": nonzero_singular_floor,
        "diagonal_min": float(np.min(diagonal, initial=0.0)),
        "diagonal_median": float(np.median(diagonal)) if diagonal.size else 0.0,
        "diagonal_max": float(np.max(diagonal, initial=0.0)),
        "expected_nullity_components": components,
        "condition_estimate_normal_plus_lambda": float(
            (largest * largest + regularization) / regularization
        ),
        "normal_spectrum_max_estimate": largest * largest,
        "lambda_crossover_singular": float(np.sqrt(regularization)),
        "small_singular_values_json": (
            None if small_values is None else json.dumps(small_values.tolist())
        ),
        "large_singular_values_json": (
            None if large_values is None else json.dumps(large_values.tolist())
        ),
        "small_mode_direct_transfer_json": (
            None
            if small_values is None
            else json.dumps(
                (regularization / (np.square(small_values) + regularization)).tolist()
            )
        ),
        "large_mode_direct_transfer_json": (
            None
            if large_values is None
            else json.dumps(
                (regularization / (np.square(large_values) + regularization)).tolist()
            )
        ),
    }


def _mesh_row(
    split: str, static: dict[str, Any], epsilon: float
) -> tuple[dict[str, Any], np.ndarray, csr_matrix, csr_matrix]:
    vertices_np = np.asarray(static["vertices"], dtype=np.float64)
    faces_np = np.asarray(static["faces"], dtype=np.int64)
    vertices = torch.from_numpy(vertices_np)
    faces = torch.from_numpy(faces_np)
    topology_edges, incidence = _topology_edges(faces_np)
    u, v = topology_edges.T
    graph = coo_matrix(
        (
            np.ones(2 * len(topology_edges)),
            (np.concatenate((u, v)), np.concatenate((v, u))),
        ),
        shape=(len(vertices_np), len(vertices_np)),
    ).tocsr()
    component_count, labels = connected_components(graph, directed=False)
    degree = np.diff(graph.indptr)

    tri = vertices_np[faces_np]
    ab, ac, bc = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], tri[:, 2] - tri[:, 1]
    twice_area = np.linalg.norm(np.cross(ab, ac), axis=1)
    lengths = np.stack(
        (
            np.linalg.norm(ab, axis=1),
            np.linalg.norm(ac, axis=1),
            np.linalg.norm(bc, axis=1),
        ),
        axis=1,
    )
    max_edge_squared = np.square(lengths).max(axis=1)
    repeated = (
        (faces_np[:, 0] == faces_np[:, 1])
        | (faces_np[:, 1] == faces_np[:, 2])
        | (faces_np[:, 2] == faces_np[:, 0])
    )
    degenerate = repeated | (twice_area <= np.finfo(np.float64).tiny)
    near_zero = (~degenerate) & (twice_area <= 1e-10 * np.maximum(max_edge_squared, 1e-300))
    safe_area = np.maximum(twice_area, 1e-300)
    aspect = max_edge_squared / safe_area
    cosine_a = np.einsum("ij,ij->i", ab, ac) / np.maximum(lengths[:, 0] * lengths[:, 1], 1e-300)
    cosine_b = np.einsum("ij,ij->i", -ab, bc) / np.maximum(lengths[:, 0] * lengths[:, 2], 1e-300)
    angle_a = np.arccos(np.clip(cosine_a, -1.0, 1.0))
    angle_b = np.arccos(np.clip(cosine_b, -1.0, 1.0))
    angle_c = np.pi - angle_a - angle_b
    angles = np.stack((angle_a, angle_b, angle_c), axis=1)

    cotangent, weights, construction = _cotangent_matrix(vertices, faces, epsilon)
    uniform = _uniform_matrix(static)
    clean = np.asarray(static["clean_reference_vertices"], dtype=np.float64)
    correction = clean - vertices_np
    row = {
        "split": split,
        "sample_id": str(static["sample_id"]),
        "recipe": str(static["sample_id"]).rsplit("__", 1)[-1],
        "vertices": int(len(vertices_np)),
        "faces": int(len(faces_np)),
        "unique_topology_edges": int(len(topology_edges)),
        "boundary_edges": int((incidence == 1).sum()),
        "nonmanifold_edges": int((incidence > 2).sum()),
        "degenerate_triangles": int(degenerate.sum()),
        "near_zero_area_triangles_audit_1e-10": int(near_zero.sum()),
        "protected_triangles_relative_1e-12": int(construction["protected_triangles"]),
        "negative_cotangent_weights": int((weights < 0).sum()),
        "nonfinite_cotangent_weights": int((~np.isfinite(weights)).sum()),
        "maximum_absolute_cotangent_weight": float(np.max(np.abs(weights), initial=0.0)),
        "isolated_vertices": int((degree == 0).sum()),
        "components": int(component_count),
        "largest_component_fraction": float(np.max(np.bincount(labels)) / len(labels)),
        "mean_triangle_aspect_proxy": float(np.mean(aspect)),
        "p95_triangle_aspect_proxy": float(np.quantile(aspect, 0.95)),
        "minimum_angle_degrees": float(np.degrees(np.min(angles))),
        "maximum_angle_degrees": float(np.degrees(np.max(angles))),
        "obtuse_triangle_fraction": float(np.mean(np.max(angles, axis=1) > np.pi / 2)),
        "negative_weight_fraction": float(np.mean(weights < 0)) if weights.size else 0.0,
        "boundary_edge_fraction": float(np.mean(incidence == 1)) if incidence.size else 0.0,
        "nonmanifold_edge_fraction": float(np.mean(incidence > 2)) if incidence.size else 0.0,
        "correction_rms": float(np.sqrt(np.mean(np.sum(np.square(correction), axis=1)))),
    }
    return row, weights, uniform, cotangent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--representative-per-split", type=int, default=3)
    parser.add_argument("--lambda", dest="regularization", type=float, default=3e-2)
    parser.add_argument("--relative-area-epsilon", type=float, default=1e-12)
    args = parser.parse_args()
    if args.representative_per_split < 1 or args.regularization <= 0:
        raise ValueError("representative count and lambda must be positive.")

    mesh_rows: list[dict[str, Any]] = []
    operator_rows: list[dict[str, Any]] = []
    weight_arrays: list[np.ndarray] = []
    for split in ("train", "validation", "test"):
        dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), split)
        representatives = set(
            np.linspace(
                0, len(dataset) - 1, args.representative_per_split, dtype=int
            ).tolist()
        )
        for index in range(len(dataset)):
            static = dataset.load_static(index)
            row, weights, uniform, cotangent = _mesh_row(
                split, static, args.relative_area_epsilon
            )
            mesh_rows.append(row)
            weight_arrays.append(weights)
            if index in representatives:
                for name, matrix in (
                    ("uniform_random_walk", uniform),
                    ("symmetric_cotangent_stiffness", cotangent),
                ):
                    operator_rows.append(
                        _operator_row(
                            split,
                            row["sample_id"],
                            name,
                            matrix,
                            int(row["components"]),
                            args.regularization,
                        )
                    )

    all_weights = np.concatenate(weight_arrays) if weight_arrays else np.empty(0)
    percentile_points = [0, 1, 5, 25, 50, 75, 95, 99, 99.9, 100]
    total_faces = sum(int(row["faces"]) for row in mesh_rows)
    payload = {
        "contract_audit": all(
            int(row["nonfinite_cotangent_weights"]) == 0 for row in mesh_rows
        ),
        "operator_definition": {
            "uniform": "I-D^-1 A on the input undirected topology",
            "cotangent": "C_ij=-w_ij; C_ii=sum_j w_ij; w_ij=0.5 sum_incident cot(opposite_angle)",
            "boundary": "one-sided contribution only",
            "negative_weights": "retained",
            "mass_normalization": False,
            "geometry_source": "input coarse mesh vertices and faces",
            "near_degenerate_protection": (
                "triangle contributes zero iff twice_area <= 1e-12 * max_edge_squared"
            ),
            "nonmanifold_policy": "sum every actual incident-face contribution",
        },
        "meshes": len(mesh_rows),
        "splits": {
            split: sum(row["split"] == split for row in mesh_rows)
            for split in ("train", "validation", "test")
        },
        "vertices": {
            "total": sum(int(row["vertices"]) for row in mesh_rows),
            "min": min(int(row["vertices"]) for row in mesh_rows),
            "max": max(int(row["vertices"]) for row in mesh_rows),
        },
        "faces": {
            "total": total_faces,
            "min": min(int(row["faces"]) for row in mesh_rows),
            "max": max(int(row["faces"]) for row in mesh_rows),
        },
        "totals": {
            field: sum(int(row[field]) for row in mesh_rows)
            for field in (
                "boundary_edges",
                "nonmanifold_edges",
                "degenerate_triangles",
                "near_zero_area_triangles_audit_1e-10",
                "protected_triangles_relative_1e-12",
                "negative_cotangent_weights",
                "nonfinite_cotangent_weights",
                "isolated_vertices",
                "components",
            )
        },
        "protected_triangle_fraction": (
            sum(int(row["protected_triangles_relative_1e-12"]) for row in mesh_rows)
            / total_faces
        ),
        "maximum_absolute_cotangent_weight": float(
            np.max(np.abs(all_weights), initial=0.0)
        ),
        "cotangent_weight_percentiles": {
            str(point): float(np.percentile(all_weights, point))
            for point in percentile_points
        },
        "absolute_cotangent_weight_percentiles": {
            str(point): float(np.percentile(np.abs(all_weights), point))
            for point in percentile_points
        },
        "representative_operator_rows": len(operator_rows),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "mesh_audit.csv", mesh_rows)
    _write_csv(args.output_dir / "operator_representatives.csv", operator_rows)
    (args.output_dir / "audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not payload["contract_audit"]:
        raise RuntimeError("Cotangent construction produced non-finite weights.")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
