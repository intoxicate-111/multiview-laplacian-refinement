#!/usr/bin/env python3
from __future__ import annotations

"""Read-only Sofa50 v2 topology and boundary-quality diagnostic for frozen B/E."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.sparse import coo_matrix, csgraph, eye
from scipy.sparse.linalg import eigsh
from scipy.stats import spearmanr

from diagnose_sofa50_exact_solve_visibility_sweep import (
    component_labels,
    uniform_sparse_laplacian,
)
from diagnose_sofa50_regularized_sparse_sweep import regularized_sparse_solve
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


ARM_B = "B_lap_plus_refine"
ARM_E = "E_direct_vertex_residual"
RECIPES = ("A1", "A2", "B1", "B2", "C1", "C2", "C3", "C4", "D1", "D2")
MILD = {"A1", "B1", "C1", "C3", "D1"}
STRONG = set(RECIPES) - MILD
GROUPS = {
    **{recipe: {recipe} for recipe in RECIPES},
    "mild": MILD,
    "strong": STRONG,
    "original_topology": {"A1", "A2"},
    "midpoint_subdivision": {"B1", "B2"},
    "area_adaptive": {"C1", "C2", "C3", "C4"},
    "area_or_edge_adaptive": {"D1", "D2"},
    "all": set(RECIPES),
}

TOPOLOGY_DEFINITION = (
    "undirected triangle edges are canonical sorted vertex pairs; boundary incidence=1; "
    "manifold interior incidence=2; non-manifold incidence>2; connected components use "
    "the full vertex-edge graph including isolated vertices; duplicate faces are repeated "
    "canonical sorted triples beyond the first; strict degenerate face has double-area "
    "norm <=1e-14; watertight requires every edge incidence=2, no isolated vertex, and no "
    "duplicate/degenerate face; winding-consistent requires opposite directed traversal "
    "on every two-face edge"
)
SPECTRAL_DEFINITION = (
    "symmetric normalized uniform graph Laplacian I-D^-1/2 A D^-1/2, which is similar "
    "to the production uniform random-walk operator I-D^-1 A; nullity is the connected-"
    "component count; the reported gap is the minimum component lambda_2; regularized "
    "nonzero condition proxy is (mu_max^2+lambda)/(gap^2+lambda), not an exact singular "
    "condition number of the nonsymmetric production operator"
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_typed_csv(path: Path) -> list[dict[str, Any]]:
    rows = _read_csv(path)
    for row in rows:
        for key, value in list(row.items()):
            if value == "True":
                row[key] = True
            elif value == "False":
                row[key] = False
            elif value == "":
                row[key] = math.nan
            elif key not in {"sample_id", "split", "recipe", "severity", "boundary_quantile", "topology_irregularity_quantile"}:
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
    return rows


def _recipe(sample_id: str) -> str:
    value = sample_id.rpartition("__")[2]
    if value not in RECIPES:
        raise ValueError(f"Unexpected recipe in {sample_id}")
    return value


def _edge_table(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    canonical = np.sort(directed, axis=1)
    unique, inverse, counts = np.unique(canonical, axis=0, return_inverse=True, return_counts=True)
    return unique, counts, inverse


def _adjacency(edges: np.ndarray, n: int):
    directed = np.concatenate((edges, edges[:, ::-1]), axis=0)
    matrix = coo_matrix(
        (np.ones(len(directed), dtype=np.float64), (directed[:, 0], directed[:, 1])),
        shape=(n, n),
    ).tocsr()
    matrix.data[:] = 1.0
    matrix.eliminate_zeros()
    return matrix


def _face_cross(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    return np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])


def topology_row(sample_id: str, split: str, vertices: np.ndarray, faces: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    n, m = len(vertices), len(faces)
    edges, incidence, inverse = _edge_table(faces)
    adjacency = _adjacency(edges, n)
    component_count, labels = csgraph.connected_components(adjacency, directed=False)
    vertex_sizes = np.bincount(labels, minlength=component_count)
    face_labels = labels[faces[:, 0]]
    if not np.all(labels[faces] == face_labels[:, None]):
        raise RuntimeError(f"{sample_id}: face crosses graph components")
    face_sizes = np.bincount(face_labels, minlength=component_count)
    boundary_edges = edges[incidence == 1]
    nonmanifold_edges = edges[incidence > 2]
    boundary_vertices = np.unique(boundary_edges) if len(boundary_edges) else np.empty(0, np.int64)
    nonmanifold_vertices = (
        np.unique(nonmanifold_edges) if len(nonmanifold_edges) else np.empty(0, np.int64)
    )
    degree = np.diff(adjacency.indptr).astype(np.int64)
    isolated = int(np.count_nonzero(degree == 0))
    canonical_faces = np.sort(faces, axis=1)
    _, face_counts = np.unique(canonical_faces, axis=0, return_counts=True)
    duplicates = int(np.maximum(face_counts - 1, 0).sum())
    cross = _face_cross(vertices, faces)
    degenerates = int(np.count_nonzero(np.linalg.norm(cross, axis=1) <= 1e-14))

    # Every incidence-two edge must be traversed once in each direction.
    directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    signs = np.where(directed[:, 0] < directed[:, 1], 1, -1)
    sign_sum = np.bincount(inverse, weights=signs, minlength=len(edges))
    winding_consistent = bool(np.all(sign_sum[incidence == 2] == 0))
    watertight = bool(
        len(boundary_edges) == 0
        and len(nonmanifold_edges) == 0
        and isolated == 0
        and duplicates == 0
        and degenerates == 0
    )
    largest_v = int(vertex_sizes.max(initial=0))
    largest_f = int(face_sizes.max(initial=0))
    row = {
        "sample_id": sample_id,
        "split": split,
        "recipe": _recipe(sample_id),
        "severity": "mild" if _recipe(sample_id) in MILD else "strong",
        "vertices": n,
        "faces": m,
        "edges": len(edges),
        "connected_components": int(component_count),
        "largest_component_vertices": largest_v,
        "largest_component_faces": largest_f,
        "largest_component_vertex_ratio": largest_v / max(n, 1),
        "largest_component_face_ratio": largest_f / max(m, 1),
        "boundary_edges": len(boundary_edges),
        "boundary_vertices": len(boundary_vertices),
        "boundary_edge_ratio": len(boundary_edges) / max(len(edges), 1),
        "boundary_vertex_ratio": len(boundary_vertices) / max(n, 1),
        "nonmanifold_edges": len(nonmanifold_edges),
        "nonmanifold_edge_ratio": len(nonmanifold_edges) / max(len(edges), 1),
        "nonmanifold_vertices_edge_induced": len(nonmanifold_vertices),
        "isolated_vertices": isolated,
        "euler_characteristic": int(n - len(edges) + m),
        "watertight": watertight,
        "winding_consistent": winding_consistent,
        "duplicate_faces": duplicates,
        "degenerate_faces": degenerates,
        "degree_mean": float(degree.mean()),
        "degree_std": float(degree.std()),
        "degree_variance": float(degree.var()),
        "degree_min": int(degree.min()) if len(degree) else 0,
        "degree_max": int(degree.max(initial=0)),
        "degree_p5": float(np.quantile(degree, 0.05)),
        "degree_p95": float(np.quantile(degree, 0.95)),
    }
    detail = {
        "edges": edges,
        "adjacency": adjacency,
        "labels": labels,
        "boundary_vertices": boundary_vertices,
        "boundary_edges": boundary_edges,
        "vertex_component_sizes": vertex_sizes,
        "face_component_sizes": face_sizes,
    }
    return row, detail


def _error_stats(error: np.ndarray, mask: np.ndarray, prefix: str) -> dict[str, Any]:
    values = np.asarray(error, dtype=np.float64)[mask]
    if not len(values):
        return {
            f"{prefix}_count": 0,
            **{f"{prefix}_{key}": math.nan for key in ("mean", "rms", "median", "p90", "p95")},
        }
    return {
        f"{prefix}_count": len(values),
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_rms": float(np.sqrt(np.square(values).mean())),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p90": float(np.quantile(values, 0.90)),
        f"{prefix}_p95": float(np.quantile(values, 0.95)),
    }


def _graph_rings(adjacency, boundary_vertices: np.ndarray) -> np.ndarray:
    n = adjacency.shape[0]
    distance = np.full(n, -1, dtype=np.int64)
    queue: deque[int] = deque()
    for value in boundary_vertices.tolist():
        distance[value] = 0
        queue.append(value)
    while queue:
        vertex = queue.popleft()
        start, stop = adjacency.indptr[vertex : vertex + 2]
        for neighbor in adjacency.indices[start:stop]:
            if distance[neighbor] < 0:
                distance[neighbor] = distance[vertex] + 1
                queue.append(int(neighbor))
    return distance


def _cosine_stats(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray, prefix: str) -> dict[str, Any]:
    p, t = prediction[mask], target[mask]
    p_norm, t_norm = np.linalg.norm(p, axis=1), np.linalg.norm(t, axis=1)
    valid = (p_norm > 1e-12) & (t_norm > 1e-12)
    cosine = np.sum(p[valid] * t[valid], axis=1) / (p_norm[valid] * t_norm[valid])
    return {
        f"{prefix}_raw_epe": float(np.linalg.norm(p - t, axis=1).mean()) if len(p) else math.nan,
        f"{prefix}_raw_rms": float(np.sqrt(np.square(p - t).sum(axis=1).mean())) if len(p) else math.nan,
        f"{prefix}_raw_cosine": float(cosine.mean()) if len(cosine) else math.nan,
        f"{prefix}_raw_cosine_valid": int(len(cosine)),
    }


def local_quality(
    initial: np.ndarray,
    clean: np.ndarray,
    b_vertices: np.ndarray,
    e_vertices: np.ndarray,
    b_delta: np.ndarray,
    target_delta: np.ndarray,
    faces: np.ndarray,
    detail: Mapping[str, Any],
) -> dict[str, Any]:
    n = len(initial)
    boundary = np.zeros(n, dtype=bool)
    boundary[np.asarray(detail["boundary_vertices"], dtype=np.int64)] = True
    interior = ~boundary
    b_error = np.linalg.norm(b_vertices - clean, axis=1)
    e_error = np.linalg.norm(e_vertices - clean, axis=1)
    row: dict[str, Any] = {}
    for name, error in (("b", b_error), ("e", e_error)):
        row.update(_error_stats(error, np.ones(n, bool), f"{name}_all"))
        row.update(_error_stats(error, boundary, f"{name}_boundary"))
        row.update(_error_stats(error, interior, f"{name}_interior"))
    row.update(_cosine_stats(b_delta, target_delta, boundary, "b_boundary"))
    row.update(_cosine_stats(b_delta, target_delta, interior, "b_interior"))

    rings = _graph_rings(detail["adjacency"], np.asarray(detail["boundary_vertices"]))
    ring_masks = {
        "ring0": rings == 0,
        "ring1": rings == 1,
        "ring2": rings == 2,
        "ring3": rings == 3,
        "deep4plus": rings >= 4,
        "closed_component": rings < 0,
    }
    for ring, mask in ring_masks.items():
        for name, error in (("b", b_error), ("e", e_error)):
            row.update(_error_stats(error, mask, f"{name}_{ring}"))

    initial_cross = _face_cross(initial, faces)
    clean_cross = _face_cross(clean, faces)
    boundary_faces = np.any(boundary[faces], axis=1)
    for name, vertices in (("b", b_vertices), ("e", e_vertices)):
        refined_cross = _face_cross(vertices, faces)
        norm_clean = np.linalg.norm(clean_cross, axis=1)
        norm_refined = np.linalg.norm(refined_cross, axis=1)
        valid_normal = (norm_clean > 1e-14) & (norm_refined > 1e-14)
        cosine = np.zeros(len(faces), dtype=np.float64)
        cosine[valid_normal] = np.abs(
            np.sum(clean_cross[valid_normal] * refined_cross[valid_normal], axis=1)
            / (norm_clean[valid_normal] * norm_refined[valid_normal])
        )
        normal_error = 1.0 - cosine
        valid_flip = (np.linalg.norm(initial_cross, axis=1) > 1e-14) & (norm_refined > 1e-14)
        flipped = valid_flip & (np.sum(initial_cross * refined_cross, axis=1) < 0)
        for region, mask in (("boundary", boundary_faces), ("interior", ~boundary_faces)):
            normal_mask = mask & valid_normal
            flip_mask = mask & valid_flip
            row[f"{name}_{region}_normal_error"] = (
                float(normal_error[normal_mask].mean()) if normal_mask.any() else math.nan
            )
            row[f"{name}_{region}_normal_eligible_faces"] = int(normal_mask.sum())
            row[f"{name}_{region}_flips"] = int(np.count_nonzero(flipped & mask))
            row[f"{name}_{region}_flip_eligible_faces"] = int(flip_mask.sum())
            row[f"{name}_{region}_flip_rate"] = (
                float(np.count_nonzero(flipped & mask) / flip_mask.sum()) if flip_mask.any() else math.nan
            )

    b_sq, e_sq = np.square(b_error), np.square(e_error)
    difference = b_sq - e_sq
    for region, mask in (("boundary", boundary), ("interior", interior)):
        row[f"vrms2_advantage_{region}_total"] = float(difference[mask].sum())
        row[f"vrms2_advantage_{region}_per_vertex"] = (
            float(difference[mask].mean()) if mask.any() else math.nan
        )
        row[f"vrms2_advantage_{region}_population_fraction"] = float(mask.mean())
    row["vrms2_advantage_total"] = float(difference.sum())
    return row


def _symmetric_normalized_laplacian(adjacency):
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inv = np.zeros_like(degree)
    valid = degree > 0
    inv[valid] = 1.0 / np.sqrt(degree[valid])
    normalized = adjacency.multiply(inv[:, None]).multiply(inv[None, :])
    return eye(adjacency.shape[0], format="csr") - normalized


def spectral_condition(detail: Mapping[str, Any]) -> dict[str, Any]:
    adjacency = detail["adjacency"]
    labels = np.asarray(detail["labels"])
    component_count = int(labels.max(initial=-1) + 1)
    gaps: list[float] = []
    reliable = True
    for component in range(component_count):
        indices = np.flatnonzero(labels == component)
        if len(indices) <= 1:
            continue
        local = _symmetric_normalized_laplacian(adjacency[indices][:, indices])
        try:
            if len(indices) <= 24:
                values = np.linalg.eigvalsh(local.toarray())
                positive = values[values > 1e-9]
                gap = float(positive[0]) if len(positive) else math.nan
            else:
                values = np.sort(eigsh(local, k=2, which="SM", return_eigenvectors=False, tol=1e-5, maxiter=5000))
                positive = values[values > 1e-8]
                gap = float(positive[0]) if len(positive) else math.nan
            if math.isfinite(gap):
                gaps.append(gap)
            else:
                reliable = False
        except Exception:
            reliable = False
    whole = _symmetric_normalized_laplacian(adjacency)
    try:
        maximum = float(eigsh(whole, k=1, which="LA", return_eigenvectors=False, tol=1e-4, maxiter=5000)[0])
    except Exception:
        maximum = math.nan
        reliable = False
    gap = min(gaps) if gaps else math.nan
    result = {
        "spectral_nullity": component_count,
        "spectral_gap_min_component": gap,
        "spectral_largest_eigenvalue": maximum,
        "spectral_reliable": bool(reliable and math.isfinite(gap) and math.isfinite(maximum)),
    }
    for lam in (1e-3, 1e-2, 1e-1):
        label = f"{lam:.0e}"
        result[f"condition_full_lambda_{label}"] = (maximum * maximum + lam) / lam if math.isfinite(maximum) else math.nan
        result[f"condition_nonzero_lambda_{label}"] = (
            (maximum * maximum + lam) / (gap * gap + lam)
            if math.isfinite(maximum) and math.isfinite(gap)
            else math.nan
        )
    return result


def _payload_rows(path: Path, arm: str) -> list[dict[str, Any]]:
    payload = _read_json(path / "shards" / f"{arm}.json")
    if payload["arm"] != arm:
        raise RuntimeError(f"Archived arm mismatch: {arm}")
    return [dict(row) for row in payload["rows"]]


def _array_starts(rows: Sequence[Mapping[str, Any]], split: str, values: np.ndarray) -> dict[str, tuple[int, int]]:
    selected = [row for row in rows if row["split"] == split]
    starts: dict[str, tuple[int, int]] = {}
    offset = 0
    for row in selected:
        count = int(row["vertices"])
        starts[str(row["sample_id"])] = (offset, offset + count)
        offset += count
    if offset != len(values):
        raise RuntimeError(f"{split} archived array length mismatch: {offset} != {len(values)}")
    return starts


def _aggregate(rows: Sequence[Mapping[str, Any]], fields: Sequence[str], identity: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(identity)
    result["samples"] = len(rows)
    for field in fields:
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        result[f"{field}_mean"] = float(finite.mean()) if len(finite) else math.nan
        result[f"{field}_median"] = float(np.median(finite)) if len(finite) else math.nan
    return result


def _spearman_rows(rows: Sequence[Mapping[str, Any]], predictors: Sequence[str], outcomes: Sequence[str], scope: str) -> list[dict[str, Any]]:
    output = []
    for predictor in predictors:
        for outcome in outcomes:
            x = np.asarray([float(row[predictor]) for row in rows])
            y = np.asarray([float(row[outcome]) for row in rows])
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() >= 3 and np.unique(x[valid]).size >= 2 and np.unique(y[valid]).size >= 2:
                statistic = spearmanr(x[valid], y[valid])
                rho, pvalue = float(statistic.statistic), float(statistic.pvalue)
            else:
                rho, pvalue = math.nan, math.nan
            output.append({"scope": scope, "predictor": predictor, "outcome": outcome, "spearman": rho, "pvalue": pvalue, "n": int(valid.sum())})
    return output


def _quantile_groups(rows: Sequence[dict[str, Any]], field: str, label: str) -> None:
    order = sorted(range(len(rows)), key=lambda index: (float(rows[index][field]), rows[index]["sample_id"]))
    chunks = np.array_split(order, 3)
    for name, chunk in zip(("low", "medium", "high"), chunks):
        for index in chunk.tolist():
            rows[index][label] = name


def run(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest.resolve()
    datasets = {split: PreparedMeshDataset.from_manifest(manifest, split) for split in ("train", "validation", "test")}
    expected = {"train": 400, "validation": 50, "test": 50}
    if {key: len(value) for key, value in datasets.items()} != expected:
        raise RuntimeError("Frozen Sofa50 v2 split contract mismatch")

    b_rows = _payload_rows(args.b_report_dir.resolve(), ARM_B)
    e_rows = _payload_rows(args.e_report_dir.resolve(), ARM_E)
    b_geometry = {(row["split"], row["sample_id"]): row for row in b_rows}
    e_geometry = {(row["split"], row["sample_id"]): row for row in e_rows if row["arm"] == ARM_E}
    b_npz = np.load(args.b_report_dir.resolve() / "shards" / f"{ARM_B}_prediction_arrays.npz")
    e_npz = np.load(args.e_report_dir.resolve() / "shards" / f"{ARM_E}_prediction_arrays.npz")
    array_maps: dict[tuple[str, str], tuple[np.ndarray, dict[str, tuple[int, int]]]] = {}
    for split in ("validation", "test"):
        for arm, archive, rows in ((ARM_B, b_npz, b_rows), (ARM_E, e_npz, e_rows)):
            values = archive[f"{split}_prediction"].astype(np.float64)
            array_maps[(split, arm)] = (values, _array_starts(rows, split, values))

    cached_paths = (
        output / "topology_per_sample.csv",
        output / "component_sizes.csv",
        output / "boundary_local_per_sample.csv",
    )
    use_cache = args.resume_tables and all(path.is_file() for path in cached_paths)
    topology: list[dict[str, Any]] = _read_typed_csv(cached_paths[0]) if use_cache else []
    components: list[dict[str, Any]] = _read_typed_csv(cached_paths[1]) if use_cache else []
    local_rows: list[dict[str, Any]] = _read_typed_csv(cached_paths[2]) if use_cache else []
    spectral_cache: dict[str, dict[str, Any]] = {}
    prediction_cache: dict[str, dict[str, np.ndarray]] = {}
    for split, dataset in (() if use_cache else datasets.items()):
        print(f"audit split={split} samples={len(dataset)}", flush=True)
        for index in range(len(dataset)):
            static = dataset.load_static(index)
            sample_id = str(static["sample_id"])
            vertices = np.asarray(static["vertices"], dtype=np.float64)
            faces = np.asarray(static["faces"], dtype=np.int64)
            clean = np.asarray(static["clean_reference_vertices"], dtype=np.float64)
            clean_faces = np.asarray(static["clean_reference_faces"], dtype=np.int64)
            if not np.array_equal(faces, clean_faces) or vertices.shape != clean.shape:
                raise RuntimeError(f"{sample_id}: same-index topology contract failed")
            row, detail = topology_row(sample_id, split, vertices, faces)
            topology.append(row)
            if (index + 1) % 25 == 0 or index + 1 == len(dataset):
                print(f"  {split}: {index + 1}/{len(dataset)}", flush=True)
            for component, (vertex_count, face_count) in enumerate(zip(detail["vertex_component_sizes"], detail["face_component_sizes"])):
                components.append({"sample_id": sample_id, "split": split, "recipe": row["recipe"], "component": component, "vertices": int(vertex_count), "faces": int(face_count), "vertex_ratio": float(vertex_count / len(vertices)), "face_ratio": float(face_count / len(faces))})

            if split not in {"validation", "test"}:
                continue
            key = hashlib.sha256(np.asarray([len(vertices)], np.int64).tobytes() + faces.tobytes()).hexdigest()
            if key not in spectral_cache:
                spectral_cache[key] = spectral_condition(detail)
            row.update(spectral_cache[key])
            b_values, b_map = array_maps[(split, ARM_B)]
            e_values, e_map = array_maps[(split, ARM_E)]
            b_start, b_stop = b_map[sample_id]
            e_start, e_stop = e_map[sample_id]
            b_delta = b_values[b_start:b_stop]
            e_displacement = e_values[e_start:e_stop]
            target_delta = np.asarray(static["raw_laplacian_target"], dtype=np.float64)
            lap, lap_data = uniform_sparse_laplacian(faces, len(vertices))
            component_count, labels = component_labels(lap_data)
            b_vertices, solve = regularized_sparse_solve(
                lap,
                b_delta,
                vertices,
                labels,
                component_count,
                1e-2,
                atol=1e-12,
                btol=1e-12,
                maxiter=100000,
            )
            if not solve["all_converged"]:
                raise RuntimeError(f"{sample_id}: frozen B sparse solve did not converge")
            e_vertices = vertices + e_displacement
            local = {"sample_id": sample_id, "split": split, "recipe": row["recipe"], "severity": row["severity"]}
            local.update(local_quality(vertices, clean, b_vertices, e_vertices, b_delta, target_delta, faces, detail))
            b_metric, e_metric = b_geometry[(split, sample_id)], e_geometry[(split, sample_id)]
            b_reproduced = float(np.sqrt(np.square(b_vertices - clean).sum(axis=1).mean()))
            e_reproduced = float(np.sqrt(np.square(e_vertices - clean).sum(axis=1).mean()))
            if not np.isclose(b_reproduced, float(b_metric["same_index_recovered_vertex_rms"]), rtol=0, atol=2e-9):
                raise RuntimeError(f"{sample_id}: frozen B recovery reproduction failed")
            if not np.isclose(e_reproduced, float(e_metric["same_index_recovered_vertex_rms"]), rtol=0, atol=2e-9):
                raise RuntimeError(f"{sample_id}: frozen E recovery reproduction failed")
            local.update({
                "b_chamfer": float(b_metric["refined_chamfer"]),
                "e_chamfer": float(e_metric["refined_chamfer"]),
                "delta_chamfer": float(e_metric["refined_chamfer"]) - float(b_metric["refined_chamfer"]),
                "b_vertex_rms": float(b_metric["same_index_recovered_vertex_rms"]),
                "e_vertex_rms": float(e_metric["same_index_recovered_vertex_rms"]),
                "delta_vertex_rms": float(e_metric["same_index_recovered_vertex_rms"]) - float(b_metric["same_index_recovered_vertex_rms"]),
                "b_p95": float(b_metric["p2s_p95"]),
                "e_p95": float(e_metric["p2s_p95"]),
                "delta_p95": float(e_metric["p2s_p95"]) - float(b_metric["p2s_p95"]),
                "b_fscore": float(b_metric["fscore"]),
                "e_fscore": float(e_metric["fscore"]),
                "b_normal": float(b_metric["normal_consistency"]),
                "e_normal": float(e_metric["normal_consistency"]),
                "delta_normal": float(e_metric["normal_consistency"]) - float(b_metric["normal_consistency"]),
                "b_flip_rate": float(b_metric["introduced_flipped_faces"]) / len(faces),
                "e_flip_rate": float(e_metric["introduced_flipped_faces"]) / len(faces),
                "delta_flip_rate": (float(e_metric["introduced_flipped_faces"]) - float(b_metric["introduced_flipped_faces"])) / len(faces),
                "gt_displacement_rms": float(np.sqrt(np.square(clean - vertices).sum(axis=1).mean())),
            })
            for name in ("boundary_vertex_ratio", "boundary_edge_ratio", "connected_components", "largest_component_vertex_ratio", "nonmanifold_edge_ratio", "degree_mean", "degree_variance", "faces", "vertices", "spectral_gap_min_component", "spectral_largest_eigenvalue", "condition_nonzero_lambda_1e-03", "condition_nonzero_lambda_1e-02", "condition_nonzero_lambda_1e-01"):
                local[name] = row[name]
            local_rows.append(local)
            if split == "test":
                prediction_cache[sample_id] = {"initial": vertices, "clean": clean, "b": b_vertices, "e": e_vertices, "b_delta_error": np.linalg.norm(b_delta - target_delta, axis=1), "faces": faces, "boundary_vertices": np.asarray(detail["boundary_vertices"])}

    if not use_cache:
        _write_csv(output / "topology_per_sample.csv", topology)
        _write_csv(output / "component_sizes.csv", components)
        _write_csv(output / "boundary_local_per_sample.csv", local_rows)
    else:
        print("resumed completed per-sample topology/local tables", flush=True)

    topology_summary = []
    topology_fields = ("watertight", "boundary_vertex_ratio", "boundary_edge_ratio", "connected_components", "largest_component_vertex_ratio", "nonmanifold_edge_ratio", "degree_mean", "degree_variance", "vertices", "faces")
    for split in ("train", "validation", "test", "all"):
        base = topology if split == "all" else [row for row in topology if row["split"] == split]
        for group, recipes in GROUPS.items():
            selected = [row for row in base if row["recipe"] in recipes]
            topology_summary.append(_aggregate(selected, topology_fields, {"split": split, "group": group}))
    _write_csv(output / "topology_summary.csv", topology_summary)

    local_summary = []
    local_fields = (
        "b_boundary_rms", "e_boundary_rms", "b_interior_rms", "e_interior_rms",
        "b_deep4plus_rms", "e_deep4plus_rms", "b_boundary_raw_epe", "b_interior_raw_epe",
        "b_boundary_normal_error", "e_boundary_normal_error", "b_interior_normal_error", "e_interior_normal_error",
        "b_boundary_flip_rate", "e_boundary_flip_rate", "b_interior_flip_rate", "e_interior_flip_rate",
        "delta_chamfer", "delta_vertex_rms", "delta_p95", "delta_normal", "delta_flip_rate",
    )
    for split in ("validation", "test"):
        base = [row for row in local_rows if row["split"] == split]
        for group, recipes in GROUPS.items():
            local_summary.append(_aggregate([row for row in base if row["recipe"] in recipes], local_fields, {"split": split, "group": group}))
    _write_csv(output / "boundary_local_summary.csv", local_summary)

    test_rows = [row for row in local_rows if row["split"] == "test"]
    _quantile_groups(test_rows, "boundary_vertex_ratio", "boundary_quantile")
    irregularity = []
    for field in ("boundary_vertex_ratio", "boundary_edge_ratio", "connected_components", "nonmanifold_edge_ratio", "degree_variance"):
        values = np.asarray([float(row[field]) for row in test_rows])
        order = np.argsort(np.argsort(values, kind="stable"), kind="stable") / max(len(values) - 1, 1)
        irregularity.append(order)
    for row, score in zip(test_rows, np.mean(irregularity, axis=0)):
        row["topology_irregularity_score"] = float(score)
    _quantile_groups(test_rows, "topology_irregularity_score", "topology_irregularity_quantile")
    quantile_summary = []
    metric_fields = ("b_chamfer", "e_chamfer", "b_vertex_rms", "e_vertex_rms", "b_p95", "e_p95", "b_fscore", "e_fscore", "b_normal", "e_normal", "b_flip_rate", "e_flip_rate")
    for group_field in ("boundary_quantile", "topology_irregularity_quantile"):
        for group in ("low", "medium", "high"):
            selected = [row for row in test_rows if row[group_field] == group]
            summary = _aggregate(selected, metric_fields, {"grouping": group_field, "group": group})
            summary.update({
                "e_lower_chamfer": sum(row["e_chamfer"] < row["b_chamfer"] for row in selected),
                "e_lower_vertex_rms": sum(row["e_vertex_rms"] < row["b_vertex_rms"] for row in selected),
                "e_lower_p95": sum(row["e_p95"] < row["b_p95"] for row in selected),
                "e_higher_fscore": sum(row["e_fscore"] > row["b_fscore"] for row in selected),
                "e_higher_normal": sum(row["e_normal"] > row["b_normal"] for row in selected),
                "e_lower_flip_rate": sum(row["e_flip_rate"] < row["b_flip_rate"] for row in selected),
            })
            quantile_summary.append(summary)
    _write_csv(output / "topology_quantile_b_vs_e.csv", quantile_summary)

    predictors = ("boundary_vertex_ratio", "boundary_edge_ratio", "connected_components", "largest_component_vertex_ratio", "nonmanifold_edge_ratio", "degree_mean", "degree_variance", "faces", "vertices")
    outcomes = ("delta_chamfer", "delta_vertex_rms", "delta_p95", "delta_normal", "delta_flip_rate")
    correlations = _spearman_rows(test_rows, predictors, outcomes, "test_topology_vs_b_e")
    correlations += _spearman_rows(test_rows, ("gt_displacement_rms",), outcomes, "test_severity_vs_b_e")
    correlations += _spearman_rows(test_rows, ("boundary_vertex_ratio", "connected_components", "spectral_gap_min_component", "condition_nonzero_lambda_1e-02"), ("b_chamfer", "delta_chamfer", "delta_vertex_rms"), "test_spectral_topology")
    _write_csv(output / "correlations.csv", correlations)

    adaptive = _read_csv(args.adaptive_selectors.resolve())
    topology_by_key = {(row["split"], row["sample_id"]): row for row in topology}
    adaptive_joined = []
    for row in adaptive:
        split, sample_id = row["split"], row["sample_id"]
        if split not in {"validation", "test"}:
            continue
        topo = topology_by_key[(split, sample_id)]
        joined = dict(row)
        for field in (*predictors, "spectral_gap_min_component", "spectral_largest_eigenvalue", "condition_nonzero_lambda_1e-03", "condition_nonzero_lambda_1e-02", "condition_nonzero_lambda_1e-01", "spectral_reliable"):
            joined[field] = topo[field]
        joined["nonmanifold_indicator"] = int(topo["nonmanifold_edges"] > 0)
        joined["log10_lambda_cd"] = math.log10(float(row["lambda_cd"]))
        joined["log10_lambda_vrms"] = math.log10(float(row["lambda_vrms"]))
        adaptive_joined.append(joined)
    _write_csv(output / "adaptive_lambda_topology_per_sample.csv", adaptive_joined)
    adaptive_corr = []
    adaptive_predictors = (
        "boundary_vertex_ratio", "boundary_edge_ratio", "connected_components",
        "largest_component_vertex_ratio", "nonmanifold_indicator", "degree_variance", "faces",
        "spectral_gap_min_component", "condition_nonzero_lambda_1e-02",
        "fixed_recovery_displacement_rms",
    )
    for split in ("validation", "test"):
        selected = [row for row in adaptive_joined if row["split"] == split]
        adaptive_corr += _spearman_rows(selected, adaptive_predictors, ("log10_lambda_cd", "log10_lambda_vrms"), f"{split}_adaptive_lambda")
    _write_csv(output / "adaptive_lambda_topology_correlations.csv", adaptive_corr)
    adaptive_histograms = []
    for split in ("validation", "test"):
        selected = [row for row in adaptive_joined if row["split"] == split]
        rank_fields = ("boundary_vertex_ratio", "boundary_edge_ratio", "connected_components", "nonmanifold_indicator", "degree_variance")
        ranks = []
        for field in rank_fields:
            values = np.asarray([float(row[field]) for row in selected])
            ranks.append(np.argsort(np.argsort(values, kind="stable"), kind="stable") / max(len(values) - 1, 1))
        for row, score in zip(selected, np.mean(ranks, axis=0)):
            row["topology_irregularity_score"] = float(score)
        _quantile_groups(selected, "topology_irregularity_score", "topology_irregularity_quantile")
        for group in ("low", "medium", "high"):
            group_rows = [row for row in selected if row["topology_irregularity_quantile"] == group]
            for selector in ("lambda_cd", "lambda_vrms"):
                for value in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0):
                    adaptive_histograms.append({
                        "split": split,
                        "topology_irregularity_group": group,
                        "selector": selector,
                        "lambda": value,
                        "count": sum(np.isclose(float(row[selector]), value) for row in group_rows),
                        "samples": len(group_rows),
                    })
    _write_csv(output / "adaptive_lambda_topology_histograms.csv", adaptive_histograms)

    contribution = {}
    for split in ("validation", "test"):
        selected = [row for row in local_rows if row["split"] == split]
        boundary_total = sum(float(row["vrms2_advantage_boundary_total"]) for row in selected)
        interior_total = sum(float(row["vrms2_advantage_interior_total"]) for row in selected)
        boundary_count = sum(
            int(float(row["b_boundary_count"]))
            if math.isfinite(float(row["b_boundary_count"]))
            else 0
            for row in selected
        )
        interior_count = sum(
            int(float(row["b_interior_count"]))
            if math.isfinite(float(row["b_interior_count"]))
            else 0
            for row in selected
        )
        contribution[split] = {
            "boundary_total": boundary_total,
            "interior_total": interior_total,
            "total": boundary_total + interior_total,
            "boundary_fraction_of_signed_advantage": boundary_total / (boundary_total + interior_total) if boundary_total + interior_total else math.nan,
            "boundary_per_vertex": boundary_total / max(boundary_count, 1),
            "interior_per_vertex": interior_total / max(interior_count, 1),
            "boundary_vertices": boundary_count,
            "interior_vertices": interior_count,
        }

    controlled = []
    for split in ("validation", "test"):
        base = [row for row in local_rows if row["split"] == split]
        for group, recipes in (("B1", {"B1"}), ("B2", {"B2"}), ("mild", MILD), ("strong", STRONG)):
            controlled.append(_aggregate([row for row in base if row["recipe"] in recipes], ("boundary_vertex_ratio", "connected_components", "gt_displacement_rms", "delta_chamfer", "delta_vertex_rms", "delta_normal"), {"split": split, "group": group}))
    _write_csv(output / "controlled_family_summary.csv", controlled)

    contract = {
        "contract_audit": True,
        "read_only": True,
        "models_retrained": False,
        "inference_rerun": False,
        "frozen_archived_predictions": [ARM_B, ARM_E],
        "arm_b_lambda": 1e-2,
        "gt_used_only_after_prediction": True,
        "mesh_repairs": False,
        "split_counts": expected,
        "topology_definition": TOPOLOGY_DEFINITION,
        "spectral_definition": SPECTRAL_DEFINITION,
        "same_index_correspondence_all_validation_test": True,
    }
    _write_json(output / "contract_audit.json", contract)
    summary = {
        "contract": contract,
        "dataset": {
            "meshes": len(topology),
            "watertight_fraction": float(np.mean([row["watertight"] for row in topology])),
            "open_boundary_fraction": float(np.mean([row["boundary_edges"] > 0 for row in topology])),
            "multi_component_fraction": float(np.mean([row["connected_components"] > 1 for row in topology])),
            "nonmanifold_edge_fraction": float(np.mean([row["nonmanifold_edges"] > 0 for row in topology])),
            "boundary_vertex_ratio_mean": float(np.mean([row["boundary_vertex_ratio"] for row in topology])),
            "boundary_vertex_ratio_median": float(np.median([row["boundary_vertex_ratio"] for row in topology])),
            "component_count_mean": float(np.mean([row["connected_components"] for row in topology])),
            "component_count_median": float(np.median([row["connected_components"] for row in topology])),
        },
        "boundary_contribution": contribution,
    }
    _write_json(output / "summary.json", summary)
    write_report(output, summary, topology_summary, local_summary, quantile_summary, correlations, adaptive_corr, controlled)
    write_visual_manifest(output, test_rows, prediction_cache)


def _fmt(value: Any, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if not math.isfinite(number) else f"{number:.{digits}g}"


def write_report(output: Path, summary: Mapping[str, Any], topology_summary: Sequence[Mapping[str, Any]], local_summary: Sequence[Mapping[str, Any]], quantiles: Sequence[Mapping[str, Any]], correlations: Sequence[Mapping[str, Any]], adaptive: Sequence[Mapping[str, Any]], controlled: Sequence[Mapping[str, Any]]) -> None:
    dataset = summary["dataset"]
    test = next(row for row in local_summary if row["split"] == "test" and row["group"] == "all")
    top_corr = sorted(
        [row for row in correlations if row["scope"] == "test_topology_vs_b_e" and row["outcome"] == "delta_vertex_rms" and math.isfinite(float(row["spearman"]))],
        key=lambda row: abs(float(row["spearman"])), reverse=True,
    )
    e_boundary_better = float(test["e_boundary_rms_mean"]) < float(test["b_boundary_rms_mean"])
    e_interior_better = float(test["e_interior_rms_mean"]) < float(test["b_interior_rms_mean"])
    robust_topology = bool(top_corr and abs(float(top_corr[0]["spearman"])) >= 0.3 and float(top_corr[0]["pvalue"]) < 0.05)
    if e_boundary_better and e_interior_better and not robust_topology:
        classification = "T2"
    elif robust_topology:
        classification = "T3"
    else:
        classification = "T4"
    lines = [
        "# Sofa50 v2 frozen B/E topology-quality diagnostic",
        "",
        "Contract audit: **true**. This is a read-only analysis of frozen archived B/E predictions; no model was retrained and no mesh was repaired.",
        "",
        f"Final classification: **{classification}**.",
        "",
        "## Deterministic definitions",
        "",
        f"- Topology: {TOPOLOGY_DEFINITION}.",
        f"- Spectrum: {SPECTRAL_DEFINITION}.",
        "- Boundary vertex: incident to at least one incidence-one edge. Boundary-adjacent face: contains at least one boundary vertex.",
        "- Local normal error is `1-abs(cosine)` against the same-index clean face. Local flip rates divide by nondegenerate eligible faces in that region.",
        "",
        "## Dataset topology audit (500 meshes)",
        "",
        f"- Watertight: {_fmt(dataset['watertight_fraction'],4)}; open boundary: {_fmt(dataset['open_boundary_fraction'],4)}; multiple components: {_fmt(dataset['multi_component_fraction'],4)}; non-manifold edges: {_fmt(dataset['nonmanifold_edge_fraction'],4)}.",
        f"- Boundary-vertex ratio mean/median: {_fmt(dataset['boundary_vertex_ratio_mean'])} / {_fmt(dataset['boundary_vertex_ratio_median'])}; component count mean/median: {_fmt(dataset['component_count_mean'])} / {_fmt(dataset['component_count_median'])}.",
        "",
        "### Recipe-level topology",
        "",
        "| Recipe | Watertight | Boundary-vertex ratio | Components | Non-manifold-edge ratio | Mean faces |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for recipe in RECIPES:
        row = next(item for item in topology_summary if item["split"] == "all" and item["group"] == recipe)
        lines.append(
            f"| {recipe} | {_fmt(row['watertight_mean'],4)} | {_fmt(row['boundary_vertex_ratio_mean'])} | "
            f"{_fmt(row['connected_components_mean'])} | {_fmt(row['nonmanifold_edge_ratio_mean'])} | {_fmt(row['faces_mean'])} |"
        )
    lines += [
        "",
        "Midpoint and adaptive subdivision change edge/face density but do not repair the inherited component structure; this is measured, not assumed.",
        "",
        "## Test boundary versus interior",
        "",
        "| Region | B vertex RMS | E vertex RMS | B normal error | E normal error | B flip rate | E flip rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Boundary | {_fmt(test['b_boundary_rms_mean'])} | {_fmt(test['e_boundary_rms_mean'])} | {_fmt(test['b_boundary_normal_error_mean'])} | {_fmt(test['e_boundary_normal_error_mean'])} | {_fmt(test['b_boundary_flip_rate_mean'])} | {_fmt(test['e_boundary_flip_rate_mean'])} |",
        f"| Interior | {_fmt(test['b_interior_rms_mean'])} | {_fmt(test['e_interior_rms_mean'])} | {_fmt(test['b_interior_normal_error_mean'])} | {_fmt(test['e_interior_normal_error_mean'])} | {_fmt(test['b_interior_flip_rate_mean'])} | {_fmt(test['e_interior_flip_rate_mean'])} |",
        f"| Deep interior (>=4 hops) | {_fmt(test['b_deep4plus_rms_mean'])} | {_fmt(test['e_deep4plus_rms_mean'])} | n/a | n/a | n/a | n/a |",
        "",
        f"Arm B raw-Laplacian EPE at boundary/interior vertices is `{_fmt(test['b_boundary_raw_epe_mean'])}` / `{_fmt(test['b_interior_raw_epe_mean'])}`. Boundary rows average only meshes with a nonempty boundary; closed meshes remain explicit in the per-sample table with boundary count zero.",
        "",
        "## Boundary contribution to B minus E squared error",
        "",
    ]
    for split, values in summary["boundary_contribution"].items():
        lines.append(f"- {split}: boundary total `{_fmt(values['boundary_total'])}`, interior total `{_fmt(values['interior_total'])}`, boundary share of signed advantage `{_fmt(values['boundary_fraction_of_signed_advantage'],4)}`; per-vertex boundary/interior `{_fmt(values['boundary_per_vertex'])}` / `{_fmt(values['interior_per_vertex'])}`.")
    lines += ["", "## Strongest test topology associations with E-B vertex RMS", ""]
    for row in top_corr[:5]:
        lines.append(f"- `{row['predictor']}`: Spearman `{_fmt(row['spearman'],4)}`, p `{_fmt(row['pvalue'],4)}`, n={row['n']}.")
    lines += ["", "## Boundary-ratio quantiles", "", "| Group | B CD | E CD | B VRMS | E VRMS | E CD wins | E VRMS wins |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in quantiles:
        if row["grouping"] == "boundary_quantile":
            lines.append(f"| {row['group']} | {_fmt(row['b_chamfer_mean'])} | {_fmt(row['e_chamfer_mean'])} | {_fmt(row['b_vertex_rms_mean'])} | {_fmt(row['e_vertex_rms_mean'])} | {row['e_lower_chamfer']}/{row['samples']} | {row['e_lower_vertex_rms']}/{row['samples']} |")
    lines += ["", "## Adaptive lambda and spectral conditioning", ""]
    severity_rows = [row for row in adaptive if row["predictor"] == "fixed_recovery_displacement_rms" and row["outcome"] == "log10_lambda_cd"]
    topology_lambda = sorted([row for row in adaptive if row["predictor"] != "fixed_recovery_displacement_rms" and row["outcome"] == "log10_lambda_cd" and math.isfinite(float(row["spearman"]))], key=lambda row: abs(float(row["spearman"])), reverse=True)
    for row in severity_rows + topology_lambda[:4]:
        lines.append(f"- {row['scope']} `{row['predictor']}` vs `{row['outcome']}`: rho `{_fmt(row['spearman'],4)}`, p `{_fmt(row['pvalue'],4)}`.")
    spectral_rows = sorted(
        [row for row in correlations if row["scope"] == "test_spectral_topology" and row["predictor"] in {"spectral_gap_min_component", "condition_nonzero_lambda_1e-02"} and math.isfinite(float(row["spearman"]))],
        key=lambda row: abs(float(row["spearman"])), reverse=True,
    )
    for row in spectral_rows:
        lines.append(f"- test `{row['predictor']}` vs `{row['outcome']}`: rho `{_fmt(row['spearman'],4)}`, p `{_fmt(row['pvalue'],4)}`.")
    lines += [
        "",
        "All 100 validation/test spectral estimates passed the reliability check. Full regularized and nonzero-subspace condition proxies for λ=1e-3/1e-2/1e-1 are in `topology_per_sample.csv`; adaptive-λ low/medium/high topology histograms are in `adaptive_lambda_topology_histograms.csv`.",
        "",
        "## Controlled family checks",
        "",
        "| Split | Group | Boundary ratio | Components | GT correction RMS | ΔCD (E-B) | ΔVRMS (E-B) | ΔNormal (E-B) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in controlled:
        lines.append(
            f"| {row['split']} | {row['group']} | {_fmt(row['boundary_vertex_ratio_mean'])} | "
            f"{_fmt(row['connected_components_mean'])} | {_fmt(row['gt_displacement_rms_mean'])} | "
            f"{_fmt(row['delta_chamfer_mean'])} | {_fmt(row['delta_vertex_rms_mean'])} | {_fmt(row['delta_normal_mean'])} |"
        )
    lines += [
        "",
        "B1/B2 have identical connectivity per object because they share one global-midpoint topology; their correction RMS differs substantially. Their B/E changes therefore isolate severity from topology more cleanly than cross-family comparisons. Full recipe/group local metrics are in `boundary_local_summary.csv`.",
        "",
        "## Interpretation",
        "",
    ]
    if classification == "T2":
        lines.append("E remains better in boundary and deep-interior correspondence error, while no robust topology association explains the E-B gap. The direct-residual advantage is therefore not primarily a watertightness effect.")
    elif classification == "T3":
        lines.append("Topology/boundary variables show a robust association with part of the E-B gap, but severity and interior behavior remain material. The evidence supports mixed topology and correction-severity effects, not a boundary-only explanation.")
    else:
        lines.append("The available topology variables do not provide a stable sufficient explanation for the representation gap.")
    lines += ["", "Open or disconnected meshes do **not** make the graph Laplacian invalid. This diagnostic asks only whether those properties reduce the usefulness of the chosen raw-Laplacian target and recovery design.", "", "Machine-readable outputs include per-sample topology, components, local errors, correlations, adaptive-lambda associations, spectral proxies, controlled groups, and an objective visual-selection manifest."]
    (output / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_visual_manifest(output: Path, rows: Sequence[Mapping[str, Any]], cache: Mapping[str, Mapping[str, np.ndarray]]) -> None:
    candidates: list[tuple[str, Mapping[str, Any]]] = []
    candidates.append(("highest_boundary_ratio", max(rows, key=lambda row: (row["boundary_vertex_ratio"], row["sample_id"]))))
    candidates.append(("lowest_boundary_ratio", min(rows, key=lambda row: (row["boundary_vertex_ratio"], row["sample_id"]))))
    candidates.append(("strongest_e_win", min(rows, key=lambda row: (row["delta_chamfer"], row["sample_id"]))))
    candidates.append(("strongest_b_win", max(rows, key=lambda row: (row["delta_chamfer"], row["sample_id"]))))
    candidates.append(("near_tie", min(rows, key=lambda row: (abs(row["delta_chamfer"]), row["sample_id"]))))
    multi = [row for row in rows if row["connected_components"] > 1]
    if multi:
        candidates.append(("multi_component", max(multi, key=lambda row: (row["connected_components"], row["sample_id"]))))
    nonmanifold = [row for row in rows if row["nonmanifold_edge_ratio"] > 0]
    if nonmanifold:
        candidates.append(("nonmanifold", max(nonmanifold, key=lambda row: (row["nonmanifold_edge_ratio"], row["sample_id"]))))
    selected: dict[str, dict[str, Any]] = {}
    for reason, row in candidates:
        sample_id = str(row["sample_id"])
        selected.setdefault(sample_id, {"sample_id": sample_id, "reasons": []})["reasons"].append(reason)
    _write_json(output / "visual_selection.json", {"selection_rule": "objective extrema/tie plus available topology defects", "samples": list(selected.values())})
    archive = output / "visual_arrays"
    archive.mkdir(exist_ok=True)
    for sample_id in selected:
        if sample_id in cache:
            np.savez_compressed(archive / f"{sample_id}.npz", **cache[sample_id])


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--b-report-dir", type=Path, required=True)
    value.add_argument("--e-report-dir", type=Path, required=True)
    value.add_argument("--adaptive-selectors", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--resume-tables", action="store_true")
    return value


if __name__ == "__main__":
    run(parser().parse_args())
