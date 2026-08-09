#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import trimesh


EPS = 1e-12


def load_mesh(path: Path):
    mesh = trimesh.load(path, process=False)

    if isinstance(mesh, trimesh.Scene):
        geometries = list(mesh.geometry.values())
        if not geometries:
            raise RuntimeError(f"No geometry found in {path}")
        mesh = trimesh.util.concatenate(geometries)

    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"Failed to load mesh: {path}")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    return vertices, faces, mesh


def magnitude(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x, axis=1)


def stats(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return {}

    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(np.max(x)),
    }


def print_compare(
    name: str,
    values: np.ndarray,
    outlier_mask: np.ndarray,
):
    values = np.asarray(values, dtype=np.float64)

    a = values[outlier_mask]
    b = values[~outlier_mask]

    sa = stats(a)
    sb = stats(b)

    print()
    print(name)
    print("-" * 84)

    print(
        f"{'':20s}"
        f"{'outlier':>16s}"
        f"{'other 99%':>16s}"
        f"{'ratio':>16s}"
    )

    for key in ["mean", "median", "p95", "p99", "max"]:
        av = sa.get(key, float("nan"))
        bv = sb.get(key, float("nan"))

        ratio = (
            av / bv
            if np.isfinite(av)
            and np.isfinite(bv)
            and abs(bv) > 1e-15
            else float("nan")
        )

        print(
            f"{key:20s}"
            f"{av:16.8f}"
            f"{bv:16.8f}"
            f"{ratio:16.4f}"
        )


def correlation(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)

    valid = np.isfinite(a) & np.isfinite(b)

    a = a[valid]
    b = b[valid]

    if len(a) < 2:
        return float("nan")

    if np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return float("nan")

    return float(np.corrcoef(a, b)[0, 1])


def load_required(
    diag: np.lib.npyio.NpzFile,
    key: str,
) -> np.ndarray:
    if key not in diag.files:
        raise KeyError(
            f"Missing '{key}' in diagnostics NPZ.\n"
            f"Available fields: {diag.files}"
        )

    return np.asarray(diag[key], dtype=np.float64)


def compute_one_ring_edge_stats(
    vertices: np.ndarray,
    faces: np.ndarray,
):
    """
    Compute per-vertex one-ring edge-length statistics.

    Edge set is unique undirected mesh edges.
    For each vertex:
      edge_min
      edge_max
      edge_mean
      edge_std
      edge_cv = std / mean
      edge_ratio = max / min
      valence
    """

    n = len(vertices)

    edges = np.concatenate(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ],
        axis=0,
    )

    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    edge_vectors = (
        vertices[edges[:, 0]]
        - vertices[edges[:, 1]]
    )

    edge_lengths = np.linalg.norm(
        edge_vectors,
        axis=1,
    )

    adjacency_lengths = [[] for _ in range(n)]

    for (a, b), length in zip(edges, edge_lengths):
        adjacency_lengths[a].append(float(length))
        adjacency_lengths[b].append(float(length))

    edge_min = np.full(n, np.nan, dtype=np.float64)
    edge_max = np.full(n, np.nan, dtype=np.float64)
    edge_mean = np.full(n, np.nan, dtype=np.float64)
    edge_std = np.full(n, np.nan, dtype=np.float64)
    edge_cv = np.full(n, np.nan, dtype=np.float64)
    edge_ratio = np.full(n, np.nan, dtype=np.float64)
    valence = np.zeros(n, dtype=np.int64)

    for i, lengths in enumerate(adjacency_lengths):
        if not lengths:
            continue

        x = np.asarray(
            lengths,
            dtype=np.float64,
        )

        valence[i] = len(x)

        edge_min[i] = np.min(x)
        edge_max[i] = np.max(x)
        edge_mean[i] = np.mean(x)
        edge_std[i] = np.std(x)

        if edge_mean[i] > EPS:
            edge_cv[i] = (
                edge_std[i]
                / edge_mean[i]
            )

        if edge_min[i] > EPS:
            edge_ratio[i] = (
                edge_max[i]
                / edge_min[i]
            )

    return {
        "edge_min": edge_min,
        "edge_max": edge_max,
        "edge_mean": edge_mean,
        "edge_std": edge_std,
        "edge_cv": edge_cv,
        "edge_ratio": edge_ratio,
        "valence": valence,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze extreme recovery displacement against "
            "Laplacian scaling and local one-ring sampling."
        )
    )

    parser.add_argument(
        "run_dir",
        type=Path,
        help=(
            "main_confidence directory containing coarse.obj, "
            "predicted_refined.obj and per_vertex_diagnostics.npz"
        ),
    )

    parser.add_argument(
        "--top-percent",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-12,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    if not (0.0 < args.top_percent <= 100.0):
        raise ValueError(
            "--top-percent must be in (0, 100]."
        )

    root = args.run_dir.expanduser().resolve()

    initial_path = root / "coarse.obj"
    refined_path = root / "predicted_refined.obj"
    diag_path = (
        root
        / "per_vertex_diagnostics.npz"
    )

    for path in [
        initial_path,
        refined_path,
        diag_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    # ============================================================
    # Geometry
    # ============================================================

    v0, faces, _ = load_mesh(initial_path)
    v1, faces_refined, _ = load_mesh(refined_path)

    if v0.shape != v1.shape:
        raise ValueError(
            f"Vertex mismatch: "
            f"initial={v0.shape}, "
            f"refined={v1.shape}"
        )

    if (
        faces.shape != faces_refined.shape
        or not np.array_equal(
            faces,
            faces_refined,
        )
    ):
        raise ValueError(
            "Initial/refined topology differs."
        )

    displacement_vector = v1 - v0
    displacement = magnitude(
        displacement_vector
    )

    n = len(v0)

    # ============================================================
    # Diagnostics NPZ
    # ============================================================

    diag = np.load(diag_path)

    print("NPZ fields:")
    for key in diag.files:
        print(
            f"  {key}: "
            f"{diag[key].shape}"
        )

    delta_hat_pred = load_required(
        diag,
        "delta_hat_prediction",
    )

    delta_raw_pred = load_required(
        diag,
        "delta_pred_raw",
    )

    delta_raw_current = load_required(
        diag,
        "delta_current_raw",
    )

    h = load_required(
        diag,
        "h_current",
    ).reshape(-1)

    confidence = load_required(
        diag,
        "confidence_prediction",
    ).reshape(-1)

    visibility_count = load_required(
        diag,
        "visibility_count",
    ).reshape(-1)

    weight = load_required(
        diag,
        "weight",
    ).reshape(-1)

    saved_displacement = load_required(
        diag,
        "displacement",
    ).reshape(-1)

    # ============================================================
    # Normalized/current Laplacians
    # ============================================================

    h2 = h**2

    denominator = (
        h2
        + args.epsilon
    )

    delta_hat_current = (
        delta_raw_current
        / denominator[:, None]
    )

    delta_hat_change = (
        delta_hat_pred
        - delta_hat_current
    )

    delta_raw_change = (
        delta_raw_pred
        - delta_raw_current
    )

    hat_pred_mag = magnitude(
        delta_hat_pred
    )

    hat_current_mag = magnitude(
        delta_hat_current
    )

    hat_change_mag = magnitude(
        delta_hat_change
    )

    raw_pred_mag = magnitude(
        delta_raw_pred
    )

    raw_current_mag = magnitude(
        delta_raw_current
    )

    raw_change_mag = magnitude(
        delta_raw_change
    )

    # ============================================================
    # One-ring sampling diagnostic
    # ============================================================

    ring = compute_one_ring_edge_stats(
        v0,
        faces,
    )

    edge_min = ring["edge_min"]
    edge_max = ring["edge_max"]
    edge_mean = ring["edge_mean"]
    edge_std = ring["edge_std"]
    edge_cv = ring["edge_cv"]
    edge_ratio = ring["edge_ratio"]
    valence = ring["valence"]

    # ============================================================
    # Validation checks
    # ============================================================

    print()
    print("=" * 84)
    print("Recovery Outlier / Laplacian Diagnostic")
    print("=" * 84)

    print(f"vertices                 : {n}")
    print(f"faces                    : {len(faces)}")

    print(
        "saved displacement error : "
        f"{np.max(np.abs(saved_displacement - displacement)):.10e}"
    )

    reconstructed_raw = (
        delta_hat_pred
        * denominator[:, None]
    )

    print(
        "pred raw roundtrip error  : "
        f"{np.max(np.abs(reconstructed_raw - delta_raw_pred)):.10e}"
    )

    reconstructed_current = (
        delta_hat_current
        * denominator[:, None]
    )

    print(
        "current raw roundtrip err : "
        f"{np.max(np.abs(reconstructed_current - delta_raw_current)):.10e}"
    )

    # This is important:
    # h should equal arithmetic mean one-ring edge length.
    valid_h = (
        np.isfinite(edge_mean)
        & np.isfinite(h)
    )

    h_difference = np.abs(
        edge_mean[valid_h]
        - h[valid_h]
    )

    print()
    print("=" * 84)
    print("h Definition Validation")
    print("=" * 84)

    print(
        "max |recomputed edge_mean - h_current| : "
        f"{h_difference.max():.10e}"
    )

    print(
        "mean |recomputed edge_mean - h_current|: "
        f"{h_difference.mean():.10e}"
    )

    # ============================================================
    # Outlier selection
    # ============================================================

    threshold = np.percentile(
        displacement,
        100.0 - args.top_percent,
    )

    outlier_mask = (
        displacement >= threshold
    )

    outlier_indices = np.flatnonzero(
        outlier_mask
    )

    print()
    print(f"top percent              : {args.top_percent:.3f}%")
    print(f"displacement threshold   : {threshold:.8f}")
    print(f"outlier count            : {len(outlier_indices)}")

    # ============================================================
    # Laplacian diagnostics
    # ============================================================

    print()
    print("=" * 84)
    print("NORMALIZED LAPLACIAN SPACE")
    print("=" * 84)

    print_compare(
        "|delta_hat_prediction|",
        hat_pred_mag,
        outlier_mask,
    )

    print_compare(
        "|delta_hat_current|",
        hat_current_mag,
        outlier_mask,
    )

    print_compare(
        "|delta_hat_prediction - delta_hat_current|",
        hat_change_mag,
        outlier_mask,
    )

    print()
    print("=" * 84)
    print("RAW LAPLACIAN SPACE")
    print("=" * 84)

    print_compare(
        "|delta_pred_raw|",
        raw_pred_mag,
        outlier_mask,
    )

    print_compare(
        "|delta_current_raw|",
        raw_current_mag,
        outlier_mask,
    )

    print_compare(
        "|delta_pred_raw - delta_current_raw|",
        raw_change_mag,
        outlier_mask,
    )

    # ============================================================
    # Local scale
    # ============================================================

    print()
    print("=" * 84)
    print("LOCAL SCALE")
    print("=" * 84)

    print_compare(
        "h_current",
        h,
        outlier_mask,
    )

    print_compare(
        "h_current^2",
        h2,
        outlier_mask,
    )

    # ============================================================
    # NEW: one-ring sampling
    # ============================================================

    print()
    print("=" * 84)
    print("ONE-RING EDGE SAMPLING")
    print("=" * 84)

    print_compare(
        "one-ring edge min",
        edge_min,
        outlier_mask,
    )

    print_compare(
        "one-ring edge max",
        edge_max,
        outlier_mask,
    )

    print_compare(
        "one-ring edge mean",
        edge_mean,
        outlier_mask,
    )

    print_compare(
        "one-ring edge std",
        edge_std,
        outlier_mask,
    )

    print_compare(
        "one-ring edge CV (std/mean)",
        edge_cv,
        outlier_mask,
    )

    print_compare(
        "one-ring edge ratio (max/min)",
        edge_ratio,
        outlier_mask,
    )

    print_compare(
        "vertex valence",
        valence.astype(np.float64),
        outlier_mask,
    )

    # ============================================================
    # Recovery support
    # ============================================================

    print()
    print("=" * 84)
    print("RECOVERY SUPPORT")
    print("=" * 84)

    print_compare(
        "confidence_prediction",
        confidence,
        outlier_mask,
    )

    print_compare(
        "visibility_count",
        visibility_count,
        outlier_mask,
    )

    print_compare(
        "recovery weight",
        weight,
        outlier_mask,
    )

    print_compare(
        "refinement displacement",
        displacement,
        outlier_mask,
    )

    # ============================================================
    # Visibility summary
    # ============================================================

    print()
    print("=" * 84)
    print("Visibility of Outliers")
    print("=" * 84)

    for name, mask in [
        ("outliers", outlier_mask),
        ("other99", ~outlier_mask),
    ]:
        values = visibility_count[mask]

        print()
        print(name)

        print(
            f"  zero view : "
            f"{100.0 * np.mean(values == 0):.2f}%"
        )

        print(
            f"  1-2 views : "
            f"{100.0 * np.mean((values >= 1) & (values <= 2)):.2f}%"
        )

        print(
            f"  >=3 views : "
            f"{100.0 * np.mean(values >= 3):.2f}%"
        )

    # ============================================================
    # Correlations
    # ============================================================

    print()
    print("=" * 84)
    print("Correlation with refinement displacement")
    print("=" * 84)

    fields = {
        "|delta_hat_prediction|":
            hat_pred_mag,

        "|delta_hat_current|":
            hat_current_mag,

        "|delta_hat_pred-delta_hat_current|":
            hat_change_mag,

        "|delta_pred_raw|":
            raw_pred_mag,

        "|delta_current_raw|":
            raw_current_mag,

        "|delta_pred_raw-delta_current_raw|":
            raw_change_mag,

        "h_current":
            h,

        "h_current^2":
            h2,

        "edge_min":
            edge_min,

        "edge_max":
            edge_max,

        "edge_std":
            edge_std,

        "edge_cv":
            edge_cv,

        "edge_ratio":
            edge_ratio,

        "valence":
            valence.astype(np.float64),

        "confidence":
            confidence,

        "visibility_count":
            visibility_count,

        "weight":
            weight,
    }

    for name, values in fields.items():
        print(
            f"{name:46s}: "
            f"{correlation(displacement, values): .6f}"
        )

    # ============================================================
    # Scaling consistency
    # ============================================================

    predicted_raw_change_from_hat = (
        delta_hat_change
        * denominator[:, None]
    )

    scale_consistency_error = magnitude(
        predicted_raw_change_from_hat
        - delta_raw_change
    )

    print()
    print("=" * 84)
    print("h^2 Scaling Consistency")
    print("=" * 84)

    print(
        "max |Δraw - Δhat*(h^2+eps)|: "
        f"{scale_consistency_error.max():.10e}"
    )

    # ============================================================
    # Top-K
    # ============================================================

    order = np.argsort(
        displacement
    )[::-1]

    top_k = min(
        args.top_k,
        len(order),
    )

    print()
    print("=" * 210)
    print(
        f"Top {top_k} extreme vertices"
    )
    print("=" * 210)

    print(
        f"{'rk':>3} "
        f"{'idx':>7} "
        f"{'disp':>9} "
        f"{'|Δhat|':>9} "
        f"{'|Δraw|':>9} "
        f"{'h':>8} "
        f"{'emin':>8} "
        f"{'emax':>8} "
        f"{'estd':>8} "
        f"{'CV':>7} "
        f"{'ratio':>8} "
        f"{'val':>4} "
        f"{'conf':>7} "
        f"{'views':>5} "
        f"{'weight':>7}"
    )

    for rank, idx in enumerate(
        order[:top_k],
        start=1,
    ):
        print(
            f"{rank:3d} "
            f"{idx:7d} "
            f"{displacement[idx]:9.5f} "
            f"{hat_change_mag[idx]:9.4f} "
            f"{raw_change_mag[idx]:9.5f} "
            f"{h[idx]:8.4f} "
            f"{edge_min[idx]:8.4f} "
            f"{edge_max[idx]:8.4f} "
            f"{edge_std[idx]:8.4f} "
            f"{edge_cv[idx]:7.3f} "
            f"{edge_ratio[idx]:8.3f} "
            f"{valence[idx]:4d} "
            f"{confidence[idx]:7.4f} "
            f"{int(visibility_count[idx]):5d} "
            f"{weight[idx]:7.4f}"
        )

    # ============================================================
    # CSV
    # ============================================================

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else root
        / "recovery_outlier_diagnostic.csv"
    )

    rows = []

    for idx in outlier_indices:
        rows.append(
            {
                "vertex":
                    int(idx),

                "displacement":
                    float(displacement[idx]),

                "dx":
                    float(
                        displacement_vector[idx, 0]
                    ),

                "dy":
                    float(
                        displacement_vector[idx, 1]
                    ),

                "dz":
                    float(
                        displacement_vector[idx, 2]
                    ),

                "delta_hat_pred_magnitude":
                    float(
                        hat_pred_mag[idx]
                    ),

                "delta_hat_current_magnitude":
                    float(
                        hat_current_mag[idx]
                    ),

                "delta_hat_change_magnitude":
                    float(
                        hat_change_mag[idx]
                    ),

                "delta_raw_pred_magnitude":
                    float(
                        raw_pred_mag[idx]
                    ),

                "delta_raw_current_magnitude":
                    float(
                        raw_current_mag[idx]
                    ),

                "delta_raw_change_magnitude":
                    float(
                        raw_change_mag[idx]
                    ),

                "h_current":
                    float(h[idx]),

                "h_current_squared":
                    float(h2[idx]),

                "edge_min":
                    float(edge_min[idx]),

                "edge_max":
                    float(edge_max[idx]),

                "edge_mean":
                    float(edge_mean[idx]),

                "edge_std":
                    float(edge_std[idx]),

                "edge_cv":
                    float(edge_cv[idx]),

                "edge_ratio":
                    float(edge_ratio[idx]),

                "valence":
                    int(valence[idx]),

                "confidence":
                    float(
                        confidence[idx]
                    ),

                "visibility_count":
                    int(
                        visibility_count[idx]
                    ),

                "weight":
                    float(weight[idx]),
            }
        )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 84)
    print("Saved")
    print("=" * 84)
    print(output)


if __name__ == "__main__":
    main()