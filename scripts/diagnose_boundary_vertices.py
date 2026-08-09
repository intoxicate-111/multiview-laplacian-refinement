#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


def load_mesh(path: Path):
    path = path.expanduser().resolve()

    mesh = trimesh.load(path, process=False)

    if isinstance(mesh, trimesh.Scene):
        geometries = list(mesh.geometry.values())
        if not geometries:
            raise RuntimeError(f"No geometry found in: {path}")
        mesh = trimesh.util.concatenate(geometries)

    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"Failed to load triangle mesh: {path}")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    if len(vertices) == 0 or len(faces) == 0:
        raise RuntimeError(f"Empty mesh: {path}")

    return vertices, faces, mesh


def boundary_vertex_mask(
    faces: np.ndarray,
    n_vertices: int,
):
    """
    Find topological boundary vertices.

    An undirected edge belonging to exactly one triangle is a boundary edge.
    """

    edges = np.concatenate(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ],
        axis=0,
    )

    edges = np.sort(edges, axis=1)

    unique_edges, counts = np.unique(
        edges,
        axis=0,
        return_counts=True,
    )

    boundary_edges = unique_edges[counts == 1]

    mask = np.zeros(n_vertices, dtype=bool)

    if len(boundary_edges):
        mask[np.unique(boundary_edges)] = True

    return mask, boundary_edges


def print_stats(name: str, values: np.ndarray):
    values = np.asarray(values, dtype=np.float64)

    print()
    print(name)
    print("-" * len(name))
    print(f"count  : {len(values)}")

    if len(values) == 0:
        return

    print(f"mean   : {values.mean():.8f}")
    print(f"median : {np.median(values):.8f}")
    print(f"p90    : {np.percentile(values, 90):.8f}")
    print(f"p95    : {np.percentile(values, 95):.8f}")
    print(f"p99    : {np.percentile(values, 99):.8f}")
    print(f"max    : {values.max():.8f}")


def nearest_surface_distance(
    points: np.ndarray,
    target_mesh: trimesh.Trimesh,
):
    _, distances, _ = trimesh.proximity.closest_point(
        target_mesh,
        points,
    )
    return np.asarray(distances, dtype=np.float64)


def export_points(
    path: Path,
    points: np.ndarray,
    color=(255, 0, 0, 255),
):
    path.parent.mkdir(parents=True, exist_ok=True)

    colors = np.tile(
        np.asarray(color, dtype=np.uint8),
        (len(points), 1),
    )

    cloud = trimesh.points.PointCloud(
        points,
        colors=colors,
    )

    cloud.export(path)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose refinement displacement outliers and "
            "topological boundary behaviour."
        )
    )

    parser.add_argument(
        "--initial",
        required=True,
        type=Path,
        help="Initial/coarse/expanded mesh",
    )

    parser.add_argument(
        "--refined",
        required=True,
        type=Path,
        help="Refined/recovered mesh",
    )

    parser.add_argument(
        "--gt",
        type=Path,
        default=None,
        help="Optional GT mesh for point-to-surface diagnostics",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("boundary_diagnostic"),
        help="Output directory",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of largest-displacement vertices to print",
    )

    parser.add_argument(
        "--outlier-percent",
        type=float,
        default=1.0,
        help="Top percentage of displacement vertices to export",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Load meshes
    # ------------------------------------------------------------

    v0, f0, initial_mesh = load_mesh(args.initial)
    v1, f1, refined_mesh = load_mesh(args.refined)

    if v0.shape != v1.shape:
        raise ValueError(
            "Initial/refined vertex count mismatch:\n"
            f"initial: {v0.shape}\n"
            f"refined: {v1.shape}"
        )

    if f0.shape != f1.shape:
        raise ValueError(
            "Initial/refined face count mismatch:\n"
            f"initial: {f0.shape}\n"
            f"refined: {f1.shape}"
        )

    if not np.array_equal(f0, f1):
        raise ValueError(
            "Initial and refined meshes do not have identical connectivity."
        )

    n_vertices = len(v0)

    # ------------------------------------------------------------
    # Boundary detection
    # ------------------------------------------------------------

    boundary_mask, boundary_edges = boundary_vertex_mask(
        f0,
        n_vertices,
    )

    interior_mask = ~boundary_mask

    # ------------------------------------------------------------
    # Per-vertex refinement displacement
    # ------------------------------------------------------------

    displacement_vectors = v1 - v0
    displacement = np.linalg.norm(
        displacement_vectors,
        axis=1,
    )

    print("=" * 70)
    print("Refinement Vertex Diagnostic")
    print("=" * 70)

    print(f"initial : {args.initial}")
    print(f"refined : {args.refined}")
    print()

    print(f"vertices          : {n_vertices}")
    print(f"faces             : {len(f0)}")
    print(f"boundary edges    : {len(boundary_edges)}")
    print(f"boundary vertices : {boundary_mask.sum()}")
    print(f"interior vertices : {interior_mask.sum()}")
    print(
        f"boundary ratio    : "
        f"{100.0 * boundary_mask.mean():.3f}%"
    )

    # ------------------------------------------------------------
    # Global displacement
    # ------------------------------------------------------------

    print_stats(
        "All vertices — refinement displacement",
        displacement,
    )

    print_stats(
        "Boundary — refinement displacement",
        displacement[boundary_mask],
    )

    print_stats(
        "Interior — refinement displacement",
        displacement[interior_mask],
    )

    if boundary_mask.any() and interior_mask.any():
        b_mean = displacement[boundary_mask].mean()
        i_mean = displacement[interior_mask].mean()

        if i_mean > 0:
            print()
            print(
                "Boundary / interior mean displacement ratio: "
                f"{b_mean / i_mean:.4f}x"
            )

    elif not boundary_mask.any():
        print()
        print(
            "NOTE: No topological boundary vertices found. "
            "This mesh is closed under the edge-count criterion."
        )

    # ------------------------------------------------------------
    # Largest-displacement vertices
    # ------------------------------------------------------------

    top_k = max(
        1,
        min(args.top_k, n_vertices),
    )

    order = np.argsort(displacement)[::-1]
    top_indices = order[:top_k]

    print()
    print("=" * 70)
    print(f"Top {top_k} displacement vertices")
    print("=" * 70)

    print(
        f"{'rank':>5} "
        f"{'vertex':>8} "
        f"{'disp':>12} "
        f"{'boundary':>9} "
        f"{'dx':>12} "
        f"{'dy':>12} "
        f"{'dz':>12}"
    )

    for rank, idx in enumerate(top_indices, start=1):
        dv = displacement_vectors[idx]

        print(
            f"{rank:5d} "
            f"{idx:8d} "
            f"{displacement[idx]:12.8f} "
            f"{str(bool(boundary_mask[idx])):>9} "
            f"{dv[0]:12.8f} "
            f"{dv[1]:12.8f} "
            f"{dv[2]:12.8f}"
        )

    # ------------------------------------------------------------
    # Top-N-percent displacement outliers
    # ------------------------------------------------------------

    if not (0.0 < args.outlier_percent <= 100.0):
        raise ValueError(
            "--outlier-percent must be in (0, 100]."
        )

    percentile_threshold = 100.0 - args.outlier_percent

    threshold = np.percentile(
        displacement,
        percentile_threshold,
    )

    outlier_mask = displacement >= threshold
    outlier_indices = np.flatnonzero(outlier_mask)

    print()
    print("=" * 70)
    print("Displacement Outliers")
    print("=" * 70)

    print(
        f"top percent       : {args.outlier_percent:.3f}%"
    )
    print(
        f"threshold         : {threshold:.8f}"
    )
    print(
        f"outlier vertices  : {outlier_mask.sum()}"
    )
    print(
        f"fraction          : "
        f"{100.0 * outlier_mask.mean():.3f}%"
    )

    # ------------------------------------------------------------
    # Centroid / global translation diagnostic
    # ------------------------------------------------------------

    centroid_initial = v0.mean(axis=0)
    centroid_refined = v1.mean(axis=0)
    centroid_shift = centroid_refined - centroid_initial

    print()
    print("=" * 70)
    print("Global Translation Diagnostic")
    print("=" * 70)

    print(f"initial centroid : {centroid_initial}")
    print(f"refined centroid : {centroid_refined}")
    print(f"centroid shift   : {centroid_shift}")
    print(
        "shift magnitude  : "
        f"{np.linalg.norm(centroid_shift):.8f}"
    )

    # Translation-removed displacement.
    v1_translation_aligned = (
        v1 - centroid_refined + centroid_initial
    )

    aligned_displacement = np.linalg.norm(
        v1_translation_aligned - v0,
        axis=1,
    )

    print_stats(
        "Displacement after removing global centroid translation",
        aligned_displacement,
    )

    # ------------------------------------------------------------
    # Optional GT point-to-surface diagnostic
    # ------------------------------------------------------------

    result = {
        "boundary_mask": boundary_mask,
        "boundary_edges": boundary_edges,
        "displacement": displacement,
        "displacement_vectors": displacement_vectors,
        "aligned_displacement": aligned_displacement,
        "outlier_mask": outlier_mask,
        "outlier_indices": outlier_indices,
        "top_indices": top_indices,
        "centroid_initial": centroid_initial,
        "centroid_refined": centroid_refined,
        "centroid_shift": centroid_shift,
    }

    if args.gt is not None:

        _, _, gt_mesh = load_mesh(args.gt)

        print()
        print("=" * 70)
        print("GT Surface Diagnostic")
        print("=" * 70)

        initial_gt = nearest_surface_distance(
            v0,
            gt_mesh,
        )

        refined_gt = nearest_surface_distance(
            v1,
            gt_mesh,
        )

        improvement = initial_gt - refined_gt

        # Positive = moved closer to GT.
        print_stats(
            "Initial -> GT surface",
            initial_gt,
        )

        print_stats(
            "Refined -> GT surface",
            refined_gt,
        )

        print_stats(
            "GT distance improvement (positive = better)",
            improvement,
        )

        if boundary_mask.any():
            print_stats(
                "Boundary initial -> GT",
                initial_gt[boundary_mask],
            )

            print_stats(
                "Boundary refined -> GT",
                refined_gt[boundary_mask],
            )

            print_stats(
                "Boundary GT improvement",
                improvement[boundary_mask],
            )

        print_stats(
            "Interior initial -> GT",
            initial_gt[interior_mask],
        )

        print_stats(
            "Interior refined -> GT",
            refined_gt[interior_mask],
        )

        print_stats(
            "Interior GT improvement",
            improvement[interior_mask],
        )

        print_stats(
            "Top displacement outliers — initial -> GT",
            initial_gt[outlier_mask],
        )

        print_stats(
            "Top displacement outliers — refined -> GT",
            refined_gt[outlier_mask],
        )

        print_stats(
            "Top displacement outliers — GT improvement",
            improvement[outlier_mask],
        )

        print()
        print(
            "All vertices improved: "
            f"{100.0 * np.mean(improvement > 0):.2f}%"
        )

        print(
            "Top displacement outliers improved: "
            f"{100.0 * np.mean(improvement[outlier_mask] > 0):.2f}%"
        )

        result.update(
            {
                "initial_gt_distance": initial_gt,
                "refined_gt_distance": refined_gt,
                "gt_distance_improvement": improvement,
            }
        )

    # ------------------------------------------------------------
    # Export
    # ------------------------------------------------------------

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    npz_path = output_dir / "diagnostic.npz"

    np.savez_compressed(
        npz_path,
        **result,
    )

    # Red = refined positions of top displacement outliers.
    outlier_refined_path = (
        output_dir
        / "top_displacement_refined_points.ply"
    )

    export_points(
        outlier_refined_path,
        v1[outlier_mask],
        color=(255, 0, 0, 255),
    )

    # Blue = initial positions of same vertices.
    outlier_initial_path = (
        output_dir
        / "top_displacement_initial_points.ply"
    )

    export_points(
        outlier_initial_path,
        v0[outlier_mask],
        color=(0, 80, 255, 255),
    )

    # Top-k points separately.
    topk_path = (
        output_dir
        / "top_k_refined_points.ply"
    )

    export_points(
        topk_path,
        v1[top_indices],
        color=(255, 0, 255, 255),
    )

    print()
    print("=" * 70)
    print("Saved")
    print("=" * 70)

    print(npz_path)
    print(outlier_refined_path)
    print(outlier_initial_path)
    print(topk_path)


if __name__ == "__main__":
    main()