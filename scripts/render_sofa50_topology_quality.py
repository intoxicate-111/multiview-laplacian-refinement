#!/usr/bin/env python3
from __future__ import annotations

"""CPU renderer for objective Sofa50 frozen B/E topology selections."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from diagnose_sofa50_exact_solve_visibility_sweep import component_labels, uniform_sparse_laplacian
from diagnose_sofa50_regularized_sparse_sweep import regularized_sparse_solve
from diagnose_sofa50_topology_quality import ARM_B, ARM_E, _array_starts, _payload_rows
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g}\n")
        for face in faces:
            handle.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def _limits(vertices: np.ndarray):
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    center = 0.5 * (low + high)
    radius = 0.55 * max(float((high - low).max()), 1e-8)
    return [(center[i] - radius, center[i] + radius) for i in range(3)]


def _draw(ax, vertices, faces, *, title, limits, values=None, cmap="viridis", vmax=None, boundary=None):
    if len(faces) > 30000:
        face_index = np.linspace(0, len(faces) - 1, 30000, dtype=np.int64)
        shown = faces[face_index]
    else:
        shown = faces
    triangles = vertices[shown]
    if values is None:
        colors = np.tile(np.asarray([[0.72, 0.76, 0.82, 1.0]]), (len(shown), 1))
    else:
        face_values = np.asarray(values)[shown].mean(axis=1)
        scale = max(float(vmax or np.quantile(values, 0.95)), 1e-12)
        colors = plt.get_cmap(cmap)(np.clip(face_values / scale, 0, 1))
    collection = Poly3DCollection(triangles, facecolors=colors, edgecolors="none", linewidths=0)
    ax.add_collection3d(collection)
    if boundary is not None and len(boundary):
        shown_boundary = boundary
        if len(shown_boundary) > 4000:
            shown_boundary = shown_boundary[np.linspace(0, len(shown_boundary) - 1, 4000, dtype=np.int64)]
        points = vertices[shown_boundary]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="#ef233c", s=2.0, depthshade=False)
    ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1]); ax.set_zlim(*limits[2])
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=20, azim=-60)
    ax.set_axis_off(); ax.set_title(title, fontsize=9)


def _edge_boundary_vertices(faces: np.ndarray) -> np.ndarray:
    directed = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    edges, counts = np.unique(np.sort(directed, axis=1), axis=0, return_counts=True)
    return np.unique(edges[counts == 1])


def run(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    selection = json.loads((output / "visual_selection.json").read_text(encoding="utf-8"))["samples"]
    dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), "test")
    index_by_id = {sample_id: index for index, sample_id in enumerate(dataset.sample_ids)}
    b_rows = _payload_rows(args.b_report_dir.resolve(), ARM_B)
    e_rows = _payload_rows(args.e_report_dir.resolve(), ARM_E)
    b_npz = np.load(args.b_report_dir.resolve() / "shards" / f"{ARM_B}_prediction_arrays.npz")
    e_npz = np.load(args.e_report_dir.resolve() / "shards" / f"{ARM_E}_prediction_arrays.npz")
    b_values = b_npz["test_prediction"].astype(np.float64)
    e_values = e_npz["test_prediction"].astype(np.float64)
    b_map = _array_starts(b_rows, "test", b_values)
    e_map = _array_starts(e_rows, "test", e_values)
    panels = output / "visuals" / "panels"
    meshes = output / "visuals" / "meshes"
    panels.mkdir(parents=True, exist_ok=True)
    rendered = []
    for order, entry in enumerate(selection):
        sample_id = entry["sample_id"]
        static = dataset.load_static(index_by_id[sample_id])
        initial = np.asarray(static["vertices"], dtype=np.float64)
        clean = np.asarray(static["clean_reference_vertices"], dtype=np.float64)
        faces = np.asarray(static["faces"], dtype=np.int64)
        b0, b1 = b_map[sample_id]; e0, e1 = e_map[sample_id]
        b_delta = b_values[b0:b1]
        target_delta = np.asarray(static["raw_laplacian_target"], dtype=np.float64)
        lap, lap_data = uniform_sparse_laplacian(faces, len(initial))
        component_count, labels = component_labels(lap_data)
        b_vertices, solve = regularized_sparse_solve(
            lap, b_delta, initial, labels, component_count, 1e-2,
            atol=1e-12, btol=1e-12, maxiter=100000,
        )
        if not solve["all_converged"]:
            raise RuntimeError(f"{sample_id}: B solve failed")
        e_vertices = initial + e_values[e0:e1]
        b_error = np.linalg.norm(b_vertices - clean, axis=1)
        e_error = np.linalg.norm(e_vertices - clean, axis=1)
        lap_error = np.linalg.norm(b_delta - target_delta, axis=1)
        boundary = _edge_boundary_vertices(faces)
        limits = _limits(np.concatenate((initial, clean, b_vertices, e_vertices), axis=0))
        vmax = float(np.quantile(np.concatenate((b_error, e_error)), 0.95))
        lap_vmax = float(np.quantile(lap_error, 0.95))
        figure = plt.figure(figsize=(16, 8), dpi=160)
        axes = [figure.add_subplot(2, 4, i + 1, projection="3d") for i in range(8)]
        _draw(axes[0], initial, faces, title="Input + boundary", limits=limits, boundary=boundary)
        _draw(axes[1], clean, faces, title="Clean GT", limits=limits)
        _draw(axes[2], b_vertices, faces, title="B Lap + sparse", limits=limits)
        _draw(axes[3], e_vertices, faces, title="E direct residual", limits=limits)
        _draw(axes[4], b_vertices, faces, title=f"B vertex error (p95={vmax:.3g})", limits=limits, values=b_error, vmax=vmax, cmap="magma")
        _draw(axes[5], e_vertices, faces, title=f"E vertex error (same scale)", limits=limits, values=e_error, vmax=vmax, cmap="magma")
        _draw(axes[6], initial, faces, title=f"B raw Lap error (p95={lap_vmax:.3g})", limits=limits, values=lap_error, vmax=lap_vmax, cmap="viridis")
        axes[7].set_axis_off()
        axes[7].text2D(0.02, 0.75, "Selection:\n" + "\n".join(entry["reasons"]), transform=axes[7].transAxes, fontsize=11)
        axes[7].text2D(0.02, 0.35, f"boundary vertices: {len(boundary)}/{len(initial)}\ncomponents: {component_count}", transform=axes[7].transAxes, fontsize=10)
        figure.suptitle(sample_id, fontsize=12)
        figure.tight_layout()
        panel = panels / f"{order:02d}_{sample_id}.png"
        figure.savefig(panel, bbox_inches="tight")
        plt.close(figure)
        mesh_dir = meshes / sample_id
        for name, values in (("input", initial), ("clean_gt", clean), ("b_lap_plus_refine", b_vertices), ("e_direct_vertex_residual", e_vertices)):
            _write_obj(mesh_dir / f"{name}.obj", values, faces)
        rendered.append({"sample_id": sample_id, "reasons": entry["reasons"], "panel": str(panel.relative_to(output)), "mesh_dir": str(mesh_dir.relative_to(output)), "boundary_vertices": len(boundary), "vertices": len(initial), "components": component_count})
        print(f"[{order + 1}/{len(selection)}] {sample_id}", flush=True)
    (output / "visuals" / "manifest.json").write_text(json.dumps({"renderer": "matplotlib_cpu_fixed_camera", "same_camera_and_scale_per_sample": True, "samples": rendered}, indent=2) + "\n", encoding="utf-8")
    report = output / "FINAL_REPORT.md"
    text = report.read_text(encoding="utf-8")
    marker = "\n## Representative visual diagnostics\n"
    if marker not in text:
        text += marker + "\nObjective selections (boundary extrema, strongest E/B Chamfer wins, near tie, multi-component and non-manifold where available) are in `visuals/manifest.json`. Each panel uses one fixed camera/scale and includes input+boundary, clean GT, B, E, B/E vertex-error heatmaps, and B raw-Laplacian error. Matching OBJ meshes are under `visuals/meshes/`.\n"
        report.write_text(text, encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--b-report-dir", type=Path, required=True)
    value.add_argument("--e-report-dir", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    return value


if __name__ == "__main__":
    run(parser().parse_args())
