#!/usr/bin/env python3
from __future__ import annotations

"""Render objectively selected frozen B/E full and spectral-band comparisons."""

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw

from diagnose_sofa50_exact_solve_visibility_sweep import component_labels, uniform_sparse_laplacian
from diagnose_sofa50_exact_target_oracle import _clean_mesh
from diagnose_sofa50_regularized_sparse_sweep import regularized_sparse_solve
from diagnose_sofa50_representation_b_vs_e import (
    ARM_B,
    ARM_E,
    SPECTRAL_BANDS,
    spectral_band_components,
)
from mlr.data import Mesh
from mlr.io import save_mesh
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.synthetic import SyntheticRenderConfig, render_mesh_views_opengl
from render_sofa50_recovery_aware_comparisons import _camera


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _render(
    meshes: Sequence[Mesh],
    labels: Sequence[str],
    cameras,
    path: Path,
    size: int,
) -> None:
    config = SyntheticRenderConfig(
        width=size,
        height=size,
        render_mode="lit",
        normalize_mesh=False,
        backend="opengl",
    )
    results = [
        render_mesh_views_opengl(mesh.ensure_normals(), [camera for _, camera in cameras], config)
        for mesh in meshes
    ]
    label_height = 28
    canvas = Image.new(
        "RGB",
        (len(meshes) * size, len(cameras) * (size + label_height)),
        (18, 18, 18),
    )
    draw = ImageDraw.Draw(canvas)
    for column, rendered in enumerate(results):
        for row, ((view, _), (rgb, _mask, _depth)) in enumerate(zip(cameras, rendered, strict=True)):
            x, y = column * size, row * (size + label_height)
            draw.text((x + 5, y + 6), f"{labels[column]} | view {view:02d}", fill=(245, 245, 245))
            canvas.paste(Image.fromarray(rgb), (x, y + label_height))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ad-report-dir", required=True, type=Path)
    parser.add_argument("--ae-report-dir", required=True, type=Path)
    parser.add_argument("--matched-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chebyshev-order", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--views", type=int, nargs="+", default=(0, 7))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty visualization output: {output}")
    dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), "test")
    selections = _read(args.matched_dir.resolve() / "representative_selection.json")
    b_payload = _read(args.ad_report_dir.resolve() / "shards" / f"{ARM_B}.json")
    e_payload = _read(args.ae_report_dir.resolve() / "shards" / f"{ARM_E}.json")
    b_rows = [row for row in b_payload["rows"] if row["split"] == "test"]
    e_rows = [row for row in e_payload["rows"] if row["split"] == "test"]
    b_array = np.load(args.ad_report_dir.resolve() / "shards" / f"{ARM_B}_prediction_arrays.npz")["test_prediction"].astype(np.float64)
    e_array = np.load(args.ae_report_dir.resolve() / "shards" / f"{ARM_E}_prediction_arrays.npz")["test_prediction"].astype(np.float64)
    b_starts = list(np.cumsum([0, *[int(row["vertices"]) for row in b_rows[:-1]]]))
    e_starts = list(np.cumsum([0, *[int(row["vertices"]) for row in e_rows[:-1]]]))
    records = []
    for selection in selections:
        index = int(selection["index"])
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        if sample_id != selection["sample_id"]:
            raise RuntimeError("Representative selection ordering mismatch")
        initial = Mesh(
            np.asarray(static["vertices"], dtype=np.float64),
            np.asarray(static["faces"], dtype=np.int64),
        ).ensure_normals()
        clean = _clean_mesh(static)
        b_prediction = b_array[b_starts[index] : b_starts[index] + initial.num_vertices]
        e_displacement = e_array[e_starts[index] : e_starts[index] + initial.num_vertices]
        lap, lap_data = uniform_sparse_laplacian(initial.faces, initial.num_vertices)
        component_count, labels = component_labels(lap_data)
        b_vertices, solve = regularized_sparse_solve(
            lap,
            b_prediction,
            initial.vertices,
            labels,
            component_count,
            float(b_rows[index]["lambda"]),
            atol=1e-12,
            btol=1e-12,
            maxiter=100000,
        )
        if not solve["all_converged"]:
            raise RuntimeError(f"{sample_id}: Arm B recovery failed")
        b_displacement = b_vertices - initial.vertices
        gt_displacement = clean.vertices - initial.vertices
        signals = np.concatenate(
            (
                gt_displacement,
                b_displacement,
                e_displacement,
                b_displacement - gt_displacement,
                e_displacement - gt_displacement,
            ),
            axis=1,
        )
        filtered, _energy = spectral_band_components(
            signals, initial.faces, order=args.chebyshev_order
        )
        mesh_dir = output / "meshes" / sample_id
        mesh_dir.mkdir(parents=True, exist_ok=True)
        b_mesh = Mesh(b_vertices, initial.faces.copy()).ensure_normals()
        e_mesh = Mesh(initial.vertices + e_displacement, initial.faces.copy()).ensure_normals()
        for name, mesh in (
            ("initial", initial),
            ("clean_gt", clean),
            ("b_refined", b_mesh),
            ("e_refined", e_mesh),
        ):
            save_mesh(mesh, mesh_dir / f"{name}.obj")
        cameras = [(view, _camera(static, view, args.image_size)) for view in args.views]
        full_panel = output / "panels" / f"{index:02d}_{sample_id}_full.png"
        _render(
            (initial, b_mesh, e_mesh, clean),
            ("Initial", "Arm B", "Arm E", "Clean GT"),
            cameras,
            full_panel,
            args.image_size,
        )
        band_panels = {}
        for band in SPECTRAL_BANDS:
            value = filtered[band]
            gt_band = Mesh(initial.vertices + value[:, 0:3], initial.faces.copy()).ensure_normals()
            b_band = Mesh(initial.vertices + value[:, 3:6], initial.faces.copy()).ensure_normals()
            e_band = Mesh(initial.vertices + value[:, 6:9], initial.faces.copy()).ensure_normals()
            b_error = Mesh(clean.vertices + value[:, 9:12], initial.faces.copy()).ensure_normals()
            e_error = Mesh(clean.vertices + value[:, 12:15], initial.faces.copy()).ensure_normals()
            for name, mesh in (
                (f"gt_{band}_displacement", gt_band),
                (f"b_{band}_displacement", b_band),
                (f"e_{band}_displacement", e_band),
                (f"b_{band}_error", b_error),
                (f"e_{band}_error", e_error),
            ):
                save_mesh(mesh, mesh_dir / f"{name}.obj")
            panel = output / "panels" / f"{index:02d}_{sample_id}_{band}.png"
            _render(
                (gt_band, b_band, e_band, b_error, e_error),
                (f"GT {band}", f"B {band}", f"E {band}", f"B err {band}", f"E err {band}"),
                cameras,
                panel,
                args.image_size,
            )
            band_panels[band] = str(panel.relative_to(output))
        records.append(
            {
                **selection,
                "views": list(args.views),
                "full_panel": str(full_panel.relative_to(output)),
                "band_panels": band_panels,
                "mesh_directory": str(mesh_dir.relative_to(output)),
            }
        )
        print(f"rendered {sample_id}", flush=True)
    manifest = {
        "read_only": True,
        "objective_selection": True,
        "matched_camera_projection_scale_lighting": True,
        "chebyshev_order": args.chebyshev_order,
        "records": records,
    }
    _write = output / "comparison_manifest.json"
    _write.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
