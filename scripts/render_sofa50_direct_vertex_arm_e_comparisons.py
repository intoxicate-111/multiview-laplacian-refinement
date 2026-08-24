#!/usr/bin/env python3
from __future__ import annotations

"""Render fixed, strictly matched Initial/A/B/C/D/E/Clean Sofa50 panels."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from diagnose_sofa50_exact_solve_visibility_sweep import component_labels, uniform_sparse_laplacian
from diagnose_sofa50_exact_target_oracle import _clean_mesh
from diagnose_sofa50_regularized_sparse_sweep import regularized_sparse_solve
from evaluate_sofa50_recovery_aware_ablation import EXTENSION_ARMS
from evaluate_sofa50_direct_vertex_arm_e import ARM_E
from mlr.data import Mesh
from mlr.io import save_mesh
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.synthetic import SyntheticRenderConfig, render_mesh_views_opengl
from render_sofa50_recovery_aware_comparisons import _camera


FIXED_INDICES = (0, 9, 19, 29, 39, 49)
LABELS = ("Initial", "Arm A", "Arm B", "Arm C", "Arm D", "Arm E", "Clean GT")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _render(entries: list[Mesh], cameras: list[tuple[int, Any]], path: Path, size: int, closeup: bool) -> None:
    label_height = 28
    config = SyntheticRenderConfig(width=size, height=size, render_mode="lit", normalize_mesh=False, backend="opengl")
    rendered = [render_mesh_views_opengl(mesh.ensure_normals(), [camera for _, camera in cameras], config) for mesh in entries]
    tile_size = size if not closeup else size
    canvas = Image.new("RGB", (len(entries) * tile_size, len(cameras) * (tile_size + label_height)), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    for column, images in enumerate(rendered):
        for row, ((view, _), (rgb, _mask, _depth)) in enumerate(zip(cameras, images, strict=True)):
            image = Image.fromarray(rgb)
            if closeup:
                # Identical deterministic central crop for every method in a row.
                lo, hi = size // 4, 3 * size // 4
                image = image.crop((lo, lo, hi, hi)).resize((size, size), Image.Resampling.LANCZOS)
            x, y = column * tile_size, row * (tile_size + label_height)
            draw.text((x + 5, y + 6), f"{LABELS[column]} | view {view:02d}", fill=(245, 245, 245))
            canvas.paste(image, (x, y + label_height))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ad-report-dir", required=True, type=Path)
    parser.add_argument("--ae-report-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--views", type=int, nargs="+", default=(0, 7))
    args = parser.parse_args()
    dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), "test")
    if len(dataset) != 50:
        raise RuntimeError("Expected the frozen 50-sample test split")
    ad_dir = args.ad_report_dir.resolve()
    ae_dir = args.ae_report_dir.resolve()
    output = args.output_dir.resolve()

    rows: dict[str, list[dict[str, Any]]] = {}
    predictions: dict[str, np.ndarray] = {}
    for arm in EXTENSION_ARMS:
        payload = _read(ad_dir / "shards" / f"{arm}.json")
        rows[arm] = [row for row in payload["rows"] if row["split"] == "test"]
        predictions[arm] = np.load(ad_dir / "shards" / f"{arm}_prediction_arrays.npz")["test_prediction"].astype(np.float64)
    e_payload = _read(ae_dir / "shards" / f"{ARM_E}.json")
    rows[ARM_E] = [row for row in e_payload["rows"] if row["split"] == "test"]
    predictions[ARM_E] = np.load(ae_dir / "shards" / f"{ARM_E}_prediction_arrays.npz")["test_prediction"].astype(np.float64)
    for arm in (*EXTENSION_ARMS, ARM_E):
        if [row["sample_id"] for row in rows[arm]] != list(dataset.sample_ids):
            raise RuntimeError(f"{arm}: sample ordering mismatch")

    starts: dict[str, list[int]] = {}
    for arm in (*EXTENSION_ARMS, ARM_E):
        counts = [int(row["vertices"]) for row in rows[arm]]
        starts[arm] = list(np.cumsum([0, *counts[:-1]]))
        if sum(counts) != len(predictions[arm]):
            raise RuntimeError(f"{arm}: archived prediction length mismatch")

    records: list[dict[str, Any]] = []
    for index in FIXED_INDICES:
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        initial = Mesh(np.asarray(static["vertices"], dtype=np.float64), np.asarray(static["faces"], dtype=np.int64)).ensure_normals()
        clean = _clean_mesh(static)
        laplacian, lap_data = uniform_sparse_laplacian(initial.faces, initial.num_vertices)
        component_count, labels = component_labels(lap_data)
        recovered: dict[str, Mesh] = {}
        for arm in EXTENSION_ARMS:
            start = starts[arm][index]
            stop = start + initial.num_vertices
            vertices, _audit = regularized_sparse_solve(
                laplacian, predictions[arm][start:stop], initial.vertices,
                labels, component_count, float(rows[arm][index]["lambda"]),
                atol=1e-12, btol=1e-12, maxiter=100000,
            )
            recovered[arm] = Mesh(vertices, initial.faces.copy()).ensure_normals()
        start = starts[ARM_E][index]
        direct = initial.vertices + predictions[ARM_E][start:start + initial.num_vertices]
        recovered[ARM_E] = Mesh(direct, initial.faces.copy()).ensure_normals()
        entries = [initial, *(recovered[arm] for arm in EXTENSION_ARMS), recovered[ARM_E], clean]
        cameras = [(view, _camera(static, view, args.image_size)) for view in args.views]
        panel = output / "panels" / f"{index:02d}_{sample_id}.png"
        closeup = output / "closeups" / f"{index:02d}_{sample_id}.png"
        _render(entries, cameras, panel, args.image_size, False)
        _render(entries, cameras, closeup, args.image_size, True)
        mesh_dir = output / "meshes" / sample_id
        mesh_dir.mkdir(parents=True, exist_ok=True)
        save_mesh(initial, mesh_dir / "initial.obj")
        for arm in EXTENSION_ARMS:
            save_mesh(recovered[arm], mesh_dir / f"{arm}.obj")
        save_mesh(recovered[ARM_E], mesh_dir / "E_direct_vertex_residual.obj")
        save_mesh(clean, mesh_dir / "clean_gt.obj")
        records.append({
            "fixed_test_index": index,
            "sample_id": sample_id,
            "views": list(args.views),
            "panel": str(panel.relative_to(output)),
            "closeup": str(closeup.relative_to(output)),
            "crop": "central [25%,75%] in both axes, identical for all methods",
            "labels": list(LABELS),
        })
        print(f"rendered {sample_id}", flush=True)
    manifest = {
        "format": "sofa50_v2_arm_a_to_e_matched_visuals_v1",
        "selection_rule": f"fixed zero-based test indices {list(FIXED_INDICES)}, declared independently of metrics",
        "camera_projection_scale_lighting_crop_matched": True,
        "records": records,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# Sofa50 A/B/C/D/E matched visualizations\n\n"
        "Panels are ordered `Initial | Arm A | Arm B | Arm C | Arm D | Arm E | Clean GT`. "
        "Every column in a row uses the same prepared camera, projection, scale, renderer, lighting, and crop. "
        "The subset was fixed by test index before examining Arm E metrics.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
