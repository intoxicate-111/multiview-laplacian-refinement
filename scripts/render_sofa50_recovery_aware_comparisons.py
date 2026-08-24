#!/usr/bin/env python3
from __future__ import annotations

"""Export and render all Sofa50 recovery-aware Arm A/B test comparisons."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw

from diagnose_sofa50_exact_solve_visibility_sweep import (
    component_labels,
    uniform_sparse_laplacian,
)
from diagnose_sofa50_exact_target_oracle import _clean_mesh
from diagnose_sofa50_regularized_sparse_sweep import regularized_sparse_solve
from mlr.data import Camera, Mesh
from mlr.io import save_mesh
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.synthetic import SyntheticRenderConfig, render_mesh_views_opengl


ARMS = ("A_lap_only", "B_lap_plus_refine")
ARM_FILES = {
    "A_lap_only": "arm_a_refined.obj",
    "B_lap_plus_refine": "arm_b_refined.obj",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--views", type=int, nargs="+", default=(0, 7))
    parser.add_argument("--expected-count", type=int, default=50)
    parser.add_argument("--overview-size", type=int, default=640)
    return parser.parse_args()


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _camera(static: Mapping[str, Any], view: int, image_size: int) -> Camera:
    intrinsics = _numpy(static["intrinsics"])[view].astype(np.float64, copy=True)
    source_width = int(static.get("image_width", 0) or 0)
    source_height = int(static.get("image_height", 0) or 0)
    if source_width <= 0 or source_height <= 0:
        prepared_size = static.get("prepared_image_size")
        if prepared_size is not None:
            source_width = source_height = int(prepared_size)
        else:
            root = Path(str(static["_dataset_root"]))
            image_path = Path(str(static["image_paths"][view]))
            if not image_path.is_absolute():
                image_path = root / image_path
            with Image.open(image_path) as image:
                source_width, source_height = image.size
    intrinsics[0] *= image_size / source_width
    intrinsics[1] *= image_size / source_height
    intrinsics[2, 2] = 1.0
    extrinsics = _numpy(static["extrinsics"])[view].astype(np.float64, copy=False)
    return Camera(
        intrinsics=intrinsics,
        rotation=extrinsics[:3, :3],
        translation=extrinsics[:3, 3],
        image_size=(image_size, image_size),
        name=f"prepared_view_{view:02d}",
    )


def _render_panel(
    entries: list[tuple[str, Mesh]],
    cameras: list[tuple[int, Camera]],
    output_path: Path,
    image_size: int,
) -> None:
    label_height = 27
    canvas = Image.new(
        "RGB",
        (len(entries) * image_size, len(cameras) * (image_size + label_height)),
        (18, 18, 18),
    )
    draw = ImageDraw.Draw(canvas)
    config = SyntheticRenderConfig(
        width=image_size,
        height=image_size,
        render_mode="lit",
        normalize_mesh=False,
        backend="opengl",
    )
    for column, (label, mesh) in enumerate(entries):
        rendered = render_mesh_views_opengl(
            mesh.ensure_normals(), [camera for _, camera in cameras], config
        )
        for row, ((view, _camera_value), (rgb, _mask, _depth)) in enumerate(
            zip(cameras, rendered, strict=True)
        ):
            x = column * image_size
            y = row * (image_size + label_height)
            draw.text((x + 5, y + 6), f"{label} | VIEW {view:02d}", fill=(245, 245, 245))
            canvas.paste(Image.fromarray(rgb), (x, y + label_height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _overview(images: list[tuple[str, Path]], output_dir: Path, thumb_width: int) -> list[str]:
    paths: list[str] = []
    per_sheet = 10
    columns = 2
    margin = 10
    title_height = 25
    for sheet_start in range(0, len(images), per_sheet):
        selected = images[sheet_start : sheet_start + per_sheet]
        opened: list[tuple[str, Image.Image]] = []
        for sample_id, path in selected:
            source = Image.open(path).convert("RGB")
            height = round(source.height * thumb_width / source.width)
            opened.append((sample_id, source.resize((thumb_width, height))))
        rows = (len(opened) + columns - 1) // columns
        cell_height = max(image.height for _, image in opened) + title_height
        canvas = Image.new(
            "RGB",
            (
                columns * thumb_width + (columns + 1) * margin,
                rows * cell_height + (rows + 1) * margin,
            ),
            (16, 16, 16),
        )
        draw = ImageDraw.Draw(canvas)
        for index, (sample_id, image) in enumerate(opened):
            column = index % columns
            row = index // columns
            x = margin + column * (thumb_width + margin)
            y = margin + row * (cell_height + margin)
            ordinal = sheet_start + index + 1
            draw.text((x + 3, y + 5), f"{ordinal:02d}  {sample_id}", fill=(245, 245, 245))
            canvas.paste(image, (x, y + title_height))
        path = output_dir / f"overview_{sheet_start + 1:02d}_{sheet_start + len(selected):02d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path, quality=92)
        paths.append(str(path))
    return paths


def main() -> None:
    args = _parse_args()
    manifest = args.manifest.resolve()
    report_dir = args.report_dir.resolve()
    output_dir = args.output_dir.resolve()
    dataset = PreparedMeshDataset.from_manifest(manifest, "test")
    if len(dataset) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} test samples, found {len(dataset)}")
    summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
    regularization = float(summary["lambda_selection"]["selected_lambda"])

    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    predictions: dict[str, np.ndarray] = {}
    for arm in ARMS:
        shard_path = report_dir / "shards" / f"{arm}.json"
        shard = json.loads(shard_path.read_text(encoding="utf-8"))
        rows = [row for row in shard["rows"] if row["split"] == "test"]
        if len(rows) != len(dataset):
            raise ValueError(f"{arm}: expected {len(dataset)} test rows, found {len(rows)}")
        if [row["sample_id"] for row in rows] != list(dataset.sample_ids):
            raise RuntimeError(f"{arm}: test row ordering differs from prepared dataset")
        arrays = np.load(report_dir / "shards" / f"{arm}_prediction_arrays.npz")
        prediction = arrays["test_prediction"].astype(np.float64, copy=False)
        if len(prediction) != sum(int(row["vertices"]) for row in rows):
            raise RuntimeError(f"{arm}: concatenated prediction length mismatch")
        rows_by_arm[arm] = rows
        predictions[arm] = prediction

    offsets = {arm: 0 for arm in ARMS}
    records: list[dict[str, Any]] = []
    rendered: list[tuple[str, Path]] = []
    for index in range(len(dataset)):
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        initial = Mesh(
            _numpy(static["vertices"]).astype(np.float64, copy=False),
            _numpy(static["faces"]).astype(np.int64, copy=False),
        ).ensure_normals()
        clean = _clean_mesh(static)
        laplacian, lap_data = uniform_sparse_laplacian(initial.faces, initial.num_vertices)
        component_count, labels = component_labels(lap_data)
        recovered_meshes: dict[str, Mesh] = {}
        solver_audits: dict[str, dict[str, Any]] = {}
        for arm in ARMS:
            start = offsets[arm]
            stop = start + initial.num_vertices
            prediction = predictions[arm][start:stop]
            offsets[arm] = stop
            recovered, solver = regularized_sparse_solve(
                laplacian,
                prediction,
                initial.vertices,
                labels,
                component_count,
                regularization,
                atol=1e-12,
                btol=1e-12,
                maxiter=100000,
            )
            recovered_meshes[arm] = Mesh(recovered, initial.faces.copy()).ensure_normals()
            solver_audits[arm] = solver

        sample_dir = output_dir / "meshes" / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "gt": sample_dir / "gt.obj",
            "coarse": sample_dir / "coarse.obj",
            "A_lap_only": sample_dir / ARM_FILES["A_lap_only"],
            "B_lap_plus_refine": sample_dir / ARM_FILES["B_lap_plus_refine"],
        }
        save_mesh(clean, paths["gt"])
        save_mesh(initial, paths["coarse"])
        for arm in ARMS:
            save_mesh(recovered_meshes[arm], paths[arm])

        row_a = rows_by_arm["A_lap_only"][index]
        row_b = rows_by_arm["B_lap_plus_refine"][index]
        entries = [
            ("GT", clean),
            (f"COARSE CD={float(row_a['initial_chamfer']):.5f}", initial),
            (f"ARM A CD={float(row_a['refined_chamfer']):.5f}", recovered_meshes["A_lap_only"]),
            (f"ARM B CD={float(row_b['refined_chamfer']):.5f}", recovered_meshes["B_lap_plus_refine"]),
        ]
        cameras = [(view, _camera(static, view, args.image_size)) for view in args.views]
        panel = output_dir / "panels" / f"{index + 1:02d}_{sample_id}.png"
        _render_panel(entries, cameras, panel, args.image_size)
        rendered.append((sample_id, panel))
        records.append(
            {
                "index": index,
                "sample_id": sample_id,
                "views": list(args.views),
                "panel": str(panel.relative_to(output_dir)),
                "mesh_paths": {key: str(path.relative_to(output_dir)) for key, path in paths.items()},
                "metrics": {
                    "initial_chamfer": float(row_a["initial_chamfer"]),
                    "arm_a_refined_chamfer": float(row_a["refined_chamfer"]),
                    "arm_b_refined_chamfer": float(row_b["refined_chamfer"]),
                    "arm_a_vertex_rms": float(row_a["same_index_recovered_vertex_rms"]),
                    "arm_b_vertex_rms": float(row_b["same_index_recovered_vertex_rms"]),
                },
                "solver": solver_audits,
            }
        )
        print(f"[{index + 1:02d}/{len(dataset):02d}] {sample_id}", flush=True)

    for arm in ARMS:
        if offsets[arm] != len(predictions[arm]):
            raise RuntimeError(f"{arm}: did not consume every archived prediction")
    overview_paths = _overview(rendered, output_dir / "overviews", args.overview_size)
    payload = {
        "format": "sofa50_recovery_aware_comparison_bundle_v1",
        "count": len(records),
        "labels": ["GT", "COARSE", "ARM A LAP-ONLY", "ARM B LAP+REFINE"],
        "views": list(args.views),
        "image_size": args.image_size,
        "lambda": regularization,
        "source_manifest": str(manifest),
        "source_report": str(report_dir / "FINAL_REPORT.md"),
        "source_report_sha256": _sha256(report_dir / "FINAL_REPORT.md"),
        "prediction_source": "archived final evaluation arrays; no model inference rerun",
        "recovery": "same regularized scipy LSMR solve as final evaluation",
        "overviews": [str(Path(path).relative_to(output_dir)) for path in overview_paths],
        "samples": records,
    }
    (output_dir / "comparison_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# Sofa50 recovery-aware comparison images\n\n"
        "Each panel uses the same two prepared cameras for GT, coarse, Arm A, and Arm B. "
        "The meshes are reconstructed from the archived evaluation predictions with the "
        "same regularized sparse solver (`lambda=1e-2`); model inference was not rerun.\n\n"
        "- `panels/`: 50 full-resolution two-view comparisons\n"
        "- `overviews/`: five 10-sample contact sheets\n"
        "- `meshes/`: GT, coarse, Arm A refined, and Arm B refined OBJ files\n"
        "- `comparison_manifest.json`: provenance, metrics, and file mapping\n",
        encoding="utf-8",
    )
    print(f"Rendered {len(records)} comparisons to {output_dir}")


if __name__ == "__main__":
    main()
