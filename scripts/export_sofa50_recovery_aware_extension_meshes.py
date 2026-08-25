#!/usr/bin/env python3
from __future__ import annotations

"""Export Arm C/D recovered OBJ meshes into an existing A/B comparison bundle."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from diagnose_sofa50_exact_solve_visibility_sweep import (
    component_labels,
    uniform_sparse_laplacian,
)
from diagnose_sofa50_exact_target_oracle import _clean_mesh
from diagnose_sofa50_regularized_sparse_sweep import regularized_sparse_solve
from mlr.data import Mesh
from mlr.io import save_mesh
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


ARMS = {
    "C_lap_plus_refine_lambda1e-3": "arm_c_refined.obj",
    "D_lap_plus_refine_lambda1e-4": "arm_d_refined.obj",
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--comparison-dir", required=True, type=Path)
    parser.add_argument("--expected-count", type=int, default=50)
    args = parser.parse_args()

    dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), "test")
    if len(dataset) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count} test samples, found {len(dataset)}"
        )
    report_dir = args.report_dir.resolve()
    comparison_dir = args.comparison_dir.resolve()

    rows: dict[str, list[dict[str, Any]]] = {}
    predictions: dict[str, np.ndarray] = {}
    starts: dict[str, list[int]] = {}
    for arm in ARMS:
        shard = _read(report_dir / "shards" / f"{arm}.json")
        arm_rows = [row for row in shard["rows"] if row["split"] == "test"]
        if [row["sample_id"] for row in arm_rows] != list(dataset.sample_ids):
            raise RuntimeError(f"{arm}: archived rows do not match the test ordering")
        prediction = np.load(
            report_dir / "shards" / f"{arm}_prediction_arrays.npz"
        )["test_prediction"].astype(np.float64, copy=False)
        counts = [int(row["vertices"]) for row in arm_rows]
        if sum(counts) != len(prediction):
            raise RuntimeError(f"{arm}: concatenated prediction length mismatch")
        rows[arm] = arm_rows
        predictions[arm] = prediction
        starts[arm] = list(np.cumsum([0, *counts[:-1]]))

    manifest_path = comparison_dir / "comparison_manifest.json"
    comparison_manifest = _read(manifest_path)
    records = comparison_manifest.get("samples")
    if not isinstance(records, list) or len(records) != len(dataset):
        raise RuntimeError("Existing comparison manifest does not cover the full test split")
    records_by_id = {str(record["sample_id"]): record for record in records}

    maximum_vertex_rms_error = 0.0
    exported = 0
    for index in range(len(dataset)):
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        if sample_id not in records_by_id:
            raise RuntimeError(f"Existing comparison bundle is missing {sample_id}")
        initial = Mesh(
            np.asarray(static["vertices"], dtype=np.float64),
            np.asarray(static["faces"], dtype=np.int64),
        ).ensure_normals()
        clean = _clean_mesh(static)
        laplacian, lap_data = uniform_sparse_laplacian(
            initial.faces, initial.num_vertices
        )
        component_count, labels = component_labels(lap_data)
        sample_dir = comparison_dir / "meshes" / sample_id
        if not sample_dir.is_dir():
            raise RuntimeError(f"Existing sample directory is missing: {sample_dir}")

        record = records_by_id[sample_id]
        mesh_paths = record.setdefault("mesh_paths", {})
        metrics = record.setdefault("metrics", {})
        solver = record.setdefault("solver", {})
        for arm, filename in ARMS.items():
            start = starts[arm][index]
            stop = start + initial.num_vertices
            recovered, audit = regularized_sparse_solve(
                laplacian,
                predictions[arm][start:stop],
                initial.vertices,
                labels,
                component_count,
                float(rows[arm][index]["lambda"]),
                atol=1e-12,
                btol=1e-12,
                maxiter=100000,
            )
            if not bool(audit["all_converged"]):
                raise RuntimeError(f"{arm}/{sample_id}: sparse solve did not converge")
            vertex_rms = float(
                np.sqrt(np.mean(np.sum((recovered - clean.vertices) ** 2, axis=1)))
            )
            archived_vertex_rms = float(
                rows[arm][index]["same_index_recovered_vertex_rms"]
            )
            vertex_rms_error = abs(vertex_rms - archived_vertex_rms)
            maximum_vertex_rms_error = max(maximum_vertex_rms_error, vertex_rms_error)
            if vertex_rms_error > 1e-10:
                raise RuntimeError(
                    f"{arm}/{sample_id}: recovered vertex RMS differs from archived "
                    f"evaluation ({vertex_rms} vs {archived_vertex_rms})"
                )
            path = sample_dir / filename
            save_mesh(Mesh(recovered, initial.faces.copy()).ensure_normals(), path)
            mesh_paths[arm] = str(path.relative_to(comparison_dir))
            metrics[f"{arm}_refined_chamfer"] = float(
                rows[arm][index]["refined_chamfer"]
            )
            metrics[f"{arm}_vertex_rms"] = archived_vertex_rms
            solver[arm] = audit
            exported += 1
        print(f"[{index + 1:02d}/{len(dataset):02d}] {sample_id}", flush=True)

    comparison_manifest["format"] = "sofa50_recovery_aware_comparison_bundle_v2"
    comparison_manifest["extension_report"] = str(report_dir / "FINAL_REPORT.md")
    comparison_manifest["extension_meshes"] = {
        "arms": list(ARMS),
        "files_per_arm": len(dataset),
        "recovery": "archived prediction plus the exact per-arm regularized sparse solve",
        "maximum_vertex_rms_reproduction_error": maximum_vertex_rms_error,
    }
    manifest_path.write_text(
        json.dumps(comparison_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readme_path = comparison_dir / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    note = (
        "\nArm C (`lambda=1e-3`) and Arm D (`lambda=1e-4`) refined OBJ meshes "
        "were reconstructed from the archived C/D predictions with each arm's frozen "
        "regularized sparse solver and added as `arm_c_refined.obj` and "
        "`arm_d_refined.obj` in every sample directory.\n"
    )
    if note.strip() not in readme:
        readme_path.write_text(readme.rstrip() + "\n" + note, encoding="utf-8")
    print(
        f"Exported {exported} C/D meshes; maximum vertex-RMS reproduction error "
        f"{maximum_vertex_rms_error:.3e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
