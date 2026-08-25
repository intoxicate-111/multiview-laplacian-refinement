#!/usr/bin/env python3
from __future__ import annotations

"""Export all test meshes reconstructed by the frozen direct-vertex Arm E."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from mlr.data import Mesh
from mlr.io import save_mesh
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


ARM = "E_direct_vertex_residual"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--arm-e-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    report = args.arm_e_report.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    shard = json.loads(
        (report / "shards" / f"{ARM}.json").read_text(encoding="utf-8")
    )
    archived_ids = [str(value) for value in shard["split_ids"]["test"]]
    metrics_by_id = {
        str(row["sample_id"]): row
        for row in shard["rows"]
        if row["split"] == "test" and row["arm"] == ARM
    }
    predictions = np.load(
        report / "shards" / f"{ARM}_prediction_arrays.npz"
    )["test_prediction"].astype(np.float64)

    dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), "test")
    if list(dataset.sample_ids) != archived_ids:
        raise RuntimeError("Archived prediction order does not match the test split")

    rows: list[dict[str, object]] = []
    start = 0
    for index in range(len(dataset)):
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        vertices = np.asarray(static["vertices"], dtype=np.float64)
        faces = np.asarray(static["faces"], dtype=np.int64)
        stop = start + len(vertices)
        if stop > len(predictions):
            raise RuntimeError("Archived prediction array is shorter than the test split")
        refined = Mesh(vertices + predictions[start:stop], faces.copy()).ensure_normals()
        mesh_path = output / f"{index:02d}_{sample_id}__arm_e_refined.obj"
        save_mesh(refined, mesh_path)
        metric = metrics_by_id[sample_id]
        rows.append(
            {
                "test_index": index,
                "sample_id": sample_id,
                "mesh": mesh_path.name,
                "vertices": refined.num_vertices,
                "faces": refined.num_faces,
                "sha256": _sha256(mesh_path),
                "refined_chamfer": metric["refined_chamfer"],
                "relative_chamfer_gain": metric["relative_chamfer_gain"],
                "same_index_recovered_vertex_rms": metric[
                    "same_index_recovered_vertex_rms"
                ],
            }
        )
        start = stop

    if start != len(predictions) or len(rows) != 50:
        raise RuntimeError(
            f"Export contract failed: consumed={start}/{len(predictions)}, rows={len(rows)}"
        )

    fields = list(rows[0])
    with (output / "mesh_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / "mesh_manifest.json").write_text(
        json.dumps(
            {
                "arm": ARM,
                "split": "test",
                "count": len(rows),
                "prediction_source": str(report),
                "contract_audit": True,
                "meshes": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "meshes": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
