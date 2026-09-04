#!/usr/bin/env python3
"""Export the 50 paired Sofa50 A+E/B+E lambda=0.01 result meshes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import evaluate_sofa50_direct_lap_positional_matched_fusion as base
from diagnose_sofa50_exact_target_oracle import _clean_mesh
from diagnose_sofa50_frozen_hybrid_recovery import _pcg
from mlr.data import Mesh
from mlr.io import load_mesh, save_mesh
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


FUSION_LAMBDA = 0.01
METHODS = {
    "A_plus_E": (base.ARM_A, "A_plus_E_lambda1e2.obj"),
    "B_plus_E": (base.ARM_B, "B_plus_E_lambda1e2.obj"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--arm-ab-report", required=True, type=Path)
    parser.add_argument("--arm-e-report", required=True, type=Path)
    parser.add_argument("--evaluation-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def metric_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    labels = {
        "Direct-Lap A+E, lambda=0.01": "A_plus_E",
        "Proposed B+E, lambda=0.01": "B_plus_E",
    }
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {
        (row["sample_id"], labels[row["method"]]): row
        for row in rows
        if row["method"] in labels
    }
    if len(result) != 100:
        raise RuntimeError(f"Expected 100 paired metric rows, found {len(result)}")
    return result


def main() -> int:
    args = parse_args()
    if args.device != "cpu":
        raise RuntimeError("This export is intentionally restricted to local CPU")
    device = torch.device("cpu")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    a_payload = base.read_json(args.arm_ab_report / "shards" / f"{base.ARM_A}.json")
    b_payload = base.read_json(args.arm_ab_report / "shards" / f"{base.ARM_B}.json")
    e_payload = base.read_json(args.arm_e_report / "shards" / f"{base.ARM_E}.json")
    for arm, payload in (
        (base.ARM_A, a_payload),
        (base.ARM_B, b_payload),
        (base.ARM_E, e_payload),
    ):
        if payload["checkpoint_sha256"] != base.EXPECTED_SHA[arm]:
            raise RuntimeError(f"{arm}: prediction metadata checkpoint SHA mismatch")

    dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), "test")
    expected_ids = list(dataset.sample_ids)
    if len(expected_ids) != 50:
        raise RuntimeError(f"Expected 50 test samples, found {len(expected_ids)}")
    payload_rows = {
        base.ARM_A: base.split_rows(a_payload, "test"),
        base.ARM_B: base.split_rows(b_payload, "test"),
        base.ARM_E: base.split_rows(e_payload, "test"),
    }
    for arm, rows in payload_rows.items():
        if [str(row["sample_id"]) for row in rows] != expected_ids:
            raise RuntimeError(f"{arm}: IDs/order differ from manifest")

    predictions: dict[str, np.ndarray] = {}
    starts: dict[str, list[int]] = {}
    artifact_paths: dict[str, Path] = {}
    for arm, report in (
        (base.ARM_A, args.arm_ab_report),
        (base.ARM_B, args.arm_ab_report),
        (base.ARM_E, args.arm_e_report),
    ):
        prediction, _, path = base.prediction_array(report, arm, "test")
        predictions[arm] = prediction
        starts[arm] = base.starts(payload_rows[arm])
        artifact_paths[arm] = path

    archived_metrics = metric_rows(args.evaluation_report / "per_mesh_metrics.csv")
    records: list[dict[str, Any]] = []
    maximum_metric_vrms_error = 0.0
    maximum_roundtrip_vertex_error = 0.0
    for index, sample_id in enumerate(expected_ids):
        static = dataset.load_static(index)
        initial = Mesh(
            np.asarray(static["vertices"], dtype=np.float64),
            np.asarray(static["faces"], dtype=np.int64),
        ).ensure_normals()
        clean = _clean_mesh(static)
        count = initial.num_vertices
        e_start = starts[base.ARM_E][index]
        direct = initial.vertices + predictions[base.ARM_E][e_start : e_start + count]
        sample_dir = output / f"{index:02d}_{sample_id}"
        record: dict[str, Any] = {
            "index": index,
            "sample_id": sample_id,
            "object_id": base.object_id(sample_id),
            "vertices": initial.num_vertices,
            "faces": initial.num_faces,
            "lambda": FUSION_LAMBDA,
            "mesh_paths": {},
            "mesh_sha256": {},
            "solver": {},
            "metric_reproduction": {},
        }
        for label, (arm, filename) in METHODS.items():
            start = starts[arm][index]
            field = predictions[arm][start : start + count]
            recovered, solver = _pcg(field, direct, static, FUSION_LAMBDA, device)
            if not solver["pcg_converged"]:
                raise RuntimeError(f"{sample_id}/{label}: PCG did not converge")
            vrms = float(
                np.sqrt(np.mean(np.sum((recovered - clean.vertices) ** 2, axis=1)))
            )
            expected_vrms = float(
                archived_metrics[(sample_id, label)]["same_index_recovered_vertex_rms"]
            )
            vrms_error = abs(vrms - expected_vrms)
            maximum_metric_vrms_error = max(maximum_metric_vrms_error, vrms_error)
            if vrms_error > 1e-12:
                raise RuntimeError(
                    f"{sample_id}/{label}: reconstructed VRMS differs from evaluation"
                )
            mesh = Mesh(recovered, initial.faces.copy()).ensure_normals()
            path = sample_dir / filename
            save_mesh(mesh, path)
            loaded = load_mesh(path)
            if not np.array_equal(loaded.faces, initial.faces):
                raise RuntimeError(f"{sample_id}/{label}: OBJ face round trip changed topology")
            roundtrip_error = float(np.max(np.abs(loaded.vertices - recovered)))
            maximum_roundtrip_vertex_error = max(
                maximum_roundtrip_vertex_error, roundtrip_error
            )
            record["mesh_paths"][label] = str(path.relative_to(output))
            record["mesh_sha256"][label] = base.sha256_file(path)
            record["solver"][label] = solver
            record["metric_reproduction"][label] = {
                "computed_vrms": vrms,
                "archived_vrms": expected_vrms,
                "absolute_error": vrms_error,
                "obj_roundtrip_max_vertex_absolute_error": roundtrip_error,
                "faces_exact": True,
            }
        records.append(record)
        print(f"exported paired meshes {index + 1}/50 {sample_id}", flush=True)

    manifest = {
        "format": "sofa50_direct_lap_positional_matched_fusion_meshes_v1",
        "split": "test",
        "samples": len(records),
        "meshes_per_sample": 2,
        "total_obj_files": 2 * len(records),
        "lambda": FUSION_LAMBDA,
        "operator": "L_U=I-D^{-1}A; (L_U^T L_U+0.01I)V=L_U^T delta+0.01V_P",
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": base.sha256_file(args.manifest.resolve()),
        "source_evaluation_report": str(args.evaluation_report.resolve()),
        "source_prediction_artifacts": {
            arm: {"path": str(path.resolve()), "sha256": base.sha256_file(path)}
            for arm, path in artifact_paths.items()
        },
        "maximum_metric_vrms_reproduction_error": maximum_metric_vrms_error,
        "maximum_obj_roundtrip_vertex_absolute_error": maximum_roundtrip_vertex_error,
        "all_faces_exact": True,
        "records": records,
    }
    base.write_json(output / "MANIFEST.json", manifest)
    readme = [
        "# Sofa50 paired A+E/B+E result meshes at lambda=0.01",
        "",
        "This directory contains the exact 50-test-sample frozen fusion outputs used by the matched comparison.",
        "Each sample directory contains `A_plus_E_lambda1e2.obj` and `B_plus_E_lambda1e2.obj` with the original input connectivity.",
        "No evaluator, model inference, training, or HPC job was run during export; vertices were reconstructed from the archived A/B/E predictions with the same float64 PCG fusion solve.",
        "",
        f"- OBJ files: `{2 * len(records)}`.",
        f"- Maximum VRMS reproduction error: `{maximum_metric_vrms_error:.3e}`.",
        f"- Maximum OBJ vertex round-trip absolute error: `{maximum_roundtrip_vertex_error:.3e}`.",
        "- Exact relative paths, SHA-256 hashes, sample IDs, topology counts, and solver audits are in `MANIFEST.json`.",
    ]
    (output / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "samples": len(records),
                "obj_files": 2 * len(records),
                "maximum_metric_vrms_reproduction_error": maximum_metric_vrms_error,
                "maximum_obj_roundtrip_vertex_absolute_error": maximum_roundtrip_vertex_error,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
