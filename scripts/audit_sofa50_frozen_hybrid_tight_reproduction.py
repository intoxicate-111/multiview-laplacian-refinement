#!/usr/bin/env python3
from __future__ import annotations

"""Test whether the frozen B/E hybrid survives the requested tighter PCG solve."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from diagnose_sofa50_exact_target_oracle import _clean_mesh, _geometry_row
from mlr.data import Mesh
from mlr.learned_laplacian.differentiable_sparse_recovery import (
    differentiable_regularized_sparse_recovery_with_audit,
)
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


ARM_B = "B_lap_plus_refine"
ARM_E = "E_direct_vertex_residual"
ARM_H = "Hybrid_B_laplacian_E_anchor"


def _payload(report: Path, arm: str) -> dict[str, object]:
    path = report / "shards" / f"{arm}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("arm") != arm:
        raise RuntimeError(f"Archived arm mismatch for {arm}.")
    return value


def _archived_predictions(report: Path, arm: str, split: str) -> np.ndarray:
    path = report / "shards" / f"{arm}_prediction_arrays.npz"
    return np.load(path)[f"{split}_prediction"].astype(np.float64)


def _starts(rows: list[dict[str, object]], array: np.ndarray) -> list[int]:
    counts = [int(row["vertices"]) for row in rows]
    if sum(counts) != len(array):
        raise RuntimeError("Archived prediction length does not match row metadata.")
    return list(np.cumsum([0, *counts[:-1]]))


def _read_rows(path: Path, split: str) -> dict[str, dict[str, object]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            str(row["sample_id"]): dict(row)
            for row in csv.DictReader(handle)
            if row["split"] == split and row["arm"] == ARM_H
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--arm-b-report", required=True, type=Path)
    parser.add_argument("--arm-e-report", required=True, type=Path)
    parser.add_argument("--frozen-report", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("validation", "test"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), args.split)
    b_payload = _payload(args.arm_b_report.resolve(), ARM_B)
    e_payload = _payload(args.arm_e_report.resolve(), ARM_E)
    b_rows = [dict(row) for row in b_payload["rows"] if row["split"] == args.split]
    e_rows = [dict(row) for row in e_payload["rows"] if row["split"] == args.split]
    expected = list(dataset.sample_ids)
    if [row["sample_id"] for row in b_rows] != expected:
        raise RuntimeError("Arm-B archive order does not match the manifest.")
    if [row["sample_id"] for row in e_rows] != expected:
        raise RuntimeError("Arm-E archive order does not match the manifest.")
    b_array = _archived_predictions(args.arm_b_report.resolve(), ARM_B, args.split)
    e_array = _archived_predictions(args.arm_e_report.resolve(), ARM_E, args.split)
    b_starts, e_starts = _starts(b_rows, b_array), _starts(e_rows, e_array)
    frozen_rows = _read_rows(
        args.frozen_report.resolve() / "matched_per_sample.csv", args.split
    )
    if set(frozen_rows) != set(expected):
        raise RuntimeError("Frozen-hybrid reference IDs do not match the manifest.")

    device = torch.device(args.device)
    rows: list[dict[str, object]] = []
    for index in range(len(dataset)):
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        vertices = np.asarray(static["vertices"], dtype=np.float64)
        faces = np.asarray(static["faces"], dtype=np.int64)
        count = len(vertices)
        prediction = torch.as_tensor(
            b_array[b_starts[index] : b_starts[index] + count],
            dtype=torch.float64,
            device=device,
        )
        direct = torch.as_tensor(
            vertices + e_array[e_starts[index] : e_starts[index] + count],
            dtype=torch.float64,
            device=device,
        )
        edge_index = torch.as_tensor(
            static["edge_index"], dtype=torch.long, device=device
        )
        degree = torch.as_tensor(
            static["vertex_degree"], dtype=torch.float64, device=device
        )
        solves = {}
        for name, tolerance in (("loose", 1e-4), ("tight", 1e-8)):
            recovered, audit = differentiable_regularized_sparse_recovery_with_audit(
                prediction,
                direct,
                edge_index,
                degree,
                regularization=3e-2,
                maximum_iterations=2048,
                tolerance=tolerance,
            )
            if not audit.converged:
                raise RuntimeError(f"{sample_id}: {name} PCG failed: {audit}")
            solves[name] = (recovered.detach().cpu().numpy(), audit)
        loose, loose_audit = solves["loose"]
        tight, tight_audit = solves["tight"]
        initial = Mesh(vertices, faces).ensure_normals()
        clean = _clean_mesh(static)
        tight_metric = _geometry_row(
            args.split,
            sample_id,
            "tight_frozen_hybrid",
            Mesh(tight, faces.copy()).ensure_normals(),
            clean,
            initial,
        )
        reference = frozen_rows[sample_id]
        rows.append(
            {
                "split": args.split,
                "sample_id": sample_id,
                "loose_iterations": loose_audit.iterations,
                "tight_iterations": tight_audit.iterations,
                "loose_relative_residual": loose_audit.relative_residual,
                "tight_relative_residual": tight_audit.relative_residual,
                "tight_minus_loose_vertex_rms": float(
                    np.sqrt(np.mean(np.sum(np.square(tight - loose), axis=1)))
                ),
                "tight_minus_loose_max_coordinate": float(
                    np.max(np.abs(tight - loose))
                ),
                "reference_frozen_chamfer": float(reference["refined_chamfer"]),
                "tight_chamfer": float(tight_metric["chamfer"]),
                "tight_minus_reference_chamfer": float(tight_metric["chamfer"])
                - float(reference["refined_chamfer"]),
            }
        )
        print(f"{args.split} {index + 1}/{len(dataset)} {sample_id}", flush=True)

    differences = np.asarray(
        [float(row["tight_minus_reference_chamfer"]) for row in rows]
    )
    vertex_differences = np.asarray(
        [float(row["tight_minus_loose_vertex_rms"]) for row in rows]
    )
    summary = {
        "split": args.split,
        "samples": len(rows),
        "lambda": 3e-2,
        "old_tolerance": 1e-4,
        "requested_tolerance": 1e-8,
        "maximum_iterations": 2048,
        "reference_mean_chamfer": float(
            np.mean([float(row["reference_frozen_chamfer"]) for row in rows])
        ),
        "tight_mean_chamfer": float(
            np.mean([float(row["tight_chamfer"]) for row in rows])
        ),
        "mean_chamfer_difference": float(differences.mean()),
        "maximum_absolute_per_sample_chamfer_difference": float(
            np.max(np.abs(differences))
        ),
        "mean_tight_minus_loose_vertex_rms": float(vertex_differences.mean()),
        "maximum_tight_minus_loose_vertex_rms": float(vertex_differences.max()),
        "maximum_tight_residual": float(
            max(float(row["tight_relative_residual"]) for row in rows)
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
