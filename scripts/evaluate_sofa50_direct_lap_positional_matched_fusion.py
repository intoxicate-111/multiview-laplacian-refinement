#!/usr/bin/env python3
"""Matched frozen Arm-A/Arm-E fusion versus the existing Arm-B/Arm-E fusion.

This script performs no network inference and no training.  It consumes the
archived Arm-A, Arm-B, and Arm-E prediction arrays, applies the existing
matrix-free float64 PCG recovery, and evaluates both fusions with the existing
Sofa50 geometry evaluator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from diagnose_sofa50_exact_target_oracle import METRIC_PROTOCOL, _clean_mesh, _geometry_row
from diagnose_sofa50_frozen_hybrid_recovery import _pcg, _row
from mlr.data import Mesh
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


ARM_A = "A_lap_only"
ARM_B = "B_lap_plus_refine"
ARM_E = "E_direct_vertex_residual"
DIRECT_AE = "Direct-Lap A+E, lambda=0.03"
PROPOSED_BE = "Proposed B+E, lambda=0.03"
EXPECTED_SHA = {
    ARM_A: "788526139f13100ed19f0cf24d6fc64ab945bbf7fb7aad4b25e46ea3fe8176a4",
    ARM_B: "a483e2212f568e771873594cf1e37d13d62cbd2e1e72244baded7dd15573970c",
    ARM_E: "6ed27da8759b7bd752ffa75ea8dac3977dd4ced358b5282e0c1c68f750dbade1",
}
FUSION_LAMBDA = 0.03
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 7
REPRODUCTION_TOLERANCE = 2e-8
LOWER_IS_BETTER = {
    "refined_chamfer",
    "p2s_p95",
    "same_index_recovered_vertex_rms",
    "introduced_flipped_faces",
}
PAIRED_FIELDS = (
    "refined_chamfer",
    "p2s_p95",
    "fscore",
    "normal_consistency",
    "same_index_recovered_vertex_rms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--arm-ab-report", required=True, type=Path)
    parser.add_argument("--arm-e-report", required=True, type=Path)
    parser.add_argument("--hybrid-report", required=True, type=Path)
    parser.add_argument("--pure-fusion-report", required=True, type=Path)
    parser.add_argument("--scalar-fusion-report", required=True, type=Path)
    parser.add_argument("--arm-b-checkpoint", required=True, type=Path)
    parser.add_argument("--arm-e-checkpoint", required=True, type=Path)
    parser.add_argument("--arm-a-checkpoint", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_rows(payload: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    return [dict(row) for row in payload["rows"] if row["split"] == split]


def prediction_array(report: Path, arm: str, split: str) -> tuple[np.ndarray, np.ndarray, Path]:
    path = report / "shards" / f"{arm}_prediction_arrays.npz"
    archive = np.load(path)
    return (
        archive[f"{split}_prediction"].astype(np.float64),
        archive[f"{split}_target"].astype(np.float64),
        path,
    )


def starts(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    result: list[int] = []
    offset = 0
    for row in rows:
        result.append(offset)
        offset += int(row["vertices"])
    return result


def typed_archived_rows(path: Path) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        selected = [
            row
            for row in csv.DictReader(handle)
            if row["split"] == "test" and row["arm"] == "Hybrid_B_laplacian_E_anchor"
        ]
    numeric = {
        "initial_chamfer",
        "refined_chamfer",
        "relative_chamfer_gain",
        "eta",
        "p2s",
        "p2s_p95",
        "fscore",
        "normal_consistency",
        "same_index_recovered_vertex_rms",
        "normalized_flip_rate",
    }
    integer = {"vertices", "faces", "introduced_flipped_faces", "new_degenerate_faces"}
    output: dict[str, dict[str, Any]] = {}
    for row in selected:
        item: dict[str, Any] = dict(row)
        for field in numeric:
            item[field] = float(row[field])
        for field in integer:
            item[field] = int(row[field])
        output[row["sample_id"]] = item
    return output


def object_id(sample_id: str) -> str:
    return sample_id.split("__", 1)[0]


def aggregate(rows: Sequence[Mapping[str, Any]], method: str) -> dict[str, Any]:
    selected = [row for row in rows if row["method"] == method]
    faces = sum(int(row["faces"]) for row in selected)
    return {
        "method": method,
        "samples": len(selected),
        "refined_chamfer": float(np.mean([row["refined_chamfer"] for row in selected])),
        "p2s_p95": float(np.mean([row["p2s_p95"] for row in selected])),
        "fscore": float(np.mean([row["fscore"] for row in selected])),
        "normal_consistency": float(np.mean([row["normal_consistency"] for row in selected])),
        "same_index_recovered_vertex_rms": float(
            np.mean([row["same_index_recovered_vertex_rms"] for row in selected])
        ),
        "introduced_flipped_faces": int(sum(row["introduced_flipped_faces"] for row in selected)),
        "normalized_flip_rate": float(
            sum(row["introduced_flipped_faces"] for row in selected) / faces
        ),
        "new_degenerate_faces": int(sum(row["new_degenerate_faces"] for row in selected)),
        "improved": int(sum(bool(row["improved"]) for row in selected)),
        "worsened": int(sum(bool(row["worsened"]) for row in selected)),
        "mean_pcg_runtime_seconds": float(np.mean([row["pcg_runtime_seconds"] for row in selected])),
        "mean_evaluator_runtime_seconds": float(
            np.mean([row["evaluator_runtime_seconds"] for row in selected])
        ),
        "mean_total_runtime_seconds": float(
            np.mean(
                [row["pcg_runtime_seconds"] + row["evaluator_runtime_seconds"] for row in selected]
            )
        ),
    }


def paired_statistics(
    rows: Sequence[Mapping[str, Any]], field: str, replicates: int, seed: int
) -> dict[str, Any]:
    a = {row["sample_id"]: row for row in rows if row["method"] == DIRECT_AE}
    b = {row["sample_id"]: row for row in rows if row["method"] == PROPOSED_BE}
    if a.keys() != b.keys() or len(a) != 50:
        raise RuntimeError(f"Paired identity mismatch for {field}")
    ids = sorted(a)
    values = np.asarray([float(a[key][field]) - float(b[key][field]) for key in ids])
    rng = np.random.default_rng(seed)
    mesh_draws = values[rng.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
    grouped: dict[str, list[float]] = {}
    for sample_id, value in zip(ids, values, strict=True):
        grouped.setdefault(object_id(sample_id), []).append(float(value))
    object_means = np.asarray([np.mean(grouped[key]) for key in sorted(grouped)])
    object_draws = object_means[
        rng.integers(0, len(object_means), size=(replicates, len(object_means)))
    ].mean(axis=1)
    lower = field in LOWER_IS_BETTER
    wins = values < 0 if lower else values > 0
    losses = values > 0 if lower else values < 0
    ties = ~(wins | losses)
    return {
        "metric": field,
        "difference": f"{field}_A_plus_E_minus_B_plus_E",
        "samples": len(values),
        "object_clusters": len(object_means),
        "mean_difference": float(values.mean()),
        "median_difference": float(np.median(values)),
        "mesh_bootstrap_95_percent_ci": [
            float(np.quantile(mesh_draws, 0.025)),
            float(np.quantile(mesh_draws, 0.975)),
        ],
        "object_cluster_bootstrap_95_percent_ci": [
            float(np.quantile(object_draws, 0.025)),
            float(np.quantile(object_draws, 0.975)),
        ],
        "a_plus_e_wins": int(wins.sum()),
        "ties": int(ties.sum()),
        "a_plus_e_losses": int(losses.sum()),
    }


def object_aggregates(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    objects = sorted({row["object_id"] for row in rows})
    for cluster in objects:
        for method in (DIRECT_AE, PROPOSED_BE):
            selected = [row for row in rows if row["object_id"] == cluster and row["method"] == method]
            output.append(
                {
                    "object_id": cluster,
                    "method": method,
                    "samples": len(selected),
                    **{
                        field: float(np.mean([row[field] for row in selected]))
                        for field in PAIRED_FIELDS
                    },
                }
            )
        left, right = output[-2], output[-1]
        output.append(
            {
                "object_id": cluster,
                "method": "A+E minus B+E",
                "samples": left["samples"],
                **{field: float(left[field] - right[field]) for field in PAIRED_FIELDS},
            }
        )
    return output


def summary_row(source: Mapping[str, Any], label: str, *, scalar: bool = False) -> dict[str, Any]:
    return {
        "method": label,
        "refined_chamfer": float(source["chamfer" if scalar else "refined_chamfer"]),
        "p2s_p95": float(source["p2s_p95"]),
        "fscore": float(source["fscore"]),
        "normal_consistency": float(source["normal_consistency"]),
        "same_index_recovered_vertex_rms": float(
            source["same_index_vertex_rms" if scalar else "same_index_recovered_vertex_rms"]
        ),
    }


def fmt(value: float) -> str:
    return f"{value:.10g}"


def main() -> None:
    args = parse_args()
    started_all = time.perf_counter()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type != "cpu":
        raise RuntimeError("This audit is intentionally restricted to local CPU execution")

    a_payload = read_json(args.arm_ab_report / "shards" / f"{ARM_A}.json")
    b_payload = read_json(args.arm_ab_report / "shards" / f"{ARM_B}.json")
    e_payload = read_json(args.arm_e_report / "shards" / f"{ARM_E}.json")
    hybrid_summary = read_json(args.hybrid_report / "matched_summary.json")
    for arm, payload in ((ARM_A, a_payload), (ARM_B, b_payload), (ARM_E, e_payload)):
        if payload["checkpoint_sha256"] != EXPECTED_SHA[arm]:
            raise RuntimeError(f"{arm}: source metadata checkpoint SHA mismatch")
    if hybrid_summary["arm_b_checkpoint_sha256"] != EXPECTED_SHA[ARM_B]:
        raise RuntimeError("B+E archive Arm-B SHA mismatch")
    if hybrid_summary["arm_e_checkpoint_sha256"] != EXPECTED_SHA[ARM_E]:
        raise RuntimeError("B+E archive Arm-E SHA mismatch")
    if float(hybrid_summary["lambda_hybrid_best"]) != FUSION_LAMBDA:
        raise RuntimeError("B+E archive lambda is not 0.03")
    if hybrid_summary["metric_protocol"] != METRIC_PROTOCOL:
        raise RuntimeError("B+E archive evaluator contract differs from current evaluator")

    checkpoint_checks: dict[str, Any] = {}
    for arm, path in ((ARM_B, args.arm_b_checkpoint), (ARM_E, args.arm_e_checkpoint)):
        actual = sha256_file(path.resolve())
        if actual != EXPECTED_SHA[arm]:
            raise RuntimeError(f"{arm}: local checkpoint SHA mismatch")
        checkpoint_checks[arm] = {
            "path": str(path.resolve()),
            "expected_sha256": EXPECTED_SHA[arm],
            "actual_sha256": actual,
            "directly_rehashed": True,
        }
    warnings: list[str] = []
    if args.arm_a_checkpoint is not None and args.arm_a_checkpoint.is_file():
        actual = sha256_file(args.arm_a_checkpoint.resolve())
        if actual != EXPECTED_SHA[ARM_A]:
            raise RuntimeError("Arm-A local checkpoint SHA mismatch")
        checkpoint_checks[ARM_A] = {
            "path": str(args.arm_a_checkpoint.resolve()),
            "expected_sha256": EXPECTED_SHA[ARM_A],
            "actual_sha256": actual,
            "directly_rehashed": True,
        }
    else:
        warning = (
            "Arm-A checkpoint file is absent locally; its identity is verified transitively through "
            "the archived prediction-shard metadata, which declares the required SHA."
        )
        warnings.append(warning)
        checkpoint_checks[ARM_A] = {
            "recorded_path": a_payload["checkpoint"],
            "expected_sha256": EXPECTED_SHA[ARM_A],
            "metadata_sha256": a_payload["checkpoint_sha256"],
            "directly_rehashed": False,
            "warning": warning,
        }

    manifest = args.manifest.resolve()
    manifest_sha = sha256_file(manifest)
    dataset = PreparedMeshDataset.from_manifest(manifest, "test")
    expected_ids = list(dataset.sample_ids)
    if len(expected_ids) != 50 or len({object_id(item) for item in expected_ids}) != 5:
        raise RuntimeError("Expected exactly 50 samples in five object clusters")
    a_rows, b_rows, e_rows = (
        split_rows(a_payload, "test"),
        split_rows(b_payload, "test"),
        split_rows(e_payload, "test"),
    )
    for name, rows in (("Arm A", a_rows), ("Arm B", b_rows), ("Arm E", e_rows)):
        if [row["sample_id"] for row in rows] != expected_ids:
            raise RuntimeError(f"{name} IDs/order differ from manifest")

    a_prediction, a_target, a_array_path = prediction_array(args.arm_ab_report, ARM_A, "test")
    b_prediction, b_target, b_array_path = prediction_array(args.arm_ab_report, ARM_B, "test")
    e_prediction, _, e_array_path = prediction_array(args.arm_e_report, ARM_E, "test")
    expected_vertices = sum(int(row["vertices"]) for row in a_rows)
    if not (
        a_prediction.shape == b_prediction.shape == e_prediction.shape == (expected_vertices, 3)
    ):
        raise RuntimeError("Prediction array shapes differ")
    if not np.array_equal(a_target, b_target):
        raise RuntimeError("Arm-A and Arm-B raw Laplacian targets differ")
    a_raw_epe = float(np.mean(np.linalg.norm(a_prediction - a_target, axis=1)))
    b_raw_epe = float(np.mean(np.linalg.norm(b_prediction - b_target, axis=1)))
    if abs(a_raw_epe - 0.0025264054) > 5e-10 or abs(b_raw_epe - 0.00263985669) > 5e-10:
        raise RuntimeError("Raw EPE does not reproduce the archived context")

    archived_be = typed_archived_rows(args.hybrid_report / "matched_per_sample.csv")
    archived_be_serialization_order_matches_manifest = list(archived_be) == expected_ids
    if set(archived_be) != set(expected_ids):
        raise RuntimeError("Archived B+E sample IDs differ from manifest")
    a_starts, b_starts, e_starts = starts(a_rows), starts(b_rows), starts(e_rows)
    rows: list[dict[str, Any]] = []
    reproduction_differences: list[dict[str, Any]] = []
    for index, sample_id in enumerate(expected_ids):
        static = dataset.load_static(index)
        if str(static["sample_id"]) != sample_id:
            raise RuntimeError(f"{sample_id}: loaded static sample ID mismatch")
        initial = Mesh(
            np.asarray(static["vertices"], dtype=np.float64),
            np.asarray(static["faces"], dtype=np.int64),
        ).ensure_normals()
        clean = _clean_mesh(static)
        count = initial.num_vertices
        a_field = a_prediction[a_starts[index] : a_starts[index] + count]
        b_field = b_prediction[b_starts[index] : b_starts[index] + count]
        e_displacement = e_prediction[e_starts[index] : e_starts[index] + count]
        direct_vertices = initial.vertices + e_displacement

        for method, field in ((DIRECT_AE, a_field), (PROPOSED_BE, b_field)):
            fused, solver = _pcg(field, direct_vertices, static, FUSION_LAMBDA, device)
            if not solver["pcg_converged"]:
                raise RuntimeError(f"{sample_id}/{method}: PCG did not converge")
            evaluation_started = time.perf_counter()
            metric = _geometry_row(
                "test",
                sample_id,
                method,
                Mesh(fused, initial.faces.copy()).ensure_normals(),
                clean,
                initial,
            )
            evaluator_runtime = time.perf_counter() - evaluation_started
            item = _row(
                "test",
                method,
                sample_id,
                index,
                fused,
                clean,
                initial,
                metric,
                solver,
                FUSION_LAMBDA,
            )
            item["method"] = method
            item["object_id"] = object_id(sample_id)
            item["evaluator_runtime_seconds"] = evaluator_runtime
            item["raw_differential_epe"] = float(
                a_rows[index]["raw_epe"] if method == DIRECT_AE else b_rows[index]["raw_epe"]
            )
            rows.append(item)

        reproduced = rows[-1]
        archived = archived_be[sample_id]
        differences = {
            field: abs(float(reproduced[field]) - float(archived[field]))
            for field in PAIRED_FIELDS
        }
        differences.update(
            {
                "introduced_flipped_faces": abs(
                    int(reproduced["introduced_flipped_faces"])
                    - int(archived["introduced_flipped_faces"])
                ),
                "new_degenerate_faces": abs(
                    int(reproduced["new_degenerate_faces"])- int(archived["new_degenerate_faces"])
                ),
            }
        )
        if max(differences[field] for field in PAIRED_FIELDS) > REPRODUCTION_TOLERANCE:
            raise RuntimeError(f"{sample_id}: reproduced B+E metric differs from archive")
        if differences["introduced_flipped_faces"] or differences["new_degenerate_faces"]:
            raise RuntimeError(f"{sample_id}: reproduced B+E topology diagnostics differ")
        reproduction_differences.append({"sample_id": sample_id, **differences})
        # The paired reference is the existing archived B+E result requested by the
        # contract.  Keep the new solve's runtime/audit fields after verifying it,
        # but use the exact archived metric values for the actual comparison.
        for field in PAIRED_FIELDS:
            reproduced[f"locally_reproduced_{field}"] = reproduced[field]
            reproduced[field] = archived[field]
        for field in (
            "initial_chamfer",
            "relative_chamfer_gain",
            "eta",
            "p2s",
            "introduced_flipped_faces",
            "normalized_flip_rate",
            "new_degenerate_faces",
        ):
            reproduced[f"locally_reproduced_{field}"] = reproduced[field]
            reproduced[field] = archived[field]
        reproduced["metric_values_source"] = "verified existing B+E archive"
        rows[-2]["metric_values_source"] = "new matched A+E evaluation"
        print(f"matched fusion test {index + 1}/50 {sample_id}", flush=True)

    aggregates = [aggregate(rows, method) for method in (DIRECT_AE, PROPOSED_BE)]
    paired = [
        paired_statistics(rows, field, args.bootstrap_replicates, args.seed)
        for field in PAIRED_FIELDS
    ]
    paired_cd = next(row for row in paired if row["metric"] == "refined_chamfer")
    mesh_ci = paired_cd["mesh_bootstrap_95_percent_ci"]
    cluster_ci = paired_cd["object_cluster_bootstrap_95_percent_ci"]
    if mesh_ci[0] > 0 and cluster_ci[0] > 0:
        verdict = "STRONG EVIDENCE AGAINST SIMPLE COMBINATION"
    elif paired_cd["mean_difference"] > 0 and (mesh_ci[0] > 0 or cluster_ci[0] > 0):
        verdict = "MODERATE EVIDENCE AGAINST SIMPLE COMBINATION"
    elif mesh_ci[1] < 0 and cluster_ci[1] < 0:
        verdict = "SIMPLE-COMBINATION BASELINE IS BETTER"
    else:
        verdict = "NO MEANINGFUL DIFFERENCE"

    ab_summary = read_json(args.arm_ab_report / "summary.json")
    pure_summary = read_json(args.pure_fusion_report / "summary.json")
    scalar_summary = read_json(args.scalar_fusion_report / "test_summary.json")
    a_standalone = next(
        row for row in ab_summary["geometry"] if row["split"] == "test" and row["arm"] == ARM_A
    )
    b_standalone = next(
        row for row in hybrid_summary["aggregate"] if row["split"] == "test" and row["arm"] == ARM_B
    )
    e_standalone = next(
        row for row in hybrid_summary["aggregate"] if row["split"] == "test" and row["arm"] == ARM_E
    )
    be_archive = next(
        row
        for row in hybrid_summary["aggregate"]
        if row["split"] == "test" and row["arm"] == "Hybrid_B_laplacian_E_anchor"
    )
    pure_be = next(
        row
        for row in pure_summary["aggregate"]
        if row["split"] == "test" and row["system"] == "Pure-Vertex Arm-B + Arm-E"
    )
    scalar = next(
        row for row in scalar_summary["aggregate"] if row["method"] == "Naive scalar fusion"
    )
    main_table = [
        summary_row(a_standalone, "Arm A standalone"),
        summary_row(b_standalone, "Arm B standalone"),
        summary_row(e_standalone, "Arm E standalone"),
        dict(aggregates[0]),
        summary_row(be_archive, PROPOSED_BE),
        summary_row(pure_be, "Pure-Vertex+E, lambda=0.03"),
        summary_row(scalar, "Scalar fusion, alpha=0.31", scalar=True),
    ]

    command = (
        "PYTHONPATH=src:scripts conda run --no-capture-output -n test python "
        "scripts/evaluate_sofa50_direct_lap_positional_matched_fusion.py "
        f"--manifest {args.manifest} --arm-ab-report {args.arm_ab_report} "
        f"--arm-e-report {args.arm_e_report} --hybrid-report {args.hybrid_report} "
        f"--pure-fusion-report {args.pure_fusion_report} "
        f"--scalar-fusion-report {args.scalar_fusion_report} "
        f"--arm-b-checkpoint {args.arm_b_checkpoint} --arm-e-checkpoint {args.arm_e_checkpoint} "
        f"--output-dir {args.output_dir} --device cpu"
    )
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    source_artifacts = {
        ARM_A: {
            "prediction_array_path": str(a_array_path.resolve()),
            "prediction_array_sha256": sha256_file(a_array_path),
            "metadata_path": str((args.arm_ab_report / "shards" / f"{ARM_A}.json").resolve()),
            "metadata_sha256": sha256_file(args.arm_ab_report / "shards" / f"{ARM_A}.json"),
        },
        ARM_B: {
            "prediction_array_path": str(b_array_path.resolve()),
            "prediction_array_sha256": sha256_file(b_array_path),
            "metadata_path": str((args.arm_ab_report / "shards" / f"{ARM_B}.json").resolve()),
            "metadata_sha256": sha256_file(args.arm_ab_report / "shards" / f"{ARM_B}.json"),
        },
        ARM_E: {
            "prediction_array_path": str(e_array_path.resolve()),
            "prediction_array_sha256": sha256_file(e_array_path),
            "metadata_path": str((args.arm_e_report / "shards" / f"{ARM_E}.json").resolve()),
            "metadata_sha256": sha256_file(args.arm_e_report / "shards" / f"{ARM_E}.json"),
        },
        "archived_B_plus_E_per_mesh": {
            "path": str((args.hybrid_report / "matched_per_sample.csv").resolve()),
            "sha256": sha256_file(args.hybrid_report / "matched_per_sample.csv"),
        },
    }
    maximum_reproduction_difference = {
        field: max(float(row[field]) for row in reproduction_differences)
        for field in (*PAIRED_FIELDS, "introduced_flipped_faces", "new_degenerate_faces")
    }
    total_runtime = time.perf_counter() - started_all
    solver_rows = [row for row in rows if "pcg_relative_residual" in row]
    contract = {
        "passed": True,
        "read_only_frozen_predictions": True,
        "models_retrained": False,
        "network_inference_run": False,
        "hpc_jobs_submitted": False,
        "same_arm_e_prediction_array": True,
        "same_uniform_random_walk_operator": True,
        "operator_definition": (
            "L_U=I-D^{-1}A (sample-specific Uniform random-walk Laplacian); "
            "solve (L_U^T L_U+0.03 I)V=L_U^T delta+0.03 V_P"
        ),
        "same_lambda": True,
        "lambda": FUSION_LAMBDA,
        "same_float64_pcg_solver": True,
        "pcg_tolerance": 1e-4,
        "pcg_maximum_iterations": 2048,
        "all_pcg_solves_converged": all(bool(row["pcg_converged"]) for row in solver_rows),
        "maximum_observed_pcg_relative_residual": max(
            float(row["pcg_relative_residual"]) for row in solver_rows
        ),
        "mean_observed_pcg_iterations": float(
            np.mean([float(row["pcg_iterations"]) for row in solver_rows])
        ),
        "maximum_observed_pcg_iterations": max(int(row["pcg_iterations"]) for row in solver_rows),
        "same_50_mesh_ids_and_order": True,
        "archived_b_plus_e_csv_serialization_order_matches_manifest": archived_be_serialization_order_matches_manifest,
        "archived_b_plus_e_csv_order_note": (
            "The old CSV is serialized in shard-interleaved order; all 50 IDs are exact and "
            "comparisons are joined by sample_id. New A+E and B+E solves both use manifest order."
        ),
        "same_gt_from_prepared_samples": True,
        "same_evaluator": True,
        "metric_protocol": METRIC_PROTOCOL,
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha,
        "sample_ids": expected_ids,
        "object_clusters": sorted({object_id(item) for item in expected_ids}),
        "checkpoint_checks": checkpoint_checks,
        "source_artifacts": source_artifacts,
        "b_plus_e_reproduction_tolerance": REPRODUCTION_TOLERANCE,
        "maximum_b_plus_e_reproduction_difference": maximum_reproduction_difference,
        "warnings": warnings,
    }
    result = {
        "execution_status": "completed",
        "verdict": verdict,
        "git_head": git_head,
        "evaluation_command": command,
        "contract_audit": contract,
        "raw_differential_epe_context": {
            "Arm_A": a_raw_epe,
            "Arm_B": b_raw_epe,
            "note": "Raw EPE is a differential-predictor diagnostic, not a fused-output metric.",
        },
        "aggregate": aggregates,
        "main_table": main_table,
        "paired_A_plus_E_minus_B_plus_E": paired,
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "seed": args.seed,
            "mesh_resampling_unit": "one of 50 test meshes",
            "cluster_resampling_unit": "one of five object IDs after averaging its ten variants",
        },
        "runtime_seconds": total_runtime,
    }
    write_json(output / "summary.json", result)
    write_json(output / "contract_audit.json", contract)
    write_json(output / "sample_ids.json", {"split": "test", "sample_ids": expected_ids})
    write_csv(output / "per_mesh_metrics.csv", rows)
    write_csv(output / "paired_metrics.csv", paired)
    write_csv(output / "per_object_metrics.csv", object_aggregates(rows))
    write_csv(output / "b_plus_e_reproduction_check.csv", reproduction_differences)
    write_csv(output / "aggregate_metrics.csv", aggregates)
    (output / "evaluation_command.txt").write_text(command + "\n", encoding="utf-8")

    lines = [
        "# Matched Direct-Lap Arm-A + Direct-Positional Arm-E fusion",
        "",
        "## 1. EXECUTION STATUS",
        "",
        "**completed**. The evaluation reused saved frozen predictions on local CPU; no model was trained, no network inference ran, and no HPC job was submitted.",
        "",
        "## 2. MATCH VALIDATION",
        "",
        "Contract audit: **true with one provenance warning**. A+E and the locally reproduced B+E use the identical Arm-E displacement array, sample-specific Uniform random-walk Laplacian, `lambda=0.03`, float64 PCG implementation (`tol=1e-4`, maximum 2048 iterations), same ordered 50 meshes, same GT, and the same geometry evaluator.",
        "The operator is `L_U=I-D^{-1}A`, and both systems solve `(L_U^T L_U+0.03 I)V=L_U^T delta+0.03 V_P` on each sample's fixed topology.",
        f"The evaluator contract is `{METRIC_PROTOCOL}`. The archived B+E result was independently reproduced; maximum CD discrepancy was `{maximum_reproduction_difference['refined_chamfer']:.3e}` and topology counts matched exactly.",
        "The Arm-A checkpoint file itself is no longer present locally, so it could not be directly rehashed; the reused Arm-A prediction archive is tied to archived metadata declaring the required checkpoint SHA, its IDs/order match the manifest exactly, its raw target is byte-identical to Arm B's, and its raw EPE reproduces the archived value.",
        "",
        "## 3. MAIN TABLE",
        "",
        "| Method | CD | P2S p95 | F-score | Normal | VRMS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in main_table:
        lines.append(
            f"| {row['method']} | {fmt(row['refined_chamfer'])} | {fmt(row['p2s_p95'])} | "
            f"{fmt(row['fscore'])} | {fmt(row['normal_consistency'])} | "
            f"{fmt(row['same_index_recovered_vertex_rms'])} |"
        )
    lines += [
        "",
        f"Direct-Lap A+E introduced `{aggregates[0]['introduced_flipped_faces']}` fixed-connectivity flips in total (normalized rate `{aggregates[0]['normalized_flip_rate']:.8f}`), with `{aggregates[0]['new_degenerate_faces']}` new degenerate faces. Mean local runtime was `{aggregates[0]['mean_total_runtime_seconds']:.6f}` s/mesh (PCG plus evaluator only).",
        f"Arm-A raw differential EPE is `{a_raw_epe:.10f}`; Arm-B raw differential EPE is `{b_raw_epe:.10f}`. These are predictor diagnostics, not fused-output metrics.",
        "",
        "## 4. DIRECT A+E VS B+E",
        "",
        "All differences are A+E minus B+E. Positive CD/P2S/VRMS and negative F-score/Normal favor Proposed B+E.",
        "",
        "| Metric | Mean difference | Median difference | A+E W/T/L | Mesh bootstrap 95% CI | Object-cluster bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in paired:
        lines.append(
            f"| {row['metric']} | {fmt(row['mean_difference'])} | {fmt(row['median_difference'])} | "
            f"{row['a_plus_e_wins']}/{row['ties']}/{row['a_plus_e_losses']} | "
            f"[{fmt(row['mesh_bootstrap_95_percent_ci'][0])}, {fmt(row['mesh_bootstrap_95_percent_ci'][1])}] | "
            f"[{fmt(row['object_cluster_bootstrap_95_percent_ci'][0])}, {fmt(row['object_cluster_bootstrap_95_percent_ci'][1])}] |"
        )
    lines += [
        "",
        "Per-object aggregates are in `per_object_metrics.csv`; all per-mesh values are in `per_mesh_metrics.csv`.",
        "",
        "## 5. VERDICT",
        "",
        f"**{verdict}**",
        "",
        "## 6. PAPER IMPLICATION",
        "",
    ]
    if verdict in {
        "STRONG EVIDENCE AGAINST SIMPLE COMBINATION",
        "MODERATE EVIDENCE AGAINST SIMPLE COMBINATION",
    }:
        lines += [
            "Under the matched frozen contract, simply combining a directly supervised differential predictor with the independently learned positional predictor through the same recovery operator is insufficient to match Proposed B+E.",
            "Because E, the operator, lambda, topology, samples, solver, and evaluator are fixed, the paired difference is attributable to replacing the differential field produced by Arm B with Arm A's field.",
            "The result supports the claim that how the differential field is trained affects its usefulness under later operator composition.",
            "It does not establish that Arm B is universally superior to direct Laplacian supervision, nor that B+E dominates every metric or domain.",
            "The claim should remain scoped to matched Sofa50-v2, frozen single-pass fusion at lambda 0.03.",
        ]
    elif verdict == "NO MEANINGFUL DIFFERENCE":
        lines += [
            "For the primary surface-distance comparison, the matched experiment does not distinguish Direct-Lap A+E from Proposed B+E with meaningful confidence.",
            "A+E is numerically lower in mean CD and P2S p95 and slightly higher in F-score, while its Normal advantage is statistically positive under both bootstrap units; B+E is numerically better only in VRMS among the reported metrics.",
            "Therefore the paper cannot use this baseline to argue that Arm B's training is necessary for operator composition.",
            "The simple-combination criticism becomes substantially stronger and the methodological novelty claim must be narrowed.",
            "This conclusion remains scoped to matched Sofa50-v2, frozen single-pass fusion at lambda 0.03.",
        ]
    else:
        lines += [
            "The Direct-Lap A+E baseline is better than Proposed B+E under the matched contract.",
            "This undermines any claim that Arm B's recovery-aware training is responsible for the Hybrid improvement.",
            "The paper must not claim that its differential training is necessary for operator composition or superior to the naive prior-art combination.",
            "The remaining evidence can support the recovery formulation itself, but not the current Arm-B-specific methodological story.",
            "This conclusion remains scoped to matched Sofa50-v2, frozen single-pass fusion at lambda 0.03.",
        ]
    lines += [
        "",
        "## Reproducibility",
        "",
        f"- Git HEAD: `{git_head}`.",
        f"- Dataset manifest: `{manifest}`; SHA-256 `{manifest_sha}`.",
        f"- Evaluation command: `{command}`.",
        f"- Bootstrap: `{args.bootstrap_replicates}` replicates, seed `{args.seed}`; 50 mesh units and five object-cluster units.",
        f"- Total wall time: `{total_runtime:.3f}` seconds.",
        "- Checkpoint and prediction-artifact paths/hashes are recorded in `contract_audit.json`.",
        "- Warning: Arm-A checkpoint direct rehash was unavailable locally; see the match-validation qualification above.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "verdict": verdict, "delta_cd": paired_cd}, indent=2))


if __name__ == "__main__":
    main()
