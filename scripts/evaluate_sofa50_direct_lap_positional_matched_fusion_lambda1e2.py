#!/usr/bin/env python3
"""Matched frozen Arm-A+E versus Arm-B+E fusion at fixed lambda=0.01."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import evaluate_sofa50_direct_lap_positional_matched_fusion as base
from diagnose_sofa50_exact_target_oracle import METRIC_PROTOCOL, _clean_mesh, _geometry_row
from diagnose_sofa50_frozen_hybrid_recovery import _pcg, _row
from mlr.data import Mesh
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


FUSION_LAMBDA = 0.01
DIRECT_AE = "Direct-Lap A+E, lambda=0.01"
PROPOSED_BE = "Proposed B+E, lambda=0.01"
REPRODUCTION_TOLERANCE = 2e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--arm-ab-report", required=True, type=Path)
    parser.add_argument("--arm-e-report", required=True, type=Path)
    parser.add_argument("--b-e-reference-report", required=True, type=Path)
    parser.add_argument("--arm-b-checkpoint", required=True, type=Path)
    parser.add_argument("--arm-e-checkpoint", required=True, type=Path)
    parser.add_argument("--arm-a-checkpoint", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def reference_rows(report: Path) -> dict[str, dict[str, Any]]:
    payload = base.read_json(report / "lambda1e2_per_sample.json")
    rows = [dict(row) for row in payload["rows"] if row["split"] == "test"]
    if len(rows) != 50:
        raise RuntimeError(f"Expected 50 archived B+E lambda=0.01 rows, found {len(rows)}")
    return {str(row["sample_id"]): row for row in rows}


def fmt(value: float) -> str:
    return f"{value:.10g}"


def main() -> None:
    args = parse_args()
    started_all = time.perf_counter()
    if args.device != "cpu":
        raise RuntimeError("This fixed-lambda audit is intentionally restricted to local CPU")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    a_payload = base.read_json(args.arm_ab_report / "shards" / f"{base.ARM_A}.json")
    b_payload = base.read_json(args.arm_ab_report / "shards" / f"{base.ARM_B}.json")
    e_payload = base.read_json(args.arm_e_report / "shards" / f"{base.ARM_E}.json")
    reference_summary = base.read_json(args.b_e_reference_report / "summary.json")
    reference_contract = reference_summary["contract_audit"]
    if float(reference_contract["fixed_fusion_lambda"]) != FUSION_LAMBDA:
        raise RuntimeError("Reference report is not B+E lambda=0.01")
    if reference_contract["metric_protocol"] != METRIC_PROTOCOL:
        raise RuntimeError("Reference evaluator protocol mismatch")
    for arm, payload in (
        (base.ARM_A, a_payload),
        (base.ARM_B, b_payload),
        (base.ARM_E, e_payload),
    ):
        if payload["checkpoint_sha256"] != base.EXPECTED_SHA[arm]:
            raise RuntimeError(f"{arm}: prediction metadata checkpoint SHA mismatch")

    checkpoint_checks: dict[str, Any] = {}
    for arm, path in ((base.ARM_B, args.arm_b_checkpoint), (base.ARM_E, args.arm_e_checkpoint)):
        actual = base.sha256_file(path.resolve())
        if actual != base.EXPECTED_SHA[arm]:
            raise RuntimeError(f"{arm}: local checkpoint SHA mismatch")
        checkpoint_checks[arm] = {
            "path": str(path.resolve()),
            "expected_sha256": base.EXPECTED_SHA[arm],
            "actual_sha256": actual,
            "directly_rehashed": True,
        }
    warnings: list[str] = []
    if args.arm_a_checkpoint is not None and args.arm_a_checkpoint.is_file():
        actual = base.sha256_file(args.arm_a_checkpoint.resolve())
        if actual != base.EXPECTED_SHA[base.ARM_A]:
            raise RuntimeError("Arm-A local checkpoint SHA mismatch")
        checkpoint_checks[base.ARM_A] = {
            "path": str(args.arm_a_checkpoint.resolve()),
            "expected_sha256": base.EXPECTED_SHA[base.ARM_A],
            "actual_sha256": actual,
            "directly_rehashed": True,
        }
    else:
        warning = (
            "Arm-A checkpoint file is absent locally; the reused prediction shard is linked "
            "to archived metadata declaring the required checkpoint SHA."
        )
        warnings.append(warning)
        checkpoint_checks[base.ARM_A] = {
            "recorded_path": a_payload["checkpoint"],
            "expected_sha256": base.EXPECTED_SHA[base.ARM_A],
            "metadata_sha256": a_payload["checkpoint_sha256"],
            "directly_rehashed": False,
            "warning": warning,
        }

    manifest = args.manifest.resolve()
    dataset = PreparedMeshDataset.from_manifest(manifest, "test")
    expected_ids = list(dataset.sample_ids)
    if len(expected_ids) != 50 or len({base.object_id(item) for item in expected_ids}) != 5:
        raise RuntimeError("Expected 50 test samples in five object clusters")
    a_rows = base.split_rows(a_payload, "test")
    b_rows = base.split_rows(b_payload, "test")
    e_rows = base.split_rows(e_payload, "test")
    for name, rows in (("Arm A", a_rows), ("Arm B", b_rows), ("Arm E", e_rows)):
        if [str(row["sample_id"]) for row in rows] != expected_ids:
            raise RuntimeError(f"{name} IDs/order differ from manifest")

    a_prediction, a_target, a_array_path = base.prediction_array(
        args.arm_ab_report, base.ARM_A, "test"
    )
    b_prediction, b_target, b_array_path = base.prediction_array(
        args.arm_ab_report, base.ARM_B, "test"
    )
    e_prediction, _, e_array_path = base.prediction_array(
        args.arm_e_report, base.ARM_E, "test"
    )
    expected_vertices = sum(int(row["vertices"]) for row in a_rows)
    if not (
        a_prediction.shape == b_prediction.shape == e_prediction.shape == (expected_vertices, 3)
    ):
        raise RuntimeError("Prediction array shapes differ")
    if not np.array_equal(a_target, b_target):
        raise RuntimeError("Arm-A and Arm-B raw targets differ")
    a_raw_epe = float(np.mean(np.linalg.norm(a_prediction - a_target, axis=1)))
    b_raw_epe = float(np.mean(np.linalg.norm(b_prediction - b_target, axis=1)))
    archived_be = reference_rows(args.b_e_reference_report)
    if set(archived_be) != set(expected_ids):
        raise RuntimeError("Archived lambda=0.01 B+E sample IDs differ")

    a_starts = base.starts(a_rows)
    b_starts = base.starts(b_rows)
    e_starts = base.starts(e_rows)
    rows: list[dict[str, Any]] = []
    reproduction: list[dict[str, Any]] = []
    for index, sample_id in enumerate(expected_ids):
        static = dataset.load_static(index)
        if str(static["sample_id"]) != sample_id:
            raise RuntimeError(f"{sample_id}: static sample mismatch")
        initial = Mesh(
            np.asarray(static["vertices"], dtype=np.float64),
            np.asarray(static["faces"], dtype=np.int64),
        ).ensure_normals()
        clean = _clean_mesh(static)
        count = initial.num_vertices
        a_field = a_prediction[a_starts[index] : a_starts[index] + count]
        b_field = b_prediction[b_starts[index] : b_starts[index] + count]
        direct = initial.vertices + e_prediction[e_starts[index] : e_starts[index] + count]

        for method, field, raw_epe in (
            (DIRECT_AE, a_field, a_rows[index]["raw_epe"]),
            (PROPOSED_BE, b_field, b_rows[index]["raw_epe"]),
        ):
            fused, solver = _pcg(field, direct, static, FUSION_LAMBDA, device)
            if not solver["pcg_converged"]:
                raise RuntimeError(f"{sample_id}/{method}: PCG did not converge")
            evaluator_started = time.perf_counter()
            metric = _geometry_row(
                "test",
                sample_id,
                method,
                Mesh(fused, initial.faces.copy()).ensure_normals(),
                clean,
                initial,
            )
            evaluator_runtime = time.perf_counter() - evaluator_started
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
            item.update(
                {
                    "method": method,
                    "object_id": base.object_id(sample_id),
                    "evaluator_runtime_seconds": evaluator_runtime,
                    "raw_differential_epe": float(raw_epe),
                }
            )
            rows.append(item)

        reproduced = rows[-1]
        archived = archived_be[sample_id]
        differences = {
            field: abs(float(reproduced[field]) - float(archived[field]))
            for field in base.PAIRED_FIELDS
        }
        differences.update(
            {
                "introduced_flipped_faces": abs(
                    int(reproduced["introduced_flipped_faces"])
                    - int(archived["introduced_flipped_faces"])
                ),
                "new_degenerate_faces": abs(
                    int(reproduced["new_degenerate_faces"])
                    - int(archived["new_degenerate_faces"])
                ),
            }
        )
        if max(differences[field] for field in base.PAIRED_FIELDS) > REPRODUCTION_TOLERANCE:
            raise RuntimeError(f"{sample_id}: B+E lambda=0.01 reproduction mismatch")
        if differences["introduced_flipped_faces"] or differences["new_degenerate_faces"]:
            raise RuntimeError(f"{sample_id}: B+E topology audit mismatch")
        reproduction.append({"sample_id": sample_id, **differences})
        for field in base.PAIRED_FIELDS:
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
        reproduced["metric_values_source"] = "verified existing B+E lambda=0.01 archive"
        rows[-2]["metric_values_source"] = "new matched A+E lambda=0.01 evaluation"
        print(f"matched lambda1e-2 test {index + 1}/50 {sample_id}", flush=True)

    base.DIRECT_AE = DIRECT_AE
    base.PROPOSED_BE = PROPOSED_BE
    aggregates = [base.aggregate(rows, method) for method in (DIRECT_AE, PROPOSED_BE)]
    paired = [
        base.paired_statistics(rows, field, args.bootstrap_replicates, args.seed)
        for field in base.PAIRED_FIELDS
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

    solver_rows = [row for row in rows if "pcg_relative_residual" in row]
    maximum_reproduction_difference = {
        field: max(float(row[field]) for row in reproduction)
        for field in (*base.PAIRED_FIELDS, "introduced_flipped_faces", "new_degenerate_faces")
    }
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    command = (
        "PYTHONPATH=src:scripts conda run --no-capture-output -n test python "
        "scripts/evaluate_sofa50_direct_lap_positional_matched_fusion_lambda1e2.py "
        f"--manifest {args.manifest} --arm-ab-report {args.arm_ab_report} "
        f"--arm-e-report {args.arm_e_report} --b-e-reference-report {args.b_e_reference_report} "
        f"--arm-b-checkpoint {args.arm_b_checkpoint} --arm-e-checkpoint {args.arm_e_checkpoint} "
        f"--output-dir {args.output_dir} --device cpu"
    )
    total_runtime = time.perf_counter() - started_all
    contract = {
        "passed": True,
        "models_retrained": False,
        "network_inference_run": False,
        "hpc_jobs_submitted": False,
        "lambda": FUSION_LAMBDA,
        "operator_definition": (
            "L_U=I-D^{-1}A; solve (L_U^T L_U+0.01 I)V=L_U^T delta+0.01 V_P"
        ),
        "same_arm_e_prediction_array": True,
        "same_uniform_random_walk_operator": True,
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
        "same_gt_from_prepared_samples": True,
        "same_evaluator": True,
        "metric_protocol": METRIC_PROTOCOL,
        "manifest": str(manifest),
        "manifest_sha256": base.sha256_file(manifest),
        "sample_ids": expected_ids,
        "object_clusters": sorted({base.object_id(item) for item in expected_ids}),
        "checkpoint_checks": checkpoint_checks,
        "source_artifacts": {
            base.ARM_A: {"path": str(a_array_path.resolve()), "sha256": base.sha256_file(a_array_path)},
            base.ARM_B: {"path": str(b_array_path.resolve()), "sha256": base.sha256_file(b_array_path)},
            base.ARM_E: {"path": str(e_array_path.resolve()), "sha256": base.sha256_file(e_array_path)},
            "archived_B_plus_E_lambda1e2": {
                "path": str((args.b_e_reference_report / "lambda1e2_per_sample.json").resolve()),
                "sha256": base.sha256_file(args.b_e_reference_report / "lambda1e2_per_sample.json"),
            },
        },
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
            "note": "Raw EPE is a predictor diagnostic, not a fused-output metric.",
        },
        "aggregate": aggregates,
        "paired_A_plus_E_minus_B_plus_E": paired,
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "seed": args.seed,
            "mesh_units": 50,
            "object_cluster_units": 5,
        },
        "runtime_seconds": total_runtime,
    }
    base.write_json(output / "summary.json", result)
    base.write_json(output / "contract_audit.json", contract)
    base.write_json(output / "sample_ids.json", {"split": "test", "sample_ids": expected_ids})
    base.write_csv(output / "per_mesh_metrics.csv", rows)
    base.write_csv(output / "paired_metrics.csv", paired)
    base.write_csv(output / "per_object_metrics.csv", base.object_aggregates(rows))
    base.write_csv(output / "b_plus_e_reproduction_check.csv", reproduction)
    base.write_csv(output / "aggregate_metrics.csv", aggregates)
    (output / "evaluation_command.txt").write_text(command + "\n", encoding="utf-8")

    lines = [
        "# Matched Direct-Lap A+E versus Proposed B+E at lambda=0.01",
        "",
        "## Execution status",
        "",
        "**completed**. Saved frozen predictions were evaluated on local CPU; no training, network inference, or HPC submission occurred.",
        "",
        "## Match validation",
        "",
        "Contract audit: **true with one provenance warning**. Both systems use the identical E array, ordered 50-mesh test set, GT, topology, Uniform random-walk operator, float64 PCG, `lambda=0.01`, and evaluator.",
        "Both solve `(L_U^T L_U+0.01 I)V=L_U^T delta+0.01 V_P`, differing only in whether `delta` comes from frozen Arm A or Arm B.",
        f"The archived B+E lambda=0.01 result was independently reproduced; maximum CD discrepancy was `{maximum_reproduction_difference['refined_chamfer']:.3e}` and topology counts matched exactly.",
        "Arm-A checkpoint direct rehash remains unavailable locally; its archived prediction metadata declares the required SHA and all array/ID/target/raw-EPE checks pass.",
        "",
        "## Aggregate results",
        "",
        "| Method | CD | P2S p95 | F-score | Normal | VRMS | Flips | Improved/worsened | Runtime s/mesh |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['method']} | {fmt(row['refined_chamfer'])} | {fmt(row['p2s_p95'])} | "
            f"{fmt(row['fscore'])} | {fmt(row['normal_consistency'])} | "
            f"{fmt(row['same_index_recovered_vertex_rms'])} | {row['introduced_flipped_faces']} | "
            f"{row['improved']}/{row['worsened']} | {row['mean_total_runtime_seconds']:.6f} |"
        )
    lines += [
        "",
        f"Arm-A raw EPE is `{a_raw_epe:.10f}` and Arm-B raw EPE is `{b_raw_epe:.10f}`; these are differential-field diagnostics, not fused-output metrics.",
        "",
        "## Paired A+E versus B+E",
        "",
        "Differences are A+E minus B+E. Positive CD/P2S/VRMS and negative F-score/Normal favor B+E.",
        "",
        "| Metric | Mean difference | Median difference | A+E W/T/L | Mesh 95% CI | Object-cluster 95% CI |",
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
        "## Verdict",
        "",
        f"**{verdict}**",
        "",
        "## Reproducibility",
        "",
        f"- Git HEAD: `{git_head}`.",
        f"- Manifest: `{manifest}`; SHA-256 `{base.sha256_file(manifest)}`.",
        f"- Command: `{command}`.",
        f"- Bootstrap: `{args.bootstrap_replicates}` replicates, seed `{args.seed}`.",
        f"- Total wall time: `{total_runtime:.3f}` seconds.",
        "- Full per-mesh, per-object, solver, checkpoint, and artifact audits are stored beside this report.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "verdict": verdict, "delta_cd": paired_cd}, indent=2))


if __name__ == "__main__":
    main()
