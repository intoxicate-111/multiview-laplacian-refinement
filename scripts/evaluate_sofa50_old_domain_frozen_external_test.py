#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate locked old-domain B, E and frozen B+E against three archived methods."""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from diagnose_sofa50_exact_target_oracle import METRIC_PROTOCOL, _clean_mesh
from evaluate_sofa50_old_domain_native1920_final_sealed_test import (
    EXPECTED_ARCHIVE,
    aggregate,
    archived_rows,
    own_geometry_row,
    read_json,
    sha256_file,
    write_csv,
)
from evaluate_sofa50_old_domain_specialists import infer_e, load_e, pcg
from evaluate_sofa50_recovery_aware_ablation import _infer_recovery_arm, _load_spec
from mlr.data import Mesh
from mlr.io import load_mesh, save_mesh
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


EXTERNAL_METHODS = ("NDS", "nvdiffrec", "ExMesh")
ARCHIVE_KEYS = {"NDS": "nds", "nvdiffrec": "nvdiffrec", "ExMesh": "exmesh"}
HISTORICAL_SECONDS = {"NDS": 227.3096, "nvdiffrec": 824.982, "ExMesh": 762.4004}


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def paired_frozen(
    frozen_rows: list[dict[str, Any]], comparator_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    left = {str(row["sample_id"]): row for row in frozen_rows}
    right = {str(row["sample_id"]): row for row in comparator_rows}
    if set(left) != set(right) or len(left) != 25:
        raise RuntimeError("Paired sample identities differ")
    rng = np.random.default_rng(7)
    result: dict[str, Any] = {}
    for field, higher_is_better in (
        ("chamfer", False),
        ("p2s_p95", False),
        ("fscore", True),
        ("normal_consistency", True),
    ):
        ids = sorted(left)
        difference = np.asarray(
            [float(left[key][field]) - float(right[key][field]) for key in ids],
            dtype=np.float64,
        )
        mesh_draws = difference[
            rng.integers(0, len(difference), size=(10_000, len(difference)))
        ].mean(axis=1)
        by_object: dict[str, list[float]] = {}
        for sample_id, value in zip(ids, difference, strict=True):
            by_object.setdefault(sample_id.split("__", 1)[0], []).append(float(value))
        object_means = np.asarray(
            [np.mean(by_object[key]) for key in sorted(by_object)], dtype=np.float64
        )
        object_draws = object_means[
            rng.integers(0, len(object_means), size=(10_000, len(object_means)))
        ].mean(axis=1)
        wins = difference > 0 if higher_is_better else difference < 0
        losses = difference < 0 if higher_is_better else difference > 0
        result[field] = {
            "frozen_minus_comparator_mean": float(difference.mean()),
            "frozen_minus_comparator_median": float(np.median(difference)),
            "mesh_bootstrap_95_percent_ci": [
                float(np.quantile(mesh_draws, 0.025)),
                float(np.quantile(mesh_draws, 0.975)),
            ],
            "object_cluster_bootstrap_95_percent_ci": [
                float(np.quantile(object_draws, 0.025)),
                float(np.quantile(object_draws, 0.975)),
            ],
            "frozen_wins": int(wins.sum()),
            "frozen_losses": int(losses.sum()),
            "ties": int((difference == 0).sum()),
        }
    return result


def report_markdown(payload: dict[str, Any]) -> str:
    aggregates = {row["method"]: row for row in payload["aggregate"]}
    order = (
        "Initial mesh",
        "NDS",
        "nvdiffrec",
        "ExMesh",
        "Old-domain Arm B",
        "Old-domain Arm E",
        "Old-domain Frozen B+E",
    )
    lines = [
        "# Old-domain native-1920 frozen B+E final test",
        "",
        f"Contract audit: **{str(payload['contract_audit']).lower()}**.",
        "",
        "Arm-E and Frozen B+E were opened on the test split once, after both specialist "
        "checkpoints and the validation-selected fusion lambda were locked. Arm-B test metrics "
        "had previously been opened in the authorized Arm-B-only comparison, so this is sealed "
        "for E/Hybrid rather than a claim that no method had ever touched the test set.",
        "",
        f"Locked fusion: `lambda_old={payload['lambda_old']}`; B SHA "
        f"`{payload['checkpoint_identity']['arm_b_sha256']}`; E SHA "
        f"`{payload['checkpoint_identity']['arm_e_sha256']}`.",
        "",
        "## Unified same-input comparison",
        "",
        "| Method | CD | CD gain | P2S p95 | F-score | Normal | Improved/worsened |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in order:
        row = aggregates[method]
        lines.append(
            f"| {method} | {row['chamfer']:.10f} | "
            f"{100.0 * row['aggregate_relative_gain']:+.2f}% | {row['p2s_p95']:.10f} | "
            f"{row['fscore']:.9f} | {row['normal_consistency']:.9f} | "
            f"{row['improved']}/{row['worsened']} |"
        )
    lines += [
        "",
        "## Paired Frozen B+E comparisons",
        "",
        "Differences are Frozen B+E minus comparator. Negative CD/P2S and positive "
        "F-score/normal favor Frozen. CIs are reported both over 25 meshes and over the "
        "five object clusters (five variants per object).",
        "",
        "| Comparator | CD difference [mesh 95% CI] | Object-cluster 95% CI | CD W/L/T | P2S-p95 difference | F-score difference | Normal difference |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("Old-domain Arm B", "Old-domain Arm E", *EXTERNAL_METHODS):
        comparison = payload["paired_frozen_comparisons"][method]
        cd = comparison["chamfer"]
        p95 = comparison["p2s_p95"]
        fscore = comparison["fscore"]
        normal = comparison["normal_consistency"]
        mesh_ci = cd["mesh_bootstrap_95_percent_ci"]
        object_ci = cd["object_cluster_bootstrap_95_percent_ci"]
        lines.append(
            f"| {method} | {cd['frozen_minus_comparator_mean']:.10f} "
            f"[{mesh_ci[0]:.10f}, {mesh_ci[1]:.10f}] | "
            f"[{object_ci[0]:.10f}, {object_ci[1]:.10f}] | "
            f"{cd['frozen_wins']}/{cd['frozen_losses']}/{cd['ties']} | "
            f"{p95['frozen_minus_comparator_mean']:.10f} | "
            f"{fscore['frozen_minus_comparator_mean']:.9f} | "
            f"{normal['frozen_minus_comparator_mean']:.9f} |"
        )
    lines += [
        "",
        "## Geometry trade-offs",
        "",
        "| Method | Vertex RMS | Introduced flips / rate | New degeneracies |",
        "|---|---:|---:|---:|",
    ]
    for method in ("Old-domain Arm B", "Old-domain Arm E", "Old-domain Frozen B+E"):
        row = aggregates[method]
        lines.append(
            f"| {method} | {row['same_index_recovered_vertex_rms']:.10f} | "
            f"{row['introduced_flipped_faces']} / {100.0 * row['normalized_flip_rate']:.3f}% | "
            f"{row['new_degenerate_faces']} |"
        )
    lines += [
        "",
        "## Compute time",
        "",
        "Our timing excludes mesh export and the common evaluator. Frozen model-forward time "
        "is the sum of independent B and E forward calls; total time is model forward plus the "
        "float64 sparse fusion solve. External totals are historical pipeline measurements and "
        "are not hardware-normalized.",
        "",
        "| Method | Model forward s/mesh | Sparse solve s/mesh | Total compute s/mesh |",
        "|---|---:|---:|---:|",
    ]
    for method in ("Old-domain Arm B", "Old-domain Arm E", "Old-domain Frozen B+E"):
        row = payload["compute_time"][method]
        lines.append(
            f"| {method} | {row['model_forward_seconds_per_mesh']:.6f} | "
            f"{row['sparse_solve_seconds_per_mesh']:.6f} | "
            f"{row['total_seconds_per_mesh']:.6f} |"
        )
    for method in EXTERNAL_METHODS:
        lines.append(f"| {method} | n/a | n/a | {HISTORICAL_SECONDS[method]:.6f} |")
    lines += [
        "",
        "## Audit",
        "",
        f"- Samples: `{payload['samples']}` exact common native-1920 inputs.",
        f"- Archived NDS/nvdiffrec/ExMesh reproduction: `{str(payload['archived_comparator_reproduction']).lower()}`.",
        f"- Solver: float64 PCG, tolerance `1e-8`; all converged: `{str(payload['solver']['all_converged']).lower()}`; maximum residual `{payload['solver']['relative_residual_max']:.3e}`.",
        f"- Metric protocol: `{payload['metric_protocol']}`.",
        "- Test was not used to choose either checkpoint or lambda, and no test lambda sweep was run.",
        "",
    ]
    return "\n".join(lines)


def validate_authorization(payload: dict[str, Any]) -> None:
    required = (
        payload.get("contract_audit") is True
        and payload.get("final_selection_locked") is True
        and payload.get("validation_only_selection") is True
        and payload.get("authorize_single_test_open") is True
        and payload.get("scope") == "old_domain_frozen_b_e_vs_nds_nvdiffrec_exmesh"
        and payload.get("arm_e_or_frozen_test_open_count_before_authorization") == 0
        and payload.get("test_metric_used_to_select_e_or_lambda") is False
    )
    if not required:
        raise RuntimeError("Frozen B+E test authorization is invalid")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--benchmark-manifest", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--arm-b-run", required=True, type=Path)
    parser.add_argument("--arm-e-run", required=True, type=Path)
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    authorization = read_json(args.authorization.resolve())
    validate_authorization(authorization)
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError("Frozen final-test output exists; refusing to open the test twice")
    if str(output) != authorization["test_output_directory"]:
        raise RuntimeError("Authorized test output directory differs")
    if sha256_file(args.benchmark_manifest.resolve()) != authorization["benchmark_manifest_sha256"]:
        raise RuntimeError("Benchmark manifest changed after lock")

    benchmark = read_json(args.benchmark_manifest.resolve())
    dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), "test")
    if len(dataset) != 25 or list(dataset.sample_ids) != list(benchmark["sample_ids"]):
        raise RuntimeError("Prepared test and exact benchmark identities differ")
    device = torch.device(args.device)
    b_spec = _load_spec(args.arm_b_run.resolve(), device)
    e_spec = load_e(args.arm_e_run.resolve(), device)
    if b_spec["checkpoint_sha256"] != authorization["arm_b_checkpoint_sha256"]:
        raise RuntimeError("Arm-B checkpoint changed after lock")
    if e_spec["checkpoint_sha256"] != authorization["arm_e_checkpoint_sha256"]:
        raise RuntimeError("Arm-E checkpoint changed after lock")
    archive_root = args.archive_root.resolve()
    archive_rows: list[dict[str, Any]] = []
    for method in EXTERNAL_METHODS:
        archive_rows.extend(archived_rows(archive_root, ARCHIVE_KEYS[method], method))
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "preflight": True,
                    "samples": len(dataset),
                    "lambda_old": authorization["lambda_old"],
                    "arm_b_sha256": b_spec["checkpoint_sha256"],
                    "arm_e_sha256": e_spec["checkpoint_sha256"],
                    "archived_rows": len(archive_rows),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    output.mkdir(parents=True)
    (output / "TEST_OPENED.json").write_text(
        json.dumps(
            {
                "opened_once_for_e_and_frozen": True,
                "authorization_sha256": sha256_file(args.authorization.resolve()),
                "all_e_and_frozen_selections_locked_before_open": True,
                "intermediate_test_trajectory": False,
                "test_lambda_sweep": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    provenance = {row["sample_id"]: row for row in benchmark["samples"]}
    regularization = float(authorization["lambda_old"])
    own_rows: list[dict[str, Any]] = []
    solver_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, float | str]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for index in range(len(dataset)):
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        source = provenance[sample_id]
        initial_path = Path(source["common_initial_mesh"])
        if sha256_file(initial_path) != source["common_initial_mesh_sha256"]:
            raise RuntimeError(f"{sample_id}: common initial SHA mismatch")
        vertices = np.asarray(static["vertices"], dtype=np.float64)
        faces = np.asarray(static["faces"], dtype=np.int64)
        initial = Mesh(vertices, faces).ensure_normals()
        initial_file = load_mesh(initial_path)
        if not np.array_equal(initial_file.faces, faces) or np.max(
            np.abs(initial_file.vertices - vertices)
        ) > 1e-6:
            raise RuntimeError(f"{sample_id}: common initial identity mismatch")
        clean = _clean_mesh(static)

        synchronize(device)
        tick = time.perf_counter()
        b_values = _infer_recovery_arm(dataset, index, b_spec, device)
        synchronize(device)
        b_forward = time.perf_counter() - tick
        delta_b = b_values["prediction_raw"].numpy().astype(np.float64)

        synchronize(device)
        tick = time.perf_counter()
        delta_v_e, _ = infer_e(dataset, index, e_spec, device)
        synchronize(device)
        e_forward = time.perf_counter() - tick
        direct_e = vertices + delta_v_e

        synchronize(device)
        tick = time.perf_counter()
        b_vertices, b_audit = pcg(delta_b, vertices, static, 0.01, device)
        synchronize(device)
        b_solve = time.perf_counter() - tick
        synchronize(device)
        tick = time.perf_counter()
        frozen_vertices, frozen_audit = pcg(
            delta_b, direct_e, static, regularization, device
        )
        synchronize(device)
        frozen_solve = time.perf_counter() - tick
        if not b_audit["pcg_converged"] or not frozen_audit["pcg_converged"]:
            raise RuntimeError(f"{sample_id}: PCG failed")

        meshes = {
            "Initial mesh": initial,
            "Old-domain Arm B": Mesh(b_vertices, faces.copy()).ensure_normals(),
            "Old-domain Arm E": Mesh(direct_e, faces.copy()).ensure_normals(),
            "Old-domain Frozen B+E": Mesh(frozen_vertices, faces.copy()).ensure_normals(),
        }
        for method, mesh in meshes.items():
            row = own_geometry_row(method, sample_id, mesh, initial, clean)
            if method.startswith("Old-domain"):
                sample_dir = output / "refined_meshes" / sample_id
                sample_dir.mkdir(parents=True, exist_ok=True)
                mesh_path = sample_dir / (
                    method.lower().replace(" ", "_").replace("+", "plus") + ".obj"
                )
                save_mesh(mesh, mesh_path)
                row["final_mesh"] = str(mesh_path)
            own_rows.append(row)
        solver_rows.extend(
            (
                {"sample_id": sample_id, "state": "Arm B", **b_audit},
                {"sample_id": sample_id, "state": "Frozen B+E", **frozen_audit},
            )
        )
        timing_rows.extend(
            (
                {
                    "sample_id": sample_id,
                    "method": "Old-domain Arm B",
                    "model_forward_seconds": b_forward,
                    "sparse_solve_seconds": b_solve,
                },
                {
                    "sample_id": sample_id,
                    "method": "Old-domain Arm E",
                    "model_forward_seconds": e_forward,
                    "sparse_solve_seconds": 0.0,
                },
                {
                    "sample_id": sample_id,
                    "method": "Old-domain Frozen B+E",
                    "model_forward_seconds": b_forward + e_forward,
                    "sparse_solve_seconds": frozen_solve,
                },
            )
        )
        print(f"frozen sealed test {index + 1}/25 {sample_id}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    all_rows = own_rows + archive_rows
    method_order = (
        "Initial mesh",
        *EXTERNAL_METHODS,
        "Old-domain Arm B",
        "Old-domain Arm E",
        "Old-domain Frozen B+E",
    )
    aggregates = [aggregate(all_rows, method) for method in method_order]
    aggregate_by_method = {row["method"]: row for row in aggregates}
    archive_checks = {
        method: {
            field: abs(float(aggregate_by_method[method][field]) - expected)
            <= (1e-8 if field == "chamfer" else 1e-6)
            for field, expected in EXPECTED_ARCHIVE[method].items()
        }
        for method in EXTERNAL_METHODS
    }
    frozen_rows = [
        row for row in own_rows if row["method"] == "Old-domain Frozen B+E"
    ]
    paired = {}
    for method in ("Old-domain Arm B", "Old-domain Arm E", *EXTERNAL_METHODS):
        paired[method] = paired_frozen(
            frozen_rows, [row for row in all_rows if row["method"] == method]
        )
    compute_time = {}
    for method in ("Old-domain Arm B", "Old-domain Arm E", "Old-domain Frozen B+E"):
        selected = [row for row in timing_rows if row["method"] == method]
        forward = float(np.mean([float(row["model_forward_seconds"]) for row in selected]))
        solve = float(np.mean([float(row["sparse_solve_seconds"]) for row in selected]))
        compute_time[method] = {
            "model_forward_seconds_per_mesh": forward,
            "sparse_solve_seconds_per_mesh": solve,
            "total_seconds_per_mesh": forward + solve,
        }
    max_residual = max(float(row["pcg_relative_residual"]) for row in solver_rows)
    contract = bool(
        all(all(fields.values()) for fields in archive_checks.values())
        and all(bool(row["pcg_converged"]) for row in solver_rows)
        and max_residual <= 1.05e-8
        and len(own_rows) == 100
        and len(archive_rows) == 75
        and all(
            math.isfinite(float(row[field]))
            for row in all_rows
            for field in ("chamfer", "p2s_p95", "fscore", "normal_consistency")
        )
    )
    payload = {
        "contract_audit": contract,
        "authorization": authorization,
        "test_opened_once_for_e_and_frozen": True,
        "test_used_for_selection": False,
        "test_lambda_sweep": False,
        "fully_sealed_all_methods": False,
        "sealed_e_and_frozen_final": True,
        "samples": 25,
        "sample_ids": list(dataset.sample_ids),
        "checkpoint_identity": {
            "arm_b_sha256": b_spec["checkpoint_sha256"],
            "arm_e_sha256": e_spec["checkpoint_sha256"],
        },
        "lambda_old": regularization,
        "aggregate": aggregates,
        "paired_frozen_comparisons": paired,
        "archived_comparator_checks": archive_checks,
        "archived_comparator_reproduction": all(
            all(fields.values()) for fields in archive_checks.values()
        ),
        "solver": {
            "all_converged": all(bool(row["pcg_converged"]) for row in solver_rows),
            "relative_residual_max": max_residual,
            "iterations_mean": float(
                np.mean([float(row["pcg_iterations"]) for row in solver_rows])
            ),
            "iterations_max": int(max(int(row["pcg_iterations"]) for row in solver_rows)),
            "rows": solver_rows,
        },
        "compute_time": compute_time,
        "runtime": {
            "seconds": time.perf_counter() - started,
            "peak_gpu_memory_mb": (
                torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
                if device.type == "cuda"
                else 0.0
            ),
        },
        "metric_protocol": METRIC_PROTOCOL,
        "rows": all_rows,
        "timing_rows": timing_rows,
    }
    (output / "frozen_test_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(output / "frozen_test_per_sample.csv", all_rows)
    write_csv(output / "compute_time_per_sample.csv", timing_rows)
    (output / "REPORT.md").write_text(report_markdown(payload), encoding="utf-8")
    if not contract:
        raise RuntimeError("Frozen B+E external test contract failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
