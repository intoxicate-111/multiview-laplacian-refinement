#!/usr/bin/env python3
from __future__ import annotations

"""Compare validation-selected old-domain Arm B with three archived external methods."""

import argparse
import csv
import json
import math
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
from evaluate_sofa50_old_domain_specialists import pcg
from evaluate_sofa50_recovery_aware_ablation import _infer_recovery_arm, _load_spec
from mlr.data import Mesh
from mlr.io import load_mesh, save_mesh
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


METHODS = ("NDS", "nvdiffrec", "ExMesh")
ARCHIVE_KEYS = {"NDS": "nds", "nvdiffrec": "nvdiffrec", "ExMesh": "exmesh"}


def paired_arm_b(
    arm_b_rows: list[dict[str, Any]], comparator_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    left = {str(row["sample_id"]): row for row in arm_b_rows}
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
        difference = np.asarray(
            [float(left[key][field]) - float(right[key][field]) for key in sorted(left)],
            dtype=np.float64,
        )
        samples = difference[
            rng.integers(0, len(difference), size=(10_000, len(difference)))
        ].mean(axis=1)
        wins = difference > 0 if higher_is_better else difference < 0
        losses = difference < 0 if higher_is_better else difference > 0
        result[field] = {
            "arm_b_minus_comparator_mean": float(difference.mean()),
            "arm_b_minus_comparator_median": float(np.median(difference)),
            "bootstrap_95_percent_ci": [
                float(np.quantile(samples, 0.025)),
                float(np.quantile(samples, 0.975)),
            ],
            "arm_b_wins": int(wins.sum()),
            "arm_b_losses": int(losses.sum()),
            "ties": int((difference == 0).sum()),
        }
    return result


def report_markdown(payload: dict[str, Any]) -> str:
    aggregates = {row["method"]: row for row in payload["aggregate"]}
    lines = [
        "# Old-domain native-1920 Arm-B versus external methods",
        "",
        f"Contract audit: **{str(payload['contract_audit']).lower()}**.",
        "",
        "This is a user-authorized Arm-B-only test opening, not the sealed final B/E/fusion evaluation. The Arm-B checkpoint was selected only by validation objective. NDS, nvdiffrec, and ExMesh rows are read from the existing same-input archive and were already recomputed with the identical unified evaluator.",
        "",
        f"Arm-B selected checkpoint: `{payload['arm_b_checkpoint']}`; SHA-256 `{payload['arm_b_checkpoint_sha256']}`; selected epoch `{payload['arm_b_selected_epoch']}` / optimizer step `{payload['arm_b_selected_step']}`.",
        "",
        "## Unified same-input comparison",
        "",
        "| Method | CD | P2S p95 | F-score | Normal | Improved/worsened |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in ("Old-domain Arm B", "NDS", "nvdiffrec", "ExMesh", "Initial mesh"):
        row = aggregates[method]
        lines.append(
            f"| {method} | {row['chamfer']:.10f} | {row['p2s_p95']:.10f} | "
            f"{row['fscore']:.9f} | {row['normal_consistency']:.9f} | "
            f"{row['improved']}/{row['worsened']} |"
        )
    lines += [
        "",
        "## Paired Arm-B comparisons",
        "",
        "Differences are Arm B minus comparator. Negative CD/P2S differences and positive F-score/normal differences favor Arm B.",
        "",
        "| Comparator | CD difference [95% CI] | CD W/L/T | P2S-p95 difference | F-score difference | Normal difference |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        comparison = payload["paired_comparisons"][method]
        cd = comparison["chamfer"]
        p95 = comparison["p2s_p95"]
        fscore = comparison["fscore"]
        normal = comparison["normal_consistency"]
        lines.append(
            f"| {method} | {cd['arm_b_minus_comparator_mean']:.10f} "
            f"[{cd['bootstrap_95_percent_ci'][0]:.10f}, {cd['bootstrap_95_percent_ci'][1]:.10f}] | "
            f"{cd['arm_b_wins']}/{cd['arm_b_losses']}/{cd['ties']} | "
            f"{p95['arm_b_minus_comparator_mean']:.10f} | "
            f"{fscore['arm_b_minus_comparator_mean']:.9f} | "
            f"{normal['arm_b_minus_comparator_mean']:.9f} |"
        )
    lines += [
        "",
        "## Audit",
        "",
        f"- Samples: `{payload['samples']}` exact common native-1920 `v00`--`v04` inputs.",
        f"- Arm-B recovery: Uniform random-walk Laplacian, `lambda=1e-2`, float64 PCG, tolerance `1e-8`; maximum residual `{payload['solver']['relative_residual_max']:.3e}`.",
        f"- Archived comparator reproduction: `{str(payload['archived_comparator_reproduction']).lower()}`.",
        f"- Metric protocol: `{payload['metric_protocol']}`.",
        "- Test access occurred before old-domain Arm-E/fusion/continuous final selection; these results must not be described as a sealed full-model final test.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--benchmark-manifest", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--arm-b-run", required=True, type=Path)
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    authorization = read_json(args.authorization.resolve())
    if not (
        authorization.get("contract_audit") is True
        and authorization.get("authorize_test_open") is True
        and authorization.get("scope") == "old_domain_arm_b_vs_nds_nvdiffrec_exmesh"
    ):
        raise RuntimeError("Arm-B external-comparison authorization is invalid")
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError("Output exists; refusing to overwrite the Arm-B test comparison")
    output.mkdir(parents=True)

    benchmark = read_json(args.benchmark_manifest.resolve())
    dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), "test")
    if len(dataset) != 25 or list(dataset.sample_ids) != list(benchmark["sample_ids"]):
        raise RuntimeError("Prepared test and benchmark identities differ")
    provenance = {row["sample_id"]: row for row in benchmark["samples"]}
    device = torch.device(args.device)
    b_spec = _load_spec(args.arm_b_run.resolve(), device)
    if b_spec["checkpoint_sha256"] != authorization["arm_b_checkpoint_sha256"]:
        raise RuntimeError("Arm-B checkpoint SHA differs from authorization")
    if int(b_spec["parameter_count"]) != 826115:
        raise RuntimeError("Arm-B parameter count mismatch")

    own_rows: list[dict[str, Any]] = []
    solver_rows: list[dict[str, Any]] = []
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
        values = _infer_recovery_arm(dataset, index, b_spec, device)
        delta_b = values["prediction_raw"].numpy().astype(np.float64)
        b_vertices, audit = pcg(delta_b, vertices, static, 0.01, device)
        if not audit["pcg_converged"]:
            raise RuntimeError(f"{sample_id}: Arm-B PCG failed")
        b_mesh = Mesh(b_vertices, faces.copy()).ensure_normals()
        initial_row = own_geometry_row("Initial mesh", sample_id, initial, initial, clean)
        b_row = own_geometry_row("Old-domain Arm B", sample_id, b_mesh, initial, clean)
        sample_dir = output / "refined_meshes" / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        mesh_path = sample_dir / "old_domain_arm_b.obj"
        save_mesh(b_mesh, mesh_path)
        b_row["final_mesh"] = str(mesh_path)
        own_rows.extend((initial_row, b_row))
        solver_rows.append({"sample_id": sample_id, **audit})
        print(f"Arm-B test {index + 1}/25 {sample_id}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    archive_rows: list[dict[str, Any]] = []
    for method in METHODS:
        archive_rows.extend(
            archived_rows(args.archive_root.resolve(), ARCHIVE_KEYS[method], method)
        )
    arm_b_rows = [row for row in own_rows if row["method"] == "Old-domain Arm B"]
    aggregates = [
        aggregate(own_rows + archive_rows, method)
        for method in ("Old-domain Arm B", *METHODS, "Initial mesh")
    ]
    aggregate_by_method = {row["method"]: row for row in aggregates}
    archive_checks = {
        method: {
            field: abs(float(aggregate_by_method[method][field]) - expected)
            <= (1e-8 if field == "chamfer" else 1e-6)
            for field, expected in EXPECTED_ARCHIVE[method].items()
        }
        for method in METHODS
    }
    paired = {
        method: paired_arm_b(
            arm_b_rows, [row for row in archive_rows if row["method"] == method]
        )
        for method in METHODS
    }
    max_residual = max(float(row["pcg_relative_residual"]) for row in solver_rows)
    contract = bool(
        all(all(fields.values()) for fields in archive_checks.values())
        and all(bool(row["pcg_converged"]) for row in solver_rows)
        and max_residual <= 1.05e-8
        and len(own_rows) == 50
        and len(archive_rows) == 75
        and all(
            math.isfinite(float(row[field]))
            for row in own_rows + archive_rows
            for field in ("chamfer", "p2s_p95", "fscore", "normal_consistency")
        )
    )
    payload = {
        "contract_audit": contract,
        "authorization": authorization,
        "test_opened": True,
        "sealed_full_model_final": False,
        "samples": 25,
        "sample_ids": list(dataset.sample_ids),
        "arm_b_checkpoint": b_spec["checkpoint"],
        "arm_b_checkpoint_sha256": b_spec["checkpoint_sha256"],
        "arm_b_selected_epoch": int(authorization["arm_b_selected_epoch"]),
        "arm_b_selected_step": int(authorization["arm_b_selected_step"]),
        "aggregate": aggregates,
        "paired_comparisons": paired,
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
        "metric_protocol": METRIC_PROTOCOL,
        "rows": own_rows + archive_rows,
    }
    (output / "comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(output / "per_sample.csv", own_rows + archive_rows)
    (output / "REPORT.md").write_text(report_markdown(payload), encoding="utf-8")
    if not contract:
        raise RuntimeError("Arm-B external-comparison contract failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
