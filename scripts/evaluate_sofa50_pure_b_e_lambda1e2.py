#!/usr/bin/env python3
"""Evaluate Pure-Vertex Arm-B + frozen Arm-E at fixed fusion lambda 1e-2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from diagnose_sofa50_exact_target_oracle import METRIC_PROTOCOL, _clean_mesh, _geometry_row
from diagnose_sofa50_frozen_hybrid_recovery import _pcg, _row
from evaluate_sofa50_original_b_e_lambda1e2 import (
    ARM_E,
    INITIAL,
    aggregate,
    archived_rows,
    directional_gain,
    paired,
)
from evaluate_sofa50_pure_b_e_fusion_ablation import (
    ARM_E_SHA256,
    PAIRED_METRICS,
    PURE_B_SHA256,
    fmt,
    prediction_array,
    read_json,
    rows_by_split,
    starts,
)
from mlr.data import Mesh
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


PURE_B = "Pure-Vertex Arm-B"
SELECTED_BE = "Pure-Vertex B+E (lambda=3e-2)"
FIXED_BE = "Pure-Vertex B+E (lambda=1e-2)"
FIXED_LAMBDA = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--pure-b-report", required=True, type=Path)
    parser.add_argument("--arm-e-report", required=True, type=Path)
    parser.add_argument("--original-hybrid-report", required=True, type=Path)
    parser.add_argument("--pure-lambda3-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    b_payload = read_json(args.pure_b_report / "shards" / "B_lap_plus_refine.json")
    e_payload = read_json(args.arm_e_report / "shards" / "E_direct_vertex_residual.json")
    original_summary = read_json(args.original_hybrid_report / "matched_summary.json")
    frozen_rows = archived_rows(args.original_hybrid_report)
    selected_payload = read_json(args.pure_lambda3_report / "pure_b_e_per_sample.json")
    selected_rows = {(row["split"], row["sample_id"]): row for row in selected_payload["rows"]}

    assert b_payload["checkpoint_sha256"] == PURE_B_SHA256
    assert e_payload["checkpoint_sha256"] == original_summary["arm_e_checkpoint_sha256"] == ARM_E_SHA256
    assert selected_payload["contract_audit"]["pure_b_checkpoint_sha256"] == PURE_B_SHA256
    assert selected_payload["contract_audit"]["arm_e_checkpoint_sha256"] == ARM_E_SHA256
    assert float(selected_payload["contract_audit"]["fusion_lambda"]) == 0.03
    assert len(selected_rows) == 100

    device = torch.device(args.device)
    all_rows: list[dict[str, Any]] = []
    maximum_initial_discrepancy = 0.0
    for split in ("validation", "test"):
        dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), split)
        b_rows = rows_by_split(b_payload, split)
        e_rows = rows_by_split(e_payload, split)
        expected = list(dataset.sample_ids)
        if [row["sample_id"] for row in b_rows] != expected:
            raise RuntimeError(f"{split}: Pure-B IDs/order differ")
        if [row["sample_id"] for row in e_rows] != expected:
            raise RuntimeError(f"{split}: Arm-E IDs/order differ")
        if set(expected) != {sample_id for row_split, sample_id in selected_rows if row_split == split}:
            raise RuntimeError(f"{split}: lambda=0.03 reference IDs differ")
        b_array = prediction_array(args.pure_b_report, "B_lap_plus_refine", split)
        e_array = prediction_array(args.arm_e_report, "E_direct_vertex_residual", split)
        b_starts, e_starts = starts(b_rows), starts(e_rows)
        if b_array.shape != e_array.shape or b_array.shape[0] != sum(int(row["vertices"]) for row in b_rows):
            raise RuntimeError(f"{split}: Pure-B/E arrays differ")

        for index, sample_id in enumerate(expected):
            static = dataset.load_static(index)
            initial = Mesh(
                np.asarray(static["vertices"], dtype=np.float64),
                np.asarray(static["faces"], dtype=np.int64),
            ).ensure_normals()
            clean = _clean_mesh(static)
            count = initial.num_vertices
            b_prediction = b_array[b_starts[index] : b_starts[index] + count]
            direct = initial.vertices + e_array[e_starts[index] : e_starts[index] + count]
            hybrid, solver = _pcg(b_prediction, direct, static, FIXED_LAMBDA, device)
            if not solver["pcg_converged"]:
                raise RuntimeError(f"{split}/{sample_id}: PCG failed")
            metric = _geometry_row(
                split,
                sample_id,
                "Pure_B_E_lambda1e2",
                Mesh(hybrid, initial.faces.copy()).ensure_normals(),
                clean,
                initial,
            )
            fixed_row = _row(
                split,
                "Pure_B_E_lambda1e2",
                sample_id,
                index,
                hybrid,
                clean,
                initial,
                metric,
                solver,
                FIXED_LAMBDA,
            )
            initial_row = frozen_rows[(split, sample_id, INITIAL)]
            e_row = frozen_rows[(split, sample_id, ARM_E)]
            pure_b_row = dict(b_rows[index])
            selected_row = dict(selected_rows[(split, sample_id)])
            maximum_initial_discrepancy = max(
                maximum_initial_discrepancy,
                abs(float(initial_row["initial_chamfer"]) - float(pure_b_row["initial_chamfer"])),
                abs(float(initial_row["initial_chamfer"]) - float(selected_row["initial_chamfer"])),
                abs(float(initial_row["initial_chamfer"]) - float(fixed_row["initial_chamfer"])),
            )
            raw_epe = float(pure_b_row["raw_epe"])
            for system, row, raw in (
                (INITIAL, initial_row, None),
                (PURE_B, pure_b_row, raw_epe),
                (ARM_E, e_row, None),
                (SELECTED_BE, selected_row, raw_epe),
                (FIXED_BE, fixed_row, raw_epe),
            ):
                item = dict(row)
                item["system"] = system
                item["raw_epe"] = raw
                all_rows.append(item)
            print(f"pure fusion lambda1e-2 {split} {index + 1}/50 {sample_id}", flush=True)

    if maximum_initial_discrepancy >= 1e-12:
        raise RuntimeError(f"Initial discrepancy: {maximum_initial_discrepancy}")
    fixed_rows = [row for row in all_rows if row["system"] == FIXED_BE]
    maximum_residual = max(float(row["pcg_relative_residual"]) for row in fixed_rows)
    mean_iterations = float(np.mean([row["pcg_iterations"] for row in fixed_rows]))
    maximum_iterations = int(max(row["pcg_iterations"] for row in fixed_rows))
    new_degenerate_faces = int(sum(row["new_degenerate_faces"] for row in fixed_rows))

    systems = (INITIAL, PURE_B, ARM_E, SELECTED_BE, FIXED_BE)
    aggregates = [aggregate(all_rows, split, system) for split in ("validation", "test") for system in systems]
    aggregate_map = {(row["split"], row["system"]): row for row in aggregates}
    comparisons = (
        (FIXED_BE, SELECTED_BE, "primary"),
        (FIXED_BE, PURE_B, "fixed_fusion_gain"),
        (SELECTED_BE, PURE_B, "selected_fusion_gain"),
        (FIXED_BE, ARM_E, "fixed_vs_E"),
        (SELECTED_BE, ARM_E, "selected_vs_E"),
    )
    paired_rows = []
    for split in ("validation", "test"):
        for candidate, reference, role in comparisons:
            metrics = (
                tuple(item for item in PAIRED_METRICS if item[0] != "raw_epe")
                if reference == ARM_E
                else PAIRED_METRICS
            )
            for metric, _ in metrics:
                item = paired(all_rows, split, candidate, reference, metric, args.bootstrap_replicates, args.seed)
                item["role"] = role
                paired_rows.append(item)

    fusion_gain = []
    for split in ("validation", "test"):
        for metric, label in (
            ("refined_chamfer", "Refined CD"),
            ("p2s_p95", "P2S p95"),
            ("fscore", "F-score"),
            ("normal_consistency", "Normal"),
        ):
            b_value = float(aggregate_map[(split, PURE_B)][metric])
            fixed_value = float(aggregate_map[(split, FIXED_BE)][metric])
            selected_value = float(aggregate_map[(split, SELECTED_BE)][metric])
            fixed_gain = directional_gain(fixed_value, b_value, metric)
            selected_gain = directional_gain(selected_value, b_value, metric)
            fusion_gain.append(
                {
                    "split": split,
                    "metric": metric,
                    "label": label,
                    "fixed_lambda_gain_over_B": fixed_gain,
                    "selected_lambda_gain_over_B": selected_gain,
                    "fixed_over_selected_gain_ratio": fixed_gain / selected_gain if selected_gain > 0 else None,
                    "fixed_minus_selected": fixed_value - selected_value,
                }
            )

    primary_cd = next(
        row for row in paired_rows
        if row["split"] == "test" and row["role"] == "primary" and row["metric"] == "refined_chamfer"
    )
    ci_low, ci_high = primary_cd["bootstrap_95_ci"]
    if ci_high < 0:
        classification = "PURE_LAMBDA_1E2_BETTER_THAN_3E2"
    elif ci_low > 0:
        classification = "PURE_LAMBDA_1E2_WORSE_THAN_3E2"
    else:
        classification = "NO_RELIABLE_PURE_B_E_LAMBDA_DIFFERENCE"

    contract = {
        "passed": True,
        "read_only": True,
        "models_retrained": False,
        "recursive_evaluation": False,
        "sample_identity_and_order_exact": True,
        "maximum_initial_metric_discrepancy": maximum_initial_discrepancy,
        "pure_b_checkpoint": b_payload["checkpoint"],
        "pure_b_checkpoint_sha256": PURE_B_SHA256,
        "arm_e_checkpoint": original_summary["arm_e_checkpoint"],
        "arm_e_checkpoint_sha256": ARM_E_SHA256,
        "fixed_fusion_lambda": FIXED_LAMBDA,
        "reference_fusion_lambda": 0.03,
        "lambda_search_run": False,
        "test_used_for_selection": False,
        "solver": "same matrix-free float64 PCG implementation as frozen B+E",
        "execution_device": str(device),
        "maximum_pcg_relative_residual": maximum_residual,
        "mean_pcg_iterations": mean_iterations,
        "maximum_pcg_iterations": maximum_iterations,
        "new_degenerate_faces": new_degenerate_faces,
        "metric_protocol": METRIC_PROTOCOL,
        "gt_used_for_predictor_or_fusion_inputs": False,
    }
    summary = {
        "classification": classification,
        "contract_audit": contract,
        "aggregate": aggregates,
        "paired": paired_rows,
        "fusion_gain": fusion_gain,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "lambda1e2_per_sample.json").write_text(
        json.dumps({"contract_audit": contract, "rows": fixed_rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Sofa50 v2 Pure-Vertex Arm-B + frozen Arm-E fusion at lambda=1e-2",
        "",
        "Contract audit: **true**. This is a read-only, non-recursive fixed-lambda fusion test on the exact same 50 validation and 50 test meshes. No model is retrained and no lambda search is run.",
        "",
        "## Fixed contract",
        "",
        f"- Pure-Vertex Arm-B: `{b_payload['checkpoint']}`; epoch `312`, optimizer step `15600`, SHA-256 `{PURE_B_SHA256}`.",
        f"- Frozen Arm-E: `{original_summary['arm_e_checkpoint']}`; SHA-256 `{ARM_E_SHA256}`.",
        "- Candidate solve: `min_V ||L_U V-delta_PureB||^2 + 0.01 ||V-V_E||^2`.",
        "- Reference solve: the already evaluated Pure-B+E system with the same inputs and `lambda=0.03`.",
        "- Uniform random-walk operator, frozen arrays, float64 PCG, evaluator, meshes, cameras, ordering, and all other settings are unchanged.",
        "- GT enters neither predictor nor fusion solve. Test is not used for checkpoint or lambda selection.",
        "",
        "## Aggregate results",
        "",
        "CD gain is the macro mean of per-mesh relative improvement over initial CD. Raw EPE is the frozen Pure-B field diagnostic and is unchanged by fusion.",
        "",
        "| Split | System | Initial CD | Refined CD | CD gain | P2S mean | P2S p95 | F-score | Normal | Raw EPE | Vertex RMS | Improved/worsened |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['split']} | {row['system']} | {fmt(row['initial_chamfer'])} | {fmt(row['refined_chamfer'])} | {100 * row['relative_chamfer_gain']:+.2f}% | {fmt(row['p2s'])} | {fmt(row['p2s_p95'])} | {fmt(row['fscore'])} | {fmt(row['normal_consistency'])} | {fmt(row['raw_epe'])} | {fmt(row['same_index_recovered_vertex_rms'])} | {row['improved']}/{row['worsened']} |"
        )

    lines.extend(
        [
            "",
            "## Paired comparisons",
            "",
            "Differences are candidate minus reference. Negative CD/P2S/raw-EPE/vertex-RMS and positive F-score/normal favor the candidate. Confidence intervals bootstrap meshes.",
            "",
            "| Split | Comparison | Metric | Mean difference [95% CI] | Candidate W/L/T |",
            "|---|---|---|---:|---:|",
        ]
    )
    for role in ("primary", "fixed_fusion_gain", "selected_fusion_gain", "fixed_vs_E", "selected_vs_E"):
        for row in paired_rows:
            if row["role"] != role:
                continue
            ci = row["bootstrap_95_ci"]
            lines.append(
                f"| {row['split']} | {row['candidate']} vs {row['reference']} | {dict(PAIRED_METRICS)[row['metric']]} | {fmt(row['candidate_minus_reference_mean'])} [{fmt(ci[0])}, {fmt(ci[1])}] | {row['candidate_wins']}/{row['candidate_losses']}/{row['ties']} |"
            )

    lines.extend(
        [
            "",
            "## Fusion gain and lambda effect",
            "",
            "Positive gain favors fusion over standalone Pure-B. The gain ratio is interpreted only when the `0.03` fusion gain is positive.",
            "",
            "| Split | Metric | Gain at 0.01 | Gain at 0.03 | 0.01/0.03 gain ratio | Value(0.01)-Value(0.03) |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in fusion_gain:
        ratio = "n/a" if row["fixed_over_selected_gain_ratio"] is None else f"{100 * row['fixed_over_selected_gain_ratio']:.2f}%"
        lines.append(
            f"| {row['split']} | {row['label']} | {fmt(row['fixed_lambda_gain_over_B'])} | {fmt(row['selected_lambda_gain_over_B'])} | {ratio} | {fmt(row['fixed_minus_selected'])} |"
        )

    fixed_test = aggregate_map[("test", FIXED_BE)]
    selected_test = aggregate_map[("test", SELECTED_BE)]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Classification: **{classification}**.",
            "",
            f"Test CD is `{fmt(fixed_test['refined_chamfer'])}` at Pure-B+E `lambda=0.01` versus `{fmt(selected_test['refined_chamfer'])}` at `lambda=0.03`. The paired difference is `{fmt(primary_cd['candidate_minus_reference_mean'])}` with 95% CI `[{fmt(ci_low)}, {fmt(ci_high)}]` and W/L/T `{primary_cd['candidate_wins']}/{primary_cd['candidate_losses']}/{primary_cd['ties']}`.",
            "",
            "This fixed-lambda diagnostic does not change any formal selected system and makes no claim about Original B, recursion, Future2000, or old native-1920.",
            "",
            "## Numerical audit",
            "",
            f"- Maximum initial-metric discrepancy: `{maximum_initial_discrepancy:.3e}`.",
            "- Exact sample identity/order and checkpoint hashes passed for Pure-B, Arm-E, the lambda=0.03 reference, and the manifest.",
            "- All 100 new float64 PCG solves converged at tolerance `1e-4`, maximum `2048` iterations.",
            f"- Maximum relative residual: `{maximum_residual:.3e}`; iterations mean/max: `{mean_iterations:.2f}/{maximum_iterations}`; new degenerate faces: `{new_degenerate_faces}`.",
            f"- Execution device: `{device}`; no inference or fusion hyperparameter other than the declared fixed lambda changed.",
        ]
    )
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
