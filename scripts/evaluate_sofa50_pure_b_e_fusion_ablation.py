#!/usr/bin/env python3
"""Fixed-contract Pure-Vertex Arm-B + frozen Arm-E fusion ablation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from diagnose_sofa50_exact_target_oracle import METRIC_PROTOCOL, _clean_mesh, _geometry_row
from diagnose_sofa50_frozen_hybrid_recovery import _pcg, _row
from mlr.data import Mesh
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


PURE_B_SHA256 = "3f29d66302f30a487e3aac9c7c09a5875328602cbcc715f3780aa24ba5b6367a"
ORIGINAL_B_SHA256 = "a483e2212f568e771873594cf1e37d13d62cbd2e1e72244baded7dd15573970c"
ARM_E_SHA256 = "6ed27da8759b7bd752ffa75ea8dac3977dd4ced358b5282e0c1c68f750dbade1"
PURE_B_CHECKPOINT = "/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/runs/learned_laplacian/sofa50_v2_arm_b_recovery_only_lambda1e-2_20k_seed7_bw4_v1/checkpoint_best.pt"

INITIAL = "Initial mesh"
ORIGINAL_B = "Original Arm-B"
PURE_B = "Pure-Vertex Arm-B"
ORIGINAL_BE = "Original Arm-B + Arm-E"
PURE_BE = "Pure-Vertex Arm-B + Arm-E"

LOWER_IS_BETTER = {
    "refined_chamfer",
    "p2s",
    "p2s_p95",
    "raw_epe",
    "same_index_recovered_vertex_rms",
}
HIGHER_IS_BETTER = {"fscore", "normal_consistency"}
PAIRED_METRICS = (
    ("refined_chamfer", "Refined CD"),
    ("p2s_p95", "P2S p95"),
    ("fscore", "F-score"),
    ("normal_consistency", "Normal"),
    ("raw_epe", "Raw EPE"),
    ("same_index_recovered_vertex_rms", "Vertex RMS"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--pure-b-report", required=True, type=Path)
    parser.add_argument("--original-b-report", required=True, type=Path)
    parser.add_argument("--arm-e-report", required=True, type=Path)
    parser.add_argument("--original-hybrid-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rows_by_split(payload: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    return [dict(row) for row in payload["rows"] if row["split"] == split]


def prediction_array(report: Path, arm: str, split: str) -> np.ndarray:
    archive = np.load(report / "shards" / f"{arm}_prediction_arrays.npz")
    return archive[f"{split}_prediction"].astype(np.float64)


def starts(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    result: list[int] = []
    offset = 0
    for row in rows:
        result.append(offset)
        offset += int(row["vertices"])
    return result


def typed_csv_row(row: Mapping[str, str]) -> dict[str, Any]:
    integer_fields = {"vertices", "faces", "introduced_flipped_faces", "new_degenerate_faces"}
    boolean_fields = {"improved", "worsened"}
    result: dict[str, Any] = dict(row)
    for field in integer_fields:
        result[field] = int(row[field])
    for field in boolean_fields:
        result[field] = row[field] == "True"
    for field in (
        "initial_chamfer",
        "refined_chamfer",
        "relative_chamfer_gain",
        "eta",
        "p2s",
        "p2s_p95",
        "fscore",
        "normal_consistency",
        "same_index_recovered_vertex_rms",
    ):
        result[field] = float(row[field])
    return result


def original_rows(report: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    mapping = {
        "initial": INITIAL,
        "B_lap_plus_refine": ORIGINAL_B,
        "Hybrid_B_laplacian_E_anchor": ORIGINAL_BE,
    }
    with (report / "matched_per_sample.csv").open(encoding="utf-8", newline="") as handle:
        rows = [typed_csv_row(row) for row in csv.DictReader(handle) if row["arm"] in mapping]
    return {(row["split"], row["sample_id"], mapping[row["arm"]]): row for row in rows}


def bootstrap_ci(values: np.ndarray, replicates: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    return [float(value) for value in np.quantile(values[indices].mean(axis=1), [0.025, 0.975])]


def aggregate(rows: Sequence[Mapping[str, Any]], split: str, system: str) -> dict[str, Any]:
    selected = [row for row in rows if row["split"] == split and row["system"] == system]
    if len(selected) != 50:
        raise RuntimeError(f"Expected 50 rows for {split}/{system}, found {len(selected)}")
    output = {
        "split": split,
        "system": system,
        "samples": len(selected),
        "initial_chamfer": float(np.mean([row["initial_chamfer"] for row in selected])),
        "refined_chamfer": float(np.mean([row["refined_chamfer"] for row in selected])),
        "relative_chamfer_gain": float(np.mean([row["relative_chamfer_gain"] for row in selected])),
        "p2s": float(np.mean([row["p2s"] for row in selected])),
        "p2s_p95": float(np.mean([row["p2s_p95"] for row in selected])),
        "fscore": float(np.mean([row["fscore"] for row in selected])),
        "normal_consistency": float(np.mean([row["normal_consistency"] for row in selected])),
        "same_index_recovered_vertex_rms": float(
            np.mean([row["same_index_recovered_vertex_rms"] for row in selected])
        ),
        "improved": int(sum(bool(row["improved"]) for row in selected)),
        "worsened": int(sum(bool(row["worsened"]) for row in selected)),
    }
    available_raw = [row for row in selected if row.get("raw_epe") is not None]
    output["raw_epe"] = (
        float(np.average([row["raw_epe"] for row in available_raw], weights=[row["vertices"] for row in available_raw]))
        if available_raw
        else None
    )
    return output


def paired(
    rows: Sequence[Mapping[str, Any]],
    split: str,
    candidate: str,
    reference: str,
    metric: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    left = {row["sample_id"]: row for row in rows if row["split"] == split and row["system"] == candidate}
    right = {row["sample_id"]: row for row in rows if row["split"] == split and row["system"] == reference}
    if left.keys() != right.keys() or len(left) != 50:
        raise RuntimeError(f"Paired keys differ for {split}: {candidate} vs {reference}")
    values = np.asarray([float(left[key][metric]) - float(right[key][metric]) for key in sorted(left)])
    favorable = values < 0 if metric in LOWER_IS_BETTER else values > 0
    unfavorable = values > 0 if metric in LOWER_IS_BETTER else values < 0
    return {
        "split": split,
        "candidate": candidate,
        "reference": reference,
        "metric": metric,
        "candidate_minus_reference_mean": float(values.mean()),
        "bootstrap_95_ci": bootstrap_ci(values, replicates, seed),
        "candidate_wins": int(favorable.sum()),
        "candidate_losses": int(unfavorable.sum()),
        "ties": int((~favorable & ~unfavorable).sum()),
    }


def directional_improvement(candidate: float, reference: float, metric: str) -> float:
    return reference - candidate if metric in LOWER_IS_BETTER else candidate - reference


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.9g}"


def main() -> None:
    args = parse_args()
    pure_payload = read_json(args.pure_b_report / "shards" / "B_lap_plus_refine.json")
    original_b_payload = read_json(args.original_b_report / "shards" / "B_lap_plus_refine.json")
    e_payload = read_json(args.arm_e_report / "shards" / "E_direct_vertex_residual.json")
    original_summary = read_json(args.original_hybrid_report / "matched_summary.json")
    original = original_rows(args.original_hybrid_report)

    assert pure_payload["checkpoint_sha256"] == PURE_B_SHA256
    assert original_b_payload["checkpoint_sha256"] == original_summary["arm_b_checkpoint_sha256"] == ORIGINAL_B_SHA256
    assert e_payload["checkpoint_sha256"] == original_summary["arm_e_checkpoint_sha256"] == ARM_E_SHA256
    fusion_lambda = float(original_summary["lambda_hybrid_best"])
    assert fusion_lambda == 0.03
    assert original_summary["lambda_selection_split"] == "validation"
    assert original_summary["contract_audit"] is True
    assert pure_payload["parameter_count"] == original_b_payload["parameter_count"] == 826115

    all_rows: list[dict[str, Any]] = []
    maximum_initial_discrepancy = 0.0
    device = torch.device(args.device)
    for split in ("validation", "test"):
        dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), split)
        pure_rows = rows_by_split(pure_payload, split)
        original_b_rows = rows_by_split(original_b_payload, split)
        e_rows = rows_by_split(e_payload, split)
        expected = list(dataset.sample_ids)
        for name, source in (("pure B", pure_rows), ("original B", original_b_rows), ("Arm E", e_rows)):
            if [row["sample_id"] for row in source] != expected:
                raise RuntimeError(f"{split}: {name} sample IDs/order differ from manifest")
        pure_array = prediction_array(args.pure_b_report, "B_lap_plus_refine", split)
        e_array = prediction_array(args.arm_e_report, "E_direct_vertex_residual", split)
        pure_starts, e_starts = starts(pure_rows), starts(e_rows)
        if pure_array.shape != e_array.shape or pure_array.shape[0] != sum(row["vertices"] for row in pure_rows):
            raise RuntimeError(f"{split}: archived array shapes do not match")

        for index, sample_id in enumerate(expected):
            static = dataset.load_static(index)
            initial = Mesh(
                np.asarray(static["vertices"], dtype=np.float64),
                np.asarray(static["faces"], dtype=np.int64),
            ).ensure_normals()
            clean = _clean_mesh(static)
            count = initial.num_vertices
            b_prediction = pure_array[pure_starts[index] : pure_starts[index] + count]
            e_displacement = e_array[e_starts[index] : e_starts[index] + count]
            direct = initial.vertices + e_displacement
            hybrid, solver = _pcg(b_prediction, direct, static, fusion_lambda, device)
            if not solver["pcg_converged"]:
                raise RuntimeError(f"{split}/{sample_id}: fusion PCG did not converge")
            metric = _geometry_row(
                split,
                sample_id,
                "Pure_Vertex_B_plus_E",
                Mesh(hybrid, initial.faces.copy()).ensure_normals(),
                clean,
                initial,
            )
            hybrid_row = _row(
                split,
                "Pure_Vertex_B_plus_E",
                sample_id,
                index,
                hybrid,
                clean,
                initial,
                metric,
                solver,
                fusion_lambda,
            )

            old_initial = original[(split, sample_id, INITIAL)]
            old_b = original[(split, sample_id, ORIGINAL_B)]
            old_be = original[(split, sample_id, ORIGINAL_BE)]
            pure_b = dict(pure_rows[index])
            maximum_initial_discrepancy = max(
                maximum_initial_discrepancy,
                abs(float(old_initial["initial_chamfer"]) - float(pure_b["initial_chamfer"])),
                abs(float(old_initial["initial_chamfer"]) - float(hybrid_row["initial_chamfer"])),
            )

            raw_original = float(original_b_rows[index]["raw_epe"])
            raw_pure = float(pure_b["raw_epe"])
            for system, row, raw_epe in (
                (INITIAL, old_initial, None),
                (ORIGINAL_B, old_b, raw_original),
                (PURE_B, pure_b, raw_pure),
                (ORIGINAL_BE, old_be, raw_original),
                (PURE_BE, hybrid_row, raw_pure),
            ):
                item = dict(row)
                item["system"] = system
                item["raw_epe"] = raw_epe
                all_rows.append(item)
            print(f"fusion {split} {index + 1}/50 {sample_id}", flush=True)

    if maximum_initial_discrepancy >= 1e-12:
        raise RuntimeError(f"Initial metric mismatch: {maximum_initial_discrepancy}")

    pure_hybrid_rows = [row for row in all_rows if row["system"] == PURE_BE]
    maximum_pcg_residual = max(float(row["pcg_relative_residual"]) for row in pure_hybrid_rows)
    mean_pcg_iterations = float(np.mean([row["pcg_iterations"] for row in pure_hybrid_rows]))
    maximum_pcg_iterations = int(max(row["pcg_iterations"] for row in pure_hybrid_rows))
    new_degenerate_faces = int(sum(row["new_degenerate_faces"] for row in pure_hybrid_rows))

    systems = (INITIAL, ORIGINAL_B, PURE_B, ORIGINAL_BE, PURE_BE)
    aggregates = [aggregate(all_rows, split, system) for split in ("validation", "test") for system in systems]
    aggregate_map = {(row["split"], row["system"]): row for row in aggregates}

    comparisons = (
        (PURE_BE, ORIGINAL_BE, "primary"),
        (PURE_BE, PURE_B, "pure_E_gain"),
        (ORIGINAL_BE, ORIGINAL_B, "original_E_gain"),
    )
    paired_rows = []
    for split in ("validation", "test"):
        for candidate, reference, role in comparisons:
            for metric, _ in PAIRED_METRICS:
                item = paired(
                    all_rows,
                    split,
                    candidate,
                    reference,
                    metric,
                    args.bootstrap_replicates,
                    args.seed,
                )
                item["role"] = role
                paired_rows.append(item)

    compensation = []
    for split in ("validation", "test"):
        for metric, label in (("refined_chamfer", "Refined CD"), ("p2s_p95", "P2S p95"), ("fscore", "F-score")):
            original_b = float(aggregate_map[(split, ORIGINAL_B)][metric])
            pure_b = float(aggregate_map[(split, PURE_B)][metric])
            original_be = float(aggregate_map[(split, ORIGINAL_BE)][metric])
            pure_be = float(aggregate_map[(split, PURE_BE)][metric])
            degradation_b = -directional_improvement(pure_b, original_b, metric)
            degradation_be = -directional_improvement(pure_be, original_be, metric)
            original_e_gain = directional_improvement(original_be, original_b, metric)
            pure_e_gain = directional_improvement(pure_be, pure_b, metric)
            ratio = 1.0 - degradation_be / degradation_b if degradation_b > 0 else None
            compensation.append(
                {
                    "split": split,
                    "metric": metric,
                    "label": label,
                    "degradation_B": degradation_b,
                    "residual_degradation_BE": degradation_be,
                    "compensation_ratio": ratio,
                    "original_E_gain": original_e_gain,
                    "pure_E_gain": pure_e_gain,
                    "original_E_relative_gain": original_e_gain / abs(original_b),
                    "pure_E_relative_gain": pure_e_gain / abs(pure_b),
                }
            )

    primary_cd = next(
        row for row in paired_rows
        if row["split"] == "test" and row["role"] == "primary" and row["metric"] == "refined_chamfer"
    )
    test_compensation = next(
        row for row in compensation if row["split"] == "test" and row["metric"] == "refined_chamfer"
    )
    cd_ratio = float(test_compensation["compensation_ratio"])
    ci_low, ci_high = primary_cd["bootstrap_95_ci"]
    if ci_high < 0:
        classification = "PURE_B_E_EXCEEDS_ORIGINAL_B_E"
    elif ci_low <= 0 <= ci_high and cd_ratio >= 0.9:
        classification = "PURE_B_E_FULLY_RECOVERS"
    elif cd_ratio > 0:
        classification = "PURE_B_E_PARTIALLY_RECOVERS"
    else:
        classification = "PURE_B_E_DOES_NOT_RECOVER"

    contract = {
        "passed": True,
        "read_only": True,
        "models_retrained": False,
        "recursive_evaluation": False,
        "sample_identity_and_order_exact": True,
        "maximum_initial_metric_discrepancy": maximum_initial_discrepancy,
        "pure_b_checkpoint_sha256": PURE_B_SHA256,
        "pure_b_checkpoint": PURE_B_CHECKPOINT,
        "original_b_checkpoint_sha256": ORIGINAL_B_SHA256,
        "original_b_checkpoint": original_summary["arm_b_checkpoint"],
        "arm_e_checkpoint_sha256": ARM_E_SHA256,
        "arm_e_checkpoint": original_summary["arm_e_checkpoint"],
        "arm_e_predictions_reused_read_only": True,
        "original_b_e_results_reused_read_only": True,
        "standalone_b_lambda": 0.01,
        "fusion_lambda": fusion_lambda,
        "fusion_lambda_source": "existing Original B+E validation-only selection",
        "pure_b_specific_retuning": False,
        "solver": "same matrix-free float64 PCG implementation as existing frozen B+E",
        "execution_device": str(device),
        "maximum_pcg_relative_residual": maximum_pcg_residual,
        "mean_pcg_iterations": mean_pcg_iterations,
        "maximum_pcg_iterations": maximum_pcg_iterations,
        "new_degenerate_faces": new_degenerate_faces,
        "metric_protocol": METRIC_PROTOCOL,
        "gt_used_for_predictor_or_fusion_inputs": False,
        "gt_use": "training supervision for completed checkpoints and evaluation only",
        "test_used_for_selection": False,
    }

    summary = {
        "classification": classification,
        "contract_audit": contract,
        "aggregate": aggregates,
        "paired": paired_rows,
        "compensation": compensation,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "pure_b_e_per_sample.json").write_text(
        json.dumps(
            {
                "contract_audit": contract,
                "rows": [row for row in all_rows if row["system"] == PURE_BE],
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Sofa50 v2 Pure-Vertex Arm-B + frozen Arm-E fusion ablation",
        "",
        "Contract audit: **true**. This is a read-only, non-recursive, exact paired comparison on 50 validation and 50 test meshes. No model is retrained and no Pure-B-specific hyperparameter is selected.",
        "",
        "## Fixed contract",
        "",
        f"- Pure-Vertex Arm-B: `{PURE_B_CHECKPOINT}`; epoch `312`, optimizer step `15600`, SHA-256 `{PURE_B_SHA256}`.",
        f"- Original Arm-B: `{original_summary['arm_b_checkpoint']}`; SHA-256 `{ORIGINAL_B_SHA256}`.",
        f"- Frozen Arm-E: `{original_summary['arm_e_checkpoint']}`; SHA-256 `{ARM_E_SHA256}`; its archived predictions are reused unchanged.",
        "- Standalone B recovery retains `lambda=1e-2`. Fusion retains the existing Original B+E validation-selected `lambda=3e-2`; no lambda sweep is run for Pure-B.",
        "- Fusion is the unchanged solve `min_V ||L_U V-delta_B||^2 + 0.03 ||V-V_E||^2`, using the same Uniform random-walk operator and float64 PCG implementation.",
        "- The existing Original B+E per-sample artifacts are reused read-only. GT enters neither predictor nor fusion solve.",
        "",
        "The two lambda values are distinct established roles: `1e-2` is the standalone Arm-B input-anchor recovery, while `3e-2` is the frozen B+E positional-anchor fusion weight. Using `1e-2` for fusion would change the existing Original B+E system and violate the requested checkpoint-only ablation.",
        "",
        "## Aggregate results",
        "",
        "CD gain is the macro mean of each mesh's relative improvement over its initial CD. Raw EPE is the B-field diagnostic and is therefore inherited unchanged by the corresponding B+E row.",
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
    for role in ("primary", "pure_E_gain", "original_E_gain"):
        for row in paired_rows:
            if row["role"] != role:
                continue
            label = dict(PAIRED_METRICS)[row["metric"]]
            ci = row["bootstrap_95_ci"]
            lines.append(
                f"| {row['split']} | {row['candidate']} vs {row['reference']} | {label} | {fmt(row['candidate_minus_reference_mean'])} [{fmt(ci[0])}, {fmt(ci[1])}] | {row['candidate_wins']}/{row['candidate_losses']}/{row['ties']} |"
            )

    lines.extend(
        [
            "",
            "## Arm-E compensation",
            "",
            "All quantities below use a positive-is-better directional convention. `D_B` is the degradation from Original B to Pure-B; `D_BE` is the residual degradation after adding the same E; compensation is `1-D_BE/D_B` when `D_B>0`.",
            "",
            "| Split | Metric | D_B | D_BE | Compensation | E gain on Original B | E gain on Pure-B |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in compensation:
        ratio = "n/a" if row["compensation_ratio"] is None else f"{100 * row['compensation_ratio']:.2f}%"
        lines.append(
            f"| {row['split']} | {row['label']} | {fmt(row['degradation_B'])} | {fmt(row['residual_degradation_BE'])} | {ratio} | {fmt(row['original_E_gain'])} | {fmt(row['pure_E_gain'])} |"
        )

    test_original_gain = float(test_compensation["original_E_gain"])
    test_pure_gain = float(test_compensation["pure_E_gain"])
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Classification: **{classification}**.",
            "",
            f"On test, Pure-B causes CD degradation `D_B={fmt(test_compensation['degradation_B'])}`. With the same frozen E and fusion rule, residual degradation is `D_BE={fmt(test_compensation['residual_degradation_BE'])}`, giving a compensation ratio of `{100 * cd_ratio:.2f}%`.",
            f"The primary Pure-B+E minus Original-B+E paired CD difference is `{fmt(primary_cd['candidate_minus_reference_mean'])}` with 95% CI `[{fmt(ci_low)}, {fmt(ci_high)}]` and W/L/T `{primary_cd['candidate_wins']}/{primary_cd['candidate_losses']}/{primary_cd['ties']}`.",
            "",
            f"Arm-E provides a {'larger' if test_pure_gain > test_original_gain else 'smaller'} absolute test-CD gain on Pure-B (`{fmt(test_pure_gain)}`) than on Original B (`{fmt(test_original_gain)}`). Relative gains are `{100 * test_compensation['pure_E_relative_gain']:.2f}%` and `{100 * test_compensation['original_E_relative_gain']:.2f}%`, respectively.",
            "",
            "This decision concerns only the matched Sofa50 v2, single-pass frozen fusion contract. It makes no claim about recursion, Future2000, old native-1920, or any other configuration.",
            "",
            "## Numerical audit",
            "",
            f"- Maximum initial-metric discrepancy: `{maximum_initial_discrepancy:.3e}`.",
            "- Exact sample identity and ordering passed for Pure-B, Original B, Arm-E, and the prepared manifest.",
            "- All 100 new float64 PCG fusion solves converged at the existing tolerance `1e-4` and maximum `2048` iterations.",
            f"- Maximum relative residual: `{maximum_pcg_residual:.3e}`; PCG iterations mean/max: `{mean_pcg_iterations:.2f}/{maximum_pcg_iterations}`; new degenerate faces: `{new_degenerate_faces}`.",
            f"- Execution device: `{device}`. Device choice changes neither the float64 equation nor the frozen fusion parameters.",
            "- Test was evaluated only after all checkpoints and the Original B+E validation-selected fusion lambda were frozen.",
        ]
    )
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
