#!/usr/bin/env python3
"""Consolidate recent Sofa50 Arm-B ablations and the old-domain comparison."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]

ABLATION_BASE = (
    ROOT
    / "reports/sofa50_multitopology_rawlap500_v2"
    / "pure_vertex_b_e_fusion_ablation_v1/summary.json"
)
ORIGINAL_LAMBDA = (
    ROOT
    / "reports/sofa50_multitopology_rawlap500_v2"
    / "original_b_e_fusion_lambda1e2_v1/summary.json"
)
PURE_LAMBDA = (
    ROOT
    / "reports/sofa50_multitopology_rawlap500_v2"
    / "pure_vertex_b_e_fusion_lambda1e2_v1/summary.json"
)
PURE_SINGLE = (
    ROOT
    / "reports/sofa50_multitopology_rawlap500_v2"
    / "arm_b_recovery_only_single_pass_v1/comparison_summary.json"
)
RECOVERY_SPECTRUM = (
    ROOT
    / "reports/sofa50_multitopology_rawlap500_v2"
    / "recovery_operator_spectrum_v1/recovery_operator_spectrum.json"
)
OLD_COMPARISON = (
    ROOT
    / "reports/sofa50_old_domain_native1920_b_e_v1"
    / "arm_b_external_comparison_v1/comparison.json"
)
PREVIOUS_OURS = (
    ROOT / "reports/synthetic_same_initial_benchmark_20260820/full_report/per_sample.json"
)
OLD_ARM_B_TIMING = (
    ROOT
    / "reports/sofa50_old_domain_native1920_b_e_v1"
    / "arm_b_recursive_refinement_v1/per_round.csv"
)
OLD_ARM_B_TIMING_SUMMARY = (
    ROOT
    / "reports/sofa50_old_domain_native1920_b_e_v1"
    / "arm_b_recursive_refinement_v1/summary.json"
)


def load(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fmt(value, digits: int = 9) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}g}"


def pct(value) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:+.2f}%"


def aggregate_index(rows):
    return {(row["split"], row["system"]): row for row in rows}


def spectrum_index(rows):
    return {(row["split"], row["scheme"], row["signal"]): row for row in rows}


def find_paired(rows, split, candidate, reference, metric="refined_chamfer"):
    matches = [
        row
        for row in rows
        if row["split"] == split
        and row["candidate"] == candidate
        and row["reference"] == reference
        and row["metric"] == metric
    ]
    if len(matches) != 1:
        raise AssertionError((split, candidate, reference, metric, len(matches)))
    return matches[0]


def bootstrap(values, *, seed=7, replicates=10_000):
    rng = random.Random(seed)
    n = len(values)
    samples = sorted(mean(values[rng.randrange(n)] for _ in range(n)) for _ in range(replicates))
    return [samples[int(0.025 * replicates)], samples[int(0.975 * replicates) - 1]]


def directional_wlt(differences, higher_is_better):
    wins = losses = ties = 0
    for value in differences:
        if value == 0:
            ties += 1
        elif (value > 0) == higher_is_better:
            wins += 1
        else:
            losses += 1
    return [wins, losses, ties]


def old_domain_previous_rows(rows):
    selected = [row for row in rows if row["method"] == "ours" and row["status"] == "completed"]
    if len(selected) != 25:
        raise AssertionError(f"expected 25 previous-Ours rows, got {len(selected)}")
    return selected


def old_domain_aggregate_from_previous(rows):
    return {
        "method": "Previous Ours (original architecture predict)",
        "samples": len(rows),
        "initial_chamfer": mean(row["initial_chamfer"] for row in rows),
        "chamfer": mean(row["final_chamfer"] for row in rows),
        "p2s_p95": mean(row["final_p2s_p95"] for row in rows),
        "fscore": mean(row["final_fscore"] for row in rows),
        "normal_consistency": mean(row["final_normal_consistency"] for row in rows),
        "aggregate_relative_gain": mean(
            (row["initial_chamfer"] - row["final_chamfer"]) / row["initial_chamfer"]
            for row in rows
        ),
        "improved": sum(row["final_chamfer"] < row["initial_chamfer"] for row in rows),
        "worsened": sum(row["final_chamfer"] > row["initial_chamfer"] for row in rows),
        "introduced_flipped_faces": sum(row["introduced_flipped_faces"] for row in rows),
        "introduced_flipped_faces_comparable": True,
        "connectivity": "preserved",
    }


def normalize_old_row(row):
    normalized = dict(row)
    normalized["connectivity"] = "changed" if row["method"] == "ExMesh" else "preserved"
    return normalized


def old_per_sample(old_rows, previous_rows):
    result = {}
    for row in old_rows:
        result[(row["method"], row["sample_id"])] = {
            "chamfer": row["chamfer"],
            "p2s_p95": row["p2s_p95"],
            "fscore": row["fscore"],
            "normal_consistency": row["normal_consistency"],
            "initial_chamfer": row["initial_chamfer"],
        }
    for row in previous_rows:
        result[("Previous Ours (original architecture predict)", row["sample_id"])] = {
            "chamfer": row["final_chamfer"],
            "p2s_p95": row["final_p2s_p95"],
            "fscore": row["final_fscore"],
            "normal_consistency": row["final_normal_consistency"],
            "initial_chamfer": row["initial_chamfer"],
        }
    return result


def old_paired(candidate, reference, sample_ids, rows):
    metrics = {
        "CD": ("chamfer", False),
        "P2S p95": ("p2s_p95", False),
        "F-score": ("fscore", True),
        "Normal": ("normal_consistency", True),
    }
    output = []
    for label, (field, higher_is_better) in metrics.items():
        differences = [rows[(candidate, sample_id)][field] - rows[(reference, sample_id)][field] for sample_id in sample_ids]
        output.append(
            {
                "candidate": candidate,
                "reference": reference,
                "metric": label,
                "candidate_minus_reference_mean": mean(differences),
                "bootstrap_95_ci": bootstrap(differences),
                "candidate_wlt": directional_wlt(differences, higher_is_better),
            }
        )
    return output


def markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "reports/sofa50_multitopology_rawlap500_v2"
            / "recent_ablation_and_old_domain_comparison_v1"
        ),
    )
    args = parser.parse_args()

    ablation = load(ABLATION_BASE)
    original_lambda = load(ORIGINAL_LAMBDA)
    pure_lambda = load(PURE_LAMBDA)
    pure_single = load(PURE_SINGLE)
    spectrum = load(RECOVERY_SPECTRUM)
    old = load(OLD_COMPARISON)
    previous_all = load(PREVIOUS_OURS)
    old_arm_b_timing_rows = load_csv(OLD_ARM_B_TIMING)
    old_arm_b_timing_summary = load(OLD_ARM_B_TIMING_SUMMARY)

    assert ablation["classification"] == "PURE_B_E_DOES_NOT_RECOVER"
    assert original_lambda["classification"] == "LAMBDA_1E2_WORSE_THAN_SELECTED_3E2"
    assert pure_lambda["classification"] == "PURE_LAMBDA_1E2_WORSE_THAN_3E2"
    assert pure_single["classification"] == "PURE_VERTEX_ERROR_WORSE"
    assert ablation["contract_audit"]["passed"] is True
    assert original_lambda["contract_audit"]["passed"] is True
    assert pure_lambda["contract_audit"]["passed"] is True
    assert pure_single["contract_audit"] is True
    assert spectrum["contract_audit"] is True
    assert spectrum["read_only"] is True
    assert spectrum["models_retrained"] is False
    assert spectrum["lambda"] == 0.03
    assert spectrum["arm_b_checkpoint_sha256"] == ablation["contract_audit"]["original_b_checkpoint_sha256"]
    assert spectrum["arm_e_checkpoint_sha256"] == ablation["contract_audit"]["arm_e_checkpoint_sha256"]
    assert old["contract_audit"] is True
    assert old["archived_comparator_reproduction"] is True
    assert old_arm_b_timing_summary["contract_audit"]["passed"] is True
    assert old_arm_b_timing_summary["reference_r1_audit"]["passed"] is True
    assert old_arm_b_timing_summary["checkpoint_sha256"] == old["arm_b_checkpoint_sha256"]

    base = aggregate_index(ablation["aggregate"])
    original_lam = aggregate_index(original_lambda["aggregate"])
    pure_lam = aggregate_index(pure_lambda["aggregate"])
    systems = [
        ("Initial mesh", base, "Initial mesh"),
        ("Original Arm-B", base, "Original Arm-B"),
        ("Pure-Vertex Arm-B", base, "Pure-Vertex Arm-B"),
        ("Arm-E", original_lam, "Arm-E"),
        ("Original B+E (lambda=3e-2)", base, "Original Arm-B + Arm-E"),
        ("Original B+E (lambda=1e-2)", original_lam, "Original B+E (lambda=1e-2)"),
        ("Pure-Vertex B+E (lambda=3e-2)", base, "Pure-Vertex Arm-B + Arm-E"),
        ("Pure-Vertex B+E (lambda=1e-2)", pure_lam, "Pure-Vertex B+E (lambda=1e-2)"),
    ]
    matched_rows = []
    for split in ("validation", "test"):
        for display, source, source_name in systems:
            row = dict(source[(split, source_name)])
            row["system"] = display
            matched_rows.append(row)

    pure_single_test_cd = pure_single["splits"]["test"]["paired"]["refined_chamfer"]
    pure_single_effect = {
        "bootstrap_95_ci": pure_single_test_cd["bootstrap_95_ci"],
        "candidate": "Pure-Vertex Arm-B",
        "candidate_losses": pure_single_test_cd["new_losses"],
        "candidate_minus_reference_mean": pure_single_test_cd["new_minus_reference_mean"],
        "candidate_wins": pure_single_test_cd["new_wins"],
        "metric": "refined_chamfer",
        "reference": "Original Arm-B",
        "role": "primary",
        "split": "test",
        "ties": pure_single_test_cd["ties"],
    }
    effects = [
        (
            "Pure B vs Original B",
            pure_single_effect,
        ),
        (
            "Pure B+E 0.03 vs Original B+E 0.03",
            find_paired(
                ablation["paired"],
                "test",
                "Pure-Vertex Arm-B + Arm-E",
                "Original Arm-B + Arm-E",
            ),
        ),
        (
            "Original B+E 0.01 vs 0.03",
            find_paired(
                original_lambda["paired"],
                "test",
                "Original B+E (lambda=1e-2)",
                "Original B+E (lambda=3e-2)",
            ),
        ),
        (
            "Pure B+E 0.01 vs 0.03",
            find_paired(
                pure_lambda["paired"],
                "test",
                "Pure-Vertex B+E (lambda=1e-2)",
                "Pure-Vertex B+E (lambda=3e-2)",
            ),
        ),
    ]

    lambda_metric_labels = (
        ("refined_chamfer", "CD"),
        ("p2s_p95", "P2S p95"),
        ("fscore", "F-score"),
        ("normal_consistency", "Normal"),
        ("raw_epe", "Raw EPE"),
        ("same_index_recovered_vertex_rms", "Vertex RMS"),
    )
    lambda_paired_test = []
    for variant, source, candidate, reference in (
        (
            "Original B+E",
            original_lambda["paired"],
            "Original B+E (lambda=1e-2)",
            "Original B+E (lambda=3e-2)",
        ),
        (
            "Pure-Vertex B+E",
            pure_lambda["paired"],
            "Pure-Vertex B+E (lambda=1e-2)",
            "Pure-Vertex B+E (lambda=3e-2)",
        ),
    ):
        for metric, metric_label in lambda_metric_labels:
            row = dict(find_paired(source, "test", candidate, reference, metric))
            row["variant"] = variant
            row["metric_label"] = metric_label
            lambda_paired_test.append(row)

    previous_rows = old_domain_previous_rows(previous_all)
    previous_ids = sorted(row["sample_id"] for row in previous_rows)
    old_ids = sorted(old["sample_ids"])
    if previous_ids != old_ids:
        raise AssertionError("old-domain Previous-Ours and Arm-B sample identities differ")

    old_aggregates = {row["method"]: normalize_old_row(row) for row in old["aggregate"]}
    old_aggregates["Previous Ours (original architecture predict)"] = old_domain_aggregate_from_previous(previous_rows)

    archived_runtime_names = {
        "Initial mesh": "initial",
        "Previous Ours (original architecture predict)": "ours",
        "NDS": "nds",
        "nvdiffrec": "nvdiffrec",
        "ExMesh": "exmesh",
    }
    for display, archive_name in archived_runtime_names.items():
        runtime_rows = [
            row
            for row in previous_all
            if row["method"] == archive_name and row["status"] == "completed"
        ]
        if len(runtime_rows) != 25:
            raise AssertionError(f"expected 25 timing rows for {archive_name}, got {len(runtime_rows)}")
        old_aggregates[display]["runtime_seconds_per_mesh"] = mean(
            float(row["runtime_seconds"]) for row in runtime_rows
        )
        old_aggregates[display]["runtime_scope"] = "archived method pipeline total"

    r1_timing = [row for row in old_arm_b_timing_rows if row["round"] == "1"]
    if len(r1_timing) != 25 or sorted(row["sample_id"] for row in r1_timing) != old_ids:
        raise AssertionError("Old-domain Arm-B R1 timing rows do not match the formal 25 samples")
    arm_b_forward_seconds = mean(float(row["runtime_inference_seconds"]) for row in r1_timing)
    arm_b_solve_seconds = mean(float(row["runtime_solve_seconds"]) for row in r1_timing)
    arm_b_compute_seconds = arm_b_forward_seconds + arm_b_solve_seconds
    old_aggregates["Old-domain Arm B"].update(
        {
            "runtime_forward_seconds_per_mesh": arm_b_forward_seconds,
            "runtime_sparse_solve_seconds_per_mesh": arm_b_solve_seconds,
            "runtime_seconds_per_mesh": arm_b_compute_seconds,
            "runtime_scope": "model forward + sparse solve; evaluator excluded",
            "runtime_device": old_arm_b_timing_summary["execution"],
        }
    )
    old_order = [
        "Initial mesh",
        "Previous Ours (original architecture predict)",
        "NDS",
        "nvdiffrec",
        "ExMesh",
        "Old-domain Arm B",
    ]
    old_aggregate_rows = [old_aggregates[name] for name in old_order]
    per_sample = old_per_sample(old["rows"], previous_rows)
    initial_differences = [
        per_sample[("Initial mesh", sample_id)]["initial_chamfer"]
        - per_sample[("Previous Ours (original architecture predict)", sample_id)]["initial_chamfer"]
        for sample_id in old_ids
    ]
    max_initial_discrepancy = max(abs(value) for value in initial_differences)

    old_pairs = []
    for comparator in (
        "Previous Ours (original architecture predict)",
        "NDS",
        "nvdiffrec",
        "ExMesh",
    ):
        old_pairs.extend(old_paired("Old-domain Arm B", comparator, old_ids, per_sample))

    matched_table_rows = []
    for row in matched_rows:
        matched_table_rows.append(
            [
                row["split"],
                row["system"],
                fmt(row["refined_chamfer"]),
                fmt(row["p2s_p95"]),
                fmt(row["fscore"]),
                fmt(row["normal_consistency"]),
                fmt(row["raw_epe"]),
                fmt(row["same_index_recovered_vertex_rms"]),
                f'{row["improved"]}/{row["worsened"]}',
            ]
        )

    effect_rows = []
    for label, row in effects:
        ci = row["bootstrap_95_ci"]
        effect_rows.append(
            [
                label,
                f'{fmt(row["candidate_minus_reference_mean"])} [{fmt(ci[0])}, {fmt(ci[1])}]',
                f'{row["candidate_wins"]}/{row["candidate_losses"]}/{row["ties"]}',
            ]
        )

    lambda_effect_rows = []
    for row in lambda_paired_test:
        ci = row["bootstrap_95_ci"]
        lambda_effect_rows.append(
            [
                row["variant"],
                row["metric_label"],
                f'{fmt(row["candidate_minus_reference_mean"])} [{fmt(ci[0])}, {fmt(ci[1])}]',
                f'{row["candidate_wins"]}/{row["candidate_losses"]}/{row["ties"]}',
            ]
        )

    spectral = spectrum_index(spectrum["aggregate"])
    spectral_error_rows = []
    for signal, label in (
        ("archived_b_error", "Original Arm-B error"),
        ("e_error", "Arm-E error"),
        ("hybrid_error", "Original B+E error"),
    ):
        row = spectral[("test", "relative", signal)]
        spectral_error_rows.append(
            [
                label,
                fmt(row["total_energy"], 8),
                pct(row["low_fraction"]),
                pct(row["mid_fraction"]),
                pct(row["high_fraction"]),
            ]
        )
    spectral_change_rows = []
    for signal, label in (
        ("hybrid_minus_b_dagger", "Hybrid - unanchored B reference"),
        ("hybrid_minus_archived_b", "Hybrid - archived Arm-B"),
        ("hybrid_minus_e", "Hybrid - Arm-E"),
    ):
        row = spectral[("test", "fusion", signal)]
        spectral_change_rows.append(
            [
                label,
                pct(row["e_dominant_fraction"]),
                pct(row["transition_fraction"]),
                pct(row["b_dominant_fraction"]),
            ]
        )
    maximum_spectral_residual = max(
        row["normal_equation_relative_residual"] for row in spectrum["audits"]
    )
    maximum_transfer_vrms = max(row["transfer_identity_vertex_rms"] for row in spectrum["audits"])

    old_table_rows = []
    for row in old_aggregate_rows:
        comparable = row.get("introduced_flipped_faces_comparable", False)
        flips = fmt(row["introduced_flipped_faces"] / row["samples"], 7) if comparable else "n/a"
        old_table_rows.append(
            [
                row["method"],
                fmt(row["chamfer"]),
                fmt(row["p2s_p95"]),
                fmt(row["fscore"]),
                fmt(row["normal_consistency"]),
                pct(row.get("aggregate_relative_gain")),
                f'{row["improved"]}/{row["worsened"]}',
                flips,
                row["connectivity"],
                fmt(row["runtime_seconds_per_mesh"], 7),
            ]
        )

    runtime_rows = [
        [
            "Old-domain Arm B",
            fmt(arm_b_forward_seconds, 7),
            fmt(arm_b_solve_seconds, 7),
            fmt(arm_b_compute_seconds, 7),
            "R1 single-pass; Quadro RTX 5000; evaluator excluded",
        ],
        [
            "Previous Ours (original architecture predict)",
            "not separately archived",
            "not separately archived",
            fmt(
                old_aggregates["Previous Ours (original architecture predict)"][
                    "runtime_seconds_per_mesh"
                ],
                7,
            ),
            "archived predictor + legacy recovery pipeline",
        ],
        [
            "NDS",
            "n/a",
            "n/a",
            fmt(old_aggregates["NDS"]["runtime_seconds_per_mesh"], 7),
            "archived method pipeline",
        ],
        [
            "nvdiffrec",
            "n/a",
            "n/a",
            fmt(old_aggregates["nvdiffrec"]["runtime_seconds_per_mesh"], 7),
            "archived method pipeline",
        ],
        [
            "ExMesh",
            "n/a",
            "n/a",
            fmt(old_aggregates["ExMesh"]["runtime_seconds_per_mesh"], 7),
            "archived method pipeline",
        ],
    ]

    old_pair_rows = []
    for row in old_pairs:
        ci = row["bootstrap_95_ci"]
        wlt = row["candidate_wlt"]
        old_pair_rows.append(
            [
                row["reference"],
                row["metric"],
                f'{fmt(row["candidate_minus_reference_mean"])} [{fmt(ci[0])}, {fmt(ci[1])}]',
                f"{wlt[0]}/{wlt[1]}/{wlt[2]}",
            ]
        )

    report = f"""# Recent Sofa50 Arm-B ablations and old-domain same-input comparison

Contract audit: **true**.

This report consolidates four recent matched-v2, single-pass Arm-B/Arm-E ablations and a separate old native-1920 same-input comparison. The two domains are reported in separate sections and their absolute metric values are not compared across sections.

## Executive findings

- The formal matched-v2 Arm-B should retain the mixed objective `L_raw-Laplacian-Huber + 1e-2 L_recovered-vertex`. Pure recovered-vertex training lowers same-index vertex RMS but worsens surface CD and raw-Laplacian EPE.
- Frozen Arm-E does not rescue the Pure-Vertex Arm-B field. At fusion `lambda=0.03`, Pure-B+E is worse than Original-B+E on every one of the 50 test meshes by CD.
- The existing validation-selected fusion `lambda=0.03` is better than the fixed diagnostic `lambda=0.01` for both B variants. The degradation is especially severe for Pure-B+E.
- Exact recovery-operator spectra show the intended division of labor at `lambda=0.03`: E changes B primarily in low-response modes, while B changes E primarily in high-response modes.
- On the separate old native-1920 inputs, the domain-trained Old-domain Arm-B has the best CD, P2S p95 and F-score among Previous Ours, NDS, nvdiffrec and ExMesh. It improves all 25 meshes. This was an authorized Arm-B-only test opening, not a sealed full B/E final evaluation.

## A. Matched-v2 loss and fusion ablations

All rows use the same 50 validation and 50 test meshes, 28x960 inputs, Uniform random-walk operator and frozen checkpoints declared by the source reports. `Pure-Vertex Arm-B` changes only the training objective to recovered-vertex MSE. B+E rows use `min_V ||L_U V-delta_B||^2 + lambda ||V-V_E||^2`. Raw EPE is a B-field diagnostic and is inherited unchanged by the corresponding fused row.

{markdown_table(
    ["Split", "System", "CD", "P2S p95", "F-score", "Normal", "Raw EPE", "Vertex RMS", "Improved/worsened"],
    matched_table_rows,
)}

### Primary paired test effects

Differences are candidate minus reference. Positive CD differences favor the reference. Confidence intervals bootstrap meshes; W/L/T is from the candidate's perspective.

{markdown_table(["Comparison", "CD difference [95% CI]", "Candidate W/L/T"], effect_rows)}

### Fixed lambda=0.01 paired test ablation

This table expands the newly completed fixed-lambda comparison beyond CD. Every difference is `value(lambda=0.01) - value(lambda=0.03)` on the same 50 test meshes. Negative values favor `lambda=0.01` for CD, P2S p95, Raw EPE and Vertex RMS; positive values favor `lambda=0.01` for F-score and Normal. W/L/T is from the `lambda=0.01` candidate's perspective.

{markdown_table(
    ["B variant", "Metric", "0.01 - 0.03 mean difference [95% CI]", "lambda=0.01 W/L/T"],
    lambda_effect_rows,
)}

For Original B+E, lowering lambda to `0.01` worsens test CD by `{fmt(find_paired(original_lambda["paired"], "test", "Original B+E (lambda=1e-2)", "Original B+E (lambda=3e-2)")["candidate_minus_reference_mean"])}` and loses on 41/50 meshes. For Pure-Vertex B+E, it worsens CD by `{fmt(find_paired(pure_lambda["paired"], "test", "Pure-Vertex B+E (lambda=1e-2)", "Pure-Vertex B+E (lambda=3e-2)")["candidate_minus_reference_mean"])}` and loses on all 50 meshes. Raw EPE is unchanged within each B variant because lambda changes only the frozen fusion solve, not the predicted B field.

The Pure-Vertex objective does optimize its direct target: test vertex RMS is `0.0105424394` versus `0.0115531855` for Original B. But its test CD rises from `0.00358497023` to `0.00397816927`, and raw EPE rises from `0.00263985669` to `0.00857208259`. The fused result shows that lower same-index vertex RMS alone does not preserve the differential field needed for B/E complementarity.

Source reports:

- [Pure-Vertex Arm-B single-pass comparison](../arm_b_recovery_only_single_pass_v1/REPORT.md)
- [Pure-Vertex B+E versus Original B+E at lambda=0.03](../pure_vertex_b_e_fusion_ablation_v1/REPORT.md)
- [Original B+E fixed lambda=0.01](../original_b_e_fusion_lambda1e2_v1/REPORT.md)
- [Pure-Vertex B+E fixed lambda=0.01](../pure_vertex_b_e_fusion_lambda1e2_v1/REPORT.md)

## B. Exact recovery-operator spectral analysis

This is the existing read-only analysis of the real Original B+E recovery operator `A_R=L_U^T L_U` on all 50 validation and 50 test meshes at the selected `lambda=0.03`. If `A_R V_B_dagger=L_U^T delta_B`, with the component-nullspace gauge copied from `V_E`, then every recovery eigenmode obeys the exact transfer

```text
v_H,k = Lambda_k/(Lambda_k+lambda) v_B_dagger,k
      + lambda/(Lambda_k+lambda) v_E,k.
```

The first table partitions test error using each mesh's relative recovery spectrum: low `[0,1/3)`, mid `[1/3,2/3)` and high `[2/3,1]` of `Lambda/Lambda_max`. Energies sum XYZ error over all test meshes.

{markdown_table(["Test signal", "Total error energy", "Low fraction", "Mid fraction", "High fraction"], spectral_error_rows)}

Hybrid has lower mid/high error energy than either standalone branch, but Arm-E has lower total error energy. The spectral result therefore supports frequency-dependent complementarity; it does not claim Hybrid dominates E in total vertex error.

For the actual fusion crossover, E-dominant means `Lambda<lambda/2`, transition means `lambda/2<=Lambda<2lambda`, and B-dominant means `Lambda>=2lambda`. These correspond to B transfer weights below `1/3`, between `1/3` and `2/3`, and above `2/3`.

{markdown_table(["Test change signal", "E-dominant", "Transition", "B-dominant"], spectral_change_rows)}

Thus `80.932%` of the Hybrid-versus-archived-B change lies in the E-dominant interval, while `73.240%` of the Hybrid-versus-E change lies in the B-dominant interval. The numerical identity is tight: maximum normal-equation residual `{maximum_spectral_residual:.3e}` and maximum transfer reconstruction VRMS `{maximum_transfer_vrms:.3e}`.

The lambda ablations are consistent with this transfer law. Lowering `lambda` from `0.03` to `0.01` decreases E's weight for every non-null mode and makes the solution trust B more strongly. Combined with Pure-B's much larger raw EPE, this provides a mechanism-level explanation for why `lambda=0.01` hurts Pure-B+E most severely. This last sentence is an inference from the exact Original-B operator spectrum plus the ablation results; no Pure-B-specific spectral decomposition was run.

The recovery spectrum is an operator-response spectrum, not automatically an intrinsic Laplace--Beltrami spectrum. A separate audit found partial correspondence with the Uniform random-walk spectrum (test reverse Spearman `0.93090`, but sampled same-band subspace overlap only `0.54110`) and only a coarse partial proxy for cotangent intrinsic frequency (test Spearman `0.74094` / `0.65287` in the two directions). Therefore the paper can describe low/high **recovery-response modes**, but should not relabel `Lambda` as cotangent frequency.

Source reports:

- [Exact recovery-operator spectrum](../recovery_operator_spectrum_v1/REPORT.md)
- [Uniform random-walk versus recovery spectrum](../uniform_rw_recovery_spectrum_correspondence_v1/REPORT.md)
- [Recovery versus cotangent spectrum](../uniform_cotangent_spectrum_correspondence_v1/REPORT.md)

## C. Old native-1920 same-input multi-metric comparison

This section uses the exact same 25 `v00`--`v04` input meshes, 28 native-1920 images/cameras and unified evaluator for every row. `Previous Ours (original architecture predict)` is the archived pre-domain-retraining predictor from the 2026-08-20 controlled benchmark; it is not a Future2000 transfer result. CD gain is the macro mean of per-mesh relative CD improvement. Mean introduced flips are shown only for connectivity-preserving outputs. Compute time is mean seconds per mesh and excludes the common evaluator for Old-domain Arm-B.

{markdown_table(
    ["Method", "CD", "P2S p95", "F-score", "Normal", "CD gain", "Improved/worsened", "Mean introduced flips", "Connectivity", "Compute s/mesh"],
    old_table_rows,
)}

The initial mesh has normal consistency `0.955190949`, so the Old-domain Arm-B's `0.948320643` is the best refined-method normal score in this table but does not exceed the unchanged input normal. ExMesh changes topology, so same-index flip counts are not comparable and are reported as `n/a`.

### Compute-time breakdown

For our current Old-domain Arm-B, the declared compute-time formula is

```text
method compute time = model forward time + sparse-matrix solve time
```

Image/model preparation included inside the measured forward call remains part of forward time. Mesh export, topology diagnostics and the unified geometry evaluator are excluded. Only the single R1 application is used; recursive R2--R5 timings and results are outside this formal comparison.

{markdown_table(["Method", "Model forward s/mesh", "Sparse solve s/mesh", "Total compute s/mesh", "Timing provenance"], runtime_rows)}

The Old-domain Arm-B total is therefore `{arm_b_forward_seconds:.6f} + {arm_b_solve_seconds:.6f} = {arm_b_compute_seconds:.6f}` seconds per mesh. The R1 timing uses the exact formal checkpoint SHA and the same 25 inputs; its re-executed geometry passed the archived-result tolerance audit. The external totals and Previous Ours total are historical implementation/hardware measurements from their completed adapters. They are useful operational references, but are not hardware-normalized algorithmic complexity comparisons.

### Paired Old-domain Arm-B comparisons

Differences are Old-domain Arm-B minus comparator. Negative CD/P2S and positive F-score/normal favor Arm-B. Confidence intervals use 10,000 paired mesh bootstraps with seed 7.

{markdown_table(["Comparator", "Metric", "Arm-B minus comparator [95% CI]", "Arm-B W/L/T"], old_pair_rows)}

Against the Previous Ours prediction, the new domain-trained Arm-B reduces mean CD by `{fmt(old_aggregates['Old-domain Arm B']['chamfer'] - old_aggregates['Previous Ours (original architecture predict)']['chamfer'])}` and wins 23/25 paired meshes. It wins 25/25 against NDS and ExMesh and 24/25 against nvdiffrec. These results establish an aggregate and paired benefit of old-domain retraining for Arm-B, but they do not establish a final B/E system result.

Source reports:

- [Old-domain Arm-B versus external methods](../../../sofa50_old_domain_native1920_b_e_v1/arm_b_external_comparison_v1/REPORT.md)
- [Previous Ours controlled same-input archive](../../../synthetic_same_initial_benchmark_20260820/full_report/FINAL_REPORT.md)

## D. How the 28-view RGB observations are generated and used

The 28 observations follow the deterministic nested layout
`cube_surface_nested_fps_antipodal_14_28_56_cpu_master_v3`. They are not 28
random orbit samples:

1. **Base 14 views.** Six cameras lie at the positive and negative coordinate-axis
   face centres of a cube with half extent `1.5`; eight cameras lie at its corners.
2. **Added 14 views.** Seven farthest-point-selected directions are added together
   with their antipodal partners. This fills the largest angular gaps left by the
   base layout while preserving opposite-view balance. The 28-view set is an exact
   prefix of the 56-view master, so the 14/28/56 view-count ablation is nested rather
   than comparing unrelated camera samples.
3. **Camera model.** Every camera looks at the origin under the right-handed CV
   convention (`+Z` forward, `+X` image-right, `+Y` image-down). At 960 resolution,
   the field of view is `90` degrees, `fx=fy=480`, and `cx=cy=480`.

For Future2000, the first 14 prepared RGB images are reused unchanged from the
upstream object observation. Only views 15--28 are newly rendered. The additional
views use the same clean source surface, fixed cameras and OpenGL/EGL renderer at
`960x960`, with four-sample MSAA, CCW winding and lit shading. RGB rendering is
two-sided; visibility is handled separately. The object is already in the prepared
canonical coordinate frame and is not renormalized during this extension.

Each object is rendered once, not once per coarse mesh. Its five synthetic-current
variants therefore share exactly the same 28 RGB images and camera matrices. The
variant changes the current vertices/connectivity and query graph, while the visual
observation stays fixed. This prevents view changes from becoming a confound in the
refinement comparison.

Masks and depth images for the added views are not persisted or supplied to the
predictor. Instead, for every current-mesh variant, renderer visibility is recomputed
from that variant's own graph with a depth-tested face-ID rasterization. Training uses
the `backface_and_occlusion` mask, so a current vertex contributes image evidence only
when it is inside the camera frustum and visible under the current graph. GT depth,
GT visibility and vertex correspondence are not model inputs.

Inside Arm-B, the 28 RGB images are processed by a shared image encoder. The HF model
does not render extra images: from each encoded feature map `F`, it constructs
`[F, F-G_sigma(F)]` using a fixed `5x5`, `sigma=1` Gaussian blur. Current vertex/query
positions are projected with each view's intrinsics and extrinsics; bilinear features
are sampled at those projected locations, invalid/hidden samples are masked, and the
remaining per-view features are mean-aggregated for each vertex. These visual features
are then combined with current-graph geometry before predicting the raw Laplacian.
Execution may process four views at a time to control memory, but all 28 views enter
the same final per-vertex aggregation.

The choice of 28 is empirical as well as structural: it preserves the nested camera
protocol and gave a better validation-loss trade-off than 14 or 56 in the completed
Sofa50 view-count study. The 56-view arm improved some raw-error tails but did not
consistently improve the selection metric and cost substantially more runtime.

Implementation sources:

- [Future2000 28-view preparation](../../../scripts/prepare_future2000_synthetic_current_28view.py)
- [Camera and OpenGL renderer](../../../src/mlr/synthetic.py)
- [Projection and bilinear feature sampling](../../../src/mlr/learned_laplacian/projection.py)
- [HF feature construction](../../../src/mlr/learned_laplacian/image_encoder.py)

## Contract and numerical audit

- Matched-v2: 50 validation and 50 test meshes; all four source classifications and contract audits were checked before aggregation.
- Recovery spectrum: all 100 matched-v2 meshes passed; Original B/E checkpoint hashes match the ablation inputs; no Pure-B-specific spectral run is claimed.
- Old domain: 25 exact sample IDs match between the Old-domain Arm-B report and the Previous Ours archive.
- Old-domain Arm-B timing: exact checkpoint and samples; R1 forward plus sparse solve only; evaluator excluded; archived-result reproduction gate passed.
- Maximum repeated initial-CD discrepancy between the two old-domain archives: `{max_initial_discrepancy:.3e}`.
- Old-domain metric protocol: `{old['metric_protocol']}`.
- Archived NDS, nvdiffrec and ExMesh aggregates reproduce the prior same-input archive: `{str(old['archived_comparator_reproduction']).lower()}`.
- No model was trained, checkpoint selected or lambda searched to create this consolidation. It reads completed reports and per-sample archives only.
- Old-domain full-model sealed final: `{str(old['sealed_full_model_final']).lower()}`.
"""

    summary = {
        "contract_audit": True,
        "scope": {
            "matched_v2_samples": {"validation": 50, "test": 50},
            "old_domain_samples": len(old_ids),
            "cross_domain_ranking_permitted": False,
        },
        "source_classifications": {
            "pure_vertex_arm_b": pure_single["classification"],
            "pure_b_e_lambda_3e2": ablation["classification"],
            "original_b_e_lambda_1e2": original_lambda["classification"],
            "pure_b_e_lambda_1e2": pure_lambda["classification"],
        },
        "matched_v2_aggregate": matched_rows,
        "matched_v2_primary_test_effects": [dict(row, comparison=label) for label, row in effects],
        "matched_v2_lambda_1e2_paired_test": lambda_paired_test,
        "recovery_operator_spectrum": {
            "contract_audit": spectrum["contract_audit"],
            "lambda": spectrum["lambda"],
            "operator": spectrum["operator"],
            "exact_characterization": spectrum["exact_characterization"],
            "test_error_energy": [
                spectral[("test", "relative", signal)]
                for signal in ("archived_b_error", "e_error", "hybrid_error")
            ],
            "test_fusion_change": [
                spectral[("test", "fusion", signal)]
                for signal in (
                    "hybrid_minus_b_dagger",
                    "hybrid_minus_archived_b",
                    "hybrid_minus_e",
                )
            ],
            "maximum_normal_equation_relative_residual": maximum_spectral_residual,
            "maximum_transfer_identity_vertex_rms": maximum_transfer_vrms,
            "pure_b_specific_spectrum_run": False,
        },
        "old_domain_aggregate": old_aggregate_rows,
        "old_domain_runtime": {
            "definition": "method_compute_time=model_forward_time+sparse_matrix_solve_time",
            "old_domain_arm_b": {
                "forward_seconds_per_mesh": arm_b_forward_seconds,
                "sparse_solve_seconds_per_mesh": arm_b_solve_seconds,
                "total_compute_seconds_per_mesh": arm_b_compute_seconds,
                "evaluator_included": False,
                "round": "R1 single-pass only",
                "execution": old_arm_b_timing_summary["execution"],
                "checkpoint_sha256": old_arm_b_timing_summary["checkpoint_sha256"],
                "reference_r1_audit_passed": old_arm_b_timing_summary["reference_r1_audit"][
                    "passed"
                ],
            },
            "archived_pipeline_seconds_per_mesh": {
                method: old_aggregates[method]["runtime_seconds_per_mesh"]
                for method in (
                    "Previous Ours (original architecture predict)",
                    "NDS",
                    "nvdiffrec",
                    "ExMesh",
                )
            },
            "hardware_normalized": False,
        },
        "old_domain_arm_b_paired": old_pairs,
        "audit": {
            "old_domain_sample_ids_match": True,
            "old_domain_sample_ids": old_ids,
            "maximum_initial_chamfer_discrepancy": max_initial_discrepancy,
            "archived_comparator_reproduction": old["archived_comparator_reproduction"],
            "sealed_full_model_final": old["sealed_full_model_final"],
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output_dir / "REPORT.md")


if __name__ == "__main__":
    main()
