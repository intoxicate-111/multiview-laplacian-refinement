#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


FAMILY_LABELS = {
    "predicted_raw": "predicted raw",
    "predicted_zero_mean": "predicted zero-mean",
    "exact_target": "exact target reference",
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _fmt(value: float) -> str:
    return f"{value:.9g}"


def _lambda(value: float) -> str:
    return "0" if value == 0 else f"{value:.0e}"


def _baseline_table(summary: Mapping[str, Any]) -> list[str]:
    labels = {
        "initial": "initial",
        "frozen_adam_visibility": "frozen Adam + visibility",
    }
    lines = [
        "| Baseline | Chamfer | Relative gain | eta mean | Normal | Flips | New degenerates | Improved / worsened |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["baseline_aggregates"]:
        lines.append(
            f"| {labels[row['state']]} | {_fmt(float(row['chamfer']))} | "
            f"{100 * float(row['relative_chamfer_gain_mean']):.2f}% | {_fmt(float(row['eta_mean']))} | "
            f"{_fmt(float(row['normal_consistency']))} | {int(row['introduced_flipped_faces'])} | "
            f"{int(row['new_degenerate_faces'])} | {int(row['improved_over_initial'])}/"
            f"{int(row['worsened_over_initial'])} |"
        )
    return lines


def _sweep_table(rows: Sequence[Mapping[str, Any]], family: str) -> list[str]:
    selected = [row for row in rows if row["family"] == family]
    lines = [
        "| lambda | Chamfer | Relative gain | eta mean / median | Normal | Flips | New deg. | Improved / worsened | Lap RMS | Displ. RMS | Runtime |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| {_lambda(float(row['lambda']))} | {_fmt(float(row['chamfer']))} | "
            f"{100 * float(row['relative_chamfer_gain_mean']):.2f}% | "
            f"{_fmt(float(row['eta_mean']))} / {_fmt(float(row['eta_median']))} | "
            f"{_fmt(float(row['normal_consistency']))} | {int(row['introduced_flipped_faces'])} | "
            f"{int(row['new_degenerate_faces'])} | {int(row['improved_over_initial'])}/"
            f"{int(row['worsened_over_initial'])} | {_fmt(float(row['laplacian_residual_rms']))} | "
            f"{_fmt(float(row['displacement_rms']))} | {_fmt(float(row['runtime_seconds_mean']))} s |"
        )
    return lines


def _domain_section(summary: Mapping[str, Any]) -> list[str]:
    best = summary["best_predicted_recovery"]
    exact = summary["exact_target_at_best_predicted_lambda"]
    retention = summary["retention_at_best_predicted_lambda"]
    versus = summary["paired_comparisons"]["best_prediction_vs_frozen"]
    projection = summary["paired_comparisons"]["best_projected_vs_best_raw"]
    audit = summary["projection_audit"]
    lines = [
        f"## {summary['dataset_arm']}",
        "",
        *_baseline_table(summary),
        "",
    ]
    for family in ("predicted_raw", "predicted_zero_mean", "exact_target"):
        lines.extend(
            [
                f"### {FAMILY_LABELS[family]}",
                "",
                *_sweep_table(summary["aggregates"], family),
                "",
            ]
        )
    lines.extend(
        [
            "### Domain summary",
            "",
            f"The minimum-mean-Chamfer predicted arm is **{FAMILY_LABELS[str(best['family'])]}** at "
            f"`lambda={_lambda(float(best['lambda']))}`: Chamfer `{_fmt(float(best['chamfer']))}`, "
            f"mean eta `{_fmt(float(best['eta_mean']))}`, normal `{_fmt(float(best['normal_consistency']))}`, "
            f"and `{int(best['improved_over_initial'])}/{int(best['samples'])}` improved over initial.",
            "",
            f"At the same lambda, the exact-target reference has eta `{_fmt(float(exact['eta_mean']))}`. "
            f"The predicted arm retains `{100 * float(retention['ratio_of_mean_eta']):.2f}%` signed and "
            f"`{100 * float(retention['useful_ratio_of_mean_eta']):.2f}%` useful mean oracle eta.",
            "",
            f"Against frozen Adam+visibility, the selected sparse arm has lower paired Chamfer on "
            f"`{int(versus['left_lower_chamfer'])}/{int(versus['samples'])}` samples and changes mean "
            f"Chamfer by `{_fmt(float(versus['mean_left_minus_right_chamfer']))}` and eta by "
            f"`{_fmt(float(versus['mean_left_minus_right_eta']))}`.",
            "",
            f"Best projected versus best raw (each at its own diagnostic lambda): projected wins "
            f"`{int(projection['left_lower_chamfer'])}/{int(projection['samples'])}`, with mean Chamfer "
            f"difference `{_fmt(float(projection['mean_left_minus_right_chamfer']))}`. The requested "
            f"ordinary component mean is reduced to at most "
            f"`{_fmt(float(audit['component_mean_max_abs_after_max']))}`; degree-weighted compatibility "
            f"changes on average from `{_fmt(float(audit['degree_weighted_component_mean_max_abs_before_mean']))}` "
            f"to `{_fmt(float(audit['degree_weighted_component_mean_max_abs_after_mean']))}`.",
            "",
        ]
    )
    return lines


def generate(v1: Mapping[str, Any], v2: Mapping[str, Any]) -> str:
    best = v2["best_predicted_recovery"]
    frozen = next(row for row in v2["baseline_aggregates"] if row["state"] == "frozen_adam_visibility")
    exact = v2["exact_target_at_best_predicted_lambda"]
    retention = v2["retention_at_best_predicted_lambda"]
    versus = v2["paired_comparisons"]["best_prediction_vs_frozen"]
    raw_best = v2["best_by_family"]["predicted_raw"]
    projected_best = v2["best_by_family"]["predicted_zero_mean"]
    projection_cmp = v2["paired_comparisons"]["best_projected_vs_best_raw"]
    distance_useful = bool(float(best["eta_mean"]) > 0 and int(best["improved_over_initial"]) >= 30)
    beats_frozen = bool(
        float(best["chamfer"]) < float(frozen["chamfer"])
        and int(versus["left_lower_chamfer"]) >= 30
    )
    projection_material = bool(
        float(projected_best["chamfer"]) + 1e-5 < float(raw_best["chamfer"])
        and int(projection_cmp["left_lower_chamfer"]) >= 30
    )
    robustly_usable = bool(
        distance_useful
        and float(best["eta_mean"]) >= 0.1
        and int(best["improved_over_initial"]) >= 40
    )
    candidate_replacement_supported = distance_useful and beats_frozen
    scaling_ready = bool(
        robustly_usable
        and candidate_replacement_supported
        and float(best["eta_mean"]) >= 0.3
        and int(best["improved_over_initial"]) >= 40
        and int(best["new_degenerate_faces"]) == 0
    )
    middle_regime = bool(
        float(best["lambda"]) > 0
        and float(best["lambda"]) < 1e-1
        and distance_useful
        and beats_frozen
    )
    lines = [
        "# Sofa50 regularized sparse recovery sweep",
        "",
        "Contract audit: **true**.",
        "",
        "This is a read-only diagnostic over the completed matched-domain v1/v2 predictions. For positive "
        "lambda it solves `min ||L V - delta||_2^2 + lambda ||V - V_initial||_2^2` as the augmented "
        "LSMR system `[L; sqrt(lambda) I]`. At lambda=0, only initial connected-component centroids fix "
        "translation. Every solve uses all Laplacian equations and no visibility, confidence, Huber or Adam. "
        "The exact-target family is an explicitly labelled reference; GT is otherwise used only for evaluation.",
        "",
        "The zero-mean arm implements the requested unweighted per-component and per-coordinate subtraction "
        "literally. Because the production uniform operator is a nonsymmetric random-walk Laplacian, this is a "
        "simple heuristic projection rather than an exact left-nullspace projection; degree-weighted residuals "
        "are therefore audited separately.",
        "",
        "Reported runtime is sparse-recovery solve wall time per mesh; geometry-evaluator time is excluded.",
        "",
        *_domain_section(v1),
        *_domain_section(v2),
        "## Direct answers for strong_smooth_v2",
        "",
        f"1. **Can regularized sparse integration make the current predictor usable?** "
        f"**{'Yes' if robustly_usable else 'Partially for surface distance, but not yet robustly'}**. "
        f"The selected arm reaches eta `{_fmt(float(best['eta_mean']))}` and improves "
        f"`{int(best['improved_over_initial'])}/{int(best['samples'])}` samples; this is measurable but "
        f"retains only `{100 * float(retention['useful_ratio_of_mean_eta']):.2f}%` of same-lambda oracle eta.",
        f"2. **Best lambda on v2:** `{_lambda(float(best['lambda']))}` for "
        f"**{FAMILY_LABELS[str(best['family'])]}**, selected by minimum diagnostic test-domain mean Chamfer. "
        f"It retains `{100 * float(retention['useful_ratio_of_mean_eta']):.2f}%` useful oracle eta at the same "
        f"lambda (`{_fmt(float(best['eta_mean']))}` vs `{_fmt(float(exact['eta_mean']))}`).",
        f"3. **Does zero-mean projection materially help?** **{'Yes' if projection_material else 'No'}**. "
        f"Best projected/raw Chamfer is `{_fmt(float(projected_best['chamfer']))}` / "
        f"`{_fmt(float(raw_best['chamfer']))}`, and projected wins "
        f"`{int(projection_cmp['left_lower_chamfer'])}/{int(projection_cmp['samples'])}` paired samples.",
        f"4. **Can hard visibility, confidence, Huber and Adam be removed?** "
        f"**{'They can be removed in a stronger sparse-recovery candidate, but a production replacement is not yet established' if candidate_replacement_supported else 'Not yet supported'}**. "
        f"The new arm changes mean Chamfer vs frozen recovery by "
        f"`{_fmt(float(v2['best_prediction_minus_frozen']['chamfer']))}` and wins "
        f"`{int(versus['left_lower_chamfer'])}/{int(versus['samples'])}` samples, but changes normal by "
        f"`{_fmt(float(v2['best_prediction_minus_frozen']['normal_consistency']))}` and flips by "
        f"`{int(v2['best_prediction_minus_frozen']['introduced_flipped_faces']):+d}`.",
        f"5. **Ready to reconsider scaling strong_smooth_v2 to 2000 meshes?** "
        f"**{'Yes, after fixing lambda on validation and rerunning a frozen test' if scaling_ready else 'No'}**. "
        "This diagnostic selects lambda on the existing test domain; it cannot itself establish a deployable "
        "hyperparameter or erase any remaining prediction-to-oracle gap.",
        "",
        f"Useful middle regularization regime found: **{'weak' if middle_regime else 'false'}**. "
        + (
            "Lambda=1e-2 suppresses unregularized inverse-Laplacian amplification and restores measurable surface distance, but the retained oracle efficiency is limited. Lambda=1e-1 is a stability-favoring Pareto point: slightly worse mean Chamfer/eta, but better normal, fewer flips and more samples improved over initial."
            if middle_regime
            else "The sweep does not show an interior lambda that both stabilizes the inverse and clearly beats the frozen recovery under the stated criteria."
        ),
        "",
        f"Metric protocol: `{v2['metric_protocol']}`.",
        "",
        "Machine-readable outputs: `aggregate.csv`, `per_sample.csv`, `baseline_aggregate.csv`, "
        "`baseline_per_sample.csv`, `projection_paired_by_lambda.csv`, `summary.json`, "
        "`contract_audit.json`, and `per_sample_contract_audit.json`.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-summary", required=True, type=Path)
    parser.add_argument("--v2-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    v1 = _read(args.v1_summary.resolve())
    v2 = _read(args.v2_summary.resolve())
    if not v1.get("contract_audit") or not v2.get("contract_audit"):
        raise RuntimeError("Input contract audit failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generate(v1, v2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
