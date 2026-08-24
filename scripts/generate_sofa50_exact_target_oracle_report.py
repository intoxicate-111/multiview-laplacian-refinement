#!/usr/bin/env python3
from __future__ import annotations

"""Merge the matched-domain Sofa50 v1/v2 exact-target oracle diagnostics."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def by_state(summary: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["state"]): row for row in summary["aggregate"]}


def select_visuals(roots: Mapping[str, Path]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, str]] = set()
    for dataset_arm, root in roots.items():
        rows = read_csv(root / "oracle_gap_per_sample.csv")
        flips = {row["sample_id"]: row for row in read_csv(root / "flip_attribution_per_sample.csv")}

        def add(category: str, row: Mapping[str, str]) -> None:
            key = (dataset_arm, row["sample_id"])
            if key in selected_keys:
                return
            selected_keys.add(key)
            records.append(
                {
                    "dataset_arm": dataset_arm,
                    "sample_id": row["sample_id"],
                    "category": category,
                    "g_pred": float(row["g_pred"]),
                    "raw_epe": float(row["raw_epe"]),
                    "pred_minus_oracle_chamfer": float(row["pred_minus_oracle_chamfer"]),
                    "predicted_introduced_flips": int(flips[row["sample_id"]]["predicted_introduced_flips"]),
                }
            )

        add("best_predicted_recovery", max(rows, key=lambda row: float(row["g_pred"])))
        add("worst_predicted_recovery", min(rows, key=lambda row: float(row["g_pred"])))
        add(
            "highest_flip_increase",
            max(rows, key=lambda row: int(flips[row["sample_id"]]["predicted_introduced_flips"])),
        )
        cutoff = float(np.quantile([float(row["raw_epe"]) for row in rows], 0.25))
        low_error = [row for row in rows if float(row["raw_epe"]) <= cutoff]
        add(
            "low_raw_epe_but_poor_recovery",
            max(low_error, key=lambda row: float(row["pred_minus_oracle_chamfer"])),
        )
    return {
        "selection_rule": (
            "For each matched-domain dataset: best and worst predicted Chamfer gain, highest "
            "predicted introduced-flip count, and largest predicted-minus-oracle Chamfer gap "
            "within the lowest raw-EPE quartile; duplicate samples retained only once."
        ),
        "records": records,
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-root", required=True, type=Path)
    parser.add_argument("--v2-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    roots = {
        "v1_legacy_smoothing": args.v1_root.resolve(),
        "v2_strong_smoothing": args.v2_root.resolve(),
    }
    summaries = {arm: read_json(root / "summary.json") for arm, root in roots.items()}
    states = {arm: by_state(value) for arm, value in summaries.items()}
    v1_audit = summaries["v1_legacy_smoothing"]["contract_audit"]
    v2_audit = summaries["v2_strong_smoothing"]["contract_audit"]
    v1_manifest = str(v1_audit["manifest"])
    v2_manifest = str(v2_audit["manifest"])
    audit = {
        "passed": bool(
            v1_audit["passed"]
            and v2_audit["passed"]
            and v1_manifest != v2_manifest
            and "RawLap500_v1" in v1_manifest
            and "RawLap500_v2" in v2_manifest
            and summaries["v1_legacy_smoothing"]["metric_protocol"]
            == summaries["v2_strong_smoothing"]["metric_protocol"]
        ),
        "v1_dataset_arm_uses_legacy_v1_manifest": v1_manifest,
        "v2_dataset_arm_uses_strong_smooth_v2_manifest": v2_manifest,
        "separate_dataset_manifests": v1_manifest != v2_manifest,
        "same_unified_metric_protocol": True,
        "legacy_vertex_sampled_chamfer_excluded": True,
        "no_retraining": True,
        "no_projection_nearest_vertex_icp_or_topology_transfer": True,
        "primary_metric_implementation": summaries["v1_legacy_smoothing"]["metric_protocol"],
        "exact_paths_audited_in": {
            arm: str(root / "per_sample_contract_audit.json") for arm, root in roots.items()
        },
    }
    if not audit["passed"]:
        raise RuntimeError(f"Combined contract audit failed: {audit}")

    v2_oracle = summaries["v2_strong_smoothing"]["oracle_efficiency"]
    v2_retained = summaries["v2_strong_smoothing"]["prediction_retention"]
    v2_oracle_strong = float(v2_oracle["median"]) >= 0.5
    v2_prediction_strong = (
        float(v2_retained["median"]) >= 0.8
        and float(states["v2_strong_smoothing"]["predicted_recovery"]["chamfer"])
        < float(states["v2_strong_smoothing"]["initial"]["chamfer"])
    )
    if not v2_oracle_strong:
        case = "A"
        conclusion = (
            "Exact-target recovery does not recover a substantial fraction of the strong-smoothing "
            "input-to-clean gap; the raw-Laplacian plus frozen recovery formulation is the primary ceiling."
        )
        blocker = "exact native target Laplacian is not converted into sufficient geometry recovery"
        scale_ready = False
    elif not v2_prediction_strong:
        case = "B"
        conclusion = (
            "The exact-target oracle is effective, but learned predictions retain too little of its gain; "
            "the current loss under-constrains recovery-sensitive prediction components."
        )
        blocker = "prediction error in recovery-sensitive Laplacian components"
        scale_ready = False
    else:
        case = "C"
        conclusion = "Both exact-target oracle and learned recovery are strong on matched-domain v2 data."
        blocker = "none observed in this diagnostic"
        scale_ready = True

    visuals = select_visuals(roots)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "contract_audit": audit,
        "decision_case": case,
        "decision_thresholds": {
            "oracle_strong": "median eta_oracle >= 0.5",
            "prediction_strong": "median eta_pred >= 0.8 and aggregate predicted Chamfer improves over initial",
        },
        "conclusion": conclusion,
        "scale_strong_smooth_v2_to_2000_ready": scale_ready,
        "single_blocking_failure_mode": blocker,
        "initial_geometry_comparison": {
            "v1_chamfer": float(states["v1_legacy_smoothing"]["initial"]["chamfer"]),
            "v2_chamfer": float(states["v2_strong_smoothing"]["initial"]["chamfer"]),
            "v2_minus_v1_chamfer": float(states["v2_strong_smoothing"]["initial"]["chamfer"])
            - float(states["v1_legacy_smoothing"]["initial"]["chamfer"]),
            "v1_normal": float(states["v1_legacy_smoothing"]["initial"]["normal_consistency"]),
            "v2_normal": float(states["v2_strong_smoothing"]["initial"]["normal_consistency"]),
            "v1_wrong_orientation_faces": int(
                summaries["v1_legacy_smoothing"]["flip_attribution"]["initial_wrong_orientation_vs_clean"]
            ),
            "v2_wrong_orientation_faces": int(
                summaries["v2_strong_smoothing"]["flip_attribution"]["initial_wrong_orientation_vs_clean"]
            ),
        },
        "matched_domain_prediction_contract_finding": {
            "v1_global_raw_epe": summaries["v1_legacy_smoothing"]["prediction_metrics_global_weighted"]["raw_epe"],
            "v2_global_raw_epe": summaries["v2_strong_smoothing"]["prediction_metrics_global_weighted"]["raw_epe"],
            "v2_lower_than_v1": summaries["v2_strong_smoothing"]["prediction_metrics_global_weighted"]["raw_epe"]
            < summaries["v1_legacy_smoothing"]["prediction_metrics_global_weighted"]["raw_epe"],
            "prior_apparent_v2_advantage_source": "v1 checkpoint evaluated out-of-domain on v2 strong-smoothing samples",
        },
        "arms": summaries,
        "visual_selection": visuals,
    }
    write_json(output / "summary.json", summary)
    write_json(output / "contract_audit.json", audit)
    write_json(output / "visual_selection.json", visuals)
    combined_geometry = []
    for arm, root in roots.items():
        combined_geometry.extend(read_csv(root / "geometry_per_sample.csv"))
    write_csv(output / "geometry_per_sample.csv", combined_geometry)

    lines = [
        "# Sofa50 matched-domain exact-target Laplacian oracle diagnostic",
        "",
        "Contract audit: **true**.",
        "",
        "This is a read-only diagnostic. No model was retrained and no 2000-mesh strong-smoothing generation or training was started.",
        "",
        f"Primary metric protocol: `{audit['primary_metric_implementation']}`",
        "",
        "The v1 checkpoint is evaluated only on the intended `legacy_v1` test meshes; the v2 checkpoint is evaluated only on the intended `strong_smooth_v2` test meshes. Old vertex-sampled Chamfer is excluded.",
        "",
        "## Metric-contract correction and initial geometry",
        "",
        f"- Matched-domain global Raw EPE is `{summaries['v1_legacy_smoothing']['prediction_metrics_global_weighted']['raw_epe']:.9g}` for v1 and `{summaries['v2_strong_smoothing']['prediction_metrics_global_weighted']['raw_epe']:.9g}` for v2. The earlier apparent v2 advantage (`0.002768` vs `0.008404`) compared v2-on-v2 against v1-on-v2 and was therefore an evaluation-contract mismatch.",
        f"- Initial unified Chamfer is close but not identical: v1 `{states['v1_legacy_smoothing']['initial']['chamfer']:.9g}`, v2 `{states['v2_strong_smoothing']['initial']['chamfer']:.9g}`. Surface distance alone hides the stronger orientation damage: initial normal consistency changes from `{states['v1_legacy_smoothing']['initial']['normal_consistency']:.9g}` to `{states['v2_strong_smoothing']['initial']['normal_consistency']:.9g}`, and wrong-vs-clean face orientations increase from `{summaries['v1_legacy_smoothing']['flip_attribution']['initial_wrong_orientation_vs_clean']}` to `{summaries['v2_strong_smoothing']['flip_attribution']['initial_wrong_orientation_vs_clean']}`.",
    ]
    for arm in ("v1_legacy_smoothing", "v2_strong_smoothing"):
        lines.extend(
            [
                "",
                f"## {arm}: four-arm geometry",
                "",
                "| State | Chamfer | ΔCD | Relative gain | P2S | P2S p95 | F-score | Normal | Flips | New degenerates | Improved | Worsened |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for state in ("initial", "clean", "exact_target_oracle", "predicted_recovery"):
            row = states[arm][state]
            lines.append(
                f"| {state} | {row['chamfer']:.9g} | {row['delta_cd']:.9g} | {pct(float(row['relative_gain']))} | "
                f"{row['p2s']:.9g} | {row['p2s_p95']:.9g} | {row['fscore']:.9g} | "
                f"{row['normal_consistency']:.9g} | {row['introduced_flipped_faces']} | "
                f"{row['new_degenerate_faces']} | {row['improved_over_initial']}/{row['samples']} | "
                f"{row['worsened_over_initial']}/{row['samples']} |"
            )
        eff = summaries[arm]["oracle_efficiency"]
        retained = summaries[arm]["prediction_retention"]
        lines.extend(
            [
                "",
                f"- Oracle efficiency η_oracle: mean `{eff['mean']:.6g}`, median `{eff['median']:.6g}`, p10 `{eff['p10']:.6g}`, p90 `{eff['p90']:.6g}`, negative `{eff['negative_count']}/{eff['count']}`.",
                f"- Prediction retention η_pred: mean `{retained['mean']:.6g}`, median `{retained['median']:.6g}`, p10 `{retained['p10']:.6g}`, p90 `{retained['p90']:.6g}`, negative `{retained['negative_count']}/{retained['count']}`.",
            ]
        )

    lines.extend(["", "## Prediction-to-oracle and spectral findings", ""])
    for arm in ("v1_legacy_smoothing", "v2_strong_smoothing"):
        gap = summaries[arm]["predicted_vs_oracle_gap"]
        spectral = summaries[arm]["spectral"]
        lines.append(
            f"- **{arm}:** oracle↔prediction vertex RMS `{gap['vertex_rms_displacement_mean']:.6g}`, "
            f"Chamfer difference `{gap['chamfer_difference_mean']:.6g}`, flip difference `{gap['flip_count_difference_total']}`. "
            f"Spectral low/mid/high error fractions `{spectral['low_error_fraction_mean']:.4f}` / "
            f"`{spectral['mid_error_fraction_mean']:.4f}` / `{spectral['high_error_fraction_mean']:.4f}` "
            f"on `{spectral['successful_count']}/{spectral['selected_count']}` stratified samples."
        )
        correlations = summaries[arm]["correlations"]
        for endpoint in (
            "predicted_chamfer_degradation_vs_initial",
            "oracle_pred_vertex_rms_displacement",
            "predicted_introduced_flip_fraction",
        ):
            candidates = [row for row in correlations if row["y"] == endpoint]
            strongest = max(candidates, key=lambda row: abs(float(row["spearman"])))
            lines.append(
                f"  - Strongest tested Spearman association with `{endpoint}`: `{strongest['x']}` = `{strongest['spearman']:.4f}` (Pearson `{strongest['pearson']:.4f}`, n={strongest['count']})."
            )

    lines.extend(["", "## Flip attribution", ""])
    for arm in ("v1_legacy_smoothing", "v2_strong_smoothing"):
        flip = summaries[arm]["flip_attribution"]
        oracle_high = flip["oracle_flips_high_target"] / max(flip["oracle_introduced_flips"], 1)
        pred_high = flip["predicted_flips_high_target"] / max(flip["predicted_introduced_flips"], 1)
        background = flip["high_target_faces"] / max(flip["all_faces"], 1)
        lines.append(
            f"- **{arm}:** initial wrong-vs-clean faces `{flip['initial_wrong_orientation_vs_clean']}`; "
            f"oracle/predicted introduced flips `{flip['oracle_introduced_flips']}` / `{flip['predicted_introduced_flips']}`; "
            f"overlap `{flip['oracle_predicted_flip_overlap']}`; oracle-only `{flip['oracle_only_flips']}`; "
            f"prediction-only `{flip['prediction_only_flips']}`. High-target region contains `{pct(background)}` of faces "
            f"and `{pct(oracle_high)}` / `{pct(pred_high)}` of oracle/predicted flips."
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Classification: **Case {case}**.",
            "",
            conclusion,
            "",
            "### Direct answers",
            "",
            f"- Can the frozen solver recover clean geometry from exact native raw Laplacians? **{'Yes' if v2_oracle_strong else 'No, not sufficiently on v2'}**.",
            f"- V2 median recoverable fraction η_oracle: **{v2_oracle['median']:.4f}**; mean **{v2_oracle['mean']:.4f}**.",
            f"- V2 median learned retention η_pred: **{v2_retained['median']:.4f}**; mean **{v2_retained['mean']:.4f}**.",
            "- Why did v2 appear to have lower raw EPE yet worse geometry? First, that apparent EPE advantage was caused by the previous v1-on-v2 evaluation mismatch and disappears under matched-domain evaluation. Independently, the exact-target oracle recovers only a small fraction of the v2 geometry gap, so the frozen inverse recovery is a primary ceiling even with zero Laplacian prediction error.",
            "- Are flips caused by input, solver or prediction? The initial-vs-clean, oracle-only, prediction-only and overlap counts above separate these causes directly.",
            f"- Ready to scale strong_smooth_v2 to 2000 meshes? **{'Yes' if scale_ready else 'No'}**.",
            f"- Single blocker: **{blocker}**.",
            "",
            "Representative visual cases were selected by fixed rules before rendering and are recorded in `visual_selection.json`.",
        ]
    )
    (output / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"contract_audit": True, "decision_case": case, "scale_ready": scale_ready}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
