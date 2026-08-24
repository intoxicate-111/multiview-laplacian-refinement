#!/usr/bin/env python3
from __future__ import annotations

"""Generate the combined exact-solve and visibility-weight-sweep report."""

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _fmt(value: float) -> str:
    return f"{value:.9g}"


def _exact_table(summary: Mapping[str, Any]) -> list[str]:
    labels = {
        "stored_target_clean_gauge": "stored float32 target + clean gauge",
        "float64_target_clean_gauge": "recomputed float64 target + clean gauge",
        "float64_target_initial_gauge": "recomputed float64 target + initial gauge",
    }
    lines = [
        "| Exact sparse solve | Vertex RMS to clean | Vertex max | Equation RMS | Chamfer | eta mean | eta median |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["exact_solve"]:
        lines.append(
            f"| {labels[row['state']]} | {_fmt(float(row['vertex_rms_to_clean_mean']))} | "
            f"{_fmt(float(row['vertex_max_to_clean_max']))} | "
            f"{_fmt(float(row['equation_residual_rms_mean']))} | {_fmt(float(row['chamfer']))} | "
            f"{_fmt(float(row['eta_mean']))} | {_fmt(float(row['eta_median']))} |"
        )
    return lines


def _sweep_table(summary: Mapping[str, Any]) -> list[str]:
    lines = [
        "| Invisible weight alpha | Chamfer | eta mean | eta median | Normal | Flips | Improved |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["visibility_sweep"]:
        lines.append(
            f"| {_fmt(float(row['alpha']))} | {_fmt(float(row['chamfer']))} | "
            f"{_fmt(float(row['eta_mean']))} | {_fmt(float(row['eta_median']))} | "
            f"{_fmt(float(row['normal_consistency']))} | {int(row['introduced_flipped_faces'])} | "
            f"{int(row['improved_over_initial'])}/{int(row['samples'])} |"
        )
    return lines


def generate(v1: Mapping[str, Any], v2: Mapping[str, Any]) -> str:
    lines = [
        "# Sofa50 exact sparse-solve sanity check and visibility-weight sweep",
        "",
        "Contract audit: **true**.",
        "",
        "No model was retrained. Both diagnostics use the matched-domain test meshes, exact native raw "
        "Laplacian target, current connectivity and the frozen unified surface evaluator.",
        "",
        "The exact solve uses LSMR on the sparse uniform-Laplacian system. Component-centroid constraints "
        "only fix the per-connected-component translation nullspace; they do not add shape information.",
        "",
    ]
    for summary in (v1, v2):
        monotonic = summary["monotonicity"]
        lines.extend(
            [
                f"## {summary['dataset_arm']}",
                "",
                "### Exact sparse solve",
                "",
                *_exact_table(summary),
                "",
                "### Visibility sweep",
                "",
                *_sweep_table(summary),
                "",
                f"Mean eta monotonic non-decreasing: **{str(bool(monotonic['mean_eta_non_decreasing_with_alpha'])).lower()}**; "
                f"median: **{str(bool(monotonic['median_eta_non_decreasing_with_alpha'])).lower()}**; "
                f"per-sample monotonic: `{int(monotonic['per_sample_eta_non_decreasing_count'])}/"
                f"{int(monotonic['per_sample_eta_non_decreasing_total'])}`.",
                "",
                f"Restoring invisible equations from alpha=0 to alpha=1 changes mean eta by "
                f"`{_fmt(float(monotonic['mean_eta_gain_alpha0_to_1']))}`.",
                "",
            ]
        )

    v2_exact = {row["state"]: row for row in v2["exact_solve"]}
    v2_monotonic = v2["monotonicity"]
    mathematical = float(v2_exact["float64_target_clean_gauge"]["vertex_rms_to_clean_mean"])
    hard_bad = bool(v2_monotonic["mean_eta_non_decreasing_with_alpha"])
    lines.extend(
        [
            "## Answers",
            "",
            "### 1. Can exact delta reconstruct the clean vertices mathematically?",
            "",
            f"**{'Yes' if mathematical < 1e-6 else 'Not to numerical precision'}.** With the translation nullspace fixed "
            f"by clean component centroids, the v2 float64 exact sparse solve has mean vertex RMS "
            f"`{_fmt(mathematical)}` to clean. The stored float32 target result separately measures target "
            "quantization, while the initial-gauge row shows the result using only an inference-available translation gauge.",
            "",
            "This separates algebraic invertibility from the 200-step Adam recovery used by the production pipeline.",
            "",
            "### 2. Is hard visibility gating the recovery-design failure?",
            "",
            f"**{'Supported' if hard_bad else 'Not established as monotonic'}.** On v2, mean eta changes from "
            f"`{_fmt(float(v2_monotonic['mean_eta_alpha0']))}` at alpha=0 to "
            f"`{_fmt(float(v2_monotonic['mean_eta_alpha1']))}` at alpha=1, and aggregate mean eta is "
            f"{'monotonic' if hard_bad else 'not monotonic'} over the requested sweep.",
            "",
            "Because the target is exact at every vertex in this diagnostic, hard alpha=0 discards valid equations. "
            "The sweep is therefore a direct recovery-stage causal intervention, not a correlation analysis.",
            "",
            "## Fixed contracts",
            "",
            "- Sweep: exact stored target, `lambda_anchor=0.01`, L2, 200 iterations, learning rate 0.01.",
            "- Visible equations always have weight 1; invisible equations use only the listed alpha.",
            "- Confidence, Huber, edge loss and unseen-anchor loss are disabled in the sweep.",
            "- Alpha=0 and alpha=1 are replay-checked against the previous `+visibility` and `+anchor` arms.",
            f"- Metric protocol: `{v2['metric_protocol']}`.",
            "",
            "Machine-readable outputs include exact/sweep per-sample CSVs, aggregate CSVs, summaries and contract audits.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-summary", type=Path, required=True)
    parser.add_argument("--v2-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    v1 = _read(args.v1_summary.resolve())
    v2 = _read(args.v2_summary.resolve())
    if not v1.get("contract_audit") or not v2.get("contract_audit"):
        raise RuntimeError("Input contract audit failed.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generate(v1, v2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
