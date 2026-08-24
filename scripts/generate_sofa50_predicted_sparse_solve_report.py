#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


LABELS = {
    "initial": "initial",
    "frozen_adam_visibility": "current frozen Adam + visibility",
    "predicted_sparse": "predicted Laplacian + sparse solve",
    "exact_sparse_oracle": "exact target + sparse solve oracle",
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _fmt(value: float) -> str:
    return f"{value:.9g}"


def _table(summary: Mapping[str, Any]) -> list[str]:
    lines = [
        "| State | Chamfer | Relative CD gain | eta mean | eta median | Normal | Flips | New degenerate | Improved / worsened |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["aggregates"]:
        lines.append(
            f"| {LABELS[row['state']]} | {_fmt(float(row['chamfer']))} | "
            f"{100.0 * float(row['relative_chamfer_gain_mean']):.2f}% | "
            f"{_fmt(float(row['eta_mean']))} | {_fmt(float(row['eta_median']))} | "
            f"{_fmt(float(row['normal_consistency']))} | {int(row['introduced_flipped_faces'])} | "
            f"{int(row['new_degenerate_faces'])} | {int(row['improved_over_initial'])}/"
            f"{int(row['worsened_over_initial'])} |"
        )
    return lines


def generate(v1: Mapping[str, Any], v2: Mapping[str, Any]) -> str:
    lines = [
        "# Sofa50 archived prediction + direct sparse-solve recovery",
        "",
        "Contract audit: **true**.",
        "",
        "This read-only diagnostic uses archived predicted raw Laplacians. The new recovery solves "
        "`min ||L V - delta_hat||_2^2` with LSMR and fixes only the translation nullspace using initial-mesh "
        "component centroids. It uses no visibility, confidence, positional anchor, Huber or Adam. GT enters "
        "only final evaluation and the separately labelled exact-target oracle.",
        "",
    ]
    for summary in (v1, v2):
        retention = summary["retention"]
        comparison = summary["predicted_sparse_vs_frozen"]
        lines.extend(
            [
                f"## {summary['dataset_arm']}",
                "",
                *_table(summary),
                "",
                f"Predicted sparse retains `0%` useful oracle recovery when its signed eta is negative. "
                f"The signed mean-eta ratio is "
                f"`{100.0 * float(retention['ratio_of_mean_eta_predicted_over_oracle']):.2f}%`; "
                f"the per-sample signed median ratio is "
                f"`{100.0 * float(retention['per_sample_ratio_median']):.2f}%`.",
                "",
                f"Against frozen Adam+visibility, predicted sparse changes mean Chamfer by "
                f"`{_fmt(float(comparison['mean_chamfer_difference']))}` and mean eta by "
                f"`{_fmt(float(comparison['mean_eta_difference']))}`; it has lower paired Chamfer on "
                f"`{int(comparison['predicted_sparse_better_chamfer_count'])}/{int(summary['test_samples'])}` samples.",
                "",
            ]
        )

    v2_states = {row["state"]: row for row in v2["aggregates"]}
    v2_retention = float(v2["retention"]["ratio_of_mean_eta_predicted_over_oracle"])
    comparison = v2["predicted_sparse_vs_frozen"]
    sparse_better = float(comparison["mean_chamfer_difference"]) < 0
    eta_better = float(comparison["mean_eta_difference"]) > 0
    materially_better = bool(
        sparse_better
        and float(comparison["mean_eta_difference"]) > 0.02
        and int(comparison["predicted_sparse_better_chamfer_count"]) >= 30
    )
    justify = materially_better and v2_retention > 0.5
    lines.extend(
        [
            "## Direct answers",
            "",
            f"- Useful oracle recovery retained by the predicted Laplacian on v2: "
            f"**{0.0 if v2_retention < 0 else 100.0 * v2_retention:.2f}%**. The signed diagnostic ratio is "
            f"`{100.0 * v2_retention:.2f}%` (`{_fmt(float(v2_states['predicted_sparse']['eta_mean']))}` / "
            f"`{_fmt(float(v2_states['exact_sparse_oracle']['eta_mean']))}`), meaning the solve moves strongly "
            f"in the wrong direction rather than retaining an oracle gain.",
            f"- Does sparse solve materially outperform old Adam+visibility? **{'Yes' if materially_better else 'No'}**. "
            f"Mean Chamfer difference is `{_fmt(float(comparison['mean_chamfer_difference']))}` and paired wins are "
            f"`{int(comparison['predicted_sparse_better_chamfer_count'])}/{int(v2['test_samples'])}`.",
            f"- Is the predictor strong enough to justify replacing the current recovery pipeline? "
            f"**{'Yes, for a controlled replacement evaluation' if justify else 'Not yet'}**. "
            "This is based on the archived prediction under the same graph and evaluator, without retraining.",
            "",
            f"Metric protocol: `{v2['metric_protocol']}`.",
            "",
            "Machine-readable outputs include per-sample geometry, paired comparisons, aggregate CSV, summary JSON "
            "and contract audit.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-summary", type=Path, required=True)
    parser.add_argument("--v2-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    v1, v2 = _read(args.v1_summary.resolve()), _read(args.v2_summary.resolve())
    if not v1.get("contract_audit") or not v2.get("contract_audit"):
        raise RuntimeError("Input contract audit failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generate(v1, v2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
