#!/usr/bin/env python3
from __future__ import annotations

"""Combine matched v1/v2 incremental recovery-ablation summaries."""

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


LABELS = {
    "pure_laplacian_l2": "pure Laplacian / L2",
    "plus_anchor": "+ anchor",
    "plus_visibility": "+ visibility",
    "plus_confidence": "+ confidence",
    "plus_huber": "+ Huber",
    "full_solver": "full solver",
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _fmt(value: float) -> str:
    return f"{value:.9g}"


def _aggregate(summary: Mapping[str, Any], arm: str) -> Mapping[str, Any]:
    return next(row for row in summary["aggregates"] if row["arm"] == arm)


def _increment(summary: Mapping[str, Any], arm: str) -> Mapping[str, Any]:
    return next(row for row in summary["increments"] if row["arm"] == arm)


def _table(summary: Mapping[str, Any]) -> list[str]:
    lines = [
        "| Cumulative arm | Chamfer | eta mean | eta median | Normal | Flips | Improved | Huber active |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in summary["arm_order"]:
        row = _aggregate(summary, arm)
        lines.append(
            f"| {LABELS[arm]} | {_fmt(float(row['chamfer']))} | "
            f"{_fmt(float(row['eta_mean']))} | {_fmt(float(row['eta_median']))} | "
            f"{_fmt(float(row['normal_consistency']))} | {int(row['introduced_flipped_faces'])} | "
            f"{int(row['improved_over_initial'])}/{int(row['samples'])} | "
            f"{int(row['huber_active_samples'])}/{int(row['samples'])} |"
        )
    return lines


def _increment_table(summary: Mapping[str, Any]) -> list[str]:
    lines = [
        "| Added component | Mean delta eta | Mean delta Chamfer (positive=better) | Better / worse / same |",
        "|---|---:|---:|---:|",
    ]
    for arm in summary["arm_order"]:
        row = _increment(summary, arm)
        lines.append(
            f"| {LABELS[arm]} | {_fmt(float(row['mean_eta_delta_from_previous']))} | "
            f"{_fmt(float(row['mean_chamfer_improvement_from_previous']))} | "
            f"{int(row['improved_samples'])} / {int(row['worsened_samples'])} / {int(row['unchanged_samples'])} |"
        )
    return lines


def generate(v1: Mapping[str, Any], v2: Mapping[str, Any]) -> str:
    lines: list[str] = [
        "# Sofa50 incremental exact-target recovery ablation",
        "",
        "Contract audit: **true**.",
        "",
        "This diagnostic holds the matched-domain initial mesh, current graph, exact native raw-Laplacian target, "
        "200 optimizer iterations, learning rate, and unified surface evaluator fixed. It cumulatively adds only "
        "the requested recovery components. No model was retrained and no benchmark result was overwritten.",
        "",
        "Arm order: `pure Laplacian/L2 -> anchor -> visibility -> confidence -> Huber -> full solver`.",
        "",
    ]
    for summary in (v1, v2):
        arm = str(summary["dataset_arm"])
        routing = summary["solver_routing"]
        collapse = summary["collapse"]
        lines.extend(
            [
                f"## {arm}",
                "",
                *_table(summary),
                "",
                *_increment_table(summary),
                "",
                f"Largest incremental efficiency drop: **{LABELS.get(str(collapse['arm']), collapse['arm'])}** "
                f"after **{LABELS.get(str(collapse.get('previous_arm')), collapse.get('previous_arm'))}**, "
                f"mean delta eta `{_fmt(float(collapse['mean_eta_delta']))}`.",
                "",
                f"Solver routing: dense `{int(routing['dense_test_samples'])}` / sparse "
                f"`{int(routing['sparse_test_samples'])}` samples; configured Huber was actually active on "
                f"`{int(routing['huber_actually_active_test_samples'])}/{int(summary['test_samples'])}` samples.",
                "",
                f"Visibility covers `{100.0 * float(summary['visibility']['mean_visible_fraction']):.2f}%` of vertices. "
                f"Learned confidence sample-mean is `{_fmt(float(summary['confidence']['mean_of_sample_means']))}` "
                f"(mean within-sample std `{_fmt(float(summary['confidence']['mean_of_sample_stds']))}`).",
                "",
            ]
        )

    v2_increments = {row["arm"]: row for row in v2["increments"]}
    v2_aggregates = {row["arm"]: row for row in v2["aggregates"]}
    collapse = v2["collapse"]
    lines.extend(
        [
            "## Where recovery efficiency collapses",
            "",
            f"On the target strong-smoothing v2 domain, the largest cumulative loss of mean recovery efficiency "
            f"occurs when **{LABELS.get(str(collapse['arm']), collapse['arm'])}** is added: "
            f"delta eta `{_fmt(float(collapse['mean_eta_delta']))}`. The paired better/worse/same counts are "
            f"`{int(v2_increments[collapse['arm']]['improved_samples'])}/"
            f"{int(v2_increments[collapse['arm']]['worsened_samples'])}/"
            f"{int(v2_increments[collapse['arm']]['unchanged_samples'])}`.",
            "",
            "The Huber and full-solver rows must be interpreted as implementation audit results, not as evidence "
            "that robustification is intrinsically ineffective: the production sparse solver used for meshes above "
            "the 5,000-vertex threshold computes an L2 objective and does not branch on `robust_loss`. The frozen "
            "full configuration also has `lambda_edge=0` and `unseen_anchor_weight=0`, so it adds no active term "
            "after the Huber-labelled row under this routing.",
            "",
            f"The final v2 full solver recovers mean eta `{_fmt(float(v2_aggregates['full_solver']['eta_mean']))}` "
            f"and median eta `{_fmt(float(v2_aggregates['full_solver']['eta_median']))}`; it improves "
            f"`{int(v2_aggregates['full_solver']['improved_over_initial'])}/{int(v2['test_samples'])}` samples.",
            "",
            "## Exact definitions",
            "",
            "- `eta = (CD_initial - CD_arm) / (CD_initial - CD_clean)` per sample.",
            "- Visibility is the hard any-view renderer mask; invisible Laplacian equation rows receive exactly zero weight.",
            "- Confidence multiplies the visibility weight and is not passed a second time through solver confidence.",
            "- L2, anchor, visibility and confidence are cumulative. The Huber row changes only `robust_loss`.",
            "- Full solver replays the frozen recovery config exactly (`lambda_lap=1`, `lambda_anchor=0.01`, "
            "`lambda_edge=0`, `unseen_anchor_weight=0`, 200 iterations, learning rate 0.01, configured Huber delta 0.01).",
            f"- Geometry metric protocol: `{v2['metric_protocol']}`.",
            "",
            "Machine-readable outputs include `summary.json`, `aggregate.csv`, `incremental_effects.csv`, "
            "`per_sample.csv`, and `contract_audit.json` for each dataset arm.",
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
        raise RuntimeError("Input recovery-ablation contract audit failed.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generate(v1, v2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
