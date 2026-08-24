#!/usr/bin/env python3
from __future__ import annotations

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


def _table(summary: Mapping[str, Any]) -> list[str]:
    lines = [
        "| Adam steps | Chamfer | eta mean | eta median | Normal | Flips | Weighted Lap RMS | Improved | Runtime/mesh |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["aggregates"]:
        lines.append(
            f"| {int(row['steps'])} | {_fmt(float(row['chamfer']))} | {_fmt(float(row['eta_mean']))} | "
            f"{_fmt(float(row['eta_median']))} | {_fmt(float(row['normal_consistency']))} | "
            f"{int(row['introduced_flipped_faces'])} | {_fmt(float(row['weighted_laplacian_residual_rms']))} | "
            f"{int(row['improved_over_initial'])}/{int(row['samples'])} | {_fmt(float(row['runtime_seconds_mean']))} s |"
        )
    return lines


def generate(v1: Mapping[str, Any], v2: Mapping[str, Any]) -> str:
    lines = [
        "# Sofa50 frozen recovery Adam-step sweep",
        "",
        "Contract audit: **true**.",
        "",
        "Each arm independently restarts from the same initial mesh and uses the same exact target, current graph, "
        "anchor=0.01, visibility x learned-confidence weights, optimizer, learning rate and unified evaluator. "
        "Only the Adam iteration budget changes: 200, 500, 1000 or 2000.",
        "",
    ]
    for summary in (v1, v2):
        convergence = summary["convergence"]
        lines.extend(
            [
                f"## {summary['dataset_arm']}",
                "",
                *_table(summary),
                "",
                f"Mean eta non-decreasing with steps: **{str(bool(convergence['mean_eta_non_decreasing_with_steps'])).lower()}**; "
                f"per-sample monotonic: `{int(convergence['per_sample_eta_non_decreasing_count'])}/"
                f"{int(convergence['per_sample_total'])}`; mean eta gain 200->2000: "
                f"`{_fmt(float(convergence['mean_eta_gain_200_to_2000']))}`.",
                "",
            ]
        )
    v2_gain = float(v2["convergence"]["mean_eta_gain_200_to_2000"])
    lines.extend(
        [
            "## Decision",
            "",
            f"On v2, extending the frozen recovery from 200 to 2000 steps changes mean eta by `{_fmt(v2_gain)}`. "
            + ("This supports under-convergence at 200 steps." if v2_gain > 0.02 else "This does not explain the main recovery-efficiency loss; the hard visibility intervention remains the primary tested cause."),
            "",
            "All test meshes use the production sparse path. Although the config says Huber, that path currently "
            "executes L2; this experiment deliberately preserves that actual original behavior.",
            "",
            f"Metric protocol: `{v2['metric_protocol']}`.",
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
        raise RuntimeError("Input contract audit failed.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generate(v1, v2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
