#!/usr/bin/env python3
from __future__ import annotations

"""Compare the views_28 + GT-adaptive combination with both parent arms."""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from analyze_sofa50_view_query_resolution_ablation import (
    METRICS,
    SPACES,
    comparison_row,
    scalar,
    summarize_arm,
)


VERSION = "2026-08-11-v1"
ARMS = ("views_28", "gt_adaptive", "views_28_gt_adaptive")


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def fmt(value: Any, digits: int = 6) -> str:
    return "—" if not finite(value) else f"{float(value):.{digits}f}"


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def report(summary: Mapping[str, Any]) -> str:
    arms = summary["arms"]
    decision = summary["decision_checks"]
    lines = [
        "# Sofa50 views_28 + GT-adaptive combination arm",
        "",
        f"Analyzer version: `{summary['analyzer_version']}`",
        "",
        "Metrics are validation-mesh macro means over the five matched validation sample IDs.",
        "",
        "## Checkpoint comparison",
        "",
        "| metric | views_28 | gt_adaptive | views_28_gt_adaptive |",
        "|---|---:|---:|---:|",
    ]
    definitions = (
        ("best validation loss", None, "best_selection_loss"),
        ("raw endpoint", "recovered_raw_space", "vector_endpoint_error"),
        ("raw top-10% endpoint", "recovered_raw_space", "top_10_percent_vector_endpoint_error"),
        ("raw top-1% endpoint", "recovered_raw_space", "top_1_percent_vector_endpoint_error"),
        ("raw global cosine", "recovered_raw_space", "global_cosine"),
        ("runtime hours", None, "runtime_hours"),
    )
    for label, space, metric in definitions:
        values: list[str] = []
        for name in ARMS:
            if metric == "runtime_hours":
                value = float(arms[name]["run"]["runtime_seconds"]) / 3600.0
            elif space is None:
                value = arms[name]["run"][metric]
            else:
                value = arms[name]["validation_macro"][space][metric]["mean"]
            values.append(fmt(value))
        lines.append(f"| {label} | " + " | ".join(values) + " |")

    lines.extend([
        "",
        "## Pairwise changes",
        "",
        "Positive endpoint improvement means the combination arm has lower endpoint error.",
        "",
        "| comparison | best-val improvement | raw endpoint improvement | raw top-10% improvement | raw top-1% improvement | cosine change | runtime multiplier |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary["pairwise_changes"]:
        lines.append(
            f"| {row['comparison']} | {float(row['best_validation_loss_relative_improvement']):+.2%} | "
            f"{float(row['recovered_raw_space.vector_endpoint_error.relative_improvement']):+.2%} | "
            f"{float(row['recovered_raw_space.top_10_percent_vector_endpoint_error.relative_improvement']):+.2%} | "
            f"{float(row['recovered_raw_space.top_1_percent_vector_endpoint_error.relative_improvement']):+.2%} | "
            f"{float(row['recovered_raw_space.global_cosine.change']):+.6f} | {float(row['runtime_multiplier']):.3f}x |"
        )
    lines.extend([
        "",
        "## Contract audit",
        "",
        "```json",
        json.dumps(summary["contract_audit"], indent=2),
        "```",
        "",
        "## Decision checks",
        "",
        f"- Raw top-10% endpoint no higher than GT-adaptive: `{str(decision['adaptive_top10_retained']).lower()}`.",
        f"- Raw top-1% endpoint no higher than GT-adaptive: `{str(decision['adaptive_top1_retained']).lower()}`.",
        f"- Best validation loss no higher than GT-adaptive: `{str(decision['validation_gain_vs_adaptive']).lower()}`.",
        f"- Raw global cosine no lower than GT-adaptive: `{str(decision['cosine_gain_vs_adaptive']).lower()}`.",
        f"- All four conditions: `{str(decision['all_conditions']).lower()}`.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views28-metrics", required=True, type=Path)
    parser.add_argument("--adaptive-metrics", required=True, type=Path)
    parser.add_argument("--combo-metrics", required=True, type=Path)
    parser.add_argument("--combo-data-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    inputs = {
        "views_28": args.views28_metrics,
        "gt_adaptive": args.adaptive_metrics,
        "views_28_gt_adaptive": args.combo_metrics,
    }
    arms: dict[str, Any] = {}
    per_mesh: list[dict[str, Any]] = []
    for name, path in inputs.items():
        result, rows = summarize_arm(name, path.expanduser().resolve())
        arms[name] = result
        per_mesh.extend(rows)
    sample_sets = [set(arms[name]["validation_sample_ids"]) for name in ARMS]
    combo_data = json.loads(args.combo_data_summary.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(combo_data, dict):
        raise ValueError("Combination data summary must be a JSON object")
    adaptive_vertices = {
        row["sample_id"]: row["vertex_count"]
        for row in per_mesh
        if row["arm"] == "gt_adaptive"
    }
    combo_vertices = {
        row["sample_id"]: row["vertex_count"]
        for row in per_mesh
        if row["arm"] == "views_28_gt_adaptive"
    }
    audit = {
        "validation_sample_count": len(sample_sets[0]),
        "validation_sample_sets_match": all(value == sample_sets[0] for value in sample_sets[1:]),
        "combo_view_count_is_28": (
            arms["views_28_gt_adaptive"]["validation_macro"]["view_count"]["minimum"] == 28
            and arms["views_28_gt_adaptive"]["validation_macro"]["view_count"]["maximum"] == 28
        ),
        "combo_validation_vertex_counts_match_gt_adaptive": combo_vertices == adaptive_vertices,
        "combo_completed_20000_steps": arms["views_28_gt_adaptive"]["run"]["optimizer_steps"] == 20000,
        "data_controls": combo_data.get("controls"),
    }
    combo = arms["views_28_gt_adaptive"]
    adaptive = arms["gt_adaptive"]
    decision = {
        "adaptive_top10_retained": scalar(combo, "top_10_percent_vector_endpoint_error", "recovered_raw_space")
        <= scalar(adaptive, "top_10_percent_vector_endpoint_error", "recovered_raw_space"),
        "adaptive_top1_retained": scalar(combo, "top_1_percent_vector_endpoint_error", "recovered_raw_space")
        <= scalar(adaptive, "top_1_percent_vector_endpoint_error", "recovered_raw_space"),
        "validation_gain_vs_adaptive": float(combo["run"]["best_selection_loss"])
        <= float(adaptive["run"]["best_selection_loss"]),
        "cosine_gain_vs_adaptive": scalar(combo, "global_cosine", "recovered_raw_space")
        >= scalar(adaptive, "global_cosine", "recovered_raw_space"),
    }
    decision["all_conditions"] = all(decision.values())
    pairs = [
        comparison_row("combination", "views_28", "views_28_gt_adaptive", arms["views_28"], combo),
        comparison_row("combination", "gt_adaptive", "views_28_gt_adaptive", adaptive, combo),
    ]
    summary = {
        "analyzer_version": VERSION,
        "experiment": "Sofa50 views_28 + GT-adaptive combination arm",
        "aggregation": "validation mesh macro mean",
        "arms": arms,
        "pairwise_changes": pairs,
        "contract_audit": audit,
        "decision_checks": decision,
    }
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "contract_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (output / "REPORT.md").write_text(report(summary), encoding="utf-8")
    main_rows: list[dict[str, Any]] = []
    for space in SPACES:
        for metric in METRICS:
            row = {"space": space, "metric": metric, "aggregation": "validation_mesh_macro_mean"}
            for name in ARMS:
                row[name] = arms[name]["validation_macro"][space][metric]["mean"]
            main_rows.append(row)
    write_csv(output / "main_comparison.csv", main_rows)
    write_csv(output / "pairwise_changes.csv", pairs)
    write_csv(output / "per_mesh_metrics.csv", per_mesh)
    print(f"Wrote {output / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
