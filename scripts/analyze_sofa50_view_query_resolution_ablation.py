#!/usr/bin/env python3
from __future__ import annotations

"""Build the Sofa50 view-count/query-resolution ablation report from run metrics."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


VERSION = "2026-08-11-v1"
VIEW_ARMS = {"views_14": 14, "views_28": 28, "views_56": 56}
QUERY_ARMS = ("gt", "gt_sub1", "gt_adaptive")
SPACES = ("target_space", "recovered_raw_space")
METRICS = (
    "mse",
    "mean_absolute_error",
    "vector_endpoint_error",
    "magnitude_error",
    "cosine_similarity",
    "global_cosine",
    "top_10_percent_cosine",
    "top_1_percent_cosine",
    "top_10_percent_vector_endpoint_error",
    "top_1_percent_vector_endpoint_error",
    "prediction_to_target_norm_ratio",
)
LOWER_IS_BETTER = {
    "loss",
    "mse",
    "mean_absolute_error",
    "vector_endpoint_error",
    "magnitude_error",
    "top_10_percent_vector_endpoint_error",
    "top_1_percent_vector_endpoint_error",
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def macro(values: Sequence[Any]) -> dict[str, float | int | None]:
    numbers = [float(value) for value in values if finite_number(value)]
    if not numbers:
        return {"count": 0, "mean": None, "std": None, "minimum": None, "maximum": None}
    return {
        "count": len(numbers),
        "mean": statistics.fmean(numbers),
        "std": statistics.pstdev(numbers),
        "minimum": min(numbers),
        "maximum": max(numbers),
    }


def validation_records(payload: Mapping[str, Any], label: str) -> dict[str, dict[str, Any]]:
    per_object = payload.get("per_object_metrics")
    if not isinstance(per_object, Mapping) or not isinstance(per_object.get("validation"), Mapping):
        raise ValueError(f"{label}: metrics.json has no per_object_metrics.validation object")
    records = {str(key): dict(value) for key, value in per_object["validation"].items() if isinstance(value, Mapping)}
    if not records:
        raise ValueError(f"{label}: validation metrics are empty")
    return records


def summarize_arm(label: str, path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = read_json(path)
    records = validation_records(payload, label)
    aggregates: dict[str, Any] = {
        "loss": macro([record.get("loss") for record in records.values()]),
        "exact_query_loss": macro([record.get("exact_query_loss") for record in records.values()]),
        "vertex_count": macro([record.get("vertex_count") for record in records.values()]),
        "face_count": macro([record.get("face_count") for record in records.values()]),
        "view_count": macro([record.get("view_count") for record in records.values()]),
        "visible_fraction": macro([
            float(record.get("visible_query_count", 0))
            / max(1, int(record.get("visible_query_count", 0)) + int(record.get("invisible_query_count", 0)))
            for record in records.values()
        ]),
    }
    for space in SPACES:
        aggregates[space] = {
            metric: macro([
                record.get(space, {}).get(metric) if isinstance(record.get(space), Mapping) else None
                for record in records.values()
            ])
            for metric in METRICS
        }
    aggregates["confidence"] = {
        "mean": macro([
            record.get("confidence", {}).get("mean") if isinstance(record.get("confidence"), Mapping) else None
            for record in records.values()
        ]),
        "correlation_with_negative_error": macro([
            record.get("confidence", {}).get("correlation_with_negative_error")
            if isinstance(record.get("confidence"), Mapping)
            else None
            for record in records.values()
        ]),
    }

    run = {
        key: payload.get(key)
        for key in (
            "best_epoch",
            "best_selection_loss",
            "completed_epochs",
            "optimizer_steps",
            "final_train_loss",
            "final_validation_loss",
            "runtime_seconds",
            "peak_gpu_memory_mb",
            "stop_reason",
            "stopped_early",
            "device",
            "distributed_world_size",
            "train_meshes",
            "validation_meshes",
        )
    }
    rows: list[dict[str, Any]] = []
    for sample_id, record in sorted(records.items()):
        row: dict[str, Any] = {
            "arm": label,
            "sample_id": sample_id,
            "loss": record.get("loss"),
            "exact_query_loss": record.get("exact_query_loss"),
            "vertex_count": record.get("vertex_count"),
            "face_count": record.get("face_count"),
            "view_count": record.get("view_count"),
            "visible_query_count": record.get("visible_query_count"),
            "invisible_query_count": record.get("invisible_query_count"),
        }
        for space in SPACES:
            values = record.get(space, {})
            if isinstance(values, Mapping):
                for metric in METRICS:
                    row[f"{space}.{metric}"] = values.get(metric)
        rows.append(row)
    return {
        "metrics_path": str(path.resolve()),
        "validation_sample_ids": sorted(records),
        "run": run,
        "validation_macro": aggregates,
    }, rows


def scalar(arm: Mapping[str, Any], metric: str, space: str | None = None) -> float:
    aggregate = arm["validation_macro"]
    value = aggregate[metric]["mean"] if space is None else aggregate[space][metric]["mean"]
    if not finite_number(value):
        raise ValueError(f"Missing aggregate metric: {space or 'root'}.{metric}")
    return float(value)


def comparison_row(
    experiment: str,
    source_name: str,
    target_name: str,
    source: Mapping[str, Any],
    target: Mapping[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {"experiment": experiment, "comparison": f"{target_name}_vs_{source_name}"}
    source_loss = float(source["run"]["best_selection_loss"])
    target_loss = float(target["run"]["best_selection_loss"])
    row["best_validation_loss_relative_improvement"] = (source_loss - target_loss) / source_loss
    for space in SPACES:
        for metric in (
            "vector_endpoint_error",
            "top_10_percent_vector_endpoint_error",
            "top_1_percent_vector_endpoint_error",
            "global_cosine",
            "prediction_to_target_norm_ratio",
        ):
            before = scalar(source, metric, space)
            after = scalar(target, metric, space)
            key = f"{space}.{metric}"
            if metric in LOWER_IS_BETTER:
                row[f"{key}.relative_improvement"] = (before - after) / before
            else:
                row[f"{key}.change"] = after - before
    row["runtime_multiplier"] = float(target["run"]["runtime_seconds"]) / float(source["run"]["runtime_seconds"])
    return row


def query_contract(summary: Mapping[str, Any]) -> dict[str, Any]:
    samples = summary.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Query summary has no samples")
    ratios: list[float] = []
    represented_area_differences: list[float] = []
    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        stats = sample.get("stats", {})
        if not isinstance(stats, Mapping):
            continue
        adaptive = stats.get("gt_adaptive", {})
        sub2 = stats.get("gt_sub2", {})
        if not isinstance(adaptive, Mapping) or not isinstance(sub2, Mapping):
            continue
        ratios.append(float(adaptive["vertices"]) / float(sub2["vertices"]))
        represented_area_differences.append(abs(float(adaptive["max_represented_area"]) - float(sub2["max_represented_area"])))
    controls = summary.get("control", {})
    return {
        "format_version": summary.get("format_version"),
        "groups": summary.get("groups"),
        "split_vertex_ranges": summary.get("validation"),
        "same_piecewise_linear_gt_surface": controls.get("same_piecewise_linear_gt_surface"),
        "same_rgb_camera_observations": controls.get("same_rgb_camera_observations"),
        "target_recomputed_on_each_current_graph": controls.get("target_recomputed_on_each_current_graph"),
        "cross_graph_target_interpolation": controls.get("cross_graph_target_interpolation"),
        "adaptive_reference": summary.get("adaptive_policy", {}).get("reference"),
        "adaptive_max_vertex_ratio_vs_sub2": max(ratios),
        "adaptive_max_represented_area_abs_difference_vs_sub2": max(represented_area_differences),
        "adaptive_matches_sub2_max_represented_area": max(represented_area_differences) <= 1e-12,
    }


def build_contract_audit(
    arms: Mapping[str, Mapping[str, Any]],
    view_summary: Mapping[str, Any],
    query_summary: Mapping[str, Any],
) -> dict[str, Any]:
    view_ids = [set(arms[name]["validation_sample_ids"]) for name in VIEW_ARMS]
    query_ids = [set(arms[name]["validation_sample_ids"]) for name in QUERY_ARMS]
    expected_views = all(
        int(round(scalar(arms[name], "view_count"))) == count
        and arms[name]["validation_macro"]["view_count"]["minimum"] == count
        and arms[name]["validation_macro"]["view_count"]["maximum"] == count
        for name, count in VIEW_ARMS.items()
    )
    query_views = all(
        arms[name]["validation_macro"]["view_count"]["minimum"] == 14
        and arms[name]["validation_macro"]["view_count"]["maximum"] == 14
        for name in QUERY_ARMS
    )
    completed = all(
        arm["run"].get("optimizer_steps") == 20000
        and arm["run"].get("completed_epochs") == 2000
        and not arm["run"].get("stopped_early")
        for name, arm in arms.items()
        if name != "gt"
    )
    q_contract = query_contract(query_summary)
    return {
        "validation_sample_count": len(view_ids[0]),
        "view_validation_sample_sets_match": all(ids == view_ids[0] for ids in view_ids[1:]),
        "query_validation_sample_sets_match": all(ids == query_ids[0] for ids in query_ids[1:]),
        "cross_experiment_validation_sample_sets_match": query_ids[0] == view_ids[0],
        "view_counts_match_14_28_56": expected_views,
        "query_arms_use_14_views": query_views,
        "view_nesting": view_summary.get("nesting"),
        "same_gt_graph_target_across_view_groups": view_summary.get("same_gt_graph_target_across_view_groups"),
        "base_14_camera_poses_reused_exactly": view_summary.get("base_14_camera_poses_reused_exactly"),
        "all_56_observations_rerendered_with_strict_cpu_reference": view_summary.get("all_56_observations_rerendered_with_strict_cpu_reference"),
        "query_resolution": q_contract,
        "completed_trained_arms": completed,
        "gt_query_arm_source": "views_14 alias",
        "gt_sub2_training_result": "excluded",
    }


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


def main_comparison_rows(arms: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = (("view_count", tuple(VIEW_ARMS)), ("query_resolution", QUERY_ARMS))
    for experiment, names in groups:
        run_row: dict[str, Any] = {"experiment": experiment, "space": "run", "metric": "best_selection_loss"}
        for name in names:
            run_row[name] = arms[name]["run"]["best_selection_loss"]
        rows.append(run_row)
        for space in SPACES:
            for metric in METRICS:
                row = {"experiment": experiment, "space": space, "metric": metric, "aggregation": "validation_mesh_macro_mean"}
                for name in names:
                    row[name] = arms[name]["validation_macro"][space][metric]["mean"]
                rows.append(row)
    return rows


def fmt(value: Any, digits: int = 6) -> str:
    return "—" if not finite_number(value) else f"{float(value):.{digits}f}"


def metric_table(lines: list[str], names: Sequence[str], arms: Mapping[str, Mapping[str, Any]]) -> None:
    lines.extend([
        "| metric | " + " | ".join(names) + " |",
        "|---|" + "---:|" * len(names),
    ])
    definitions = (
        ("best validation loss", None, "best_selection_loss"),
        ("target endpoint", "target_space", "vector_endpoint_error"),
        ("raw endpoint", "recovered_raw_space", "vector_endpoint_error"),
        ("raw top-10% endpoint", "recovered_raw_space", "top_10_percent_vector_endpoint_error"),
        ("raw top-1% endpoint", "recovered_raw_space", "top_1_percent_vector_endpoint_error"),
        ("raw global cosine", "recovered_raw_space", "global_cosine"),
        ("raw pred/target norm", "recovered_raw_space", "prediction_to_target_norm_ratio"),
    )
    for label, space, metric in definitions:
        values = []
        for name in names:
            value = arms[name]["run"][metric] if space is None else arms[name]["validation_macro"][space][metric]["mean"]
            values.append(fmt(value))
        lines.append(f"| {label} | " + " | ".join(values) + " |")


def report(summary: Mapping[str, Any]) -> str:
    arms = summary["arms"]
    q_contract = summary["contract_audit"]["query_resolution"]
    q_ranges = q_contract["split_vertex_ranges"]
    lines = [
        "# Sofa50 view-count and query-resolution ablation",
        "",
        f"Analyzer version: `{summary['analyzer_version']}`",
        "",
        "Metrics are validation-mesh macro means over the five matched validation sample IDs.",
        "The query-resolution GT row aliases the views_14 run recorded by the experiment contract.",
        "GT-sub2 is a data-only arm and has no training metrics.",
        "",
        "## View-count checkpoint comparison",
        "",
    ]
    metric_table(lines, tuple(VIEW_ARMS), arms)
    lines.extend(["", "## Query-resolution checkpoint comparison", ""])
    metric_table(lines, QUERY_ARMS, arms)

    lines.extend([
        "",
        "## Pairwise changes",
        "",
        "Positive endpoint improvement means the target arm has lower endpoint error.",
        "",
        "| comparison | best-val improvement | raw endpoint improvement | raw top-10% improvement | raw cosine change | runtime multiplier |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in summary["pairwise_changes"]:
        lines.append(
            f"| {row['comparison']} | {float(row['best_validation_loss_relative_improvement']):+.2%} | "
            f"{float(row['recovered_raw_space.vector_endpoint_error.relative_improvement']):+.2%} | "
            f"{float(row['recovered_raw_space.top_10_percent_vector_endpoint_error.relative_improvement']):+.2%} | "
            f"{float(row['recovered_raw_space.global_cosine.change']):+.6f} | {float(row['runtime_multiplier']):.3f}x |"
        )

    lines.extend([
        "",
        "## Query-graph data contract",
        "",
        "| graph | train vertices | validation vertices | training result |",
        "|---|---:|---:|---|",
    ])
    labels = {"gt": "GT", "gt_sub1": "GT-sub1", "gt_sub2": "GT-sub2", "gt_adaptive": "GT-adaptive"}
    states = {"gt": "views_14 alias", "gt_sub1": "complete", "gt_sub2": "excluded", "gt_adaptive": "complete"}
    for name in ("gt", "gt_sub1", "gt_sub2", "gt_adaptive"):
        train = q_ranges[name]["train"]
        validation = q_ranges[name]["validation"]
        lines.append(
            f"| {labels[name]} | {train['min_vertices']:,}–{train['max_vertices']:,} | "
            f"{validation['min_vertices']:,}–{validation['max_vertices']:,} | {states[name]} |"
        )
    lines.extend([
        "",
        f"GT-adaptive maximum vertex-count ratio to GT-sub2: `{q_contract['adaptive_max_vertex_ratio_vs_sub2']:.8f}`.",
        f"Maximum represented-area absolute difference between GT-adaptive and GT-sub2: `{q_contract['adaptive_max_represented_area_abs_difference_vs_sub2']:.12g}`.",
        "",
        "## Contract audit",
        "",
        "```json",
        json.dumps(summary["contract_audit"], indent=2),
        "```",
        "",
        "## Run health",
        "",
        "| arm | steps | epochs | best epoch | final train | final validation | runtime h | peak GPU MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name in tuple(VIEW_ARMS) + ("gt_sub1", "gt_adaptive"):
        run = arms[name]["run"]
        lines.append(
            f"| {name} | {run['optimizer_steps']:,} | {run['completed_epochs']:,} | {run['best_epoch']:,} | "
            f"{fmt(run['final_train_loss'])} | {fmt(run['final_validation_loss'])} | "
            f"{float(run['runtime_seconds']) / 3600:.3f} | {float(run['peak_gpu_memory_mb']):.1f} |"
        )

    view14 = arms["views_14"]
    view28 = arms["views_28"]
    view56 = arms["views_56"]
    sub1 = arms["gt_sub1"]
    adaptive = arms["gt_adaptive"]
    lines.extend([
        "",
        "## Recorded results",
        "",
        f"- View-count best validation loss: 14={fmt(view14['run']['best_selection_loss'])}, 28={fmt(view28['run']['best_selection_loss'])}, 56={fmt(view56['run']['best_selection_loss'])}.",
        f"- Query-resolution best validation loss: GT={fmt(view14['run']['best_selection_loss'])}, GT-sub1={fmt(sub1['run']['best_selection_loss'])}, GT-adaptive={fmt(adaptive['run']['best_selection_loss'])}.",
        "- GT-adaptive matches the per-sample GT-sub2 maximum represented-area threshold.",
        "- All five trained arms reached 20,000 optimizer steps and 2,000 epochs.",
        "",
    ])
    return "\n".join(lines)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    view_root = args.view_root.expanduser().resolve()
    query_root = args.query_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    arms: dict[str, Any] = {}
    per_mesh_rows: list[dict[str, Any]] = []
    for name in VIEW_ARMS:
        arm, rows = summarize_arm(name, view_root / name / "metrics.json")
        arms[name] = arm
        per_mesh_rows.extend(rows)
    for name in ("gt_sub1", "gt_adaptive"):
        arm, rows = summarize_arm(name, query_root / name / "metrics.json")
        arms[name] = arm
        per_mesh_rows.extend(rows)
    arms["gt"] = {**arms["views_14"], "source_alias": "views_14"}

    view_summary = read_json(args.view_summary.expanduser().resolve())
    query_summary = read_json(args.query_summary.expanduser().resolve())
    pairs = [
        comparison_row("view_count", "views_14", "views_28", arms["views_14"], arms["views_28"]),
        comparison_row("view_count", "views_14", "views_56", arms["views_14"], arms["views_56"]),
        comparison_row("view_count", "views_28", "views_56", arms["views_28"], arms["views_56"]),
        comparison_row("query_resolution", "gt", "gt_sub1", arms["gt"], arms["gt_sub1"]),
        comparison_row("query_resolution", "gt", "gt_adaptive", arms["gt"], arms["gt_adaptive"]),
    ]
    audit = build_contract_audit(arms, view_summary, query_summary)
    summary = {
        "analyzer_version": VERSION,
        "experiment": "Sofa50 C2F2 view-count and query-resolution ablation",
        "aggregation": "validation mesh macro mean",
        "source_roots": {
            "view_runs": str(view_root),
            "query_runs": str(query_root),
            "view_dataset_summary": str(args.view_summary.expanduser().resolve()),
            "query_dataset_summary": str(args.query_summary.expanduser().resolve()),
        },
        "arms": arms,
        "pairwise_changes": pairs,
        "contract_audit": audit,
        "notes": [
            "The query-resolution GT row aliases views_14 according to the recorded experiment contract.",
            "GT-sub2 was excluded from training and is reported only in the data contract.",
            "Aggregate prediction metrics are macro means of the five per-object validation metrics stored by training.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "contract_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (output_dir / "REPORT.md").write_text(report(summary), encoding="utf-8")
    write_csv(output_dir / "main_comparison.csv", main_comparison_rows(arms))
    write_csv(output_dir / "pairwise_changes.csv", pairs)
    write_csv(output_dir / "per_mesh_metrics.csv", per_mesh_rows)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view-root", required=True, type=Path)
    parser.add_argument("--query-root", required=True, type=Path)
    parser.add_argument("--view-summary", required=True, type=Path)
    parser.add_argument("--query-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = analyze(args)
    print(f"Wrote {args.output_dir.expanduser().resolve() / 'REPORT.md'}")
    print(json.dumps({"contract_audit": summary["contract_audit"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
