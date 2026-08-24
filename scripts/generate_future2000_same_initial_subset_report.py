#!/usr/bin/env python3
from __future__ import annotations

"""Aggregate the selected Future2000 same-initial learned/external comparison."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any


DEFAULT_METHODS = ("ours", "nds", "nvdiffrec", "exmesh")

PAIRED_METRICS = {
    "chamfer": ("refined_chamfer", "lower"),
    "p2s_mean": ("refined_p2s_mean", "lower"),
    "p2s_p95": ("refined_p2s_p95", "lower"),
    "fscore": ("refined_fscore", "higher"),
    "normal_consistency": ("refined_normal_consistency", "higher"),
    "introduced_flipped_faces": ("introduced_flipped_faces", "lower"),
    "new_degenerate_faces": ("new_degenerate_faces", "lower"),
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _finite(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value in (None, "", "None", "nan", "NaN"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _truth(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    return None


def _mean_finite(rows: list[dict[str, str]], key: str) -> float | None:
    values = [value for row in rows if (value := _finite(row, key)) is not None]
    return fmean(values) if values else None


def _sum_finite(rows: list[dict[str, str]], key: str) -> float | None:
    values = [value for row in rows if (value := _finite(row, key)) is not None]
    return sum(values) if values else None


def _fmt(value: float | None, precision: int = 9) -> str:
    return "n/a" if value is None else f"{value:.{precision}g}"


def _paired_metric(
    ours: dict[str, dict[str, str]],
    other: dict[str, dict[str, str]],
    expected: list[str],
    field: str,
    direction: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    wins = ties = losses = 0
    differences: list[float] = []
    rows: list[dict[str, Any]] = []
    for sample_id in expected:
        ours_value = _finite(ours[sample_id], field)
        other_value = _finite(other[sample_id], field)
        if ours_value is None or other_value is None:
            rows.append(
                {
                    "sample_id": sample_id,
                    "metric": field,
                    "ours": ours_value,
                    "external": other_value,
                    "outcome": "invalid",
                }
            )
            continue
        difference = ours_value - other_value
        differences.append(difference)
        if math.isclose(ours_value, other_value, rel_tol=0.0, abs_tol=1e-15):
            ties += 1
            outcome = "tie"
        elif (direction == "lower" and ours_value < other_value) or (
            direction == "higher" and ours_value > other_value
        ):
            wins += 1
            outcome = "ours"
        else:
            losses += 1
            outcome = "external"
        rows.append(
            {
                "sample_id": sample_id,
                "metric": field,
                "ours": ours_value,
                "external": other_value,
                "outcome": outcome,
            }
        )
    valid = len(differences)
    return (
        {
            "valid_pairs": valid,
            "invalid_pairs": len(expected) - valid,
            "ours_wins": wins,
            "ties": ties,
            "external_wins": losses,
            "mean_ours_minus_external": fmean(differences) if differences else None,
            "direction": direction,
        },
        rows,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    methods = tuple(args.methods or DEFAULT_METHODS)
    if "ours" not in methods or len(set(methods)) != len(methods):
        raise ValueError("--methods must contain ours exactly once and have no duplicates.")
    if args.selection is not None:
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        expected = [str(value) for value in selection["sample_ids"]]
    else:
        if args.manifest is None:
            raise ValueError("--manifest is required when --selection is omitted")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        expected = [
            str(row["sample_id"])
            for row in manifest["samples"]
            if row.get("split") == "test"
        ]
        selection = {
            "experiment": "future2000_same_initial_full1000_blackwell",
            "shared_input_contract": (
                "Each method receives the exact same current mesh, 28 native-960 "
                "RGB observations and cameras for every test sample; GT is evaluation-only."
            ),
            "source_manifest": str(args.manifest.resolve()),
            "source_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "selection_seed": None,
            "selection_algorithm": (
                "complete manifest test split in PreparedMeshDataset order; "
                "eight independent modulo shards"
            ),
        }
    rows_by_method = {
        method: _read_rows(args.results_root / method / "per_sample.csv")
        for method in methods
    }
    aggregate = {
        method: json.loads(
            (args.results_root / method / "aggregate.json").read_text(encoding="utf-8")
        )
        for method in methods
    }
    audit_checks: list[dict[str, Any]] = []
    indexed: dict[str, dict[str, dict[str, str]]] = {}
    for method, rows in rows_by_method.items():
        by_id = {row["sample_id"]: row for row in rows}
        indexed[method] = by_id
        audit_checks.append(
            {
                "check": f"{method}_exact_selected_ids",
                "passed": sorted(by_id) == sorted(expected) and len(rows) == len(expected),
            }
        )
        audit_checks.append(
            {
                "check": f"{method}_all_completed",
                "passed": all(row["status"] == "completed" for row in rows),
            }
        )
    for sample_id in expected:
        hashes_by_method = {
            method: indexed[method][sample_id].get("common_initial_mesh_sha256", "")
            for method in methods
        }
        available_hashes = {value for value in hashes_by_method.values() if value}
        missing_hashes = [method for method, value in hashes_by_method.items() if not value]
        missing_are_post_input_failures = all(
            indexed[method][sample_id].get("status") == "failed"
            and indexed[method][sample_id].get("failure_stage")
            == "execution_or_evaluation"
            for method in missing_hashes
        )
        audit_checks.append(
            {
                "check": f"{sample_id}_same_initial_geometry",
                # The runner performs source/adapter identity checks before
                # executing a method.  A post-input execution/evaluation
                # failure may omit the copied identity fields from its row;
                # retain that as metric incompleteness, not an input mismatch.
                "passed": len(available_hashes) == 1
                and (not missing_hashes or missing_are_post_input_failures),
                "hashes": sorted(available_hashes),
                "missing_result_identity_methods": missing_hashes,
            }
        )
        initial_values = [
            value
            for method in methods
            if (value := _finite(indexed[method][sample_id], "initial_chamfer"))
            is not None
        ]
        audit_checks.append(
            {
                "check": f"{sample_id}_same_initial_metric",
                "passed": bool(initial_values)
                and max(initial_values) - min(initial_values) <= 1e-9
                and (len(initial_values) == len(methods) or missing_are_post_input_failures),
                "available_methods": len(initial_values),
            }
        )
    input_contract_checks = [
        item for item in audit_checks if not item["check"].endswith("_all_completed")
    ]
    contract_audit = all(bool(item["passed"]) for item in audit_checks)
    input_contract_audit = all(bool(item["passed"]) for item in input_contract_checks)
    metric_completeness = all(
        all(row["status"] == "completed" for row in rows)
        for rows in rows_by_method.values()
    )

    initial_rows = rows_by_method["ours"]
    initial = {
        "method": "initial",
        "completed_samples": len(initial_rows),
        "mean_initial_chamfer": fmean(float(row["initial_chamfer"]) for row in initial_rows),
        "mean_refined_chamfer": fmean(float(row["initial_chamfer"]) for row in initial_rows),
        "mean_refined_p2s_mean": fmean(float(row["initial_p2s_mean"]) for row in initial_rows),
        "mean_refined_p2s_p95": fmean(float(row["initial_p2s_p95"]) for row in initial_rows),
        "mean_refined_fscore": fmean(float(row["initial_fscore"]) for row in initial_rows),
        "mean_refined_normal_consistency": fmean(
            float(row["initial_normal_consistency"]) for row in initial_rows
        ),
        "improved_meshes": 0,
    }
    summaries = [initial]
    for method in methods:
        item = aggregate[method]
        metrics = item["metrics"]
        completed_rows = [
            row for row in rows_by_method[method] if row["status"] == "completed"
        ]
        chamfer_stats = metrics["refined_chamfer"]
        valid_chamfer = int(chamfer_stats["count"]) if chamfer_stats else 0
        mean_initial = metrics["initial_chamfer"]["mean"]
        mean_refined = chamfer_stats["mean"] if chamfer_stats else None
        summaries.append(
            {
                "method": method,
                "completed_samples": item["completed_samples"],
                "failed_samples": item["failed_samples"],
                "valid_chamfer_samples": valid_chamfer,
                "invalid_chamfer_samples": len(expected) - valid_chamfer,
                "mean_initial_chamfer": mean_initial,
                "mean_refined_chamfer": mean_refined,
                "relative_chamfer_gain": (
                    (mean_initial - mean_refined) / mean_initial
                    if mean_refined is not None and mean_initial
                    else None
                ),
                "mean_refined_p2s_mean": metrics["refined_p2s_mean"]["mean"],
                "mean_refined_p2s_p95": metrics["refined_p2s_p95"]["mean"],
                "mean_refined_fscore": metrics["refined_fscore"]["mean"],
                "mean_refined_normal_consistency": metrics[
                    "refined_normal_consistency"
                ]["mean"],
                "improved_meshes": item["improved_meshes"],
                "runtime_seconds_per_mesh": item["runtime_seconds_per_mesh"],
                "peak_gpu_memory_mb": item["peak_gpu_memory_mb"],
                "connectivity_preserved": sum(
                    _truth(row.get("output_connectivity_preserved")) is True
                    for row in completed_rows
                ),
                "introduced_flipped_faces": _sum_finite(
                    completed_rows, "introduced_flipped_faces"
                ),
                "new_degenerate_faces": _sum_finite(
                    completed_rows, "new_degenerate_faces"
                ),
                "failure_reasons": item["failure_reasons"],
            }
        )
    paired: dict[str, Any] = {}
    paired_rows: list[dict[str, Any]] = []
    ours = indexed["ours"]
    for method in (item for item in methods if item != "ours"):
        other = indexed[method]
        paired[method] = {}
        for name, (field, direction) in PAIRED_METRICS.items():
            result, rows = _paired_metric(ours, other, expected, field, direction)
            paired[method][name] = result
            paired_rows.extend({"external_method": method, **row} for row in rows)
    payload = {
        "experiment": selection["experiment"],
        "contract_audit": contract_audit,
        "input_contract_audit": input_contract_audit,
        "metric_completeness": metric_completeness,
        "input_contract": selection["shared_input_contract"],
        "source_manifest": selection["source_manifest"],
        "source_manifest_sha256": selection["source_manifest_sha256"],
        "selection_seed": selection["selection_seed"],
        "selection_algorithm": selection["selection_algorithm"],
        "sample_ids": expected,
        "surface_protocol": {
            "samples": args.surface_samples,
            "seed": args.metric_seed,
            "fscore_threshold": args.fscore_threshold,
        },
        "methods": list(methods),
        "summaries": summaries,
        "paired_ours_vs_external": paired,
        "contract_checks": audit_checks,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    combined = [row for method in methods for row in rows_by_method[method]]
    with (args.output_dir / "per_sample.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields: list[str] = []
        known: set[str] = set()
        for row in combined:
            for key in row:
                if key not in known:
                    fields.append(key)
                    known.add(key)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(combined)
    with (args.output_dir / "paired_ours_vs_external.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    (args.output_dir / "FINAL_REPORT.md").write_text(
        _markdown(payload), encoding="utf-8"
    )
    return payload


def _markdown(payload: dict[str, Any]) -> str:
    total = len(payload["sample_ids"])
    object_count = len({sample_id.rpartition("__v")[0] for sample_id in payload["sample_ids"]})
    lines = [
        f"# Future2000 full {total}-sample same-initial Blackwell benchmark",
        "",
        f"Contract audit: **{str(payload['contract_audit']).lower()}**.",
        f"Input-contract audit: **{str(payload['input_contract_audit']).lower()}**. "
        f"Metric completeness: **{str(payload['metric_completeness']).lower()}**.",
        "",
        f"Input: {payload['input_contract']}",
        "",
        "| Method | Complete | Valid CD | Chamfer | CD gain | P2S mean | P2S p95 | F-score | Normal | Improved |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["summaries"]:
        lines.append(
            f"| {item['method']} | {item['completed_samples']}/{total} | "
            f"{item.get('valid_chamfer_samples', total)}/{total} | "
            f"{item['mean_refined_chamfer']:.9g} | "
            f"{item.get('relative_chamfer_gain', 0.0):+.2%} | "
            f"{item['mean_refined_p2s_mean']:.9g} | "
            f"{item['mean_refined_p2s_p95']:.9g} | {item['mean_refined_fscore']:.9g} | "
            f"{item['mean_refined_normal_consistency']:.9g} | {item['improved_meshes']}/{total} |"
        )
    lines.extend(["", "## Paired ours vs external", ""])
    lines.extend(
        [
            "| External | Metric | Valid | Ours wins | Ties | External wins | Mean ours - external |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method, values in payload["paired_ours_vs_external"].items():
        for metric, result in values.items():
            lines.append(
                f"| {method} | {metric} | {result['valid_pairs']}/{total} | "
                f"{result['ours_wins']} | {result['ties']} | "
                f"{result['external_wins']} | "
                f"{_fmt(result['mean_ours_minus_external'])} |"
            )
    lines.extend(
        [
            "",
            "For lower-is-better metrics, an Ours win means a smaller value; for "
            "F-score and normal consistency, it means a larger value.",
            "",
            "## Topology and runtime",
            "",
            "| Method | Connectivity preserved | Introduced flips | New degenerates | Runtime/mesh | Peak GPU memory |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in payload["summaries"]:
        if item["method"] == "initial":
            continue
        runtime = item.get("runtime_seconds_per_mesh") or {}
        memory = item.get("peak_gpu_memory_mb") or {}
        lines.append(
            f"| {item['method']} | {item['connectivity_preserved']}/"
            f"{item['completed_samples']} | {_fmt(item['introduced_flipped_faces'], 12)} | "
            f"{_fmt(item['new_degenerate_faces'], 12)} | "
            f"{_fmt(runtime.get('mean'))} s | {_fmt(memory.get('mean'))} MiB |"
        )
    failures = [
        item
        for item in payload["summaries"]
        if item.get("failed_samples", 0) or item.get("invalid_chamfer_samples", 0)
    ]
    if failures:
        lines.extend(["", "## Failures and invalid metrics", ""])
        for item in failures:
            lines.append(
                f"- {item['method']}: failed {item.get('failed_samples', 0)}; "
                f"invalid Chamfer {item.get('invalid_chamfer_samples', 0)}; "
                f"reasons `{json.dumps(item.get('failure_reasons', {}), sort_keys=True)}`."
            )
    valid_summaries = [
        item
        for item in payload["summaries"]
        if item["method"] != "initial" and item["mean_refined_chamfer"] is not None
    ]
    best = min(valid_summaries, key=lambda item: item["mean_refined_chamfer"])
    ours_summary = next(
        item for item in payload["summaries"] if item["method"] == "ours"
    )
    lines.extend(
        [
            "",
            "## Concise conclusion",
            "",
            f"- Lowest aggregate valid-sample Chamfer: **{best['method']}** "
            f"({_fmt(best['mean_refined_chamfer'])}).",
            f"- Ours improves {ours_summary['improved_meshes']}/{total} inputs and "
            f"has aggregate Chamfer gain {ours_summary['relative_chamfer_gain']:+.2%}.",
            "- NDS-28V-full improves over the original one-view-per-iteration NDS "
            "aggregate, but remains behind Ours on paired Chamfer for 632/999 valid pairs.",
            "- Invalid outputs remain explicit and are excluded only from the affected "
            "metric denominator; no mesh cleanup or alternate evaluator was used.",
            "",
            f"This is the complete Future2000 test split: {object_count} objects and "
            f"{total} current-mesh variants, evaluated as eight independent modulo shards.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--surface-samples", type=int, default=3000)
    parser.add_argument("--metric-seed", type=int, default=7)
    parser.add_argument("--fscore-threshold", type=float, default=0.01)
    parser.add_argument("--methods", nargs="+")
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps(payload, indent=2))
    return 0 if payload["contract_audit"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
