#!/usr/bin/env python3
from __future__ import annotations

"""Generate an interim full-test report for the completed Ours arm only."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import fmean, median
from typing import Any


LOWER_IS_BETTER = (
    ("chamfer", "initial_chamfer", "refined_chamfer"),
    ("p2s_mean", "initial_p2s_mean", "refined_p2s_mean"),
    ("p2s_p95", "initial_p2s_p95", "refined_p2s_p95"),
)
HIGHER_IS_BETTER = (
    ("fscore", "initial_fscore", "refined_fscore"),
    (
        "normal_consistency",
        "initial_normal_consistency",
        "refined_normal_consistency",
    ),
)


def _truth(value: str) -> bool:
    return value.lower() in {"true", "1"}


def _mean(rows: list[dict[str, str]], field: str) -> float:
    return fmean(float(row[field]) for row in rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    aggregate = json.loads(args.aggregate.read_text(encoding="utf-8"))
    with args.per_sample.open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]

    object_counts = Counter(row["sample_id"].rpartition("__v")[0] for row in rows)
    checkpoints = {row["checkpoint_sha256"] for row in rows}
    optimizer_steps = {int(row["checkpoint_optimizer_steps"]) for row in rows}
    audits = {
        "exactly_1000_unique_samples": len(rows) == len({row["sample_id"] for row in rows}) == 1000,
        "exactly_200_objects_x_5_variants": len(object_counts) == 200
        and set(object_counts.values()) == {5},
        "all_completed": all(row["status"] == "completed" for row in rows),
        "method_is_ours": {row["method"] for row in rows} == {"ours"},
        "single_checkpoint": len(checkpoints) == 1,
        "checkpoint_is_200k": optimizer_steps == {200000},
        "all_have_28_views": {int(row["view_count"]) for row in rows} == {28},
        "source_identity_passed": all(
            _truth(row["common_initial_source_identity_audit"]) for row in rows
        ),
        "adapter_initial_matches_common_initial": all(
            row["adapter_initial_mesh_sha256"] == row["common_initial_mesh_sha256"]
            and float(row["adapter_initial_max_abs_vertex_error"]) == 0.0
            and _truth(row["adapter_initial_faces_exact"])
            and _truth(row["common_initial_identity_audit"])
            for row in rows
        ),
        "output_connectivity_preserved": all(
            _truth(row["output_connectivity_preserved"]) for row in rows
        ),
    }

    paired: dict[str, Any] = {}
    for name, initial, refined in LOWER_IS_BETTER:
        paired[name] = {
            "improved": sum(float(row[refined]) < float(row[initial]) for row in rows),
            "equal": sum(float(row[refined]) == float(row[initial]) for row in rows),
            "worsened": sum(float(row[refined]) > float(row[initial]) for row in rows),
        }
    for name, initial, refined in HIGHER_IS_BETTER:
        paired[name] = {
            "improved": sum(float(row[refined]) > float(row[initial]) for row in rows),
            "equal": sum(float(row[refined]) == float(row[initial]) for row in rows),
            "worsened": sum(float(row[refined]) < float(row[initial]) for row in rows),
        }

    flips = [int(row["introduced_flipped_faces"]) for row in rows]
    degenerates = [int(row["new_degenerate_faces"]) for row in rows]
    metrics = aggregate["metrics"]
    summary_metrics: dict[str, Any] = {}
    for name, initial_field, refined_field in LOWER_IS_BETTER + HIGHER_IS_BETTER:
        initial = _mean(rows, initial_field)
        refined = _mean(rows, refined_field)
        summary_metrics[name] = {
            "initial_mean": initial,
            "refined_mean": refined,
            "absolute_change": refined - initial,
            "relative_change": (refined - initial) / initial,
            "refined_bootstrap_95_ci": metrics[refined_field]["bootstrap_95_ci"],
        }

    runtime_values = [float(row["runtime_seconds"]) for row in rows]
    memory_values = [float(row["peak_gpu_memory_mb"]) for row in rows]
    payload = {
        "experiment": "future2000_same_initial_full1000_ours_200k_interim",
        "status": "completed",
        "contract_audit": all(audits.values()),
        "contract_checks": audits,
        "samples": len(rows),
        "objects": len(object_counts),
        "variants_per_object": sorted(set(object_counts.values())),
        "checkpoint_sha256": next(iter(checkpoints)) if len(checkpoints) == 1 else sorted(checkpoints),
        "optimizer_steps": sorted(optimizer_steps),
        "metrics": summary_metrics,
        "mean_per_sample_chamfer_improvement_rate": metrics[
            "chamfer_improvement_rate"
        ]["mean"],
        "paired": paired,
        "improved_over_initial": aggregate["improved_meshes"],
        "topology": {
            "connectivity_preserved": sum(
                _truth(row["output_connectivity_preserved"]) for row in rows
            ),
            "introduced_flipped_faces_total": sum(flips),
            "introduced_flipped_faces_mean": fmean(flips),
            "introduced_flipped_faces_median": median(flips),
            "zero_introduced_flips_samples": sum(value == 0 for value in flips),
            "new_degenerate_faces_total": sum(degenerates),
            "zero_new_degenerate_faces_samples": sum(value == 0 for value in degenerates),
        },
        "runtime": {
            "mean_seconds_per_mesh": fmean(runtime_values),
            "median_seconds_per_mesh": median(runtime_values),
            "summed_gpu_hours": sum(runtime_values) / 3600.0,
            "mean_peak_gpu_memory_mb": fmean(memory_values),
            "eight_gpu_stage_wall_time_range_minutes": [39.4, 43.1],
        },
        "evaluation": {
            "surface_samples": 3000,
            "surface_sampling_seed": 7,
            "fscore_threshold": 0.01,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "OURS_REPORT.md").write_text(
        _markdown(payload), encoding="utf-8"
    )
    return payload


def _markdown(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    paired = payload["paired"]
    topology = payload["topology"]
    runtime = payload["runtime"]
    lines = [
        "# Future2000 full-test Ours 200k interim report",
        "",
        f"Contract audit: **{str(payload['contract_audit']).lower()}**.",
        "",
        "This report covers the completed learned-Laplacian arm only. External baseline jobs are still running.",
        "",
        f"- Test coverage: {payload['objects']} objects x 5 variants = {payload['samples']} meshes",
        f"- Checkpoint optimizer steps: {payload['optimizer_steps'][0]}",
        f"- Checkpoint SHA-256: `{payload['checkpoint_sha256']}`",
        f"- Improved over initial: {payload['improved_over_initial']}/{payload['samples']}",
        "",
        "## Geometry metrics",
        "",
        "| Metric | Initial mean | Refined mean | Absolute change | Relative change | Refined 95% bootstrap CI | Paired improved |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("chamfer", "p2s_mean", "p2s_p95", "fscore", "normal_consistency"):
        item = metrics[name]
        ci = item["refined_bootstrap_95_ci"]
        lines.append(
            f"| {name} | {item['initial_mean']:.9g} | {item['refined_mean']:.9g} | "
            f"{item['absolute_change']:+.9g} | {item['relative_change']:+.2%} | "
            f"[{ci[0]:.9g}, {ci[1]:.9g}] | {paired[name]['improved']}/{payload['samples']} |"
        )
    lines.extend(
        [
            "",
            "The mean of the 1,000 per-sample Chamfer improvement rates is "
            f"{payload['mean_per_sample_chamfer_improvement_rate']:.2%}.",
            "",
            "## Topology and runtime",
            "",
            f"- Connectivity preserved: {topology['connectivity_preserved']}/{payload['samples']}",
            f"- Introduced flipped faces: total {topology['introduced_flipped_faces_total']}, "
            f"mean {topology['introduced_flipped_faces_mean']:.3f}, median {topology['introduced_flipped_faces_median']:.3f}; "
            f"zero-flip samples {topology['zero_introduced_flips_samples']}/{payload['samples']}",
            f"- New degenerate faces: total {topology['new_degenerate_faces_total']}; "
            f"zero-degenerate samples {topology['zero_new_degenerate_faces_samples']}/{payload['samples']}",
            f"- Runtime: mean {runtime['mean_seconds_per_mesh']:.3f} s/mesh, "
            f"median {runtime['median_seconds_per_mesh']:.3f} s/mesh, "
            f"summed GPU time {runtime['summed_gpu_hours']:.3f} h",
            f"- Mean peak GPU memory: {runtime['mean_peak_gpu_memory_mb']:.1f} MiB",
            f"- Eight-GPU stage wall time: {runtime['eight_gpu_stage_wall_time_range_minutes'][0]:.1f}-"
            f"{runtime['eight_gpu_stage_wall_time_range_minutes'][1]:.1f} min across shards",
            "",
            "## Interim conclusion",
            "",
            "The 200k learned model produces a strong and broad surface-distance improvement: "
            f"Chamfer is lower for {paired['chamfer']['improved']}/{payload['samples']} meshes, "
            f"and mean P2S p95 changes by {metrics['p2s_p95']['relative_change']:+.2%}. "
            "The gain does not transfer to mean normal consistency, which changes by "
            f"{metrics['normal_consistency']['relative_change']:+.2%}; this is a real distance-versus-normal trade-off, "
            "not a reason to alter the frozen evaluation contract mid-benchmark.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--per-sample", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2))
    return 0 if result["contract_audit"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
