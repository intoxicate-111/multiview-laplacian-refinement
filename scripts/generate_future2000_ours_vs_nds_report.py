#!/usr/bin/env python3
from __future__ import annotations

"""Generate a paired Ours-vs-NDS report from completed Future2000 results."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


LOWER_IS_BETTER = {
    "chamfer": "refined_chamfer",
    "p2s_mean": "refined_p2s_mean",
    "p2s_p95": "refined_p2s_p95",
    "introduced_flipped_faces": "introduced_flipped_faces",
    "new_degenerate_faces": "new_degenerate_faces",
}
HIGHER_IS_BETTER = {
    "fscore": "refined_fscore",
    "normal_consistency": "refined_normal_consistency",
}


def _rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    return {row["sample_id"]: row for row in rows}


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _bool(row: dict[str, str], key: str) -> bool:
    return row[key].lower() in {"true", "1"}


def _stats(values: list[float], seed: int = 7) -> dict[str, Any]:
    raw = np.asarray(values, dtype=np.float64)
    array = raw[np.isfinite(raw)]
    if not len(array):
        return {
            "count": 0,
            "invalid_count": int(len(raw)),
            "mean": None,
            "median": None,
            "std": None,
            "bootstrap_95_ci": None,
        }
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray(
        [rng.choice(array, size=len(array), replace=True).mean() for _ in range(2000)]
    )
    return {
        "count": int(len(array)),
        "invalid_count": int(len(raw) - len(array)),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
        "bootstrap_95_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
    }


def _method_summary(rows: dict[str, dict[str, str]]) -> dict[str, Any]:
    values = list(rows.values())
    metrics = {}
    for name, field in {**LOWER_IS_BETTER, **HIGHER_IS_BETTER}.items():
        metrics[name] = _stats([_float(row, field) for row in values])
    return {
        "samples": len(values),
        "improved_over_initial": sum(_bool(row, "improved") for row in values),
        "connectivity_preserved": sum(
            _bool(row, "output_connectivity_preserved") for row in values
        ),
        "metrics": metrics,
        "runtime_seconds": _stats([_float(row, "runtime_seconds") for row in values]),
        "peak_gpu_memory_mb": _stats(
            [_float(row, "peak_gpu_memory_mb") for row in values]
        ),
        "introduced_flipped_faces_total": int(
            sum(_float(row, "introduced_flipped_faces") for row in values)
        ),
        "new_degenerate_faces_total": int(
            sum(_float(row, "new_degenerate_faces") for row in values)
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    ours = _rows(args.ours / "per_sample.csv")
    nds = _rows(args.nds / "per_sample.csv")
    ids = sorted(ours)
    checks = {
        "ours_1000_unique_samples": len(ours) == 1000,
        "nds_1000_unique_samples": len(nds) == 1000,
        "same_sample_ids": ids == sorted(nds),
        "ours_all_completed": all(row["status"] == "completed" for row in ours.values()),
        "nds_all_completed": all(row["status"] == "completed" for row in nds.values()),
        "same_initial_mesh_sha256_per_sample": all(
            ours[sid]["common_initial_mesh_sha256"]
            == nds[sid]["common_initial_mesh_sha256"]
            != ""
            for sid in ids
        ),
        "same_initial_metrics_per_sample": all(
            all(
                abs(_float(ours[sid], field) - _float(nds[sid], field)) <= 1e-9
                for field in (
                    "initial_chamfer",
                    "initial_p2s_mean",
                    "initial_p2s_p95",
                    "initial_fscore",
                    "initial_normal_consistency",
                )
            )
            for sid in ids
        ),
        "same_28_views": all(
            ours[sid]["view_count"] == nds[sid]["view_count"] == "28" for sid in ids
        ),
    }
    if not checks["same_sample_ids"]:
        raise ValueError("Ours and NDS sample IDs do not match.")

    initial = {
        name: _stats([_float(ours[sid], f"initial_{field}") for sid in ids])
        for name, field in {
            "chamfer": "chamfer",
            "p2s_mean": "p2s_mean",
            "p2s_p95": "p2s_p95",
            "fscore": "fscore",
            "normal_consistency": "normal_consistency",
        }.items()
    }
    summaries = {"ours": _method_summary(ours), "nds": _method_summary(nds)}
    paired: dict[str, Any] = {}
    paired_rows: list[dict[str, Any]] = []
    for metric, field in LOWER_IS_BETTER.items():
        all_differences = [
            _float(ours[sid], field) - _float(nds[sid], field) for sid in ids
        ]
        differences = [value for value in all_differences if np.isfinite(value)]
        paired[metric] = {
            "direction": "lower_is_better",
            "valid_pairs": len(differences),
            "invalid_pairs": len(all_differences) - len(differences),
            "ours_wins": sum(value < 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
            "nds_wins": sum(value > 0 for value in differences),
            "ours_minus_nds": _stats(differences),
        }
    for metric, field in HIGHER_IS_BETTER.items():
        all_differences = [
            _float(ours[sid], field) - _float(nds[sid], field) for sid in ids
        ]
        differences = [value for value in all_differences if np.isfinite(value)]
        paired[metric] = {
            "direction": "higher_is_better",
            "valid_pairs": len(differences),
            "invalid_pairs": len(all_differences) - len(differences),
            "ours_wins": sum(value > 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
            "nds_wins": sum(value < 0 for value in differences),
            "ours_minus_nds": _stats(differences),
        }
    for sid in ids:
        item: dict[str, Any] = {
            "sample_id": sid,
            "initial_mesh_sha256": ours[sid]["common_initial_mesh_sha256"],
            "initial_chamfer": _float(ours[sid], "initial_chamfer"),
        }
        for metric, field in {**LOWER_IS_BETTER, **HIGHER_IS_BETTER}.items():
            ours_value = _float(ours[sid], field)
            nds_value = _float(nds[sid], field)
            item[f"ours_{metric}"] = ours_value
            item[f"nds_{metric}"] = nds_value
            item[f"ours_minus_nds_{metric}"] = ours_value - nds_value
        paired_rows.append(item)

    payload = {
        "experiment": "future2000_same_initial_full1000_ours_vs_nds_interim",
        "report_scope": "completed Ours-200k and NDS arms only; external benchmark continues",
        "contract_audit": all(checks.values()),
        "metric_completeness": all(
            item["invalid_pairs"] == 0 for item in paired.values()
        ),
        "invalid_distance_sample_ids": [
            sid
            for sid in ids
            if not all(
                np.isfinite(_float(nds[sid], field))
                for field in (
                    "refined_chamfer",
                    "refined_p2s_mean",
                    "refined_p2s_p95",
                )
            )
        ],
        "contract_checks": checks,
        "samples": len(ids),
        "objects": len({sid.rpartition("__v")[0] for sid in ids}),
        "variants_per_object": 5,
        "initial": initial,
        "methods": summaries,
        "paired_ours_vs_nds": paired,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output / "per_sample_paired.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    (args.output / "OURS_VS_NDS_REPORT.md").write_text(
        _markdown(payload), encoding="utf-8"
    )
    return payload


def _markdown(payload: dict[str, Any]) -> str:
    total = payload["samples"]
    initial = payload["initial"]
    ours = payload["methods"]["ours"]
    nds = payload["methods"]["nds"]
    lines = [
        "# Future2000 Ours-200k vs NDS interim report",
        "",
        f"Contract audit: **{str(payload['contract_audit']).lower()}**.",
        f"Metric completeness: **{str(payload['metric_completeness']).lower()}**.",
        "",
        (
            f"Scope: {payload['objects']} test objects x {payload['variants_per_object']} "
            f"variants = {total} paired meshes. This report covers only the completed "
            "Ours and NDS arms; nvdiffrec/ExMesh and the final benchmark remain in progress."
        ),
        "",
        "## Geometry metrics",
        "",
        "| Method | Chamfer | P2S mean | P2S p95 | F-score | Normal consistency | Improved / valid distance |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| Initial | {initial['chamfer']['mean']:.9g} | "
            f"{initial['p2s_mean']['mean']:.9g} | {initial['p2s_p95']['mean']:.9g} | "
            f"{initial['fscore']['mean']:.9g} | "
            f"{initial['normal_consistency']['mean']:.9g} | - |"
        ),
    ]
    for label, item in (("Ours-200k", ours), ("NDS (distance n=998)", nds)):
        metric = item["metrics"]
        lines.append(
            f"| {label} | {metric['chamfer']['mean']:.9g} | "
            f"{metric['p2s_mean']['mean']:.9g} | {metric['p2s_p95']['mean']:.9g} | "
            f"{metric['fscore']['mean']:.9g} | "
            f"{metric['normal_consistency']['mean']:.9g} | "
            f"{item['improved_over_initial']}/{metric['chamfer']['count']} |"
        )
    lines.extend(
        [
            "",
            "## Paired comparison",
            "",
            "| Metric | Valid pairs | Ours wins | Ties | NDS wins | Mean Ours - NDS | 95% bootstrap CI |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for metric in (
        "chamfer",
        "p2s_mean",
        "p2s_p95",
        "fscore",
        "normal_consistency",
        "introduced_flipped_faces",
        "new_degenerate_faces",
    ):
        item = payload["paired_ours_vs_nds"][metric]
        diff = item["ours_minus_nds"]
        lines.append(
            f"| {metric} | {item['valid_pairs']}/{total} | "
            f"{item['ours_wins']}/{item['valid_pairs']} | "
            f"{item['ties']}/{item['valid_pairs']} | "
            f"{item['nds_wins']}/{item['valid_pairs']} | {diff['mean']:+.9g} | "
            f"[{diff['bootstrap_95_ci'][0]:+.9g}, {diff['bootstrap_95_ci'][1]:+.9g}] |"
        )
    lines.extend(
        [
            "",
            "Distance-metric coverage is 998/1000 because NDS produced finite meshes but "
            "the frozen point-to-surface evaluator returned NaN for two outputs containing "
            "degenerate triangles. They remain explicit invalid outcomes; no mesh cleanup or "
            "alternative evaluator was applied.",
            "",
            "Invalid NDS distance samples: "
            + ", ".join(f"`{sid}`" for sid in payload["invalid_distance_sample_ids"])
            + ".",
            "",
            "## Topology and runtime",
            "",
            "| Method | Connectivity preserved | Introduced flips total | New degenerates total | Runtime mean | Peak memory mean |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label, item in (("Ours-200k", ours), ("NDS", nds)):
        lines.append(
            f"| {label} | {item['connectivity_preserved']}/{total} | "
            f"{item['introduced_flipped_faces_total']} | "
            f"{item['new_degenerate_faces_total']} | "
            f"{item['runtime_seconds']['mean']:.3f} s/mesh | "
            f"{item['peak_gpu_memory_mb']['mean']:.1f} MiB |"
        )
    chamfer = payload["paired_ours_vs_nds"]["chamfer"]
    normal = payload["paired_ours_vs_nds"]["normal_consistency"]
    lines.extend(
        [
            "",
            "## Interim conclusion",
            "",
            (
                f"Ours has lower paired Chamfer on {chamfer['ours_wins']}/"
                f"{chamfer['valid_pairs']} valid pairs and higher normal consistency "
                f"on {normal['ours_wins']}/{normal['valid_pairs']}. "
                "This conclusion is limited to Ours vs NDS; it is not the final external-baseline ranking."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours", type=Path, required=True)
    parser.add_argument("--nds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps(payload, indent=2))
    return 0 if payload["contract_audit"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
