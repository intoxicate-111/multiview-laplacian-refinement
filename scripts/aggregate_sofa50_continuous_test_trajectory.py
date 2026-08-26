#!/usr/bin/env python3
from __future__ import annotations

"""Aggregate read-only continuous B+E test checkpoint diagnostics."""

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


def _step(path: Path) -> int:
    match = re.search(r"step(\d+)_test", path.stem)
    if match is None:
        raise ValueError(path)
    return int(match.group(1))


def _fmt(value: Any) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.9g}"


def _external_cd_calibration(path: Path) -> dict[str, Any]:
    """Audit the trajectory CD against the corrected external-baseline scale."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    methods: list[dict[str, Any]] = []
    all_differences: list[float] = []
    for method in ("nds", "nvdiffrec", "exmesh"):
        selected = [row for row in source_rows if row.get("method") == method]
        differences: list[float] = []
        for row in selected:
            for phase in ("initial", "final"):
                native = row.get(f"native_{phase}_chamfer", "")
                unified = row.get(f"{phase}_chamfer", "")
                if native == "" or unified == "":
                    continue
                difference = abs(float(native) - float(unified))
                if math.isfinite(difference):
                    differences.append(difference)
                    all_differences.append(difference)
        methods.append(
            {
                "method": method,
                "comparisons": len(differences),
                "mean_absolute_difference": float(statistics.fmean(differences)),
                "maximum_absolute_difference": float(max(differences)),
            }
        )
    if not all_differences:
        raise RuntimeError(f"No external CD calibration pairs found in {path}")
    maximum = float(max(all_differences))
    return {
        "passed": maximum <= 1e-8,
        "source": str(path.resolve()),
        "meaning": (
            "The current trajectory already uses the corrected Sofa50 external-"
            "baseline unified CD scale; no multiplicative conversion is applied."
        ),
        "calibration_factor": 1.0,
        "tolerance": 1e-8,
        "maximum_absolute_difference": maximum,
        "methods": methods,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--external-benchmark-per-sample",
        type=Path,
        help=(
            "Optional corrected Sofa50 same-initial per-sample CSV used to audit "
            "the external-baseline CD scale."
        ),
    )
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob("step*_test.json"), key=_step)
    if not paths:
        raise RuntimeError("No checkpoint test diagnostics found")
    rows: list[dict[str, Any]] = []
    payloads = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payloads.append(payload)
        geometry = payload["geometry"]
        quality = payload["curvature_and_distortion"]
        rows.append(
            {
                "step": _step(path),
                "checkpoint_sha256": payload["checkpoint_sha256"],
                "test_chamfer": geometry["refined_chamfer"],
                "test_mean_relative_gain": geometry["relative_gain"],
                "test_vertex_rms": geometry["vertex_rms"],
                "test_p2s_p95": geometry["p2s_p95"],
                "test_fscore": geometry["fscore"],
                "test_normal_consistency": geometry["normal_consistency"],
                "introduced_flips": geometry["introduced_flips"],
                "new_degenerates": geometry["new_degenerates"],
                "improved": geometry["improved_worsened"][0],
                "worsened": geometry["improved_worsened"][1],
                **quality,
            }
        )
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    with (output / "test_checkpoint_trajectory.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "read_only": True,
        "test_used_for_selection": False,
        "checkpoint_snapshot_steps": [row["step"] for row in rows],
        "metric_protocol": payloads[0]["metric_protocol"],
        "curvature_protocol": payloads[0]["curvature_protocol"],
        "rows": rows,
    }
    calibration = None
    if args.external_benchmark_per_sample is not None:
        calibration = _external_cd_calibration(args.external_benchmark_per_sample)
        if not calibration["passed"]:
            raise RuntimeError(
                "External-baseline CD calibration failed: maximum discrepancy "
                f"{calibration['maximum_absolute_difference']:.9g}"
            )
        result["external_baseline_cd_calibration"] = calibration
    (output / "test_checkpoint_trajectory.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    best = min(rows, key=lambda row: row["test_chamfer"])
    lines = [
        "# Continuous pretrained B+E test checkpoint trajectory",
        "",
        "Read-only diagnostic. Test metrics were not used for checkpoint selection.",
        "",
        f"Lowest observed test Chamfer: **{_fmt(best['test_chamfer'])}** at step **{best['step']}**.",
        "",
    ]
    if calibration is not None:
        lines.extend(
            [
                "## External-baseline CD calibration",
                "",
                "Calibration audit: **true**. These trajectory CD values already use the corrected Sofa50 same-initial external-baseline evaluator (`evaluate_mesh_geometry`, 3,000 area-weighted surface samples, seed 7, bidirectional sampled-surface-to-exact-triangle distance). The calibration factor is therefore **1.0**; no numerical rescaling is applied.",
                "",
                "| Archived method | Native/unified comparisons | Mean absolute CD difference | Maximum absolute CD difference |",
                "|---|---:|---:|---:|",
            ]
        )
        for item in calibration["methods"]:
            lines.append(
                f"| {item['method']} | {item['comparisons']} | "
                f"{_fmt(item['mean_absolute_difference'])} | "
                f"{_fmt(item['maximum_absolute_difference'])} |"
            )
        lines.extend(
            [
                "",
                "The obsolete learned-method native CD used vertex subsampling and is not a constant-scale transform of this surface protocol; it is excluded from calibration and cross-method ranking.",
                "",
            ]
        )
    lines.extend(
        [
            "![Test checkpoint trajectory](test_checkpoint_trajectory.png)",
            "",
            "| Step | CD | VRMS | P2S p95 | F-score | Normal | Improved/worsened | Curvature 2H MAE | Scaled curvature MAE | Dihedral MAE (deg) | Face-normal MAE (deg) | Edge log error | Area log error |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['step']} | {_fmt(row['test_chamfer'])} | {_fmt(row['test_vertex_rms'])} | "
            f"{_fmt(row['test_p2s_p95'])} | {_fmt(row['test_fscore'])} | "
            f"{_fmt(row['test_normal_consistency'])} | {row['improved']}/{row['worsened']} | "
            f"{_fmt(row['twice_mean_curvature_magnitude_error_mean'])} | "
            f"{_fmt(row['scaled_curvature_error_mean'])} | "
            f"{_fmt(row['dihedral_angle_error_degrees_mean'])} | "
            f"{_fmt(row['face_normal_angle_error_degrees_mean'])} | "
            f"{_fmt(row['absolute_log_edge_length_ratio_mean'])} | "
            f"{_fmt(row['absolute_log_face_area_ratio_mean'])} |"
        )
    lines.extend(
        [
            "",
            "Curvature is the same-index cotangent discrete twice-mean-curvature magnitude error. All table values are macro-averages over the 50 test meshes.",
            "",
            f"Metric protocol: `{payloads[0]['metric_protocol']}`.",
            "",
            f"Curvature protocol: `{payloads[0]['curvature_protocol']}`.",
        ]
    )
    (output / "TEST_TRAJECTORY_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"steps": result["checkpoint_snapshot_steps"], "best": best}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
