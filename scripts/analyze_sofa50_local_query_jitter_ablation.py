#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def f(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.8g}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-summary", required=True, type=Path)
    parser.add_argument("--openmvs-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prediction = read_json(args.prediction_summary)
    openmvs = read_json(args.openmvs_summary)
    arms = ("A_no_jitter", "B_local_jitter")
    native = prediction["native_training"]
    test = prediction["deterministic_evaluation"]["test"]
    recovery = {row["arm"]: row for row in openmvs["aggregate"]}

    lines = [
        "# Sofa50 local query-position jitter ablation",
        "",
        "## Contract",
        "",
        "- Capacity: C2.",
        "- Feature resolution: F2.",
        "- Views per sample: 28.",
        "- Stored current variants per GT object: 5.",
        "- Seed: 7.",
        "- Optimizer-step budget per arm: 20000.",
        "- Arm A training query: stored current vertex position.",
        "- Arm B training query: `q_i = c_i + eta_i`, where `eta_i = h_i * clip_l2(N(0, 0.003^2 I), 0.009)`.",
        "- Proxy, target, `h_current`, graph connectivity, Laplacian operator and recovery settings are shared.",
        "- Validation and test query jitter: disabled.",
        "",
        "## Training",
        "",
        "| arm | best validation loss | runtime seconds | best epoch | peak GPU MB |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in arms:
        row = native[arm]
        lines.append(
            f"| {arm} | {f(row['best_validation_loss'])} | {f(row['runtime_seconds'])} | "
            f"{row['best_epoch']} | {f(row.get('peak_gpu_memory_mb'))} |"
        )
    lines.extend(
        [
            "",
            "## Deterministic test prediction",
            "",
            "| arm | raw endpoint | raw top10 endpoint | raw top1 endpoint | raw cosine | raw norm ratio | raw top10 cosine | raw top1 cosine | zero-RGB raw endpoint |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in arms:
        correct = test[arm]["correct_rgb"]
        zero = test[arm]["zero_rgb"]
        lines.append(
            f"| {arm} | {f(correct['raw_endpoint'])} | {f(correct['raw_top10_endpoint'])} | "
            f"{f(correct['raw_top1_endpoint'])} | {f(correct['raw_global_cosine'])} | "
            f"{f(correct['raw_prediction_to_target_norm_ratio'])} | "
            f"{f(correct['raw_top10_cosine'])} | {f(correct['raw_top1_cosine'])} | "
            f"{f(zero['raw_endpoint'])} |"
        )
    lines.extend(
        [
            "",
            "## OpenMVS48 current-mesh recovery",
            "",
            "| arm | meshes | initial Chamfer | refined Chamfer | initial point-to-surface | refined point-to-surface | initial normal consistency | refined normal consistency | improved meshes | flips | degeneracies |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in arms:
        row = recovery[arm]
        lines.append(
            f"| {arm} | {row['mesh_count']} | {f(row['mean_initial_chamfer'])} | "
            f"{f(row['mean_refined_chamfer'])} | {f(row['mean_initial_point_to_surface'])} | "
            f"{f(row['mean_refined_point_to_surface'])} | {f(row['mean_initial_normal_consistency'])} | "
            f"{f(row['mean_refined_normal_consistency'])} | {row['better_than_initial_meshes']} | "
            f"{row['introduced_flips']} | {row['new_degeneracies']} |"
        )

    b_minus_a = prediction["paired"]["test"]["B_minus_A"]
    rec_delta = float(recovery[arms[1]]["mean_refined_chamfer"]) - float(
        recovery[arms[0]]["mean_refined_chamfer"]
    )
    lines.extend(
        [
            "",
            "## Conclusions",
            "",
            f"- Test raw endpoint B-A: {f(b_minus_a['raw_endpoint']['mean'])}.",
            f"- Test raw top10 endpoint B-A: {f(b_minus_a['raw_top10_endpoint']['mean'])}.",
            f"- Test raw top1 endpoint B-A: {f(b_minus_a['raw_top1_endpoint']['mean'])}.",
            f"- Test raw cosine B-A: {f(b_minus_a['raw_global_cosine']['mean'])}.",
            f"- OpenMVS mean refined Chamfer B-A: {f(rec_delta)}.",
            f"- Runtime ratio B/A: {f(float(native[arms[1]]['runtime_seconds']) / float(native[arms[0]]['runtime_seconds']))}.",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
