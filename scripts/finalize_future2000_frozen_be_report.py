#!/usr/bin/env python3
from __future__ import annotations

"""Merge frozen Future2000 B+E results and generate the comprehensive report."""

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from statistics import fmean
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_future2000_same_initial_subset_report import run as generate_report
from merge_future2000_external_baseline import merge


METHOD_LABELS = {
    "initial": "Initial mesh",
    "ours": "B+E Hybrid",
    "arm_b": "Arm-B",
    "arm_e": "Arm-E",
    "old_structure": "Old structure",
    "nds": "NDS",
    "nvdiffrec": "nvdiffrec",
    "exmesh": "ExMesh",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for row in rows:
        for key in row:
            if key not in names:
                names.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _materialize_new_methods(args: argparse.Namespace) -> None:
    by_method: dict[str, list[list[dict[str, Any]]]] = {"arm_e": [], "hybrid": []}
    for index in range(args.shard_count):
        path = args.run_dir / "test" / "shards" / f"test_shard_{index:03d}.csv"
        metadata_path = path.with_suffix(".metadata.json")
        rows = _read_csv(path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not metadata.get("contract_audit"):
            raise RuntimeError(f"Test shard {index} failed contract audit")
        peak_mb = float(metadata["peak_gpu_memory_bytes"]) / (1024.0 * 1024.0)
        for method in by_method:
            selected = []
            for source in rows:
                if source["method"] != method:
                    continue
                row = dict(source)
                row["status"] = "completed"
                row["runtime_seconds"] = row["total_compute_seconds"]
                row["peak_gpu_memory_mb"] = peak_mb
                row["failure_stage"] = ""
                row["failure_reason"] = ""
                selected.append(row)
            by_method[method].append(selected)
            method_shards = args.run_dir / "results" / method / "shards"
            _write_csv(method_shards / f"per_sample_shard_{index:03d}.csv", selected)
            (method_shards / f"metadata_shard_{index:03d}.json").write_text(
                json.dumps({
                    **metadata,
                    "method": method,
                    "pinned_commit": metadata[
                        "arm_e_checkpoint_sha256" if method == "arm_e" else "arm_b_checkpoint_sha256"
                    ],
                    "repository": str(ROOT),
                }, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    for method, shards in by_method.items():
        rows = [row for shard in shards for row in shard]
        if len(rows) != args.expected_samples or len({row["sample_id"] for row in rows}) != args.expected_samples:
            raise RuntimeError(f"{method}: expected {args.expected_samples} unique test rows")
        merge(args.manifest, args.run_dir / "results", method, args.shard_count)


def _link(path: Path, target: Path) -> None:
    target = target.resolve()
    if path.is_symlink():
        if path.resolve() != target:
            raise RuntimeError(f"Existing symlink has wrong target: {path}")
        return
    if path.exists():
        raise RuntimeError(f"Refusing to replace existing path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target, target_is_directory=True)


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def _mesh_bootstrap(differences: list[float], seed: int, replicates: int = 10_000) -> list[float]:
    generator = random.Random(seed)
    samples = [
        fmean(generator.choices(differences, k=len(differences)))
        for _ in range(replicates)
    ]
    return [_quantile(samples, 0.025), _quantile(samples, 0.975)]


def _fmt(value: Any, precision: int = 9) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{precision}g}"


def _report(payload: dict[str, Any], lock: dict[str, Any], sweep: list[dict[str, str]], args: argparse.Namespace) -> str:
    total = len(payload["sample_ids"])
    objects = len({item.rpartition("__v")[0] for item in payload["sample_ids"]})
    summaries = {row["method"]: row for row in payload["summaries"]}
    lines = [
        f"# Future2000 frozen Arm-B + Arm-E fusion and baseline comparison",
        "",
        f"Contract audit: **{str(payload['input_contract_audit']).lower()}**. "
        f"Metric completeness: **{str(payload['metric_completeness']).lower()}**.",
        "",
        f"This report uses the complete Future2000 test split ({objects} objects, {total} current-mesh variants). "
        "Every method receives the same current mesh and the same 28 native-960 RGB images/cameras; GT is evaluation-only. "
        "The Arm-B and Arm-E checkpoints were frozen before fusion selection.",
        "",
        "## Validation-only fusion selection",
        "",
        f"The fusion solves `(L_U^T L_U + lambda I)V = L_U^T delta_B + lambda V_E` with the Uniform random-walk operator. "
        f"A {len(sweep)}-point grid was evaluated on {lock['validation_sample_count']} validation meshes; mean CD selected "
        f"`lambda={lock['selected_lambda']}`. Test data were not used for selection.",
        "",
        "| Lambda | Validation CD | P2S p95 | F-score | Normal | VRMS | Improved |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sweep:
        marker = "**" if math.isclose(float(row["lambda"]), float(lock["selected_lambda"])) else ""
        lines.append(
            f"| {marker}{row['lambda']}{marker} | {marker}{_fmt(row['mean_cd'])}{marker} | "
            f"{_fmt(row['mean_p2s_p95'])} | {_fmt(row['mean_fscore'])} | {_fmt(row['mean_normal'])} | "
            f"{_fmt(row['mean_vrms'])} | {row['improved']}/{row['sample_count']} |"
        )
    lines.extend([
        "",
        "## Test comparison",
        "",
        "| Method | Complete | Valid CD | CD | CD gain | P2S p95 | F-score | Normal | Improved | Runtime s/mesh |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    order = ("initial", "arm_b", "arm_e", "ours", "old_structure", "nds", "nvdiffrec", "exmesh")
    for method in order:
        item = summaries[method]
        label = METHOD_LABELS[method]
        if method == "ours":
            label = f"**{label} (lambda={lock['selected_lambda']})**"
        runtime = item.get("runtime_seconds_per_mesh") or {}
        lines.append(
            f"| {label} | {item['completed_samples']}/{total} | {item.get('valid_chamfer_samples', total)}/{total} | "
            f"{_fmt(item['mean_refined_chamfer'])} | {float(item.get('relative_chamfer_gain') or 0):+.2%} | "
            f"{_fmt(item['mean_refined_p2s_p95'])} | {_fmt(item['mean_refined_fscore'])} | "
            f"{_fmt(item['mean_refined_normal_consistency'])} | {item['improved_meshes']}/{total} | "
            f"{_fmt(runtime.get('mean'))} |"
        )
    lines.extend([
        "",
        "Chamfer and bidirectional P2S mean are identical under this evaluator, so P2S mean is omitted; P2S p95 remains a distinct tail metric.",
        "",
        "## Paired Hybrid comparisons",
        "",
        "Differences are Hybrid minus comparator. Negative CD/P2S and positive F-score/Normal favor Hybrid.",
        "",
        "| Comparator | CD difference | Mesh 95% CI | Object-cluster 95% CI | Hybrid W/T/L | P2S p95 diff | F-score diff | Normal diff |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for method in order[1:]:
        if method == "ours":
            continue
        paired = payload["paired_ours_vs_external"][method]
        cd = paired["chamfer"]
        cluster = cd["object_cluster"]
        per_sample = [
            row for row in payload["_paired_rows"]
            if row["external_method"] == method and row["metric"] == "refined_chamfer" and row["outcome"] != "invalid"
        ]
        differences = [float(row["ours"]) - float(row["external"]) for row in per_sample]
        mesh_ci = _mesh_bootstrap(differences, args.metric_seed) if differences else None
        cluster_ci = cluster["ci95"]
        mesh_text = "n/a" if mesh_ci is None else f"[{_fmt(mesh_ci[0])}, {_fmt(mesh_ci[1])}]"
        cluster_text = "n/a" if cluster_ci is None else f"[{_fmt(cluster_ci[0])}, {_fmt(cluster_ci[1])}]"
        lines.append(
            f"| {METHOD_LABELS[method]} | {_fmt(cd['mean_ours_minus_external'])} | {mesh_text} | {cluster_text} | "
            f"{cd['ours_wins']}/{cd['ties']}/{cd['external_wins']} | "
            f"{_fmt(paired['p2s_p95']['mean_ours_minus_external'])} | "
            f"{_fmt(paired['fscore']['mean_ours_minus_external'])} | "
            f"{_fmt(paired['normal_consistency']['mean_ours_minus_external'])} |"
        )
    hybrid = summaries["ours"]
    e = summaries["arm_e"]
    b = summaries["arm_b"]
    lines.extend([
        "",
        "## Topology, runtime, and audit",
        "",
        "| Method | Connectivity preserved | Introduced flips | New degenerates | Mean compute s/mesh |",
        "|---|---:|---:|---:|---:|",
    ])
    for method in ("arm_b", "arm_e", "ours", "old_structure", "nds", "nvdiffrec", "exmesh"):
        item = summaries[method]
        runtime = item.get("runtime_seconds_per_mesh") or {}
        lines.append(
            f"| {METHOD_LABELS[method]} | {item['connectivity_preserved']}/{item['completed_samples']} | "
            f"{_fmt(item['introduced_flipped_faces'], 12)} | {_fmt(item['new_degenerate_faces'], 12)} | {_fmt(runtime.get('mean'))} |"
        )
    lines.extend([
        "",
        f"- Arm-B checkpoint SHA-256: `{lock['arm_b_checkpoint_sha256']}`.",
        f"- Arm-E checkpoint SHA-256: `{lock['arm_e_checkpoint_sha256']}`.",
        f"- Manifest SHA-256: `{lock['manifest_sha256']}`.",
        f"- All Hybrid float64 PCG solves used tolerance `{lock['pcg_tolerance']}` and maximum `{lock['pcg_maximum_iterations']}` iterations.",
        "- External-method invalid outputs remain explicit and are excluded only from affected metric denominators; no cleanup or alternate evaluator was used.",
        "- Arm-B and external test results pre-existed. Arm-E and Hybrid test evaluation was opened once only after the validation lambda lock; therefore the new E/Hybrid comparison is sealed, but the full all-method table is not a wholly fresh sealed benchmark.",
        "",
        "## Conclusion",
        "",
        f"B+E Hybrid test CD is `{_fmt(hybrid['mean_refined_chamfer'])}`, compared with Arm-B `{_fmt(b['mean_refined_chamfer'])}` and Arm-E `{_fmt(e['mean_refined_chamfer'])}`. "
        "Interpretation should follow the paired confidence intervals above and retain any Normal/topology trade-off rather than claiming uniform dominance.",
        "",
        "Raw merged tables, the validation sweep, lambda lock, per-sample metrics, topology diagnostics, solver audits, result meshes, and machine-readable summary are stored beside this report or in the linked run directory.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--baseline-results-root", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--expected-samples", type=int, default=1000)
    parser.add_argument("--surface-samples", type=int, default=3000)
    parser.add_argument("--metric-seed", type=int, default=7)
    parser.add_argument("--fscore-threshold", type=float, default=0.01)
    args = parser.parse_args()

    _materialize_new_methods(args)
    combined = args.run_dir / "combined_results"
    _link(combined / "ours", args.run_dir / "results" / "hybrid")
    _link(combined / "arm_e", args.run_dir / "results" / "arm_e")
    _link(combined / "arm_b", args.baseline_results_root / "ours")
    for method in ("old_structure", "nds", "nvdiffrec", "exmesh"):
        _link(combined / method, args.baseline_results_root / method)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    namespace = SimpleNamespace(
        selection=None,
        manifest=args.manifest,
        results_root=combined,
        output_dir=args.report_dir,
        surface_samples=args.surface_samples,
        metric_seed=args.metric_seed,
        fscore_threshold=args.fscore_threshold,
        methods=["ours", "arm_b", "arm_e", "old_structure", "nds", "nvdiffrec", "exmesh"],
    )
    payload = generate_report(namespace)
    paired_rows = _read_csv(args.report_dir / "paired_ours_vs_external.csv")
    payload["_paired_rows"] = paired_rows
    lock = json.loads((args.run_dir / "validation" / "lambda_lock.json").read_text(encoding="utf-8"))
    sweep = _read_csv(args.run_dir / "validation" / "lambda_sweep.csv")
    report = _report(payload, lock, sweep, args)
    (args.report_dir / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    summary = dict(payload)
    summary.pop("_paired_rows", None)
    summary["lambda_lock"] = lock
    (args.report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(args.report_dir / "FINAL_REPORT.md")
    return 0 if payload["input_contract_audit"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
