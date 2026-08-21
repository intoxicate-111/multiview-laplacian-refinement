#!/usr/bin/env python3
from __future__ import annotations

"""Aggregate the controlled Sofa50 same-initial benchmark without hiding failures."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

from mlr.data import Mesh
from mlr.io import load_mesh
from mlr.learned_laplacian.evaluation import evaluate_mesh_geometry
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


METHODS = ("ours", "exmesh", "nds", "nvdiffrec")


def _load_status(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = payload.get("row", payload)
    if not isinstance(row, dict):
        raise ValueError(f"Invalid status row: {path}")
    return row


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _mean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _standardize(method: str, row: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    ours = method == "ours"
    native_final_key = "reconstruction_chamfer" if ours else "refined_chamfer"
    native_initial_p2s_key = "initial_point_to_surface" if ours else "initial_p2s_mean"
    native_final_p2s_key = "reconstruction_point_to_surface" if ours else "refined_p2s_mean"
    native_final_normal_key = (
        "reconstruction_normal_consistency" if ours else "refined_normal_consistency"
    )
    return {
        "sample_id": source["sample_id"],
        "method": method,
        "status": row.get("status", "failed"),
        "failure_stage": row.get("failure_stage", ""),
        "failure_reason": row.get("failure_reason", ""),
        "common_initial_mesh": source["common_initial_mesh"],
        "common_initial_mesh_sha256": source["common_initial_mesh_sha256"],
        "observed_common_initial_mesh_sha256": row.get(
            "common_initial_mesh_sha256", ""
        ),
        "initial_vertex_count": source["initial_vertex_count"],
        "initial_face_count": source["initial_face_count"],
        "image_directory": source["image_directory"],
        "camera_and_gt_container": source["camera_and_gt_container"],
        "view_count": source["view_count"],
        # Native method metrics are provenance only.  The primary fields below
        # are populated by one common evaluator before aggregation.
        "native_initial_chamfer": _number(row.get("initial_chamfer")),
        "native_final_chamfer": _number(row.get(native_final_key)),
        "native_initial_p2s": _number(row.get(native_initial_p2s_key)),
        "native_final_p2s": _number(row.get(native_final_p2s_key)),
        "native_initial_normal_consistency": _number(
            row.get("initial_normal_consistency")
        ),
        "native_final_normal_consistency": _number(row.get(native_final_normal_key)),
        "initial_chamfer": None,
        "final_chamfer": None,
        "chamfer_improvement_percent": None,
        "initial_p2s": None,
        "final_p2s": None,
        "initial_p2s_p95": None,
        "final_p2s_p95": None,
        "initial_fscore": None,
        "final_fscore": None,
        "initial_normal_consistency": None,
        "final_normal_consistency": None,
        "unified_metric_audit": False,
        "metric_protocol": "",
        "introduced_flipped_faces": _number(row.get("introduced_flipped_faces")),
        "introduced_flipped_faces_comparable": row.get("introduced_flipped_faces_comparable", ours),
        "output_connectivity_preserved": row.get("output_connectivity_preserved", ours),
        "final_vertex_count": row.get("final_vertex_count", row.get("vertex_count", "")),
        "final_face_count": row.get("final_face_count", row.get("face_count", "")),
        "runtime_seconds": _number(row.get("runtime_seconds")),
        "peak_gpu_memory_mb": _number(row.get("peak_gpu_memory_mb")),
        "final_mesh": row.get("final_mesh", ""),
        "coordinate_transform_to_gt": row.get("coordinate_transform_to_gt", ""),
        "source_identity_audit": row.get("common_initial_identity_audit" if ours else "common_initial_source_identity_audit", False),
        "adapter_identity_audit": True if ours else row.get("common_initial_identity_audit", False),
    }


def _metric_values(metrics: dict[str, float]) -> dict[str, float]:
    return {
        "chamfer": float(metrics["chamfer"]),
        "p2s": float(metrics["point_to_surface_bidirectional_mean"]),
        "p2s_p95": float(metrics["point_to_surface_bidirectional_p95"]),
        "fscore": float(metrics["fscore"]),
        "normal_consistency": float(metrics["normal_consistency"]),
    }


def _assign_unified_metrics(
    row: dict[str, Any],
    initial: dict[str, float],
    final: dict[str, float],
    protocol: str,
) -> None:
    row.update(
        {
            "initial_chamfer": initial["chamfer"],
            "final_chamfer": final["chamfer"],
            "initial_p2s": initial["p2s"],
            "final_p2s": final["p2s"],
            "initial_p2s_p95": initial["p2s_p95"],
            "final_p2s_p95": final["p2s_p95"],
            "initial_fscore": initial["fscore"],
            "final_fscore": final["fscore"],
            "initial_normal_consistency": initial["normal_consistency"],
            "final_normal_consistency": final["normal_consistency"],
            "chamfer_improvement_percent": (
                100.0 * (initial["chamfer"] - final["chamfer"]) / initial["chamfer"]
                if initial["chamfer"] > 0
                else 0.0
            ),
            "unified_metric_audit": True,
            "metric_protocol": protocol,
        }
    )


def _unified_reevaluate(
    manifest_path: Path,
    sources: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    surface_samples: int,
    seed: int,
    fscore_threshold: float,
) -> None:
    """Recompute every primary geometry metric with exactly one evaluator."""

    dataset = PreparedMeshDataset.from_manifest(manifest_path, "test")
    index_by_id = {
        str(sample_id): index for index, sample_id in enumerate(dataset.sample_ids)
    }
    protocol = (
        "mlr.learned_laplacian.evaluation.evaluate_mesh_geometry;"
        f"surface_samples={surface_samples};seed={seed};"
        f"fscore_threshold={fscore_threshold}"
    )
    rows_by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_sample.setdefault(str(row["sample_id"]), []).append(row)
    for source in sources:
        sample_id = str(source["sample_id"])
        if sample_id not in index_by_id:
            raise ValueError(f"Unified evaluator cannot locate sample {sample_id!r}.")
        static = dataset.load_static(index_by_id[sample_id])
        gt = Mesh(
            static["gt_vertices"].detach().cpu().numpy(),
            static["gt_faces"].detach().cpu().numpy(),
        ).ensure_normals()
        initial_mesh = load_mesh(Path(str(source["common_initial_mesh"]))).ensure_normals()
        initial = _metric_values(
            evaluate_mesh_geometry(
                initial_mesh,
                gt,
                surface_samples=surface_samples,
                seed=seed,
                fscore_threshold=fscore_threshold,
            )
        )
        for row in rows_by_sample.get(sample_id, []):
            if row["status"] != "completed":
                continue
            try:
                final_path = Path(str(row["final_mesh"]))
                if not final_path.is_file():
                    raise FileNotFoundError(f"Missing final mesh: {final_path}")
                final_mesh = load_mesh(final_path).ensure_normals()
                final = _metric_values(
                    evaluate_mesh_geometry(
                        final_mesh,
                        gt,
                        surface_samples=surface_samples,
                        seed=seed,
                        fscore_threshold=fscore_threshold,
                    )
                )
                _assign_unified_metrics(row, initial, final, protocol)
            except Exception as exc:
                row.update(
                    {
                        "status": "failed",
                        "failure_stage": "unified_evaluation",
                        "failure_reason": f"{exc.__class__.__name__}: {exc}",
                    }
                )


def _failure(method: str, source: dict[str, Any], reason: str) -> dict[str, Any]:
    return _standardize(
        method,
        {"status": "failed", "failure_stage": "missing_status", "failure_reason": reason},
        source,
    )


def _aggregate(method: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row["status"] == "completed"]
    values = lambda key: [float(row[key]) for row in completed if row[key] is not None]
    improvements = values("chamfer_improvement_percent")
    final_cd = values("final_chamfer")
    initial_cd = values("initial_chamfer")
    mean_initial = _mean(initial_cd)
    mean_final = _mean(final_cd)
    aggregate_improvement = (
        100.0 * (mean_initial - mean_final) / mean_initial
        if mean_initial is not None and mean_final is not None and mean_initial > 0
        else None
    )
    return {
        "method": method,
        "expected_samples": len(rows),
        "completed_samples": len(completed),
        "failed_samples": len(rows) - len(completed),
        "mean_initial_chamfer": mean_initial,
        "mean_final_chamfer": mean_final,
        "aggregate_chamfer_improvement_percent": aggregate_improvement,
        "mean_per_sample_chamfer_improvement_percent": _mean(improvements),
        "median_per_sample_chamfer_improvement_percent": _median(improvements),
        "mean_final_p2s": _mean(values("final_p2s")),
        "mean_final_normal_consistency": _mean(values("final_normal_consistency")),
        "improved_samples": sum(float(row["final_chamfer"]) < float(row["initial_chamfer"]) for row in completed if row["final_chamfer"] is not None and row["initial_chamfer"] is not None),
        "worsened_samples": sum(float(row["final_chamfer"]) > float(row["initial_chamfer"]) for row in completed if row["final_chamfer"] is not None and row["initial_chamfer"] is not None),
        "unchanged_samples": sum(float(row["final_chamfer"]) == float(row["initial_chamfer"]) for row in completed if row["final_chamfer"] is not None and row["initial_chamfer"] is not None),
        "mean_vertices": _mean([float(row["final_vertex_count"]) for row in completed if row["final_vertex_count"] not in (None, "")]),
        "mean_faces": _mean([float(row["final_face_count"]) for row in completed if row["final_face_count"] not in (None, "")]),
        "mean_runtime_seconds": _mean(values("runtime_seconds")),
        "total_runtime_seconds": sum(values("runtime_seconds")),
        "mean_peak_gpu_memory_mb": _mean(values("peak_gpu_memory_mb")),
        "max_peak_gpu_memory_mb": max(values("peak_gpu_memory_mb"), default=None),
        "connectivity_preserved_samples": sum(row["output_connectivity_preserved"] is True for row in completed),
        "topology_changed_samples": sum(row["output_connectivity_preserved"] is False for row in completed),
        "mean_introduced_flipped_faces_when_comparable": _mean([
            float(row["introduced_flipped_faces"])
            for row in completed
            if row["introduced_flipped_faces_comparable"] is True and row["introduced_flipped_faces"] is not None
        ]),
        "failures": [
            {"sample_id": row["sample_id"], "stage": row["failure_stage"], "reason": row["failure_reason"]}
            for row in rows
            if row["status"] != "completed"
        ],
    }


def _initial_rows(sources: list[dict[str, Any]], standardized: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_sample: dict[str, dict[str, Any]] = {}
    for row in standardized:
        if row["status"] == "completed" and row["initial_chamfer"] is not None:
            by_sample.setdefault(row["sample_id"], row)
    result = []
    for source in sources:
        reference = by_sample.get(source["sample_id"])
        if reference is None:
            result.append(_failure("initial", source, "No completed method row supplied common initial metrics"))
            continue
        row = dict(reference)
        row.update(
            {
                "method": "initial",
                "status": "completed",
                "failure_stage": "",
                "failure_reason": "",
                "final_chamfer": reference["initial_chamfer"],
                "chamfer_improvement_percent": 0.0,
                "final_p2s": reference["initial_p2s"],
                "final_p2s_p95": reference["initial_p2s_p95"],
                "final_fscore": reference["initial_fscore"],
                "final_normal_consistency": reference["initial_normal_consistency"],
                "introduced_flipped_faces": 0.0,
                "introduced_flipped_faces_comparable": True,
                "output_connectivity_preserved": True,
                "final_vertex_count": source["initial_vertex_count"],
                "final_face_count": source["initial_face_count"],
                "runtime_seconds": 0.0,
                "peak_gpu_memory_mb": 0.0,
                "final_mesh": source["common_initial_mesh"],
            }
        )
        result.append(row)
    return result


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# Sofa50 controlled same-initial-mesh benchmark",
        "",
        "Primary claim scope: `same prepared synthetic mesh + same 28-view RGB/cameras -> different refinement methods`.",
        "",
        f"Contract audit: **{str(summary['contract_audit']).lower()}**. Completed methods are evaluated with the same deterministic 3,000-point surface protocol (seed 7).",
        "",
        "## Group A aggregate",
        "",
        "| Method | Complete | Mean initial CD | Mean final CD | CD improvement | Mean P2S | Normal | Improved | Worsened | Vertices | Faces | Runtime/sample | Peak GPU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["aggregate"]:
        lines.append(
            "| " + " | ".join(
                [
                    row["method"],
                    f"{row['completed_samples']}/{row['expected_samples']}",
                    _fmt(row["mean_initial_chamfer"]),
                    _fmt(row["mean_final_chamfer"]),
                    _fmt(row["aggregate_chamfer_improvement_percent"]),
                    _fmt(row["mean_final_p2s"]),
                    _fmt(row["mean_final_normal_consistency"]),
                    str(row["improved_samples"]),
                    str(row["worsened_samples"]),
                    _fmt(row["mean_vertices"]),
                    _fmt(row["mean_faces"]),
                    _fmt(row["mean_runtime_seconds"]),
                    _fmt(row["max_peak_gpu_memory_mb"]),
                ]
            ) + " |"
        )
    lines.extend(
        [
            "",
            "CD improvement is `(mean initial CD - mean final CD) / mean initial CD * 100%`; per-sample mean and median values are retained in `summary.json`.",
            "",
            "## Input and method contract",
            "",
            f"- Dataset: `{summary['dataset']}` ({summary['sample_count']} canonical test samples).",
            "- Common input: the exact existing prepared current OBJ, the same 28 native-1920 RGB images, and the same prepared cameras.",
            "- GT is consumed only by the common evaluator.",
            "- Group A: initial, ours, ExMesh, NDS, nvdiffrec.",
            "- Group B: Neuralangelo (neural SDF/marching-cubes reconstruction) and MAtCha (point-map/Gaussian/chart reconstruction); neither officially refines an arbitrary supplied triangular mesh.",
            "- ExMesh PGSR and NDS visual-hull initialization are bypassed. nvdiffrec uses its official fixed-topology DLMesh path with the supplied base mesh.",
            "",
            "## Topology and metric limitations",
            "",
            "Introduced flipped faces are reported only when output V/F and face ordering preserve the common connectivity. For a topology-changing output such as ExMesh this metric is unavailable rather than inferred from unrelated faces. The configured NDS and nvdiffrec adapters preserve the supplied connectivity. Runtime and memory are implementation/hardware measurements, not algorithm-independent complexity estimates. Ours peak memory is PyTorch peak allocated memory; external-method peak memory is process-tree GPU usage sampled through nvidia-smi, so the two columns are useful operational measurements but not byte-identical profiler definitions.",
            "",
            "## Failures",
            "",
        ]
    )
    failures = [item for row in summary["aggregate"] for item in row["failures"]]
    if failures:
        for item in failures:
            lines.append(f"- `{item['sample_id']}`: {item['stage']}: {item['reason']}")
    else:
        lines.append("No failed Group A runs.")
    lines.extend(
        [
            "",
            "The obsolete `ours_exmesh_initial_zero_shot = 0.616526 mm` result is excluded from every table in this benchmark and remains provenance-only.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    sources = [dict(row) for row in manifest["samples"]]
    all_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for source in sources:
            status_path = args.results_root / method / "samples" / source["sample_id"] / "status.json"
            row = _load_status(status_path) if status_path.is_file() else None
            all_rows.append(
                _standardize(method, row, source)
                if row is not None
                else _failure(method, source, f"Missing status file: {status_path}")
            )
    surface_samples = int(getattr(args, "surface_samples", 3000))
    metric_seed = int(getattr(args, "metric_seed", 7))
    fscore_threshold = float(getattr(args, "fscore_threshold", 0.01))
    _unified_reevaluate(
        args.manifest,
        sources,
        all_rows,
        surface_samples=surface_samples,
        seed=metric_seed,
        fscore_threshold=fscore_threshold,
    )
    initial = _initial_rows(sources, all_rows)
    all_rows = initial + all_rows
    aggregates = [_aggregate(method, [row for row in all_rows if row["method"] == method]) for method in ("initial", *METHODS)]
    completed_group_a = [row for row in all_rows if row["method"] != "initial" and row["status"] == "completed"]
    contract_audit = bool(
        len(completed_group_a) == len(METHODS) * len(sources)
        and all(row["source_identity_audit"] is True and row["adapter_identity_audit"] is True for row in completed_group_a)
        and all(row["observed_common_initial_mesh_sha256"] == row["common_initial_mesh_sha256"] for row in completed_group_a)
        and all(int(row["view_count"]) == 28 for row in completed_group_a)
        and all(row["unified_metric_audit"] is True for row in completed_group_a)
        and all(
            len(
                {
                    (
                        row["initial_chamfer"],
                        row["initial_p2s"],
                        row["initial_normal_consistency"],
                    )
                    for row in completed_group_a
                    if row["sample_id"] == source["sample_id"]
                }
            )
            == 1
            for source in sources
        )
    )
    summary = {
        "contract_audit": contract_audit,
        "dataset": manifest["source_manifest"],
        "benchmark_manifest": str(args.manifest.resolve()),
        "sample_count": len(sources),
        "sample_ids": [row["sample_id"] for row in sources],
        "common_input_contract": manifest["common_input_contract"],
        "metric_contract": {
            "implementation": "mlr.learned_laplacian.evaluation.evaluate_mesh_geometry",
            "surface_samples": surface_samples,
            "seed": metric_seed,
            "fscore_threshold": fscore_threshold,
            "native_method_metrics_role": "provenance_only",
        },
        "group_a": ["initial", *METHODS],
        "group_b": {
            name: config["methods"][name]["reason"] for name in ("neuralangelo", "matcha")
        },
        "aggregate": aggregates,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    (args.output_dir / "per_sample.json").write_text(json.dumps(all_rows, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "FINAL_REPORT.md").write_text(_report(summary), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--surface-samples", type=int, default=3000)
    parser.add_argument("--metric-seed", type=int, default=7)
    parser.add_argument("--fscore-threshold", type=float, default=0.01)
    result = run(parser.parse_args())
    print(json.dumps({"contract_audit": result["contract_audit"], "sample_count": result["sample_count"], "output": str(Path(result["benchmark_manifest"]).parent)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
