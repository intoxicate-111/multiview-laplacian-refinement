#!/usr/bin/env python3
from __future__ import annotations

"""Aggregate the unequal-view Sofa50 stress test with one geometry evaluator."""

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from mlr.data import Mesh
from mlr.io import load_mesh
from mlr.learned_laplacian.evaluation import evaluate_mesh_geometry
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


EXTERNAL = ("nds", "nvdiffrec", "exmesh")


def _status(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = payload.get("row", payload)
    if not isinstance(row, dict):
        raise ValueError(f"Invalid status payload: {path}")
    if row.get("status") != "completed":
        raise RuntimeError(f"Incomplete result {path}: {row.get('failure_reason', row.get('status'))}")
    return row


def _metrics(mesh_path: Path, gt: Mesh, args: argparse.Namespace) -> dict[str, float]:
    mesh = load_mesh(mesh_path).ensure_normals()
    value = evaluate_mesh_geometry(
        mesh,
        gt,
        surface_samples=args.surface_samples,
        seed=args.metric_seed,
        fscore_threshold=args.fscore_threshold,
    )
    return {
        "cd": float(value["chamfer"]),
        "p2s_p95": float(value["point_to_surface_bidirectional_p95"]),
        "fscore": float(value["fscore"]),
        "normal": float(value["normal_consistency"]),
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(statistics.fmean(float(row[key]) for row in rows))


def _aggregate(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    selected = [row for row in rows if row["method"] == method]
    return {
        "method": method,
        "samples": len(selected),
        "views": int(selected[0]["views"]),
        "cd": _mean(selected, "cd"),
        "p2s_p95": _mean(selected, "p2s_p95"),
        "fscore": _mean(selected, "fscore"),
        "normal": _mean(selected, "normal"),
        "improved_vs_initial": sum(row["cd"] < row["initial_cd"] for row in selected),
        "worsened_vs_initial": sum(row["cd"] > row["initial_cd"] for row in selected),
    }


def _mesh_from_status(root: Path, method: str, sample_id: str) -> tuple[Path, dict[str, Any]]:
    path = root / method / "samples" / sample_id / "status.json"
    row = _status(path)
    mesh = Path(str(row["final_mesh"]))
    if not mesh.is_file():
        raise FileNotFoundError(f"Missing refined mesh: {mesh}")
    return mesh, row


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    sources = {str(row["sample_id"]): dict(row) for row in manifest["samples"]}
    dataset = PreparedMeshDataset.from_manifest(args.manifest, "test")
    if len(dataset) != 25:
        raise ValueError(f"Expected 25 samples, got {len(dataset)}")
    rows: list[dict[str, Any]] = []
    identity_audits = []
    for index, sample_id in enumerate(dataset.sample_ids):
        static = dataset.load_static(index)
        source = sources[sample_id]
        gt = Mesh(
            static["gt_vertices"].detach().cpu().numpy(),
            static["gt_faces"].detach().cpu().numpy(),
        ).ensure_normals()
        initial_path = Path(str(source["common_initial_mesh"]))
        initial = _metrics(initial_path, gt, args)
        sample_rows = []

        ours_path, ours_status = _mesh_from_status(args.old_results_root, "ours", sample_id)
        if int(ours_status["view_count"]) != 28:
            raise RuntimeError(f"{sample_id}: archived Ours is not 28-view")
        sample_rows.append(("ours_28v", 28, ours_path))

        for method in EXTERNAL:
            old_path, old_status = _mesh_from_status(args.old_results_root, method, sample_id)
            if int(old_status["view_count"]) != 28:
                raise RuntimeError(f"{sample_id}: archived {method} is not 28-view")
            sample_rows.append((f"{method}_28v", 28, old_path))
            new_path, new_status = _mesh_from_status(args.external_results_root, method, sample_id)
            if int(new_status["view_count"]) != 56:
                raise RuntimeError(f"{sample_id}: stress {method} is not 56-view")
            if not (
                new_status.get("common_initial_source_identity_audit") is True
                and new_status.get("common_initial_identity_audit") is True
                and str(new_status.get("common_initial_mesh_sha256"))
                == str(source["common_initial_mesh_sha256"])
            ):
                raise RuntimeError(f"{sample_id}: {method} input identity audit failed")
            sample_rows.append((f"{method}_56v", 56, new_path))

        for method, views, mesh_path in sample_rows:
            measured = _metrics(mesh_path, gt, args)
            rows.append(
                {
                    "sample_id": sample_id,
                    "method": method,
                    "views": views,
                    "initial_cd": initial["cd"],
                    **measured,
                    "mesh": str(mesh_path),
                }
            )
        identity_audits.append({"sample_id": sample_id, "passed": True})

    method_order = (
        "ours_28v",
        "nds_28v",
        "nds_56v",
        "nvdiffrec_28v",
        "nvdiffrec_56v",
        "exmesh_28v",
        "exmesh_56v",
    )
    aggregate = [_aggregate(rows, method) for method in method_order]
    by_key = {(row["sample_id"], row["method"]): row for row in rows}
    paired = []
    increments = []
    for method in EXTERNAL:
        differences = [
            by_key[(sample_id, f"{method}_56v")]["cd"]
            - by_key[(sample_id, "ours_28v")]["cd"]
            for sample_id in dataset.sample_ids
        ]
        paired.append(
            {
                "baseline": f"{method}_56v",
                "baseline_minus_ours_cd": float(statistics.fmean(differences)),
                "ours_wins": sum(value > 0 for value in differences),
                "baseline_wins": sum(value < 0 for value in differences),
                "ties": sum(value == 0 for value in differences),
            }
        )
        changes = [
            by_key[(sample_id, f"{method}_56v")]["cd"]
            - by_key[(sample_id, f"{method}_28v")]["cd"]
            for sample_id in dataset.sample_ids
        ]
        increments.append(
            {
                "method": method,
                "cd_56v_minus_28v": float(statistics.fmean(changes)),
                "improved_with_56v": sum(value < 0 for value in changes),
                "worsened_with_56v": sum(value > 0 for value in changes),
                "ties": sum(value == 0 for value in changes),
            }
        )

    summary = {
        "contract_audit": all(row["passed"] for row in identity_audits),
        "classification": "unequal_view_external_stress_not_fairness_replacement",
        "sample_count": len(dataset),
        "ours_views": 28,
        "external_views": 56,
        "aggregate": aggregate,
        "paired_ours28_vs_external56": paired,
        "external_view_increment": increments,
        "metric_protocol": (
            "mlr.learned_laplacian.evaluation.evaluate_mesh_geometry;"
            f"surface_samples={args.surface_samples};seed={args.metric_seed};"
            f"fscore_threshold={args.fscore_threshold};alignment=no_ICP"
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "FINAL_REPORT.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _fmt(value: float) -> str:
    return f"{value:.9g}"


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# Sofa50 56-view external-baseline stress comparison",
        "",
        f"Contract audit: **{str(summary['contract_audit']).lower()}**.",
        "",
        "This is deliberately unequal-view: Ours uses 28 native-1920 views; external methods use 56. It strengthens the baselines and does not replace the 28-vs-28 fairness table.",
        "",
        "| Method | Views | CD | P2S p95 | F-score | Normal | Improved/worsened |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["aggregate"]:
        lines.append(
            f"| {row['method']} | {row['views']} | {_fmt(row['cd'])} | {_fmt(row['p2s_p95'])} | "
            f"{_fmt(row['fscore'])} | {_fmt(row['normal'])} | "
            f"{row['improved_vs_initial']}/{row['worsened_vs_initial']} |"
        )
    lines.extend(
        [
            "",
            "## Ours-28V versus external-56V",
            "",
            "Positive baseline-minus-Ours CD means Ours is better.",
            "",
            "| Baseline | Baseline minus Ours CD | Ours wins | Baseline wins | Ties |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["paired_ours28_vs_external56"]:
        lines.append(
            f"| {row['baseline']} | {_fmt(row['baseline_minus_ours_cd'])} | "
            f"{row['ours_wins']} | {row['baseline_wins']} | {row['ties']} |"
        )
    lines.extend(
        [
            "",
            "## External-method gain from 28 to 56 views",
            "",
            "Negative 56V-minus-28V CD means additional views helped.",
            "",
            "| Method | 56V minus 28V CD | Improved | Worsened | Ties |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["external_view_increment"]:
        lines.append(
            f"| {row['method']} | {_fmt(row['cd_56v_minus_28v'])} | "
            f"{row['improved_with_56v']} | {row['worsened_with_56v']} | {row['ties']} |"
        )
    lines.extend(["", f"Metric protocol: `{summary['metric_protocol']}`.", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--external-results-root", type=Path, required=True)
    parser.add_argument("--old-results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--surface-samples", type=int, default=3000)
    parser.add_argument("--metric-seed", type=int, default=7)
    parser.add_argument("--fscore-threshold", type=float, default=0.01)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
