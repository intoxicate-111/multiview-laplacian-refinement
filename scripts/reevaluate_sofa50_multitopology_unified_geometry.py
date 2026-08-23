#!/usr/bin/env python3
from __future__ import annotations

"""Re-evaluate archived Sofa50 refinements with the unified surface evaluator."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mlr.data import Mesh
from mlr.io import load_mesh
from mlr.learned_laplacian.evaluation import evaluate_mesh_geometry
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


ARMS = ("old_960_HF", "new_multitopology_rawlap")
PROTOCOL = (
    "mlr.learned_laplacian.evaluation.evaluate_mesh_geometry;"
    "area_weighted_triangle_surface_sampling;"
    "bidirectional_sampled_surface_to_exact_triangle_surface;"
    "surface_samples=3000;seed=7;fscore_threshold=0.01;"
    "alignment=shared_prepared_coordinate_frame_no_ICP"
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _gt_mesh(static: Mapping[str, Any]) -> Mesh:
    for vertex_key, face_key in (
        ("gt_vertices", "gt_faces"),
        ("clean_reference_vertices", "clean_reference_faces"),
    ):
        if static.get(vertex_key) is not None and static.get(face_key) is not None:
            vertices = static[vertex_key].detach().cpu().numpy()
            faces = static[face_key].detach().cpu().numpy()
            return Mesh(vertices, faces).ensure_normals()
    raise KeyError("Prepared sample has neither GT nor clean-reference mesh tensors.")


def _geometry(mesh: Mesh, gt: Mesh) -> dict[str, Any]:
    return evaluate_mesh_geometry(
        mesh.ensure_normals(), gt, surface_samples=3000, seed=7, fscore_threshold=0.01
    )


def _find_source_rows(source: Path, mode: str) -> list[dict[str, str]]:
    name = "per_sample.csv" if mode == "openmvs" else "recovery_per_sample.csv"
    path = source / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return read_csv(path)


def evaluate_shard(args: argparse.Namespace) -> None:
    arms = (args.old_arm_name, args.new_arm_name)
    if len(set(arms)) != 2:
        raise ValueError("Evaluation arm names must be distinct.")
    manifest = args.manifest.resolve()
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    dataset = PreparedMeshDataset.from_manifest(manifest, "test")
    source_rows = _find_source_rows(source, args.mode)
    source_by_key = {
        (str(row["sample_id"]), str(row["arm"])): row for row in source_rows
    }
    rows: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        if index % args.shard_count != args.shard_index:
            continue
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        gt = _gt_mesh(static)
        arm_meshes: dict[str, tuple[Mesh, Mesh]] = {}
        for arm in arms:
            mesh_dir = source / "reconstruction" / arm / sample_id
            arm_meshes[arm] = (
                load_mesh(mesh_dir / "coarse.obj"),
                load_mesh(mesh_dir / "predicted_refined.obj"),
            )
        coarse_old, coarse_new = arm_meshes[arms[0]][0], arm_meshes[arms[1]][0]
        common_initial_exact = bool(
            np.array_equal(coarse_old.faces, coarse_new.faces)
            and np.array_equal(coarse_old.vertices, coarse_new.vertices)
        )
        if not common_initial_exact:
            raise RuntimeError(f"Common-initial identity failed for {sample_id}.")
        initial = _geometry(coarse_old, gt)
        for arm in arms:
            source_row = source_by_key.get((sample_id, arm))
            if source_row is None:
                raise KeyError(f"Missing source metric row for {sample_id}/{arm}.")
            final = _geometry(arm_meshes[arm][1], gt)
            row: dict[str, Any] = {
                "sample_id": sample_id,
                "arm": arm,
                "metric_protocol": PROTOCOL,
                "unified_metric_audit": True,
                "coordinate_alignment": "shared_prepared_coordinate_frame; identity; no ICP",
                "common_initial_mesh_exact": common_initial_exact,
                "initial_chamfer": initial["chamfer"],
                "reconstruction_chamfer": final["chamfer"],
                "initial_point_to_surface": initial["point_to_surface_bidirectional_mean"],
                "reconstruction_point_to_surface": final["point_to_surface_bidirectional_mean"],
                "initial_p2s_p95": initial["point_to_surface_bidirectional_p95"],
                "reconstruction_p2s_p95": final["point_to_surface_bidirectional_p95"],
                "initial_fscore": initial["fscore"],
                "reconstruction_fscore": final["fscore"],
                "initial_normal_consistency": initial["normal_consistency"],
                "reconstruction_normal_consistency": final["normal_consistency"],
                "improved_over_initial": final["chamfer"] < initial["chamfer"],
                "forward_engine": final["forward_engine"],
                "reverse_engine": final["reverse_engine"],
            }
            for field in (
                "object_id",
                "variant",
                "source_split",
                "vertices",
                "faces",
                "introduced_flipped_faces",
                "new_degenerate_faces",
                "inference_seconds",
                "recovery_seconds",
            ):
                if source_row.get(field, "") != "":
                    row[field] = source_row[field]
            rows.append(row)
            print(
                f"{sample_id} {arm} unified_chamfer={final['chamfer']:.9g}", flush=True
            )
    write_json(
        output / "shards" / f"shard_{args.shard_index:02d}.json",
        {
            "arms": list(arms),
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "manifest": str(manifest),
            "source_dir": str(source),
            "mode": args.mode,
            "metric_protocol": PROTOCOL,
            "rows": rows,
        },
    )


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def merge(args: argparse.Namespace) -> None:
    arms = (args.old_arm_name, args.new_arm_name)
    if len(set(arms)) != 2:
        raise ValueError("Evaluation arm names must be distinct.")
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    payloads = [
        read_json(output / "shards" / f"shard_{index:02d}.json")
        for index in range(args.shard_count)
    ]
    if any(tuple(payload.get("arms", ARMS)) != arms for payload in payloads):
        raise RuntimeError("Shard evaluation arms differ from merge arms.")
    rows = [row for payload in payloads for row in payload["rows"]]
    expected = 2 * len(PreparedMeshDataset.from_manifest(args.manifest.resolve(), "test"))
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} unified rows, found {len(rows)}.")
    if not all(
        row["metric_protocol"] == PROTOCOL
        and bool(row["unified_metric_audit"])
        and bool(row["common_initial_mesh_exact"])
        for row in rows
    ):
        raise RuntimeError("Unified geometry contract audit failed.")
    aggregates: list[dict[str, Any]] = []
    for arm in arms:
        selected = [row for row in rows if row["arm"] == arm]
        aggregates.append(
            {
                "arm": arm,
                "samples": len(selected),
                "initial_chamfer": _mean(selected, "initial_chamfer"),
                "reconstruction_chamfer": _mean(selected, "reconstruction_chamfer"),
                "chamfer": _mean(selected, "reconstruction_chamfer"),
                "reconstruction_point_to_surface": _mean(selected, "reconstruction_point_to_surface"),
                "p2s": _mean(selected, "reconstruction_point_to_surface"),
                "reconstruction_p2s_p95": _mean(selected, "reconstruction_p2s_p95"),
                "reconstruction_fscore": _mean(selected, "reconstruction_fscore"),
                "reconstruction_normal_consistency": _mean(selected, "reconstruction_normal_consistency"),
                "normal_consistency": _mean(selected, "reconstruction_normal_consistency"),
                "introduced_flipped_faces": int(sum(int(row.get("introduced_flipped_faces", 0)) for row in selected)),
                "new_degenerate_faces": int(sum(int(row.get("new_degenerate_faces", 0)) for row in selected)),
                "improved_over_initial": int(sum(bool(row["improved_over_initial"]) for row in selected)),
            }
        )
    by_arm = {
        arm: {str(row["sample_id"]): row for row in rows if row["arm"] == arm}
        for arm in arms
    }
    paired: list[dict[str, Any]] = []
    for sample_id in sorted(by_arm[arms[0]]):
        old, new = by_arm[arms[0]][sample_id], by_arm[arms[1]][sample_id]
        paired.append(
            {
                "sample_id": sample_id,
                "old_chamfer": old["reconstruction_chamfer"],
                "new_chamfer": new["reconstruction_chamfer"],
                "new_lower_chamfer": float(new["reconstruction_chamfer"]) < float(old["reconstruction_chamfer"]),
                "old_p2s": old["reconstruction_point_to_surface"],
                "new_p2s": new["reconstruction_point_to_surface"],
                "new_lower_p2s": float(new["reconstruction_point_to_surface"]) < float(old["reconstruction_point_to_surface"]),
                "old_normal": old["reconstruction_normal_consistency"],
                "new_normal": new["reconstruction_normal_consistency"],
                "new_higher_normal": float(new["reconstruction_normal_consistency"]) > float(old["reconstruction_normal_consistency"]),
            }
        )
    source_summary = read_json(source / "summary.json")
    audit = {
        "passed": True,
        "unified_metric_audit": True,
        "same_prepared_samples": set(by_arm[arms[0]]) == set(by_arm[arms[1]]),
        "common_initial_mesh_exact_all": True,
        "coordinate_alignment": "shared_prepared_coordinate_frame; identity; no ICP",
        "legacy_geometry_metrics_used_for_primary_result": False,
        "source_contract_audit": source_summary.get("contract_audit"),
    }
    if args.mode == "openmvs":
        summary = {
            "arms": list(arms),
            "contract_audit": audit,
            "metric_protocol": PROTOCOL,
            "aggregate": aggregates,
        }
    else:
        summary = {
            "arms": list(arms),
            "contract_audit": audit,
            "metric_protocol": PROTOCOL,
            "prediction": source_summary["prediction"],
            "recovery": aggregates,
            "paired_sample_count": len(paired),
        }
    write_json(output / "summary.json", summary)
    write_json(output / "contract_audit.json", audit)
    write_csv(output / "per_sample.csv", rows)
    write_csv(output / "paired_old_vs_new.csv", paired)
    write_csv(output / "aggregate.csv", aggregates)
    lines = [
        "# Sofa50 unified-v2 geometry re-evaluation",
        "",
        "Contract audit: **true**.",
        "",
        "Primary geometry metrics use area-weighted surface sampling and exact bidirectional point-to-triangle-surface distances in the shared prepared coordinate frame. No ICP or test-time alignment was applied. Legacy vertex-sampled Chamfer values are excluded.",
        "",
        f"Metric protocol: `{PROTOCOL}`",
        "",
        "| Arm | Initial Chamfer | Refined Chamfer | P2S | P2S p95 | F-score | Normal | Improved |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            f"| {row['arm']} | {row['initial_chamfer']:.9g} | {row['reconstruction_chamfer']:.9g} | "
            f"{row['reconstruction_point_to_surface']:.9g} | {row['reconstruction_p2s_p95']:.9g} | "
            f"{row['reconstruction_fscore']:.9g} | {row['reconstruction_normal_consistency']:.9g} | "
            f"{row['improved_over_initial']}/{row['samples']} |"
        )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=("standard", "openmvs"), default="standard")
    parser.add_argument("--old-arm-name", default=ARMS[0])
    parser.add_argument("--new-arm-name", default=ARMS[1])
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    if args.merge_only:
        merge(args)
    else:
        evaluate_shard(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
