#!/usr/bin/env python3
from __future__ import annotations

"""Export GT/Ours/NDS meshes with the largest paired refined-Chamfer gaps."""

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlr.data import Mesh
from mlr.io import load_mesh, save_mesh
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


def _read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["sample_id"]: dict(row) for row in csv.DictReader(handle)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    ours = _read_rows(args.results_root / "ours/per_sample.csv")
    nds = _read_rows(args.results_root / "nds/per_sample.csv")
    if set(ours) != set(nds):
        raise ValueError("Ours and NDS sample IDs do not match.")
    candidates = []
    for sample_id in ours:
        ours_chamfer = float(ours[sample_id]["refined_chamfer"])
        nds_chamfer = float(nds[sample_id]["refined_chamfer"])
        face_count = int(ours[sample_id]["face_count"])
        if (
            face_count >= args.minimum_face_count
            and face_count <= args.maximum_face_count
            and math.isfinite(ours_chamfer)
            and math.isfinite(nds_chamfer)
        ):
            score = (
                nds_chamfer - ours_chamfer
                if args.selection_metric == "chamfer_gap"
                else ours_chamfer
            )
            candidates.append((score, sample_id))
    selected = sorted(
        candidates,
        reverse=args.selection_metric == "chamfer_gap",
    )[: args.count]
    if len(selected) != args.count:
        raise ValueError(f"Only {len(selected)} finite paired samples are available.")

    dataset = PreparedMeshDataset.from_manifest(args.manifest, "test")
    index_by_id = {sample_id: index for index, sample_id in enumerate(dataset.sample_ids)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for rank, (_, sample_id) in enumerate(selected, start=1):
        static = dataset.load_static(index_by_id[sample_id])
        sample_dir = args.output_dir / f"rank_{rank:02d}_{sample_id}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        outputs = {
            "gt": sample_dir / "GT.obj",
            "coarse": sample_dir / "COARSE.obj",
            "ours_refined": sample_dir / "OURS_REFINED.obj",
            "nds_refined": sample_dir / "NDS_REFINED.obj",
        }
        gt = Mesh(
            static["gt_vertices"].detach().cpu().numpy(),
            static["gt_faces"].detach().cpu().numpy(),
        ).ensure_normals()
        save_mesh(gt, outputs["gt"])
        coarse = Mesh(
            static["vertices"].detach().cpu().numpy(),
            static["faces"].detach().cpu().numpy(),
        ).ensure_normals()
        save_mesh(coarse, outputs["coarse"])
        shutil.copy2(Path(ours[sample_id]["final_mesh"]), outputs["ours_refined"])
        shutil.copy2(Path(nds[sample_id]["final_mesh"]), outputs["nds_refined"])
        meshes = {name: load_mesh(path) for name, path in outputs.items()}
        gap = float(nds[sample_id]["refined_chamfer"]) - float(
            ours[sample_id]["refined_chamfer"]
        )
        record = {
            "rank": rank,
            "sample_id": sample_id,
            "selection_metric": args.selection_metric,
            "chamfer_gap": gap,
            "initial_face_count": int(ours[sample_id]["initial_face_count"]),
            "initial_chamfer": float(ours[sample_id]["initial_chamfer"]),
            "ours_refined_chamfer": float(ours[sample_id]["refined_chamfer"]),
            "nds_refined_chamfer": float(nds[sample_id]["refined_chamfer"]),
            "ours_refined_p2s_p95": float(ours[sample_id]["refined_p2s_p95"]),
            "nds_refined_p2s_p95": float(nds[sample_id]["refined_p2s_p95"]),
            "ours_normal_consistency": float(
                ours[sample_id]["refined_normal_consistency"]
            ),
            "nds_normal_consistency": float(
                nds[sample_id]["refined_normal_consistency"]
            ),
            "meshes": {
                name: {
                    "path": str(outputs[name].resolve()),
                    "sha256": _sha256(outputs[name]),
                    "vertices": meshes[name].num_vertices,
                    "faces": meshes[name].num_faces,
                }
                for name in outputs
            },
        }
        (sample_dir / "metrics.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )
        records.append(record)

    (args.output_dir / "selection.json").write_text(
        json.dumps(
            {
                "definition": (
                    "Finite paired samples selected after the configured common-topology "
                    "face-count interval, then ranked by the configured metric. "
                    "chamfer_gap is descending; ours_refined_chamfer is ascending."
                ),
                "count": args.count,
                "minimum_face_count_inclusive": args.minimum_face_count,
                "maximum_face_count_inclusive": args.maximum_face_count,
                "selection_metric": args.selection_metric,
                "records": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    flat = [
        {key: value for key, value in record.items() if key != "meshes"}
        for record in records
    ]
    with (args.output_dir / "selection.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    lines = [
        f"# Ours vs NDS: {args.count} selected paired mesh comparisons",
        "",
        "Each directory contains `GT.obj`, `COARSE.obj`, `OURS_REFINED.obj`, and "
        "`NDS_REFINED.obj` in the same evaluation coordinate system.",
        "",
        f"Minimum common-topology face count (inclusive): {args.minimum_face_count}.",
        f"Maximum common-topology face count (inclusive): {args.maximum_face_count}.",
        f"Selection metric: {args.selection_metric}.",
        "",
        "| Rank | Sample | Initial | Ours | NDS | NDS - Ours |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            f"| {record['rank']} | `{record['sample_id']}` | "
            f"{record['initial_chamfer']:.9g} | "
            f"{record['ours_refined_chamfer']:.9g} | "
            f"{record['nds_refined_chamfer']:.9g} | {record['chamfer_gap']:.9g} |"
        )
    (args.output_dir / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--minimum-face-count", type=int, default=0)
    parser.add_argument("--maximum-face-count", type=int, default=2**63 - 1)
    parser.add_argument(
        "--selection-metric",
        choices=("chamfer_gap", "ours_refined_chamfer"),
        default="chamfer_gap",
    )
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
