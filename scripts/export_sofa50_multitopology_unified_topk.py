#!/usr/bin/env python3
from __future__ import annotations

"""Export top unified-Chamfer Sofa50 examples with all comparison meshes."""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mlr.data import Mesh
from mlr.io import load_mesh, save_mesh
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


OLD_ARM = "old_960_HF"
NEW_ARM = "new_multitopology_rawlap"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gt_mesh(static: Mapping[str, Any]) -> Mesh:
    for vertex_key, face_key in (
        ("gt_vertices", "gt_faces"),
        ("clean_reference_vertices", "clean_reference_faces"),
    ):
        if static.get(vertex_key) is not None and static.get(face_key) is not None:
            return Mesh(
                static[vertex_key].detach().cpu().numpy(),
                static[face_key].detach().cpu().numpy(),
            ).ensure_normals()
    raise KeyError("Prepared sample has no GT/clean-reference mesh.")


def _mesh_info(path: Path, root: Path) -> dict[str, Any]:
    mesh = load_mesh(path)
    return {
        "path": str(path.relative_to(root)),
        "sha256": _sha256(path),
        "vertices": mesh.num_vertices,
        "faces": mesh.num_faces,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--evaluation-dir", required=True, type=Path)
    parser.add_argument("--source-evaluation-dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--label", default="Sofa50 new-test50")
    args = parser.parse_args()

    evaluation = args.evaluation_dir.resolve()
    source_evaluation = (
        args.source_evaluation_dir.resolve()
        if args.source_evaluation_dir is not None
        else evaluation.parent.parent / "evaluation" / "in_domain"
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(evaluation / "per_sample.csv")
    by_key = {(row["sample_id"], row["arm"]): row for row in rows}
    candidates = [row for row in rows if row["arm"] == NEW_ARM]
    candidates.sort(
        key=lambda row: float(row["initial_chamfer"])
        - float(row["reconstruction_chamfer"]),
        reverse=True,
    )
    selected = candidates[: args.top_k]
    dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), "test")
    index_by_id = {sample_id: index for index, sample_id in enumerate(dataset.sample_ids)}

    selection_rows: list[dict[str, Any]] = []
    package: list[dict[str, Any]] = []
    for rank, new_row in enumerate(selected, 1):
        sample_id = new_row["sample_id"]
        old_row = by_key[(sample_id, OLD_ARM)]
        sample_dir = output / f"{rank:02d}_{sample_id}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        old_source = source_evaluation / "reconstruction" / OLD_ARM / sample_id
        new_source = source_evaluation / "reconstruction" / NEW_ARM / sample_id
        coarse_old = load_mesh(old_source / "coarse.obj")
        coarse_new = load_mesh(new_source / "coarse.obj")
        if not (
            np.array_equal(coarse_old.vertices, coarse_new.vertices)
            and np.array_equal(coarse_old.faces, coarse_new.faces)
        ):
            raise RuntimeError(f"Common initial mesh mismatch for {sample_id}.")
        static = dataset.load_static(index_by_id[sample_id])
        save_mesh(_gt_mesh(static), sample_dir / "gt.obj")
        shutil.copy2(old_source / "coarse.obj", sample_dir / "coarse.obj")
        shutil.copy2(
            old_source / "predicted_refined.obj",
            sample_dir / "old_960_HF_refined.obj",
        )
        shutil.copy2(
            new_source / "predicted_refined.obj",
            sample_dir / "new_multitopology_refined.obj",
        )
        improvement = float(new_row["initial_chamfer"]) - float(
            new_row["reconstruction_chamfer"]
        )
        relative = 100.0 * improvement / max(float(new_row["initial_chamfer"]), 1e-12)
        selection = {
            "rank": rank,
            "sample_id": sample_id,
            "variant": new_row.get("variant"),
            "initial_chamfer": float(new_row["initial_chamfer"]),
            "old_960_HF_chamfer": float(old_row["reconstruction_chamfer"]),
            "new_multitopology_chamfer": float(new_row["reconstruction_chamfer"]),
            "new_absolute_improvement_over_initial": improvement,
            "new_relative_improvement_over_initial_percent": relative,
            "new_minus_old_chamfer": float(new_row["reconstruction_chamfer"])
            - float(old_row["reconstruction_chamfer"]),
            "metric_protocol": new_row["metric_protocol"],
        }
        selection_rows.append(selection)
        files = {
            name: _mesh_info(sample_dir / name, output)
            for name in (
                "gt.obj",
                "coarse.obj",
                "old_960_HF_refined.obj",
                "new_multitopology_refined.obj",
            )
        }
        detail = {"selection": selection, "old_metrics": old_row, "new_metrics": new_row, "files": files}
        (sample_dir / "metrics.json").write_text(
            json.dumps(detail, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        package.append(detail)

    _write_csv(output / "selection.csv", selection_rows)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "selection_rule": "largest unified-v2 initial_chamfer - new_multitopology_refined_chamfer",
                "label": args.label,
                "top_k": args.top_k,
                "contract_audit": True,
                "samples": package,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# {args.label} top unified-v2 improvements",
        "",
        "Selection rule: the ten largest absolute reductions from common-initial Chamfer to the new multi-topology model, using the audited unified-v2 surface evaluator.",
        "",
        "Each sample directory contains `gt.obj`, `coarse.obj`, `old_960_HF_refined.obj`, `new_multitopology_refined.obj`, and `metrics.json`.",
        "",
        "| Rank | Sample | Variant | Initial | Old HF | New | New improvement |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in selection_rows:
        lines.append(
            f"| {row['rank']} | `{row['sample_id']}` | {row['variant']} | "
            f"{row['initial_chamfer']:.9g} | {row['old_960_HF_chamfer']:.9g} | "
            f"{row['new_multitopology_chamfer']:.9g} | "
            f"{row['new_relative_improvement_over_initial_percent']:.2f}% |"
        )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "samples": len(selection_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
