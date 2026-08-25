#!/usr/bin/env python3
from __future__ import annotations

"""Audit deterministic per-component hard anchors for all Sofa50 v2 meshes."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from mlr.learned_laplacian.hard_anchor_sparse_recovery import (
    deterministic_component_anchor_indices,
)
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), split)
        for index in range(len(dataset)):
            sample = dataset.load_static(index)
            vertices = int(torch.as_tensor(sample["vertices"]).shape[0])
            anchors = deterministic_component_anchor_indices(
                torch.as_tensor(sample["edge_index"]), vertices
            )
            if anchors.numel() < 1 or int(anchors[0]) != 0:
                raise RuntimeError(
                    f"Deterministic anchor contract failed for {sample['sample_id']}."
                )
            rows.append(
                {
                    "split": split,
                    "sample_id": str(sample["sample_id"]),
                    "vertices": vertices,
                    "faces": int(torch.as_tensor(sample["faces"]).shape[0]),
                    "connected_components": int(anchors.numel()),
                    "hard_anchor_count": int(anchors.numel()),
                    "anchor_indices": ";".join(str(int(value)) for value in anchors),
                    "multiple_components": bool(anchors.numel() > 1),
                    "anchor_zero_for_first_component": int(anchors[0]) == 0,
                }
            )
    fields = list(rows[0])
    with (output / "per_sample.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    counts = [int(row["connected_components"]) for row in rows]
    by_split = {
        split: {
            "meshes": sum(row["split"] == split for row in rows),
            "multi_component_meshes": sum(
                row["split"] == split and bool(row["multiple_components"])
                for row in rows
            ),
            "hard_anchors": sum(
                int(row["hard_anchor_count"])
                for row in rows
                if row["split"] == split
            ),
        }
        for split in ("train", "validation", "test")
    }
    summary = {
        "contract_audit": bool(
            len(rows) == 500
            and all(int(row["hard_anchor_count"]) == int(row["connected_components"]) for row in rows)
            and all(bool(row["anchor_zero_for_first_component"]) for row in rows)
        ),
        "manifest": str(args.manifest.resolve()),
        "meshes": len(rows),
        "meshes_with_multiple_components": sum(bool(row["multiple_components"]) for row in rows),
        "connected_components_minimum": min(counts),
        "connected_components_mean": sum(counts) / len(counts),
        "connected_components_maximum": max(counts),
        "hard_anchors_total": sum(counts),
        "anchor_rule": "lowest_global_vertex_index_per_undirected_component",
        "anchor_uses_gt": False,
        "connectivity_modified": False,
        "by_split": by_split,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["contract_audit"]:
        raise RuntimeError("Hard-anchor connectivity contract failed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
