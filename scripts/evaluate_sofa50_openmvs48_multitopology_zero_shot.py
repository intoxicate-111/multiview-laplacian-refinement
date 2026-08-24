#!/usr/bin/env python3
from __future__ import annotations

"""Compare old/new direct-raw HF checkpoints on existing OpenMVS coarse meshes."""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from evaluate_sofa50_multitopology_rawlap import ARMS, load_spec
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.synthetic_current_h2_ablation import _infer_one, _recover_raw_one


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    manifest = args.manifest.resolve()
    output = args.output_dir.resolve()
    dataset = PreparedMeshDataset.from_manifest(manifest, "test")
    specs = {
        ARMS[0]: load_spec(args.old_run.resolve(), device),
        ARMS[1]: load_spec(args.new_run.resolve(), device),
    }
    configs = {arm: spec["config"] for arm, spec in specs.items()}
    preflight = {
        "optimizer_steps": {arm: spec["optimizer_steps"] for arm, spec in specs.items()},
        "target_modes": {arm: config.get("target_mode") for arm, config in configs.items()},
        "feature_modes": {
            arm: config.get("image_encoder", {}).get("feature_construction", {}).get("mode")
            for arm, config in configs.items()
        },
        "same_architecture": all(
            configs[ARMS[0]].get(key) == configs[ARMS[1]].get(key)
            for key in ("model", "image_encoder", "input_mode")
        ),
        "same_recovery": configs[ARMS[0]].get("recovery") == configs[ARMS[1]].get("recovery"),
        "fine_tuning_used": False,
        "target_placeholders_used": False,
        "gt_used_after_prediction_only": True,
    }
    preflight["passed"] = bool(
        all(value == 20_000 for value in preflight["optimizer_steps"].values())
        and all(value == "raw_laplacian" for value in preflight["target_modes"].values())
        and all(value == "original_plus_high_frequency" for value in preflight["feature_modes"].values())
        and preflight["same_architecture"]
        and preflight["same_recovery"]
    )
    if not preflight["passed"]:
        raise RuntimeError(f"Preflight failed: {preflight}")
    rows = []
    for index in range(len(dataset)):
        if index % args.shard_count != args.shard_index:
            continue
        static = dataset.load_static(index)
        metadata = dict(static.get("metadata", {}))
        for arm, spec in specs.items():
            torch.cuda.synchronize()
            start = time.perf_counter()
            values = _infer_one(dataset, index, spec, device, current_faces=static["faces"])
            torch.cuda.synchronize()
            inference_seconds = time.perf_counter() - start
            recovery, _ = _recover_raw_one(
                static,
                values["prediction_raw"],
                values["prediction_normalized"],
                values["confidence"],
                args.output_dir.resolve() / "reconstruction" / arm / str(static["sample_id"]),
                spec["config"],
            )
            rows.append(
                {
                    "arm": arm,
                    "sample_id": str(static["sample_id"]),
                    "source_split": metadata.get("source_split"),
                    "vertices": int(static["vertices"].shape[0]),
                    "faces": int(static["faces"].shape[0]),
                    "inference_seconds": inference_seconds,
                    **recovery,
                }
            )
            print(
                f"{arm} {static['sample_id']} chamfer={recovery['reconstruction_chamfer']:.8g}",
                flush=True,
            )
            del values
            torch.cuda.empty_cache()
    write_json(
        output / "shards" / f"shard_{args.shard_index:02d}.json",
        {
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "preflight": preflight,
            "rows": rows,
        },
    )


def mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def merge(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    payloads = [read_json(output / "shards" / f"shard_{index:02d}.json") for index in range(args.shard_count)]
    rows = [row for payload in payloads for row in payload["rows"]]
    aggregates = []
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        aggregates.append(
            {
                "arm": arm,
                "samples": len(selected),
                "initial_chamfer": mean(selected, "initial_chamfer"),
                "chamfer": mean(selected, "reconstruction_chamfer"),
                "p2s": mean(selected, "reconstruction_point_to_surface"),
                "normal_consistency": mean(selected, "reconstruction_normal_consistency"),
                "introduced_flipped_faces": int(sum(int(row["introduced_flipped_faces"]) for row in selected)),
                "new_degenerate_faces": int(sum(int(row["new_degenerate_faces"]) for row in selected)),
                "improved_over_initial": int(sum(bool(row["improved_over_initial"]) for row in selected)),
                "inference_seconds": mean(selected, "inference_seconds"),
            }
        )
    by_arm = {
        arm: {str(row["sample_id"]): row for row in rows if row["arm"] == arm}
        for arm in ARMS
    }
    paired = []
    for sample_id in sorted(by_arm[ARMS[1]]):
        old, new = by_arm[ARMS[0]][sample_id], by_arm[ARMS[1]][sample_id]
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
    audit = {
        "passed": all(bool(payload["preflight"]["passed"]) for payload in payloads),
        "same_prepared_samples": len(by_arm[ARMS[0]]) == len(by_arm[ARMS[1]]),
        "same_initial_mesh_visibility_observations": True,
        "fine_tuning_used": False,
        "target_placeholders_used": False,
        "gt_differential_transfer_used": False,
        "gt_usage": "post_prediction geometry metrics only",
    }
    write_json(output / "summary.json", {"contract_audit": audit, "aggregate": aggregates})
    write_json(output / "contract_audit.json", audit)
    write_csv(output / "per_sample.csv", rows)
    write_csv(output / "paired_old_vs_new.csv", paired)
    write_csv(output / "aggregate.csv", aggregates)
    lines = [
        "# Sofa50 OpenMVS48 real-coarse zero-shot comparison",
        "",
        f"Contract audit: **{str(audit['passed']).lower()}**.",
        "",
        "The 48-view OpenMVS meshes are the common initial geometry; both learned arms use the same original 14-view Sofa RGB/cameras. No fine-tuning or GT differential target is used.",
        "",
        "| Arm | Chamfer | P2S | Normal | Flips | New degenerates | Improved |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(f"| {row['arm']} | {row['chamfer']:.9g} | {row['p2s']:.9g} | {row['normal_consistency']:.9g} | {row['introduced_flipped_faces']} | {row['new_degenerate_faces']} | {row['improved_over_initial']}/{row['samples']} |")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"contract_audit": audit, "aggregate": aggregates}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--old-run", type=Path)
    parser.add_argument("--new-run", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    if args.merge_only:
        merge(args)
    else:
        if args.old_run is None or args.new_run is None:
            parser.error("--old-run and --new-run are required")
        evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
