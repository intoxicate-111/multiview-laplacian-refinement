from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mlr.io import load_mesh
from mlr.learned_laplacian.canonical_experiment import _topology_change
from mlr.learned_laplacian.evaluation import reconstruct_and_evaluate
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate exact current-graph proxy targets through the recovery solver."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument(
        "--one-variant-per-object",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = PreparedMeshDataset.from_manifest(manifest, args.split)
    config = _read_json(args.config.resolve())
    recovery = dict(config["recovery"])
    recovery.update(
        {
            "dense_vertex_limit": 5000,
            "chamfer_samples": 3000,
            "metric_seed": 7,
            "evaluate_oracle": False,
        }
    )
    epsilon = float(config["target_scaling"]["epsilon"])
    rows = []
    seen_objects: set[str] = set()
    for index in range(len(dataset)):
        sample = dataset.load_static(index)
        metadata = dict(sample.get("metadata", {}))
        object_id = str(metadata.get("object_id", sample["sample_id"]))
        if args.one_variant_per_object and object_id in seen_objects:
            continue
        seen_objects.add(object_id)
        target_raw = sample["raw_laplacian_target"]
        target_normalized = sample["normalized_laplacian_target"]
        sample_dir = output / str(sample["sample_id"])
        result = reconstruct_and_evaluate(
            sample,
            target_raw,
            sample_dir,
            recovery,
            normalized_prediction=target_normalized,
            edge_scale_epsilon=epsilon,
            laplacian_weight=np.ones(len(target_raw), dtype=np.float64),
            unseen_anchor_weight=0.0,
            evaluate_laplacian_prediction=True,
            evaluate_initial_geometry=True,
            solver_confidence=np.ones(len(target_raw), dtype=np.float64),
        )
        recovered = load_mesh(sample_dir / "predicted_refined.obj")
        initial = sample["vertices"].numpy()
        faces = sample["faces"].numpy()
        topology = _topology_change(initial, recovered.vertices, faces)
        initial_geometry = result["geometry"]["coarse"]
        recovered_geometry = result["geometry"]["predicted"]
        initial_chamfer = float(initial_geometry["chamfer"])
        recovered_chamfer = float(recovered_geometry["chamfer"])
        row = {
            "sample_id": str(sample["sample_id"]),
            "object_id": object_id,
            "variant_index": int(metadata.get("variant_index", -1)),
            "initial_chamfer": initial_chamfer,
            "oracle_recovered_chamfer": recovered_chamfer,
            "initial_point_to_surface": float(
                initial_geometry["point_to_surface_bidirectional_mean"]
            ),
            "oracle_recovered_point_to_surface": float(
                recovered_geometry["point_to_surface_bidirectional_mean"]
            ),
            "initial_normal_consistency": float(initial_geometry["normal_consistency"]),
            "oracle_recovered_normal_consistency": float(
                recovered_geometry["normal_consistency"]
            ),
            "introduced_flipped_faces": int(topology["introduced_flips"]),
            "new_degenerate_faces": int(topology["new_degeneracies"]),
            "chamfer_improved": bool(recovered_chamfer < initial_chamfer),
            "all_finite": bool(result["reconstruction"]["all_finite"]),
        }
        rows.append(row)
        print(
            f"{row['sample_id']}: initial={initial_chamfer:.8g} "
            f"oracle={recovered_chamfer:.8g} improved={row['chamfer_improved']}",
            flush=True,
        )
    if not rows:
        raise RuntimeError("No samples selected for oracle recovery validation.")
    summary = {
        "manifest": str(manifest),
        "split": args.split,
        "one_variant_per_object": bool(args.one_variant_per_object),
        "sample_count": len(rows),
        "all_finite": all(row["all_finite"] for row in rows),
        "all_chamfer_improved": all(row["chamfer_improved"] for row in rows),
        "improved_count": sum(row["chamfer_improved"] for row in rows),
        "introduced_flipped_faces": sum(row["introduced_flipped_faces"] for row in rows),
        "mean_initial_chamfer": float(np.mean([row["initial_chamfer"] for row in rows])),
        "mean_oracle_recovered_chamfer": float(
            np.mean([row["oracle_recovered_chamfer"] for row in rows])
        ),
        "per_sample": rows,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "per_sample"}, indent=2))
    if not summary["all_finite"] or not summary["all_chamfer_improved"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
