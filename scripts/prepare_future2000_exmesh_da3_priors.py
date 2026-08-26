#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


def object_id_from_sample_id(sample_id: str) -> str:
    head, separator, variant = sample_id.rpartition("__v")
    if not separator or not head or len(variant) != 2 or not variant.isdigit():
        raise ValueError(f"Unexpected Future2000 sample_id: {sample_id!r}")
    return head


def representative_indices(dataset: PreparedMeshDataset) -> list[tuple[str, int]]:
    groups: dict[str, list[int]] = {}
    for index, sample_id in enumerate(dataset.sample_ids):
        groups.setdefault(object_id_from_sample_id(sample_id), []).append(index)
    malformed = {key: values for key, values in groups.items() if len(values) != 5}
    if malformed:
        raise ValueError(f"Expected five variants per test object, found {malformed}")
    return [(object_id, indices[0]) for object_id, indices in sorted(groups.items())]


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count).")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    prior_config = config["methods"]["exmesh"]["depth_prior"]
    checkout = _git_commit(args.da3_root)
    if checkout != prior_config["commit"]:
        raise ValueError(
            f"Depth Anything 3 checkout {checkout} does not match {prior_config['commit']}."
        )
    from depth_anything_3.api import DepthAnything3

    import torch

    dataset = PreparedMeshDataset.from_manifest(args.manifest, "test")
    if len(dataset) != args.expected_test_samples:
        raise ValueError(
            f"Expected {args.expected_test_samples} test samples, found {len(dataset)}."
        )
    representatives = representative_indices(dataset)
    if len(representatives) != args.expected_test_objects:
        raise ValueError(
            f"Expected {args.expected_test_objects} test objects, found {len(representatives)}."
        )
    _validate_shared_observations(dataset)
    if args.object_id is not None:
        representatives = [item for item in representatives if item[0] == args.object_id]
        if len(representatives) != 1:
            raise ValueError(f"Expected exactly one --object-id match, found {representatives}")
        assigned = representatives
    else:
        assigned = representatives[args.shard_index :: args.shard_count]
    device = torch.device(args.device)
    model = DepthAnything3.from_pretrained(prior_config["model"]).to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    skipped = 0
    for ordinal, (object_id, index) in enumerate(assigned, start=1):
        sample = dataset.load_static(index)
        image_paths = _resolve_images(sample)
        intrinsics = _numpy(sample["intrinsics"])
        extrinsics = _numpy(sample["extrinsics"])
        views = int(args.expected_view_count)
        if (
            len(image_paths) != views
            or intrinsics.shape != (views, 3, 3)
            or extrinsics.shape != (views, 4, 4)
        ):
            raise ValueError(
                f"{object_id} does not contain exactly {views} RGB/camera inputs."
            )
        object_dir = args.output_dir / object_id
        if _is_complete(object_dir, views):
            skipped += 1
            continue
        object_dir.mkdir(parents=True, exist_ok=True)
        prediction = model.inference(
            [str(path) for path in image_paths],
            extrinsics=extrinsics,
            intrinsics=intrinsics,
        )
        if tuple(prediction.depth.shape[:1]) != (views,):
            raise ValueError(f"DA3 returned {prediction.depth.shape[0]} views for {object_id}.")
        for view in range(views):
            stem = f"{view + 1:08d}"
            payload = {"depth": np.round(prediction.depth[view], 6)}
            if prediction.conf is not None:
                payload["conf"] = np.round(prediction.conf[view], 2)
            np.savez_compressed(object_dir / f"{stem}.npz", **payload)
        metadata = {
            "status": "completed",
            "object_id": object_id,
            "representative_sample_id": str(sample["sample_id"]),
            "view_count": views,
            "input_contract": "same RGB images and cameras only; no GT",
            "consumed_sample_fields": [
                "sample_id",
                "image_paths",
                "_dataset_root",
                "intrinsics",
                "extrinsics",
            ],
            "da3_repository": prior_config["repository"],
            "da3_commit": checkout,
            "model": prior_config["model"],
            "image_list_sha256": _path_digest(image_paths),
        }
        (object_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        completed += 1
        print(
            f"DA3 shard={args.shard_index} {ordinal}/{len(assigned)} "
            f"object={object_id} status=completed",
            flush=True,
        )
    summary = {
        "status": "completed",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "assigned_objects": len(assigned),
        "completed_objects": completed,
        "skipped_complete_objects": skipped,
        "da3_commit": checkout,
        "model": prior_config["model"],
    }
    (args.output_dir / f"shard_{args.shard_index:03d}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _resolve_images(sample: dict[str, Any]) -> list[Path]:
    # Deliberately select only RGB input fields; labels and GT are never copied.
    root = Path(str(sample["_dataset_root"])).resolve()
    result = []
    for value in sample["image_paths"]:
        path = Path(value)
        result.append((path if path.is_absolute() else root / path).resolve())
    missing = [str(path) for path in result if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing DA3 RGB inputs: " + ", ".join(missing))
    return result


def _validate_shared_observations(dataset: PreparedMeshDataset) -> None:
    """Prove that one DA3 inference can be reused across an object's variants."""

    reference: dict[str, tuple[tuple[str, ...], np.ndarray, np.ndarray]] = {}
    counts: dict[str, int] = {}
    for index, sample_id in enumerate(dataset.sample_ids):
        object_id = object_id_from_sample_id(sample_id)
        sample = dataset.load_static(index)
        signature = (
            tuple(str(value) for value in sample["image_paths"]),
            _numpy(sample["intrinsics"]),
            _numpy(sample["extrinsics"]),
        )
        if object_id in reference:
            expected = reference[object_id]
            if (
                signature[0] != expected[0]
                or not np.array_equal(signature[1], expected[1])
                or not np.array_equal(signature[2], expected[2])
            ):
                raise ValueError(
                    f"Variants of {object_id} do not share identical RGB/camera inputs."
                )
        else:
            reference[object_id] = signature
        counts[object_id] = counts.get(object_id, 0) + 1
    if set(counts.values()) != {5}:
        raise ValueError("DA3 reuse requires exactly five variants per test object.")


def _is_complete(path: Path, views: int) -> bool:
    metadata = path / "metadata.json"
    if not metadata.is_file():
        return False
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    return (
        payload.get("status") == "completed"
        and payload.get("view_count") == views
        and all((path / f"{index:08d}.npz").is_file() for index in range(1, views + 1))
    )


def _path_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _git_commit(path: Path) -> str:
    import subprocess

    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--da3-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-test-samples", type=int, default=1000)
    parser.add_argument("--expected-test-objects", type=int, default=200)
    parser.add_argument("--expected-view-count", type=int, default=28)
    parser.add_argument("--object-id")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
