#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SPLIT_OBJECTS = {"train": 2, "validation": 1, "test": 1}


def _object_id(sample: dict[str, Any]) -> str:
    sample_id = str(sample.get("sample_id", ""))
    if "__v" not in sample_id:
        raise ValueError(f"Cannot recover object id from sample_id {sample_id!r}.")
    return sample_id.rsplit("__v", 1)[0]


def prepare(
    manifest_path: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    config_path = config_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise ValueError("Manifest must contain a samples list.")

    by_split: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for source in samples:
        sample = dict(source)
        split = str(sample.get("split", ""))
        if split not in SPLIT_OBJECTS:
            continue
        by_split[split][_object_id(sample)].append(sample)

    selected: list[dict[str, Any]] = []
    selected_objects: dict[str, list[str]] = {}
    for split, wanted in SPLIT_OBJECTS.items():
        object_ids = sorted(by_split[split])[:wanted]
        if len(object_ids) != wanted:
            raise ValueError(f"Split {split!r} does not contain {wanted} objects.")
        selected_objects[split] = object_ids
        for object_id in object_ids:
            variants = sorted(
                by_split[split][object_id], key=lambda item: str(item["sample_id"])
            )
            if len(variants) != 5:
                raise ValueError(
                    f"Smoke object {object_id!r} has {len(variants)} variants, expected 5."
                )
            for sample in variants:
                path = Path(str(sample["path"]))
                if not path.is_absolute():
                    path = manifest_path.parent / path
                sample["path"] = str(path.resolve())
                selected.append(sample)

    counts = {
        split: sum(item["split"] == split for item in selected)
        for split in SPLIT_OBJECTS
    }
    smoke_manifest = {
        key: value for key, value in manifest.items() if key != "samples"
    }
    smoke_manifest.update(
        {
            "format_version": "future2000_training_smoke_v1",
            "source_manifest": str(manifest_path),
            "dataset_root": str(manifest_path.parent),
            "object_count": sum(SPLIT_OBJECTS.values()),
            "object_split_counts": SPLIT_OBJECTS,
            "variant_split_counts": counts,
            "samples": selected,
        }
    )
    smoke_manifest_path = output_dir / "manifest.json"
    smoke_manifest_path.write_text(
        json.dumps(smoke_manifest, indent=2) + "\n", encoding="utf-8"
    )

    source_config = json.loads(config_path.read_text(encoding="utf-8"))
    config_paths = {}
    for phase, max_steps, checkpoints in (("phase1", 5, [5]), ("phase2", 10, [10])):
        config = copy.deepcopy(source_config)
        config["method"] = f"{source_config['method']}_smoke_{phase}"
        config["dataset"] = {
            **config["dataset"],
            "name": "Future2000GTAdaptiveExpandedCurrent28ViewSmoke",
            "expected_split_counts": counts,
            "objects": sum(SPLIT_OBJECTS.values()),
        }
        config["multi_object_training"].update(
            {
                "epochs": 2,
                "max_optimizer_steps": max_steps,
                "validation_every_epochs": 1,
                "checkpoint_every_epochs": 0,
                "checkpoint_epochs": [],
                "checkpoint_optimizer_steps": checkpoints,
            }
        )
        config["experiment_metadata"] = {
            **config["experiment_metadata"],
            "experiment": f"future2000_training_smoke_{phase}",
            "smoke_test": True,
            "source_config": str(config_path),
        }
        path = output_dir / f"config_{phase}.json"
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        config_paths[phase] = str(path)

    summary = {
        "manifest": str(smoke_manifest_path),
        "configs": config_paths,
        "counts": counts,
        "selected_objects": selected_objects,
        "views": 28,
        "phase1_steps": 5,
        "phase2_resumed_total_steps": 10,
    }
    (output_dir / "preparation.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest, args.config, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
