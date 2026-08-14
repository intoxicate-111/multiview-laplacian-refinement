from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _module():
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "prepare_future2000_training_smoke.py"
    )
    spec = importlib.util.spec_from_file_location("future2000_training_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_smoke_inputs_keep_complete_objects_and_two_phase_resume_budget(tmp_path) -> None:
    samples = []
    object_counts = {"train": 3, "validation": 2, "test": 2}
    for split, count in object_counts.items():
        for object_index in range(count):
            object_id = f"{split}_{object_index:02d}"
            for variant in range(5):
                samples.append(
                    {
                        "sample_id": f"{object_id}__v{variant:02d}",
                        "split": split,
                        "path": f"prepared/{object_id}_{variant:02d}.pt",
                    }
                )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "method": "fixture",
                "dataset": {},
                "multi_object_training": {},
                "experiment_metadata": {},
            }
        ),
        encoding="utf-8",
    )

    result = _module().prepare(manifest, config, tmp_path / "smoke")

    assert result["counts"] == {"train": 10, "validation": 5, "test": 5}
    prepared_manifest = json.loads(
        Path(result["manifest"]).read_text(encoding="utf-8")
    )
    assert len(prepared_manifest["samples"]) == 20
    assert prepared_manifest["dataset_root"] == str(manifest.parent)
    assert all(Path(item["path"]).is_absolute() for item in prepared_manifest["samples"])
    phase1 = json.loads(Path(result["configs"]["phase1"]).read_text())
    phase2 = json.loads(Path(result["configs"]["phase2"]).read_text())
    assert phase1["multi_object_training"]["max_optimizer_steps"] == 5
    assert phase1["multi_object_training"]["checkpoint_optimizer_steps"] == [5]
    assert phase2["multi_object_training"]["max_optimizer_steps"] == 10
    assert phase2["multi_object_training"]["checkpoint_optimizer_steps"] == [10]
