from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from mlr.learned_laplacian.dataset import save_prepared_sample
from mlr.learned_laplacian.multi_trainer import train_multi_object
from mlr.learned_laplacian.local_query_jitter import local_query_jitter_settings
from mlr.learned_laplacian.synthetic_current_stage2_adaptation import (
    ARMS,
    BASELINE,
    EXPECTED_B_SHA256,
    _aggregate_stage2_rows,
    _make_image_paths_absolute,
    merge_stage2_dataset_shards,
)

from .helpers import tiny_sample


def test_stage2_continuation_config_preserves_B_training_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    base = json.loads(
        (root / "configs/learned_laplacian/train_sofa50_synthetic_current_28view_direct_raw_20k.json").read_text()
    )
    continuation = json.loads(
        (root / "configs/learned_laplacian/train_sofa50_synthetic_current_stage2_continuation_20k.json").read_text()
    )
    assert continuation["target_mode"] == "raw_laplacian"
    assert continuation["model"] == base["model"]
    assert continuation["image_encoder"] == base["image_encoder"]
    assert continuation["confidence"] == base["confidence"]
    assert continuation["training"] == base["training"]
    assert continuation["recovery"] == base["recovery"]
    assert continuation["multi_object_training"]["gradient_accumulation_meshes"] == 2
    assert continuation["multi_object_training"]["max_optimizer_steps"] == 40_000
    assert continuation["local_query_jitter"]["enabled"] is False
    assert local_query_jitter_settings(continuation).enabled is False
    assert continuation["query_training"]["enabled"] is False


def test_stage2_image_paths_are_made_absolute(tmp_path: Path) -> None:
    source = {"_dataset_root": str(tmp_path)}
    stage2 = {"image_paths": ["rgb/00.png", str(tmp_path / "rgb/01.png")]}
    _make_image_paths_absolute(stage2, source)
    assert stage2["image_paths"] == [
        str((tmp_path / "rgb/00.png").resolve()),
        str((tmp_path / "rgb/01.png").resolve()),
    ]
    assert stage2["image_path_root"] == "/"


def test_stage2_aggregate_tracks_retained_gained_and_lost() -> None:
    categories = ["retained"] * 18 + ["lost"] + ["gained"] * 2 + ["remained_failed"] * 4
    rows = []
    for index, category in enumerate(categories):
        row = {
            "arm": ARMS[1],
            "checkpoint_kind": "best",
            "sample_id": f"sample_{index:02d}",
            "stage2_chamfer": 0.003,
            "stage2_point_to_surface": 0.0031,
            "stage2_normal_consistency": 0.95,
            "stage2_cumulative_introduced_flipped_faces": 2,
            "stage2_step_introduced_flipped_faces": 1,
            "stage2_improved_vs_initial": category in {"retained", "gained"},
            "stage2_improved_vs_stage1": index < 10,
            "transition_category": category,
        }
        for field in (
            "raw_epe",
            "raw_top_1_percent_epe",
            "raw_top_10_percent_epe",
            "raw_top_20_percent_epe",
            "raw_top_50_percent_epe",
            "raw_global_cosine",
            "prediction_to_target_raw_norm_ratio",
            "raw_residual_rms",
            "raw_residual_maximum",
            "recovery_weighted_raw_residual_rms",
        ):
            row[field] = 0.1
            row[f"zero_rgb_{field}"] = 0.2
        rows.append(row)
    aggregate = _aggregate_stage2_rows(rows, ARMS[1], "best")
    assert aggregate["cumulative_improved_over_original"] == 20
    assert aggregate["retained_from_original_19"] == 18
    assert aggregate["gained_from_original_failed_6"] == 2
    assert aggregate["lost_from_original_19"] == 1
    assert aggregate["remained_failed_from_original_6"] == 4
    assert aggregate["improved_over_stage1"] == 10


def test_merge_stage2_shards_builds_matched_manifests_and_passes_audit(
    tmp_path: Path,
) -> None:
    items = []
    rows_by_shard = {0: [], 1: [], 2: []}
    split_counts = {"train": 200, "validation": 25, "test": 25}
    global_index = 0
    test_index = 0
    for split, count in split_counts.items():
        for local_index in range(count):
            sample_id = f"{split}_{local_index:03d}"
            original_path = tmp_path / "original" / f"{sample_id}.pt"
            stage2_path = tmp_path / "stage2" / f"{sample_id}.pt"
            original_sample = tiny_sample()
            original_sample["sample_id"] = sample_id
            save_prepared_sample(original_sample, original_path)
            items.append({"sample_id": sample_id, "split": split, "path": str(original_path)})
            row = {
                "global_index": global_index,
                "split": split,
                "sample_id": sample_id,
                "original_path": str(original_path),
                "stage2_path": str(stage2_path),
                "faces_preserved": True,
                "target_exact_equivalence": True,
                "target_max_abs_difference": 0.0,
                "visibility_backend": "opengl",
                "visibility_raster_size": 960,
            }
            if split == "test":
                improved = test_index < 19
                row.update(
                    {
                        "initial_chamfer": (
                            BASELINE["reconstruction_chamfer"] + 1.0
                            if improved
                            else BASELINE["reconstruction_chamfer"] - 0.001
                        ),
                        "stage1_chamfer": BASELINE["reconstruction_chamfer"],
                        "stage1_point_to_surface": BASELINE["reconstruction_point_to_surface"],
                        "stage1_normal_consistency": BASELINE["reconstruction_normal_consistency"],
                        "stage1_introduced_flipped_faces": 6566 if test_index == 0 else 0,
                        "raw_epe": BASELINE["raw_epe"],
                        "raw_top_1_percent_epe": BASELINE["raw_top_1_percent_epe"],
                        "raw_top_10_percent_epe": BASELINE["raw_top_10_percent_epe"],
                        "raw_global_cosine": BASELINE["raw_global_cosine"],
                        "recovery_weighted_raw_residual_rms": BASELINE[
                            "recovery_weighted_raw_residual_rms"
                        ],
                    }
                )
                test_index += 1
            rows_by_shard[global_index % 3].append(row)
            global_index += 1
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"samples": items}))
    manifest_sha = _sha256(manifest)
    output = tmp_path / "output"
    for shard_index in range(3):
        path = output / "shards" / f"dataset_shard_{shard_index}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "shard_index": shard_index,
                    "shard_count": 3,
                    "manifest_sha256": manifest_sha,
                    "checkpoint_sha256": EXPECTED_B_SHA256,
                    "visibility_policy": "recomputed_opengl_960",
                    "visibility_size": 960,
                    "rows": rows_by_shard[shard_index],
                }
            )
        )
    audit = merge_stage2_dataset_shards(manifest, output)
    assert audit["passed"] is True
    assert audit["mix_counts"]["train_X0"] == 100
    assert audit["mix_counts"]["train_X1"] == 100
    for name in (
        "continue_original_manifest.json",
        "continue_B_result_manifest.json",
        "continue_mix_50_50_manifest.json",
        "stage2_manifest.json",
    ):
        payload = json.loads((output / "manifests" / name).read_text())
        assert len(payload["samples"]) == 250


def test_reset_resume_tracking_requires_checkpoint() -> None:
    with pytest.raises(ValueError, match="requires resume_checkpoint"):
        train_multi_object(
            [{}],
            [],
            {"device": "cpu"},
            progress=False,
            reset_resume_tracking=True,
        )


def test_reset_resume_tracking_keeps_optimizer_step_but_resets_stage_history(
    tmp_path: Path,
) -> None:
    sample = tiny_sample()
    sample["sample_id"] = "train"
    validation = tiny_sample()
    validation["sample_id"] = "validation"
    config = {
        "seed": 7,
        "device": "cpu",
        "input_mode": "coarse_only",
        "target_mode": "raw_laplacian",
        "target_scaling": {"epsilon": 1e-12},
        "image_encoder": {"feature_dim": 8},
        "model": {"hidden_dim": 8, "num_graph_layers": 1, "dropout": 0.0},
        "training": {
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "loss": "huber",
            "huber_delta": 0.01,
            "gradient_clip_norm": 1.0,
        },
        "multi_object_training": {
            "epochs": 3,
            "max_optimizer_steps": 1,
            "gradient_accumulation_meshes": 1,
            "shuffle": False,
            "validation_every_epochs": 1,
        },
    }
    first_dir = tmp_path / "first"
    first = train_multi_object(
        [sample], [validation], config, output_dir=first_dir, progress=False
    )
    assert first.optimizer_steps == 1

    continued_config = json.loads(json.dumps(config))
    continued_config["multi_object_training"]["max_optimizer_steps"] = 2
    second = train_multi_object(
        [sample],
        [validation],
        continued_config,
        output_dir=tmp_path / "second",
        progress=False,
        resume_checkpoint=first_dir / "checkpoint_latest.pt",
        reset_resume_tracking=True,
    )
    assert second.optimizer_steps == 2
    assert second.continuation_optimizer_steps == 1
    assert len(second.history) == 1
    assert second.history[0]["optimizer_steps"] == 2


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
