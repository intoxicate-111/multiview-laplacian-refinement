from __future__ import annotations

import json
from pathlib import Path

import torch

from mlr.learned_laplacian.synthetic_current_hf_resolution_ablation import (
    _controlled_config,
    _initial_state_hash,
    _sample_gt_groups,
    _validation_interval,
)


ROOT = Path(__file__).resolve().parents[2]


def _config(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value.get("experiment_config", value)


def test_hf1920_config_preserves_model_and_step_schedule_contract():
    baseline = _config(
        ROOT
        / "configs/learned_laplacian/train_sofa50_synthetic_current_28view_direct_raw_original_plus_high_frequency_20k_2gpu.json"
    )
    candidate = _config(
        ROOT
        / "configs/learned_laplacian/train_sofa50_synthetic_current_28view_direct_raw_original_plus_high_frequency_1920_20k_4gpu.json"
    )

    assert _controlled_config(baseline) == _controlled_config(candidate)
    assert _initial_state_hash(baseline) == _initial_state_hash(candidate)
    assert _validation_interval(baseline, 2) == 500
    assert _validation_interval(candidate, 4) == 500
    assert baseline["multi_object_training"]["max_optimizer_steps"] == 20_000
    assert candidate["multi_object_training"]["max_optimizer_steps"] == 20_000
    assert baseline["image_encoder"]["feature_construction"] == candidate[
        "image_encoder"
    ]["feature_construction"]
    assert candidate["image_encoder"]["view_chunk_size"] == 4
    assert candidate["image_encoder"]["gradient_checkpointing"] is True


def test_per_sample_tail_groups_rank_by_gt_magnitude():
    target = torch.zeros((100, 3), dtype=torch.float64)
    target[:, 0] = torch.arange(1, 101, dtype=torch.float64)
    prediction = target.clone()
    prediction[:, 1] = 1.0
    prediction[-10:, 1] = 5.0
    prediction[-1, 1] = 9.0

    groups = _sample_gt_groups(
        prediction, target, torch.ones(100, dtype=torch.bool)
    )

    assert groups["gt_bottom90_epe"] == 1.0
    assert groups["gt_top10_epe"] == 5.4
    assert groups["gt_top1_epe"] == 9.0
