from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mlr.learned_laplacian.synthetic_current_image_feature_ablation import (
    ARMS,
    FEATURE_MODES,
    _controlled_config,
    _feature_mode,
    _prediction_aggregates,
    _validation_step_interval,
)


ROOT = Path(__file__).resolve().parents[2]


def _config(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value.get("experiment_config", value)


def test_image_feature_configs_preserve_effective_baseline_contract() -> None:
    baseline = _config(
        ROOT
        / "runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7/B_direct_raw_laplacian/run_config.json"
    )
    gaussian = _config(
        ROOT
        / "configs/learned_laplacian/train_sofa50_synthetic_current_28view_direct_raw_gaussian_feature_20k_2gpu.json"
    )
    high = _config(
        ROOT
        / "configs/learned_laplacian/train_sofa50_synthetic_current_28view_direct_raw_original_plus_high_frequency_20k_2gpu.json"
    )

    configs = dict(zip(ARMS, (baseline, gaussian, high), strict=True))
    assert {arm: _feature_mode(config) for arm, config in configs.items()} == FEATURE_MODES
    assert _controlled_config(baseline) == _controlled_config(gaussian)
    assert _controlled_config(baseline) == _controlled_config(high)
    assert _validation_step_interval(baseline, 1) == 500
    assert _validation_step_interval(gaussian, 2) == 500
    assert _validation_step_interval(high, 2) == 500
    assert baseline["multi_object_training"]["gradient_accumulation_meshes"] == 2
    assert gaussian["multi_object_training"]["gradient_accumulation_meshes"] == 1
    assert high["multi_object_training"]["gradient_accumulation_meshes"] == 1
    assert gaussian["multi_object_training"]["report_every_optimizer_steps"] == 200
    assert high["multi_object_training"]["report_every_optimizer_steps"] == 200


def test_gt_magnitude_groups_use_global_target_ranking() -> None:
    arrays = {}
    target = np.zeros((100, 3), dtype=np.float64)
    target[:, 0] = np.arange(1, 101)
    for split in ("validation", "test"):
        for arm in ARMS:
            prediction = target.copy()
            prediction[:, 1] = 1.0
            prediction[-10:, 1] = 5.0
            prediction[-1, 1] = 9.0
            prefix = f"{split}__{arm}"
            arrays[f"{prefix}__prediction"] = prediction
            arrays[f"{prefix}__target"] = target.copy()
            arrays[f"{prefix}__recovery_weight"] = np.ones(100)

    aggregate, groups, targets_equal = _prediction_aggregates(arrays)

    assert targets_equal
    assert len(aggregate) == 6
    row = next(
        value
        for value in groups
        if value["split"] == "test" and value["arm"] == ARMS[0]
    )
    assert row["bottom_90_percent_mean_raw_error_epe"] == 1.0
    assert row["top_10_percent_mean_raw_error_epe"] == 5.4
    assert row["top_1_percent_mean_raw_error_epe"] == 9.0
