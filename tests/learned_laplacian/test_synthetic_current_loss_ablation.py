import json
from pathlib import Path

import numpy as np

from mlr.learned_laplacian.synthetic_current_loss_ablation import (
    ARMS,
    _controlled_config,
    _global_prediction_aggregates,
    _validation_step_interval,
)


ROOT = Path(__file__).resolve().parents[2]


def _config(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value.get("experiment_config", value)


def test_3gpu_mse_config_matches_huber_after_authorized_normalization() -> None:
    huber = _config(
        ROOT
        / "runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7/B_direct_raw_laplacian/run_config.json"
    )
    mse = _config(
        ROOT
        / "configs/learned_laplacian/train_sofa50_synthetic_current_28view_direct_raw_mse_20k_3gpu.json"
    )

    assert huber["training"]["loss"] == "huber"
    assert mse["training"]["loss"] == "mse"
    assert _controlled_config(huber) == _controlled_config(mse)
    assert _validation_step_interval(huber, world_size=1) == 500
    assert _validation_step_interval(mse, world_size=3) == 510


def test_global_tail_groups_rank_by_target_magnitude_not_residual() -> None:
    arrays = {}
    for split in ("validation", "test"):
        for arm in ARMS:
            prefix = f"{split}__{arm}"
            target = np.zeros((100, 3), dtype=np.float64)
            target[:, 0] = np.arange(1, 101)
            prediction = target.copy()
            prediction[:, 1] = 1.0
            prediction[-10:, 1] = 5.0
            prediction[-1, 1] = 9.0
            arrays[f"{prefix}__prediction"] = prediction
            arrays[f"{prefix}__target"] = target
            arrays[f"{prefix}__recovery_weight"] = np.ones(100)

    aggregate, groups = _global_prediction_aggregates(arrays)

    assert len(aggregate) == 4
    row = next(
        value
        for value in groups
        if value["split"] == "test" and value["arm"] == ARMS[0]
    )
    assert row["bottom_90_percent_mean_raw_error_epe"] == 1.0
    assert row["top_10_percent_mean_raw_error_epe"] == 5.4
    assert row["top_1_percent_mean_raw_error_epe"] == 9.0
