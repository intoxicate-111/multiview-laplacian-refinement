import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mlr.learned_laplacian.synthetic_current_h2_ablation import (
    ARMS,
    GEOMETRY_FIELDS,
    PERCENTAGES,
    RAW_METRIC_FIELDS,
    _aggregate_small_h,
    _infer_one,
    _raw_metrics,
    _write_h2_shard,
    merge_h2_normalization_ablation_shards,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "learned_laplacian"


def _load(name: str) -> dict:
    return json.loads((CONFIG_DIR / name).read_text(encoding="utf-8"))


def _controlled(config: dict) -> dict:
    result = copy.deepcopy(config)
    result.pop("method", None)
    result.pop("target_mode", None)
    result.pop("target_definition", None)
    result.get("training", {}).pop("prediction_loss_space", None)
    result.get("recovery", {}).pop("denormalization", None)
    result.pop("experiment_metadata", None)
    return result


def test_three_arm_configs_differ_only_in_ablation_variables() -> None:
    arm_a = _load("train_sofa50_synthetic_current_28view_no_jitter_20k.json")
    arm_b = _load("train_sofa50_synthetic_current_28view_direct_raw_20k.json")
    arm_c = _load(
        "train_sofa50_synthetic_current_28view_normalized_output_raw_loss_20k.json"
    )

    assert _controlled(arm_a) == _controlled(arm_b) == _controlled(arm_c)
    assert arm_a["target_mode"] == "edge_scale_normalized_laplacian"
    assert arm_b["target_mode"] == "raw_laplacian"
    assert arm_c["target_mode"] == "edge_scale_normalized_laplacian"
    assert arm_a["training"].get(
        "prediction_loss_space", "output_representation"
    ) == "output_representation"
    assert arm_b["training"]["prediction_loss_space"] == "output_representation"
    assert arm_c["training"]["prediction_loss_space"] == "raw_laplacian"


def test_three_arm_fixed_training_contract() -> None:
    configs = (
        _load("train_sofa50_synthetic_current_28view_no_jitter_20k.json"),
        _load("train_sofa50_synthetic_current_28view_direct_raw_20k.json"),
        _load(
            "train_sofa50_synthetic_current_28view_normalized_output_raw_loss_20k.json"
        ),
    )
    for config in configs:
        assert config["seed"] == 7
        assert config["device"] == "cuda"
        assert config["experiment_metadata"]["views"] == 28
        assert config["model"]["hidden_dim"] == 256
        assert config["model"]["num_graph_layers"] == 3
        assert config["model"]["geometry_mode"] == "query_fourier"
        assert config["query_training"]["enabled"] is False
        assert config["local_query_jitter"]["enabled"] is False
        assert config["training"]["vertex_sampling"] == {"mode": "full"}
        assert config["multi_object_training"]["max_optimizer_steps"] == 20_000
        assert config["multi_object_training"]["gradient_accumulation_meshes"] == 2
        assert config["confidence"]["recovery_weight"] == (
            "renderer_visible_any_times_confidence_prediction"
        )


def test_inference_uses_explicit_faces_after_training_sample_pruning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    faces = torch.tensor([[0, 1, 2]])
    prepared = SimpleNamespace(
        sample={
            "vertices": vertices,
            "local_edge_length": torch.ones(3),
            "valid_scale_mask": torch.ones(3, dtype=torch.bool),
            "visibility": torch.ones((1, 3), dtype=torch.bool),
            "target_confidence": torch.ones(3),
        },
        raw_target=torch.zeros_like(vertices),
    )

    monkeypatch.setattr(
        "mlr.learned_laplacian.synthetic_current_h2_ablation._load_device_item",
        lambda *_args, **_kwargs: prepared,
    )

    class _Model:
        def __call__(self, _sample):
            return SimpleNamespace(
                predicted_laplacian=torch.zeros_like(vertices),
                confidence_prediction=torch.ones(3),
            )

    inferred = _infer_one(
        object(),
        0,
        {
            "config": {
                "target_mode": "raw_laplacian",
                "target_scaling": {"epsilon": 1e-12},
            },
            "model": _Model(),
            "amp_dtype": torch.float16,
            "amp_enabled": False,
        },
        torch.device("cpu"),
        current_faces=faces,
    )

    assert torch.equal(inferred["prediction_raw"], torch.zeros_like(vertices))
    assert torch.equal(inferred["recovery_weight"], torch.ones(3))


def test_three_shards_merge_into_complete_audit(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"samples": []}\n', encoding="utf-8")
    output = tmp_path / "analysis"
    preflight = {"passed": True, "contract": "test"}
    arm_metadata = {
        arm: {
            "run_dir": f"/runs/{arm}",
            "checkpoint": f"/runs/{arm}/checkpoint_latest.pt",
            "checkpoint_sha256": arm,
            "optimizer_steps": 20_000,
            "target_mode": "raw_laplacian",
            "prediction_loss_space": "raw_laplacian",
            "native_best_validation_loss": 1e-6,
            "native_final_validation_loss": 2e-6,
            "runtime_seconds": 1.0,
        }
        for arm in ARMS
    }

    for shard_index in range(3):
        indices = [index for index in range(25) if index % 3 == shard_index]
        prediction_rows = []
        recovery_rows = []
        selection_checks = []
        formula_checks = []
        roundtrip_checks = []
        for split in ("validation", "test"):
            for index in indices:
                sample_id = f"{split}_{index:02d}"
                formula_checks.append(
                    {
                        "sample_id": sample_id,
                        "current_graph_proxy_raw_target_max_abs_error": 0.0,
                    }
                )
                for arm in ARMS:
                    prediction_rows.append(
                        {
                            "split": split,
                            "arm": arm,
                            "sample_id": sample_id,
                            **{field: 1.0 for field in RAW_METRIC_FIELDS},
                            "mean_confidence": 1.0,
                            "visible_vertex_fraction": 1.0,
                        }
                    )
                    roundtrip_checks.append(
                        {
                            "split": split,
                            "arm": arm,
                            "sample_id": sample_id,
                            "max_abs_output_to_raw_roundtrip_error": 0.0,
                        }
                    )
                    if split != "test":
                        continue
                    selection_checks.append(
                        {"sample_id": sample_id, "checkpoint": arm, "passed": True}
                    )
                    for percentage in PERCENTAGES:
                        recovery_rows.append(
                            {
                                "arm": arm,
                                "replacement_percent": percentage,
                                "sample_id": sample_id,
                                "object_id": f"object_{index // 5}",
                                "actual_replacement_percent": float(percentage),
                                "raw_residual_energy_replaced_fraction": percentage / 100,
                                **{field: 1.0 for field in GEOMETRY_FIELDS},
                                "introduced_flipped_faces": 0,
                                "new_degenerate_faces": 0,
                                "improved_over_initial": False,
                            }
                        )

        shard_csv = output / "shards" / f"small_h_per_vertex_shard_{shard_index}.csv"
        shard_csv.parent.mkdir(parents=True, exist_ok=True)
        shard_csv.write_text("sample_id,value\nshard,1\n", encoding="utf-8")
        small_h_arrays = {
            split: {
                "h_current": [np.arange(1, len(indices) + 1, dtype=np.float64)],
                "normalized_residual": [np.ones(len(indices))],
                "raw_residual": [np.ones(len(indices))],
                "weighted_normalized_loss": [np.ones(len(indices))],
                "weighted_raw_residual": [np.ones(len(indices))],
                "recovered_distance": [np.ones(len(indices))],
                "valid": [np.ones(len(indices), dtype=bool)],
            }
            for split in ("validation", "test")
        }
        _write_h2_shard(
            output,
            shard_index=shard_index,
            shard_count=3,
            manifest=manifest,
            preflight=preflight,
            arm_metadata=arm_metadata,
            prediction_rows=prediction_rows,
            recovery_rows=recovery_rows,
            selection_checks=selection_checks,
            target_formula_checks=formula_checks,
            roundtrip_checks=roundtrip_checks,
            small_h_arrays=small_h_arrays,
            small_h_path=shard_csv,
        )

    summary = merge_h2_normalization_ablation_shards(
        manifest, output, shard_count=3
    )

    assert summary["contract_audit"]["passed"] is True
    assert summary["contract_audit"]["counts"] == {
        "prediction_rows": 150,
        "recovery_rows": 450,
        "formula_checks": 50,
        "roundtrip_checks": 150,
        "selection_checks": 75,
    }
    assert (output / "REPORT.md").is_file()
    assert len((output / "small_h_per_vertex.csv").read_text().splitlines()) == 4


def test_raw_metrics_use_residual_magnitude_for_tail_percentiles() -> None:
    target = torch.zeros((100, 3))
    prediction = torch.zeros((100, 3))
    prediction[:, 0] = torch.arange(1, 101, dtype=torch.float32)
    weight = torch.ones(100)
    weight[-1] = 4.0

    metrics = _raw_metrics(prediction, target, weight, torch.ones(100, dtype=torch.bool))

    assert metrics["raw_epe"] == pytest.approx(50.5)
    assert metrics["raw_top_1_percent_epe"] == pytest.approx(100.0)
    assert metrics["raw_top_10_percent_epe"] == pytest.approx(95.5)
    assert metrics["raw_top_20_percent_epe"] == pytest.approx(90.5)
    assert metrics["raw_top_50_percent_epe"] == pytest.approx(75.5)
    expected_weighted_rms = np.sqrt(
        (sum(value * value for value in range(1, 100)) + 4 * 100**2) / 103
    )
    assert metrics["recovery_weighted_raw_residual_rms"] == pytest.approx(
        expected_weighted_rms
    )


def test_small_h_groups_are_disjoint_and_loss_contributions_sum_to_one() -> None:
    count = 100
    split_values = {
        "h_current": [np.arange(1, count + 1, dtype=np.float64)],
        "normalized_residual": [np.arange(1, count + 1, dtype=np.float64)],
        "raw_residual": [np.ones(count)],
        "weighted_normalized_loss": [np.arange(1, count + 1, dtype=np.float64)],
        "weighted_raw_residual": [np.full(count, 2.0)],
        "recovered_distance": [np.full(count, 3.0)],
        "valid": [np.ones(count, dtype=bool)],
    }

    rows = _aggregate_small_h(
        {"validation": split_values, "test": copy.deepcopy(split_values)}
    )

    for split in ("validation", "test"):
        selected = [row for row in rows if row["split"] == split]
        assert [row["vertex_count"] for row in selected] == [1, 9, 15, 25, 50]
        assert sum(row["vertex_count"] for row in selected) == count
        assert sum(
            row["normalized_loss_contribution_fraction"] for row in selected
        ) == pytest.approx(1.0)
        assert selected[0]["normalized_residual_mean"] == pytest.approx(1.0)
        assert selected[-1]["recovered_vertex_to_gt_surface_distance_mean"] == 3.0


def test_arm_labels_are_fixed() -> None:
    assert ARMS == (
        "A_canonical_h2_normalized",
        "B_direct_raw_laplacian",
        "C_normalized_output_raw_loss",
    )
