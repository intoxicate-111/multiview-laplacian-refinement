from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs" / "learned_laplacian"


def _load(name: str) -> dict:
    return json.loads((CONFIGS / name).read_text(encoding="utf-8"))


def test_future2000_main_is_a_controlled_sofa50_template_clone() -> None:
    mother = _load("train_sofa50_synthetic_current_28view_direct_raw_20k.json")
    main = _load(
        "train_future2000_gt_adaptive_2000mesh_expanded_current_28view_direct_raw_20k.json"
    )
    allowed = {"method", "dataset", "experiment_metadata"}
    assert {
        key: value for key, value in main.items() if key not in allowed
    } == {
        key: value for key, value in mother.items() if key not in allowed
    }
    assert main["dataset"]["objects"] == 2000
    assert main["dataset"]["variants_per_object"] == 5
    assert main["dataset"]["expected_split_counts"] == {
        "train": 8000,
        "validation": 1000,
        "test": 1000,
    }
    assert main["target_definition"] == "delta_target_raw=L_current@P_proxy"
    assert main["target_mode"] == "raw_laplacian"
    assert main["query_training"]["enabled"] is False
    assert main["local_query_jitter"]["enabled"] is False
    assert main["experiment_metadata"]["views"] == 28
    assert main["experiment_metadata"]["stage_2_adaptation"] is False


def test_displacement_arm_changes_only_target_and_recovery_semantics() -> None:
    main = _load(
        "train_future2000_gt_adaptive_2000mesh_expanded_current_28view_direct_raw_20k.json"
    )
    displacement = _load(
        "train_future2000_gt_adaptive_2000mesh_expanded_current_28view_displacement_20k.json"
    )
    allowed = {
        "method",
        "prediction_semantics",
        "target_semantics",
        "target_definition",
        "recovery",
        "experiment_metadata",
    }
    assert {
        key: value for key, value in displacement.items() if key not in allowed
    } == {
        key: value for key, value in main.items() if key not in allowed
    }
    assert displacement["prediction_semantics"] == "direct_vertex_displacement"
    assert displacement["target_definition"] == (
        "displacement_target=P_proxy-P_current"
    )
    assert displacement["recovery"] == {
        "mode": "direct_vertex_addition",
        "definition": "P_refined=P_current+displacement_prediction",
        "laplacian_solver": "disabled",
    }
