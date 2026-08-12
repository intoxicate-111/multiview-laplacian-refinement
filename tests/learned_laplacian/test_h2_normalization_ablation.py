import copy
import json
from pathlib import Path


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
