from mlr.learned_laplacian.renderer_visibility_training import build_short_training_config


def test_short_training_config_changes_only_controlled_training_budget():
    source = {
        "seed": 99,
        "training": {
            "learning_rate": 1.0e-3,
            "loss": "charbonnier",
            "huber_delta": 9.0,
            "lr_scheduler": {"type": "reduce_on_plateau"},
        },
        "multi_object_training": {
            "gradient_accumulation_meshes": 4,
            "epochs": 5000,
            "max_optimizer_steps": 50000,
        },
    }
    config = build_short_training_config(
        source,
        condition="backface_and_occlusion",
        mesh_count=16,
        validation_mesh_count=5,
        optimizer_steps=100,
        seed=7,
    )

    assert source["training"]["loss"] == "charbonnier"
    assert config["seed"] == 7
    assert config["training"]["learning_rate"] == 1.0e-3
    assert config["training"]["loss"] == "huber"
    assert config["training"]["huber_delta"] == 0.01
    assert config["training"]["target_magnitude_weight_lambda"] == 0.0
    assert config["training"]["lr_scheduler"] == {"type": "none"}
    assert config["multi_object_training"]["max_optimizer_steps"] == 100
    assert config["multi_object_training"]["validation_every_epochs"] == 3
    assert config["renderer_visibility"]["condition"] == "backface_and_occlusion"
    assert config["renderer_visibility"]["depth_image_used"] is False
