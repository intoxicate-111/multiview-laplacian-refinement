from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_external_config_pins_methods_and_forbids_test_tuning() -> None:
    config = json.loads(
        (ROOT / "configs/baselines/future2000_external_baselines.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["test_samples"] == 1000
    assert config["no_test_tuning"] is True
    assert set(config["methods"]) == {
        "openmvs_refinemesh",
        "nds",
        "nerf2mesh",
        "exmesh",
    }
    assert all(len(method["commit"]) == 40 for method in config["methods"].values())
    assert all(
        len(commit) == 40
        for name, commit in config["environment_pins"].items()
        if name != "torch_runtime"
    )
    assert len(config["methods"]["exmesh"]["depth_prior"]["commit"]) == 40
    assert "no GT" in config["methods"]["exmesh"]["depth_prior"]["input_contract"]
    assert "no GT" in config["input_contract"]
