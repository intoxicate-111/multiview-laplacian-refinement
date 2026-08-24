from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "diagnose_sofa50_recovery_step_sweep.py"
SPEC = importlib.util.spec_from_file_location("recovery_step_sweep", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_only_num_iters_changes_across_step_budgets() -> None:
    base = {
        "operator_type": "uniform",
        "lambda_lap": 1.0,
        "lambda_anchor": 0.01,
        "lambda_edge": 0.0,
        "unseen_anchor_weight": 0.0,
        "learning_rate": 0.01,
        "robust_loss": "huber",
        "huber_delta": 0.01,
    }
    configs = [MODULE.recovery_config(base, steps) for steps in MODULE.STEP_BUDGETS]
    assert [config.num_iters for config in configs] == [200, 500, 1000, 2000]
    frozen = [
        (config.operator_type, config.lambda_lap, config.lambda_anchor, config.lambda_edge, config.learning_rate, config.robust_loss)
        for config in configs
    ]
    assert len(set(frozen)) == 1
