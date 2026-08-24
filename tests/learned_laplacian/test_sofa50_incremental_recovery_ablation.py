from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "diagnose_sofa50_incremental_recovery_ablation.py"
SPEC = importlib.util.spec_from_file_location("incremental_recovery_ablation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_recovery_arm_specs_are_strictly_cumulative() -> None:
    specs = MODULE.recovery_arm_specs(
        {
            "operator_type": "uniform",
            "lambda_lap": 1.0,
            "lambda_anchor": 0.01,
            "lambda_edge": 0.0,
            "unseen_anchor_weight": 0.0,
            "num_iters": 200,
            "learning_rate": 0.01,
            "robust_loss": "huber",
            "huber_delta": 0.01,
        }
    )
    assert tuple(specs) == MODULE.ARM_ORDER
    assert specs["pure_laplacian_l2"]["lambda_anchor"] == 0
    assert specs["plus_anchor"]["lambda_anchor"] == 0.01
    assert specs["plus_visibility"]["weight_mode"] == "hard_any_view_visibility"
    assert specs["plus_confidence"]["weight_mode"] == "visibility_times_learned_confidence"
    assert specs["plus_confidence"]["robust_loss"] == "l2"
    assert specs["plus_huber"]["robust_loss"] == "huber"
    assert specs["full_solver"]["lambda_edge"] == 0


def test_sparse_effective_terms_expose_huber_noop() -> None:
    spec = {"robust_loss": "huber", "lambda_edge": 0.0, "lambda_anchor": 0.01, "weight_mode": "visibility_times_learned_confidence"}
    sparse = MODULE.effective_terms(spec, "sparse_uniform_oracle_core")
    dense = MODULE.effective_terms(spec, "dense_refinement")
    assert sparse["effective_robust_loss"] == "l2"
    assert sparse["huber_actually_active"] is False
    assert dense["effective_robust_loss"] == "huber"
    assert dense["huber_actually_active"] is True


def test_identify_collapse_uses_largest_negative_eta_step() -> None:
    rows = [
        {"arm": "pure_laplacian_l2", "previous_arm": "initial", "mean_eta_delta_from_previous": 0.4},
        {"arm": "plus_anchor", "previous_arm": "pure_laplacian_l2", "mean_eta_delta_from_previous": -0.1},
        {"arm": "plus_visibility", "previous_arm": "plus_anchor", "mean_eta_delta_from_previous": -0.3},
    ]
    collapse = MODULE.identify_collapse(rows)
    assert collapse["arm"] == "plus_visibility"
    assert collapse["classification"] == "largest_efficiency_collapse"
