import torch

from mlr.learned_laplacian.synthetic_current_oracle_recovery import (
    _decision,
    _distribution,
    _formula_audit,
    _geometry_changes,
    _lost_comparison,
    _weighted_rms,
)


def test_distribution_and_weighted_rms_are_vertexwise():
    values = torch.tensor([1.0, 2.0, 3.0])
    distribution = _distribution(values)
    assert distribution["mean"] == 2.0
    assert distribution["median"] == 2.0
    assert distribution["maximum"] == 3.0
    assert _weighted_rms(values, torch.tensor([1.0, 0.0, 0.0])) == 1.0


def test_formula_audit_requires_all_three_exact_target_checks():
    row = {
        "raw_round_trip_max_abs_error": 0.0,
        "current_graph_proxy_raw_target_max_abs_error": 1e-6,
        "normalized_formula_max_abs_error": 2e-6,
    }
    assert _formula_audit([row])["passed"] is True
    row["normalized_formula_max_abs_error"] = 2e-5
    assert _formula_audit([row])["passed"] is False


def test_geometry_change_sign_is_refined_minus_initial():
    result = _geometry_changes(
        {"initial_chamfer": 0.004, "reconstruction_chamfer": 0.003}
    )
    assert result["absolute_chamfer_change"] == -0.001
    assert result["percent_chamfer_change"] == -25.0


def test_lost_comparison_separates_global_and_shared_weighted_error():
    payload = {
        "current_query_20k_pred": {
            "target_epe": 2.0,
            "top10_target_magnitude_normalized_residual_mean": 3.0,
            "shared_50k_recovery_weighted_normalized_residual_rms": 2.0,
            "recovery": {"refined_chamfer": 0.003},
        },
        "current_query_50k_pred": {
            "target_epe": 1.0,
            "top10_target_magnitude_normalized_residual_mean": 4.0,
            "shared_50k_recovery_weighted_normalized_residual_rms": 3.0,
            "recovery": {"refined_chamfer": 0.004},
        },
    }
    result = _lost_comparison(payload)
    assert result["global_target_epe_lower_at_50k"] is True
    assert result["top10_normalized_residual_lower_at_50k"] is False
    assert result["shared_weighted_normalized_residual_lower_at_50k"] is False
    assert result["chamfer_lower_at_50k"] is False


def test_decision_uses_required_oracle_logic():
    learned20 = {
        "vector_l2": 2.4,
        "reconstruction_chamfer": 0.0041,
        "improved_over_initial": 5,
    }
    learned50 = {
        "vector_l2": 2.2,
        "reconstruction_chamfer": 0.0042,
        "improved_over_initial": 3,
    }
    oracle = {
        "reconstruction_chamfer": 0.002,
        "improved_over_initial": 25,
    }
    per_object = [
        {
            "arm": "current_graph_exact_target_oracle",
            "object_id": f"object-{index}",
            "improved_samples": 5,
        }
        for index in range(5)
    ]
    lost = {
        "sample": {
            "comparison_50k_vs_20k": {
                "global_target_epe_lower_at_50k": True,
                "shared_weighted_normalized_residual_lower_at_50k": False,
            }
        }
    }
    result = _decision(
        {
            "current_query_20k_pred": learned20,
            "current_query_50k_pred": learned50,
            "current_graph_exact_target_oracle": oracle,
        },
        per_object,
        lost,
    )
    assert result["classification"] == "learned_prediction_error_and_recovery_interaction"
    assert result["oracle_improves_all"] is True
    assert result["oracle_removes_single_object_success_pattern"] is True
    assert result["lost_success_solver_sensitive_pattern"]["sample"] is True
