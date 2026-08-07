import numpy as np
import torch

from mlr.learned_laplacian.current_graph_identity import (
    analytic_current_graph_identity_inputs,
)
from mlr.learned_laplacian.evaluation import reconstruct_and_evaluate


def _mesh():
    vertices = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
    return vertices, faces


def test_identity_normalization_and_canonical_denormalization_reconstruct_lx0():
    vertices, faces = _mesh()
    result = analytic_current_graph_identity_inputs(
        vertices,
        faces,
        torch.tensor([[True, False, True], [False, False, True]]),
        torch.tensor([0.2, 0.9, 0.7]),
        epsilon=1e-12,
    )

    torch.testing.assert_close(
        result.recovery.delta_pred_raw,
        result.delta_identity_raw,
        rtol=1e-6,
        atol=1e-7,
    )
    torch.testing.assert_close(
        result.recovery.delta_current_raw,
        result.delta_identity_raw,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(result.recovery.weight, torch.tensor([0.2, 0.0, 0.7]))


def test_recovery_evaluation_does_not_require_placeholder_target(tmp_path):
    vertices, faces = _mesh()
    identity = analytic_current_graph_identity_inputs(
        vertices,
        faces,
        torch.ones((1, 3), dtype=torch.bool),
        torch.ones(3),
    )
    sample = {
        "vertices": vertices,
        "faces": faces,
        "local_edge_length": identity.recovery.h_current,
        "valid_scale_mask": torch.ones(3, dtype=torch.bool),
    }
    metrics = reconstruct_and_evaluate(
        sample,
        identity.recovery.delta_pred_raw,
        tmp_path,
        {
            "operator_type": "uniform",
            "lambda_lap": 1.0,
            "lambda_anchor": 0.01,
            "lambda_edge": 0.0,
            "num_iters": 1,
            "learning_rate": 0.01,
            "robust_loss": "huber",
            "huber_delta": 0.01,
            "evaluate_oracle": False,
            "write_legacy_prediction_names": False,
        },
        normalized_prediction=identity.delta_identity_hat,
        laplacian_weight=np.ones(3),
        solver_confidence=np.ones(3),
        evaluate_laplacian_prediction=False,
    )

    assert metrics["laplacian_prediction"] is None
    assert metrics["reconstruction"]["oracle_evaluated"] is False
