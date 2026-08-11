import math

import pytest
import torch

from mlr.laplacian import compute_laplacian_coordinates
from mlr.learned_laplacian.model import FourierPositionEncoding, LearnedLaplacianModel
from mlr.learned_laplacian.multi_trainer import train_multi_object
from mlr.learned_laplacian.query_training import (
    QueryAugmentationSettings,
    apply_query_augmentation,
    query_augmentation_settings,
    validate_gt_query_contract,
)
from mlr.learned_laplacian.sample_io import prepare_gt_query_sample_from_prepared

from .helpers import tiny_sample


def _gt_query_sample(sample_id: str = "gt_query") -> dict:
    source = tiny_sample()
    source["sample_id"] = sample_id
    source["metadata"] = {
        "coarse_mesh_path": "ignored-expanded.obj",
        "target_constructor": "old_closest_surface_target",
    }
    return prepare_gt_query_sample_from_prepared(source)


def _settings(**overrides) -> QueryAugmentationSettings:
    values = {
        "enabled": True,
        "exact_fraction": 0.25,
        "normal_std_h": 0.0003,
        "tangent_std_h": 0.0003,
        "max_offset_h": 0.001,
        "apply_to_validation": True,
        "zero_initial_laplacian": True,
    }
    values.update(overrides)
    return QueryAugmentationSettings(**values)


def _config() -> dict:
    return {
        "seed": 5,
        "device": "cpu",
        "input_mode": "coarse_plus_multiview",
        "target_mode": "edge_scale_normalized_laplacian",
        "target_scaling": {"epsilon": 1e-12, "clip_max_norm": None},
        "query_training": {
            "enabled": True,
            "exact_fraction": 0.25,
            "normal_std_h": 0.0003,
            "tangent_std_h": 0.0003,
            "max_offset_h": 0.001,
            "apply_to_validation": True,
            "zero_initial_laplacian": True,
        },
        "image_encoder": {"feature_dim": 8},
        "model": {
            "hidden_dim": 16,
            "num_graph_layers": 1,
            "dropout": 0.0,
            "geometry_mode": "query_fourier",
            "position_encoding": {"num_frequencies": 3, "include_input": True},
        },
        "training": {
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "loss": "huber",
            "huber_delta": 0.1,
            "gradient_clip_norm": 1.0,
        },
        "multi_object_training": {
            "epochs": 1,
            "gradient_accumulation_meshes": 1,
            "shuffle": False,
            "validation_every_epochs": 1,
            "checkpoint_every_epochs": 0,
        },
    }


def test_gt_query_conversion_recomputes_target_on_gt_graph_and_zeros_leakage_input():
    source = tiny_sample()
    converted = prepare_gt_query_sample_from_prepared(source)
    expected = torch.from_numpy(
        compute_laplacian_coordinates(
            source["gt_vertices"].numpy(), source["gt_faces"].numpy(), "uniform"
        )
    ).float()

    validate_gt_query_contract(converted)
    torch.testing.assert_close(converted["vertices"], source["gt_vertices"])
    torch.testing.assert_close(converted["raw_laplacian_target"], expected)
    assert torch.count_nonzero(converted["initial_laplacian"]) == 0
    assert converted["visibility"] is None
    assert (
        converted["metadata"]["target_constructor"]
        == "direct_gt_graph_sparse_uniform_laplacian"
    )
    assert converted["metadata"]["source_coarse_mesh_ignored"] is None


def test_query_training_rejects_unconverted_coarse_sample():
    with pytest.raises(ValueError, match="convert the manifest first"):
        train_multi_object([tiny_sample()], None, _config(), progress=False)


def test_query_perturbation_defaults_and_screening_safety_bound():
    settings = query_augmentation_settings({"query_training": {"enabled": True}})
    assert settings.normal_std_h == 0.0003
    assert settings.tangent_std_h == 0.0003
    assert settings.max_offset_h == 0.001

    with pytest.raises(ValueError, match="must not exceed 0.1"):
        query_augmentation_settings(
            {"query_training": {"enabled": True, "max_offset_h": 0.1001}}
        )


def test_query_perturbation_is_deterministic_bounded_and_preserves_target():
    sample = _gt_query_sample()
    raw_target_before = sample["raw_laplacian_target"].clone()
    target_before = sample["normalized_laplacian_target"].clone()
    first = apply_query_augmentation(sample, _settings(), base_seed=9, epoch=3)
    second = apply_query_augmentation(sample, _settings(), base_seed=9, epoch=3)
    different_seed = apply_query_augmentation(sample, _settings(), base_seed=10, epoch=3)

    torch.testing.assert_close(first["query_positions"], second["query_positions"])
    assert not torch.equal(first["query_positions"], different_seed["query_positions"])
    torch.testing.assert_close(first["raw_laplacian_target"], raw_target_before)
    torch.testing.assert_close(first["normalized_laplacian_target"], target_before)
    assert first["query_positions"].shape == sample["vertices"].shape
    assert first["query_positions"].dtype == sample["vertices"].dtype
    assert first["query_positions"].device == sample["vertices"].device
    assert int(first["query_is_exact"].sum()) == 1
    offsets = first["query_positions"] - sample["vertices"]
    assert torch.count_nonzero(offsets[first["query_is_exact"]]) == 0
    assert torch.count_nonzero(offsets[~first["query_is_exact"]]) > 0
    assert torch.all(
        torch.linalg.vector_norm(offsets, dim=-1)
        <= 0.001 * sample["local_edge_length"] + 1e-7
    )
    diagnostics = first["query_perturbation_diagnostics"]
    assert diagnostics["max_offset_norm_over_h"] <= 0.001 + 1e-7
    assert diagnostics["bound_violations"] == 0
    assert diagnostics["invalid_or_zero_h_nonzero_offsets"] == 0


def test_normal_and_tangent_query_components_follow_vertex_frame():
    sample = _gt_query_sample()
    normal_only = apply_query_augmentation(
        sample,
        _settings(normal_std_h=0.0003, tangent_std_h=0.0, exact_fraction=0.25),
        base_seed=3,
        epoch=1,
    )
    tangent_only = apply_query_augmentation(
        sample,
        _settings(normal_std_h=0.0, tangent_std_h=0.0003, exact_fraction=0.25),
        base_seed=3,
        epoch=1,
    )
    normals = sample["vertex_normals"]
    normal_offsets = normal_only["query_positions"] - sample["vertices"]
    tangent_offsets = tangent_only["query_positions"] - sample["vertices"]
    assert (
        torch.linalg.vector_norm(
            torch.linalg.cross(normal_offsets, normals, dim=-1), dim=-1
        ).max()
        < 1e-6
    )
    assert torch.abs((tangent_offsets * normals).sum(dim=-1)).max() < 1e-6


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_combined_query_offset_uses_local_limits_and_zeros_isolated_vertices(dtype):
    vertices = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
            [8.0, 8.0, 8.0],
        ],
        dtype=dtype,
    )
    local_h = torch.tensor([1.0, 1.0, 2.0, 2.0, 0.0], dtype=dtype)
    target = torch.arange(15, dtype=dtype).reshape(5, 3)
    sample = {
        "sample_id": "local_scales",
        "vertices": vertices,
        "vertex_normals": torch.tensor(
            [[0.0, 0.0, 1.0]] * 5, dtype=dtype
        ),
        "local_edge_length": local_h,
        "valid_scale_mask": torch.tensor([True, True, True, True, False]),
        "initial_laplacian": torch.ones_like(vertices),
        "raw_laplacian_target": target,
        "normalized_laplacian_target": target.square(),
    }
    augmented = apply_query_augmentation(
        sample,
        _settings(normal_std_h=1.0, tangent_std_h=1.0),
        base_seed=17,
        epoch=4,
    )

    offsets = augmented["query_positions"] - vertices
    norms = torch.linalg.vector_norm(offsets, dim=-1)
    perturbed = ~augmented["query_is_exact"]
    valid_perturbed = perturbed & sample["valid_scale_mask"]
    assert augmented["query_positions"].shape == vertices.shape
    assert augmented["query_positions"].dtype == dtype
    assert augmented["query_positions"].device == vertices.device
    assert torch.all(norms <= 0.001 * local_h + 1e-7)
    torch.testing.assert_close(
        norms[valid_perturbed] / local_h[valid_perturbed],
        torch.full_like(norms[valid_perturbed], 0.001),
        rtol=2e-4,
        atol=1e-7,
    )
    assert torch.count_nonzero(offsets[-1]).item() == 0
    assert augmented["query_perturbation_diagnostics"]["invalid_or_zero_h_vertices"] == 1
    assert (
        augmented["query_perturbation_diagnostics"][
            "invalid_or_zero_h_nonzero_offsets"
        ]
        == 0
    )
    torch.testing.assert_close(augmented["raw_laplacian_target"], target)
    torch.testing.assert_close(
        augmented["normalized_laplacian_target"], target.square()
    )


def test_float32_rounding_cannot_push_realised_offset_outside_local_bound():
    vertices = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0001, 1.0, 1.0],
            [1.0, 1.0001, 1.0],
            [1.0001, 1.0001, 1.0],
        ],
        dtype=torch.float32,
    )
    local_h = torch.full((4,), 1e-4, dtype=torch.float32)
    sample = {
        "sample_id": "float32_rounding",
        "vertices": vertices,
        "vertex_normals": torch.tensor([[0.0, 0.0, 1.0]] * 4),
        "local_edge_length": local_h,
        "valid_scale_mask": torch.ones(4, dtype=torch.bool),
        "initial_laplacian": torch.zeros_like(vertices),
    }

    augmented = apply_query_augmentation(
        sample,
        _settings(normal_std_h=1.0, tangent_std_h=1.0),
        base_seed=23,
        epoch=1,
    )
    realised_norm = torch.linalg.vector_norm(
        augmented["query_positions"] - vertices, dim=-1
    )

    assert torch.all(realised_norm / local_h <= 0.001 + 1e-7)
    assert augmented["query_perturbation_diagnostics"]["bound_violations"] == 0


def test_fourier_position_encoding_has_expected_dimension_and_finite_values():
    encoder = FourierPositionEncoding(num_frequencies=4, include_input=True)
    positions = torch.tensor([[0.0, -0.5, 1.0], [0.25, 0.75, -1.0]])
    encoded = encoder(positions)
    assert encoded.shape == (2, 27)
    assert torch.isfinite(encoded).all()


def test_zero_frequency_position_encoding_is_raw_xyz():
    encoder = FourierPositionEncoding(num_frequencies=0, include_input=True)
    positions = torch.tensor([[0.0, -0.5, 1.0], [0.25, 0.75, -1.0]])
    encoded = encoder(positions)
    assert encoder.output_dim == 3
    torch.testing.assert_close(encoded, positions, rtol=0.0, atol=0.0)


def test_query_fourier_model_cannot_copy_initial_laplacian():
    sample = _gt_query_sample()
    sample = apply_query_augmentation(sample, _settings(), base_seed=2, epoch=1)
    model = LearnedLaplacianModel(
        image_feature_dim=8,
        hidden_dim=16,
        num_graph_layers=1,
        geometry_mode="query_fourier",
        position_num_frequencies=3,
    ).eval()
    changed = dict(sample)
    changed["initial_laplacian"] = torch.randn_like(sample["initial_laplacian"]) * 1e6
    with torch.no_grad():
        reference = model(sample).predicted_laplacian
        altered = model(changed).predicted_laplacian
    torch.testing.assert_close(reference, altered, rtol=0.0, atol=0.0)


def test_query_fourier_model_accepts_prediction_mesh_vertices_as_inference_queries():
    coarse_sample = tiny_sample()
    model = LearnedLaplacianModel(
        image_feature_dim=8,
        hidden_dim=16,
        num_graph_layers=1,
        geometry_mode="query_fourier",
        position_num_frequencies=3,
    ).eval()
    assert "query_positions" not in coarse_sample
    with torch.no_grad():
        output = model(coarse_sample)
    assert output.predicted_laplacian.shape == coarse_sample["vertices"].shape
    assert torch.isfinite(output.predicted_laplacian).all()


def test_query_training_records_exact_and_perturbed_losses(tmp_path):
    train = _gt_query_sample("train_gt")
    validation = _gt_query_sample("validation_gt")
    result = train_multi_object(
        [train], [validation], _config(), output_dir=tmp_path, progress=False
    )
    record = result.history[0]
    for name in (
        "train_exact_query_loss",
        "train_perturbed_query_loss",
        "validation_exact_query_loss",
        "validation_perturbed_query_loss",
    ):
        assert record[name] is not None
        assert math.isfinite(record[name])
    assert result.per_object_metrics["validation"]["validation_gt"]["exact_query_loss"] >= 0
    assert result.per_object_metrics["validation"]["validation_gt"]["perturbed_query_loss"] >= 0
    diagnostics = result.per_object_metrics["validation"]["validation_gt"][
        "query_perturbation"
    ]
    assert diagnostics["max_offset_norm_over_h"] <= 0.001 + 1e-7
    assert diagnostics["bound_violations"] == 0


def test_float32_rounding_guard_also_holds_for_screening_support():
    sample = _gt_query_sample()
    augmented = apply_query_augmentation(
        sample,
        _settings(normal_std_h=0.003, tangent_std_h=0.003, max_offset_h=0.01),
        base_seed=7,
        epoch=7,
    )
    ratio = torch.linalg.vector_norm(
        augmented["query_positions"] - sample["vertices"], dim=-1
    ) / sample["local_edge_length"]
    assert float(ratio.max()) <= 0.0100001
    assert augmented["query_perturbation_diagnostics"]["bound_violations"] == 0
