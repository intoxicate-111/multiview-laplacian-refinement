import math

import pytest
import torch

from mlr.laplacian import compute_laplacian_coordinates
from mlr.learned_laplacian.model import FourierPositionEncoding, LearnedLaplacianModel
from mlr.learned_laplacian.multi_trainer import train_multi_object
from mlr.learned_laplacian.query_training import (
    QueryAugmentationSettings,
    apply_query_augmentation,
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
        "normal_std_h": 0.1,
        "tangent_std_h": 0.1,
        "max_offset_h": 0.25,
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
            "normal_std_h": 0.1,
            "tangent_std_h": 0.1,
            "max_offset_h": 0.25,
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


def test_query_perturbation_is_deterministic_bounded_and_preserves_target():
    sample = _gt_query_sample()
    target_before = sample["normalized_laplacian_target"].clone()
    first = apply_query_augmentation(sample, _settings(), base_seed=9, epoch=3)
    second = apply_query_augmentation(sample, _settings(), base_seed=9, epoch=3)

    torch.testing.assert_close(first["query_positions"], second["query_positions"])
    torch.testing.assert_close(first["normalized_laplacian_target"], target_before)
    assert int(first["query_is_exact"].sum()) == 1
    offsets = first["query_positions"] - sample["vertices"]
    assert torch.count_nonzero(offsets[first["query_is_exact"]]) == 0
    assert torch.count_nonzero(offsets[~first["query_is_exact"]]) > 0
    assert torch.all(
        torch.linalg.vector_norm(offsets, dim=-1)
        <= 0.25 * sample["local_edge_length"] + 1e-7
    )


def test_normal_and_tangent_query_components_follow_vertex_frame():
    sample = _gt_query_sample()
    normal_only = apply_query_augmentation(
        sample,
        _settings(normal_std_h=0.1, tangent_std_h=0.0, exact_fraction=0.25),
        base_seed=3,
        epoch=1,
    )
    tangent_only = apply_query_augmentation(
        sample,
        _settings(normal_std_h=0.0, tangent_std_h=0.1, exact_fraction=0.25),
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


def test_fourier_position_encoding_has_expected_dimension_and_finite_values():
    encoder = FourierPositionEncoding(num_frequencies=4, include_input=True)
    positions = torch.tensor([[0.0, -0.5, 1.0], [0.25, 0.75, -1.0]])
    encoded = encoder(positions)
    assert encoded.shape == (2, 27)
    assert torch.isfinite(encoded).all()


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
