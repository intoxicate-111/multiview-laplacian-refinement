from __future__ import annotations

import torch

from mlr.learned_laplacian.image_encoder import ImageFeatureConstructor
from mlr.learned_laplacian.model import LearnedLaplacianModel
from mlr.learned_laplacian.multi_trainer import _build_model


def test_gaussian_feature_constructor_is_fixed_depthwise_blur():
    constructor = ImageFeatureConstructor(
        mode="gaussian_blur", gaussian_kernel_size=5, gaussian_sigma=1.0
    )
    feature_maps = torch.zeros((2, 3, 9, 9))
    feature_maps[:, :, 4, 4] = 1.0
    blurred = constructor(feature_maps)

    assert blurred.shape == feature_maps.shape
    assert tuple(constructor.parameters()) == ()
    torch.testing.assert_close(blurred[0, 0], blurred[1, 2])
    torch.testing.assert_close(blurred[0, 0].sum(), torch.tensor(1.0))
    assert 0.0 < float(blurred[0, 0, 4, 4]) < 1.0


def test_high_frequency_constructor_concatenates_original_and_residual():
    constructor = ImageFeatureConstructor(
        mode="original_plus_high_frequency",
        gaussian_kernel_size=5,
        gaussian_sigma=1.0,
    )
    feature_maps = torch.randn((2, 4, 11, 13))
    output = constructor(feature_maps)
    blur = ImageFeatureConstructor(
        mode="gaussian_blur", gaussian_kernel_size=5, gaussian_sigma=1.0
    )(feature_maps)

    assert output.shape == (2, 8, 11, 13)
    torch.testing.assert_close(output[:, :4], feature_maps)
    torch.testing.assert_close(output[:, 4:], feature_maps - blur)


def test_feature_modes_adjust_only_projected_image_input_dimension():
    common = {
        "image_feature_dim": 8,
        "hidden_dim": 16,
        "num_graph_layers": 2,
        "predict_confidence": True,
    }
    original = LearnedLaplacianModel(**common)
    blurred = LearnedLaplacianModel(
        **common, image_feature_construction_mode="gaussian_blur"
    )
    high = LearnedLaplacianModel(
        **common, image_feature_construction_mode="original_plus_high_frequency"
    )

    assert original.projected_image_feature_dim == blurred.projected_image_feature_dim == 8
    assert high.projected_image_feature_dim == 16
    assert original.predictor.input_mlp[0].in_features == blurred.predictor.input_mlp[0].in_features
    assert high.predictor.input_mlp[0].in_features == original.predictor.input_mlp[0].in_features + 8
    assert len(original.predictor.blocks) == len(blurred.predictor.blocks) == len(high.predictor.blocks) == 2
    assert original.predictor.input_mlp[0].out_features == high.predictor.input_mlp[0].out_features == 16


def test_build_model_reads_feature_construction_config():
    model = _build_model(
        {
            "image_encoder": {
                "feature_dim": 8,
                "feature_construction": {
                    "mode": "original_plus_high_frequency",
                    "kernel_size": 5,
                    "sigma": 1.0,
                },
            },
            "model": {"hidden_dim": 16, "num_graph_layers": 1},
        },
        None,
        False,
    )
    architecture = model.architecture_config()
    assert architecture["image_feature_construction_mode"] == "original_plus_high_frequency"
    assert architecture["image_gaussian_kernel_size"] == 5
    assert architecture["image_gaussian_sigma"] == 1.0
    assert model.projected_image_feature_dim == 16


def _multiview_sample() -> dict[str, torch.Tensor | tuple[int, int]]:
    vertices = torch.tensor(
        [
            [-0.20, -0.20, 1.0],
            [0.20, -0.20, 1.0],
            [0.20, 0.20, 1.0],
            [-0.20, 0.20, 1.0],
        ],
        dtype=torch.float32,
    )
    intrinsics = torch.tensor(
        [[8.0, 0.0, 8.0], [0.0, 8.0, 8.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    ).repeat(6, 1, 1)
    return {
        "images": torch.randn((6, 3, 16, 16)),
        "image_size": (16, 16),
        "vertices": vertices,
        "intrinsics": intrinsics,
        "extrinsics": torch.eye(4).repeat(6, 1, 1),
        "visibility": torch.ones((6, 4), dtype=torch.bool),
        "faces": torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long),
        "vertex_normals": torch.tensor([[0.0, 0.0, 1.0]]).repeat(4, 1),
        "initial_laplacian": torch.zeros((4, 3)),
    }


def test_chunked_checkpointed_views_match_full_forward_and_gradients():
    common = {
        "image_feature_dim": 8,
        "image_feature_construction_mode": "original_plus_high_frequency",
        "hidden_dim": 16,
        "num_graph_layers": 1,
        "predict_confidence": True,
    }
    torch.manual_seed(7)
    full = LearnedLaplacianModel(**common)
    chunked = LearnedLaplacianModel(
        **common,
        image_view_chunk_size=2,
        image_gradient_checkpointing=True,
    )
    chunked.load_state_dict(full.state_dict())
    sample = _multiview_sample()

    full.train()
    chunked.train()
    full_output = full(sample)
    chunked_output = chunked(sample)
    torch.testing.assert_close(
        chunked_output.predicted_laplacian,
        full_output.predicted_laplacian,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        chunked_output.aggregated_image_features,
        full_output.aggregated_image_features,
        rtol=1e-5,
        atol=1e-6,
    )

    full_output.predicted_laplacian.square().mean().backward()
    chunked_output.predicted_laplacian.square().mean().backward()
    for (full_name, full_parameter), (chunk_name, chunk_parameter) in zip(
        full.named_parameters(), chunked.named_parameters(), strict=True
    ):
        assert full_name == chunk_name
        if full_parameter.grad is None:
            assert chunk_parameter.grad is None
        else:
            torch.testing.assert_close(
                chunk_parameter.grad,
                full_parameter.grad,
                rtol=2e-4,
                atol=2e-6,
            )
