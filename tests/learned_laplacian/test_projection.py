import torch

from mlr.learned_laplacian.projection import project_vertices, sample_vertex_features


def _camera():
    intrinsics = torch.tensor(
        [[[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]]
    )
    extrinsics = torch.eye(4).unsqueeze(0)
    return intrinsics, extrinsics


def test_known_point_projects_to_expected_pixel():
    intrinsics, extrinsics = _camera()
    result = project_vertices(
        torch.tensor([[0.0, 0.0, 1.0]]), intrinsics, extrinsics, image_size=(3, 3)
    )
    torch.testing.assert_close(result.pixels[0, 0], torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(result.grid[0, 0], torch.tensor([0.0, 0.0]))
    assert result.frustum_valid[0, 0]
    assert result.renderer_visible[0, 0]
    assert result.valid[0, 0]


def test_renderer_visibility_is_intersected_without_depth_inference():
    intrinsics, extrinsics = _camera()
    result = project_vertices(
        torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        intrinsics,
        extrinsics,
        image_size=(3, 3),
        visibility=torch.tensor([[True, False]]),
    )
    assert result.frustum_valid.tolist() == [[True, True]]
    assert result.renderer_visible.tolist() == [[True, False]]
    assert result.final_valid.tolist() == [[True, False]]


def test_behind_camera_and_out_of_frame_are_invalid():
    intrinsics, extrinsics = _camera()
    result = project_vertices(
        torch.tensor([[0.0, 0.0, -1.0], [5.0, 0.0, 1.0]]),
        intrinsics,
        extrinsics,
        image_size=(3, 3),
    )
    assert not result.valid[0, 0]
    assert not result.valid[0, 1]


def test_grid_sampling_returns_expected_feature_value():
    intrinsics, extrinsics = _camera()
    feature_map = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3)
    sampled, valid, _ = sample_vertex_features(
        feature_map,
        torch.tensor([[0.0, 0.0, 1.0]]),
        intrinsics,
        extrinsics,
        image_size=(3, 3),
    )
    torch.testing.assert_close(sampled[0, 0, 0], torch.tensor(4.0))
    assert valid[0, 0]
