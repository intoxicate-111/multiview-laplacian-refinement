from __future__ import annotations

import torch

from mlr.laplacian import compute_laplacian_coordinates


def tiny_sample() -> dict:
    vertices = torch.tensor(
        [
            [-0.2, -0.2, 1.0],
            [0.2, -0.2, 1.0],
            [0.0, 0.2, 1.0],
            [0.0, 0.0, 1.35],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor(
        [[0, 1, 2], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=torch.long
    )
    target_positions = vertices.clone()
    target_positions[:, 2] += torch.tensor([0.04, -0.03, 0.02, -0.05])
    initial = torch.from_numpy(
        compute_laplacian_coordinates(vertices.numpy(), faces.numpy(), "uniform")
    ).float()
    target = torch.from_numpy(
        compute_laplacian_coordinates(target_positions.numpy(), faces.numpy(), "uniform")
    ).float()
    normals = torch.nn.functional.normalize(vertices - vertices.mean(dim=0), dim=-1)
    intrinsics = torch.tensor(
        [[[8.0, 0.0, 7.5], [0.0, 8.0, 7.5], [0.0, 0.0, 1.0]]],
        dtype=torch.float32,
    )
    return {
        "sample_id": "tiny",
        "images": torch.rand((1, 3, 16, 16), generator=torch.Generator().manual_seed(3)),
        "intrinsics": intrinsics,
        "extrinsics": torch.eye(4).unsqueeze(0),
        "vertices": vertices,
        "faces": faces,
        "vertex_normals": normals,
        "initial_laplacian": initial,
        "laplacian_target": target,
        "target_confidence": torch.ones(4),
        "visibility": torch.ones((1, 4), dtype=torch.bool),
        "target_positions": target_positions,
        "gt_vertices": target_positions,
        "gt_faces": faces,
    }
