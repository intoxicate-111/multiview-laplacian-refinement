import torch

from mlr.learned_laplacian.graph_layers import LaplacianPredictor


def test_forward_shape_gradients_and_isolated_vertex_are_finite():
    model = LaplacianPredictor(input_dim=5, hidden_dim=16, num_graph_layers=2)
    features = torch.randn(4, 5, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    output = model(features, edge_index)
    assert output.shape == (4, 3)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_all_isolated_vertices_are_supported():
    model = LaplacianPredictor(input_dim=2, hidden_dim=8, num_graph_layers=1)
    output = model(torch.randn(3, 2), torch.empty((2, 0), dtype=torch.long))
    assert output.shape == (3, 3)
    assert torch.isfinite(output).all()
