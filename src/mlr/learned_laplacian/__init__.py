"""Isolated learned per-vertex Laplacian prediction subsystem."""

from .aggregation import masked_mean_aggregate
from .dataset import load_prepared_sample, save_prepared_sample, validate_sample
from .graph_layers import LaplacianPredictor, faces_to_edge_index
from .losses import laplacian_prediction_metrics, weighted_robust_laplacian_loss
from .model import LearnedLaplacianModel, LearnedLaplacianOutput
from .projection import ProjectionResult, project_vertices, sample_vertex_features
from .sample_io import (
    corrupt_same_topology_mesh,
    prepare_same_topology_sample,
    prepare_single_object_sample,
)
from .trainer import TrainingResult, load_checkpoint, train_single_object

__all__ = [
    "LaplacianPredictor",
    "LearnedLaplacianModel",
    "LearnedLaplacianOutput",
    "ProjectionResult",
    "TrainingResult",
    "faces_to_edge_index",
    "corrupt_same_topology_mesh",
    "laplacian_prediction_metrics",
    "load_prepared_sample",
    "masked_mean_aggregate",
    "prepare_single_object_sample",
    "prepare_same_topology_sample",
    "project_vertices",
    "sample_vertex_features",
    "save_prepared_sample",
    "load_checkpoint",
    "train_single_object",
    "validate_sample",
    "weighted_robust_laplacian_loss",
]
