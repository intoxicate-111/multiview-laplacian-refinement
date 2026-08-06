"""Isolated learned per-vertex Laplacian prediction subsystem."""

from .aggregation import masked_mean_aggregate
from .dataset import load_prepared_sample, save_prepared_sample, validate_sample
from .graph_layers import LaplacianPredictor, faces_to_edge_index
from .losses import laplacian_prediction_metrics, weighted_robust_laplacian_loss
from .model import FourierPositionEncoding, LearnedLaplacianModel, LearnedLaplacianOutput
from .multi_dataset import (
    PreparedMeshDataset,
    PreparedMeshRecord,
    validate_disjoint_splits,
)
from .multi_trainer import MultiObjectTrainingResult, train_multi_object
from .projection import ProjectionResult, project_vertices, sample_vertex_features
from .prediction_visualizer import (
    PredictionRecord,
    RunMetadata,
    VisualizationOptions,
    discover_predictions,
    discover_run_metadata,
    load_prediction_sample,
    visualize_prediction_sample,
    visualize_prediction_split,
)
from .sample_io import (
    corrupt_same_topology_mesh,
    prepare_gt_query_sample_from_prepared,
    prepare_same_topology_sample,
    prepare_single_object_sample,
)
from .trainer import TrainingResult, load_checkpoint, train_single_object
from .target_scaling import (
    EDGE_SCALE_NORMALIZED_LAPLACIAN,
    RAW_LAPLACIAN,
    denormalize_laplacian_by_edge_scale,
    graph_structure_statistics,
    incident_edge_length_and_valid_mask,
    mean_incident_edge_length,
    normalize_laplacian_by_edge_scale,
)

__all__ = [
    "LaplacianPredictor",
    "LearnedLaplacianModel",
    "LearnedLaplacianOutput",
    "FourierPositionEncoding",
    "MultiObjectTrainingResult",
    "PreparedMeshDataset",
    "PreparedMeshRecord",
    "ProjectionResult",
    "PredictionRecord",
    "RunMetadata",
    "TrainingResult",
    "VisualizationOptions",
    "EDGE_SCALE_NORMALIZED_LAPLACIAN",
    "RAW_LAPLACIAN",
    "faces_to_edge_index",
    "discover_predictions",
    "discover_run_metadata",
    "corrupt_same_topology_mesh",
    "laplacian_prediction_metrics",
    "mean_incident_edge_length",
    "load_prepared_sample",
    "load_prediction_sample",
    "masked_mean_aggregate",
    "prepare_single_object_sample",
    "prepare_gt_query_sample_from_prepared",
    "prepare_same_topology_sample",
    "project_vertices",
    "normalize_laplacian_by_edge_scale",
    "denormalize_laplacian_by_edge_scale",
    "graph_structure_statistics",
    "incident_edge_length_and_valid_mask",
    "sample_vertex_features",
    "save_prepared_sample",
    "load_checkpoint",
    "train_single_object",
    "train_multi_object",
    "validate_sample",
    "validate_disjoint_splits",
    "visualize_prediction_sample",
    "visualize_prediction_split",
    "weighted_robust_laplacian_loss",
]
