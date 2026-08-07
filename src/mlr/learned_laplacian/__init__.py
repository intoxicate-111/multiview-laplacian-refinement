"""Isolated learned per-vertex Laplacian prediction subsystem."""

from .aggregation import masked_mean_aggregate
from .dataset import load_prepared_sample, save_prepared_sample, validate_sample
from .graph_layers import LaplacianPredictor, faces_to_edge_index
from .canonical_pipeline import (
    CanonicalRecoveryInputs,
    canonical_current_graph_recovery_inputs,
)
from .losses import (
    confidence_calibration_metrics,
    confidence_reliability_loss,
    laplacian_prediction_metrics,
    weighted_robust_laplacian_loss,
)
from .model import FourierPositionEncoding, LearnedLaplacianModel, LearnedLaplacianOutput
from .multi_dataset import (
    PreparedMeshDataset,
    PreparedMeshRecord,
    validate_disjoint_splits,
)
from .multi_trainer import MultiObjectTrainingResult, train_multi_object
from .projection import ProjectionResult, project_vertices, sample_vertex_features
from .renderer_visibility import (
    RendererVisibilityResult,
    compute_renderer_visibility,
    vertex_visibility_from_face_id_buffer,
)
from .visibility_recovery import (
    HardVisibilityRecoveryMask,
    confidence_aware_recovery_weight,
    hard_any_view_recovery_mask,
    visibility_coverage_diagnostics,
)
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
    prediction_to_raw_laplacian,
    require_matching_laplacian_representations,
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
    "RendererVisibilityResult",
    "HardVisibilityRecoveryMask",
    "CanonicalRecoveryInputs",
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
    "compute_renderer_visibility",
    "normalize_laplacian_by_edge_scale",
    "denormalize_laplacian_by_edge_scale",
    "prediction_to_raw_laplacian",
    "require_matching_laplacian_representations",
    "graph_structure_statistics",
    "incident_edge_length_and_valid_mask",
    "sample_vertex_features",
    "vertex_visibility_from_face_id_buffer",
    "hard_any_view_recovery_mask",
    "confidence_aware_recovery_weight",
    "canonical_current_graph_recovery_inputs",
    "confidence_reliability_loss",
    "confidence_calibration_metrics",
    "visibility_coverage_diagnostics",
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
