"""Multi-view Laplacian refinement experiment framework."""

from .data import Camera, Mesh, ReconstructionInput, VisibilityCache
from .datasets import load_reconstruction_input
from .coarse import (
    NvidiaInstantNGPMeshGenerator,
    OpenMVSCommandMeshGenerator,
    generate_coarse_mesh,
    write_colmap_text_model,
)
from .coarse import NvidiaNvdiffrecMeshGenerator
from .coarse_lap_oracle import CoarseGraphOracleConfig, run_coarse_graph_laplacian_oracles
from .gt_laplacian import GTLaplacianTargetConfig, refine_coarse_mesh_with_gt_laplacian
from .laplacian import compute_laplacian_coordinates, compute_laplacian_target
from .refinement import RefinementConfig, refine_mesh_with_laplacian
from .synthetic import (
    SyntheticRenderConfig,
    generate_synthetic_dataset_from_mesh,
    generate_synthetic_datasets_from_mesh_dir,
)

__all__ = [
    "Camera",
    "Mesh",
    "ReconstructionInput",
    "VisibilityCache",
    "NvidiaInstantNGPMeshGenerator",
    "NvidiaNvdiffrecMeshGenerator",
    "OpenMVSCommandMeshGenerator",
    "CoarseGraphOracleConfig",
    "RefinementConfig",
    "GTLaplacianTargetConfig",
    "SyntheticRenderConfig",
    "compute_laplacian_coordinates",
    "compute_laplacian_target",
    "load_reconstruction_input",
    "generate_synthetic_dataset_from_mesh",
    "generate_synthetic_datasets_from_mesh_dir",
    "generate_coarse_mesh",
    "run_coarse_graph_laplacian_oracles",
    "write_colmap_text_model",
    "refine_mesh_with_laplacian",
    "refine_coarse_mesh_with_gt_laplacian",
]
