"""Multi-view Laplacian refinement experiment framework."""

from .data import Camera, Mesh, ReconstructionInput, VisibilityCache
from .datasets import load_reconstruction_input
from .coarse import NvidiaInstantNGPMeshGenerator, generate_coarse_mesh
from .coarse import NvidiaNvdiffrecMeshGenerator
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
    "RefinementConfig",
    "GTLaplacianTargetConfig",
    "SyntheticRenderConfig",
    "compute_laplacian_coordinates",
    "compute_laplacian_target",
    "load_reconstruction_input",
    "generate_synthetic_dataset_from_mesh",
    "generate_synthetic_datasets_from_mesh_dir",
    "generate_coarse_mesh",
    "refine_mesh_with_laplacian",
    "refine_coarse_mesh_with_gt_laplacian",
]
