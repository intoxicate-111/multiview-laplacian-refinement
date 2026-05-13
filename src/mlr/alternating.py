from __future__ import annotations

from dataclasses import dataclass, field

from .data import Array, Camera, Mesh
from .laplacian import compute_laplacian_target
from .pseudo_surface import PseudoSurfaceEstimator, estimate_pseudo_surface
from .refinement import RefinementConfig, RefinementResult, refine_mesh_with_laplacian
from .visibility import update_visibility


@dataclass
class AlternatingRefinementConfig:
    num_outer_iters: int = 3
    inner: RefinementConfig = field(default_factory=RefinementConfig)
    operator_type: str = "uniform"
    update_visibility_each_iter: bool = True


@dataclass
class AlternatingRefinementResult:
    mesh: Mesh
    outer_history: list[dict[str, object]]


def alternating_refinement_loop(
    mesh: Mesh,
    images: list[Array] | None,
    cameras: list[Camera],
    masks: list[Array] | None = None,
    estimator: PseudoSurfaceEstimator | None = None,
    config: AlternatingRefinementConfig | None = None,
) -> AlternatingRefinementResult:
    config = config or AlternatingRefinementConfig()
    current = mesh
    outer_history: list[dict[str, object]] = []

    for outer_iter in range(config.num_outer_iters):
        visibility = current.visibility
        if config.update_visibility_each_iter or visibility is None:
            visibility = update_visibility(current, cameras, masks)
            current.visibility = visibility

        p_star, confidence = estimate_pseudo_surface(
            current,
            images=images,
            cameras=cameras,
            masks=masks,
            visibility=visibility,
            estimator=estimator,
        )
        delta_pseudo = compute_laplacian_target(p_star, current.faces, config.operator_type)
        result: RefinementResult = refine_mesh_with_laplacian(
            current,
            delta_target=delta_pseudo,
            confidence=confidence,
            anchors=current.vertices,
            config=config.inner,
        )
        current = result.mesh
        outer_history.append(
            {
                "outer_iter": outer_iter,
                "mean_confidence": float(confidence.mean()),
                "inner_history": result.history,
            }
        )

    return AlternatingRefinementResult(mesh=current, outer_history=outer_history)
