from __future__ import annotations

import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from mlr.coarse_lap_oracle import (
    apply_uniform_laplacian_transpose,
    build_uniform_laplacian_data,
)
from mlr.io import load_mesh

from .canonical_pipeline import (
    CanonicalRecoveryInputs,
    canonical_current_graph_recovery_inputs,
    current_uniform_laplacian_raw,
)
from .evaluation import reconstruct_and_evaluate
from .graph_layers import faces_to_edge_index
from .multi_dataset import PreparedMeshDataset
from .target_scaling import (
    mean_incident_edge_length,
    normalize_laplacian_by_edge_scale,
)


@dataclass(frozen=True)
class AnalyticIdentityInputs:
    """Current-graph identity target before and after canonical conversion."""

    delta_identity_raw: torch.Tensor
    delta_identity_hat: torch.Tensor
    recovery: CanonicalRecoveryInputs


def analytic_current_graph_identity_inputs(
    current_vertices: torch.Tensor | np.ndarray,
    current_faces: torch.Tensor | np.ndarray,
    visibility: torch.Tensor,
    confidence_prediction: torch.Tensor | np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> AnalyticIdentityInputs:
    """Build ``L_current @ X0`` without GT and exercise canonical h² recovery."""

    vertices = torch.as_tensor(current_vertices)
    faces = torch.as_tensor(current_faces, dtype=torch.long, device=vertices.device)
    edge_index = faces_to_edge_index(faces, int(vertices.shape[0]))
    h_current = mean_incident_edge_length(vertices, edge_index, eps=epsilon)
    delta_identity_raw = torch.as_tensor(
        current_uniform_laplacian_raw(vertices, faces),
        dtype=vertices.dtype,
        device=vertices.device,
    )
    delta_identity_hat = normalize_laplacian_by_edge_scale(
        delta_identity_raw,
        h_current,
        eps=epsilon,
        valid_scale_mask=h_current > 0,
    )
    recovery = canonical_current_graph_recovery_inputs(
        vertices,
        faces,
        delta_identity_hat,
        visibility,
        torch.as_tensor(confidence_prediction),
        epsilon=epsilon,
    )
    torch.testing.assert_close(recovery.h_current, h_current, rtol=0.0, atol=0.0)
    return AnalyticIdentityInputs(
        delta_identity_raw=delta_identity_raw,
        delta_identity_hat=delta_identity_hat,
        recovery=recovery,
    )


def run_current_graph_identity_diagnostic(
    canonical_run_dir: str | Path,
    expanded_manifest: str | Path,
    output_dir: str | Path,
    *,
    sample_id: str | None = None,
    include_unit_weight_control: bool = True,
) -> dict[str, Any]:
    """Run the one-mesh analytic identity diagnostic without target transfer."""

    run = Path(canonical_run_dir).resolve()
    manifest = Path(expanded_manifest).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = _read_json(run / "config.json")
    epsilon = float(config.get("target_scaling", {}).get("epsilon", 1e-12))
    if epsilon != 1e-12:
        raise ValueError("Canonical identity diagnostic requires epsilon=1e-12.")

    dataset = PreparedMeshDataset.from_manifest(manifest, "validation")
    selected_id = sample_id or dataset.sample_ids[0]
    if selected_id not in dataset.sample_ids:
        raise ValueError(f"Unknown validation sample_id: {selected_id}")
    sample = dataset.load_static(dataset.sample_ids.index(selected_id))
    vertices = torch.as_tensor(sample["vertices"])
    faces = torch.as_tensor(sample["faces"], dtype=torch.long)
    visibility = torch.as_tensor(sample["visibility_backface_and_occlusion"])

    confidence_source = (
        run
        / "recovered_meshes"
        / selected_id
        / "main_confidence"
        / "per_vertex_diagnostics.npz"
    )
    with np.load(confidence_source) as cached:
        confidence = np.asarray(cached["confidence_prediction"], dtype=np.float32)
        cached_h = np.asarray(cached["h_current"])
        cached_visible = np.asarray(cached["visible"], dtype=bool)
        cached_weight = np.asarray(cached["weight"])
    identity = analytic_current_graph_identity_inputs(
        vertices,
        faces,
        visibility,
        confidence,
        epsilon=epsilon,
    )
    recovery = identity.recovery
    _verify_cached_recovery_contract(recovery, cached_h, cached_visible, cached_weight)

    reconstruction = dict(config.get("recovery", {}))
    unseen_anchor_weight = float(reconstruction.get("unseen_anchor_weight", 0.0))
    reconstruction.update(
        {
            "dense_vertex_limit": 5000,
            "chamfer_samples": 3000,
            "metric_seed": 7,
            "evaluate_oracle": False,
        }
    )
    evaluation_sample = dict(sample)
    for key in (
        "laplacian_target",
        "raw_laplacian_target",
        "normalized_laplacian_target",
    ):
        evaluation_sample.pop(key, None)
    evaluation_sample["local_edge_length"] = recovery.h_current
    evaluation_sample["valid_scale_mask"] = recovery.h_current > 0

    variants: list[tuple[str, np.ndarray]] = [
        ("visibility_times_confidence", recovery.weight.numpy().astype(np.float64))
    ]
    if include_unit_weight_control:
        variants.append(("unit_weight_all_rows", np.ones(len(vertices), dtype=np.float64)))

    initial_vertices = vertices.numpy().astype(np.float64)
    face_array = faces.numpy().astype(np.int64)
    delta_identity_raw = identity.delta_identity_raw.numpy()
    delta_solver_raw = recovery.delta_pred_raw.numpy()
    current_laplacian = current_uniform_laplacian_raw(vertices, faces)
    before_residual = current_laplacian - delta_identity_raw
    solver_residual_before = current_laplacian - delta_solver_raw
    rows: list[dict[str, Any]] = []
    for variant, laplacian_weight in variants:
        variant_dir = output / "variants" / variant
        metrics = reconstruct_and_evaluate(
            evaluation_sample,
            recovery.delta_pred_raw,
            variant_dir,
            reconstruction,
            normalized_prediction=identity.delta_identity_hat,
            edge_scale_epsilon=epsilon,
            laplacian_weight=laplacian_weight,
            unseen_anchor_weight=unseen_anchor_weight,
            evaluate_laplacian_prediction=False,
            evaluate_initial_geometry=True,
            solver_confidence=np.ones(len(vertices), dtype=np.float64),
        )
        recovered = load_mesh(variant_dir / "predicted_refined.obj")
        if not np.array_equal(recovered.faces, face_array):
            raise RuntimeError("Recovery changed face connectivity or ordering.")
        row, arrays = _variant_metrics(
            variant,
            initial_vertices,
            recovered.vertices,
            face_array,
            delta_identity_raw,
            delta_solver_raw,
            before_residual,
            solver_residual_before,
            laplacian_weight,
            metrics,
            reconstruction,
        )
        rows.append(row)
        np.save(variant_dir / "delta_identity_raw.npy", delta_identity_raw)
        np.save(variant_dir / "delta_identity_hat.npy", identity.delta_identity_hat.numpy())
        np.save(variant_dir / "delta_current_raw.npy", recovery.delta_current_raw.numpy())
        np.savez_compressed(
            variant_dir / "per_vertex_debug.npz",
            h_current=recovery.h_current.numpy(),
            confidence_prediction=confidence,
            visible=recovery.visible.numpy(),
            laplacian_weight=laplacian_weight,
            displacement=arrays["displacement"],
            differential_residual_before=before_residual,
            differential_residual_after=arrays["differential_residual_after"],
            solver_residual_before=solver_residual_before,
            solver_residual_after=arrays["solver_residual_after"],
        )

    roundtrip_error = np.linalg.norm(
        recovery.delta_pred_raw.numpy() - delta_identity_raw, axis=1
    )
    primary = rows[0]
    bbox_diagonal = float(
        np.linalg.norm(initial_vertices.max(axis=0) - initial_vertices.min(axis=0))
    )
    drift_tolerance = max(1e-9, bbox_diagonal * 1e-6)
    case_a = bool(
        primary["max_displacement"] <= drift_tolerance
        and primary["flipped_faces"] == 0
        and primary["new_degenerate_faces"] == 0
        and abs(primary["recovered_chamfer"] - primary["initial_chamfer"])
        <= drift_tolerance
    )
    checks = {
        "laplacian_built_from_selected_expanded_faces": True,
        "identity_uses_exact_selected_x0": True,
        "gt_vertices_used_for_target": False,
        "gt_laplacian_transferred_or_interpolated": False,
        "placeholder_expanded_target_loaded_by_evaluation": False,
        "h_current_recomputed": True,
        "epsilon": epsilon,
        "canonical_denormalization_count": 1,
        "vertex_order_unchanged": all(row["vertex_order_unchanged"] for row in rows),
        "faces_unchanged": all(row["faces_unchanged"] for row in rows),
        "same_recovery_entry_point": "reconstruct_and_evaluate",
        "cached_h_max_abs_difference": float(
            np.max(np.abs(recovery.h_current.numpy() - cached_h))
        ),
        "roundtrip_mean_vector_l2": float(roundtrip_error.mean()),
        "roundtrip_max_vector_l2": float(roundtrip_error.max()),
        "solver_input_residual_mean_vector_l2": primary[
            "solver_residual_before_mean_vector_l2"
        ],
        "solver_input_residual_max_vector_l2": primary[
            "solver_residual_before_max_vector_l2"
        ],
    }
    summary = {
        "experiment": "correspondence_free_analytic_current_graph_identity",
        "case": "A" if case_a else "B",
        "interpretation": (
            "The existing recovery path preserves a self-consistent current-graph "
            "target; failure evidence shifts toward learned prediction / cross-graph transfer."
            if case_a
            else "The fixed-step sparse Adam recovery amplifies float32 h2 round-trip "
            "noise even though X0 is already a self-consistent solution."
        ),
        "smallest_concrete_cause": (
            None
            if case_a
            else {
                "component": "sparse recovery optimizer convergence",
                "mechanism": (
                    "The canonical float32 h2 round trip leaves a tiny nonzero residual. "
                    "Sparse Adam has no stationary-point tolerance, applies finite updates "
                    "for 200 iterations, and raises rather than reduces the objective."
                ),
                "initial_objective": primary["objective_initial_total"],
                "final_objective": primary["objective_total"],
                "estimated_first_adam_step_max": primary[
                    "estimated_first_adam_step_max"
                ],
            }
        ),
        "selected_sample_id": selected_id,
        "selection_rule": "first Sofa50 validation sample in expanded manifest",
        "canonical_run_dir": str(run),
        "expanded_manifest": str(manifest),
        "confidence_source": str(confidence_source),
        "solver_config": reconstruction,
        "unseen_anchor_weight": unseen_anchor_weight,
        "drift_tolerance": drift_tolerance,
        "checks": checks,
        "variants": rows,
    }
    shutil.copyfile(run / "config.json", output / "canonical_config.json")
    shutil.copyfile(manifest, output / "expanded_manifest.json")
    _write_csv(output / "metrics.csv", rows)
    _write_json(output / "summary.json", summary)
    (output / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _verify_cached_recovery_contract(
    recovery: CanonicalRecoveryInputs,
    cached_h: np.ndarray,
    cached_visible: np.ndarray,
    cached_weight: np.ndarray,
) -> None:
    np.testing.assert_allclose(recovery.h_current.numpy(), cached_h, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(recovery.visible.numpy(), cached_visible)
    np.testing.assert_allclose(recovery.weight.numpy(), cached_weight, rtol=0.0, atol=0.0)


def _variant_metrics(
    variant: str,
    initial: np.ndarray,
    recovered: np.ndarray,
    faces: np.ndarray,
    delta_identity_raw: np.ndarray,
    delta_solver_raw: np.ndarray,
    before_residual: np.ndarray,
    solver_residual_before: np.ndarray,
    laplacian_weight: np.ndarray,
    metrics: Mapping[str, Any],
    reconstruction: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    displacement = np.linalg.norm(recovered - initial, axis=1)
    after_residual = current_uniform_laplacian_raw(
        torch.from_numpy(recovered), torch.from_numpy(faces)
    ) - delta_identity_raw
    solver_residual_after = current_uniform_laplacian_raw(
        torch.from_numpy(recovered), torch.from_numpy(faces)
    ) - delta_solver_raw
    before_error = np.linalg.norm(before_residual, axis=1)
    after_error = np.linalg.norm(after_residual, axis=1)
    topology = _topology_metrics(initial, recovered, faces)
    geometry = metrics["geometry"]
    initial_geometry = geometry["coarse"]
    recovered_geometry = geometry["predicted"]
    final_terms = metrics["reconstruction"]["predicted_final_terms"]
    operator = build_uniform_laplacian_data(faces, len(initial))
    lambda_lap = float(reconstruction.get("lambda_lap", 1.0))
    learning_rate = float(reconstruction.get("learning_rate", 0.01))
    initial_gradient = lambda_lap * (2.0 / len(initial)) * (
        apply_uniform_laplacian_transpose(
            solver_residual_before * laplacian_weight[:, None], operator
        )
    )
    first_adam_update = learning_rate * initial_gradient / (
        np.abs(initial_gradient) + 1e-8
    )
    first_adam_displacement = np.linalg.norm(first_adam_update, axis=1)
    initial_lap_loss = float(
        np.mean(
            np.sum(
                np.square(solver_residual_before) * laplacian_weight[:, None],
                axis=1,
            )
        )
    )
    initial_total = lambda_lap * initial_lap_loss
    final_total = float(final_terms["loss"])
    row = {
        "variant": variant,
        "mean_displacement": float(displacement.mean()),
        "median_displacement": float(np.median(displacement)),
        "p95_displacement": float(np.quantile(displacement, 0.95)),
        "max_displacement": float(displacement.max()),
        "differential_before_mean_vector_l2": float(before_error.mean()),
        "differential_before_max_vector_l2": float(before_error.max()),
        "differential_after_mean_vector_l2": float(after_error.mean()),
        "differential_after_max_vector_l2": float(after_error.max()),
        "solver_residual_before_mean_vector_l2": float(
            np.linalg.norm(solver_residual_before, axis=1).mean()
        ),
        "solver_residual_before_max_vector_l2": float(
            np.linalg.norm(solver_residual_before, axis=1).max()
        ),
        "solver_residual_after_mean_vector_l2": float(
            np.linalg.norm(solver_residual_after, axis=1).mean()
        ),
        "solver_residual_after_max_vector_l2": float(
            np.linalg.norm(solver_residual_after, axis=1).max()
        ),
        "objective_weighted_laplacian": float(
            final_terms.get("weighted_lap_loss", final_terms.get("laplacian", 0.0))
        ),
        "objective_anchor": float(
            final_terms.get("weighted_anchor_loss", final_terms.get("anchor", 0.0))
        ),
        "objective_unseen_anchor": float(
            final_terms.get(
                "weighted_unseen_anchor_loss", final_terms.get("unseen_anchor", 0.0)
            )
        ),
        "objective_edge": float(final_terms.get("edge", 0.0)),
        "objective_initial_total": initial_total,
        "objective_total": final_total,
        "objective_growth_factor": final_total / max(initial_total, 1e-300),
        "initial_gradient_mean_vector_l2": float(
            np.linalg.norm(initial_gradient, axis=1).mean()
        ),
        "initial_gradient_max_vector_l2": float(
            np.linalg.norm(initial_gradient, axis=1).max()
        ),
        "estimated_first_adam_step_mean": float(first_adam_displacement.mean()),
        "estimated_first_adam_step_max": float(first_adam_displacement.max()),
        "initial_chamfer": float(initial_geometry["chamfer"]),
        "recovered_chamfer": float(recovered_geometry["chamfer"]),
        "initial_point_to_surface": float(
            initial_geometry["point_to_surface_bidirectional_mean"]
        ),
        "recovered_point_to_surface": float(
            recovered_geometry["point_to_surface_bidirectional_mean"]
        ),
        "initial_normal_consistency": float(initial_geometry["normal_consistency"]),
        "recovered_normal_consistency": float(
            recovered_geometry["normal_consistency"]
        ),
        "flipped_faces": topology["flipped_faces"],
        "new_degenerate_faces": topology["new_degenerate_faces"],
        "minimum_triangle_area_before": topology["minimum_triangle_area_before"],
        "minimum_triangle_area_after": topology["minimum_triangle_area_after"],
        "enabled_differential_rows": int(np.count_nonzero(laplacian_weight > 0)),
        "zero_weight_differential_rows": int(np.count_nonzero(laplacian_weight <= 0)),
        "vertex_order_unchanged": bool(recovered.shape == initial.shape),
        "faces_unchanged": True,
        "solver": metrics["reconstruction"]["predicted_solver"],
    }
    return row, {
        "displacement": displacement,
        "differential_residual_after": after_residual,
        "solver_residual_after": solver_residual_after,
    }


def _topology_metrics(
    initial: np.ndarray, recovered: np.ndarray, faces: np.ndarray
) -> dict[str, Any]:
    before = np.cross(
        initial[faces[:, 1]] - initial[faces[:, 0]],
        initial[faces[:, 2]] - initial[faces[:, 0]],
    )
    after = np.cross(
        recovered[faces[:, 1]] - recovered[faces[:, 0]],
        recovered[faces[:, 2]] - recovered[faces[:, 0]],
    )
    before_area = 0.5 * np.linalg.norm(before, axis=1)
    after_area = 0.5 * np.linalg.norm(after, axis=1)
    before_degenerate = before_area <= 5e-15
    after_degenerate = after_area <= 5e-15
    return {
        "flipped_faces": int(np.sum(np.einsum("ij,ij->i", before, after) < 0)),
        "new_degenerate_faces": int(np.sum(after_degenerate & ~before_degenerate)),
        "minimum_triangle_area_before": float(before_area.min()),
        "minimum_triangle_area_after": float(after_area.min()),
    }


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Correspondence-free current-graph identity diagnostic",
        "",
        f"Selected validation mesh: `{summary['selected_sample_id']}` (first validation "
        "sample in the expanded manifest). The analytic target is built only as "
        "`L_current @ X0`, normalized with current-graph h², and denormalized exactly "
        "once by the canonical helper. GT geometry is used only for final geometry metrics.",
        "",
        "| Variant | Mean disp. | Median | P95 | Max | Diff. mean before | Diff. mean after | Initial Chamfer | Recovered Chamfer | Flips | New degeneracies |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["variants"]:
        lines.append(
            f"| {row['variant']} | {_fmt(row['mean_displacement'])} | "
            f"{_fmt(row['median_displacement'])} | {_fmt(row['p95_displacement'])} | "
            f"{_fmt(row['max_displacement'])} | "
            f"{_fmt(row['differential_before_mean_vector_l2'])} | "
            f"{_fmt(row['differential_after_mean_vector_l2'])} | "
            f"{_fmt(row['initial_chamfer'])} | {_fmt(row['recovered_chamfer'])} | "
            f"{row['flipped_faces']} | {row['new_degenerate_faces']} |"
        )
    primary = summary["variants"][0]
    lines.extend(
        [
            "",
            "## GT-only geometry diagnostics",
            "",
            "| Variant | Initial P2S | Recovered P2S | Initial normal | Recovered normal |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["variants"]:
        lines.append(
            f"| {row['variant']} | {_fmt(row['initial_point_to_surface'])} | "
            f"{_fmt(row['recovered_point_to_surface'])} | "
            f"{_fmt(row['initial_normal_consistency'])} | "
            f"{_fmt(row['recovered_normal_consistency'])} |"
        )
    lines.extend(
        [
            "",
            "For the primary variant, identity-target differential residual mean/max "
            f"changes from `{_fmt(primary['differential_before_mean_vector_l2'])}` / "
            f"`{_fmt(primary['differential_before_max_vector_l2'])}` to "
            f"`{_fmt(primary['differential_after_mean_vector_l2'])}` / "
            f"`{_fmt(primary['differential_after_max_vector_l2'])}`.",
            "",
            "## Objective and validity",
            "",
            f"Primary final objective: total `{_fmt(primary['objective_total'])}`, "
            f"weighted Laplacian `{_fmt(primary['objective_weighted_laplacian'])}`, "
            f"anchor `{_fmt(primary['objective_anchor'])}`, unseen anchor "
            f"`{_fmt(primary['objective_unseen_anchor'])}`, edge "
            f"`{_fmt(primary['objective_edge'])}`.",
            f"The objective starts at `{_fmt(primary['objective_initial_total'])}`. "
            "The exact first-step sparse-Adam formula predicts mean/max displacement "
            f"`{_fmt(primary['estimated_first_adam_step_mean'])}` / "
            f"`{_fmt(primary['estimated_first_adam_step_max'])}` from round-off alone.",
            "",
            f"Minimum triangle area: `{_fmt(primary['minimum_triangle_area_before'])}` "
            f"before and `{_fmt(primary['minimum_triangle_area_after'])}` after. "
            f"Normal consistency: `{_fmt(primary['initial_normal_consistency'])}` "
            f"before and `{_fmt(primary['recovered_normal_consistency'])}` after.",
            "",
            "## Interpretation",
            "",
            f"**Case {summary['case']}.** {summary['interpretation']}",
        ]
    )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    return f"{float(value):.8g}"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_sanitize(value), indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value
