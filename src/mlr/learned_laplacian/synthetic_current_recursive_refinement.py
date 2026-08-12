from __future__ import annotations

import copy
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mlr.coarse_lap_oracle import apply_uniform_laplacian, build_uniform_laplacian_data
from mlr.data import Camera, Mesh
from mlr.io import load_mesh
from mlr.synthetic import SyntheticRenderConfig

from .canonical_experiment import _exact_query_sample
from .canonical_pipeline import canonical_current_graph_recovery_inputs
from .diagnostics import _amp_settings
from .evaluation import reconstruct_and_evaluate
from .graph_layers import faces_to_edge_index
from .multi_dataset import PreparedMeshDataset
from .multi_trainer import _build_model, _prepare_item_for_use, _prepare_object_static
from .renderer_visibility import compute_renderer_visibility
from .synthetic_current_comparison import _topology_change
from .synthetic_current_h2_ablation import _raw_metrics
from .target_scaling import (
    RAW_LAPLACIAN,
    incident_edge_length_and_valid_mask,
    normalize_laplacian_by_edge_scale,
)
from .trainer import load_checkpoint


ARM = "B_direct_raw_laplacian"
POLICIES = ("recomputed_opengl_960", "fixed_prepared_visibility")
GEOMETRY_FIELDS = (
    "original_initial_chamfer",
    "round_initial_chamfer",
    "reconstruction_chamfer",
    "original_initial_point_to_surface",
    "round_initial_point_to_surface",
    "reconstruction_point_to_surface",
    "original_initial_normal_consistency",
    "round_initial_normal_consistency",
    "reconstruction_normal_consistency",
)
PREDICTION_FIELDS = (
    "raw_epe",
    "raw_top_1_percent_epe",
    "raw_top_10_percent_epe",
    "raw_top_20_percent_epe",
    "raw_top_50_percent_epe",
    "raw_global_cosine",
    "prediction_to_target_raw_norm_ratio",
    "raw_residual_rms",
    "raw_residual_maximum",
    "recovery_weighted_raw_residual_rms",
)


def run_recursive_refinement_shard(
    manifest_path: str | Path,
    run_dir: str | Path,
    baseline_analysis_dir: str | Path,
    output_dir: str | Path,
    *,
    rounds: int = 3,
    shard_index: int = 0,
    shard_count: int = 1,
    device: str = "cuda",
    visibility_size: int = 960,
) -> dict[str, Any]:
    if rounds < 1:
        raise ValueError("rounds must be positive.")
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Require shard_count > 0 and 0 <= shard_index < shard_count.")
    if visibility_size < 64:
        raise ValueError("visibility_size must be at least 64.")
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Recursive refinement requires an available CUDA device.")

    manifest = Path(manifest_path).resolve()
    run = Path(run_dir).resolve()
    baseline = Path(baseline_analysis_dir).resolve()
    output = Path(output_dir).resolve()
    dataset = PreparedMeshDataset.from_manifest(manifest, "test")
    if len(dataset) != 25:
        raise ValueError(f"Expected 25 test samples, found {len(dataset)}.")
    spec = _load_b_spec(run, resolved_device)
    baseline_rows = _load_baseline_rows(
        baseline / "topk_oracle_replacement_per_sample.csv"
    )
    baseline_predictions = _load_baseline_prediction_rows(
        baseline / "prediction_per_sample.csv"
    )
    baseline_mesh_root = (
        baseline
        / "reconstruction"
        / ARM
        / "replace_000pct"
    )
    expected_ids = set(dataset.sample_ids)
    if set(baseline_rows) != expected_ids or set(baseline_predictions) != expected_ids:
        raise RuntimeError("Baseline H2 rows do not match the test manifest sample IDs.")

    rows: list[dict[str, Any]] = []
    sample_audits: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        if index % shard_count != shard_index:
            continue
        source = dataset[index]
        sample_id = str(source["sample_id"])
        metadata = dict(source.get("metadata", {}))
        original_vertices = _np(source["vertices"])
        faces = _np(source["faces"]).astype(np.int64)
        baseline_mesh = load_mesh(baseline_mesh_root / sample_id / "predicted_refined.obj")
        if not np.array_equal(baseline_mesh.faces, faces):
            raise RuntimeError(f"{sample_id}: baseline refined topology differs from source.")
        if not np.isfinite(baseline_mesh.vertices).all():
            raise RuntimeError(f"{sample_id}: baseline refined mesh is non-finite.")
        baseline_row = baseline_rows[sample_id]
        prediction_row = baseline_predictions[sample_id]
        original_initial_chamfer = float(baseline_row["initial_chamfer"])
        baseline_chamfer = float(baseline_row["reconstruction_chamfer"])
        baseline_displacement = np.linalg.norm(
            baseline_mesh.vertices - original_vertices, axis=1
        )

        for policy in POLICIES:
            rows.append(
                _round_zero_row(
                    policy,
                    sample_id,
                    metadata,
                    baseline_row,
                    prediction_row,
                    baseline_displacement,
                )
            )
            current_vertices = baseline_mesh.vertices.copy()
            previous_chamfer = baseline_chamfer
            for round_index in range(1, rounds + 1):
                current_mesh = Mesh(current_vertices, faces).ensure_normals()
                if policy == "recomputed_opengl_960":
                    visibility, visibility_audit = _dynamic_visibility(
                        current_mesh, source, visibility_size=visibility_size
                    )
                else:
                    visibility = _np(source["visibility_backface_and_occlusion"]).astype(bool)
                    visibility_audit = _visibility_audit(visibility, visibility_size=None)
                recursive_sample = build_recursive_sample(
                    source, current_mesh, visibility
                )
                inferred = _infer_recursive(recursive_sample, spec, resolved_device)
                round_dir = (
                    output
                    / "reconstruction"
                    / policy
                    / f"round_{round_index:02d}"
                    / sample_id
                )
                metrics = reconstruct_and_evaluate(
                    recursive_sample,
                    inferred["prediction_raw"],
                    round_dir,
                    _recovery_config(spec["config"]),
                    normalized_prediction=inferred["prediction_normalized"],
                    edge_scale_epsilon=inferred["epsilon"],
                    laplacian_weight=inferred["recovery_weight"],
                    unseen_anchor_weight=float(
                        spec["config"].get("recovery", {}).get(
                            "unseen_anchor_weight", 0.0
                        )
                    ),
                    evaluate_laplacian_prediction=True,
                    evaluate_initial_geometry=True,
                    solver_confidence=np.ones(len(current_vertices), dtype=np.float64),
                )
                recovered = load_mesh(round_dir / "predicted_refined.obj")
                if not np.array_equal(recovered.faces, faces):
                    raise RuntimeError(f"{sample_id}: round {round_index} changed topology.")
                if not np.isfinite(recovered.vertices).all():
                    raise RuntimeError(f"{sample_id}: round {round_index} is non-finite.")
                coarse_geometry = metrics["geometry"]["coarse"]
                refined_geometry = metrics["geometry"]["predicted"]
                round_initial_chamfer = float(coarse_geometry["chamfer"])
                reconstruction_chamfer = float(refined_geometry["chamfer"])
                if round_index == 1 and abs(round_initial_chamfer - baseline_chamfer) > 1e-6:
                    raise RuntimeError(
                        f"{sample_id}: round-0 mesh metric changed by "
                        f"{round_initial_chamfer - baseline_chamfer:.3g}."
                    )
                cumulative_topology = _topology_change(
                    original_vertices, recovered.vertices, faces
                )
                step_topology = _topology_change(
                    current_vertices, recovered.vertices, faces
                )
                step_displacement = np.linalg.norm(
                    recovered.vertices - current_vertices, axis=1
                )
                cumulative_displacement = np.linalg.norm(
                    recovered.vertices - original_vertices, axis=1
                )
                row = {
                    "policy": policy,
                    "round": round_index,
                    "sample_id": sample_id,
                    "object_id": metadata.get("object_id"),
                    "variant_index": metadata.get("variant_index"),
                    "vertex_count": len(current_vertices),
                    "original_initial_chamfer": original_initial_chamfer,
                    "round_initial_chamfer": round_initial_chamfer,
                    "reconstruction_chamfer": reconstruction_chamfer,
                    "original_initial_point_to_surface": float(
                        baseline_row["initial_point_to_surface"]
                    ),
                    "round_initial_point_to_surface": float(
                        coarse_geometry["point_to_surface_bidirectional_mean"]
                    ),
                    "reconstruction_point_to_surface": float(
                        refined_geometry["point_to_surface_bidirectional_mean"]
                    ),
                    "original_initial_normal_consistency": float(
                        baseline_row["initial_normal_consistency"]
                    ),
                    "round_initial_normal_consistency": float(
                        coarse_geometry["normal_consistency"]
                    ),
                    "reconstruction_normal_consistency": float(
                        refined_geometry["normal_consistency"]
                    ),
                    "cumulative_improved_over_original": bool(
                        reconstruction_chamfer < original_initial_chamfer
                    ),
                    "step_improved_over_previous": bool(
                        reconstruction_chamfer < round_initial_chamfer
                    ),
                    "improved_over_round0": bool(
                        reconstruction_chamfer < baseline_chamfer
                    ),
                    "previous_recorded_chamfer": previous_chamfer,
                    "cumulative_introduced_flipped_faces": int(
                        cumulative_topology["introduced_flips"]
                    ),
                    "step_introduced_flipped_faces": int(
                        step_topology["introduced_flips"]
                    ),
                    "cumulative_new_degenerate_faces": int(
                        cumulative_topology["new_degeneracies"]
                    ),
                    "step_new_degenerate_faces": int(
                        step_topology["new_degeneracies"]
                    ),
                    "mean_step_displacement": float(step_displacement.mean()),
                    "max_step_displacement": float(step_displacement.max()),
                    "mean_cumulative_displacement": float(
                        cumulative_displacement.mean()
                    ),
                    "max_cumulative_displacement": float(
                        cumulative_displacement.max()
                    ),
                    "mean_confidence": float(inferred["confidence"].mean().item()),
                    "visible_vertex_fraction": float(
                        inferred["visible"].float().mean().item()
                    ),
                    "visibility_backend": visibility_audit["backend"],
                    "visibility_raster_size": visibility_audit["raster_size"],
                    "mean_visible_views_per_vertex": visibility_audit[
                        "mean_visible_views_per_vertex"
                    ],
                    "zero_visible_vertex_fraction": visibility_audit[
                        "zero_visible_vertex_fraction"
                    ],
                    **inferred["prediction_metrics"],
                }
                rows.append(row)
                current_vertices = recovered.vertices.copy()
                previous_chamfer = reconstruction_chamfer
                print(
                    f"[{sample_id}] {policy} round={round_index}/{rounds} "
                    f"chamfer={reconstruction_chamfer:.9g} "
                    f"cumulative_improved={row['cumulative_improved_over_original']} "
                    f"step_improved={row['step_improved_over_previous']}",
                    flush=True,
                )

        sample_audits.append(
            {
                "sample_id": sample_id,
                "faces_preserved": True,
                "baseline_mesh_finite": True,
                "baseline_metric_source": str(
                    baseline / "topk_oracle_replacement_per_sample.csv"
                ),
                "baseline_chamfer": baseline_chamfer,
            }
        )

    shard_payload = {
        "format": "synthetic_current_recursive_refinement_shard_v1",
        "shard_index": shard_index,
        "shard_count": shard_count,
        "rounds": rounds,
        "policies": list(POLICIES),
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "run_dir": str(run),
        "checkpoint": str(spec["checkpoint"]),
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "baseline_analysis_dir": str(baseline),
        "visibility_size": visibility_size,
        "rows": rows,
        "sample_audits": sample_audits,
    }
    shard_path = output / "shards" / f"shard_{shard_index}.json"
    _write_json(shard_path, shard_payload)
    return shard_payload


def build_recursive_sample(
    source: Mapping[str, Any], mesh: Mesh, visibility: np.ndarray
) -> dict[str, Any]:
    """Replace every geometry-dependent model input for one recursive round."""

    if mesh.num_vertices != int(source["vertices"].shape[0]):
        raise ValueError("Recursive mesh must preserve vertex count and correspondence.")
    source_faces = _np(source["faces"]).astype(np.int64)
    if not np.array_equal(mesh.faces, source_faces):
        raise ValueError("Recursive mesh must preserve faces and vertex order.")
    if visibility.shape != (int(source["intrinsics"].shape[0]), mesh.num_vertices):
        raise ValueError("Recursive visibility must have shape [V, N].")

    excluded = {
        "_static_prepared",
        "edge_index",
        "vertex_degree",
        "vertices",
        "vertex_normals",
        "initial_laplacian",
        "laplacian_target",
        "raw_laplacian_target",
        "normalized_laplacian_target",
        "local_edge_length",
        "local_edge_scale",
        "valid_scale_mask",
        "visibility",
        "visibility_backface_only",
        "visibility_occlusion_only",
        "visibility_backface_and_occlusion",
        "query_positions",
        "query_offsets",
        "query_is_exact",
        "position_normalization_center",
        "position_normalization_scale",
    }
    sample = {
        key: (value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value))
        for key, value in source.items()
        if key not in excluded
    }
    vertices = torch.as_tensor(mesh.vertices, dtype=torch.float32)
    faces = torch.as_tensor(mesh.faces, dtype=torch.long)
    edge_index = faces_to_edge_index(faces, mesh.num_vertices)
    local_h, valid = incident_edge_length_and_valid_mask(vertices, edge_index)
    operator = build_uniform_laplacian_data(mesh.faces, mesh.num_vertices)
    initial_raw = torch.as_tensor(
        apply_uniform_laplacian(mesh.vertices, operator), dtype=torch.float32
    )
    proxy_vertices = _np(source["gt_vertices"])
    target_raw = torch.as_tensor(
        apply_uniform_laplacian(proxy_vertices, operator), dtype=torch.float32
    )
    epsilon = float(source.get("metadata", {}).get("edge_scale_epsilon", 1e-12))
    target_normalized = normalize_laplacian_by_edge_scale(
        target_raw, local_h, eps=epsilon, valid_scale_mask=valid
    )
    center = 0.5 * (vertices.amin(dim=0) + vertices.amax(dim=0))
    scale = torch.linalg.vector_norm(vertices - center, dim=-1).amax()
    if not torch.isfinite(scale) or float(scale) <= 1e-12:
        raise ValueError("Recursive mesh has invalid spatial extent.")
    visibility_tensor = torch.as_tensor(visibility, dtype=torch.bool)
    metadata = dict(sample.get("metadata", {}))
    metadata.update(
        {
            "query_geometry_role": "recursive_refined_current_graph",
            "initial_laplacian_input": "L_current@C",
            "position_normalization": "bbox_center_max_radius_recomputed_each_round",
            "edge_scale_source": "recursive_current_graph",
            "target_constructor": "delta_target=L_current@P_proxy",
        }
    )
    sample.update(
        {
            "vertices": vertices,
            "faces": faces,
            "vertex_normals": torch.as_tensor(
                mesh.ensure_normals().normals, dtype=torch.float32
            ),
            "initial_laplacian": initial_raw,
            "laplacian_target": target_raw,
            "raw_laplacian_target": target_raw,
            "normalized_laplacian_target": target_normalized,
            "target_confidence": torch.ones(mesh.num_vertices, dtype=torch.float32),
            "local_edge_length": local_h,
            "local_edge_scale": local_h.square(),
            "valid_scale_mask": valid,
            "edge_index": edge_index,
            "visibility": visibility_tensor,
            "visibility_backface_and_occlusion": visibility_tensor,
            "visibility_backface_only": None,
            "visibility_occlusion_only": None,
            "position_normalization_center": center,
            "position_normalization_scale": scale.reshape(()),
            "metadata": metadata,
        }
    )
    return sample


def merge_recursive_refinement_shards(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    rounds: int = 3,
    shard_count: int = 3,
) -> dict[str, Any]:
    manifest = Path(manifest_path).resolve()
    output = Path(output_dir).resolve()
    payloads = [
        _read_json(output / "shards" / f"shard_{index}.json")
        for index in range(shard_count)
    ]
    if [int(payload["shard_index"]) for payload in payloads] != list(range(shard_count)):
        raise RuntimeError("Recursive shard indices are incomplete or unordered.")
    controlled = {
        (
            payload["manifest_sha256"],
            payload["checkpoint_sha256"],
            int(payload["rounds"]),
            tuple(payload["policies"]),
            int(payload["visibility_size"]),
        )
        for payload in payloads
    }
    if len(controlled) != 1 or payloads[0]["manifest_sha256"] != _sha256(manifest):
        raise RuntimeError("Recursive shard contracts do not match.")
    rows = [dict(row) for payload in payloads for row in payload["rows"]]
    rows.sort(key=lambda row: (str(row["policy"]), int(row["round"]), str(row["sample_id"])))
    expected_count = len(POLICIES) * (rounds + 1) * 25
    keys = {
        (str(row["policy"]), int(row["round"]), str(row["sample_id"]))
        for row in rows
    }
    counts_match = len(rows) == expected_count and len(keys) == expected_count
    aggregates = aggregate_recursive_rows(rows, rounds=rounds)
    baseline_counts = {
        policy: next(
            int(row["cumulative_improved_over_original"])
            for row in aggregates
            if row["policy"] == policy and int(row["round"]) == 0
        )
        for policy in POLICIES
    }
    audit = {
        "passed": bool(
            counts_match
            and all(count == 19 for count in baseline_counts.values())
            and all(bool(item["faces_preserved"]) for payload in payloads for item in payload["sample_audits"])
        ),
        "row_count": len(rows),
        "expected_row_count": expected_count,
        "unique_key_count": len(keys),
        "baseline_improved_counts": baseline_counts,
        "shard_count": shard_count,
        "rounds": rounds,
        "policies": list(POLICIES),
        "manifest_sha256": _sha256(manifest),
        "checkpoint_sha256": payloads[0]["checkpoint_sha256"],
        "geometry_inputs_recomputed_each_round": [
            "vertices",
            "vertex_normals",
            "initial_laplacian",
            "local_edge_length",
            "valid_scale_mask",
            "position_normalization_center",
            "position_normalization_scale",
        ],
        "target_values_used_by_model": False,
        "topology_and_vertex_order_preserved": True,
    }
    if not audit["passed"]:
        _write_json(output / "contract_audit.json", audit)
        raise RuntimeError("Recursive refinement merge audit failed.")
    decision = _recursive_decision(aggregates)
    summary = {
        "experiment": "B direct-raw recursive current-mesh refinement",
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "arm": ARM,
        "checkpoint": payloads[0]["checkpoint"],
        "checkpoint_sha256": payloads[0]["checkpoint_sha256"],
        "baseline_analysis_dir": payloads[0]["baseline_analysis_dir"],
        "round_definition": (
            "round 0 is the completed B coarse-to-refined result; rounds 1-3 each "
            "use the previous refined mesh as the full current-graph model input"
        ),
        "policies": {
            "recomputed_opengl_960": (
                "Recompute backface+occlusion visibility from each current mesh at 960."
            ),
            "fixed_prepared_visibility": (
                "Reuse prepared CUDA visibility to isolate recursive geometry-input effects."
            ),
        },
        "contract_audit": audit,
        "aggregate": aggregates,
        "decision": decision,
    }
    _write_csv(output / "recursive_refinement_per_sample.csv", rows)
    _write_csv(output / "recursive_refinement_aggregate.csv", aggregates)
    _write_json(output / "contract_audit.json", audit)
    _write_json(output / "recursive_refinement_summary.json", summary)
    (output / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    return summary


def aggregate_recursive_rows(
    rows: Sequence[Mapping[str, Any]], *, rounds: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["policy"]), int(row["round"]))].append(row)
    output: list[dict[str, Any]] = []
    for policy in POLICIES:
        baseline = {str(row["sample_id"]): row for row in grouped[(policy, 0)]}
        if len(baseline) != 25:
            raise RuntimeError(f"Expected 25 baseline rows for {policy}.")
        baseline_success = {
            sample_id
            for sample_id, row in baseline.items()
            if bool(row["cumulative_improved_over_original"])
        }
        for round_index in range(rounds + 1):
            selected = grouped[(policy, round_index)]
            if len(selected) != 25:
                raise RuntimeError(
                    f"Expected 25 rows for {policy}/round {round_index}."
                )
            current_success = {
                str(row["sample_id"])
                for row in selected
                if bool(row["cumulative_improved_over_original"])
            }
            result: dict[str, Any] = {
                "policy": policy,
                "round": round_index,
                "sample_count": len(selected),
                **{field: _mean(selected, field) for field in GEOMETRY_FIELDS},
                "cumulative_improved_over_original": len(current_success),
                "step_improved_over_previous": int(
                    sum(bool(row["step_improved_over_previous"]) for row in selected)
                ),
                "improved_over_round0": int(
                    sum(bool(row["improved_over_round0"]) for row in selected)
                ),
                "retained_round0_successes": len(current_success & baseline_success),
                "gained_from_round0_failures": len(current_success - baseline_success),
                "lost_round0_successes": len(baseline_success - current_success),
                "cumulative_introduced_flipped_faces": int(
                    sum(int(row["cumulative_introduced_flipped_faces"]) for row in selected)
                ),
                "step_introduced_flipped_faces": int(
                    sum(int(row["step_introduced_flipped_faces"]) for row in selected)
                ),
                "mean_step_displacement": _mean(selected, "mean_step_displacement"),
                "mean_cumulative_displacement": _mean(
                    selected, "mean_cumulative_displacement"
                ),
                "mean_confidence": _mean(selected, "mean_confidence"),
                "visible_vertex_fraction": _mean(selected, "visible_vertex_fraction"),
                "mean_visible_views_per_vertex": _mean_optional(
                    selected, "mean_visible_views_per_vertex"
                ),
            }
            for field in PREDICTION_FIELDS:
                result[field] = _mean_optional(selected, field)
            output.append(result)
    return output


def _load_b_spec(run_dir: Path, device: torch.device) -> dict[str, Any]:
    config_payload = _read_json(run_dir / "run_config.json")
    config = config_payload.get("experiment_config", config_payload)
    if config.get("target_mode") != RAW_LAPLACIAN:
        raise ValueError("Recursive experiment requires the B direct-raw checkpoint.")
    checkpoint = run_dir / "checkpoint_latest.pt"
    model = _build_model(config, None, False).to(device)
    payload = load_checkpoint(checkpoint, model, map_location=device)
    if int(payload.get("optimizer_steps", -1)) != 20_000:
        raise ValueError("Expected the completed 20,000-step B checkpoint.")
    model.eval()
    amp_enabled, amp_dtype = _amp_settings(config, device)
    return {
        "config": config,
        "checkpoint": checkpoint,
        "checkpoint_sha256": _sha256(checkpoint),
        "model": model,
        "amp_enabled": amp_enabled,
        "amp_dtype": amp_dtype,
    }


def _infer_recursive(
    sample: Mapping[str, Any], spec: Mapping[str, Any], device: torch.device
) -> dict[str, Any]:
    prepared = _prepare_item_for_use(
        _prepare_object_static(sample, spec["config"]),
        spec["config"],
        device,
        cache_on_device=False,
        non_blocking=False,
        decode_images=True,
    )
    conditioned = _exact_query_sample(prepared.sample, device)
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=spec["amp_dtype"],
        enabled=bool(spec["amp_enabled"]),
    ):
        model_output = spec["model"](conditioned)
    if model_output.confidence_prediction is None:
        raise RuntimeError("B checkpoint must provide confidence prediction.")
    prediction_raw = model_output.predicted_laplacian.float().detach().cpu()
    confidence = model_output.confidence_prediction.float().detach().cpu()
    h = prepared.sample["local_edge_length"].float().detach().cpu()
    valid = prepared.sample["valid_scale_mask"].bool().detach().cpu()
    epsilon = float(spec["config"].get("target_scaling", {}).get("epsilon", 1e-12))
    prediction_normalized = normalize_laplacian_by_edge_scale(
        prediction_raw, h, eps=epsilon, valid_scale_mask=valid
    )
    visibility = prepared.sample["visibility"].detach().cpu()
    canonical = canonical_current_graph_recovery_inputs(
        prepared.sample["vertices"].detach().cpu(),
        sample["faces"],
        prediction_normalized,
        visibility,
        confidence,
        epsilon=epsilon,
    )
    roundtrip_error = float(
        torch.max(torch.abs(canonical.delta_pred_raw.cpu() - prediction_raw)).item()
    )
    if roundtrip_error > 1e-6:
        raise RuntimeError(f"Prediction raw/normalized round trip failed: {roundtrip_error}.")
    target_raw = prepared.raw_target.float().detach().cpu()
    return {
        "prediction_raw": prediction_raw,
        "prediction_normalized": prediction_normalized,
        "confidence": confidence,
        "visible": canonical.visible.detach().cpu(),
        "recovery_weight": canonical.weight.detach().cpu(),
        "epsilon": epsilon,
        "prediction_metrics": _raw_metrics(
            prediction_raw, target_raw, canonical.weight.detach().cpu(), valid
        ),
    }


def _dynamic_visibility(
    mesh: Mesh, source: Mapping[str, Any], *, visibility_size: int
) -> tuple[np.ndarray, dict[str, Any]]:
    source_size = int(source["prepared_image_size"])
    intrinsics = _np(source["intrinsics"]).copy()
    extrinsics = _np(source["extrinsics"])
    if visibility_size != source_size:
        scale = float(visibility_size) / float(source_size)
        intrinsics[:, 0, :] *= scale
        intrinsics[:, 1, :] *= scale
    cameras = [
        Camera(
            intrinsics=intrinsics[index],
            rotation=extrinsics[index, :3, :3],
            translation=extrinsics[index, :3, 3],
            image_size=(visibility_size, visibility_size),
            name=f"recursive_{index:02d}",
        )
        for index in range(len(intrinsics))
    ]
    result = compute_renderer_visibility(
        mesh,
        cameras,
        SyntheticRenderConfig(
            num_views=len(cameras),
            width=visibility_size,
            height=visibility_size,
            backend="opengl",
            normalize_mesh=False,
            antialiasing="none",
            backface_culling=False,
            front_face_winding="ccw",
        ),
        neighborhood_radius=1,
    )
    visibility = np.asarray(result.backface_and_occlusion_visible, dtype=bool)
    return visibility, _visibility_audit(visibility, visibility_size=visibility_size)


def _visibility_audit(
    visibility: np.ndarray, visibility_size: int | None
) -> dict[str, Any]:
    counts = np.asarray(visibility, dtype=bool).sum(axis=0)
    return {
        "backend": "opengl" if visibility_size is not None else "prepared_cuda_fixed",
        "raster_size": visibility_size,
        "mean_visible_views_per_vertex": float(counts.mean()),
        "zero_visible_vertex_fraction": float(np.mean(counts == 0)),
    }


def _recovery_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(config.get("recovery", {}))
    result.update(
        {
            "dense_vertex_limit": 5000,
            "chamfer_samples": 3000,
            "metric_seed": 7,
            "evaluate_oracle": False,
        }
    )
    return result


def _round_zero_row(
    policy: str,
    sample_id: str,
    metadata: Mapping[str, Any],
    baseline: Mapping[str, str],
    prediction: Mapping[str, str],
    baseline_displacement: np.ndarray,
) -> dict[str, Any]:
    return {
        "policy": policy,
        "round": 0,
        "sample_id": sample_id,
        "object_id": metadata.get("object_id"),
        "variant_index": metadata.get("variant_index"),
        "vertex_count": int(baseline["vertex_count"]),
        "original_initial_chamfer": float(baseline["initial_chamfer"]),
        "round_initial_chamfer": float(baseline["initial_chamfer"]),
        "reconstruction_chamfer": float(baseline["reconstruction_chamfer"]),
        "original_initial_point_to_surface": float(baseline["initial_point_to_surface"]),
        "round_initial_point_to_surface": float(baseline["initial_point_to_surface"]),
        "reconstruction_point_to_surface": float(
            baseline["reconstruction_point_to_surface"]
        ),
        "original_initial_normal_consistency": float(
            baseline["initial_normal_consistency"]
        ),
        "round_initial_normal_consistency": float(
            baseline["initial_normal_consistency"]
        ),
        "reconstruction_normal_consistency": float(
            baseline["reconstruction_normal_consistency"]
        ),
        "cumulative_improved_over_original": _bool(baseline["improved_over_initial"]),
        "step_improved_over_previous": _bool(baseline["improved_over_initial"]),
        "improved_over_round0": False,
        "previous_recorded_chamfer": float(baseline["initial_chamfer"]),
        "cumulative_introduced_flipped_faces": int(
            baseline["introduced_flipped_faces"]
        ),
        "step_introduced_flipped_faces": int(baseline["introduced_flipped_faces"]),
        "cumulative_new_degenerate_faces": int(baseline["new_degenerate_faces"]),
        "step_new_degenerate_faces": int(baseline["new_degenerate_faces"]),
        "mean_step_displacement": float(baseline_displacement.mean()),
        "max_step_displacement": float(baseline_displacement.max()),
        "mean_cumulative_displacement": float(baseline_displacement.mean()),
        "max_cumulative_displacement": float(baseline_displacement.max()),
        "mean_confidence": float(baseline["mean_confidence"]),
        "visible_vertex_fraction": float(baseline["visible_vertex_fraction"]),
        "visibility_backend": "prepared_cuda",
        "visibility_raster_size": 960,
        "mean_visible_views_per_vertex": None,
        "zero_visible_vertex_fraction": 1.0 - float(baseline["visible_vertex_fraction"]),
        **{field: float(prediction[field]) for field in PREDICTION_FIELDS},
    }


def _load_baseline_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["arm"] == ARM and int(row["replacement_percent"]) == 0
        ]
    if len(rows) != 25:
        raise RuntimeError(f"Expected 25 B/0% baseline rows in {path}.")
    return {row["sample_id"]: row for row in rows}


def _load_baseline_prediction_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["split"] == "test" and row["arm"] == ARM
        ]
    if len(rows) != 25:
        raise RuntimeError(f"Expected 25 B test prediction rows in {path}.")
    return {row["sample_id"]: row for row in rows}


def _recursive_decision(aggregate: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    recursive = [row for row in aggregate if int(row["round"]) > 0]
    best_chamfer = min(recursive, key=lambda row: float(row["reconstruction_chamfer"]))
    best_count = max(
        recursive,
        key=lambda row: (
            int(row["cumulative_improved_over_original"]),
            -float(row["reconstruction_chamfer"]),
        ),
    )
    return {
        "baseline_improved_over_original": 19,
        "any_round_exceeds_19_of_25": any(
            int(row["cumulative_improved_over_original"]) > 19 for row in recursive
        ),
        "maximum_improved_over_original": int(
            best_count["cumulative_improved_over_original"]
        ),
        "maximum_improved_policy": best_count["policy"],
        "maximum_improved_round": int(best_count["round"]),
        "lowest_mean_chamfer": float(best_chamfer["reconstruction_chamfer"]),
        "lowest_mean_chamfer_policy": best_chamfer["policy"],
        "lowest_mean_chamfer_round": int(best_chamfer["round"]),
        "primary_policy": "recomputed_opengl_960",
        "sensitivity_policy": "fixed_prepared_visibility",
    }


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# B Direct-Raw Recursive Current-Mesh Refinement",
        "",
        "## Contract",
        "",
        f"- Checkpoint SHA-256: `{summary['checkpoint_sha256']}`.",
        "- Round 0 is the completed B result (19/25 below the original coarse mesh).",
        "- Rounds 1, 2 and 3 each use the previous round's refined mesh as model input.",
        "- Vertices, normals, current Laplacian, local h, position normalization and projection are rebuilt every round.",
        "- Topology and vertex correspondence are fixed; GT differential values are not model inputs.",
        "- Primary policy recomputes OpenGL backface+occlusion visibility at 960; sensitivity policy reuses prepared CUDA visibility.",
        f"- Contract audit: `{summary['contract_audit']['passed']}`.",
        "",
        "## Geometry trajectory",
        "",
        "| Visibility policy | Round | Chamfer | P2S | Normal | Improved/original | Step improved | Retained 19 | Gained from 6 | Lost from 19 | Cumulative flips |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["aggregate"]:
        lines.append(
            f"| {row['policy']} | {row['round']} | {_f(row['reconstruction_chamfer'])} | "
            f"{_f(row['reconstruction_point_to_surface'])} | "
            f"{_f(row['reconstruction_normal_consistency'])} | "
            f"{row['cumulative_improved_over_original']}/25 | "
            f"{row['step_improved_over_previous']}/25 | "
            f"{row['retained_round0_successes']}/19 | "
            f"{row['gained_from_round0_failures']}/6 | "
            f"{row['lost_round0_successes']}/19 | "
            f"{row['cumulative_introduced_flipped_faces']} |"
        )
    lines.extend(
        [
            "",
            "## Recursive prediction",
            "",
            "| Visibility policy | Round | Raw EPE | Top 1% | Top 10% | Raw cosine | Weighted RMS | Visible fraction |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["aggregate"]:
        lines.append(
            f"| {row['policy']} | {row['round']} | {_f(row['raw_epe'])} | "
            f"{_f(row['raw_top_1_percent_epe'])} | "
            f"{_f(row['raw_top_10_percent_epe'])} | "
            f"{_f(row['raw_global_cosine'])} | "
            f"{_f(row['recovery_weighted_raw_residual_rms'])} | "
            f"{_f(row['visible_vertex_fraction'])} |"
        )
    decision = summary["decision"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Any recursive round exceeds 19/25: `{decision['any_round_exceeds_19_of_25']}`.",
            f"- Maximum improved count: `{decision['maximum_improved_over_original']}/25` at `{decision['maximum_improved_policy']}` round {decision['maximum_improved_round']}.",
            f"- Lowest mean Chamfer: `{_f(decision['lowest_mean_chamfer'])}` at `{decision['lowest_mean_chamfer_policy']}` round {decision['lowest_mean_chamfer_round']}.",
            "",
        ]
    )
    return "\n".join(lines)


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def _mean_optional(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = [row.get(field) for row in rows]
    selected = [float(value) for value in values if value is not None]
    return float(np.mean(selected)) if selected else None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _np(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _f(value: Any) -> str:
    return "—" if value is None else f"{float(value):.9g}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}.")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
