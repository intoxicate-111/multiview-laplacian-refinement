#!/usr/bin/env python3
from __future__ import annotations

"""Read-only projected-GT oracle diagnostic for Sofa50 OpenMVS48."""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from evaluate_sofa50_multitopology_rawlap import load_spec
from mlr.data import Mesh
from mlr.io import load_mesh, save_mesh
from mlr.learned_laplacian.evaluation import evaluate_mesh_geometry
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.multitopology_rawlap import (
    raw_uniform_laplacian,
    uniform_midpoint_subdivide,
)
from mlr.learned_laplacian.synthetic_current_h2_ablation import (
    _recover_raw_one,
)
from mlr.learned_laplacian.target_scaling import normalize_laplacian_by_edge_scale


OLD_ARM = "old_960_HF"
NEW_ARM = "new_multitopology_rawlap"
PRIMARY_ARMS = (
    "initial",
    "projected_gt_position_oracle",
    "projected_gt_laplacian_oracle",
    "predicted_laplacian_recovery",
)
ALL_ARMS = PRIMARY_ARMS + ("old_960_hf_prediction",)
METRIC_PROTOCOL = (
    "mlr.learned_laplacian.evaluation.evaluate_mesh_geometry;"
    "area_weighted_triangle_surface_sampling;"
    "bidirectional_sampled_surface_to_exact_triangle_surface;"
    "surface_samples=3000;seed=7;fscore_threshold=0.01;"
    "alignment=shared_prepared_coordinate_frame_no_ICP"
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _gt_mesh(static: Mapping[str, Any]) -> Mesh:
    return Mesh(
        _numpy(static["gt_vertices"]), _numpy(static["gt_faces"]).astype(np.int64)
    ).ensure_normals()


def _exact_project_to_gt(
    points: np.ndarray, gt: Mesh, *, chunk_size: int = 8192
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Exact point-to-triangle projection through trimesh's R-tree path."""

    import trimesh
    from scipy.spatial import cKDTree

    query = np.asarray(points, dtype=np.float64)
    surface = trimesh.Trimesh(vertices=gt.vertices, faces=gt.faces, process=False)
    projected: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    triangle_ids: list[np.ndarray] = []
    for start in range(0, len(query), chunk_size):
        closest, distance, triangle_id = trimesh.proximity.closest_point(
            surface, query[start : start + chunk_size]
        )
        projected.append(np.asarray(closest, dtype=np.float64))
        distances.append(np.asarray(distance, dtype=np.float64))
        triangle_ids.append(np.asarray(triangle_id, dtype=np.int64))
    result = np.concatenate(projected)
    distance = np.concatenate(distances)
    triangle_id = np.concatenate(triangle_ids)
    if not (
        np.isfinite(result).all()
        and np.isfinite(distance).all()
        and np.all(triangle_id >= 0)
        and np.all(triangle_id < gt.num_faces)
    ):
        raise RuntimeError("Exact GT triangle projection produced invalid values.")
    geometric_error = float(
        np.max(np.abs(np.linalg.norm(result - query, axis=1) - distance), initial=0.0)
    )
    # Keep this inexpensive audit single-threaded so independent sample shards
    # can run concurrently without nested thread oversubscription.
    nearest_vertex_distance, _ = cKDTree(gt.vertices).query(query, workers=1)
    upper_bound_violation = float(
        np.max(distance - nearest_vertex_distance, initial=0.0)
    )
    if geometric_error > 1e-8 or upper_bound_violation > 1e-8:
        raise RuntimeError(
            f"Projection audit failed: geometric={geometric_error}, "
            f"nearest-vertex upper-bound violation={upper_bound_violation}."
        )
    return result, distance, {
        "engine": "trimesh.proximity.closest_point_rtree_exact_triangle",
        "query_count": len(query),
        "chunk_size": chunk_size,
        "maximum_distance_identity_error": geometric_error,
        "maximum_nearest_vertex_upper_bound_violation": upper_bound_violation,
        "fraction_strictly_better_than_nearest_gt_vertex": float(
            np.mean(distance < nearest_vertex_distance - 1e-10)
        ),
    }


def _topology_change(initial: Mesh, result: Mesh) -> dict[str, Any]:
    if not np.array_equal(initial.faces, result.faces):
        raise RuntimeError("Oracle/recovery connectivity differs from OpenMVS input.")
    faces = initial.faces
    before = np.cross(
        initial.vertices[faces[:, 1]] - initial.vertices[faces[:, 0]],
        initial.vertices[faces[:, 2]] - initial.vertices[faces[:, 0]],
    )
    after = np.cross(
        result.vertices[faces[:, 1]] - result.vertices[faces[:, 0]],
        result.vertices[faces[:, 2]] - result.vertices[faces[:, 0]],
    )
    before_degenerate = np.linalg.norm(before, axis=1) <= 1e-14
    after_degenerate = np.linalg.norm(after, axis=1) <= 1e-14
    return {
        "introduced_flipped_faces": int(
            np.sum(np.einsum("ij,ij->i", before, after) < 0)
        ),
        "new_degenerate_faces": int(np.sum(after_degenerate & ~before_degenerate)),
        "faces_exactly_preserved": True,
    }


def _metric_row(sample_id: str, arm: str, mesh: Mesh, gt: Mesh, initial: Mesh) -> dict[str, Any]:
    metric = evaluate_mesh_geometry(
        mesh, gt, surface_samples=3000, seed=7, fscore_threshold=0.01
    )
    topology = (
        {"introduced_flipped_faces": 0, "new_degenerate_faces": 0, "faces_exactly_preserved": True}
        if arm == "initial"
        else _topology_change(initial, mesh)
    )
    return {
        "sample_id": sample_id,
        "arm": arm,
        "chamfer": metric["chamfer"],
        "p2s": metric["point_to_surface_bidirectional_mean"],
        "p2s_p95": metric["point_to_surface_bidirectional_p95"],
        "normal_consistency": metric["normal_consistency"],
        "fscore": metric["fscore"],
        "forward_engine": metric["forward_engine"],
        "reverse_engine": metric["reverse_engine"],
        "metric_protocol": METRIC_PROTOCOL,
        **topology,
    }


def evaluate_shard(args: argparse.Namespace) -> None:
    manifest = args.manifest.resolve()
    recovery_source = args.recovery_source.resolve()
    unified_source = args.unified_source.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = PreparedMeshDataset.from_manifest(manifest, "test")
    device = torch.device(args.device)
    spec = load_spec(args.new_run.resolve(), device)
    config = spec["config"]
    epsilon = float(config.get("target_scaling", {}).get("epsilon", 1e-12))
    source_rows = _read_csv(unified_source / "per_sample.csv")
    source_by_key = {(row["sample_id"], row["arm"]): row for row in source_rows}
    recovery_rows = _read_csv(recovery_source / "per_sample.csv")
    recovery_by_key = {(row["sample_id"], row["arm"]): row for row in recovery_rows}
    checkpoint = Path(str(spec["checkpoint"]))
    metric_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        if index % args.shard_count != args.shard_index:
            continue
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        old_dir = recovery_source / "reconstruction" / OLD_ARM / sample_id
        new_dir = recovery_source / "reconstruction" / NEW_ARM / sample_id
        initial = load_mesh(new_dir / "coarse.obj").ensure_normals()
        initial_old = load_mesh(old_dir / "coarse.obj").ensure_normals()
        learned = load_mesh(new_dir / "predicted_refined.obj").ensure_normals()
        old = load_mesh(old_dir / "predicted_refined.obj").ensure_normals()
        gt = _gt_mesh(static)
        same_initial = bool(
            np.array_equal(initial.vertices, initial_old.vertices)
            and np.array_equal(initial.faces, initial_old.faces)
            and np.allclose(initial.vertices, _numpy(static["vertices"]), rtol=0.0, atol=1e-8)
            and np.array_equal(initial.faces, _numpy(static["faces"]))
        )
        if not same_initial:
            raise RuntimeError(f"Initial OpenMVS identity failed for {sample_id}.")

        projected_vertices, projection_distance, projection_audit = _exact_project_to_gt(
            initial.vertices, gt, chunk_size=args.projection_chunk_size
        )
        position_oracle = Mesh(projected_vertices, initial.faces.copy()).ensure_normals()
        delta_oracle = raw_uniform_laplacian(position_oracle)
        h = torch.as_tensor(static["local_edge_length"]).float().cpu()
        valid = torch.as_tensor(static["valid_scale_mask"]).bool().cpu()
        delta_oracle_t = torch.as_tensor(delta_oracle, dtype=torch.float32)
        delta_oracle_normalized = normalize_laplacian_by_edge_scale(
            delta_oracle_t, h, eps=epsilon, valid_scale_mask=valid
        )

        saved_prediction = np.load(new_dir / "delta_pred_raw.npy")
        archived_row = recovery_by_key[(sample_id, NEW_ARM)]
        archived_mean_confidence = float(archived_row["mean_confidence"])
        if archived_mean_confidence != 1.0:
            raise RuntimeError(
                f"Archived confidence is not exactly one for {sample_id}: "
                f"{archived_mean_confidence}"
            )
        # The original Blackwell inference reports confidence=1.0 exactly for every
        # OpenMVS sample.  Reuse the archived prediction and that frozen confidence
        # contract rather than recomputing FP16 inference on an L40, which is not
        # bitwise reproducible across GPU architectures.
        confidence = torch.ones(len(initial.vertices), dtype=torch.float32)
        saved_prediction_t = torch.as_tensor(saved_prediction, dtype=torch.float32)
        saved_prediction_normalized = normalize_laplacian_by_edge_scale(
            saved_prediction_t, h, eps=epsilon, valid_scale_mask=valid
        )
        archived_replay_dir = output / "samples" / sample_id / "archived_recovery_replay"
        _, _ = _recover_raw_one(
            static,
            saved_prediction_t,
            saved_prediction_normalized,
            confidence,
            archived_replay_dir,
            config,
        )
        archived_replay = load_mesh(
            archived_replay_dir / "predicted_refined.obj"
        ).ensure_normals()
        archived_replay_error = float(
            np.max(np.abs(archived_replay.vertices - learned.vertices), initial=0.0)
        )
        shutil.rmtree(archived_replay_dir)
        recovery_dir = output / "samples" / sample_id / "oracle_recovery"
        oracle_recovery, _ = _recover_raw_one(
            static,
            delta_oracle_t,
            delta_oracle_normalized,
            confidence,
            recovery_dir,
            config,
        )
        lap_oracle = load_mesh(recovery_dir / "predicted_refined.obj").ensure_normals()

        sample_dir = output / "samples" / sample_id
        save_mesh(gt, sample_dir / "gt.obj")
        shutil.copy2(new_dir / "coarse.obj", sample_dir / "initial.obj")
        save_mesh(position_oracle, sample_dir / "projected_gt_position_oracle.obj")
        shutil.copy2(
            recovery_dir / "predicted_refined.obj",
            sample_dir / "projected_gt_laplacian_oracle.obj",
        )
        shutil.copy2(
            new_dir / "predicted_refined.obj",
            sample_dir / "predicted_laplacian_recovery.obj",
        )
        shutil.copy2(
            old_dir / "predicted_refined.obj",
            sample_dir / "old_960_hf_prediction.obj",
        )
        shutil.rmtree(recovery_dir)
        np.save(sample_dir / "projected_gt_vertex_displacement.npy", projected_vertices - initial.vertices)
        np.save(sample_dir / "projected_gt_oracle_raw_laplacian.npy", delta_oracle)

        meshes = {
            "initial": initial,
            "projected_gt_position_oracle": position_oracle,
            "projected_gt_laplacian_oracle": lap_oracle,
            "predicted_laplacian_recovery": learned,
            "old_960_hf_prediction": old,
        }
        sample_metrics = {
            arm: _metric_row(sample_id, arm, mesh, gt, initial)
            for arm, mesh in meshes.items()
        }
        metric_rows.extend(sample_metrics.values())

        expanded_vertices, expanded_faces = uniform_midpoint_subdivide(
            initial.vertices, initial.faces, levels=1
        )
        expanded_initial = Mesh(expanded_vertices, expanded_faces).ensure_normals()
        expanded_projected, _, expanded_projection_audit = _exact_project_to_gt(
            expanded_vertices, gt, chunk_size=args.projection_chunk_size
        )
        expanded_position_oracle = Mesh(expanded_projected, expanded_faces).ensure_normals()
        expanded_initial_metric = evaluate_mesh_geometry(
            expanded_initial, gt, surface_samples=3000, seed=7, fscore_threshold=0.01
        )
        expanded_oracle_metric = evaluate_mesh_geometry(
            expanded_position_oracle, gt, surface_samples=3000, seed=7, fscore_threshold=0.01
        )
        np.savez_compressed(
            sample_dir / "expanded_position_oracle.npz",
            initial_vertices=expanded_vertices,
            projected_vertices=expanded_projected,
            faces=expanded_faces,
        )

        initial_chamfer = float(sample_metrics["initial"]["chamfer"])
        pos_chamfer = float(sample_metrics["projected_gt_position_oracle"]["chamfer"])
        lap_chamfer = float(sample_metrics["projected_gt_laplacian_oracle"]["chamfer"])
        pred_chamfer = float(sample_metrics["predicted_laplacian_recovery"]["chamfer"])
        representation_gain = initial_chamfer - pos_chamfer
        laplacian_recovery_gain = initial_chamfer - lap_chamfer
        expanded_gain = initial_chamfer - float(expanded_oracle_metric["chamfer"])
        gap_rows.append(
            {
                "sample_id": sample_id,
                "initial_chamfer": initial_chamfer,
                "projected_gt_position_oracle_chamfer": pos_chamfer,
                "projected_gt_laplacian_oracle_chamfer": lap_chamfer,
                "predicted_laplacian_recovery_chamfer": pred_chamfer,
                "old_960_hf_prediction_chamfer": float(sample_metrics["old_960_hf_prediction"]["chamfer"]),
                "representation_gap": representation_gain,
                "representation_relative_improvement": representation_gain / max(initial_chamfer, 1e-12),
                "recovery_gap_lap_minus_position": lap_chamfer - pos_chamfer,
                "recovery_retained_fraction": laplacian_recovery_gain / max(representation_gain, 1e-12),
                "prediction_gap_pred_minus_lap": pred_chamfer - lap_chamfer,
                "expanded_initial_chamfer": float(expanded_initial_metric["chamfer"]),
                "expanded_projected_gt_position_oracle_chamfer": float(expanded_oracle_metric["chamfer"]),
                "expanded_representation_gap": expanded_gain,
                "expanded_vs_coarse_oracle_chamfer": float(expanded_oracle_metric["chamfer"]) - pos_chamfer,
                "expanded_faces": len(expanded_faces),
                "expanded_vertices": len(expanded_vertices),
                "position_to_laplacian_oracle_vertex_rmse": float(
                    np.sqrt(np.mean(np.sum((lap_oracle.vertices - projected_vertices) ** 2, axis=1)))
                ),
            }
        )
        source_initial = source_by_key[(sample_id, NEW_ARM)]
        source_initial_metric_error = abs(
            initial_chamfer - float(source_initial["initial_chamfer"])
        )
        source_learned_metric_error = abs(
            pred_chamfer - float(source_initial["reconstruction_chamfer"])
        )
        audit = {
            "sample_id": sample_id,
            "same_openmvs_input_mesh": same_initial,
            "same_gt_surface": True,
            "projection_engine": projection_audit["engine"],
            "projection_exact_triangle": True,
            "projection_nearest_vertex_shortcut": False,
            "projection_audit": projection_audit,
            "expanded_projection_audit": expanded_projection_audit,
            "no_icp": True,
            "no_test_time_alignment": True,
            "position_oracle_faces_exact": np.array_equal(position_oracle.faces, initial.faces),
            "oracle_laplacian_graph": "OpenMVS F0 uniform graph",
            "oracle_laplacian_recompute_max_abs_error": float(
                np.max(np.abs(raw_uniform_laplacian(position_oracle) - delta_oracle), initial=0.0)
            ),
            "oracle_recovery_faces_exact": np.array_equal(lap_oracle.faces, initial.faces),
            "learned_recovery_faces_exact": np.array_equal(learned.faces, initial.faces),
            "same_recovery_config": True,
            "recovery_config": config.get("recovery"),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_optimizer_steps": spec["optimizer_steps"],
            "prediction_source": "archived Blackwell delta_pred_raw.npy",
            "cross_gpu_fp16_inference_rerun_used": False,
            "archived_mean_confidence_record": archived_mean_confidence,
            "archived_recovery_replay_max_abs_error": archived_replay_error,
            "confidence_min": float(confidence.min()),
            "confidence_mean": float(confidence.mean()),
            "confidence_max": float(confidence.max()),
            "source_initial_unified_metric_abs_error": source_initial_metric_error,
            "source_learned_unified_metric_abs_error": source_learned_metric_error,
            "hidden_post_recovery_smoothing_or_remeshing": False,
            "oracle_recovery_native_metrics": oracle_recovery,
        }
        audit["passed"] = bool(
            same_initial
            and audit["position_oracle_faces_exact"]
            and audit["oracle_recovery_faces_exact"]
            and audit["learned_recovery_faces_exact"]
            and audit["oracle_laplacian_recompute_max_abs_error"] == 0.0
            and archived_replay_error <= 5e-6
            and source_initial_metric_error <= 1e-12
            and source_learned_metric_error <= 1e-12
            and projection_audit["maximum_nearest_vertex_upper_bound_violation"] <= 1e-8
            and expanded_projection_audit["maximum_nearest_vertex_upper_bound_violation"] <= 1e-8
        )
        audit_rows.append(audit)
        _write_json(sample_dir / "metrics.json", sample_metrics)
        _write_json(sample_dir / "contract_audit.json", audit)
        print(
            f"{sample_id}: initial={initial_chamfer:.8g} pos={pos_chamfer:.8g} "
            f"lap={lap_chamfer:.8g} pred={pred_chamfer:.8g} audit={audit['passed']}",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    shard = output / "shards"
    _write_json(
        shard / f"shard_{args.shard_index:02d}.json",
        {
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "manifest": str(manifest),
            "manifest_sha256": _sha256(manifest),
            "metric_protocol": METRIC_PROTOCOL,
            "new_checkpoint": str(checkpoint),
            "new_checkpoint_sha256": _sha256(checkpoint),
            "metric_rows": metric_rows,
            "gap_rows": gap_rows,
            "audit_rows": audit_rows,
        },
    )


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def merge(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    payloads = [
        _read_json(output / "shards" / f"shard_{index:02d}.json")
        for index in range(args.shard_count)
    ]
    metric_rows = [row for payload in payloads for row in payload["metric_rows"]]
    gap_rows = [row for payload in payloads for row in payload["gap_rows"]]
    audit_rows = [row for payload in payloads for row in payload["audit_rows"]]
    expected = len(PreparedMeshDataset.from_manifest(args.manifest.resolve(), "test"))
    contract_audit = {
        "passed": len(gap_rows) == expected and all(bool(row["passed"]) for row in audit_rows),
        "samples": len(gap_rows),
        "expected_samples": expected,
        "metric_protocol": METRIC_PROTOCOL,
        "same_openmvs_input_all": all(bool(row["same_openmvs_input_mesh"]) for row in audit_rows),
        "same_gt_surface_all": all(bool(row["same_gt_surface"]) for row in audit_rows),
        "exact_closest_triangle_projection_all": all(bool(row["projection_exact_triangle"]) for row in audit_rows),
        "nearest_vertex_shortcut_used": any(bool(row["projection_nearest_vertex_shortcut"]) for row in audit_rows),
        "no_icp": all(bool(row["no_icp"]) for row in audit_rows),
        "no_test_time_alignment": all(bool(row["no_test_time_alignment"]) for row in audit_rows),
        "openmvs_connectivity_retained_all": all(
            bool(row["position_oracle_faces_exact"])
            and bool(row["oracle_recovery_faces_exact"])
            and bool(row["learned_recovery_faces_exact"])
            for row in audit_rows
        ),
        "oracle_laplacian_on_openmvs_graph": True,
        "same_recovery_settings": all(bool(row["same_recovery_config"]) for row in audit_rows),
        "checkpoint_changes": False,
        "hidden_smoothing_or_remeshing": False,
        "checkpoint": payloads[0]["new_checkpoint"],
        "checkpoint_sha256": payloads[0]["new_checkpoint_sha256"],
        "prediction_source": "archived Blackwell delta_pred_raw.npy",
        "cross_gpu_fp16_inference_rerun_used": False,
        "max_archived_recovery_replay_abs_error": max(
            float(row["archived_recovery_replay_max_abs_error"])
            for row in audit_rows
        ),
        "max_source_unified_metric_abs_error": max(
            max(float(row["source_initial_unified_metric_abs_error"]), float(row["source_learned_unified_metric_abs_error"]))
            for row in audit_rows
        ),
    }
    _write_json(output / "contract_audit.json", contract_audit)
    _write_csv(output / "per_sample_metrics.csv", metric_rows)
    _write_csv(output / "per_sample_oracle_gaps.csv", gap_rows)
    _write_json(output / "per_sample_contract_audit.json", audit_rows)
    if not contract_audit["passed"]:
        _write_json(output / "summary.json", {"contract_audit": contract_audit})
        raise RuntimeError("Contract audit failed; interpretation stopped.")

    aggregate: list[dict[str, Any]] = []
    initial_by_id = {
        row["sample_id"]: row for row in metric_rows if row["arm"] == "initial"
    }
    for arm in ALL_ARMS:
        selected = [row for row in metric_rows if row["arm"] == arm]
        aggregate.append(
            {
                "arm": arm,
                "samples": len(selected),
                "chamfer": _mean(selected, "chamfer"),
                "p2s": _mean(selected, "p2s"),
                "p2s_p95": _mean(selected, "p2s_p95"),
                "normal_consistency": _mean(selected, "normal_consistency"),
                "fscore": _mean(selected, "fscore"),
                "introduced_flipped_faces": int(sum(int(row["introduced_flipped_faces"]) for row in selected)),
                "new_degenerate_faces": int(sum(int(row["new_degenerate_faces"]) for row in selected)),
                "improved_over_initial": int(sum(float(row["chamfer"]) < float(initial_by_id[row["sample_id"]]["chamfer"]) for row in selected)),
                "worsened_over_initial": int(sum(float(row["chamfer"]) > float(initial_by_id[row["sample_id"]]["chamfer"]) for row in selected)),
            }
        )
    agg = {row["arm"]: row for row in aggregate}
    initial = agg["initial"]["chamfer"]
    for row in aggregate:
        improvement = initial - float(row["chamfer"])
        row["absolute_improvement_vs_initial"] = improvement
        row["relative_improvement_vs_initial"] = improvement / max(initial, 1e-12)
    position = agg["projected_gt_position_oracle"]["chamfer"]
    laplacian = agg["projected_gt_laplacian_oracle"]["chamfer"]
    predicted = agg["predicted_laplacian_recovery"]["chamfer"]
    representation_gain = initial - position
    recovery_gain = initial - laplacian
    expanded_position = _mean(gap_rows, "expanded_projected_gt_position_oracle_chamfer")
    expanded_gain = initial - expanded_position
    analysis = {
        "representation_absolute_improvement": representation_gain,
        "representation_relative_improvement": representation_gain / max(initial, 1e-12),
        "recovery_gap_lap_minus_position": laplacian - position,
        "recovery_retained_fraction_of_position_oracle_gain": recovery_gain / max(representation_gain, 1e-12),
        "prediction_gap_pred_minus_lap": predicted - laplacian,
        "prediction_gap_fraction_of_position_oracle_gain": (predicted - laplacian) / max(representation_gain, 1e-12),
        "prediction_retained_fraction_of_laplacian_oracle_gain": (initial - predicted) / max(recovery_gain, 1e-12),
        "expanded_projected_gt_position_oracle_chamfer": expanded_position,
        "expanded_representation_absolute_improvement": expanded_gain,
        "expanded_representation_relative_improvement": expanded_gain / max(initial, 1e-12),
        "expanded_vs_coarse_position_oracle_chamfer": expanded_position - position,
        "expanded_gain_ratio_vs_coarse": expanded_gain / max(representation_gain, 1e-12),
    }
    visual_selection = _select_visual_cases(gap_rows)
    summary = {
        "contract_audit": contract_audit,
        "aggregate": aggregate,
        "oracle_gap_analysis": analysis,
        "representative_cases": visual_selection,
        "terminology": {
            "position": "projected-GT position oracle",
            "laplacian": "projected-GT oracle Laplacian on the OpenMVS graph",
        },
    }
    _write_json(output / "summary.json", summary)
    _write_csv(output / "aggregate.csv", aggregate)
    _write_json(output / "visual_selection_request.json", visual_selection)
    _write_report(output / "REPORT.md", summary)
    print(json.dumps(summary, indent=2))


def _select_visual_cases(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    learned_delta = lambda row: float(row["initial_chamfer"]) - float(row["predicted_laplacian_recovery_chamfer"])
    best = sorted(rows, key=learned_delta, reverse=True)[:3]
    worst = sorted(rows, key=learned_delta)[:3]
    used = {str(row["sample_id"]) for row in best + worst}
    difficult = [
        row for row in sorted(rows, key=lambda row: float(row["initial_chamfer"]), reverse=True)
        if str(row["sample_id"]) not in used
    ][:3]
    return {
        "selection_rule": "3 best learned Chamfer changes + 3 worst + 3 highest-initial visually difficult non-overlapping cases",
        "best": [row["sample_id"] for row in best],
        "worst": [row["sample_id"] for row in worst],
        "difficult": [row["sample_id"] for row in difficult],
        "sample_ids": [row["sample_id"] for row in best + worst + difficult],
    }


def _write_report(path: Path, summary: Mapping[str, Any]) -> None:
    aggregate = summary["aggregate"]
    analysis = summary["oracle_gap_analysis"]
    lines = [
        "# Sofa50 OpenMVS48 projected-GT oracle diagnostic",
        "",
        "Contract audit: **true**.",
        "",
        "No model was trained. The exact existing checkpoint, OpenMVS meshes, shared prepared coordinates, visibility/confidence weighting and recovery solver were retained. No ICP or test-time alignment was used.",
        "",
        "**Policy:** OpenMVS is a low-quality OOD stress input, not a target, pseudo-GT, model-selection endpoint or quality ceiling. This diagnostic has zero decision weight for future training and scale-up.",
        "",
        f"Metric protocol: `{METRIC_PROTOCOL}`",
        "",
        "| Arm | Chamfer | Abs. gain | Rel. gain | P2S | P2S p95 | Normal | F-score | Flips | New degenerates | Improved | Worsened |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['arm']} | {row['chamfer']:.9g} | {row['absolute_improvement_vs_initial']:.9g} | "
            f"{100*row['relative_improvement_vs_initial']:.2f}% | {row['p2s']:.9g} | {row['p2s_p95']:.9g} | "
            f"{row['normal_consistency']:.9g} | {row['fscore']:.9g} | {row['introduced_flipped_faces']} | "
            f"{row['new_degenerate_faces']} | {row['improved_over_initial']}/{row['samples']} | "
            f"{row['worsened_over_initial']}/{row['samples']} |"
        )
    lines.extend(
        [
            "",
            "## Oracle gaps",
            "",
            f"- Current-topology representation improvement: `{analysis['representation_absolute_improvement']:.9g}` ({100*analysis['representation_relative_improvement']:.2f}%).",
            f"- Recovery-retained fraction of that attainable gain: `{100*analysis['recovery_retained_fraction_of_position_oracle_gain']:.2f}%`.",
            f"- Learned prediction minus Laplacian-oracle Chamfer gap: `{analysis['prediction_gap_pred_minus_lap']:.9g}`.",
            f"- One-level expanded position-oracle Chamfer: `{analysis['expanded_projected_gt_position_oracle_chamfer']:.9g}`; gain ratio versus coarse topology: `{analysis['expanded_gain_ratio_vs_coarse']:.3f}`.",
            "",
            "Per-sample values are in `per_sample_oracle_gaps.csv`; visual selection is in `visual_selection_request.json`.",
            "",
            "## Representative cases selected before rendering",
            "",
            f"- Learned improvements: `{', '.join(summary['representative_cases']['best'])}`",
            f"- Learned worsenings: `{', '.join(summary['representative_cases']['worst'])}`",
            f"- High-initial-error difficult cases: `{', '.join(summary['representative_cases']['difficult'])}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--new-run", type=Path)
    parser.add_argument("--recovery-source", required=True, type=Path)
    parser.add_argument("--unified-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--projection-chunk-size", type=int, default=8192)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    if args.merge_only:
        merge(args)
    else:
        if args.new_run is None:
            parser.error("--new-run is required unless --merge-only is used")
        evaluate_shard(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
