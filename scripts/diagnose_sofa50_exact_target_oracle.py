#!/usr/bin/env python3
from __future__ import annotations

"""Exact-native-target Laplacian recovery diagnostic for one Sofa50 dataset arm."""

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from evaluate_sofa50_multitopology_rawlap import load_spec
from mlr.data import Mesh
from mlr.io import load_mesh, save_mesh
from mlr.learned_laplacian.evaluation import evaluate_mesh_geometry
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.multitopology_rawlap import raw_uniform_laplacian
from mlr.learned_laplacian.synthetic_current_h2_ablation import _infer_one, _recover_raw_one
from mlr.learned_laplacian.target_scaling import normalize_laplacian_by_edge_scale


STATES = ("initial", "clean", "exact_target_oracle", "predicted_recovery")
METRIC_PROTOCOL = (
    "mlr.learned_laplacian.evaluation.evaluate_mesh_geometry;"
    "area_weighted_triangle_surface_sampling;"
    "bidirectional_sampled_surface_to_exact_triangle_surface;"
    "surface_samples=3000;seed=7;fscore_threshold=0.01;"
    "alignment=shared_prepared_coordinate_frame_no_ICP"
)
SPECTRAL_PROTOCOL = (
    "symmetric_normalized_graph_laplacian_exact_smallest_eigenvectors;"
    "k=32;low=first_8_eigenvectors;mid=eigenvectors_8_to_31;"
    "high=orthogonal_residual;one_lexicographically_first_test_object_per_topology_variant"
)


def _numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _clean_mesh(static: Mapping[str, Any]) -> Mesh:
    vertices = static.get("clean_reference_vertices", static.get("gt_vertices"))
    faces = static.get("clean_reference_faces", static.get("gt_faces"))
    if vertices is None or faces is None:
        raise KeyError("Missing clean-reference mesh.")
    return Mesh(_numpy(vertices), _numpy(faces).astype(np.int64)).ensure_normals()


def _crosses(mesh: Mesh) -> np.ndarray:
    tri = mesh.vertices[mesh.faces]
    return np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])


def _topology(initial: Mesh, result: Mesh) -> tuple[np.ndarray, np.ndarray]:
    if not np.array_equal(initial.faces, result.faces):
        raise RuntimeError("Recovery changed face connectivity.")
    before = _crosses(initial)
    after = _crosses(result)
    flips = np.einsum("ij,ij->i", before, after) < 0
    before_deg = np.linalg.norm(before, axis=1) <= 1e-14
    after_deg = np.linalg.norm(after, axis=1) <= 1e-14
    return flips, after_deg & ~before_deg


def _geometry_row(
    dataset_arm: str,
    sample_id: str,
    state: str,
    mesh: Mesh,
    clean: Mesh,
    initial: Mesh,
) -> dict[str, Any]:
    metric = evaluate_mesh_geometry(
        mesh.ensure_normals(), clean, surface_samples=3000, seed=7, fscore_threshold=0.01
    )
    if state == "initial":
        flips = np.zeros(initial.num_faces, dtype=bool)
        degenerates = np.zeros(initial.num_faces, dtype=bool)
    else:
        flips, degenerates = _topology(initial, mesh)
    return {
        "dataset_arm": dataset_arm,
        "sample_id": sample_id,
        "state": state,
        "chamfer": float(metric["chamfer"]),
        "p2s": float(metric["point_to_surface_bidirectional_mean"]),
        "p2s_p95": float(metric["point_to_surface_bidirectional_p95"]),
        "fscore": float(metric["fscore"]),
        "normal_consistency": float(metric["normal_consistency"]),
        "introduced_flipped_faces": int(flips.sum()),
        "new_degenerate_faces": int(degenerates.sum()),
        "metric_protocol": METRIC_PROTOCOL,
        "forward_engine": metric["forward_engine"],
        "reverse_engine": metric["reverse_engine"],
    }


def _prediction_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    visible: torch.Tensor,
) -> dict[str, float]:
    pred = prediction[valid].double()
    gt = target[valid].double()
    vis = visible[valid].bool()
    vector_error = pred - gt
    error = torch.linalg.vector_norm(vector_error, dim=-1)
    gt_magnitude = torch.linalg.vector_norm(gt, dim=-1)
    order = torch.argsort(gt_magnitude, stable=True)
    count = len(order)
    top10_count = max(1, int(math.ceil(0.10 * count)))
    top1_count = max(1, int(math.ceil(0.01 * count)))
    bottom = order[: count - top10_count]
    top10 = order[count - top10_count :]
    top1 = order[count - top1_count :]
    mean_vector = vector_error.mean(dim=0)

    def group_mean(mask: torch.Tensor) -> float:
        return float(error[mask].mean()) if bool(mask.any()) else float("nan")

    cosine = F.cosine_similarity(pred.reshape(1, -1), gt.reshape(1, -1), dim=-1, eps=1e-12)
    return {
        "raw_epe": float(error.mean()),
        "raw_rms": float(torch.sqrt(error.square().mean())),
        "raw_max": float(error.max()),
        "raw_cosine": float(cosine.item()),
        "mean_vector_bias_x": float(mean_vector[0]),
        "mean_vector_bias_y": float(mean_vector[1]),
        "mean_vector_bias_z": float(mean_vector[2]),
        "mean_vector_bias_norm": float(torch.linalg.vector_norm(mean_vector)),
        "mean_magnitude_error": float(torch.abs(torch.linalg.vector_norm(pred, dim=-1) - gt_magnitude).mean()),
        "bottom90_epe": float(error[bottom].mean()),
        "top10_epe": float(error[top10].mean()),
        "top1_epe": float(error[top1].mean()),
        "visible_epe": group_mean(vis),
        "invisible_epe": group_mean(~vis),
        "visible_fraction": float(vis.float().mean()),
    }


def _spectral_metrics(error: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    from scipy.sparse import coo_matrix, csgraph
    from scipy.sparse.linalg import eigsh

    n = len(error)
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    edges = np.concatenate((edges, edges[:, ::-1]), axis=0)
    adjacency = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
    adjacency.data[:] = 1.0
    laplacian = csgraph.laplacian(adjacency, normed=True)
    k = min(32, n - 2)
    if k < 3:
        raise RuntimeError("Mesh is too small for spectral diagnostic.")
    eigenvalues, eigenvectors = eigsh(laplacian, k=k, which="SM", tol=1e-4, maxiter=5000)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    low_k = min(8, k)
    low_coeff = eigenvectors[:, :low_k].T @ error
    mid_coeff = eigenvectors[:, low_k:].T @ error
    low_energy = float(np.square(low_coeff).sum())
    mid_energy = float(np.square(mid_coeff).sum())
    projected = eigenvectors @ (eigenvectors.T @ error)
    high_energy = float(np.square(error - projected).sum())
    total = float(np.square(error).sum())
    return {
        "spectral_protocol": SPECTRAL_PROTOCOL,
        "spectral_vertex_count": n,
        "spectral_eigenvector_count": k,
        "spectral_low_band_max_eigenvalue": float(eigenvalues[low_k - 1]),
        "spectral_mid_band_max_eigenvalue": float(eigenvalues[-1]),
        "spectral_low_error_energy": low_energy,
        "spectral_mid_error_energy": mid_energy,
        "spectral_high_error_energy": high_energy,
        "spectral_total_error_energy": total,
        "spectral_low_error_fraction": low_energy / max(total, 1e-30),
        "spectral_mid_error_fraction": mid_energy / max(total, 1e-30),
        "spectral_high_error_fraction": high_energy / max(total, 1e-30),
    }


def _spectral_selection(dataset: PreparedMeshDataset) -> set[str]:
    selected: dict[str, str] = {}
    for index, sample_id in enumerate(dataset.sample_ids):
        static = dataset.load_static(index)
        variant = str(dict(static.get("metadata", {})).get("variant", "unknown"))
        if variant not in selected or sample_id < selected[variant]:
            selected[variant] = sample_id
    return set(selected.values())


def _flip_attribution(
    dataset_arm: str,
    sample_id: str,
    initial: Mesh,
    clean: Mesh,
    oracle: Mesh,
    predicted: Mesh,
    target: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    initial_cross = _crosses(initial)
    clean_cross = _crosses(clean)
    initial_wrong = np.einsum("ij,ij->i", initial_cross, clean_cross) < 0
    oracle_flip, _ = _topology(initial, oracle)
    pred_flip, _ = _topology(initial, predicted)
    overlap = oracle_flip & pred_flip
    area = 0.5 * np.linalg.norm(initial_cross, axis=1)
    diagonal = float(np.linalg.norm(initial.vertices.max(axis=0) - initial.vertices.min(axis=0)))
    normalized_area = area / max(diagonal * diagonal, 1e-30)
    target_vertex_magnitude = np.linalg.norm(target, axis=1)
    target_face_magnitude = target_vertex_magnitude[initial.faces].mean(axis=1)
    threshold = float(np.quantile(target_face_magnitude, 0.9))
    high_target = target_face_magnitude >= threshold
    oracle_displacement = np.linalg.norm(oracle.vertices - initial.vertices, axis=1)
    pred_displacement = np.linalg.norm(predicted.vertices - initial.vertices, axis=1)
    oracle_face_displacement = oracle_displacement[initial.faces].max(axis=1)
    pred_face_displacement = pred_displacement[initial.faces].max(axis=1)

    def mean_at(values: np.ndarray, mask: np.ndarray) -> float:
        return float(values[mask].mean()) if mask.any() else float("nan")

    row = {
        "dataset_arm": dataset_arm,
        "sample_id": sample_id,
        "faces": initial.num_faces,
        "initial_wrong_orientation_vs_clean": int(initial_wrong.sum()),
        "oracle_introduced_flips": int(oracle_flip.sum()),
        "predicted_introduced_flips": int(pred_flip.sum()),
        "oracle_predicted_flip_overlap": int(overlap.sum()),
        "oracle_only_flips": int((oracle_flip & ~pred_flip).sum()),
        "prediction_only_flips": int((pred_flip & ~oracle_flip).sum()),
        "high_target_faces": int(high_target.sum()),
        "oracle_flips_high_target": int((oracle_flip & high_target).sum()),
        "predicted_flips_high_target": int((pred_flip & high_target).sum()),
        "oracle_flip_mean_face_displacement": mean_at(oracle_face_displacement, oracle_flip),
        "oracle_nonflip_mean_face_displacement": mean_at(oracle_face_displacement, ~oracle_flip),
        "predicted_flip_mean_face_displacement": mean_at(pred_face_displacement, pred_flip),
        "predicted_nonflip_mean_face_displacement": mean_at(pred_face_displacement, ~pred_flip),
    }
    arrays = {
        "oracle_flip_area": normalized_area[oracle_flip],
        "predicted_flip_area": normalized_area[pred_flip],
    }
    return row, arrays


def evaluate_shard(args: argparse.Namespace) -> None:
    manifest = args.manifest.resolve()
    run_dir = args.run_dir.resolve()
    source = args.prediction_source_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = PreparedMeshDataset.from_manifest(manifest, "test")
    selected_spectral = _spectral_selection(dataset)
    device = torch.device(args.device)
    spec = load_spec(run_dir, device)
    config = spec["config"]
    checkpoint = Path(str(spec["checkpoint"]))
    epsilon = float(config.get("target_scaling", {}).get("epsilon", 1e-12))
    geometry_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    flip_rows: list[dict[str, Any]] = []
    spectral_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    area_arrays: dict[str, list[np.ndarray]] = defaultdict(list)

    for index in range(len(dataset)):
        if index % args.shard_count != args.shard_index:
            continue
        static = dataset.load_static(index)
        record = dataset.records[index]
        sample_id = str(static["sample_id"])
        metadata = dict(static.get("metadata", {}))
        initial = Mesh(_numpy(static["vertices"]), _numpy(static["faces"]).astype(np.int64)).ensure_normals()
        clean = _clean_mesh(static)
        if not np.array_equal(initial.faces, clean.faces):
            raise RuntimeError(f"Initial/clean connectivity mismatch for {sample_id}.")
        stored_target = torch.as_tensor(
            static.get("raw_laplacian_target", static["laplacian_target"]), dtype=torch.float32
        ).cpu()
        recomputed_target = torch.as_tensor(raw_uniform_laplacian(clean), dtype=torch.float32)
        target_error = float(torch.max(torch.abs(stored_target - recomputed_target)))
        if target_error > args.target_tolerance:
            raise RuntimeError(f"Stored target mismatch for {sample_id}: {target_error}")

        archived_dir = source / "reconstruction" / args.prediction_arm_name / sample_id
        archived_prediction_path = archived_dir / "delta_pred_raw.npy"
        archived_mesh_path = archived_dir / "predicted_refined.obj"
        for path in (archived_prediction_path, archived_mesh_path, archived_dir / "coarse.obj"):
            if not path.is_file():
                raise FileNotFoundError(path)
        archived_initial = load_mesh(archived_dir / "coarse.obj").ensure_normals()
        predicted = load_mesh(archived_mesh_path).ensure_normals()
        same_archived_initial = bool(
            np.array_equal(initial.faces, archived_initial.faces)
            and np.allclose(initial.vertices, archived_initial.vertices, rtol=0.0, atol=1e-8)
        )
        if not same_archived_initial:
            raise RuntimeError(f"Archived input mismatch for {sample_id}.")
        archived_prediction = torch.as_tensor(np.load(archived_prediction_path), dtype=torch.float32)

        # Archived recovery directories do not contain the learned per-vertex
        # confidence. Re-run the exact checkpoint solely to recover confidence and
        # visibility weights, while retaining archived delta and recovered geometry.
        inferred = _infer_one(dataset, index, spec, device, current_faces=static["faces"])
        inference_prediction_error = float(
            torch.max(torch.abs(torch.as_tensor(inferred["prediction_raw"]) - archived_prediction))
        )
        confidence = torch.as_tensor(inferred["confidence"]).float().cpu()
        visible = torch.as_tensor(inferred["visibility_count"]).cpu() > 0
        valid = torch.as_tensor(inferred["valid"]).bool().cpu()
        h = torch.as_tensor(static["local_edge_length"]).float().cpu()
        target_normalized = normalize_laplacian_by_edge_scale(
            stored_target, h, eps=epsilon, valid_scale_mask=valid
        )
        archived_prediction_normalized = normalize_laplacian_by_edge_scale(
            archived_prediction, h, eps=epsilon, valid_scale_mask=valid
        )
        replay_dir = output / "archived_recovery_replay" / sample_id
        _, replay_vertices = _recover_raw_one(
            static,
            archived_prediction,
            archived_prediction_normalized,
            confidence,
            replay_dir,
            config,
        )
        replay_delta = replay_vertices - predicted.vertices
        replay_max_error = float(np.max(np.abs(replay_delta), initial=0.0))
        replay_vertex_rms_error = float(
            np.sqrt(np.mean(np.sum(replay_delta**2, axis=1)))
        )
        shutil.rmtree(replay_dir)
        oracle_dir = output / "oracle_recovery" / sample_id
        _, oracle_vertices = _recover_raw_one(
            static, stored_target, target_normalized, confidence, oracle_dir, config
        )
        oracle = load_mesh(oracle_dir / "predicted_refined.obj").ensure_normals()

        meshes = {
            "initial": initial,
            "clean": clean,
            "exact_target_oracle": oracle,
            "predicted_recovery": predicted,
        }
        sample_geometry = {
            state: _geometry_row(args.dataset_arm, sample_id, state, mesh, clean, initial)
            for state, mesh in meshes.items()
        }
        geometry_rows.extend(sample_geometry.values())
        prediction_metric = {
            "dataset_arm": args.dataset_arm,
            "sample_id": sample_id,
            "variant": metadata.get("variant"),
            **_prediction_metrics(archived_prediction, stored_target, valid, visible),
        }
        prediction_rows.append(prediction_metric)

        initial_cd = float(sample_geometry["initial"]["chamfer"])
        clean_cd = float(sample_geometry["clean"]["chamfer"])
        oracle_cd = float(sample_geometry["exact_target_oracle"]["chamfer"])
        pred_cd = float(sample_geometry["predicted_recovery"]["chamfer"])
        available = initial_cd - clean_cd
        oracle_gain = initial_cd - oracle_cd
        pred_gain = initial_cd - pred_cd
        vertex_delta = predicted.vertices - oracle.vertices
        gap_rows.append(
            {
                "dataset_arm": args.dataset_arm,
                "sample_id": sample_id,
                "variant": metadata.get("variant"),
                "initial_chamfer": initial_cd,
                "clean_chamfer": clean_cd,
                "oracle_chamfer": oracle_cd,
                "predicted_chamfer": pred_cd,
                "predicted_chamfer_degradation_vs_initial": pred_cd - initial_cd,
                "g_available": available,
                "g_oracle": oracle_gain,
                "g_pred": pred_gain,
                "eta_oracle": oracle_gain / available if available > 1e-12 else float("nan"),
                "eta_pred": pred_gain / oracle_gain if oracle_gain > 1e-12 else float("nan"),
                "oracle_pred_vertex_rms_displacement": float(np.sqrt(np.mean(np.sum(vertex_delta**2, axis=1)))),
                "oracle_pred_vertex_max_displacement": float(np.linalg.norm(vertex_delta, axis=1).max()),
                "pred_minus_oracle_chamfer": pred_cd - oracle_cd,
                "pred_minus_oracle_normal": float(sample_geometry["predicted_recovery"]["normal_consistency"])
                - float(sample_geometry["exact_target_oracle"]["normal_consistency"]),
                "pred_minus_oracle_flips": int(sample_geometry["predicted_recovery"]["introduced_flipped_faces"])
                - int(sample_geometry["exact_target_oracle"]["introduced_flipped_faces"]),
                "predicted_introduced_flips": int(
                    sample_geometry["predicted_recovery"]["introduced_flipped_faces"]
                ),
                **{key: value for key, value in prediction_metric.items() if key not in ("dataset_arm", "sample_id", "variant")},
            }
        )
        flip_row, flip_arrays = _flip_attribution(
            args.dataset_arm,
            sample_id,
            initial,
            clean,
            oracle,
            predicted,
            stored_target.numpy(),
        )
        flip_rows.append(flip_row)
        for key, value in flip_arrays.items():
            if len(value):
                area_arrays[key].append(value)

        spectral_error = ""
        if sample_id in selected_spectral:
            try:
                spectral_rows.append(
                    {
                        "dataset_arm": args.dataset_arm,
                        "sample_id": sample_id,
                        "variant": metadata.get("variant"),
                        **_spectral_metrics(
                            (archived_prediction - stored_target).double().numpy(), initial.faces
                        ),
                    }
                )
            except Exception as exc:  # keep the main oracle audit usable and report explicitly
                spectral_error = f"{type(exc).__name__}: {exc}"
                spectral_rows.append(
                    {
                        "dataset_arm": args.dataset_arm,
                        "sample_id": sample_id,
                        "variant": metadata.get("variant"),
                        "spectral_protocol": SPECTRAL_PROTOCOL,
                        "spectral_error": spectral_error,
                    }
                )

        audit = {
            "dataset_arm": args.dataset_arm,
            "sample_id": sample_id,
            "manifest": str(manifest),
            "manifest_sha256": _sha256(manifest),
            "prepared_sample_path": str(record.path),
            "input_mesh_source": f"{record.path}:vertices/faces",
            "clean_mesh_source": f"{record.path}:clean_reference_vertices/clean_reference_faces",
            "stored_target_source": f"{record.path}:raw_laplacian_target",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_optimizer_steps": int(spec["optimizer_steps"]),
            "archived_prediction": str(archived_prediction_path),
            "archived_recovered_mesh": str(archived_mesh_path),
            "archived_input_exact": same_archived_initial,
            "initial_clean_faces_exact": bool(np.array_equal(initial.faces, clean.faces)),
            "target_recompute_max_abs_float32_error": target_error,
            "target_recompute_tolerance": args.target_tolerance,
            "target_convention": "raw_uniform_laplacian(clean_reference, sample faces)",
            "inference_rerun_role": "confidence_and_visibility_audit_only",
            "archived_delta_used": True,
            "archived_recovered_mesh_used": True,
            "inference_rerun_vs_archived_delta_max_abs_error": inference_prediction_error,
            "archived_recovery_replay_max_abs_vertex_error": replay_max_error,
            "archived_recovery_replay_vertex_rms_error": replay_vertex_rms_error,
            "archived_recovery_replay_tolerance": args.replay_tolerance,
            "confidence_mean": float(confidence.mean()),
            "same_recovery_config": True,
            "recovery_config": config.get("recovery"),
            "no_projection": True,
            "no_nearest_vertex": True,
            "no_icp": True,
            "no_topology_transfer": True,
            "metric_protocol": METRIC_PROTOCOL,
            "spectral_selected": sample_id in selected_spectral,
            "spectral_error": spectral_error,
        }
        audit["passed"] = bool(
            same_archived_initial
            and audit["initial_clean_faces_exact"]
            and target_error <= args.target_tolerance
            and np.array_equal(initial.faces, oracle.faces)
            and np.array_equal(initial.faces, predicted.faces)
            and int(spec["optimizer_steps"]) == 20_000
            and replay_max_error <= args.replay_tolerance
        )
        audit_rows.append(audit)
        _write_json(output / "sample_audits" / f"{sample_id}.json", audit)
        print(
            f"{args.dataset_arm} {sample_id}: initial={initial_cd:.8g} oracle={oracle_cd:.8g} "
            f"pred={pred_cd:.8g} eta_oracle={oracle_gain/max(available,1e-12):.4g} audit={audit['passed']}",
            flush=True,
        )
        del inferred
        if device.type == "cuda":
            torch.cuda.empty_cache()

    shard_dir = output / "shards"
    _write_json(
        shard_dir / f"shard_{args.shard_index:02d}.json",
        {
            "dataset_arm": args.dataset_arm,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "manifest": str(manifest),
            "manifest_sha256": _sha256(manifest),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "prediction_source_dir": str(source),
            "prediction_arm_name": args.prediction_arm_name,
            "metric_protocol": METRIC_PROTOCOL,
            "spectral_protocol": SPECTRAL_PROTOCOL,
            "geometry_rows": geometry_rows,
            "prediction_rows": prediction_rows,
            "gap_rows": gap_rows,
            "flip_rows": flip_rows,
            "spectral_rows": spectral_rows,
            "audit_rows": audit_rows,
        },
    )
    np.savez_compressed(
        shard_dir / f"flip_areas_{args.shard_index:02d}.npz",
        **{
            key: np.concatenate(values) if values else np.empty(0, dtype=np.float64)
            for key, values in area_arrays.items()
        },
    )


def _mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    return {
        "count": int(len(finite)),
        "mean": float(finite.mean()) if len(finite) else float("nan"),
        "median": float(np.median(finite)) if len(finite) else float("nan"),
        "p10": float(np.quantile(finite, 0.1)) if len(finite) else float("nan"),
        "p90": float(np.quantile(finite, 0.9)) if len(finite) else float("nan"),
        "negative_count": int((finite < 0).sum()),
    }


def _correlation(rows: Sequence[Mapping[str, Any]], x: str, y: str) -> dict[str, Any]:
    from scipy.stats import spearmanr

    values = np.asarray(
        [(float(row[x]), float(row[y])) for row in rows if np.isfinite(float(row[x])) and np.isfinite(float(row[y]))]
    )
    if len(values) < 3 or np.std(values[:, 0]) == 0 or np.std(values[:, 1]) == 0:
        return {"x": x, "y": y, "count": int(len(values)), "pearson": float("nan"), "spearman": float("nan")}
    return {
        "x": x,
        "y": y,
        "count": int(len(values)),
        "pearson": float(np.corrcoef(values[:, 0], values[:, 1])[0, 1]),
        "spearman": float(spearmanr(values[:, 0], values[:, 1]).statistic),
    }


def merge(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    payloads = [_read_json(output / "shards" / f"shard_{i:02d}.json") for i in range(args.shard_count)]
    if any(str(payload["dataset_arm"]) != args.dataset_arm for payload in payloads):
        raise RuntimeError("Shard dataset-arm mismatch.")
    geometry = [row for payload in payloads for row in payload["geometry_rows"]]
    predictions = [row for payload in payloads for row in payload["prediction_rows"]]
    gaps = [row for payload in payloads for row in payload["gap_rows"]]
    flips = [row for payload in payloads for row in payload["flip_rows"]]
    spectral = [row for payload in payloads for row in payload["spectral_rows"]]
    audits = [row for payload in payloads for row in payload["audit_rows"]]
    expected = len(PreparedMeshDataset.from_manifest(args.manifest.resolve(), "test"))
    dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), "test")
    valid_counts = {
        str(dataset.load_static(index)["sample_id"]): int(
            torch.as_tensor(dataset.load_static(index)["valid_scale_mask"]).sum().item()
        )
        for index in range(len(dataset))
    }
    flip_by_sample = {str(row["sample_id"]): row for row in flips}
    for row in gaps:
        flip = flip_by_sample[str(row["sample_id"])]
        row["predicted_introduced_flip_fraction"] = int(
            flip["predicted_introduced_flips"]
        ) / max(int(flip["faces"]), 1)
    audit = {
        "passed": bool(
            len(gaps) == expected
            and len(geometry) == expected * len(STATES)
            and all(bool(row["passed"]) for row in audits)
            and len({row["prepared_sample_path"] for row in audits}) == expected
        ),
        "dataset_arm": args.dataset_arm,
        "expected_test_samples": expected,
        "evaluated_test_samples": len(gaps),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": payloads[0]["manifest_sha256"],
        "checkpoint": payloads[0]["checkpoint"],
        "checkpoint_sha256": payloads[0]["checkpoint_sha256"],
        "prediction_source_dir": payloads[0]["prediction_source_dir"],
        "prediction_arm_name": payloads[0]["prediction_arm_name"],
        "all_exact_native_target_recomputations_pass": all(
            float(row["target_recompute_max_abs_float32_error"]) <= float(row["target_recompute_tolerance"])
            for row in audits
        ),
        "maximum_target_recompute_error": max(float(row["target_recompute_max_abs_float32_error"]) for row in audits),
        "all_archived_inputs_match_dataset": all(bool(row["archived_input_exact"]) for row in audits),
        "maximum_inference_rerun_vs_archived_delta_error": max(
            float(row["inference_rerun_vs_archived_delta_max_abs_error"]) for row in audits
        ),
        "maximum_archived_recovery_replay_max_abs_vertex_error": max(
            float(row["archived_recovery_replay_max_abs_vertex_error"]) for row in audits
        ),
        "maximum_archived_recovery_replay_vertex_rms_error": max(
            float(row["archived_recovery_replay_vertex_rms_error"]) for row in audits
        ),
        "archived_recovery_replay_tolerance": args.replay_tolerance,
        "all_archived_recoveries_replayed_with_frozen_weights": all(
            float(row["archived_recovery_replay_max_abs_vertex_error"])
            <= float(row["archived_recovery_replay_tolerance"])
            for row in audits
        ),
        "archived_delta_and_recovered_mesh_reused": True,
        "inference_rerun_used_only_for_confidence_visibility_audit": True,
        "no_projection_nearest_vertex_icp_or_topology_transfer": True,
        "metric_protocol": METRIC_PROTOCOL,
        "legacy_vertex_sampled_chamfer_excluded": True,
    }
    _write_json(output / "contract_audit.json", audit)
    if not audit["passed"]:
        raise RuntimeError("Contract audit failed.")

    by_sample_state = {(row["sample_id"], row["state"]): row for row in geometry}
    aggregates = []
    for state in STATES:
        selected = [row for row in geometry if row["state"] == state]
        initial_rows = [by_sample_state[(row["sample_id"], "initial")] for row in selected]
        initial_mean = _mean(initial_rows, "chamfer")
        chamfer = _mean(selected, "chamfer")
        aggregates.append(
            {
                "dataset_arm": args.dataset_arm,
                "state": state,
                "samples": len(selected),
                "chamfer": chamfer,
                "p2s": _mean(selected, "p2s"),
                "p2s_p95": _mean(selected, "p2s_p95"),
                "fscore": _mean(selected, "fscore"),
                "normal_consistency": _mean(selected, "normal_consistency"),
                "introduced_flipped_faces": int(sum(int(row["introduced_flipped_faces"]) for row in selected)),
                "new_degenerate_faces": int(sum(int(row["new_degenerate_faces"]) for row in selected)),
                "improved_over_initial": int(sum(float(row["chamfer"]) < float(by_sample_state[(row["sample_id"], "initial")]["chamfer"]) for row in selected)),
                "worsened_over_initial": int(sum(float(row["chamfer"]) > float(by_sample_state[(row["sample_id"], "initial")]["chamfer"]) for row in selected)),
                "delta_cd": initial_mean - chamfer,
                "relative_gain": (initial_mean - chamfer) / max(initial_mean, 1e-12),
            }
        )

    area_values: dict[str, list[np.ndarray]] = defaultdict(list)
    for index in range(args.shard_count):
        with np.load(output / "shards" / f"flip_areas_{index:02d}.npz") as archive:
            for key in archive.files:
                area_values[key].append(np.asarray(archive[key]))
    flip_summary = {
        "initial_wrong_orientation_vs_clean": int(sum(int(row["initial_wrong_orientation_vs_clean"]) for row in flips)),
        "oracle_introduced_flips": int(sum(int(row["oracle_introduced_flips"]) for row in flips)),
        "predicted_introduced_flips": int(sum(int(row["predicted_introduced_flips"]) for row in flips)),
        "oracle_predicted_flip_overlap": int(sum(int(row["oracle_predicted_flip_overlap"]) for row in flips)),
        "oracle_only_flips": int(sum(int(row["oracle_only_flips"]) for row in flips)),
        "prediction_only_flips": int(sum(int(row["prediction_only_flips"]) for row in flips)),
        "high_target_faces": int(sum(int(row["high_target_faces"]) for row in flips)),
        "all_faces": int(sum(int(row["faces"]) for row in flips)),
        "oracle_flips_high_target": int(sum(int(row["oracle_flips_high_target"]) for row in flips)),
        "predicted_flips_high_target": int(sum(int(row["predicted_flips_high_target"]) for row in flips)),
        "oracle_flip_area_normalized_by_bbox_diagonal_squared": _distribution(
            np.concatenate(area_values.get("oracle_flip_area", [np.empty(0)]))
        ),
        "predicted_flip_area_normalized_by_bbox_diagonal_squared": _distribution(
            np.concatenate(area_values.get("predicted_flip_area", [np.empty(0)]))
        ),
        "oracle_flip_mean_face_displacement": float(np.nanmean([float(row["oracle_flip_mean_face_displacement"]) for row in flips])),
        "oracle_nonflip_mean_face_displacement": float(np.nanmean([float(row["oracle_nonflip_mean_face_displacement"]) for row in flips])),
        "predicted_flip_mean_face_displacement": float(np.nanmean([float(row["predicted_flip_mean_face_displacement"]) for row in flips])),
        "predicted_nonflip_mean_face_displacement": float(np.nanmean([float(row["predicted_nonflip_mean_face_displacement"]) for row in flips])),
    }
    spectral_valid = [row for row in spectral if not row.get("spectral_error")]
    spectral_summary = {
        "protocol": SPECTRAL_PROTOCOL,
        "selected_count": len(spectral),
        "successful_count": len(spectral_valid),
        "failures": [row for row in spectral if row.get("spectral_error")],
        "low_error_fraction_mean": _mean(spectral_valid, "spectral_low_error_fraction") if spectral_valid else float("nan"),
        "mid_error_fraction_mean": _mean(spectral_valid, "spectral_mid_error_fraction") if spectral_valid else float("nan"),
        "high_error_fraction_mean": _mean(spectral_valid, "spectral_high_error_fraction") if spectral_valid else float("nan"),
        "low_error_energy_mean": _mean(spectral_valid, "spectral_low_error_energy") if spectral_valid else float("nan"),
        "mid_error_energy_mean": _mean(spectral_valid, "spectral_mid_error_energy") if spectral_valid else float("nan"),
        "high_error_energy_mean": _mean(spectral_valid, "spectral_high_error_energy") if spectral_valid else float("nan"),
    }
    correlations = []
    for x in ("raw_epe", "raw_rms", "mean_vector_bias_norm", "mean_magnitude_error", "bottom90_epe", "top10_epe", "top1_epe", "visible_epe", "invisible_epe"):
        for y in (
            "predicted_chamfer_degradation_vs_initial",
            "pred_minus_oracle_chamfer",
            "oracle_pred_vertex_rms_displacement",
            "predicted_introduced_flips",
            "predicted_introduced_flip_fraction",
        ):
            correlations.append(_correlation(gaps, x, y))

    total_valid = sum(valid_counts.values())
    prediction_by_sample = {str(row["sample_id"]): row for row in predictions}
    global_prediction = {
        "valid_vertices": total_valid,
        "raw_epe": sum(
            float(prediction_by_sample[sample_id]["raw_epe"]) * count
            for sample_id, count in valid_counts.items()
        )
        / total_valid,
        "raw_rms": math.sqrt(
            sum(
                float(prediction_by_sample[sample_id]["raw_rms"]) ** 2 * count
                for sample_id, count in valid_counts.items()
            )
            / total_valid
        ),
        "aggregation": "global_vertex_weighted_from_exact_per_sample_sums",
    }

    summary = {
        "contract_audit": audit,
        "dataset_arm": args.dataset_arm,
        "metric_protocol": METRIC_PROTOCOL,
        "aggregate": aggregates,
        "oracle_efficiency": _distribution([float(row["eta_oracle"]) for row in gaps]),
        "prediction_retention": _distribution([float(row["eta_pred"]) for row in gaps]),
        "predicted_vs_oracle_gap": {
            "vertex_rms_displacement_mean": _mean(gaps, "oracle_pred_vertex_rms_displacement"),
            "vertex_max_displacement_mean": _mean(gaps, "oracle_pred_vertex_max_displacement"),
            "chamfer_difference_mean": _mean(gaps, "pred_minus_oracle_chamfer"),
            "normal_difference_mean": _mean(gaps, "pred_minus_oracle_normal"),
            "flip_count_difference_total": int(sum(int(row["pred_minus_oracle_flips"]) for row in gaps)),
        },
        "prediction_metrics_mean": {
            field: _mean(predictions, field)
            for field in (
                "raw_epe", "raw_rms", "raw_max", "raw_cosine", "mean_vector_bias_x",
                "mean_vector_bias_y", "mean_vector_bias_z", "mean_vector_bias_norm",
                "mean_magnitude_error", "bottom90_epe", "top10_epe", "top1_epe",
                "visible_epe", "invisible_epe", "visible_fraction",
            )
        },
        "prediction_metrics_global_weighted": global_prediction,
        "correlations": correlations,
        "spectral": spectral_summary,
        "flip_attribution": flip_summary,
    }
    _write_json(output / "summary.json", summary)
    _write_csv(output / "geometry_per_sample.csv", geometry)
    _write_csv(output / "prediction_per_sample.csv", predictions)
    _write_csv(output / "oracle_gap_per_sample.csv", gaps)
    _write_csv(output / "flip_attribution_per_sample.csv", flips)
    _write_csv(output / "spectral_per_sample.csv", spectral)
    _write_csv(output / "correlations.csv", correlations)
    _write_csv(output / "aggregate.csv", aggregates)
    _write_json(output / "per_sample_contract_audit.json", audits)
    print(json.dumps(summary, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--prediction-source-dir", type=Path)
    parser.add_argument("--prediction-arm-name")
    parser.add_argument("--dataset-arm", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-tolerance", type=float, default=1e-7)
    parser.add_argument("--replay-tolerance", type=float, default=5e-5)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    if args.merge_only:
        merge(args)
    else:
        for name in ("run_dir", "prediction_source_dir", "prediction_arm_name"):
            if getattr(args, name) is None:
                parser.error(f"--{name.replace('_', '-')} is required unless --merge-only is used")
        evaluate_shard(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
