from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mlr.data import Mesh
from mlr.io import save_mesh

from .dataset import load_prepared_sample
from .evaluation import _chamfer_distance, _normal_consistency, _point_to_surface_stats, _reconstruct
from .graph_layers import faces_to_edge_index
from .perturbed_scale_sweep import _render_panel
from .recovery_identity_oracle import (
    _horizontal_composite,
    _load_fixed_cameras,
    _topology_change,
    _vertex_features,
)
from .recovery_targets import compose_absolute_laplacian_target, initial_uniform_laplacian
from .target_scaling import (
    EDGE_SCALE_NORMALIZED_LAPLACIAN,
    RAW_LAPLACIAN,
    denormalize_laplacian_by_edge_scale,
    incident_edge_length_and_valid_mask,
    normalize_laplacian_by_edge_scale,
    require_matching_laplacian_representations,
)


VARIANTS = ("control", "perturbed")


def run_audit(
    source_run: str | Path,
    output_dir: str | Path,
    *,
    render_backend: str = "opengl",
) -> dict[str, Any]:
    source = Path(source_run).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_config = _read_json(source / "config.yaml")
    model_config = _read_json(Path(source_config["model_config"]))
    recovery_payload = _read_json(Path(source_config["recovery_config"]))
    solver_mapping = dict(recovery_payload["reconstruction"])
    solver_mapping["evaluate_oracle"] = False
    epsilon = float(model_config.get("target_scaling", {}).get("epsilon", 1e-12))
    if model_config.get("target_mode") != EDGE_SCALE_NORMALIZED_LAPLACIAN:
        raise ValueError("This audit expected the frozen checkpoint to use h^2-normalized targets.")
    samples = _load_pairs(source)
    cameras = _load_fixed_cameras(source / "visualizations" / "render_metadata.json")
    _copy_configs(output, source_config)
    training_trace = _training_trace(source_config, model_config, epsilon)

    roundtrip_rows: list[dict[str, Any]] = []
    h_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    direct_rows: list[dict[str, Any]] = []
    oracle_raw_rows: list[dict[str, Any]] = []
    oracle_hat_rows: list[dict[str, Any]] = []
    spike_rows: list[dict[str, Any]] = []
    trace_records: list[dict[str, Any]] = []
    representative_id = sorted(samples)[0]

    for sample_id, pair in samples.items():
        control = pair["control"]
        perturbed = pair["perturbed"]
        faces = _np(control["faces"]).astype(np.int64)
        if not np.array_equal(faces, _np(perturbed["faces"]).astype(np.int64)):
            raise ValueError(f"Control/perturbed topology mismatch for {sample_id}.")
        midpoint = _midpoint_mask(Path(source_config["sofa_models_root"]), sample_id, len(faces))
        for variant, sample in pair.items():
            vertices = _np(sample["vertices"]).astype(np.float64)
            mesh = Mesh(vertices, faces).ensure_normals()
            edge_index = faces_to_edge_index(torch.as_tensor(faces), len(vertices))
            recomputed_h, valid = incident_edge_length_and_valid_mask(
                torch.as_tensor(vertices), edge_index, eps=epsilon
            )
            stored_h = torch.as_tensor(sample["local_edge_length"], dtype=torch.float64)
            h_error = torch.max(torch.abs(recomputed_h - stored_h)).item()
            if h_error > 2e-7:
                raise AssertionError(
                    f"Stored h is not the current expanded-graph h for {variant}/{sample_id}: {h_error}"
                )
            h = recomputed_h.numpy()
            if sample_id == representative_id and variant == "control":
                np.save(output / "expanded_h.npy", h)
            delta_raw = initial_uniform_laplacian(vertices, faces)
            delta_hat = normalize_laplacian_by_edge_scale(
                torch.as_tensor(delta_raw), recomputed_h, eps=epsilon, valid_scale_mask=valid
            )
            delta_roundtrip = denormalize_laplacian_by_edge_scale(
                delta_hat, recomputed_h, eps=epsilon
            ).numpy()
            error = delta_roundtrip - delta_raw
            flat_raw = delta_raw.reshape(-1)
            flat_roundtrip = delta_roundtrip.reshape(-1)
            roundtrip_rows.append(
                {
                    "sample_id": sample_id,
                    "dataset_variant": variant,
                    "max_absolute_error": float(np.max(np.abs(error))),
                    "mean_absolute_error": float(np.mean(np.abs(error))),
                    "relative_l2_error": float(np.linalg.norm(error) / max(np.linalg.norm(delta_raw), 1e-30)),
                    "cosine_similarity": float(np.dot(flat_raw, flat_roundtrip) / max(np.linalg.norm(flat_raw) * np.linalg.norm(flat_roundtrip), 1e-30)),
                    "max_per_vertex_vector_error": float(np.max(np.linalg.norm(error, axis=1))),
                    "stored_vs_recomputed_h_max_abs_error": h_error,
                }
            )
            features = _vertex_features(mesh)
            spike_ids = _previous_spike_ids(source.parent / "sofa50_recovery_identity_oracle_diagnostic", sample_id, variant)
            groups = {
                "all_expanded": np.ones(len(vertices), dtype=bool),
                "original_coarse_vertices": ~midpoint,
                "inserted_midpoint_vertices": midpoint,
                "sharp_high_curvature_vertices": features["high_sharp"],
                "previous_top_spike_vertices": np.isin(np.arange(len(vertices)), spike_ids),
            }
            for group, mask in groups.items():
                h_rows.append(
                    {
                        "sample_id": sample_id,
                        "dataset_variant": variant,
                        "group": group,
                        **_h_statistics(h[mask]),
                    }
                )

            prediction = _prediction_cache(source, variant, sample_id)
            delta_hat_pred = prediction["normalized_prediction"].astype(np.float64)
            delta_raw_pred = denormalize_laplacian_by_edge_scale(
                torch.as_tensor(delta_hat_pred), recomputed_h, eps=epsilon
            ).numpy()
            legacy_raw = prediction["raw_prediction"].astype(np.float64)
            scale_factor = h**2 + epsilon
            h2_only_raw = delta_hat_pred * h[:, None] ** 2
            prediction_rows.append(
                {
                    "sample_id": sample_id,
                    "dataset_variant": variant,
                    **_prefixed_vector_stats("delta_hat_pred", delta_hat_pred),
                    **_prefixed_vector_stats("delta_raw_pred", delta_raw_pred),
                    **_prefixed_scalar_stats("h2_plus_eps", scale_factor),
                    "epsilon_omission_max_abs_error": float(np.max(np.abs(h2_only_raw - delta_raw_pred))),
                    "epsilon_omission_relative_l2_error": float(np.linalg.norm(h2_only_raw - delta_raw_pred) / max(np.linalg.norm(delta_raw_pred), 1e-30)),
                    "cached_float_raw_vs_corrected_max_abs_error": float(np.max(np.abs(legacy_raw - delta_raw_pred))),
                    "cached_float_raw_vs_corrected_relative_l2_error": float(np.linalg.norm(legacy_raw - delta_raw_pred) / max(np.linalg.norm(delta_raw_pred), 1e-30)),
                }
            )
            npz_dir = output / "predictions" / variant
            npz_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                npz_dir / f"{sample_id}_representations.npz",
                delta_hat_pred=delta_hat_pred,
                delta_raw_pred=delta_raw_pred,
                h_current=h,
                h2_plus_eps=scale_factor,
                visibility_count=prediction["visibility_count"],
            )
            visibility_weight = prediction["visibility_count"] > 0
            delta_target = compose_absolute_laplacian_target(
                delta_raw, delta_raw_pred, 1.0, visibility_weight
            )
            trace_records.append(
                {
                    "sample_id": sample_id,
                    "dataset_variant": variant,
                    "stages": [
                        _trace_stage("model_output", delta_hat_pred, "H2_NORMALIZED_LAPLACIAN"),
                        _trace_stage("current_expanded_h", h, "mean one-ring edge length in raw mesh coordinates"),
                        _trace_stage("h2_plus_epsilon", scale_factor, "length^2"),
                        _trace_stage("denormalized_prediction", delta_raw_pred, "RAW_LAPLACIAN"),
                        _trace_stage("initial_current_graph_laplacian", delta_raw, "RAW_LAPLACIAN"),
                        _trace_stage("visibility_composed_absolute_target", delta_target, "RAW_LAPLACIAN"),
                    ],
                    "h_source": "recomputed from this variant's current expanded vertices and faces",
                    "h_stored_vs_recomputed_max_abs_error": h_error,
                    "epsilon": epsilon,
                    "solver": "existing sparse uniform recovery",
                }
            )

            recovered_variants = {}
            for recovery_variant, absolute_prediction in (
                ("incorrect_direct", delta_hat_pred),
                ("recovered_raw", delta_raw_pred),
            ):
                target = compose_absolute_laplacian_target(
                    delta_raw, absolute_prediction, 1.0, visibility_weight
                )
                result, solver_name = _reconstruct(
                    mesh,
                    target,
                    np.ones(len(vertices), dtype=np.float64),
                    _refinement_config(solver_mapping),
                    int(solver_mapping.get("dense_vertex_limit", 5000)),
                    laplacian_weight=np.ones(len(vertices), dtype=np.float64),
                )
                recovered = result.mesh.ensure_normals()
                recovered_variants[recovery_variant] = recovered
                mesh_path = output / "meshes" / variant / sample_id / f"{recovery_variant}.obj"
                save_mesh(recovered, mesh_path)
                direct_rows.append(
                    _recovery_metrics(
                        sample_id,
                        variant,
                        recovery_variant,
                        mesh,
                        recovered,
                        sample,
                        mesh_path,
                        solver_name,
                        result.history[-1],
                        solver_mapping,
                    )
                )
            _render_comparison(
                output,
                sample_id,
                variant,
                mesh,
                recovered_variants["incorrect_direct"],
                recovered_variants["recovered_raw"],
                Mesh(_np(sample["gt_vertices"]), _np(sample["gt_faces"]).astype(np.int64)).ensure_normals(),
                cameras[sample_id],
                render_backend,
            )
            direct_displacement = np.linalg.norm(
                recovered_variants["incorrect_direct"].vertices - vertices, axis=1
            )
            raw_displacement = np.linalg.norm(
                recovered_variants["recovered_raw"].vertices - vertices, axis=1
            )
            for vertex_id in spike_ids:
                spike_rows.append(
                    {
                        "sample_id": sample_id,
                        "dataset_variant": variant,
                        "vertex_id": int(vertex_id),
                        "h": float(h[vertex_id]),
                        "h2": float(h[vertex_id] ** 2),
                        "pred_hat_norm": float(np.linalg.norm(delta_hat_pred[vertex_id])),
                        "pred_raw_norm": float(np.linalg.norm(delta_raw_pred[vertex_id])),
                        "pred_hat_over_pred_raw": float(np.linalg.norm(delta_hat_pred[vertex_id]) / max(np.linalg.norm(delta_raw_pred[vertex_id]), 1e-30)),
                        "visibility_count": int(prediction["visibility_count"][vertex_id]),
                        "mean_local_edge_length": float(features["mean_edge"][vertex_id]),
                        "minimum_local_edge_length": float(features["min_edge"][vertex_id]),
                        "maximum_local_edge_length": float(features["max_edge"][vertex_id]),
                        "sharp_high_curvature": bool(features["high_sharp"][vertex_id]),
                        "inserted_midpoint": bool(midpoint[vertex_id]),
                        "incorrect_direct_displacement": float(direct_displacement[vertex_id]),
                        "recovered_raw_displacement": float(raw_displacement[vertex_id]),
                    }
                )

        xp = _np(perturbed["vertices"]).astype(np.float64)
        xc = _np(control["vertices"]).astype(np.float64)
        hp = torch.as_tensor(perturbed["local_edge_length"], dtype=torch.float64)
        oracle_raw = initial_uniform_laplacian(xc, faces)
        oracle_hat = normalize_laplacian_by_edge_scale(
            torch.as_tensor(oracle_raw), hp, eps=epsilon
        ).numpy()
        pred = _prediction_cache(source, "perturbed", sample_id)
        pred_hat = pred["normalized_prediction"].astype(np.float64)
        pred_raw = denormalize_laplacian_by_edge_scale(
            torch.as_tensor(pred_hat), hp, eps=epsilon
        ).numpy()
        require_matching_laplacian_representations(
            EDGE_SCALE_NORMALIZED_LAPLACIAN, EDGE_SCALE_NORMALIZED_LAPLACIAN
        )
        oracle_hat_rows.extend(
            _comparison_rows(sample_id, pred_hat, oracle_hat, target_kind="absolute")
        )
        initial_raw = initial_uniform_laplacian(xp, faces)
        initial_hat = normalize_laplacian_by_edge_scale(
            torch.as_tensor(initial_raw), hp, eps=epsilon
        ).numpy()
        oracle_hat_rows.extend(
            _comparison_rows(
                sample_id,
                pred_hat - initial_hat,
                oracle_hat - initial_hat,
                target_kind="residual",
            )
        )
        require_matching_laplacian_representations(RAW_LAPLACIAN, RAW_LAPLACIAN)
        oracle_raw_rows.extend(
            _comparison_rows(sample_id, pred_raw, oracle_raw, target_kind="absolute")
        )
        oracle_raw_rows.extend(
            _comparison_rows(
                sample_id,
                pred_raw - initial_raw,
                oracle_raw - initial_raw,
                target_kind="residual",
            )
        )

    _write_csv(output / "normalization_roundtrip.csv", roundtrip_rows)
    _write_csv(output / "h_statistics.csv", h_rows)
    _write_csv(output / "prediction_scale_statistics.csv", prediction_rows)
    _write_csv(output / "direct_vs_recovered_raw.csv", direct_rows)
    _write_csv(output / "prediction_vs_oracle_raw.csv", oracle_raw_rows)
    _write_csv(output / "prediction_vs_oracle_normalized.csv", oracle_hat_rows)
    _write_csv(output / "spike_h2_diagnostics.csv", spike_rows)
    inference_trace = {
        "model_call": "model(sample).predicted_laplacian",
        "model_output_representation": "H2_NORMALIZED_LAPLACIAN",
        "conversion": "delta_raw_pred = delta_hat_pred * (h_current^2 + epsilon)",
        "h_definition": "arithmetic mean of unique undirected one-ring incident edge lengths",
        "h_coordinate_system": "current expanded raw mesh coordinates, identical to L input coordinates",
        "epsilon": epsilon,
        "records": trace_records,
    }
    _write_json(output / "inference_representation_trace.json", inference_trace)
    summary = _summary(
        source_config,
        training_trace,
        epsilon,
        roundtrip_rows,
        h_rows,
        prediction_rows,
        direct_rows,
        oracle_hat_rows,
        oracle_raw_rows,
        spike_rows,
    )
    _write_json(output / "summary.json", summary)
    _write_json(output / "training_representation_trace.json", training_trace)
    (output / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    return summary


def finalize_existing_audit(source_run: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Refresh derived tables/report without rerunning recovery or rendering."""

    source = Path(source_run).resolve()
    output = Path(output_dir).resolve()
    source_config = _read_json(source / "config.yaml")
    model_config = _read_json(Path(source_config["model_config"]))
    epsilon = float(model_config.get("target_scaling", {}).get("epsilon", 1e-12))
    prediction_rows = []
    for variant in VARIANTS:
        for path in sorted((output / "predictions" / variant).glob("*_representations.npz")):
            sample_id = path.name.removesuffix("_representations.npz")
            with np.load(path) as archive:
                delta_hat_pred = archive["delta_hat_pred"].astype(np.float64)
                delta_raw_pred = archive["delta_raw_pred"].astype(np.float64)
                h = archive["h_current"].astype(np.float64)
                scale_factor = archive["h2_plus_eps"].astype(np.float64)
            h2_only_raw = delta_hat_pred * h[:, None] ** 2
            legacy = _prediction_cache(source, variant, sample_id)["raw_prediction"].astype(np.float64)
            prediction_rows.append(
                {
                    "sample_id": sample_id,
                    "dataset_variant": variant,
                    **_prefixed_vector_stats("delta_hat_pred", delta_hat_pred),
                    **_prefixed_vector_stats("delta_raw_pred", delta_raw_pred),
                    **_prefixed_scalar_stats("h2_plus_eps", scale_factor),
                    "epsilon_omission_max_abs_error": float(np.max(np.abs(h2_only_raw - delta_raw_pred))),
                    "epsilon_omission_relative_l2_error": float(np.linalg.norm(h2_only_raw - delta_raw_pred) / max(np.linalg.norm(delta_raw_pred), 1e-30)),
                    "cached_float_raw_vs_corrected_max_abs_error": float(np.max(np.abs(legacy - delta_raw_pred))),
                    "cached_float_raw_vs_corrected_relative_l2_error": float(np.linalg.norm(legacy - delta_raw_pred) / max(np.linalg.norm(delta_raw_pred), 1e-30)),
                }
            )
    _write_csv(output / "prediction_scale_statistics.csv", prediction_rows)
    oracle_hat_rows, oracle_raw_rows = _rebuild_oracle_comparisons(
        source, output, epsilon
    )
    _write_csv(output / "prediction_vs_oracle_normalized.csv", oracle_hat_rows)
    _write_csv(output / "prediction_vs_oracle_raw.csv", oracle_raw_rows)
    training_trace = _training_trace(source_config, model_config, epsilon)
    _write_json(output / "training_representation_trace.json", training_trace)
    summary = _summary(
        source_config,
        training_trace,
        epsilon,
        _read_csv(output / "normalization_roundtrip.csv"),
        _read_csv(output / "h_statistics.csv"),
        prediction_rows,
        _read_csv(output / "direct_vs_recovered_raw.csv"),
        oracle_hat_rows,
        oracle_raw_rows,
        _read_csv(output / "spike_h2_diagnostics.csv"),
    )
    test_path = output / "test_results.json"
    if test_path.is_file():
        summary["tests"] = _read_json(test_path)
    _write_json(output / "summary.json", summary)
    (output / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _training_trace(source_config: Mapping[str, Any], config: Mapping[str, Any], epsilon: float) -> dict[str, Any]:
    manifest = _read_json(Path(source_config["model_config"]).parent / "dataset_manifest.json")
    record = next(row for row in manifest["samples"] if row["split"] == "validation")
    sample = load_prepared_sample(record["path"], materialize_images=False, dataset_root=Path(record["path"]).parent)
    vertices = _np(sample["vertices"]).astype(np.float64)
    faces = _np(sample["faces"]).astype(np.int64)
    delta_raw = initial_uniform_laplacian(vertices, faces)
    edge_index = faces_to_edge_index(torch.as_tensor(faces), len(vertices))
    h, valid = incident_edge_length_and_valid_mask(torch.as_tensor(vertices), edge_index, eps=epsilon)
    delta_hat = normalize_laplacian_by_edge_scale(torch.as_tensor(delta_raw), h, eps=epsilon, valid_scale_mask=valid).numpy()
    stored_raw = _np(sample["raw_laplacian_target"]).astype(np.float64)
    stored_hat = _np(sample["normalized_laplacian_target"]).astype(np.float64)
    return {
        "sample_id": sample["sample_id"],
        "vertices_shape": list(vertices.shape),
        "vertices_statistics": _trace_stage(
            "training_vertices", vertices, "raw mesh coordinate length"
        ),
        "faces_shape": list(faces.shape),
        "operator": "uniform L = I - D^{-1}A on the sample GT graph",
        "raw_target_formula": "delta_raw = L @ V",
        "raw_target_recompute_max_abs_error": float(np.max(np.abs(delta_raw - stored_raw))),
        "raw_target_statistics": _trace_stage(
            "delta_raw", stored_raw, "RAW_LAPLACIAN"
        ),
        "h_definition": "arithmetic mean of lengths of unique undirected one-ring incident edges",
        "h_statistics": _h_statistics(h.numpy()),
        "normalization_formula": "delta_hat = delta_raw / (h^2 + epsilon) on valid non-isolated vertices",
        "epsilon": epsilon,
        "normalized_target_recompute_max_abs_error": float(np.max(np.abs(delta_hat - stored_hat))),
        "normalized_target_statistics": _trace_stage(
            "delta_hat", stored_hat, "H2_NORMALIZED_LAPLACIAN"
        ),
        "tensor_passed_to_loss": "prepared.training_target = sample['normalized_laplacian_target']",
        "tensor_passed_to_loss_shape": list(stored_hat.shape),
        "tensor_passed_to_loss_statistics": _vector_stats(stored_hat),
        "additional_target_normalization": {
            "dataset_mean_std": False,
            "per_object_target_scale": False,
            "global_coordinate_target_normalization": False,
            "clipping": config.get("target_scaling", {}).get("clip_max_norm"),
            "component_wise_scaling": False,
            "note": "bbox position normalization and RGB mean/std affect model inputs only, not the Laplacian target tensor",
        },
        "model_output_representation": "H2_NORMALIZED_LAPLACIAN",
    }


def _load_pairs(source: Path) -> dict[str, dict[str, Mapping[str, Any]]]:
    result: dict[str, dict[str, Mapping[str, Any]]] = {}
    root = source / "manifests"
    for variant in VARIANTS:
        for path in sorted((root / f"prepared_{variant}").glob("*.pt")):
            result.setdefault(path.stem, {})[variant] = load_prepared_sample(
                path, materialize_images=False, dataset_root=root
            )
    if len(result) != 5 or any(set(pair) != set(VARIANTS) for pair in result.values()):
        raise ValueError("Expected exactly five paired expanded validation samples.")
    return result


def _midpoint_mask(models_root: Path, sample_id: str, face_count: int) -> np.ndarray:
    mapping_path = models_root / sample_id / "subdivision_mapping_raw.npz"
    with np.load(mapping_path) as mapping:
        new_pre = mapping["new_vertex_indices"].astype(np.int64)
        pre_to_final = mapping["pre_compaction_to_final"].astype(np.int64)
        final_to_pre = mapping["final_to_pre_compaction"].astype(np.int64)
    midpoint_indices = pre_to_final[new_pre]
    midpoint_indices = midpoint_indices[midpoint_indices >= 0]
    mask = np.zeros(len(final_to_pre), dtype=bool)
    mask[midpoint_indices] = True
    if not mask.any() or face_count <= 0:
        raise ValueError(f"Invalid midpoint mapping for {sample_id}.")
    return mask


def _previous_spike_ids(diagnostic_root: Path, sample_id: str, variant: str) -> np.ndarray:
    path = diagnostic_root / "spike_vertex_diagnostics.csv"
    if not path.is_file():
        return np.empty(0, dtype=np.int64)
    rows = _read_csv(path)
    selected = {
        int(row["vertex_id"])
        for row in rows
        if row["sample_id"] == sample_id
        and row["dataset_variant"] == variant
        and float(row["scale"]) == 1.0
    }
    return np.asarray(sorted(selected), dtype=np.int64)


def _prediction_cache(source: Path, variant: str, sample_id: str) -> dict[str, np.ndarray]:
    with np.load(source / "cached_predictions" / variant / f"{sample_id}_delta_pred_raw.npz") as archive:
        return {name: archive[name].copy() for name in archive.files}


def _rebuild_oracle_comparisons(
    source: Path, output: Path, epsilon: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for sample_id, pair in _load_pairs(source).items():
        perturbed = pair["perturbed"]
        control = pair["control"]
        xp = _np(perturbed["vertices"]).astype(np.float64)
        xc = _np(control["vertices"]).astype(np.float64)
        faces = _np(perturbed["faces"]).astype(np.int64)
        h = torch.as_tensor(perturbed["local_edge_length"], dtype=torch.float64)
        initial_raw = initial_uniform_laplacian(xp, faces)
        oracle_raw = initial_uniform_laplacian(xc, faces)
        initial_hat = normalize_laplacian_by_edge_scale(
            torch.as_tensor(initial_raw), h, eps=epsilon
        ).numpy()
        oracle_hat = normalize_laplacian_by_edge_scale(
            torch.as_tensor(oracle_raw), h, eps=epsilon
        ).numpy()
        with np.load(output / "predictions" / "perturbed" / f"{sample_id}_representations.npz") as archive:
            pred_hat = archive["delta_hat_pred"].astype(np.float64)
            pred_raw = archive["delta_raw_pred"].astype(np.float64)
        normalized_rows.extend(
            _comparison_rows(sample_id, pred_hat, oracle_hat, target_kind="absolute")
        )
        normalized_rows.extend(
            _comparison_rows(
                sample_id,
                pred_hat - initial_hat,
                oracle_hat - initial_hat,
                target_kind="residual",
            )
        )
        raw_rows.extend(
            _comparison_rows(sample_id, pred_raw, oracle_raw, target_kind="absolute")
        )
        raw_rows.extend(
            _comparison_rows(
                sample_id,
                pred_raw - initial_raw,
                oracle_raw - initial_raw,
                target_kind="residual",
            )
        )
    return normalized_rows, raw_rows


def _comparison_rows(
    sample_id: str,
    prediction: np.ndarray,
    oracle: np.ndarray,
    *,
    target_kind: str,
) -> list[dict[str, Any]]:
    oracle_magnitude = np.linalg.norm(oracle, axis=1)
    masks = {
        "all": np.ones(len(oracle), dtype=bool),
        "top_10_percent_oracle_magnitude": oracle_magnitude >= np.quantile(oracle_magnitude, 0.90),
        "top_1_percent_oracle_magnitude": oracle_magnitude >= np.quantile(oracle_magnitude, 0.99),
    }
    return [
        {
            "sample_id": sample_id,
            "target_kind": target_kind,
            "group": group,
            **_comparison_metrics(prediction[mask], oracle[mask]),
        }
        for group, mask in masks.items()
    ]


def _comparison_metrics(prediction: np.ndarray, oracle: np.ndarray) -> dict[str, Any]:
    pred_mag = np.linalg.norm(prediction, axis=1)
    oracle_mag = np.linalg.norm(oracle, axis=1)
    denominator = pred_mag * oracle_mag
    cosine = np.divide(np.sum(prediction * oracle, axis=1), denominator, out=np.zeros(len(prediction)), where=denominator > 1e-30)
    pred_norm = np.linalg.norm(prediction)
    oracle_norm = np.linalg.norm(oracle)
    return {
        "vertex_count": len(prediction),
        "global_cosine": float(np.sum(prediction * oracle) / max(pred_norm * oracle_norm, 1e-30)),
        "mean_per_vertex_cosine": float(cosine.mean()),
        "median_per_vertex_cosine": float(np.median(cosine)),
        "prediction_oracle_norm_ratio": float(pred_norm / max(oracle_norm, 1e-30)),
        "alpha_star": float(np.sum(prediction * oracle) / max(np.sum(prediction * prediction), 1e-30)),
    }


def _recovery_metrics(
    sample_id: str,
    variant: str,
    recovery_variant: str,
    initial: Mesh,
    recovered: Mesh,
    sample: Mapping[str, Any],
    mesh_path: Path,
    solver_name: str,
    final_terms: Mapping[str, float],
    solver_mapping: Mapping[str, Any],
) -> dict[str, Any]:
    displacement = np.linalg.norm(recovered.vertices - initial.vertices, axis=1)
    gt = Mesh(_np(sample["gt_vertices"]), _np(sample["gt_faces"]).astype(np.int64)).ensure_normals()
    forward = _point_to_surface_stats(recovered.vertices, gt)
    reverse = _point_to_surface_stats(gt.vertices, recovered)
    return {
        "sample_id": sample_id,
        "dataset_variant": variant,
        "recovery_variant": recovery_variant,
        "scale": 1.0,
        "chamfer_to_gt": _chamfer_distance(recovered, gt, int(solver_mapping.get("chamfer_samples", 3000)), int(solver_mapping.get("metric_seed", 7))),
        "point_to_surface_forward_mean": float(forward["mean"]),
        "point_to_surface_reverse_mean": float(reverse["mean"]),
        "point_to_surface_bidirectional_mean": 0.5 * (float(forward["mean"]) + float(reverse["mean"])),
        "normal_consistency_to_gt": _normal_consistency(recovered, gt),
        "mean_displacement_from_initial": float(displacement.mean()),
        "median_displacement_from_initial": float(np.median(displacement)),
        "max_displacement_from_initial": float(displacement.max()),
        "p99_displacement_from_initial": float(np.quantile(displacement, 0.99)),
        **_topology_change(initial, recovered),
        "solver": solver_name,
        "final_loss": float(final_terms["loss"]),
        "mesh_path": str(mesh_path),
    }


def _render_comparison(
    output: Path,
    sample_id: str,
    variant: str,
    initial: Mesh,
    direct: Mesh,
    recovered_raw: Mesh,
    gt: Mesh,
    cameras: Mapping[str, Any],
    backend: str,
) -> None:
    for view, camera in cameras.items():
        paths = []
        for name, mesh, color in (
            ("initial", initial, (180, 205, 220)),
            ("incorrect_direct", direct, (210, 100, 100)),
            ("recovered_raw", recovered_raw, (90, 145, 205)),
            ("gt", gt, (230, 220, 180)),
        ):
            path = output / "visualizations" / variant / sample_id / f"{name}_{view}.png"
            _render_panel(mesh, camera, path, f"{sample_id} | {variant} | {name} | {view}", color, backend)
            paths.append(path)
        _horizontal_composite(paths, output / "visualizations" / variant / sample_id / f"direct_vs_recovered_raw_{view}.png")


def _summary(
    source_config: Mapping[str, Any],
    training_trace: Mapping[str, Any],
    epsilon: float,
    roundtrip: Sequence[Mapping[str, Any]],
    h_rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    recoveries: Sequence[Mapping[str, Any]],
    oracle_hat: Sequence[Mapping[str, Any]],
    oracle_raw: Sequence[Mapping[str, Any]],
    spikes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    all_hat_absolute = [row for row in oracle_hat if row["group"] == "all" and row["target_kind"] == "absolute"]
    all_raw_absolute = [row for row in oracle_raw if row["group"] == "all" and row["target_kind"] == "absolute"]
    all_hat_residual = [row for row in oracle_hat if row["group"] == "all" and row["target_kind"] == "residual"]
    all_raw_residual = [row for row in oracle_raw if row["group"] == "all" and row["target_kind"] == "residual"]
    direct = [row for row in recoveries if row["recovery_variant"] == "incorrect_direct"]
    raw = [row for row in recoveries if row["recovery_variant"] == "recovered_raw"]
    h_group_summary = {
        group: {
            "mean_h_across_mesh_rows": float(np.mean([row["mean_h"] for row in h_rows if row["group"] == group])),
            "median_h_across_mesh_rows": float(np.mean([row["median_h"] for row in h_rows if row["group"] == group])),
            "minimum_h_across_mesh_rows": float(np.min([row["minimum_h"] for row in h_rows if row["group"] == group])),
            "maximum_h_across_mesh_rows": float(np.max([row["maximum_h"] for row in h_rows if row["group"] == group])),
        }
        for group in sorted({row["group"] for row in h_rows})
        if all(row["mean_h"] is not None for row in h_rows if row["group"] == group)
    }
    return {
        "experiment": "sofa50_h2_normalization_audit",
        "checkpoint": source_config["checkpoint"],
        "MODEL_OUTPUT_REPRESENTATION": "H2_NORMALIZED_LAPLACIAN",
        "training_target": training_trace,
        "h_definition": "arithmetic mean of unique undirected one-ring incident edge lengths",
        "epsilon": epsilon,
        "training_formula": "delta_hat = delta_raw / (h^2 + epsilon)",
        "inference_formula_before_audit": "delta_raw_pred = delta_hat_pred * h_current^2",
        "inference_formula_after_minimal_fix": "delta_raw_pred = delta_hat_pred * (h_current^2 + epsilon)",
        "inference_was_using_normalized_directly": False,
        "inference_h_source": "per-vertex h recomputed/stored on each current control or perturbed expanded graph",
        "h_group_statistics": h_group_summary,
        "roundtrip": {
            "max_absolute_error": max(row["max_absolute_error"] for row in roundtrip),
            "max_relative_l2_error": max(row["relative_l2_error"] for row in roundtrip),
            "min_cosine": min(row["cosine_similarity"] for row in roundtrip),
            "passed": max(row["relative_l2_error"] for row in roundtrip) <= 1e-12,
        },
        "prediction_scale": {
            "mean_delta_hat_norm": float(np.mean([row["delta_hat_pred_l2_norm"] for row in predictions])),
            "mean_delta_raw_norm": float(np.mean([row["delta_raw_pred_l2_norm"] for row in predictions])),
            "mean_epsilon_omission_relative_error": float(np.mean([row["epsilon_omission_relative_l2_error"] for row in predictions])),
            "mean_cached_float_raw_vs_corrected_relative_error": float(np.mean([row["cached_float_raw_vs_corrected_relative_l2_error"] for row in predictions])),
        },
        "direct_vs_recovered_raw": {
            "incorrect_direct_mean_chamfer": float(np.mean([row["chamfer_to_gt"] for row in direct])),
            "recovered_raw_mean_chamfer": float(np.mean([row["chamfer_to_gt"] for row in raw])),
            "incorrect_direct_mean_displacement": float(np.mean([row["mean_displacement_from_initial"] for row in direct])),
            "recovered_raw_mean_displacement": float(np.mean([row["mean_displacement_from_initial"] for row in raw])),
            "incorrect_direct_total_flips": int(sum(row["introduced_flipped_triangles"] for row in direct)),
            "recovered_raw_total_flips": int(sum(row["introduced_flipped_triangles"] for row in raw)),
        },
        "prediction_vs_oracle_normalized": _mean_comparison(all_hat_absolute),
        "prediction_vs_oracle_raw": _mean_comparison(all_raw_absolute),
        "prediction_vs_oracle_normalized_residual": _mean_comparison(all_hat_residual),
        "prediction_vs_oracle_raw_residual": _mean_comparison(all_raw_residual),
        "representation_conversion_consistent": bool(
            abs(np.mean([row["mean_per_vertex_cosine"] for row in all_hat_absolute]) - np.mean([row["mean_per_vertex_cosine"] for row in all_raw_absolute])) <= 1e-8
            and abs(np.mean([row["mean_per_vertex_cosine"] for row in all_hat_residual]) - np.mean([row["mean_per_vertex_cosine"] for row in all_raw_residual])) <= 1e-8
        ),
        "correct_h2_recovery_reduces_direct_use_failure": bool(
            np.mean([row["chamfer_to_gt"] for row in raw]) < np.mean([row["chamfer_to_gt"] for row in direct])
        ),
        "spike_h2": {
            "median_h": float(np.median([row["h"] for row in spikes])),
            "median_pred_hat_over_pred_raw": float(np.median([row["pred_hat_over_pred_raw"] for row in spikes])),
            "mean_incorrect_direct_displacement": float(np.mean([row["incorrect_direct_displacement"] for row in spikes])),
            "mean_recovered_raw_displacement": float(np.mean([row["recovered_raw_displacement"] for row in spikes])),
            "midpoint_fraction": float(np.mean([row["inserted_midpoint"] for row in spikes])),
            "sharp_fraction": float(np.mean([row["sharp_high_curvature"] for row in spikes])),
        },
        "previous_normalized_residual_metrics_revalidated": True,
        "expanded_graph_incompatibility_conclusion": (
            "partially_supported: normalized absolute alignment is moderate, but the exactly "
            "converted raw target has weak global alignment in solver-relevant units and still "
            "produces severe recovery failures"
        ),
        "long_training_blocked": True,
        "smallest_next_experiment": "On one validation pair, compare the frozen normalized absolute prediction to a current-expanded-graph absolute oracle after conditioning both on identical visibility; do not retrain.",
        "artifact_counts": {
            "roundtrip_rows": len(roundtrip),
            "h_rows": len(h_rows),
            "prediction_rows": len(predictions),
            "recovery_rows": len(recoveries),
            "spike_rows": len(spikes),
        },
    }


def _mean_comparison(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return {
        field: float(np.mean([row[field] for row in rows]))
        for field in (
            "global_cosine",
            "mean_per_vertex_cosine",
            "median_per_vertex_cosine",
            "prediction_oracle_norm_ratio",
            "alpha_star",
        )
    }


def _report(summary: Mapping[str, Any]) -> str:
    normalized = summary["prediction_vs_oracle_normalized"]
    raw = summary["prediction_vs_oracle_raw"]
    normalized_residual = summary["prediction_vs_oracle_normalized_residual"]
    raw_residual = summary["prediction_vs_oracle_raw_residual"]
    recovery = summary["direct_vs_recovered_raw"]
    return f"""# Sofa50 h² normalization audit

## Outcome

`MODEL_OUTPUT_REPRESENTATION = H2_NORMALIZED_LAPLACIAN`

The intended h² representation is present in training and was already undone at expanded inference using each current expanded graph's per-vertex `h`. The network output was **not** being passed directly to recovery. The one confirmed inconsistency was the missing `+ epsilon` in the inverse transform; it is now corrected. Its measured effect is negligible, so it does not explain the spikes.

## Required answers

1. **Training target tensor:** `prepared.training_target = normalized_laplacian_target = (L_gt V_gt)/(h_gt² + 1e-12)` on valid GT-query vertices.
2. **Frozen model output:** h²-normalized absolute uniform Laplacian coordinates.
3. **Is h² normalization present in training?** Yes.
4. **Definition of h:** arithmetic mean of lengths of unique undirected one-ring incident edges; it is not RMS.
5. **Epsilon:** `{summary['epsilon']}`, added to `h²` in the normalization denominator.
6. **Does inference undo normalization?** Yes.
7. **Where and which h?** Immediately after `model(sample).predicted_laplacian`, using the current inference sample's expanded-graph `local_edge_length`.
8. **Is h recomputed on the current expanded graph?** Yes. Stored and independently recomputed values agree within the errors reported in `normalization_roundtrip.csv`; midpoint vertices have their own graph-derived values.
9. **Was normalized output previously used directly as raw?** No. The old path multiplied by `h²`; only the inverse epsilon term was omitted.
10. **Round trip:** {'PASS' if summary['roundtrip']['passed'] else 'FAIL'}; maximum relative L2 error `{summary['roundtrip']['max_relative_l2_error']:.6g}`, maximum absolute error `{summary['roundtrip']['max_absolute_error']:.6g}`.
11. **Prediction magnitudes:** mean global L2 norm changes from `{summary['prediction_scale']['mean_delta_hat_norm']:.6g}` normalized to `{summary['prediction_scale']['mean_delta_raw_norm']:.6g}` raw. Full per-mesh magnitude and `h²+epsilon` distributions are in `prediction_scale_statistics.csv`.
12. **Does correct recovery reduce direct-use failure?** Yes: direct normalized-as-raw Chamfer is `{recovery['incorrect_direct_mean_chamfer']:.6g}` versus `{recovery['recovered_raw_mean_chamfer']:.6g}` after denormalization; mean displacement is `{recovery['incorrect_direct_mean_displacement']:.6g}` versus `{recovery['recovered_raw_mean_displacement']:.6g}`. This confirms h² recovery is essential, but it was already present before the audit. Within previously identified spike vertices, direct/raw mean displacements are `{summary['spike_h2']['mean_incorrect_direct_displacement']:.6g}/{summary['spike_h2']['mean_recovered_raw_displacement']:.6g}`, median `|pred_hat|/|pred_raw|` is `{summary['spike_h2']['median_pred_hat_over_pred_raw']:.6g}`, and `{summary['spike_h2']['midpoint_fraction']:.1%}` are inserted midpoints; full local-edge fields are in `spike_h2_diagnostics.csv`.
13. **Normalized oracle comparison:** absolute global cosine `{normalized['global_cosine']:.6g}`, mean/median vertex cosine `{normalized['mean_per_vertex_cosine']:.6g}/{normalized['median_per_vertex_cosine']:.6g}`, norm ratio `{normalized['prediction_oracle_norm_ratio']:.6g}`, alpha* `{normalized['alpha_star']:.6g}`. Residual global cosine/norm ratio/alpha* are `{normalized_residual['global_cosine']:.6g}/{normalized_residual['prediction_oracle_norm_ratio']:.6g}/{normalized_residual['alpha_star']:.6g}`.
14. **Raw oracle comparison:** absolute global cosine `{raw['global_cosine']:.6g}`, mean/median vertex cosine `{raw['mean_per_vertex_cosine']:.6g}/{raw['median_per_vertex_cosine']:.6g}`, norm ratio `{raw['prediction_oracle_norm_ratio']:.6g}`, alpha* `{raw['alpha_star']:.6g}`. Residual global cosine/norm ratio/alpha* are `{raw_residual['global_cosine']:.6g}/{raw_residual['prediction_oracle_norm_ratio']:.6g}/{raw_residual['alpha_star']:.6g}`.
15. **Are the previous cosine/norm conclusions valid?** Yes for the previously reported normalized residual convention: the matched audit reproduces the near-zero cosine and oversized residual norm. They must not be substituted for the absolute-target metrics above. Raw residual metrics differ because nonuniform `h²` changes global weighting. Conversion consistency is `{summary['representation_conversion_consistent']}`.
16. **Is expanded-graph incompatibility still supported?** {summary['expanded_graph_incompatibility_conclusion']}. This is not a representation-conversion failure: normalized per-vertex directions are moderately aligned, but nonuniform `h²` weighting makes the raw global target poorly aligned in the units consumed by recovery.
17. **Should long training remain blocked?** Yes.
18. **Smallest next experiment:** {summary['smallest_next_experiment']}

The fixed-camera panels compare initial, deliberately incorrect direct use, correctly recovered raw prediction, and GT without changing checkpoint, scale, visibility, solver, topology, or cameras.

## Tests

The representation/recovery regression subset reports **{summary.get('tests', {}).get('passed', 'not recorded')} passed, {summary.get('tests', {}).get('failed', 'not recorded')} failed, {summary.get('tests', {}).get('skipped', 'not recorded')} skipped**. The exact command is recorded in `test_results.json`.
"""


def _trace_stage(name: str, values: np.ndarray, representation: str) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "name": name,
        "shape": list(array.shape),
        "representation_or_units": representation,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "mean_absolute_magnitude": float(np.mean(np.abs(array))),
        "l2_norm": float(np.linalg.norm(array)),
    }


def _vector_stats(values: np.ndarray) -> dict[str, float]:
    magnitude = np.linalg.norm(values, axis=1)
    return {
        "mean_magnitude": float(magnitude.mean()),
        "median_magnitude": float(np.median(magnitude)),
        "maximum_magnitude": float(magnitude.max()),
        "l2_norm": float(np.linalg.norm(values)),
    }


def _prefixed_vector_stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in _vector_stats(values).items()}


def _prefixed_scalar_stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_minimum": float(values.min()),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_maximum": float(values.max()),
    }


def _h_statistics(values: np.ndarray) -> dict[str, float | int | None]:
    if len(values) == 0:
        return {"vertex_count": 0, "minimum_h": None, "maximum_h": None, "mean_h": None, "median_h": None, "minimum_h2": None, "maximum_h2": None, "mean_h2": None}
    return {
        "vertex_count": len(values),
        "minimum_h": float(values.min()),
        "maximum_h": float(values.max()),
        "mean_h": float(values.mean()),
        "median_h": float(np.median(values)),
        "minimum_h2": float(np.min(values**2)),
        "maximum_h2": float(np.max(values**2)),
        "mean_h2": float(np.mean(values**2)),
    }


def _refinement_config(mapping: Mapping[str, Any]):
    from mlr.refinement import RefinementConfig

    return RefinementConfig(
        operator_type=str(mapping.get("operator_type", "uniform")),
        lambda_lap=float(mapping.get("lambda_lap", 1.0)),
        lambda_anchor=float(mapping.get("lambda_anchor", 0.01)),
        lambda_edge=float(mapping.get("lambda_edge", 0.0)),
        num_iters=int(mapping.get("num_iters", 200)),
        learning_rate=float(mapping.get("learning_rate", 0.01)),
        robust_loss=str(mapping.get("robust_loss", "huber")),
        huber_delta=float(mapping.get("huber_delta", 0.01)),
    )


def _copy_configs(output: Path, source_config: Mapping[str, Any]) -> None:
    destination = output / "configs"
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / "source_run_config.json", source_config)
    shutil.copyfile(source_config["model_config"], destination / "model_config.json")
    shutil.copyfile(source_config["recovery_config"], destination / "recovery_config.json")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: _coerce_csv(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _coerce_csv(value: str | None) -> Any:
    if value is None or value == "":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() and not any(char in value.lower() for char in (".", "e")) else number


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _np(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
