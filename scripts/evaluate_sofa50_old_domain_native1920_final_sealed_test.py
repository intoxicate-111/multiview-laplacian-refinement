#!/usr/bin/env python3
from __future__ import annotations

"""Open the sealed old-domain test once and evaluate every frozen final method."""

import argparse
import csv
import glob
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from diagnose_sofa50_exact_target_oracle import METRIC_PROTOCOL, _clean_mesh, _geometry_row
from evaluate_sofa50_continuous_checkpoint_validation import CURVATURE_PROTOCOL, _curvature_quality
from evaluate_sofa50_old_domain_specialists import infer_e, load_e, pcg
from evaluate_sofa50_recovery_aware_ablation import _infer_recovery_arm, _load_spec
from mlr.data import Mesh
from mlr.io import load_mesh, save_mesh
from mlr.learned_laplacian.canonical_experiment import _exact_query_sample, _load_device_item
from mlr.learned_laplacian.diagnostics import _amp_settings
from mlr.learned_laplacian.differentiable_sparse_recovery import recovery_forward_audit
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.multi_trainer import _build_model
from mlr.learned_laplacian.trainer import load_checkpoint
from mlr.learned_laplacian.two_branch_hybrid import TwoBranchPretrainedHybridModel


EXPECTED_ARCHIVE = {
    "Initial mesh": {"chamfer": 0.0170704684},
    "NDS": {
        "chamfer": 0.0112049924,
        "p2s_p95": 0.0398475607,
        "fscore": 0.652827299,
        "normal_consistency": 0.873805125,
    },
    "Previous Ours (native-1920 HF)": {
        "chamfer": 0.0113478004,
        "p2s_p95": 0.0403952873,
        "fscore": 0.647196717,
        "normal_consistency": 0.944514414,
    },
    "nvdiffrec": {
        "chamfer": 0.0136546596,
        "p2s_p95": 0.0457457720,
        "fscore": 0.558673128,
        "normal_consistency": 0.848122276,
    },
    "ExMesh": {
        "chamfer": 0.0201706152,
        "p2s_p95": 0.0696287605,
        "fscore": 0.478513280,
        "normal_consistency": 0.845336664,
    },
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def uniform_laplacian(values: np.ndarray, edge_index: np.ndarray, degree: np.ndarray) -> np.ndarray:
    source, target = edge_index
    neighbor_sum = np.zeros_like(values, dtype=np.float64)
    np.add.at(neighbor_sum, source, values[target])
    return values - neighbor_sum / degree[:, None]


def uniform_laplacian_transpose(
    values: np.ndarray, edge_index: np.ndarray, degree: np.ndarray
) -> np.ndarray:
    source, target = edge_index
    contribution = np.zeros_like(values, dtype=np.float64)
    np.add.at(contribution, target, values[source] / degree[source, None])
    return values - contribution


def vector_diagnostic(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left_flat, right_flat = left.reshape(-1), right.reshape(-1)
    left_norm = float(np.linalg.norm(left_flat))
    right_norm = float(np.linalg.norm(right_flat))
    denominator = left_norm * right_norm
    return {
        "rms_difference": float(np.sqrt(np.mean((left - right) ** 2))),
        "relative_rms_difference": float(
            np.sqrt(np.mean((left - right) ** 2))
            / (0.5 * (np.sqrt(np.mean(left**2)) + np.sqrt(np.mean(right**2))) + 1e-12)
        ),
        "cosine": float(np.dot(left_flat, right_flat) / denominator) if denominator else 0.0,
        "norm_ratio": left_norm / (right_norm + 1e-12),
    }


def rhs_diagnostic(
    delta: np.ndarray,
    direct: np.ndarray,
    clean: np.ndarray,
    edge_index: np.ndarray,
    degree: np.ndarray,
    regularization: float,
) -> dict[str, float]:
    delta_star = uniform_laplacian(clean, edge_index, degree)
    e_l = uniform_laplacian_transpose(delta - delta_star, edge_index, degree)
    e_d = regularization * (direct - clean)
    e_q = e_l + e_d
    norm_l = float(np.linalg.norm(e_l))
    norm_d = float(np.linalg.norm(e_d))
    norm_q = float(np.linalg.norm(e_q))
    denominator = norm_l * norm_d
    return {
        "e_L_norm": norm_l,
        "e_D_norm": norm_d,
        "e_q_norm": norm_q,
        "e_L_e_D_cosine": float(np.vdot(e_l.reshape(-1), e_d.reshape(-1)) / denominator)
        if denominator
        else 0.0,
        "cancellation_ratio": norm_q / (norm_l + norm_d + 1e-12),
    }


def own_geometry_row(
    method: str,
    sample_id: str,
    refined: Mesh,
    initial: Mesh,
    clean: Mesh,
) -> dict[str, Any]:
    initial_metric = _geometry_row("test", sample_id, "initial", initial, clean, initial)
    metric = _geometry_row("test", sample_id, method, refined, clean, initial)
    initial_cd = float(initial_metric["chamfer"])
    refined_cd = float(metric["chamfer"])
    return {
        "method": method,
        "sample_id": sample_id,
        "faces": refined.num_faces,
        "initial_chamfer": initial_cd,
        "chamfer": refined_cd,
        "relative_gain": (initial_cd - refined_cd) / initial_cd,
        "p2s": float(metric["p2s"]),
        "p2s_p95": float(metric["p2s_p95"]),
        "fscore": float(metric["fscore"]),
        "normal_consistency": float(metric["normal_consistency"]),
        "introduced_flipped_faces": int(metric["introduced_flipped_faces"]),
        "new_degenerate_faces": int(metric["new_degenerate_faces"]),
        "flips_comparable": True,
        "same_index_recovered_vertex_rms": float(
            np.sqrt(np.mean(np.sum((refined.vertices - clean.vertices) ** 2, axis=1)))
        )
        if refined.num_vertices == clean.num_vertices
        else float("nan"),
        "improved": refined_cd < initial_cd,
        "worsened": refined_cd > initial_cd,
        **(
            _curvature_quality(refined.vertices, clean.vertices, refined.faces)
            if refined.num_vertices == clean.num_vertices
            and np.array_equal(refined.faces, clean.faces)
            else {}
        ),
    }


def aggregate(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    selected = [row for row in rows if row["method"] == method]
    if len(selected) != 25:
        raise RuntimeError(f"Expected 25 {method} rows, found {len(selected)}")
    initial = float(np.mean([float(row["initial_chamfer"]) for row in selected]))
    refined = float(np.mean([float(row["chamfer"]) for row in selected]))
    faces = sum(int(row.get("faces", 0)) for row in selected)
    flips_comparable = all(bool(row.get("flips_comparable", True)) for row in selected)
    result = {
        "method": method,
        "samples": len(selected),
        "initial_chamfer": initial,
        "chamfer": refined,
        "aggregate_relative_gain": (initial - refined) / initial,
        "p2s": float(np.mean([float(row["p2s"]) for row in selected])),
        "p2s_p95": float(np.mean([float(row["p2s_p95"]) for row in selected])),
        "fscore": float(np.mean([float(row["fscore"]) for row in selected])),
        "normal_consistency": float(
            np.mean([float(row["normal_consistency"]) for row in selected])
        ),
        "introduced_flipped_faces": int(
            sum(int(row.get("introduced_flipped_faces", 0)) for row in selected)
        ),
        "normalized_flip_rate": (
            sum(int(row.get("introduced_flipped_faces", 0)) for row in selected) / faces
            if faces and flips_comparable
            else None
        ),
        "introduced_flipped_faces_comparable": flips_comparable,
        "new_degenerate_faces": int(sum(int(row.get("new_degenerate_faces", 0)) for row in selected)),
        "improved": int(sum(bool(row["improved"]) for row in selected)),
        "worsened": int(sum(bool(row["worsened"]) for row in selected)),
    }
    optional = (
        "same_index_recovered_vertex_rms",
        "twice_mean_curvature_magnitude_error_mean",
        "scaled_curvature_error_mean",
        "dihedral_angle_error_degrees_mean",
        "face_normal_angle_error_degrees_mean",
        "absolute_log_edge_length_ratio_mean",
        "absolute_log_face_area_ratio_mean",
    )
    for field in optional:
        values = [float(row[field]) for row in selected if field in row and math.isfinite(float(row[field]))]
        result[field] = float(np.mean(values)) if values else None
    return result


def archived_rows(root: Path, method: str, label: str) -> list[dict[str, Any]]:
    paths = sorted(glob.glob(str(root / method / "shards" / "per_sample_shard_*.csv")))
    rows: list[dict[str, Any]] = []
    for path in paths:
        with Path(path).open(encoding="utf-8", newline="") as handle:
            for source in csv.DictReader(handle):
                if source.get("status") != "completed":
                    raise RuntimeError(f"Archived {label} row is not complete: {source}")
                rows.append(
                    {
                        "method": label,
                        "sample_id": source["sample_id"],
                        "faces": int(source["face_count"]),
                        "initial_chamfer": float(source["initial_chamfer"]),
                        "chamfer": float(source["refined_chamfer"]),
                        "p2s": float(source["refined_p2s_mean"]),
                        "p2s_p95": float(source["refined_p2s_p95"]),
                        "fscore": float(source["refined_fscore"]),
                        "normal_consistency": float(source["refined_normal_consistency"]),
                        "introduced_flipped_faces": int(source["introduced_flipped_faces"] or 0),
                        "new_degenerate_faces": int(source["new_degenerate_faces"] or 0),
                        "flips_comparable": source["introduced_flipped_faces_comparable"].lower()
                        == "true",
                        "improved": source["improved"].lower() == "true",
                        "worsened": source["improved"].lower() != "true",
                        "archived_final_mesh": source["final_mesh"],
                    }
                )
    if len(rows) != 25:
        raise RuntimeError(f"Expected 25 archived {label} rows, found {len(rows)}")
    return rows


def paired(continuous: dict[str, dict[str, Any]], comparison: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ids = sorted(continuous)
    if set(ids) != set(comparison):
        raise RuntimeError("Paired sample identities differ")
    result: dict[str, Any] = {}
    rng = np.random.default_rng(7)
    for field in ("chamfer", "p2s_p95", "fscore", "normal_consistency"):
        difference = np.asarray(
            [float(continuous[sample_id][field]) - float(comparison[sample_id][field]) for sample_id in ids],
            dtype=np.float64,
        )
        draws = rng.integers(0, len(difference), size=(10000, len(difference)))
        bootstrap = difference[draws].mean(axis=1)
        result[field] = {
            "continuous_minus_comparator_mean": float(np.mean(difference)),
            "continuous_minus_comparator_median": float(np.median(difference)),
            "bootstrap_95_percent_ci": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
        }
        if field == "chamfer":
            result[field].update(
                {
                    "continuous_wins": int(np.sum(difference < 0)),
                    "continuous_losses": int(np.sum(difference > 0)),
                    "ties": int(np.sum(difference == 0)),
                }
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--benchmark-manifest", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--arm-b-run", required=True, type=Path)
    parser.add_argument("--arm-e-run", required=True, type=Path)
    parser.add_argument("--continuous-run", required=True, type=Path)
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    authorization = read_json(args.authorization.resolve())
    if not (
        authorization.get("contract_audit") is True
        and authorization.get("final_selection_locked") is True
        and authorization.get("validation_only_selection") is True
        and authorization.get("authorize_single_test_open") is True
        and authorization.get("test_open_count_before_authorization") == 0
    ):
        raise RuntimeError("Final sealed-test authorization is invalid")
    output = args.output_dir.resolve()
    if output.exists():
        raise RuntimeError("Final test output exists; refusing to open the sealed test twice")
    if str(output) != authorization["test_output_directory"]:
        raise RuntimeError("Authorized final-test output directory differs")
    output.mkdir(parents=True)
    (output / "TEST_OPENED.json").write_text(
        json.dumps(
            {
                "opened_once": True,
                "authorization_sha256": sha256_file(args.authorization.resolve()),
                "all_selections_locked_before_open": True,
                "intermediate_test_trajectory": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    benchmark = read_json(args.benchmark_manifest.resolve())
    dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), "test")
    if len(dataset) != 25 or list(dataset.sample_ids) != list(benchmark["sample_ids"]):
        raise RuntimeError("Prepared test and exact benchmark identities differ")
    provenance = {row["sample_id"]: row for row in benchmark["samples"]}
    device = torch.device(args.device)
    b_spec = _load_spec(args.arm_b_run.resolve(), device)
    e_spec = load_e(args.arm_e_run.resolve(), device)
    if b_spec["checkpoint_sha256"] != authorization["arm_b_checkpoint_sha256"]:
        raise RuntimeError("Arm-B final checkpoint SHA changed after lock")
    if e_spec["checkpoint_sha256"] != authorization["arm_e_checkpoint_sha256"]:
        raise RuntimeError("Arm-E final checkpoint SHA changed after lock")

    continuous_payload = read_json(args.continuous_run.resolve() / "run_config.json")
    continuous_config = continuous_payload.get("experiment_config", continuous_payload)
    continuous_model = _build_model(continuous_config, None, False).to(device)
    if not isinstance(continuous_model, TwoBranchPretrainedHybridModel):
        raise RuntimeError("Continuous run is not an independent two-network model")
    continuous_checkpoint = Path(authorization["continuous_checkpoint"])
    if sha256_file(continuous_checkpoint) != authorization["continuous_checkpoint_sha256"]:
        raise RuntimeError("Continuous final checkpoint SHA changed after lock")
    load_checkpoint(continuous_checkpoint, continuous_model, map_location=device)
    continuous_model.eval()
    continuous_amp, continuous_dtype = _amp_settings(continuous_config, device)
    regularization = float(authorization["lambda_old"])

    rows: list[dict[str, Any]] = []
    representation_rows: list[dict[str, Any]] = []
    solver_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for index in range(len(dataset)):
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        source = provenance[sample_id]
        initial_path = Path(source["common_initial_mesh"])
        if sha256_file(initial_path) != source["common_initial_mesh_sha256"]:
            raise RuntimeError(f"{sample_id}: common initial SHA mismatch")
        vertices = np.asarray(static["vertices"], dtype=np.float64)
        faces = np.asarray(static["faces"], dtype=np.int64)
        initial = Mesh(vertices, faces).ensure_normals()
        initial_file = load_mesh(initial_path)
        if (
            not np.array_equal(initial_file.faces, faces)
            or np.max(np.abs(initial_file.vertices - vertices)) > 1e-6
        ):
            raise RuntimeError(f"{sample_id}: common initial identity mismatch")
        clean = _clean_mesh(static)
        edge_index = np.asarray(static["edge_index"], dtype=np.int64)
        degree = np.asarray(static["vertex_degree"], dtype=np.float64)

        b_values = _infer_recovery_arm(dataset, index, b_spec, device)
        delta_b = b_values["prediction_raw"].numpy().astype(np.float64)
        delta_v_e, _ = infer_e(dataset, index, e_spec, device)
        direct_e = vertices + delta_v_e
        b_vertices, b_audit = pcg(delta_b, vertices, static, 0.01, device)
        frozen_vertices, frozen_audit = pcg(delta_b, direct_e, static, regularization, device)

        prepared = _load_device_item(dataset, index, continuous_config, device)
        conditioned = _exact_query_sample(prepared.sample, device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=continuous_dtype, enabled=continuous_amp
        ):
            continuous_output = continuous_model(conditioned)
        continuous_direct_output = continuous_output.direct_vertex_displacement_prediction
        if continuous_direct_output is None:
            raise RuntimeError(f"{sample_id}: continuous direct output missing")
        continuous_delta = continuous_output.predicted_laplacian.float().detach().cpu().numpy().astype(np.float64)
        continuous_direct = vertices + continuous_direct_output.float().detach().cpu().numpy().astype(np.float64)
        with torch.no_grad():
            continuous_vertices_tensor, continuous_audit = recovery_forward_audit(
                continuous_output.predicted_laplacian.detach().double(),
                prepared.sample["vertices"].double() + continuous_direct_output.detach().double(),
                prepared.sample["edge_index"],
                prepared.sample["vertex_degree"].double(),
                regularization=regularization,
                maximum_iterations=2048,
                tolerance=1e-8,
            )
        continuous_vertices = continuous_vertices_tensor.cpu().numpy()

        method_meshes = {
            "Initial mesh": initial,
            "Old-domain Arm B": Mesh(b_vertices, faces.copy()).ensure_normals(),
            "Old-domain Arm E": Mesh(direct_e, faces.copy()).ensure_normals(),
            "Old-domain Frozen B+E": Mesh(frozen_vertices, faces.copy()).ensure_normals(),
            "Old-domain Continuous B+E": Mesh(continuous_vertices, faces.copy()).ensure_normals(),
        }
        previous_mesh = load_mesh(Path(source["canonical_ours_mesh"]))
        if not np.array_equal(previous_mesh.faces, faces):
            raise RuntimeError(f"{sample_id}: archived Previous Ours topology mismatch")
        method_meshes["Previous Ours (native-1920 HF)"] = previous_mesh.ensure_normals()
        for method, mesh in method_meshes.items():
            row = own_geometry_row(method, sample_id, mesh, initial, clean)
            sample_dir = output / "samples" / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            if method.startswith("Old-domain"):
                path = sample_dir / (method.lower().replace(" ", "_").replace("+", "plus") + ".obj")
                save_mesh(mesh, path)
                row["final_mesh"] = str(path)
            rows.append(row)

        representation_rows.extend(
            [
                {
                    "sample_id": sample_id,
                    "state": "Frozen",
                    **vector_diagnostic(delta_b, uniform_laplacian(direct_e, edge_index, degree)),
                    **rhs_diagnostic(delta_b, direct_e, clean.vertices, edge_index, degree, regularization),
                },
                {
                    "sample_id": sample_id,
                    "state": "Continuous",
                    **vector_diagnostic(
                        continuous_delta,
                        uniform_laplacian(continuous_direct, edge_index, degree),
                    ),
                    **rhs_diagnostic(
                        continuous_delta,
                        continuous_direct,
                        clean.vertices,
                        edge_index,
                        degree,
                        regularization,
                    ),
                },
            ]
        )
        for state, audit in (
            ("Arm B", b_audit),
            ("Frozen B+E", frozen_audit),
            ("Continuous B+E", continuous_audit),
        ):
            solver_rows.append(
                {
                    "sample_id": sample_id,
                    "state": state,
                    "iterations": int(audit.iterations if hasattr(audit, "iterations") else audit["pcg_iterations"]),
                    "converged": bool(audit.converged if hasattr(audit, "converged") else audit["pcg_converged"]),
                    "relative_residual": float(
                        audit.relative_residual
                        if hasattr(audit, "relative_residual")
                        else audit["pcg_relative_residual"]
                    ),
                }
            )
        print(f"sealed test {index + 1}/25 {sample_id}", flush=True)
        torch.cuda.empty_cache()

    archive_root = args.archive_root.resolve()
    rows.extend(archived_rows(archive_root, "nds", "NDS"))
    rows.extend(archived_rows(archive_root, "nvdiffrec", "nvdiffrec"))
    rows.extend(archived_rows(archive_root, "exmesh", "ExMesh"))
    method_order = (
        "Initial mesh",
        "NDS",
        "Previous Ours (native-1920 HF)",
        "nvdiffrec",
        "ExMesh",
        "Old-domain Arm B",
        "Old-domain Arm E",
        "Old-domain Frozen B+E",
        "Old-domain Continuous B+E",
    )
    aggregates = [aggregate(rows, method) for method in method_order]
    aggregate_by_method = {row["method"]: row for row in aggregates}
    archive_checks = {
        method: {
            field: abs(float(aggregate_by_method[method][field]) - expected)
            <= (1e-8 if field == "chamfer" else 1e-6)
            for field, expected in values.items()
        }
        for method, values in EXPECTED_ARCHIVE.items()
    }
    continuous_by_id = {
        row["sample_id"]: row for row in rows if row["method"] == "Old-domain Continuous B+E"
    }
    comparisons = {}
    for method in method_order[:-1]:
        comparison = {row["sample_id"]: row for row in rows if row["method"] == method}
        comparisons[method] = paired(continuous_by_id, comparison)

    representation_aggregate = []
    for state in ("Frozen", "Continuous"):
        selected = [row for row in representation_rows if row["state"] == state]
        representation_aggregate.append(
            {
                "state": state,
                **{
                    field: float(np.mean([row[field] for row in selected]))
                    for field in (
                        "rms_difference",
                        "relative_rms_difference",
                        "cosine",
                        "norm_ratio",
                        "e_L_norm",
                        "e_D_norm",
                        "e_q_norm",
                        "e_L_e_D_cosine",
                        "cancellation_ratio",
                    )
                },
            }
        )
    elapsed = time.perf_counter() - started
    peak_memory = (
        torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        if device.type == "cuda"
        else 0.0
    )
    solver_contract = all(row["converged"] for row in solver_rows) and max(
        row["relative_residual"] for row in solver_rows
    ) <= 1e-8
    contract = bool(
        all(all(fields.values()) for fields in archive_checks.values())
        and solver_contract
        and len(rows) == 25 * len(method_order)
        and all(
            math.isfinite(float(row[field]))
            for row in rows
            for field in ("chamfer", "p2s_p95", "fscore", "normal_consistency")
        )
    )
    payload = {
        "contract_audit": contract,
        "test_opened_once": True,
        "test_used_for_selection": False,
        "intermediate_test_trajectory": False,
        "samples": 25,
        "sample_ids": list(dataset.sample_ids),
        "checkpoint_identity": {
            "arm_b_sha256": authorization["arm_b_checkpoint_sha256"],
            "arm_e_sha256": authorization["arm_e_checkpoint_sha256"],
            "continuous_sha256": authorization["continuous_checkpoint_sha256"],
        },
        "lambda_old": regularization,
        "aggregate": aggregates,
        "archived_comparator_reproduction_checks": archive_checks,
        "paired_continuous_comparisons": comparisons,
        "representation_and_rhs_aggregate": representation_aggregate,
        "solver": {
            "all_converged": all(row["converged"] for row in solver_rows),
            "relative_residual_max": max(row["relative_residual"] for row in solver_rows),
            "iterations_mean": float(np.mean([row["iterations"] for row in solver_rows])),
            "iterations_max": int(max(row["iterations"] for row in solver_rows)),
            "rows": solver_rows,
        },
        "runtime": {"seconds": elapsed, "peak_gpu_memory_mb": peak_memory},
        "metric_protocol": METRIC_PROTOCOL,
        "curvature_protocol": CURVATURE_PROTOCOL,
        "rows": rows,
        "representation_and_rhs_rows": representation_rows,
    }
    (output / "final_test_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(output / "final_test_per_sample.csv", rows)
    write_csv(output / "representation_and_rhs_per_sample.csv", representation_rows)
    if not contract:
        raise RuntimeError("Final sealed-test contract failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
