#!/usr/bin/env python3
from __future__ import annotations

"""Read-only unified-v2 geometry evaluation for one continuous B+E checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from diagnose_sofa50_exact_target_oracle import METRIC_PROTOCOL, _clean_mesh, _geometry_row
from mlr.data import Mesh
from mlr.learned_laplacian.canonical_experiment import _exact_query_sample, _load_device_item
from mlr.learned_laplacian.diagnostics import _amp_settings
from mlr.learned_laplacian.differentiable_sparse_recovery import (
    differentiable_regularized_sparse_recovery_with_audit,
)
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.multi_trainer import _build_model
from mlr.learned_laplacian.trainer import load_checkpoint
from mlr.learned_laplacian.two_branch_hybrid import TwoBranchPretrainedHybridModel


CURVATURE_PROTOCOL = (
    "same-index cotangent discrete twice-mean-curvature vector "
    "2Hn=(2A_bary)^-1 sum_j(cot_alpha+cot_beta)(v_i-v_j); compare magnitudes; "
    "eligible vertices require positive refined/clean barycentric area; "
    "dihedral and face-normal angles use acos(abs(dot)) to ignore winding sign; "
    "edge-length and triangle-area distortion use absolute log ratios"
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(values: np.ndarray, prefix: str) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": float("nan"),
            f"{prefix}_rms": float("nan"),
            f"{prefix}_p95": float("nan"),
        }
    return {
        f"{prefix}_count": int(len(finite)),
        f"{prefix}_mean": float(finite.mean()),
        f"{prefix}_rms": float(np.sqrt(np.square(finite).mean())),
        f"{prefix}_p95": float(np.quantile(finite, 0.95)),
    }


def _face_geometry(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangles = vertices[faces]
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    double_area = np.linalg.norm(cross, axis=1)
    normals = np.zeros_like(cross)
    valid = double_area > 1e-14
    normals[valid] = cross[valid] / double_area[valid, None]
    return double_area * 0.5, normals, valid


def _cotangent_twice_mean_curvature(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    triangles = vertices[faces]
    edges_i = triangles[:, 1] - triangles[:, 0]
    edges_j = triangles[:, 2] - triangles[:, 1]
    edges_k = triangles[:, 0] - triangles[:, 2]
    double_area = np.linalg.norm(np.cross(edges_i, -edges_k), axis=1)
    valid_faces = double_area > 1e-14
    denominator = np.where(valid_faces, double_area, 1.0)
    cot_i = np.einsum("ij,ij->i", edges_i, -edges_k) / denominator
    cot_j = np.einsum("ij,ij->i", edges_j, -edges_i) / denominator
    cot_k = np.einsum("ij,ij->i", edges_k, -edges_j) / denominator
    cot_i[~valid_faces] = 0.0
    cot_j[~valid_faces] = 0.0
    cot_k[~valid_faces] = 0.0

    edge_pairs = np.concatenate(
        (faces[:, [1, 2]], faces[:, [2, 0]], faces[:, [0, 1]]), axis=0
    )
    weights = np.concatenate((cot_i, cot_j, cot_k), axis=0)
    first, second = edge_pairs[:, 0], edge_pairs[:, 1]
    result = np.zeros_like(vertices, dtype=np.float64)
    contribution = weights[:, None] * (vertices[first] - vertices[second])
    np.add.at(result, first, contribution)
    np.add.at(result, second, -contribution)

    barycentric_area = np.zeros(len(vertices), dtype=np.float64)
    face_area = 0.5 * double_area
    for column in range(3):
        np.add.at(barycentric_area, faces[:, column], face_area / 3.0)
    eligible = barycentric_area > 1e-14
    result[eligible] /= 2.0 * barycentric_area[eligible, None]
    result[~eligible] = np.nan
    return result, eligible


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    edges = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
    )
    return np.unique(np.sort(edges, axis=1), axis=0)


def _mean_incident_edge_length(
    vertices: np.ndarray, edges: np.ndarray
) -> np.ndarray:
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    total = np.zeros(len(vertices), dtype=np.float64)
    count = np.zeros(len(vertices), dtype=np.int64)
    for column in range(2):
        np.add.at(total, edges[:, column], lengths)
        np.add.at(count, edges[:, column], 1)
    output = np.zeros(len(vertices), dtype=np.float64)
    valid = count > 0
    output[valid] = total[valid] / count[valid]
    return output


def _interior_face_pairs(faces: np.ndarray) -> np.ndarray:
    edges = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
    )
    edges = np.sort(edges, axis=1)
    face_ids = np.tile(np.arange(len(faces), dtype=np.int64), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    sorted_edges = edges[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.any(np.diff(sorted_edges, axis=0), axis=1))]
    stops = np.r_[starts[1:], len(order)]
    pair_groups = np.flatnonzero(stops - starts == 2)
    return np.stack(
        (
            face_ids[order[starts[pair_groups]]],
            face_ids[order[starts[pair_groups] + 1]],
        ),
        axis=1,
    )


def _curvature_quality(
    refined: np.ndarray, clean: np.ndarray, faces: np.ndarray
) -> dict[str, float | int]:
    refined_area, refined_normals, refined_normal_valid = _face_geometry(refined, faces)
    clean_area, clean_normals, clean_normal_valid = _face_geometry(clean, faces)
    refined_curvature, refined_curvature_valid = _cotangent_twice_mean_curvature(
        refined, faces
    )
    clean_curvature, clean_curvature_valid = _cotangent_twice_mean_curvature(clean, faces)
    curvature_valid = refined_curvature_valid & clean_curvature_valid
    curvature_error = np.abs(
        np.linalg.norm(refined_curvature[curvature_valid], axis=1)
        - np.linalg.norm(clean_curvature[curvature_valid], axis=1)
    )
    edges = _unique_edges(faces)
    clean_h = _mean_incident_edge_length(clean, edges)[curvature_valid]
    output = _summary(curvature_error, "twice_mean_curvature_magnitude_error")
    output.update(_summary(curvature_error * clean_h, "scaled_curvature_error"))

    normal_valid = refined_normal_valid & clean_normal_valid
    normal_cosine = np.abs(
        np.einsum(
            "ij,ij->i",
            refined_normals[normal_valid],
            clean_normals[normal_valid],
        )
    )
    normal_angle = np.degrees(np.arccos(np.clip(normal_cosine, 0.0, 1.0)))
    output.update(_summary(normal_angle, "face_normal_angle_error_degrees"))

    dihedral_pairs = _interior_face_pairs(faces)
    dihedral_valid = (
        refined_normal_valid[dihedral_pairs[:, 0]]
        & refined_normal_valid[dihedral_pairs[:, 1]]
        & clean_normal_valid[dihedral_pairs[:, 0]]
        & clean_normal_valid[dihedral_pairs[:, 1]]
    )
    dihedral_pairs = dihedral_pairs[dihedral_valid]
    refined_dihedral = np.arccos(
        np.clip(
            np.abs(
                np.einsum(
                    "ij,ij->i",
                    refined_normals[dihedral_pairs[:, 0]],
                    refined_normals[dihedral_pairs[:, 1]],
                )
            ),
            0.0,
            1.0,
        )
    )
    clean_dihedral = np.arccos(
        np.clip(
            np.abs(
                np.einsum(
                    "ij,ij->i",
                    clean_normals[dihedral_pairs[:, 0]],
                    clean_normals[dihedral_pairs[:, 1]],
                )
            ),
            0.0,
            1.0,
        )
    )
    output.update(
        _summary(
            np.degrees(np.abs(refined_dihedral - clean_dihedral)),
            "dihedral_angle_error_degrees",
        )
    )

    refined_lengths = np.linalg.norm(
        refined[edges[:, 0]] - refined[edges[:, 1]], axis=1
    )
    clean_lengths = np.linalg.norm(clean[edges[:, 0]] - clean[edges[:, 1]], axis=1)
    length_valid = (refined_lengths > 1e-14) & (clean_lengths > 1e-14)
    output.update(
        _summary(
            np.abs(np.log(refined_lengths[length_valid] / clean_lengths[length_valid])),
            "absolute_log_edge_length_ratio",
        )
    )
    area_valid = (refined_area > 1e-14) & (clean_area > 1e-14)
    output.update(
        _summary(
            np.abs(np.log(refined_area[area_valid] / clean_area[area_valid])),
            "absolute_log_face_area_ratio",
        )
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_payload = _read(args.run.resolve() / "run_config.json")
    config = run_payload.get("experiment_config", run_payload)
    settings = config["training"]["hybrid_single_geometry_loss"]
    device = torch.device(args.device)
    model = _build_model(config, None, False).to(device)
    if not isinstance(model, TwoBranchPretrainedHybridModel):
        raise RuntimeError("Run config did not instantiate two complete B/E networks")
    checkpoint_payload = load_checkpoint(
        args.checkpoint.resolve(), model, map_location=device
    )
    model.eval()
    dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), args.split)
    amp_enabled, amp_dtype = _amp_settings(config, device)

    rows: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        static = dataset.load_static(index)
        prepared = _load_device_item(dataset, index, config, device)
        conditioned = _exact_query_sample(prepared.sample, device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            prediction = model(conditioned)
        direct = prediction.direct_vertex_displacement_prediction
        if direct is None:
            raise RuntimeError("Continuous B/E checkpoint omitted the direct branch")
        recovered, audit = differentiable_regularized_sparse_recovery_with_audit(
            prediction.predicted_laplacian.detach().double(),
            prepared.sample["vertices"].double() + direct.detach().double(),
            prepared.sample["edge_index"],
            prepared.sample["vertex_degree"].double(),
            regularization=float(settings["lambda"]),
            maximum_iterations=int(settings["maximum_iterations"]),
            tolerance=float(settings["tolerance"]),
        )
        if not audit.converged:
            raise RuntimeError(f"{static['sample_id']}: PCG did not converge")
        vertices = np.asarray(static["vertices"], dtype=np.float64)
        faces = np.asarray(static["faces"], dtype=np.int64)
        clean = _clean_mesh(static)
        initial_mesh = Mesh(vertices, faces).ensure_normals()
        recovered_mesh = Mesh(
            recovered.detach().cpu().numpy(), faces.copy()
        ).ensure_normals()
        initial_metric = _geometry_row(
            args.split,
            str(static["sample_id"]),
            "initial",
            initial_mesh,
            clean,
            initial_mesh,
        )
        metric = _geometry_row(
            args.split,
            str(static["sample_id"]),
            args.label,
            recovered_mesh,
            clean,
            initial_mesh,
        )
        clean_vertices = np.asarray(clean.vertices, dtype=np.float64)
        initial_chamfer = float(initial_metric["chamfer"])
        rows.append(
            {
                **metric,
                **_curvature_quality(
                    recovered_mesh.vertices, clean_vertices, faces
                ),
                "initial_chamfer": initial_chamfer,
                "relative_gain": (initial_chamfer - float(metric["chamfer"]))
                / initial_chamfer,
                "same_index_recovered_vertex_rms": float(
                    np.sqrt(
                        np.mean(
                            np.sum(
                                (recovered_mesh.vertices - clean_vertices) ** 2,
                                axis=1,
                            )
                        )
                    )
                ),
                "sample_index": index,
                "pcg_iterations": int(audit.iterations),
                "pcg_relative_residual": float(audit.relative_residual),
            }
        )
        print(
            f"continuous {args.split} {args.label} {index + 1}/{len(dataset)}",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def mean(field: str) -> float:
        return float(np.mean([float(row[field]) for row in rows]))

    payload = {
        "read_only": True,
        "selection_eligible": args.split == "validation",
        "split": args.split,
        "label": args.label,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint_payload.get("epoch"),
        "samples": len(rows),
        "geometry": {
            "initial_chamfer": mean("initial_chamfer"),
            "refined_chamfer": mean("chamfer"),
            "relative_gain": mean("relative_gain"),
            "p2s_mean": mean("p2s"),
            "p2s_p95": mean("p2s_p95"),
            "fscore": mean("fscore"),
            "normal_consistency": mean("normal_consistency"),
            "introduced_flips": int(sum(int(row["introduced_flipped_faces"]) for row in rows)),
            "new_degenerates": int(sum(int(row["new_degenerate_faces"]) for row in rows)),
            "vertex_rms": mean("same_index_recovered_vertex_rms"),
            "improved_worsened": [
                sum(float(row["chamfer"]) < float(row["initial_chamfer"]) for row in rows),
                sum(float(row["chamfer"]) >= float(row["initial_chamfer"]) for row in rows),
            ],
        },
        "curvature_and_distortion": {
            field: mean(field)
            for field in (
                "twice_mean_curvature_magnitude_error_mean",
                "twice_mean_curvature_magnitude_error_rms",
                "twice_mean_curvature_magnitude_error_p95",
                "scaled_curvature_error_mean",
                "scaled_curvature_error_rms",
                "scaled_curvature_error_p95",
                "face_normal_angle_error_degrees_mean",
                "face_normal_angle_error_degrees_p95",
                "dihedral_angle_error_degrees_mean",
                "dihedral_angle_error_degrees_p95",
                "absolute_log_edge_length_ratio_mean",
                "absolute_log_edge_length_ratio_p95",
                "absolute_log_face_area_ratio_mean",
                "absolute_log_face_area_ratio_p95",
            )
        },
        "solver": {
            "lambda": float(settings["lambda"]),
            "tolerance": float(settings["tolerance"]),
            "maximum_iterations": int(settings["maximum_iterations"]),
            "iterations_mean": float(np.mean([row["pcg_iterations"] for row in rows])),
            "iterations_max": int(max(row["pcg_iterations"] for row in rows)),
            "relative_residual_max": float(max(row["pcg_relative_residual"] for row in rows)),
            "failed": 0,
        },
        "metric_protocol": METRIC_PROTOCOL,
        "curvature_protocol": CURVATURE_PROTOCOL,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "checkpoint_sha256": payload["checkpoint_sha256"],
                "geometry": payload["geometry"],
                "solver": payload["solver"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
