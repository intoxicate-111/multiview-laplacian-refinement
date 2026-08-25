#!/usr/bin/env python3
from __future__ import annotations

"""Frozen matched/OOD evaluation for Uniform and Cotangent single-loss hybrids."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from diagnose_sofa50_exact_target_oracle import _clean_mesh, _geometry_row
from mlr.data import Mesh
from mlr.learned_laplacian.canonical_experiment import (
    _exact_query_sample,
    _load_device_item,
)
from mlr.learned_laplacian.cotangent_sparse_recovery import (
    build_symmetric_cotangent_stiffness,
    differentiable_cotangent_sparse_recovery_with_audit,
)
from mlr.learned_laplacian.diagnostics import _amp_settings
from mlr.learned_laplacian.differentiable_sparse_recovery import (
    differentiable_regularized_sparse_recovery_with_audit,
)
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.multi_trainer import _build_model
from mlr.learned_laplacian.trainer import load_checkpoint


RECIPES = ("A1", "A2", "B1", "B2", "C1", "C2", "C3", "C4", "D1", "D2")


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


def _checkpoint(run: Path) -> Path:
    for name in ("checkpoint_best.pt", "best.pt"):
        path = run / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No validation-selected checkpoint in {run}.")


def _recipe(sample_id: str) -> str:
    value = sample_id.rsplit("__", 1)[-1]
    return value if value in RECIPES else "unknown"


def _splits(manifest: Path, requested: list[str]) -> list[str]:
    available = []
    for split in requested:
        try:
            dataset = PreparedMeshDataset.from_manifest(manifest, split)
        except (KeyError, ValueError):
            continue
        if len(dataset):
            available.append(split)
    if not available:
        raise RuntimeError(f"No requested splits {requested} exist in {manifest}.")
    return available


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("uniform", "cotangent"))
    parser.add_argument("--domain", required=True)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run = args.run.resolve()
    config_payload = _read(run / "run_config.json")
    config = config_payload.get("experiment_config", config_payload)
    settings = config["training"]["hybrid_single_geometry_loss"]
    expected_operator = (
        "uniform_random_walk"
        if args.arm == "uniform"
        else "symmetric_cotangent_stiffness"
    )
    actual_operator = str(settings.get("operator", "uniform_random_walk"))
    if actual_operator != expected_operator:
        raise RuntimeError(f"Arm/operator mismatch: {args.arm} vs {actual_operator}.")
    regularization = float(settings["lambda"])
    tolerance = float(settings["tolerance"])
    maximum_iterations = int(settings["maximum_iterations"])
    cotangent_epsilon = float(settings.get("cotangent_relative_area_epsilon", 1e-12))
    if config["training"]["recovery_aware_geometry_loss"]["enabled"]:
        raise RuntimeError("Auxiliary recovery-aware loss is unexpectedly enabled.")

    device = torch.device(args.device)
    checkpoint = _checkpoint(run)
    model = _build_model(config, None, False).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    model.eval()
    amp_enabled, amp_dtype = _amp_settings(config, device)
    if not model.hybrid_direct_head_enabled:
        raise RuntimeError("The frozen model lacks the required direct latent head.")

    rows: list[dict[str, Any]] = []
    manifest = args.manifest.resolve()
    for split in _splits(manifest, args.splits):
        dataset = PreparedMeshDataset.from_manifest(manifest, split)
        for index in range(len(dataset)):
            static = dataset.load_static(index)
            prepared = _load_device_item(dataset, index, config, device)
            conditioned = _exact_query_sample(prepared.sample, device)
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                output = model(conditioned)
            direct_prediction = output.direct_vertex_displacement_prediction
            if direct_prediction is None:
                raise RuntimeError("Missing direct latent output.")
            delta = output.predicted_laplacian.detach().to(dtype=torch.float64)
            initial_vertices = prepared.sample["vertices"].to(dtype=torch.float64)
            direct_vertices = initial_vertices + direct_prediction.detach().to(
                dtype=torch.float64
            )
            if args.arm == "uniform":
                recovered, solve = differentiable_regularized_sparse_recovery_with_audit(
                    delta,
                    direct_vertices,
                    prepared.sample["edge_index"],
                    prepared.sample["vertex_degree"].to(dtype=torch.float64),
                    regularization=regularization,
                    maximum_iterations=maximum_iterations,
                    tolerance=tolerance,
                )
                construction = None
            else:
                cot_edges, cot_weights, cot_diagonal, construction = (
                    build_symmetric_cotangent_stiffness(
                        initial_vertices.detach().cpu(),
                        prepared.sample["faces"].detach().cpu(),
                        relative_area_epsilon=cotangent_epsilon,
                    )
                )
                recovered, solve = differentiable_cotangent_sparse_recovery_with_audit(
                    delta,
                    direct_vertices,
                    cot_edges,
                    cot_weights,
                    cot_diagonal,
                    regularization=regularization,
                    maximum_iterations=maximum_iterations,
                    tolerance=tolerance,
                )
            if not solve.converged:
                raise RuntimeError(f"{static['sample_id']}: PCG did not converge.")

            vertices_np = np.asarray(static["vertices"], dtype=np.float64)
            faces_np = np.asarray(static["faces"], dtype=np.int64)
            clean = _clean_mesh(static)
            initial = Mesh(vertices_np, faces_np).ensure_normals()
            refined_vertices = recovered.detach().cpu().numpy()
            refined = Mesh(refined_vertices, faces_np.copy()).ensure_normals()
            sample_id = str(static["sample_id"])
            initial_geometry = _geometry_row(
                args.domain, sample_id, "initial", initial, clean, initial
            )
            clean_geometry = _geometry_row(
                args.domain, sample_id, "clean", clean, clean, initial
            )
            geometry = _geometry_row(
                args.domain, sample_id, args.arm, refined, clean, initial
            )
            initial_cd = float(initial_geometry["chamfer"])
            clean_cd = float(clean_geometry["chamfer"])
            refined_cd = float(geometry["chamfer"])
            flips = int(geometry["introduced_flipped_faces"])
            rows.append(
                {
                    "arm": args.arm,
                    "domain": args.domain,
                    "split": split,
                    "sample_id": sample_id,
                    "recipe": _recipe(sample_id),
                    "lambda": regularization,
                    "initial_chamfer": initial_cd,
                    "refined_chamfer": refined_cd,
                    "relative_chamfer_gain": (initial_cd - refined_cd) / initial_cd,
                    "eta": (initial_cd - refined_cd) / (initial_cd - clean_cd),
                    "p2s": float(geometry["p2s"]),
                    "p2s_p95": float(geometry["p2s_p95"]),
                    "fscore": float(geometry["fscore"]),
                    "normal_consistency": float(geometry["normal_consistency"]),
                    "same_index_recovered_vertex_rms": float(
                        np.sqrt(
                            np.mean(
                                np.sum(
                                    np.square(refined_vertices - clean.vertices), axis=1
                                )
                            )
                        )
                    ),
                    "introduced_flipped_faces": flips,
                    "normalized_flip_rate": flips / max(len(faces_np), 1),
                    "new_degenerate_faces": int(geometry["new_degenerate_faces"]),
                    "improved": refined_cd < initial_cd,
                    "worsened": refined_cd > initial_cd,
                    "vertices": int(len(vertices_np)),
                    "faces": int(len(faces_np)),
                    "pcg_iterations": int(solve.iterations),
                    "pcg_relative_residual": float(solve.relative_residual),
                    "protected_triangles": (
                        None if construction is None else construction.protected_triangles
                    ),
                    "negative_cotangent_weights": (
                        None
                        if construction is None
                        else construction.negative_edge_weights
                    ),
                }
            )
            print(
                f"{args.arm} {args.domain} {split} {index + 1}/{len(dataset)} {sample_id}",
                flush=True,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "contract_audit": True,
        "arm": args.arm,
        "operator": actual_operator,
        "lambda": regularization,
        "domain": args.domain,
        "manifest": str(manifest),
        "run": str(run),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "rows": rows,
    }
    stem = f"{args.domain}_{args.arm}"
    (args.output_dir / f"{stem}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / f"{stem}.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
