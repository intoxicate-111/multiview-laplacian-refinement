#!/usr/bin/env python3
from __future__ import annotations

"""Measure the five-run step-0 execution nondeterminism envelope."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from diagnose_sofa50_exact_target_oracle import _clean_mesh, _geometry_row
from mlr.data import Mesh
from mlr.learned_laplacian.canonical_experiment import _exact_query_sample, _load_device_item
from mlr.learned_laplacian.diagnostics import _amp_settings
from mlr.learned_laplacian.differentiable_sparse_recovery import (
    differentiable_regularized_sparse_recovery_with_audit,
)
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.multi_trainer import _build_model
from mlr.learned_laplacian.two_branch_hybrid import TwoBranchPretrainedHybridModel


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--tight-reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--realizations", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.realizations < 5:
        raise ValueError("The revised contract requires at least five realizations.")

    config = _read(args.config.resolve())
    tight = _read(args.tight_reference.resolve())
    tight_rows = {str(row["sample_id"]): row for row in tight["rows"]}
    device = torch.device(args.device)
    model = _build_model(config, None, False).to(device)
    if not isinstance(model, TwoBranchPretrainedHybridModel):
        raise RuntimeError("Expected two complete pretrained specialist networks.")
    model.eval()
    amp_enabled, amp_dtype = _amp_settings(config, device)
    dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), "validation")

    realizations: list[dict[str, Any]] = []
    latent_b: list[np.ndarray] = []
    latent_e: list[np.ndarray] = []
    per_sample_cd: list[np.ndarray] = []
    sample_ids = list(dataset.sample_ids)
    total_faces = sum(int(np.asarray(dataset.load_static(i)["faces"]).shape[0]) for i in range(len(dataset)))

    for realization in range(args.realizations):
        b_values: list[np.ndarray] = []
        e_values: list[np.ndarray] = []
        rows: list[dict[str, Any]] = []
        for index in range(len(dataset)):
            static = dataset.load_static(index)
            sample_id = str(static["sample_id"])
            prepared = _load_device_item(dataset, index, config, device)
            conditioned = _exact_query_sample(prepared.sample, device)
            with torch.no_grad(), torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                output = model(conditioned)
            if output.direct_vertex_displacement_prediction is None:
                raise RuntimeError(f"{sample_id}: missing E direct output.")
            delta = output.predicted_laplacian.float().detach().double()
            displacement = output.direct_vertex_displacement_prediction.float().detach().double()
            b_values.append(delta.cpu().numpy())
            e_values.append(displacement.cpu().numpy())
            direct = prepared.sample["vertices"].double() + displacement
            recovered, audit = differentiable_regularized_sparse_recovery_with_audit(
                delta,
                direct,
                prepared.sample["edge_index"],
                prepared.sample["vertex_degree"].double(),
                regularization=3e-2,
                maximum_iterations=2048,
                tolerance=1e-8,
            )
            if not audit.converged:
                raise RuntimeError(f"{sample_id}: PCG failed in realization {realization}.")
            vertices = np.asarray(static["vertices"], dtype=np.float64)
            faces = np.asarray(static["faces"], dtype=np.int64)
            clean = _clean_mesh(static)
            geometry = _geometry_row(
                "validation",
                sample_id,
                f"step0_realization_{realization}",
                Mesh(recovered.detach().cpu().numpy(), faces.copy()).ensure_normals(),
                clean,
                Mesh(vertices, faces).ensure_normals(),
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "chamfer": float(geometry["chamfer"]),
                    "vertex_rms": float(np.sqrt(np.mean(np.sum(np.square(recovered.detach().cpu().numpy() - clean.vertices), axis=1)))),
                    "p2s_p95": float(geometry["p2s_p95"]),
                    "fscore": float(geometry["fscore"]),
                    "normal": float(geometry["normal_consistency"]),
                    "flips": int(geometry["introduced_flipped_faces"]),
                    "pcg_residual": float(audit.relative_residual),
                }
            )
        b_array, e_array = np.concatenate(b_values), np.concatenate(e_values)
        cd_array = np.asarray([row["chamfer"] for row in rows], dtype=np.float64)
        latent_b.append(b_array)
        latent_e.append(e_array)
        per_sample_cd.append(cd_array)
        realizations.append(
            {
                "realization": realization,
                "validation_cd": float(cd_array.mean()),
                "validation_vrms": float(np.mean([row["vertex_rms"] for row in rows])),
                "validation_p2s_p95": float(np.mean([row["p2s_p95"] for row in rows])),
                "validation_fscore": float(np.mean([row["fscore"] for row in rows])),
                "validation_normal": float(np.mean([row["normal"] for row in rows])),
                "validation_flips": int(sum(row["flips"] for row in rows)),
                "validation_flip_rate": float(sum(row["flips"] for row in rows) / total_faces),
                "maximum_pcg_residual": float(max(row["pcg_residual"] for row in rows)),
                "maximum_cd_difference_from_tight_reference": float(max(abs(row["chamfer"] - float(tight_rows[row["sample_id"]]["tight_chamfer"])) for row in rows)),
                "aggregate_cd_relative_difference_from_tight_reference": float(abs(cd_array.mean() - float(tight["tight_mean_chamfer"])) / float(tight["tight_mean_chamfer"])),
            }
        )
        print(f"realization {realization + 1}/{args.realizations} complete", flush=True)

    pairs: list[dict[str, Any]] = []
    for left in range(args.realizations):
        for right in range(left + 1, args.realizations):
            b_difference = latent_b[right] - latent_b[left]
            e_difference = latent_e[right] - latent_e[left]
            cd_difference = per_sample_cd[right] - per_sample_cd[left]
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "aggregate_cd_difference": float(per_sample_cd[right].mean() - per_sample_cd[left].mean()),
                    "maximum_per_sample_cd_difference": float(np.max(np.abs(cd_difference))),
                    "b_latent_rms_difference": float(np.sqrt(np.mean(np.square(b_difference)))),
                    "e_latent_rms_difference": float(np.sqrt(np.mean(np.square(e_difference)))),
                    "b_latent_maximum_difference": float(np.max(np.abs(b_difference))),
                    "e_latent_maximum_difference": float(np.max(np.abs(e_difference))),
                }
            )

    envelope = {
        "aggregate_cd_range": float(max(row["validation_cd"] for row in realizations) - min(row["validation_cd"] for row in realizations)),
        "maximum_absolute_pairwise_aggregate_cd_difference": float(max(abs(row["aggregate_cd_difference"]) for row in pairs)),
        "maximum_pairwise_per_sample_cd_difference": float(max(row["maximum_per_sample_cd_difference"] for row in pairs)),
        "maximum_pairwise_b_latent_rms_difference": float(max(row["b_latent_rms_difference"] for row in pairs)),
        "maximum_pairwise_e_latent_rms_difference": float(max(row["e_latent_rms_difference"] for row in pairs)),
        "maximum_pairwise_b_latent_difference": float(max(row["b_latent_maximum_difference"] for row in pairs)),
        "maximum_pairwise_e_latent_difference": float(max(row["e_latent_maximum_difference"] for row in pairs)),
    }
    gate = {
        "five_realizations": len(realizations) >= 5,
        "all_aggregate_cd_relative_differences_le_0p1_percent": all(row["aggregate_cd_relative_difference_from_tight_reference"] <= 1e-3 for row in realizations),
        "all_maximum_per_sample_cd_differences_le_5e-5": all(row["maximum_cd_difference_from_tight_reference"] <= 5e-5 for row in realizations),
        "all_pcg_residuals_le_1e-8": all(row["maximum_pcg_residual"] <= 1e-8 for row in realizations),
        "all_metrics_finite": all(np.isfinite(value) for row in realizations for key, value in row.items() if key != "realization"),
    }
    payload = {
        "contract_audit": all(gate.values()),
        "contract_checks": gate,
        "checkpoint_identity": {
            "arm_b": model.arm_b_checkpoint,
            "arm_e": model.arm_e_checkpoint,
        },
        "sample_ids": sample_ids,
        "realizations": realizations,
        "pairwise": pairs,
        "execution_nondeterminism_envelope": envelope,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key not in ("sample_ids", "pairwise")}, indent=2))
    if not payload["contract_audit"]:
        raise RuntimeError(f"Five-run nondeterminism gate failed: {gate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
