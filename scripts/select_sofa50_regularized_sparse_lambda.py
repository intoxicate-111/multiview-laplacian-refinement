#!/usr/bin/env python3
from __future__ import annotations

"""Select the recovery lambda using only Sofa50 v2 validation predictions."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from diagnose_sofa50_exact_solve_visibility_sweep import (
    component_labels,
    uniform_sparse_laplacian,
)
from diagnose_sofa50_exact_target_oracle import METRIC_PROTOCOL, _clean_mesh, _geometry_row
from diagnose_sofa50_regularized_sparse_sweep import regularized_sparse_solve
from evaluate_sofa50_multitopology_rawlap import load_spec
from mlr.data import Mesh
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset
from mlr.learned_laplacian.synthetic_current_h2_ablation import _infer_one


CANDIDATE_LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_shard(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    shard_dir = output / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Validation prediction inference requires CUDA.")
    dataset = PreparedMeshDataset.from_manifest(args.manifest.resolve(), "validation")
    spec = load_spec(args.run_dir.resolve(), device)
    rows: list[dict[str, Any]] = []
    for index in range(args.shard_index, len(dataset), args.shard_count):
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        initial = Mesh(
            torch.as_tensor(static["vertices"]).cpu().numpy(),
            torch.as_tensor(static["faces"]).cpu().numpy().astype(np.int64),
        ).ensure_normals()
        clean = _clean_mesh(static)
        values = _infer_one(dataset, index, spec, device, current_faces=static["faces"])
        prediction = values["prediction_raw"].cpu().numpy().astype(np.float64)
        laplacian, lap_data = uniform_sparse_laplacian(
            initial.faces, initial.num_vertices
        )
        component_count, labels = component_labels(lap_data)
        initial_geometry = _geometry_row(
            "v2_strong_smoothing", sample_id, "initial", initial, clean, initial
        )
        clean_geometry = _geometry_row(
            "v2_strong_smoothing", sample_id, "clean", clean, clean, initial
        )
        initial_chamfer = float(initial_geometry["chamfer"])
        clean_chamfer = float(clean_geometry["chamfer"])
        available = initial_chamfer - clean_chamfer
        if available <= 0:
            raise RuntimeError(f"Invalid validation recoverable gap for {sample_id}.")
        for regularization in CANDIDATE_LAMBDAS:
            recovered, solver = regularized_sparse_solve(
                laplacian,
                prediction,
                initial.vertices,
                labels,
                component_count,
                regularization,
                atol=1e-12,
                btol=1e-12,
                maxiter=100000,
            )
            recovered_mesh = Mesh(recovered, initial.faces.copy()).ensure_normals()
            geometry = _geometry_row(
                "v2_strong_smoothing",
                sample_id,
                f"validation_predicted_lambda_{regularization:.0e}",
                recovered_mesh,
                clean,
                initial,
            )
            chamfer = float(geometry["chamfer"])
            rows.append(
                {
                    "split": "validation",
                    "sample_id": sample_id,
                    "lambda": regularization,
                    "chamfer": chamfer,
                    "p2s_mean": float(geometry["p2s"]),
                    "p2s_p95": float(geometry["p2s_p95"]),
                    "normal_consistency": float(geometry["normal_consistency"]),
                    "introduced_flipped_faces": int(
                        geometry["introduced_flipped_faces"]
                    ),
                    "new_degenerate_faces": int(geometry["new_degenerate_faces"]),
                    "initial_chamfer": initial_chamfer,
                    "clean_chamfer": clean_chamfer,
                    "relative_chamfer_gain": (
                        initial_chamfer - chamfer
                    ) / initial_chamfer,
                    "eta": (initial_chamfer - chamfer) / available,
                    "same_index_vertex_rms": float(
                        np.sqrt(np.mean(np.sum((recovered - clean.vertices) ** 2, axis=1)))
                    ),
                    "solver_runtime_seconds": float(solver["runtime_seconds"]),
                    "lsmr_all_converged": bool(solver["all_converged"]),
                    "vertices": initial.num_vertices,
                    "faces": initial.num_faces,
                }
            )
        print(f"validation {sample_id}: completed {len(CANDIDATE_LAMBDAS)} lambdas", flush=True)
    _write_json(
        shard_dir / f"shard_{args.shard_index:02d}.json",
        {
            "split": "validation",
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "candidate_lambdas": list(CANDIDATE_LAMBDAS),
            "checkpoint": spec["checkpoint"],
            "checkpoint_sha256": spec["checkpoint_sha256"],
            "checkpoint_optimizer_steps": spec["optimizer_steps"],
            "rows": rows,
        },
    )


def merge(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    payloads = [
        json.loads((output / "shards" / f"shard_{index:02d}.json").read_text())
        for index in range(args.shard_count)
    ]
    rows = [row for payload in payloads for row in payload["rows"]]
    if any(payload["split"] != "validation" for payload in payloads):
        raise RuntimeError("Lambda selection loaded a non-validation split.")
    if any(tuple(payload["candidate_lambdas"]) != CANDIDATE_LAMBDAS for payload in payloads):
        raise RuntimeError("Lambda candidates differ across shards.")
    sample_ids = sorted({str(row["sample_id"]) for row in rows})
    aggregate: list[dict[str, Any]] = []
    for regularization in CANDIDATE_LAMBDAS:
        selected = [row for row in rows if float(row["lambda"]) == regularization]
        if len(selected) != len(sample_ids):
            raise RuntimeError(f"Incomplete lambda={regularization} validation rows.")
        aggregate.append(
            {
                "lambda": regularization,
                "samples": len(selected),
                "mean_chamfer": float(np.mean([row["chamfer"] for row in selected])),
                "mean_relative_chamfer_gain": float(
                    np.mean([row["relative_chamfer_gain"] for row in selected])
                ),
                "mean_eta": float(np.mean([row["eta"] for row in selected])),
                "mean_same_index_vertex_rms": float(
                    np.mean([row["same_index_vertex_rms"] for row in selected])
                ),
                "mean_normal_consistency": float(
                    np.mean([row["normal_consistency"] for row in selected])
                ),
                "introduced_flipped_faces": int(
                    np.sum([row["introduced_flipped_faces"] for row in selected])
                ),
                "new_degenerate_faces": int(
                    np.sum([row["new_degenerate_faces"] for row in selected])
                ),
                "improved": int(
                    np.sum([row["chamfer"] < row["initial_chamfer"] for row in selected])
                ),
                "worsened": int(
                    np.sum([row["chamfer"] > row["initial_chamfer"] for row in selected])
                ),
                "all_lsmr_converged": all(row["lsmr_all_converged"] for row in selected),
            }
        )
    # Predeclared rule: minimum mean unified-v2 validation Chamfer. Same-index
    # RMS is only a deterministic tie-breaker and test is never loaded.
    chosen = min(
        aggregate,
        key=lambda row: (float(row["mean_chamfer"]), float(row["mean_same_index_vertex_rms"])),
    )
    contract = {
        "passed": bool(
            len(sample_ids) == 50
            and len(rows) == 50 * len(CANDIDATE_LAMBDAS)
            and all(row["all_lsmr_converged"] for row in aggregate)
        ),
        "selection_split": "validation",
        "test_split_loaded": False,
        "candidate_lambdas_predeclared": list(CANDIDATE_LAMBDAS),
        "selection_rule": "minimum mean unified-v2 validation Chamfer; same-index vertex RMS tie-break",
        "metric_protocol": METRIC_PROTOCOL,
        "checkpoint_sha256": payloads[0]["checkpoint_sha256"],
        "checkpoint_optimizer_steps": payloads[0]["checkpoint_optimizer_steps"],
    }
    summary = {
        "contract_audit": contract,
        "selected_lambda": chosen["lambda"],
        "selected_validation_metrics": chosen,
        "validation_curve": aggregate,
    }
    _write_csv(output / "validation_lambda_per_sample.csv", rows)
    _write_csv(output / "validation_lambda_curve.csv", aggregate)
    _write_json(output / "lambda_selection.json", summary)
    lines = [
        "# Sofa50 v2 validation-only sparse-recovery lambda selection",
        "",
        f"Contract audit: **{str(contract['passed']).lower()}**.",
        "",
        "The test split was not loaded. Lambda was selected by minimum mean unified-v2 validation Chamfer.",
        "",
        "| Lambda | Chamfer | Relative gain | Eta | Same-index RMS | Normal | Flips | Improved |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['lambda']:.0e} | {row['mean_chamfer']:.9g} | "
            f"{row['mean_relative_chamfer_gain']:.2%} | {row['mean_eta']:.9g} | "
            f"{row['mean_same_index_vertex_rms']:.9g} | {row['mean_normal_consistency']:.9g} | "
            f"{row['introduced_flipped_faces']} | {row['improved']}/50 |"
        )
    lines.extend(("", f"Selected lambda: **{chosen['lambda']:.0e}**.", ""))
    (output / "LAMBDA_SELECTION_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    if args.merge_only:
        merge(args)
    else:
        if args.manifest is None or args.run_dir is None:
            parser.error("--manifest and --run-dir are required for shard evaluation")
        evaluate_shard(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
