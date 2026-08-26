#!/usr/bin/env python3
from __future__ import annotations

"""Validation-only frozen B+E lambda sweep for the old native-1920 domain."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from diagnose_sofa50_exact_target_oracle import _clean_mesh, _geometry_row
from mlr.data import Mesh
from mlr.learned_laplacian.differentiable_sparse_recovery import recovery_forward_audit
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


LAMBDAS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def solve(
    delta: np.ndarray,
    direct: np.ndarray,
    static: dict[str, Any],
    regularization: float,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    with torch.no_grad():
        recovered, audit = recovery_forward_audit(
            torch.as_tensor(delta, dtype=torch.float64, device=device),
            torch.as_tensor(direct, dtype=torch.float64, device=device),
            torch.as_tensor(static["edge_index"], dtype=torch.long, device=device),
            torch.as_tensor(static["vertex_degree"], dtype=torch.float64, device=device),
            regularization=regularization,
            maximum_iterations=2048,
            tolerance=1e-8,
        )
    return recovered.cpu().numpy(), {
        "pcg_iterations": int(audit.iterations),
        "pcg_converged": bool(audit.converged),
        "pcg_relative_residual": float(audit.relative_residual),
    }


def shard(args: argparse.Namespace) -> None:
    manifest = args.manifest.resolve()
    summary = read_json(args.specialist_summary.resolve())
    if summary.get("split") != "validation" or summary.get("test_opened") is not False:
        raise RuntimeError("Lambda selection accepts validation predictions only")
    dataset = PreparedMeshDataset.from_manifest(manifest, "validation")
    archive = np.load(args.predictions.resolve())
    ids = archive["sample_ids"].tolist()
    offsets = archive["offsets"].astype(np.int64)
    if ids != list(dataset.sample_ids) or len(dataset) != 25:
        raise RuntimeError("Validation prediction IDs/order mismatch")
    device = torch.device(args.device)
    rows: list[dict[str, Any]] = []
    for index in range(args.shard_index, len(dataset), args.shard_count):
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        start, stop = int(offsets[index]), int(offsets[index + 1])
        delta = archive["b_prediction"][start:stop].astype(np.float64)
        displacement = archive["e_displacement"][start:stop].astype(np.float64)
        initial = Mesh(
            np.asarray(static["vertices"], dtype=np.float64),
            np.asarray(static["faces"], dtype=np.int64),
        ).ensure_normals()
        clean = _clean_mesh(static)
        direct = initial.vertices + displacement
        initial_metric = _geometry_row("validation", sample_id, "initial", initial, clean, initial)
        for regularization in LAMBDAS:
            vertices, audit = solve(delta, direct, static, regularization, device)
            if not audit["pcg_converged"]:
                raise RuntimeError(f"{sample_id} lambda={regularization}: PCG failed")
            metric = _geometry_row(
                "validation",
                sample_id,
                "Frozen_B_E",
                Mesh(vertices, initial.faces.copy()).ensure_normals(),
                clean,
                initial,
            )
            initial_cd = float(initial_metric["chamfer"])
            refined_cd = float(metric["chamfer"])
            rows.append(
                {
                    "sample_id": sample_id,
                    "index": index,
                    "lambda": regularization,
                    "vertices": initial.num_vertices,
                    "faces": initial.num_faces,
                    "initial_chamfer": initial_cd,
                    "refined_chamfer": refined_cd,
                    "relative_chamfer_gain": (initial_cd - refined_cd) / initial_cd,
                    "p2s": float(metric["p2s"]),
                    "p2s_p95": float(metric["p2s_p95"]),
                    "fscore": float(metric["fscore"]),
                    "normal_consistency": float(metric["normal_consistency"]),
                    "introduced_flipped_faces": int(metric["introduced_flipped_faces"]),
                    "new_degenerate_faces": int(metric["new_degenerate_faces"]),
                    "same_index_recovered_vertex_rms": float(
                        np.sqrt(np.mean(np.sum((vertices - clean.vertices) ** 2, axis=1)))
                    ),
                    "hybrid_to_direct_vertex_rms": float(
                        np.sqrt(np.mean(np.sum((vertices - direct) ** 2, axis=1)))
                    ),
                    "improved": refined_cd < initial_cd,
                    "worsened": refined_cd > initial_cd,
                    **audit,
                }
            )
        print(f"shard={args.shard_index} {sample_id}", flush=True)
    write_json(
        args.output_dir / "shards" / f"lambda_{args.shard_index:02d}.json",
        {
            "contract_audit": True,
            "selection_split": "validation",
            "test_accessed": False,
            "lambda_grid": list(LAMBDAS),
            "arm_b_checkpoint_sha256": summary["arm_b_checkpoint_sha256"],
            "arm_e_checkpoint_sha256": summary["arm_e_checkpoint_sha256"],
            "rows": rows,
        },
    )


def aggregate(rows: list[dict[str, Any]], regularization: float) -> dict[str, Any]:
    selected = [row for row in rows if float(row["lambda"]) == regularization]
    return {
        "lambda": regularization,
        "samples": len(selected),
        "refined_chamfer": float(np.mean([row["refined_chamfer"] for row in selected])),
        "relative_chamfer_gain": float(
            np.mean([row["relative_chamfer_gain"] for row in selected])
        ),
        "same_index_recovered_vertex_rms": float(
            np.mean([row["same_index_recovered_vertex_rms"] for row in selected])
        ),
        "p2s": float(np.mean([row["p2s"] for row in selected])),
        "p2s_p95": float(np.mean([row["p2s_p95"] for row in selected])),
        "fscore": float(np.mean([row["fscore"] for row in selected])),
        "normal_consistency": float(
            np.mean([row["normal_consistency"] for row in selected])
        ),
        "introduced_flipped_faces": int(
            sum(row["introduced_flipped_faces"] for row in selected)
        ),
        "normalized_flip_rate": float(
            sum(row["introduced_flipped_faces"] for row in selected)
            / sum(row["faces"] for row in selected)
        ),
        "new_degenerate_faces": int(sum(row["new_degenerate_faces"] for row in selected)),
        "improved": int(sum(row["improved"] for row in selected)),
        "worsened": int(sum(row["worsened"] for row in selected)),
        "pcg_iterations_mean": float(np.mean([row["pcg_iterations"] for row in selected])),
        "pcg_iterations_max": int(max(row["pcg_iterations"] for row in selected)),
        "pcg_relative_residual_max": float(
            max(row["pcg_relative_residual"] for row in selected)
        ),
    }


def merge(args: argparse.Namespace) -> None:
    payloads = [
        read_json(args.output_dir / "shards" / f"lambda_{index:02d}.json")
        for index in range(args.shard_count)
    ]
    rows = [row for payload in payloads for row in payload["rows"]]
    if len(rows) != 25 * len(LAMBDAS):
        raise RuntimeError(f"Expected {25 * len(LAMBDAS)} rows, found {len(rows)}")
    if len({(row["sample_id"], float(row["lambda"])) for row in rows}) != len(rows):
        raise RuntimeError("Duplicate sample/lambda rows")
    aggregates = [aggregate(rows, value) for value in LAMBDAS]
    selected = min(aggregates, key=lambda row: row["refined_chamfer"])
    best_lambda = float(selected["lambda"])
    boundary = best_lambda in {LAMBDAS[0], LAMBDAS[-1]}
    contract = bool(
        all(payload["contract_audit"] for payload in payloads)
        and all(payload["selection_split"] == "validation" for payload in payloads)
        and not any(payload["test_accessed"] for payload in payloads)
        and all(row["pcg_converged"] for row in rows)
        and max(row["pcg_relative_residual"] for row in rows) <= 1e-8
        and len({payload["arm_b_checkpoint_sha256"] for payload in payloads}) == 1
        and len({payload["arm_e_checkpoint_sha256"] for payload in payloads}) == 1
    )
    summary = {
        "contract_audit": contract,
        "selection_split": "validation",
        "test_accessed": False,
        "selection_metric": "macro_mean_unified_surface_chamfer",
        "lambda_grid": list(LAMBDAS),
        "selected_lambda": best_lambda,
        "selected_at_grid_boundary": boundary,
        "solver": {"dtype": "float64", "tolerance": 1e-8, "maximum_iterations": 2048},
        "arm_b_checkpoint_sha256": payloads[0]["arm_b_checkpoint_sha256"],
        "arm_e_checkpoint_sha256": payloads[0]["arm_e_checkpoint_sha256"],
        "aggregate": aggregates,
    }
    write_json(args.output_dir / "lambda_selection.json", summary)
    with (args.output_dir / "lambda_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    with (args.output_dir / "lambda_sweep_per_sample.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["index"], row["lambda"])))
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not contract:
        raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("shard", "merge"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--specialist-summary", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shard-count", type=int, default=5)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.mode == "shard":
        if args.manifest is None or args.predictions is None or args.specialist_summary is None:
            parser.error("shard mode requires manifest, predictions, and specialist summary")
        shard(args)
    else:
        merge(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
