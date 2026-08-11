#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping

import torch

import test_sofa50_openmvs48_c2f2_3seed as base


ARMS = ("A_no_jitter", "B_local_jitter")
EXPECTED_VIEWS = 28


def controlled_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(config))
    value.setdefault("local_query_jitter", {})["enabled"] = "CONTROLLED_ARM_VARIABLE"
    value.setdefault("experiment_metadata", {})["arm"] = "CONTROLLED_ARM_LABEL"
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--coarse-models-root", type=Path, required=True)
    parser.add_argument("--a-run", type=Path, required=True)
    parser.add_argument("--b-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mesh-name", default="coarse.obj")
    parser.add_argument("--split", choices=("validation", "test", "all"), default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--visibility-backend", choices=("opengl",), default="opengl")
    parser.add_argument("--visibility-size", type=int)
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()

    repo_root = base.expand(args.repo_root)
    source_manifest = base.expand(args.source_manifest)
    coarse_root = base.expand(args.coarse_models_root)
    output_dir = base.expand(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("OpenMVS A/B evaluation requires CUDA")
    run_dirs = {
        ARMS[0]: base.expand(args.a_run),
        ARMS[1]: base.expand(args.b_run),
    }
    configs = {arm: base.read_json(base.config_path(run_dirs[arm])) for arm in ARMS}
    if controlled_config(configs[ARMS[0]]) != controlled_config(configs[ARMS[1]]):
        raise ValueError("Arm configs differ outside jitter enablement and arm label")
    modules = base.load_runtime_modules(repo_root)
    records = base.manifest_records(source_manifest, args.split)
    prepared_records, missing = base.prepare_query_samples(
        records=records,
        source_manifest=source_manifest,
        coarse_models_root=coarse_root,
        mesh_name=args.mesh_name,
        output_dir=output_dir / "shared_current_queries",
        visibility_backend=args.visibility_backend,
        visibility_size=args.visibility_size,
        require_all=args.require_all,
        modules=modules,
        expected_views=EXPECTED_VIEWS,
        zero_initial_laplacian=False,
    )

    rows = []
    for arm in ARMS:
        arm_rows = base.infer_seed(
            seed=7,
            seed_dir=run_dirs[arm],
            prepared_records=prepared_records,
            output_dir=output_dir / arm,
            device=device,
            modules=modules,
            expected_views=EXPECTED_VIEWS,
        )
        for row in arm_rows:
            row["arm"] = arm
            row["variant"] = arm
        rows.extend(arm_rows)

    aggregates = []
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        aggregates.append(
            {
                "arm": arm,
                "mesh_count": len(selected),
                "mean_initial_chamfer": base._mean(selected, "initial_chamfer"),
                "mean_refined_chamfer": base._mean(selected, "refined_chamfer"),
                "median_refined_chamfer": base._median(selected, "refined_chamfer"),
                "mean_chamfer_improvement": base._mean(selected, "chamfer_improvement"),
                "mean_chamfer_ratio_to_initial": base._mean(selected, "chamfer_ratio_to_initial"),
                "better_than_initial_meshes": int(sum(bool(row["better_than_initial_chamfer"]) for row in selected)),
                "mean_initial_point_to_surface": base._mean(selected, "initial_point_to_surface"),
                "mean_refined_point_to_surface": base._mean(selected, "refined_point_to_surface"),
                "mean_initial_normal_consistency": base._mean(selected, "initial_normal_consistency"),
                "mean_refined_normal_consistency": base._mean(selected, "refined_normal_consistency"),
                "introduced_flips": int(sum(int(row.get("introduced_flips") or 0) for row in selected)),
                "new_degeneracies": int(sum(int(row.get("new_degeneracies") or 0) for row in selected)),
                "mean_vertex_displacement": base._mean(selected, "mean_vertex_displacement"),
            }
        )
    paired = []
    indexed = {(row["arm"], row["sample_id"]): row for row in rows}
    for sample_id in sorted({row["sample_id"] for row in rows}):
        a = indexed[(ARMS[0], sample_id)]
        b = indexed[(ARMS[1], sample_id)]
        paired.append(
            {
                "sample_id": sample_id,
                "A_refined_chamfer": a["refined_chamfer"],
                "B_refined_chamfer": b["refined_chamfer"],
                "B_minus_A_refined_chamfer": float(b["refined_chamfer"]) - float(a["refined_chamfer"]),
                "A_refined_point_to_surface": a["refined_point_to_surface"],
                "B_refined_point_to_surface": b["refined_point_to_surface"],
                "A_refined_normal_consistency": a["refined_normal_consistency"],
                "B_refined_normal_consistency": b["refined_normal_consistency"],
            }
        )
    summary = {
        "experiment": "Sofa50 OpenMVS48 current-mesh recovery, 28-view A/B checkpoints",
        "source_manifest": str(source_manifest),
        "coarse_models_root": str(coarse_root),
        "prediction_views": EXPECTED_VIEWS,
        "split": args.split,
        "visibility_backend": args.visibility_backend,
        "current_graph_initial_laplacian": "L_current@C",
        "gt_differential_transfer_used": False,
        "recovery_config_equal": configs[ARMS[0]].get("recovery") == configs[ARMS[1]].get("recovery"),
        "evaluated_mesh_count": len(prepared_records),
        "missing": missing,
        "aggregate": aggregates,
        "paired": paired,
        "per_mesh": rows,
    }
    base.write_csv(output_dir / "openmvs_per_mesh.csv", rows)
    base.write_csv(output_dir / "openmvs_aggregate.csv", aggregates)
    base.write_csv(output_dir / "openmvs_paired.csv", paired)
    base.write_json(output_dir / "openmvs_summary.json", summary)
    print(json.dumps({"output": str(output_dir), "aggregate": aggregates}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
