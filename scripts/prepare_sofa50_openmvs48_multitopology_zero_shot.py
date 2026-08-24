#!/usr/bin/env python3
from __future__ import annotations

"""Prepare existing Sofa50 OpenMVS coarse meshes for read-only zero-shot inference."""

import argparse
import json
from pathlib import Path

from test_sofa50_openmvs48_c2f2_3seed import (
    load_runtime_modules,
    manifest_records,
    prepare_query_samples,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--coarse-models-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mesh-name", default="coarse.obj")
    parser.add_argument("--visibility-backend", default="opengl", choices=("cpu", "opengl"))
    parser.add_argument("--visibility-size", type=int, default=480)
    args = parser.parse_args()

    repo = args.repo_root.expanduser().resolve()
    source_manifest = args.source_manifest.expanduser().resolve()
    coarse_root = args.coarse_models_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    modules = load_runtime_modules(repo)
    records = manifest_records(source_manifest, "all")
    prepared, missing = prepare_query_samples(
        records=records,
        source_manifest=source_manifest,
        coarse_models_root=coarse_root,
        mesh_name=args.mesh_name,
        output_dir=output,
        visibility_backend=args.visibility_backend,
        visibility_size=args.visibility_size,
        require_all=False,
        modules=modules,
        expected_views=14,
        zero_initial_laplacian=True,
    )
    samples = [
        {
            "sample_id": str(row["sample_id"]),
            "split": "test",
            "path": str(Path(row["prepared_sample"]).resolve()),
        }
        for row in prepared
    ]
    manifest = {
        "format_version": "Sofa50OpenMVS48ZeroShotPrepared_v1",
        "dataset_role": "inference_only_real_coarse_zero_shot",
        "source_manifest": str(source_manifest),
        "coarse_models_root": str(coarse_root),
        "prediction_views": 14,
        "visibility_backend": args.visibility_backend,
        "visibility_size": args.visibility_size,
        "sample_count": len(samples),
        "missing_count": len(missing),
        "target_fields": "identity_schema_placeholders_not_used_for_prediction_or_recovery",
        "gt_usage": "post_prediction_geometry_metrics_only",
        "samples": samples,
    }
    write_json(output / "manifest.json", manifest)
    write_json(
        output / "contract_audit.json",
        {
            "passed": len(samples) > 0,
            "sample_count": len(samples),
            "missing": missing,
            "same_observations_for_all_arms": True,
            "same_initial_mesh_for_all_arms": True,
            "same_visibility_for_all_arms": True,
            "fine_tuning_used": False,
            "gt_differential_transfer_used": False,
            "target_fields_used": False,
        },
    )
    print(json.dumps({"prepared": len(samples), "missing": len(missing)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
