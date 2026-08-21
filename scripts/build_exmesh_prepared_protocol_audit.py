#!/usr/bin/env python3
"""Build a versioned audit before a corrected prepared-mesh ExMesh run.

The script deliberately leaves the primary prepared rows unresolved when no
audited prepared input exists.  It never promotes an ExMesh/PGSR mesh to the
primary ``ours`` role.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlr.io import load_mesh  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-status", type=Path, required=True)
    parser.add_argument("--official-initial", type=Path, required=True)
    parser.add_argument("--old-coarse", type=Path, required=True)
    parser.add_argument("--old-output", type=Path, required=True)
    parser.add_argument("--old-comparison", type=Path, required=True)
    parser.add_argument("--rejected-pgsr-candidate", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mesh_record(path: Path) -> tuple[dict[str, Any], Any]:
    mesh = load_mesh(path)
    bbox_min = mesh.vertices.min(axis=0)
    bbox_max = mesh.vertices.max(axis=0)
    record = {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "vertices": mesh.num_vertices,
        "faces": mesh.num_faces,
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "bbox_diagonal": float(np.linalg.norm(bbox_max - bbox_min)),
    }
    return record, mesh


def same_connectivity(first: Any, second: Any) -> bool:
    return bool(np.array_equal(first.faces, second.faces))


def close_vertices(first: Any, second: Any, tolerance: float = 5e-8) -> bool:
    return bool(
        first.vertices.shape == second.vertices.shape
        and np.max(np.abs(first.vertices - second.vertices), initial=0.0) <= tolerance
    )


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty snapshot: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    old_status = json.loads(args.old_status.read_text(encoding="utf-8"))
    old_comparison = json.loads(args.old_comparison.read_text(encoding="utf-8"))
    official_record, official_mesh = mesh_record(args.official_initial)
    coarse_record, coarse_mesh = mesh_record(args.old_coarse)
    output_record, output_mesh = mesh_record(args.old_output)
    coarse_is_official = same_connectivity(coarse_mesh, official_mesh) and close_vertices(
        coarse_mesh, official_mesh
    )

    rejected_candidates: list[dict[str, Any]] = []
    if args.rejected_pgsr_candidate is not None:
        candidate_record, _ = mesh_record(args.rejected_pgsr_candidate)
        candidate_record.update(
            {
                "decision": "rejected",
                "reason": (
                    "The archived job command copies ExMesh PGSR "
                    "tsdf_fusion_post.ply into current_meshes/scan24_mesh.ply; "
                    "it is not a mesh prepared by the learned-Laplacian pipeline."
                ),
                "provenance": "official ExMesh PGSR output from cancelled job 15910",
            }
        )
        rejected_candidates.append(candidate_record)

    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scene_id": 24,
        "audit_state": "blocked_before_corrected_inference",
        "old_result": {
            "old_label": "ours",
            "corrected_label": "ours_exmesh_initial_zero_shot",
            "overall_cd_mm": old_status["metrics"]["overall"],
            "d2s_mm": old_status["metrics"]["mean_d2s"],
            "s2d_mm": old_status["metrics"]["mean_s2d"],
            "absolute_input_mesh_path_recorded_by_run": old_status["initial_mesh"],
            "input_mesh_sha256": official_record["sha256"],
            "vertices": official_record["vertices"],
            "faces": official_record["faces"],
            "bbox_min": official_record["bbox_min"],
            "bbox_max": official_record["bbox_max"],
            "bbox_diagonal": official_record["bbox_diagonal"],
            "is_pipeline_prepared_mesh": False,
            "byte_identical_to_official_exmesh_initial": True,
            "archived_coarse_geometry_matches_official_initial": coarse_is_official,
            "graph_operator_source": "faces_to_edge_index(F_current) from official ExMesh PGSR input; uniform current-graph Laplacian",
            "query_position_source": "V_current from official ExMesh PGSR input",
            "output_mesh_path_recorded_by_run": old_status["final_mesh"],
            "archived_output": output_record,
            "output_connectivity_matches_input": same_connectivity(
                output_mesh, official_mesh
            ),
        },
        "official_exmesh_initial_archived_copy": official_record,
        "archived_old_coarse": coarse_record,
        "rejected_candidates": rejected_candidates,
        "search_audit": {
            "searched": [
                "local ~/sofa_mesh and compares snapshots",
                "HPC project configs, manifests, docs, scripts, backups, and Slurm logs",
                "HPC runs/exmesh_baselines including cancelled jobs",
                "HPC project mesh/ and meshes/ directories",
                "HPC external_baselines scan24 mesh outputs",
                "HPC data_prepare, sofa_mesh, 48mesh_res, and recent mesh outputs",
            ],
            "valid_pipeline_prepared_dtu_scan24_mesh_found": False,
            "required_to_continue": (
                "An absolute path or manifest entry for the existing pipeline-prepared "
                "DTU scan24 mesh in the audited ExMesh normalized frame."
            ),
        },
        "corrected_primary": {
            "input_role": "ours_prepared_initial",
            "output_role": "ours",
            "status": "not_run_missing_verified_prepared_mesh",
            "checkpoint_unchanged": True,
            "checkpoint_sha256": old_status["source_checkpoint_sha256"],
        },
        "remaining_protocol_uncertainty": (
            "No existing file or metadata in the searched trees identifies a non-PGSR "
            "prepared mesh for DTU scan24. Choosing a Sofa50 mesh, a baseline output, or "
            "an anonymous generic mesh would violate the requested protocol."
        ),
    }
    (output_dir / "PROTOCOL_AUDIT.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    relabelled_old_status = {
        **old_status,
        "original_method_label": old_status.get("method"),
        "method": "ours_exmesh_initial_zero_shot",
        "comparison_label": "ours_exmesh_initial_zero_shot",
        "primary_method_role": False,
        "identical_to_official_exmesh_initial": True,
        "protocol_relabel_reason": (
            "The optimization input is byte-identical to official exmesh_initial, "
            "not a pipeline-prepared DTU mesh."
        ),
    }
    (output_dir / "ours_exmesh_initial_zero_shot_status.json").write_text(
        json.dumps(relabelled_old_status, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    source_records = {
        str(record["method"]): record for record in old_comparison["records"]
    }
    rows: list[dict[str, Any]] = [
        {"role": "ours_prepared_initial", "status": "blocked", "overall_cd_mm": None, "d2s_mm": None, "s2d_mm": None, "vertices": None, "faces": None, "notes": "verified prepared DTU scan24 mesh not found"},
        {"role": "ours", "status": "not_run", "overall_cd_mm": None, "d2s_mm": None, "s2d_mm": None, "vertices": None, "faces": None, "notes": "requires ours_prepared_initial"},
    ]
    role_map = (
        ("ours_exmesh_initial_zero_shot", "ours"),
        ("exmesh_initial", "exmesh_initial"),
        ("exmesh_official", "exmesh_official"),
        ("neural_deferred_shading", "neural_deferred_shading"),
        ("nvdiffrec", "nvdiffrec"),
        ("neuralangelo", "neuralangelo"),
        ("matcha", "matcha"),
    )
    for role, source_method in role_map:
        record = source_records[source_method]
        rows.append(
            {
                "role": role,
                "status": "success" if record["success"] else "not_run",
                "overall_cd_mm": record["official_chamfer_overall_mm"],
                "d2s_mm": record["official_accuracy_d2s_mm"],
                "s2d_mm": record["official_completeness_s2d_mm"],
                "vertices": record["vertices"],
                "faces": record["faces"],
                "notes": (
                    "exploratory only; input is official ExMesh initial"
                    if role == "ours_exmesh_initial_zero_shot"
                    else record["notes"]
                ),
            }
        )
    (output_dir / "corrected_comparison.json").write_text(
        json.dumps(
            {
                "generated_at": audit["generated_at"],
                "protocol_state": audit["audit_state"],
                "records": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with (output_dir / "corrected_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Corrected ExMesh prepared-mesh protocol audit",
        "",
        "The old `ours = 0.616526 mm` row is not the primary method result. It used the",
        "official ExMesh/PGSR initial mesh and is relabelled",
        "`ours_exmesh_initial_zero_shot`.",
        "",
        "## Old result identity",
        "",
        f"- Recorded absolute input: `{old_status['initial_mesh']}`",
        f"- SHA-256: `{official_record['sha256']}`",
        f"- Mesh: {official_record['vertices']:,} vertices / {official_record['faces']:,} faces",
        f"- Bbox: `{official_record['bbox_min']}` to `{official_record['bbox_max']}`; diagonal {official_record['bbox_diagonal']:.9f}",
        "- It is byte-identical to the archived official `exmesh_initial` mesh.",
        "- Graph/operator: constructed from that mesh's faces; queries: that mesh's vertices.",
        f"- Output: `{old_status['final_mesh']}`",
        "",
        "## Prepared-input discovery",
        "",
        "No non-PGSR mesh with metadata tying it to the learned-Laplacian DTU scan24",
        "preparation pipeline was found. The misleading cancelled-job candidate",
        "`current_meshes/scan24_mesh.ply` is explicitly created by copying ExMesh PGSR",
        "`tsdf_fusion_post.ply`, so it is rejected.",
        "",
        "## Current decision",
        "",
        "The corrected primary inference is intentionally not started. A guessed Sofa50,",
        "external-baseline, generic normalized, or PGSR mesh would repeat the protocol error.",
        "The prepared-mesh runner is ready and rejects geometry identical to the official",
        "ExMesh initial automatically.",
        "",
        "`corrected_comparison.csv` and `corrected_comparison.json` reserve separate roles",
        "for `ours_prepared_initial`, `ours`, `ours_exmesh_initial_zero_shot`, and the",
        "official/end-to-end baselines; unresolved primary cells remain empty rather than",
        "being filled with the exploratory score.",
    ]
    (output_dir / "CORRECTED_COMPARISON_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
