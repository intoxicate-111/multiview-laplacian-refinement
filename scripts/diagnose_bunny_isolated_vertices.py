#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlr.coarse_lap_oracle import apply_uniform_laplacian, build_uniform_laplacian_data
from mlr.io import load_mesh
from mlr.learned_laplacian.graph_layers import faces_to_edge_index
from mlr.learned_laplacian.target_scaling import incident_edge_length_and_valid_mask
from mlr.mesh_cleaning import remove_unreferenced_vertices


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose every isolated Bunny OBJ vertex.")
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--sample", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    mesh = load_mesh(args.mesh)
    faces = mesh.faces
    face_references = np.bincount(faces.reshape(-1), minlength=mesh.num_vertices)
    edge_index = faces_to_edge_index(torch.as_tensor(faces), mesh.num_vertices)
    degree = torch.bincount(edge_index[1], minlength=mesh.num_vertices).cpu().numpy()
    h, valid = incident_edge_length_and_valid_mask(
        torch.as_tensor(mesh.vertices, dtype=torch.float64), edge_index
    )
    corrected_delta = apply_uniform_laplacian(
        mesh.vertices, build_uniform_laplacian_data(faces, mesh.num_vertices)
    )
    confidence = np.ones(mesh.num_vertices, dtype=np.float64)
    raw_delta = corrected_delta
    raw_source = "corrected sparse uniform Laplacian"
    if args.sample is not None:
        sample = torch.load(args.sample, map_location="cpu", weights_only=False)
        if sample["vertices"].shape[0] != mesh.num_vertices:
            raise ValueError("--sample vertex count does not match --mesh")
        raw_delta = sample.get("raw_laplacian_target", sample["laplacian_target"]).numpy()
        confidence = sample["target_confidence"].numpy()
        raw_source = "prepared sample raw_laplacian_target"
    isolated = np.flatnonzero(degree == 0)
    cleaned = remove_unreferenced_vertices(mesh.vertices, faces)
    records = []
    for index in isolated:
        records.append(
            {
                "vertex_index": int(index),
                "position": mesh.vertices[index].tolist(),
                "face_reference_count": int(face_references[index]),
                "graph_degree": int(degree[index]),
                "raw_laplacian_vector": raw_delta[index].tolist(),
                "raw_laplacian_magnitude": float(np.linalg.norm(raw_delta[index])),
                "corrected_zero_row_laplacian_vector": corrected_delta[index].tolist(),
                "legacy_identity_row_laplacian_vector": mesh.vertices[index].tolist(),
                "local_edge_length": float(h[index]),
                "valid_scale": bool(valid[index]),
                "target_confidence": float(confidence[index]),
            }
        )
    report = {
        "mesh": str(args.mesh),
        "raw_laplacian_source": raw_source,
        "confirmed_cause": (
            "OBJ vertex records not referenced by any face"
            if np.array_equal(isolated, np.flatnonzero(face_references == 0))
            else "graph construction mismatch"
        ),
        "summary": {
            "total_vertices_before_cleaning": mesh.num_vertices,
            "vertices_referenced_by_faces": int((face_references > 0).sum()),
            "unreferenced_vertices": int((face_references == 0).sum()),
            "isolated_graph_vertices": int(len(isolated)),
            "faces_before_cleaning": mesh.num_faces,
            "faces_after_cleaning": int(len(cleaned.faces)),
            "degenerate_faces_removed": int(len(cleaned.degenerate_face_indices)),
            "duplicate_faces_detected": int(len(cleaned.duplicate_face_indices)),
        },
        "isolated_vertices": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "isolated_vertices"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
