from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]


def _module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path: Path, value: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def test_sanity_gate_requires_same_mesh_and_rgb_contract(tmp_path: Path) -> None:
    audit = _module("same_initial_sanity", "scripts/audit_sofa50_same_initial_sanity.py")
    sample_id = "object__v01"
    initial = _write(tmp_path / "initial.obj", "mesh")
    initial_sha = hashlib.sha256(initial.read_bytes()).hexdigest()
    images = [str(_write(tmp_path / "rgb" / f"{index:02d}.png").resolve()) for index in range(28)]
    manifest = {
        "representative_sample_id": sample_id,
        "samples": [
            {
                "sample_id": sample_id,
                "common_initial_mesh": str(initial),
                "common_initial_mesh_sha256": initial_sha,
                "image_paths": images,
            }
        ],
    }
    manifest_path = _write(tmp_path / "manifest.json", json.dumps(manifest))
    coordinate_path = _write(
        tmp_path / "coordinate.json",
        json.dumps({"contract_audit": True, "sample_id": sample_id}),
    )
    sanity = tmp_path / "sanity"
    ours_dir = sanity / "ours" / "samples" / sample_id
    ours_row = {
        "status": "completed",
        "sample_id": sample_id,
        "common_initial_mesh_sha256": initial_sha,
        "view_count": 28,
        "final_mesh": str(_write(ours_dir / "refined.obj")),
        "common_initial_identity_audit": True,
    }
    _write(ours_dir / "status.json", json.dumps(ours_row))
    for name in (
        "predicted_raw_laplacian.npy",
        "predicted_confidence.npy",
        "recovery_weight.npy",
        "visibility_used.npz",
        "recovery_config.json",
    ):
        _write(ours_dir / name)
    for method in ("exmesh", "nds", "nvdiffrec"):
        method_dir = sanity / method / "samples" / sample_id
        row = {
            "status": "completed",
            "sample_id": sample_id,
            "common_initial_mesh_sha256": initial_sha,
            "view_count": 28,
            "final_mesh": str(_write(method_dir / "refined.obj")),
            "common_initial_source_identity_audit": True,
            "common_initial_identity_audit": True,
            "output_connectivity_preserved": method == "nvdiffrec",
        }
        command = "train --initial_mesh input.obj" if method == "nds" else "train"
        _write(method_dir / "status.json", json.dumps({"row": row, "commands": [command]}))
        _write(
            method_dir / "input_contract.json",
            json.dumps(
                {
                    "source_images": images,
                    "forbidden_fields_consumed": [],
                }
            ),
        )
    result = audit.audit(
        SimpleNamespace(
            manifest=manifest_path,
            coordinate_audit=coordinate_path,
            sanity_root=sanity,
            output=tmp_path / "gate.json",
        )
    )
    assert result["contract_audit"] is True
    assert result["failed_checks"] == []


def test_aggregate_reports_all_group_a_rows(tmp_path: Path) -> None:
    aggregate = _module("same_initial_aggregate", "scripts/aggregate_sofa50_same_initial_benchmark.py")
    sample_id = "object__v01"
    source = {
        "sample_id": sample_id,
        "common_initial_mesh": str(tmp_path / "initial.obj"),
        "common_initial_mesh_sha256": "abc",
        "initial_vertex_count": 10,
        "initial_face_count": 20,
        "image_directory": str(tmp_path / "rgb"),
        "camera_and_gt_container": str(tmp_path / "sample.pt"),
        "view_count": 28,
    }
    manifest_path = _write(
        tmp_path / "manifest.json",
        json.dumps(
            {
                "source_manifest": "canonical.json",
                "common_input_contract": "same mesh + rgb + cameras",
                "samples": [source],
            }
        ),
    )
    config_path = _write(
        tmp_path / "config.json",
        json.dumps(
            {
                "methods": {
                    "neuralangelo": {"reason": "SDF"},
                    "matcha": {"reason": "chart"},
                }
            }
        ),
    )
    results = tmp_path / "full"
    base = {
        "sample_id": sample_id,
        "status": "completed",
        "common_initial_mesh_sha256": "abc",
        "common_initial_source_identity_audit": True,
        "common_initial_identity_audit": True,
        "initial_chamfer": 1.0,
        "initial_p2s_mean": 1.0,
        "initial_normal_consistency": 0.5,
        "refined_chamfer": 0.8,
        "refined_p2s_mean": 0.8,
        "refined_normal_consistency": 0.6,
        "vertex_count": 10,
        "face_count": 20,
        "runtime_seconds": 2.0,
        "peak_gpu_memory_mb": 100.0,
        "output_connectivity_preserved": False,
    }
    for method in ("exmesh", "nds", "nvdiffrec"):
        row = dict(base)
        row["output_connectivity_preserved"] = method == "nvdiffrec"
        _write(results / method / "samples" / sample_id / "status.json", json.dumps({"row": row}))
    ours = {
        **base,
        "common_initial_identity_audit": True,
        "reconstruction_chamfer": 0.7,
        "initial_point_to_surface": 1.0,
        "reconstruction_point_to_surface": 0.7,
        "reconstruction_normal_consistency": 0.7,
        "final_vertex_count": 10,
        "final_face_count": 20,
        "output_connectivity_preserved": True,
    }
    _write(results / "ours" / "samples" / sample_id / "status.json", json.dumps(ours))
    unified_final = {"ours": 0.7, "exmesh": 0.8, "nds": 0.6, "nvdiffrec": 0.75}

    def fake_unified_reevaluate(received_manifest, sources, rows, **kwargs):
        assert received_manifest == manifest_path
        assert sources == [source]
        assert kwargs == {
            "surface_samples": 3000,
            "seed": 7,
            "fscore_threshold": 0.01,
        }
        initial = {
            "chamfer": 1.25,
            "p2s": 1.5,
            "p2s_p95": 2.0,
            "fscore": 0.25,
            "normal_consistency": 0.5,
        }
        for row in rows:
            final = dict(initial)
            final["chamfer"] = unified_final[row["method"]]
            aggregate._assign_unified_metrics(row, initial, final, "test-unified")

    aggregate._unified_reevaluate = fake_unified_reevaluate
    output = tmp_path / "report"
    summary = aggregate.run(
        SimpleNamespace(
            manifest=manifest_path,
            config=config_path,
            results_root=results,
            output_dir=output,
        )
    )
    assert summary["contract_audit"] is True
    assert len(summary["aggregate"]) == 5
    assert summary["metric_contract"]["native_method_metrics_role"] == "provenance_only"
    rows = json.loads((output / "per_sample.json").read_text(encoding="utf-8"))
    method_rows = [row for row in rows if row["method"] != "initial"]
    assert {row["initial_chamfer"] for row in method_rows} == {1.25}
    assert {row["metric_protocol"] for row in method_rows} == {"test-unified"}
    assert {row["native_initial_chamfer"] for row in method_rows} == {1.0}
    assert (output / "per_sample.csv").is_file()
    assert "same prepared synthetic mesh" in (output / "FINAL_REPORT.md").read_text()
