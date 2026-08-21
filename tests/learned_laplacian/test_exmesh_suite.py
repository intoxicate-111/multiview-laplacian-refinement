from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from mlr.baselines.exmesh_suite import (
    METHODS,
    aggregate_results,
    decompose_projection_matrix,
    load_suite_config,
    nvdiffrec_projection_from_intrinsics,
    prepare_nds_scene,
    prepare_neuralangelo_scene,
    prepare_nvdiffrec_scene,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/baselines/exmesh_official_suite.json"
RUNNER = ROOT / "scripts/HPC/run_exmesh_official_scene.slurm"
NDS_RUNNER = ROOT / "scripts/HPC/run_exmesh_nds_sanity.slurm"
NVDIFFREC_RUNNER = ROOT / "scripts/HPC/run_exmesh_nvdiffrec_sanity.slurm"


def test_config_pins_official_suite_and_has_no_legacy_dataset_paths() -> None:
    config = load_suite_config(CONFIG)
    assert len(config["scene_ids"]) == 15
    assert config["sanity_scene_id"] == 24
    assert config["ours_contract"]["training_source"] is None
    paths = json.dumps(config["paths"]).lower()
    assert "sofa50" not in paths
    assert "openmvs" not in paths


def test_official_runner_is_fail_fast_and_uses_released_pgsr_mesh_path() -> None:
    runner = RUNNER.read_text()
    assert 'PGSR_MESH="${PGSR_MODEL}/mesh/tsdf_fusion_post.ply"' in runner
    assert 'cat > "${OUTPUT}/command.sh" <<EOF\n#!/bin/bash\nset -euo pipefail' in runner
    assert "export PATH=/networkhome/WMGDS/zhou_c/miniconda3/envs/exmesh_official/bin" in runner
    assert "test -f ${INITIAL_OUTPUT}/eval/results.json" in runner
    assert "test -f ${OUTPUT}/eval/results.json" in runner
    assert '--id="${CUDA_VISIBLE_DEVICES' in runner


def test_nds_runner_pins_official_code_and_uses_common_contract() -> None:
    runner = NDS_RUNNER.read_text()
    assert "set -euo pipefail" in runner
    assert "760e4549f59adaed9adf1bd705599786a00ba6b8" in runner
    assert "prepare_exmesh_nds_scene.py" in runner
    assert "--initial_mesh vh32" in runner
    assert "--image_scale 1" in runner
    assert "--iterations 2000" in runner
    assert "evaluate_single_scene.py" in runner
    assert "future_nds/bin:\\${PATH}" in runner


def test_nvdiffrec_runner_uses_exact_camera_overlay_and_official_dmtet() -> None:
    runner = NVDIFFREC_RUNNER.read_text()
    assert "set -euo pipefail" in runner
    assert "abf3a34b1eb6e782abffefc2462c7e9bcd89f9bb" in runner
    assert "prepare_exmesh_nvdiffrec_scene.py" in runner
    assert "run_nvdiffrec_exmesh.py" in runner
    assert "128_tets.npz" in runner
    assert "evaluate_single_scene.py" in runner


def test_projection_decomposition_round_trip() -> None:
    intrinsics = np.asarray(
        [[700.0, 0.0, 400.0], [0.0, 710.0, 300.0], [0.0, 0.0, 1.0]]
    )
    world_to_camera = np.eye(4)
    world_to_camera[:3, 3] = np.asarray([0.2, -0.3, 2.0])
    projection = intrinsics @ world_to_camera[:3]
    found_k, camera_to_world = decompose_projection_matrix(projection)
    np.testing.assert_allclose(found_k, intrinsics, atol=1e-7)
    np.testing.assert_allclose(camera_to_world, np.linalg.inv(world_to_camera), atol=1e-7)


def test_nds_adapter_preserves_rgba_and_camera_projection(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    pixels = np.zeros((3, 4, 4), dtype=np.uint8)
    pixels[..., :3] = 127
    pixels[..., 3] = np.asarray([[0, 0, 255, 255]] * 3, dtype=np.uint8)
    Image.fromarray(pixels, mode="RGBA").save(source)
    intrinsics = np.asarray(
        [[700.0, 0.0, 2.0], [0.0, 710.0, 1.5], [0.0, 0.0, 1.0]]
    )
    world_to_camera = np.eye(4)
    world_to_camera[:3, 3] = [0.2, -0.3, 2.0]
    projection = intrinsics @ world_to_camera[:3]
    world_mat = np.eye(4)
    world_mat[:3] = projection
    contract = {
        "contract_audit": {"valid": True},
        "scenes": [
            {
                "scene_id": 24,
                "initial_mesh": {"path": "mesh.ply", "sha256": "digest"},
                "views": [
                    {
                        "index": 0,
                        "image_id": "0000",
                        "rgb_path": str(source),
                        "mask_source": "rgba_alpha",
                        "resolution_wh": [4, 3],
                        "intrinsics": intrinsics.tolist(),
                        "world_to_camera": world_to_camera.tolist(),
                        "world_mat": world_mat.tolist(),
                        "scale_mat": np.eye(4).tolist(),
                    }
                ],
            }
        ],
    }
    manifest = prepare_nds_scene(contract, 24, tmp_path / "adapted")
    assert manifest["contract_audit"] is True
    assert manifest["image_bytes_unchanged"] is True
    assert manifest["resampling"] is False
    assert (tmp_path / "adapted/views/0000.png").is_symlink()
    np.testing.assert_allclose(
        np.loadtxt(tmp_path / "adapted/views/0000_k.txt"), intrinsics
    )
    np.testing.assert_allclose(
        np.loadtxt(tmp_path / "adapted/views/0000_r.txt"),
        world_to_camera[:3, :3],
    )
    np.testing.assert_allclose(
        np.loadtxt(tmp_path / "adapted/views/0000_t.txt"),
        world_to_camera[:3, 3],
    )


def test_nvdiffrec_exact_projection_preserves_pixels() -> None:
    k = np.asarray(
        [[700.0, 0.0, 446.0], [0.0, 710.0, 338.0], [0.0, 0.0, 1.0]]
    )
    width, height = 800, 600
    projection = nvdiffrec_projection_from_intrinsics(k, [width, height], 0.1, 10.0)
    point_cv = np.asarray([0.2, -0.1, 2.0, 1.0])
    cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0])
    clip = projection @ cv_to_gl @ point_cv
    ndc = clip[:2] / clip[3]
    pixel_from_clip = np.asarray(
        [(ndc[0] + 1.0) * width / 2.0, (ndc[1] + 1.0) * height / 2.0]
    )
    pixel_from_k = (k @ point_cv[:3])[:2] / (k @ point_cv[:3])[2]
    np.testing.assert_allclose(pixel_from_clip, pixel_from_k, atol=1e-10)


def test_nvdiffrec_adapter_preserves_rgba_and_exact_camera(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    Image.fromarray(np.full((3, 4, 4), 255, dtype=np.uint8), mode="RGBA").save(
        source
    )
    k = np.asarray([[7.0, 0.0, 2.2], [0.0, 7.1, 1.7], [0.0, 0.0, 1.0]])
    w2c = np.eye(4)
    contract = {
        "contract_audit": {"valid": True},
        "scenes": [
            {
                "scene_id": 24,
                "views": [
                    {
                        "index": 0,
                        "image_id": "0000",
                        "rgb_path": str(source),
                        "mask_source": "rgba_alpha",
                        "resolution_wh": [4, 3],
                        "intrinsics": k.tolist(),
                        "world_to_camera": w2c.tolist(),
                    }
                ],
            }
        ],
    }
    manifest = prepare_nvdiffrec_scene(contract, 24, tmp_path / "nvdiffrec")
    assert manifest["image_bytes_unchanged"] is True
    assert manifest["resampling"] is False
    transforms = json.loads(
        (tmp_path / "nvdiffrec/transforms_train.json").read_text()
    )
    frame = transforms["frames"][0]
    np.testing.assert_allclose(frame["intrinsics"], k)
    np.testing.assert_allclose(
        frame["world_to_camera_opengl"], np.diag([1.0, -1.0, -1.0, 1.0])
    )


def test_neuralangelo_adapter_uses_native_frame_without_gt(tmp_path: Path) -> None:
    source = tmp_path / "rgba.png"
    Image.fromarray(np.full((3, 4, 4), 255, dtype=np.uint8), mode="RGBA").save(
        source
    )
    k = np.asarray([[7.0, 0.0, 2.2], [0.0, 7.1, 1.7], [0.0, 0.0, 1.0]])
    c2w = np.eye(4)
    contract = {
        "contract_audit": {"valid": True},
        "scenes": [
            {
                "scene_id": 24,
                "views": [
                    {
                        "index": 0,
                        "image_id": "0000",
                        "rgb_path": str(source),
                        "resolution_wh": [4, 3],
                        "intrinsics": k.tolist(),
                        "camera_to_world": c2w.tolist(),
                    }
                ],
            }
        ],
    }
    manifest = prepare_neuralangelo_scene(
        contract, 24, tmp_path / "neuralangelo"
    )
    assert manifest["image_bytes_unchanged"] is True
    assert manifest["evaluation_gt_used_as_method_input"] is False
    transforms = json.loads(
        (tmp_path / "neuralangelo/transforms.json").read_text()
    )
    np.testing.assert_allclose(
        transforms["frames"][0]["transform_matrix"],
        np.diag([1.0, -1.0, -1.0, 1.0]),
    )
    np.testing.assert_allclose(transforms["frames"][0]["intrinsics"], k)


def test_aggregate_requires_reproduction_and_sanity_gates(tmp_path: Path) -> None:
    config = load_suite_config(CONFIG)
    payload = aggregate_results(config, tmp_path)
    assert payload["official_exmesh_reproduction_gate"]["passed"] is False
    assert payload["six_method_sanity_gate"]["passed"] is False
    assert payload["full_benchmark_authorized"] is False
    assert (tmp_path / "summary.csv").is_file()
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "BASELINE_REPORT.md").is_file()
    assert len(payload["methods"]) == len(METHODS)


def test_aggregate_opens_full_gate_only_after_reproduction_and_sanity(
    tmp_path: Path,
) -> None:
    config = load_suite_config(CONFIG)
    paper = config["official_exmesh_protocol"]["paper_overall_cd_mm"]
    for scene_id in config["scene_ids"]:
        _status(
            tmp_path,
            "exmesh_official",
            int(scene_id),
            float(paper[str(scene_id)]),
        )
    for method in (
        "ours",
        "neural_deferred_shading",
        "nvdiffrec",
        "neuralangelo",
        "matcha",
    ):
        _status(tmp_path, method, 24, 0.5)
    payload = aggregate_results(config, tmp_path)
    assert payload["official_exmesh_reproduction_gate"]["passed"] is True
    assert payload["six_method_sanity_gate"]["passed"] is True
    assert payload["full_benchmark_authorized"] is True
    paired = payload["paired_comparisons"]["exmesh_official"]
    assert paired["successful_pairs"] == 1
    assert paired["metrics"]["chamfer"]["right_better"] == 1
    report = (tmp_path / "BASELINE_REPORT.md").read_text()
    assert "Per-scene official evaluator CD" in report
    assert "Paired comparison against ours" in report


def _status(root: Path, method: str, scene_id: int, overall: float) -> None:
    path = root / method / f"scan{scene_id}" / "status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "method": method,
                "scene_id": scene_id,
                "success": True,
                "metrics": {
                    "mean_d2s": overall,
                    "mean_s2d": overall,
                    "overall": overall,
                },
            }
        )
    )
