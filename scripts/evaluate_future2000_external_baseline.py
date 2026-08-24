#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlr.baselines.future2000 import (
    ExternalSceneExport,
    export_nds_scene,
    export_nerf_scene,
    export_nvdiffrec_scene,
    export_openmvs_scene,
)
from mlr.data import Mesh
from mlr.io import load_mesh
from mlr.learned_laplacian.evaluation import evaluate_mesh_geometry
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


NDS_METHODS = ("nds", "nds_28v_full")
METHODS = ("openmvs_refinemesh", *NDS_METHODS, "nerf2mesh", "exmesh", "nvdiffrec")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.method not in METHODS:
        raise ValueError(f"Unknown method {args.method!r}.")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count).")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    method_config = config["methods"][args.method]
    if method_config["commit"] != _git_commit(args.external_root):
        raise ValueError(
            f"{args.method} checkout does not match pinned commit "
            f"{method_config['commit']}."
        )
    dataset = PreparedMeshDataset.from_manifest(args.manifest, "test")
    if len(dataset) != args.expected_test_samples:
        raise ValueError(
            f"Expected {args.expected_test_samples} test samples, found {len(dataset)}."
        )
    manifest_payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    provenance = {
        str(row["sample_id"]): dict(row) for row in manifest_payload["samples"]
    }
    output = args.output_dir.resolve() / args.method
    shard_dir = output / "shards"
    sample_root = output / "samples"
    work_root = output / "work" / f"shard_{args.shard_index:03d}"
    for path in (shard_dir, sample_root, work_root):
        path.mkdir(parents=True, exist_ok=True)
    if args.sample_id is None:
        indices = list(range(args.shard_index, len(dataset), args.shard_count))
    else:
        matches = [index for index, value in enumerate(dataset.sample_ids) if value == args.sample_id]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one --sample-id match, found {matches}")
        # A selected-sample array gives every task a unique shard index while
        # explicitly assigning its sample ID.  Do not apply modulo sharding a
        # second time, otherwise only task zero would execute.
        indices = matches
    preflight_error = _preflight(args)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for ordinal, index in enumerate(indices, start=1):
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
        source = provenance[sample_id]
        sample_dir = sample_root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        status_path = sample_dir / "status.json"
        if status_path.is_file():
            existing = json.loads(status_path.read_text(encoding="utf-8"))
            terminal_statuses = {"completed"}
            if not args.retry_failed:
                terminal_statuses.add("failed")
            if existing.get("status") in terminal_statuses:
                rows.append(existing["row"])
                continue
        if preflight_error is not None:
            row = _failure_row(static, source, args.method, "preflight", preflight_error)
            _save_status(status_path, row, [], preflight_error, method_config)
            rows.append(row)
            continue
        work_dir = work_root / sample_id
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)
        commands: list[list[str]] = []
        try:
            source_identity = _audit_source_identity(static, source)
            scene = _export(args.method, static, work_dir / "scene")
            identity = _audit_initial_identity(static, scene)
            result_mesh_path, commands, runtime, peak_memory = _execute(
                args, scene, work_dir, sample_id, method_config
            )
            result_mesh = load_mesh(result_mesh_path).ensure_normals()
            copied_mesh = sample_dir / "refined.obj"
            from mlr.io import save_mesh

            save_mesh(result_mesh, copied_mesh)
            metrics = _evaluate(static, result_mesh, args)
            row = {
                "sample_id": sample_id,
                "method": args.method,
                "status": "completed",
                "failure_stage": "",
                "failure_reason": "",
                "runtime_seconds": runtime,
                "peak_gpu_memory_mb": peak_memory,
                "vertex_count": result_mesh.num_vertices,
                "face_count": result_mesh.num_faces,
                "final_mesh": str(copied_mesh),
                "coordinate_transform_to_gt": "identity",
                "method_config_path": str(args.config.resolve()),
                **source_identity,
                **identity,
                **metrics,
            }
            shutil.copy2(scene.metadata_path, sample_dir / "input_contract.json")
            (sample_dir / "method_config.json").write_text(
                json.dumps(method_config, indent=2) + "\n", encoding="utf-8"
            )
            _save_status(status_path, row, commands, None, method_config)
        except Exception as exc:
            reason = f"{exc.__class__.__name__}: {exc}"
            row = _failure_row(
                static, source, args.method, "execution_or_evaluation", reason
            )
            for log_path in sorted(work_dir.glob("command_*.log")):
                shutil.copy2(log_path, sample_dir / log_path.name)
            (sample_dir / "traceback.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
            _save_status(status_path, row, commands, reason, method_config)
        finally:
            if not args.keep_work and work_dir.exists():
                shutil.rmtree(work_dir)
        rows.append(row)
        print(
            f"{args.method} shard={args.shard_index} {ordinal}/{len(indices)} "
            f"sample={sample_id} status={row['status']}",
            flush=True,
        )
    csv_path = shard_dir / f"per_sample_shard_{args.shard_index:03d}.csv"
    _write_csv(csv_path, rows)
    metadata = {
        "method": args.method,
        "status": "completed_with_failures"
        if any(row["status"] != "completed" for row in rows)
        else "completed",
        "pinned_commit": method_config["commit"],
        "repository": method_config["repository"],
        "manifest": str(args.manifest),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "assigned_samples": len(rows),
        "completed_samples": sum(row["status"] == "completed" for row in rows),
        "failed_samples": sum(row["status"] == "failed" for row in rows),
        "preflight_error": preflight_error,
        "runtime_seconds": time.perf_counter() - started,
        "config": method_config,
        "csv": str(csv_path),
    }
    (shard_dir / f"metadata_shard_{args.shard_index:03d}.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def _preflight(args: argparse.Namespace) -> str | None:
    missing = []
    if args.method == "openmvs_refinemesh":
        for executable in (args.interface_colmap, args.refine_mesh):
            if executable is None or not Path(executable).is_file():
                missing.append(str(executable))
    else:
        if args.external_python is None or not Path(args.external_python).is_file():
            missing.append(str(args.external_python))
        entry = args.external_root / (
            "reconstruct.py"
            if args.method in NDS_METHODS
            else "main.py"
            if args.method == "nerf2mesh"
            else "train.py"
        )
        if not entry.is_file():
            missing.append(str(entry))
    if args.method == "exmesh" and args.exmesh_depth_root is None:
        missing.append("--exmesh-depth-root")
    return None if not missing else "Missing required runtime inputs: " + ", ".join(missing)


def _export(method: str, sample: dict[str, Any], scene_dir: Path) -> ExternalSceneExport:
    if method == "openmvs_refinemesh":
        return export_openmvs_scene(sample, scene_dir)
    if method in NDS_METHODS:
        return export_nds_scene(sample, scene_dir)
    if method == "nvdiffrec":
        return export_nvdiffrec_scene(sample, scene_dir)
    return export_nerf_scene(sample, scene_dir, method=method)


def _execute(
    args: argparse.Namespace,
    scene: ExternalSceneExport,
    work_dir: Path,
    sample_id: str,
    method_config: dict[str, Any],
) -> tuple[Path, list[list[str]], float, float | None]:
    commands: list[list[str]] = []
    started = time.perf_counter()
    peak = 0.0
    if args.method == "openmvs_refinemesh":
        mvs = work_dir / "scene.mvs"
        output_mvs = work_dir / "refined.mvs"
        commands = [
            [
                str(args.interface_colmap),
                "-w",
                str(work_dir),
                "-i",
                str(scene.scene_dir / "colmap/sparse"),
                "-o",
                str(mvs),
                "--image-folder",
                str(scene.scene_dir / "colmap/images"),
            ],
            [
                str(args.refine_mesh),
                "-w",
                str(work_dir),
                "-i",
                str(mvs),
                "-m",
                str(scene.initial_ply),
                "-o",
                str(output_mvs),
                "--export-type",
                "ply",
                "--resolution-level",
                "0",
                "--min-resolution",
                "640",
                "--max-views",
                "8",
                "--decimate",
                "1",
                "--close-holes",
                "0",
                "--ensure-edge-size",
                "1",
                "--max-face-area",
                "16",
                "--scales",
                "2",
                "--scale-step",
                "0.5",
                "--regularity-weight",
                "0.2",
                "--rigidity-elasticity-ratio",
                "0.9",
                "--gradient-step",
                "45.05",
                "--reduce-memory",
                "1",
            ],
        ]
        for number, command in enumerate(commands):
            peak = max(peak, _run_command(command, work_dir / f"command_{number}.log"))
        result = work_dir / "refined.ply"
    elif args.method in NDS_METHODS:
        output = work_dir / "nds_output"
        nds_arguments = method_config["arguments"]
        command = [
            str(args.external_python),
            str(args.external_root / "reconstruct.py"),
            "--input_dir",
            str(scene.scene_dir / "views"),
            "--input_bbox",
            str(scene.scene_dir / "bbox.txt"),
            "--output_dir",
            str(output),
            "--initial_mesh",
            str(scene.initial_obj),
            "--iterations",
            str(nds_arguments["iterations"]),
            "--image_scale",
            str(nds_arguments.get("image_scale", 1)),
            "--view_sampling_mode",
            str(nds_arguments.get("view_sampling_mode", "random")),
            "--views_per_iter",
            str(nds_arguments.get("views_per_iter", 1)),
            "--lr_vertices",
            str(nds_arguments.get("lr_vertices", 0.001)),
            "--lr_shader",
            str(nds_arguments.get("lr_shader", 0.001)),
            "--upsample_iterations",
            *[str(value) for value in nds_arguments.get("upsample_iterations", [])],
            "--weight_mask",
            str(nds_arguments.get("weight_mask", 2.0)),
            "--weight_normal",
            str(nds_arguments.get("weight_normal", 0.1)),
            "--weight_laplacian",
            str(nds_arguments.get("weight_laplacian", 40.0)),
            "--weight_shading",
            str(nds_arguments.get("weight_shading", 1.0)),
            "--run_name",
            sample_id,
            "--device",
            "0",
        ]
        commands = [command]
        peak = _run_command(command, work_dir / "command_0.log", cwd=args.external_root)
        iterations = int(method_config["arguments"]["iterations"])
        result = output / sample_id / "meshes" / f"mesh_{iterations:06d}.obj"
    elif args.method == "nerf2mesh":
        workspace = work_dir / "nerf2mesh_workspace"
        common = [
            str(args.external_python),
            str(args.external_root / "main.py"),
            str(scene.scene_dir),
            "--workspace",
            str(workspace),
            "-O",
            "--bound",
            "1",
            "--scale",
            "1",
            "--sdf",
            "--dt_gamma",
            "0",
        ]
        commands = [
            common + ["--stage", "0", "--iters", "10000", "--decimate_target", "100000", "--lambda_tv", "1e-8"],
            common
            + [
                "--stage",
                "1",
                "--iters",
                "5000",
                "--mesh",
                str(scene.initial_ply),
                "--lambda_normal",
                "0.01",
                "--refine_remesh_size",
                "0.01",
            ],
        ]
        for number, command in enumerate(commands):
            peak = max(
                peak,
                _run_command(command, work_dir / f"command_{number}.log", cwd=args.external_root),
            )
        result = workspace / "mesh_stage1/mesh.obj"
    elif args.method == "exmesh":
        prior_source = args.exmesh_depth_root / sample_id.split("__v", 1)[0]
        prior_destination = scene.scene_dir / "mono_priors/da3"
        if not prior_source.is_dir():
            raise FileNotFoundError(f"Missing RGB-only DA3 prior directory: {prior_source}")
        prior_destination.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(prior_source.resolve(), prior_destination)
        output = work_dir / "exmesh_output"
        command = [
            str(args.external_python),
            str(args.external_root / "train.py"),
            "-s",
            str(scene.scene_dir),
            "-m",
            str(output),
            "--iterations",
            str(method_config["arguments"]["iterations"]),
            "--save_iterations",
            str(method_config["arguments"]["iterations"]),
            "--resolution",
            str(method_config["arguments"].get("resolution", 1)),
            "--quiet",
        ]
        commands = [command]
        peak = _run_command(command, work_dir / "command_0.log", cwd=args.external_root)
        iterations = int(method_config["arguments"]["iterations"])
        result = output / "mesh" / f"auto_mesh_iter_{iterations}.ply"
    else:
        # Official nvdiffrec prefixes its out_dir with ``out/``.  Run it from
        # the isolated per-sample work directory so the result stays scoped to
        # that sample and is cleaned with the work directory.  Its renderer
        # also loads the checked-in BSDF LUT through a hard-coded relative
        # ``data/`` path, so expose that official read-only resource here.
        os.symlink(args.external_root / "data", work_dir / "data")
        output = work_dir / "out/nvdiffrec_output"
        config_path = work_dir / "nvdiffrec_config.json"
        config = {
            "ref_mesh": str(scene.scene_dir),
            "base_mesh": str(scene.initial_obj),
            "random_textures": True,
            "iter": int(method_config["arguments"]["iterations"]),
            "save_interval": int(method_config["arguments"].get("save_interval", 100)),
            "texture_res": list(method_config["arguments"].get("texture_res", [2048, 2048])),
            "train_res": list(method_config["arguments"].get("train_res", [1920, 1920])),
            "batch": int(method_config["arguments"].get("batch", 1)),
            "spp": int(method_config["arguments"].get("spp", 1)),
            "learning_rate": list(method_config["arguments"].get("learning_rate", [0.03, 0.01])),
            "ks_min": [0, 0.08, 0.0],
            "dmtet_grid": 128,
            "mesh_scale": 2.1,
            "laplace_scale": int(method_config["arguments"].get("laplace_scale", 3000)),
            "background": "white",
            "loss": "logl1",
            "pre_load": True,
            "validate": False,
            "out_dir": "nvdiffrec_output",
        }
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        command = [
            str(args.external_python),
            str(ROOT / "scripts/run_nvdiffrec_exmesh.py"),
            "--seed",
            str(method_config.get("seed", 7)),
            "--nvdiffrec-root",
            str(args.external_root),
            "--config",
            str(config_path),
        ]
        commands = [command]
        peak = _run_command(command, work_dir / "command_0.log", cwd=work_dir)
        result = output / "mesh/mesh.obj"
    if not result.is_file():
        raise FileNotFoundError(f"Expected external result mesh was not produced: {result}")
    return result, commands, time.perf_counter() - started, peak or None


def _run_command(
    command: list[str], log_path: Path, cwd: Path | None = None
) -> float:
    with log_path.open("w", encoding="utf-8") as log:
        log.write("command=" + shlex.join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        peak = [0.0]
        stop = threading.Event()
        monitor = threading.Thread(
            target=_monitor_gpu, args=(stop, peak, process.pid), daemon=True
        )
        monitor.start()
        return_code = process.wait()
        stop.set()
        monitor.join(timeout=2.0)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    return peak[0]


def _monitor_gpu(stop: threading.Event, peak: list[float], root_pid: int) -> None:
    while not stop.wait(0.25):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            owned = _process_tree(root_pid)
            values = []
            for line in result.stdout.splitlines():
                fields = [value.strip() for value in line.split(",")]
                if len(fields) != 2 or int(fields[0]) not in owned:
                    continue
                values.append(float(fields[1]))
            if values:
                peak[0] = max(peak[0], sum(values))
        except (OSError, ValueError):
            return


def _process_tree(root_pid: int) -> set[int]:
    parents: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            lines = (entry / "status").read_text(encoding="utf-8").splitlines()
            parent = next(line for line in lines if line.startswith("PPid:"))
            parents[int(entry.name)] = int(parent.split()[1])
        except (OSError, StopIteration, ValueError):
            continue
    owned = {int(root_pid)}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in owned and pid not in owned:
                owned.add(pid)
                changed = True
    return owned


def _evaluate(
    static: dict[str, Any], result_mesh: Mesh, args: argparse.Namespace
) -> dict[str, Any]:
    gt = Mesh(
        static["gt_vertices"].detach().cpu().numpy(),
        static["gt_faces"].detach().cpu().numpy(),
    ).ensure_normals()
    initial = Mesh(
        static["vertices"].detach().cpu().numpy(),
        static["faces"].detach().cpu().numpy(),
    ).ensure_normals()
    before = evaluate_mesh_geometry(
        initial,
        gt,
        surface_samples=args.surface_samples,
        seed=args.metric_seed,
        fscore_threshold=args.fscore_threshold,
    )
    after = evaluate_mesh_geometry(
        result_mesh,
        gt,
        surface_samples=args.surface_samples,
        seed=args.metric_seed,
        fscore_threshold=args.fscore_threshold,
    )
    fields = {
        "chamfer": "chamfer",
        "p2s_mean": "point_to_surface_bidirectional_mean",
        "p2s_p95": "point_to_surface_bidirectional_p95",
        "fscore": "fscore",
        "normal_consistency": "normal_consistency",
    }
    result = {}
    for short, name in fields.items():
        result[f"initial_{short}"] = before[name]
        result[f"refined_{short}"] = after[name]
    initial_chamfer = float(before["chamfer"])
    refined_chamfer = float(after["chamfer"])
    result["chamfer_improvement_rate"] = (
        (initial_chamfer - refined_chamfer) / initial_chamfer
        if initial_chamfer > 0
        else 0.0
    )
    result["improved"] = refined_chamfer < initial_chamfer
    same_connectivity = bool(
        result_mesh.num_vertices == initial.num_vertices
        and result_mesh.num_faces == initial.num_faces
        and np.array_equal(np.asarray(result_mesh.faces), np.asarray(initial.faces))
    )
    result["output_connectivity_preserved"] = same_connectivity
    if same_connectivity:
        from mlr.learned_laplacian.canonical_experiment import _topology_change

        topology = _topology_change(initial.vertices, result_mesh.vertices, initial.faces)
        result["introduced_flipped_faces"] = int(topology["introduced_flips"])
        result["new_degenerate_faces"] = int(topology["new_degeneracies"])
        result["introduced_flipped_faces_comparable"] = True
    else:
        result["introduced_flipped_faces"] = None
        result["new_degenerate_faces"] = None
        result["introduced_flipped_faces_comparable"] = False
    return result


def _audit_initial_identity(
    static: dict[str, Any], scene: ExternalSceneExport
) -> dict[str, Any]:
    exported = load_mesh(scene.initial_obj)
    vertices = static["vertices"].detach().cpu().numpy()
    faces = static["faces"].detach().cpu().numpy()
    counts_match = exported.num_vertices == len(vertices) and exported.num_faces == len(faces)
    faces_match = counts_match and np.array_equal(np.asarray(exported.faces), faces)
    max_error = (
        float(np.max(np.abs(np.asarray(exported.vertices) - vertices)))
        if counts_match
        else float("inf")
    )
    passed = bool(counts_match and faces_match and max_error <= 1e-6)
    if not passed:
        raise RuntimeError(
            f"Adapter replaced the common initial mesh: counts={counts_match} "
            f"faces={faces_match} max_vertex_error={max_error}"
        )
    return {
        "adapter_initial_mesh_sha256": _sha256_file(scene.initial_obj),
        "adapter_initial_vertex_count": exported.num_vertices,
        "adapter_initial_face_count": exported.num_faces,
        "adapter_initial_max_abs_vertex_error": max_error,
        "adapter_initial_faces_exact": True,
        "common_initial_identity_audit": True,
    }


def _audit_source_identity(
    static: dict[str, Any], source: dict[str, Any]
) -> dict[str, Any]:
    if "common_initial_mesh" not in source:
        vertices = static["vertices"].detach().cpu().numpy()
        faces = static["faces"].detach().cpu().numpy()
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(vertices).tobytes())
        digest.update(np.ascontiguousarray(faces).tobytes())
        dataset_root = Path(str(static["_dataset_root"])).resolve()
        container = (dataset_root / str(source["path"])).resolve()
        image_paths = []
        for value in static["image_paths"]:
            path = Path(str(value))
            image_paths.append((path if path.is_absolute() else dataset_root / path).resolve())
        image_parents = {path.parent for path in image_paths}
        if len(image_paths) != 28:
            raise RuntimeError("Future2000 source does not have exactly 28 image paths")
        return {
            "common_initial_mesh": f"{container}::vertices/faces",
            "common_initial_mesh_sha256": digest.hexdigest(),
            "initial_vertex_count": len(vertices),
            "initial_face_count": len(faces),
            "common_initial_source_identity_audit": True,
            "image_directory": ";".join(str(path) for path in sorted(image_parents)),
            "view_count": len(image_paths),
            "camera_and_gt_container": str(container),
        }
    initial_path = Path(str(source["common_initial_mesh"]))
    expected_sha = str(source["common_initial_mesh_sha256"])
    observed_sha = _sha256_file(initial_path)
    if observed_sha != expected_sha:
        raise RuntimeError(
            f"Common initial mesh SHA changed: expected={expected_sha} observed={observed_sha}"
        )
    initial = load_mesh(initial_path)
    vertices = static["vertices"].detach().cpu().numpy()
    faces = static["faces"].detach().cpu().numpy()
    counts_match = initial.num_vertices == len(vertices) and initial.num_faces == len(faces)
    faces_match = counts_match and np.array_equal(np.asarray(initial.faces), faces)
    max_error = (
        float(np.max(np.abs(np.asarray(initial.vertices) - vertices)))
        if counts_match
        else float("inf")
    )
    if not (counts_match and faces_match and max_error <= 1e-6):
        raise RuntimeError(
            f"Canonical common initial changed: counts={counts_match} "
            f"faces={faces_match} max_vertex_error={max_error}"
        )
    return {
        "common_initial_mesh": str(initial_path),
        "common_initial_mesh_sha256": observed_sha,
        "initial_vertex_count": initial.num_vertices,
        "initial_face_count": initial.num_faces,
        "common_initial_source_identity_audit": True,
        "image_directory": str(source["image_directory"]),
        "view_count": int(source["view_count"]),
        "camera_and_gt_container": str(source["camera_and_gt_container"]),
    }


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _failure_row(
    static: dict[str, Any],
    source: dict[str, Any] | str,
    method: str,
    stage: str,
    reason: str | None = None,
) -> dict[str, Any]:
    # Preserve the historical helper call used by lightweight unit tests and
    # older local tooling: _failure_row(static, method, stage, reason).
    if reason is None:
        reason = stage
        stage = method
        method = str(source)
        source = {}
    return {
        "sample_id": str(static["sample_id"]),
        "method": method,
        "status": "failed",
        "failure_stage": stage,
        "failure_reason": reason,
        "runtime_seconds": "",
        "peak_gpu_memory_mb": "",
        "vertex_count": "",
        "face_count": "",
        "final_mesh": "",
        "coordinate_transform_to_gt": "",
        "method_config_path": "",
        "common_initial_mesh": str(source.get("common_initial_mesh", "")),
        "common_initial_mesh_sha256": str(
            source.get("common_initial_mesh_sha256", "")
        ),
        "initial_vertex_count": source.get("initial_vertex_count", ""),
        "initial_face_count": source.get("initial_face_count", ""),
        "common_initial_source_identity_audit": "",
        "image_directory": str(source.get("image_directory", "")),
        "view_count": source.get("view_count", ""),
        "camera_and_gt_container": str(source.get("camera_and_gt_container", "")),
        "adapter_initial_mesh_sha256": "",
        "adapter_initial_vertex_count": "",
        "adapter_initial_face_count": "",
        "adapter_initial_max_abs_vertex_error": "",
        "adapter_initial_faces_exact": "",
        "common_initial_identity_audit": "",
        "initial_chamfer": "",
        "refined_chamfer": "",
        "chamfer_improvement_rate": "",
        "initial_p2s_mean": "",
        "refined_p2s_mean": "",
        "initial_p2s_p95": "",
        "refined_p2s_p95": "",
        "initial_fscore": "",
        "refined_fscore": "",
        "initial_normal_consistency": "",
        "refined_normal_consistency": "",
        "improved": "",
        "output_connectivity_preserved": "",
        "introduced_flipped_faces": "",
        "new_degenerate_faces": "",
        "introduced_flipped_faces_comparable": "",
    }


def _save_status(
    path: Path,
    row: dict[str, Any],
    commands: list[list[str]],
    error: str | None,
    method_config: dict[str, Any],
) -> None:
    path.write_text(
        json.dumps(
            {
                "status": row["status"],
                "row": row,
                "commands": [shlex.join(command) for command in commands],
                "error": error,
                "method_config": method_config,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("External evaluation shard has no assigned samples.")
    fieldnames = list(rows[0])
    known = set(fieldnames)
    for row in rows[1:]:
        for key in row:
            if key not in known:
                fieldnames.append(key)
                known.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--external-python", type=Path)
    parser.add_argument("--interface-colmap", type=Path)
    parser.add_argument("--refine-mesh", type=Path)
    parser.add_argument("--exmesh-depth-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--surface-samples", type=int, default=3000)
    parser.add_argument("--metric-seed", type=int, default=7)
    parser.add_argument("--fscore-threshold", type=float, default=0.01)
    parser.add_argument("--expected-test-samples", type=int, default=1000)
    parser.add_argument("--sample-id")
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-run samples whose existing status.json is failed.",
    )
    parser.add_argument(
        "--fail-on-sample-error",
        action="store_true",
        help="Return a non-zero process status when any assigned sample fails.",
    )
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2))
    return 2 if args.fail_on_sample_error and result["failed_samples"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
