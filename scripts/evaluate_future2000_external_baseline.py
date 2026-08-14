#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
    export_openmvs_scene,
)
from mlr.data import Mesh
from mlr.io import load_mesh
from mlr.learned_laplacian.evaluation import evaluate_mesh_geometry
from mlr.learned_laplacian.multi_dataset import PreparedMeshDataset


METHODS = ("openmvs_refinemesh", "nds", "nerf2mesh", "exmesh")


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
    if len(dataset) != 1000:
        raise ValueError(f"Expected 1000 test samples, found {len(dataset)}.")
    output = args.output_dir.resolve() / args.method
    shard_dir = output / "shards"
    sample_root = output / "samples"
    work_root = output / "work" / f"shard_{args.shard_index:03d}"
    for path in (shard_dir, sample_root, work_root):
        path.mkdir(parents=True, exist_ok=True)
    indices = list(range(args.shard_index, len(dataset), args.shard_count))
    preflight_error = _preflight(args)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for ordinal, index in enumerate(indices, start=1):
        static = dataset.load_static(index)
        sample_id = str(static["sample_id"])
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
            row = _failure_row(static, args.method, "preflight", preflight_error)
            _save_status(status_path, row, [], preflight_error)
            rows.append(row)
            continue
        work_dir = work_root / sample_id
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True)
        commands: list[list[str]] = []
        try:
            scene = _export(args.method, static, work_dir / "scene")
            result_mesh_path, commands, runtime, peak_memory = _execute(
                args, scene, work_dir, sample_id
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
                **metrics,
            }
            shutil.copy2(scene.metadata_path, sample_dir / "input_contract.json")
            _save_status(status_path, row, commands, None)
        except Exception as exc:
            reason = f"{exc.__class__.__name__}: {exc}"
            row = _failure_row(static, args.method, "execution_or_evaluation", reason)
            (sample_dir / "traceback.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
            _save_status(status_path, row, commands, reason)
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
        entry = args.external_root / ("reconstruct.py" if args.method == "nds" else "main.py" if args.method == "nerf2mesh" else "train.py")
        if not entry.is_file():
            missing.append(str(entry))
    if args.method == "exmesh" and args.exmesh_depth_root is None:
        missing.append("--exmesh-depth-root")
    return None if not missing else "Missing required runtime inputs: " + ", ".join(missing)


def _export(method: str, sample: dict[str, Any], scene_dir: Path) -> ExternalSceneExport:
    if method == "openmvs_refinemesh":
        return export_openmvs_scene(sample, scene_dir)
    if method == "nds":
        return export_nds_scene(sample, scene_dir)
    return export_nerf_scene(sample, scene_dir, method=method)


def _execute(
    args: argparse.Namespace,
    scene: ExternalSceneExport,
    work_dir: Path,
    sample_id: str,
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
    elif args.method == "nds":
        output = work_dir / "nds_output"
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
            "2000",
            "--run_name",
            sample_id,
            "--device",
            "0",
        ]
        commands = [command]
        peak = _run_command(command, work_dir / "command_0.log", cwd=args.external_root)
        result = output / sample_id / "meshes/mesh_002000.obj"
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
    else:
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
            "10000",
            "--save_iterations",
            "10000",
            "--quiet",
        ]
        commands = [command]
        peak = _run_command(command, work_dir / "command_0.log", cwd=args.external_root)
        result = output / "mesh/auto_mesh_iter_10000.ply"
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
        monitor = threading.Thread(target=_monitor_gpu, args=(stop, peak), daemon=True)
        monitor.start()
        return_code = process.wait()
        stop.set()
        monitor.join(timeout=2.0)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    return peak[0]


def _monitor_gpu(stop: threading.Event, peak: list[float]) -> None:
    while not stop.wait(0.25):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            values = [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]
            if values:
                peak[0] = max(peak[0], sum(values))
        except (OSError, ValueError):
            return


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
    return result


def _failure_row(
    static: dict[str, Any], method: str, stage: str, reason: str
) -> dict[str, Any]:
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
    }


def _save_status(
    path: Path, row: dict[str, Any], commands: list[list[str]], error: str | None
) -> None:
    path.write_text(
        json.dumps(
            {
                "status": row["status"],
                "row": row,
                "commands": [shlex.join(command) for command in commands],
                "error": error,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("External evaluation shard has no assigned samples.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
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
    parser.add_argument("--keep-work", action="store_true")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-run samples whose existing status.json is failed.",
    )
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
