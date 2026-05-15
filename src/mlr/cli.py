from __future__ import annotations

import argparse
import json
from pathlib import Path

from .coarse import (
    ExistingMeshGenerator,
    NvidiaInstantNGPMeshGenerator,
    NvidiaNvdiffrecMeshGenerator,
    generate_coarse_mesh,
)
from .datasets import load_masks, load_reconstruction_input
from .gt_laplacian import GTLaplacianTargetConfig, refine_coarse_mesh_with_gt_laplacian
from .io import load_mesh, save_mesh
from .metrics import correspondence_metrics
from .nvdiffrec import NvdiffrecRunConfig, prepare_nvdiffrec_run
from .oracle import OracleBaselineConfig, run_oracle_baselines
from .refinement import RefinementConfig
from .synthetic import (
    SyntheticRenderConfig,
    generate_synthetic_dataset_from_mesh,
    generate_synthetic_datasets_from_mesh_dir,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mlr")
    sub = parser.add_subparsers(dest="command", required=True)

    coarse = sub.add_parser("coarse", help="Normalize/import an existing coarse mesh.")
    coarse.add_argument("--mesh", required=True, type=Path)
    coarse.add_argument("--out", required=True, type=Path)

    coarse_ngp = sub.add_parser("coarse-ngp", help="Generate a coarse mesh with an Instant-NGP command wrapper.")
    coarse_ngp.add_argument("--dataset", required=True, type=Path)
    coarse_ngp.add_argument("--scene-dir", required=True, type=Path)
    coarse_ngp.add_argument("--out", required=True, type=Path)
    coarse_ngp.add_argument("--command-template", required=True)
    coarse_ngp.add_argument("--aabb-scale", default=16, type=int)
    coarse_ngp.add_argument("--no-cv-to-gl", action="store_true")
    coarse_ngp.add_argument("--no-visibility", action="store_true")

    coarse_nvdiffrec = sub.add_parser("coarse-nvdiffrec", help="Generate a coarse mesh with NVIDIA nvdiffrec.")
    coarse_nvdiffrec.add_argument("--dataset", required=True, type=Path)
    coarse_nvdiffrec.add_argument("--nvdiffrec-root", required=True, type=Path)
    coarse_nvdiffrec.add_argument("--run-dir", required=True, type=Path)
    coarse_nvdiffrec.add_argument("--out", required=True, type=Path)
    coarse_nvdiffrec.add_argument("--command-template", default='python train.py --config "{config_path}"')
    coarse_nvdiffrec.add_argument("--result-mesh", type=Path)
    coarse_nvdiffrec.add_argument("--iters", default=1000, type=int)
    coarse_nvdiffrec.add_argument("--save-interval", default=100, type=int)
    coarse_nvdiffrec.add_argument("--train-res", default=512, type=int)
    coarse_nvdiffrec.add_argument("--texture-res", default=1024, type=int)
    coarse_nvdiffrec.add_argument("--batch", default=4, type=int)
    coarse_nvdiffrec.add_argument("--dmtet-grid", default=64, type=int)
    coarse_nvdiffrec.add_argument("--mesh-scale", default=2.4, type=float)
    coarse_nvdiffrec.add_argument("--laplace-scale", default=3000.0, type=float)
    coarse_nvdiffrec.add_argument("--background", default="white", choices=["white", "black", "checker", "reference", "random"])
    coarse_nvdiffrec.add_argument("--isosurface", choices=["dmtet", "flexicubes"])
    coarse_nvdiffrec.add_argument("--no-cv-to-gl", action="store_true")
    coarse_nvdiffrec.add_argument("--no-visibility", action="store_true")
    coarse_nvdiffrec.add_argument("--prepare-only", action="store_true")

    oracle = sub.add_parser("oracle", help="Run known-topology Laplacian oracle baselines.")
    oracle.add_argument("--init-mesh", required=True, type=Path)
    oracle.add_argument("--gt-mesh", required=True, type=Path)
    oracle.add_argument("--out-dir", required=True, type=Path)
    oracle.add_argument("--operator", default="uniform", choices=["uniform", "cotangent"])
    oracle.add_argument("--iters", default=300, type=int)
    oracle.add_argument("--lr", default=5e-3, type=float)
    oracle.add_argument("--lambda-lap", default=1.0, type=float)
    oracle.add_argument("--lambda-anchor", default=0.05, type=float)
    oracle.add_argument("--noise-sigma", default=0.01, type=float)

    gt_lap = sub.add_parser(
        "gt-laplacian-refine",
        help="Refine a coarse mesh with GT Laplacian values interpolated from a GT surface.",
    )
    gt_lap.add_argument("--coarse-mesh", required=True, type=Path)
    gt_lap.add_argument("--gt-mesh", required=True, type=Path)
    gt_lap.add_argument("--out", required=True, type=Path)
    gt_lap.add_argument("--history-out", type=Path)
    gt_lap.add_argument("--operator", default="uniform", choices=["uniform", "cotangent"])
    gt_lap.add_argument("--iters", default=300, type=int)
    gt_lap.add_argument("--lr", default=5e-3, type=float)
    gt_lap.add_argument("--lambda-lap", default=1.0, type=float)
    gt_lap.add_argument("--lambda-anchor", default=0.05, type=float)
    gt_lap.add_argument("--lambda-edge", default=0.0, type=float)
    gt_lap.add_argument("--robust-loss", default="charbonnier", choices=["charbonnier", "huber", "l2"])
    gt_lap.add_argument("--distance-confidence-scale", type=float)
    gt_lap.add_argument("--min-confidence", default=0.0, type=float)
    gt_lap.add_argument("--log-every", default=25, type=int)

    synthetic = sub.add_parser("synthetic", help="Render multi-view synthetic inputs from a mesh.")
    synthetic_input = synthetic.add_mutually_exclusive_group(required=True)
    synthetic_input.add_argument("--mesh", type=Path)
    synthetic_input.add_argument("--mesh-dir", type=Path)
    synthetic.add_argument("--out-dir", required=True, type=Path)
    synthetic.add_argument("--views", default=24, type=int)
    synthetic.add_argument("--width", default=512, type=int)
    synthetic.add_argument("--height", default=512, type=int)
    synthetic.add_argument("--trajectory", default="orbit", choices=["orbit", "sphere"])
    synthetic.add_argument("--radius-scale", default=2.5, type=float)
    synthetic.add_argument("--elevation", default=20.0, type=float)
    synthetic.add_argument("--min-elevation", default=-60.0, type=float)
    synthetic.add_argument("--max-elevation", default=60.0, type=float)
    synthetic.add_argument("--fov", default=50.0, type=float)
    synthetic.add_argument("--mode", default="lit", choices=["lit", "normal", "depth"])
    synthetic.add_argument("--backend", default="cpu", choices=["cpu", "opengl"])
    synthetic.add_argument("--no-normalize", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "coarse":
        return _run_coarse(args)
    if args.command == "coarse-ngp":
        return _run_coarse_ngp(args)
    if args.command == "coarse-nvdiffrec":
        return _run_coarse_nvdiffrec(args)
    if args.command == "oracle":
        return _run_oracle(args)
    if args.command == "gt-laplacian-refine":
        return _run_gt_laplacian_refine(args)
    if args.command == "synthetic":
        return _run_synthetic(args)
    raise ValueError(args.command)


def _run_coarse(args: argparse.Namespace) -> int:
    mesh = generate_coarse_mesh([], [], method=ExistingMeshGenerator(args.mesh, compute_visibility=False))
    save_mesh(mesh, args.out)
    print(f"Wrote normalized coarse mesh to {args.out}")
    return 0


def _run_coarse_ngp(args: argparse.Namespace) -> int:
    data = load_reconstruction_input(args.dataset)
    masks = load_masks(data.mask_paths)
    backend = NvidiaInstantNGPMeshGenerator(
        scene_dir=args.scene_dir,
        output_mesh_path=args.out,
        command_template=args.command_template,
        aabb_scale=args.aabb_scale,
        convert_cv_to_gl=not args.no_cv_to_gl,
        compute_visibility=not args.no_visibility,
    )
    print(f"Loaded dataset with {len(data.image_paths)} images from {args.dataset}", flush=True)
    print(f"Writing Instant-NGP transforms into {args.scene_dir}", flush=True)
    print(f"Expected coarse mesh output: {args.out}", flush=True)
    mesh = generate_coarse_mesh(data.image_paths, data.cameras, masks=masks, method=backend)
    print(
        json.dumps(
            {
                "coarse_mesh": str(args.out),
                "vertices": mesh.num_vertices,
                "faces": mesh.num_faces,
                "backend": mesh.attributes.get("coarse_backend", "nvidia_instant_ngp"),
            },
            indent=2,
        )
    )
    return 0


def _run_coarse_nvdiffrec(args: argparse.Namespace) -> int:
    data = load_reconstruction_input(args.dataset)
    masks = load_masks(data.mask_paths)
    run_config = NvdiffrecRunConfig(
        iterations=args.iters,
        save_interval=args.save_interval,
        texture_res=(args.texture_res, args.texture_res),
        train_res=(args.train_res, args.train_res),
        batch=args.batch,
        dmtet_grid=args.dmtet_grid,
        mesh_scale=args.mesh_scale,
        laplace_scale=args.laplace_scale,
        background=args.background,
        isosurface=args.isosurface,
    )
    if args.prepare_only:
        prepared = prepare_nvdiffrec_run(
            data,
            run_dir=args.run_dir,
            out_name="coarse",
            config=run_config,
            convert_cv_to_gl=not args.no_cv_to_gl,
            masks=masks,
        )
        print(
            json.dumps(
                {
                    "mode": "prepare_only",
                    "nerf_dataset_dir": str(prepared.nerf_dataset_dir),
                    "config_path": str(prepared.config_path),
                    "nvdiffrec_output": str(prepared.nvdiffrec_out_dir),
                    "next_command": args.command_template.format(
                        config_path=str(prepared.config_path),
                        dataset_dir=str(prepared.nerf_dataset_dir),
                        out_dir=str(prepared.nvdiffrec_out_dir),
                        output_mesh_path=str(args.out),
                        nvdiffrec_root=str(args.nvdiffrec_root),
                    ),
                },
                indent=2,
            )
        )
        return 0

    backend = NvidiaNvdiffrecMeshGenerator(
        run_dir=args.run_dir,
        output_mesh_path=args.out,
        nvdiffrec_root=args.nvdiffrec_root,
        command_template=args.command_template,
        result_mesh_path=args.result_mesh,
        run_config=run_config,
        convert_cv_to_gl=not args.no_cv_to_gl,
        compute_visibility=not args.no_visibility,
    )
    print(f"Loaded dataset with {len(data.image_paths)} images from {args.dataset}", flush=True)
    print(f"Preparing nvdiffrec run directory: {args.run_dir}", flush=True)
    print(f"Using nvdiffrec root: {args.nvdiffrec_root}", flush=True)
    mesh = generate_coarse_mesh(data.image_paths, data.cameras, masks=masks, method=backend)
    print(
        json.dumps(
            {
                "coarse_mesh": str(args.out),
                "vertices": mesh.num_vertices,
                "faces": mesh.num_faces,
                "backend": mesh.attributes.get("coarse_backend", "nvidia_nvdiffrec"),
                "nvdiffrec_config": mesh.attributes.get("nvdiffrec_config"),
                "nvdiffrec_output": mesh.attributes.get("nvdiffrec_output"),
            },
            indent=2,
        )
    )
    return 0


def _run_oracle(args: argparse.Namespace) -> int:
    init_mesh = load_mesh(args.init_mesh)
    gt_mesh = load_mesh(args.gt_mesh)
    config = OracleBaselineConfig(
        operator_type=args.operator,
        lambda_lap=args.lambda_lap,
        lambda_anchor=args.lambda_anchor,
        noisy_laplacian_sigma=args.noise_sigma,
        num_iters=args.iters,
        learning_rate=args.lr,
    )
    results = run_oracle_baselines(init_mesh, gt_mesh.vertices, config)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {}
    for name, result in results.items():
        save_mesh(result.mesh, args.out_dir / f"{name}.obj")
        metrics[name] = {
            **correspondence_metrics(result.vertices, gt_mesh.vertices),
            "final_loss": result.history[-1]["loss"] if result.history else None,
        }
    metrics["init"] = correspondence_metrics(init_mesh.vertices, gt_mesh.vertices)
    with (args.out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))
    return 0


def _run_gt_laplacian_refine(args: argparse.Namespace) -> int:
    coarse_mesh = load_mesh(args.coarse_mesh)
    gt_mesh = load_mesh(args.gt_mesh)
    target_config = GTLaplacianTargetConfig(
        operator_type=args.operator,
        distance_confidence_scale=args.distance_confidence_scale,
        min_confidence=args.min_confidence,
    )
    refinement_config = RefinementConfig(
        operator_type=args.operator,
        lambda_lap=args.lambda_lap,
        lambda_anchor=args.lambda_anchor,
        lambda_edge=args.lambda_edge,
        num_iters=args.iters,
        learning_rate=args.lr,
        robust_loss=args.robust_loss,
        log_every=args.log_every,
    )
    result = refine_coarse_mesh_with_gt_laplacian(
        coarse_mesh,
        gt_mesh,
        target_config=target_config,
        refinement_config=refinement_config,
        anchors=coarse_mesh.vertices,
    )
    save_mesh(result.mesh, args.out)

    summary = {
        "refined_mesh": str(args.out),
        "coarse_vertices": coarse_mesh.num_vertices,
        "coarse_faces": coarse_mesh.num_faces,
        "gt_vertices": gt_mesh.num_vertices,
        "gt_faces": gt_mesh.num_faces,
        "operator": args.operator,
        "mean_gt_projection_distance": float(result.target.distances.mean()),
        "max_gt_projection_distance": float(result.target.distances.max(initial=0.0)),
        "mean_confidence": float(result.target.confidence.mean()),
        "initial_loss": result.history[0]["loss"] if result.history else None,
        "final_loss": result.history[-1]["loss"] if result.history else None,
    }
    if args.history_out is not None:
        args.history_out.parent.mkdir(parents=True, exist_ok=True)
        with args.history_out.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "summary": summary,
                    "history": result.history,
                },
                handle,
                indent=2,
            )
    print(json.dumps(summary, indent=2))
    return 0


def _run_synthetic(args: argparse.Namespace) -> int:
    config = SyntheticRenderConfig(
        num_views=args.views,
        width=args.width,
        height=args.height,
        trajectory=args.trajectory,
        radius_scale=args.radius_scale,
        elevation_degrees=args.elevation,
        min_elevation_degrees=args.min_elevation,
        max_elevation_degrees=args.max_elevation,
        fov_degrees=args.fov,
        render_mode=args.mode,
        backend=args.backend,
        normalize_mesh=not args.no_normalize,
    )
    if args.mesh_dir is not None:
        datasets = generate_synthetic_datasets_from_mesh_dir(
            args.mesh_dir,
            args.out_dir,
            config=config,
            progress=_print_progress,
        )
        print(
            json.dumps(
                {
                    "out_root": str(args.out_dir),
                    "num_datasets": len(datasets),
                    "datasets": [str(dataset.dataset_path) for dataset in datasets],
                },
                indent=2,
            )
        )
        return 0

    dataset = generate_synthetic_dataset_from_mesh(
        args.mesh,
        args.out_dir,
        config=config,
        progress=_print_progress,
    )
    print(
        json.dumps(
            {
                "dataset": str(dataset.dataset_path),
                "mesh": str(dataset.mesh_path),
                "num_images": len(dataset.image_paths),
                "num_masks": len(dataset.mask_paths),
                "num_depth": len(dataset.depth_paths),
            },
            indent=2,
        )
    )
    return 0


def _print_progress(message: str) -> None:
    print(message, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
