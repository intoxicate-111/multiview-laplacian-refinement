from __future__ import annotations

import argparse
import json
from pathlib import Path

from .coarse import (
    ExistingMeshGenerator,
    NvidiaInstantNGPMeshGenerator,
    NvidiaNvdiffrecMeshGenerator,
    OpenMVSCommandMeshGenerator,
    generate_coarse_mesh,
    write_colmap_text_model,
    write_openmvg_sfm_data,
)
from .coarse_lap_oracle import CoarseGraphOracleConfig, run_coarse_graph_laplacian_oracles
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

    coarse_openmvs = sub.add_parser("coarse-openmvs", help="Generate a coarse mesh with OpenMVS.")
    coarse_openmvs.add_argument("--dataset", required=True, type=Path)
    coarse_openmvs.add_argument("--scene-dir", required=True, type=Path)
    coarse_openmvs.add_argument("--out", required=True, type=Path)
    coarse_openmvs.add_argument("--command-template", default="")
    coarse_openmvs.add_argument("--interface", default="colmap", choices=["colmap", "openmvg"])
    coarse_openmvs.add_argument("--colmap-dir", default="colmap")
    coarse_openmvs.add_argument("--sfm-data", default="sfm_data.json")
    coarse_openmvs.add_argument("--no-copy-images", action="store_true")
    coarse_openmvs.add_argument("--no-visibility", action="store_true")
    coarse_openmvs.add_argument("--prepare-only", action="store_true")

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

    coarse_lap = sub.add_parser(
        "coarse-lap-oracle",
        help="Run coarse-graph-compatible GT Laplacian oracle experiments.",
    )
    coarse_lap.add_argument("--coarse-mesh", required=True, type=Path)
    coarse_lap.add_argument("--gt-mesh", required=True, type=Path)
    coarse_lap.add_argument("--output-dir", required=True, type=Path)
    coarse_lap.add_argument("--operator", default="uniform", choices=["uniform"])
    coarse_lap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    coarse_lap.add_argument("--iters", default=3000, type=int)
    coarse_lap.add_argument("--lr", default=5e-3, type=float)
    coarse_lap.add_argument("--lambda-lap", default=1.0, type=float)
    coarse_lap.add_argument("--lambda-anchor", default=0.01, type=float)
    coarse_lap.add_argument("--lambda-pos", default=0.1, type=float)
    coarse_lap.add_argument("--lambda-edge", default=0.0, type=float)
    coarse_lap.add_argument("--normalized-eps", default=1e-8, type=float)
    coarse_lap.add_argument("--log-every", default=25, type=int)
    coarse_lap.add_argument("--print-every", default=0, type=int)
    coarse_lap.add_argument("--chamfer-samples", default=5000, type=int)
    coarse_lap.add_argument("--seed", default=7, type=int)
    coarse_lap.add_argument("--reg-surface-loss", default="point_to_plane", choices=["point_to_plane", "point_to_point"])
    coarse_lap.add_argument("--reg-lambda-surface", default=1.0, type=float)
    coarse_lap.add_argument("--reg-lambda-lap-smooth", default=0.1, type=float)
    coarse_lap.add_argument("--reg-lambda-edge", default=0.01, type=float)
    coarse_lap.add_argument("--reg-lambda-anchor", default=0.01, type=float)
    coarse_lap.add_argument("--reg-iters", default=10000, type=int)
    coarse_lap.add_argument("--reg-lr", default=1e-3, type=float)
    coarse_lap.add_argument("--previous-refined-mesh", type=Path)
    coarse_lap.add_argument("--previous-history", type=Path)
    gt_lap.add_argument("--print-every", default=0, type=int)

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
    synthetic.add_argument("--backend", default="cpu", choices=["cpu", "opengl", "cuda"])
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
    if args.command == "coarse-lap-oracle":
        return _run_coarse_lap_oracle(args)
    if args.command == "synthetic":
        return _run_synthetic(args)
    if args.command == "coarse-openmvs":
        return _run_coarse_openmvs(args)
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


def _run_coarse_openmvs(args: argparse.Namespace) -> int:
    data = load_reconstruction_input(args.dataset)
    masks = load_masks(data.mask_paths)
    scene_dir = Path(args.scene_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)
    sfm_data_path = scene_dir / args.sfm_data
    colmap_path = scene_dir / args.colmap_dir
    output_mesh_path = Path(args.out)
    output_mesh_path.parent.mkdir(parents=True, exist_ok=True)
    if args.interface == "openmvg":
        write_openmvg_sfm_data(sfm_data_path, data.image_paths, data.cameras, scene_dir=scene_dir)
    else:
        write_colmap_text_model(
            colmap_path,
            data.image_paths,
            data.cameras,
            copy_images=not args.no_copy_images,
        )

    command_template = args.command_template or _default_openmvs_command_template(args.interface)
    if args.prepare_only:
        payload = {
            "mode": "prepare_only",
            "interface": args.interface,
            "sfm_data_path": str(sfm_data_path.resolve()),
            "colmap_path": str(colmap_path.resolve()),
            "colmap_sparse_path": str((colmap_path / "sparse").resolve()),
            "colmap_images_path": str((colmap_path / "images").resolve()),
            "scene_dir": str(scene_dir.resolve()),
            "next_command": command_template.format(
                scene_dir=str(scene_dir.resolve()),
                sfm_data_path=str(sfm_data_path.resolve()),
                colmap_path=str(colmap_path.resolve()),
                colmap_sparse_path=str((colmap_path / "sparse").resolve()),
                colmap_images_path=str((colmap_path / "images").resolve()),
                output_mesh_path=str(output_mesh_path.resolve()),
            ),
        }
        print(json.dumps(payload, indent=2))
        return 0

    backend = OpenMVSCommandMeshGenerator(
        scene_dir=scene_dir,
        output_mesh_path=args.out,
        command_template=command_template,
        sfm_data_filename=args.sfm_data,
        interface_format=args.interface,
        colmap_dirname=args.colmap_dir,
        copy_images=not args.no_copy_images,
        compute_visibility=not args.no_visibility,
    )
    print(f"Loaded dataset with {len(data.image_paths)} images from {args.dataset}", flush=True)
    if args.interface == "openmvg":
        print(f"Writing OpenMVG sfm_data to {sfm_data_path}", flush=True)
    else:
        print(f"Writing COLMAP text model to {colmap_path}", flush=True)
    print(f"Expected coarse mesh output: {args.out}", flush=True)
    mesh = generate_coarse_mesh(data.image_paths, data.cameras, masks=masks, method=backend)
    print(
        json.dumps(
            {
                "coarse_mesh": str(args.out),
                "vertices": mesh.num_vertices,
                "faces": mesh.num_faces,
                "backend": mesh.attributes.get("coarse_backend", "openmvs"),
                "interface": mesh.attributes.get("openmvs_interface_format", args.interface),
                "openmvg_sfm_data": mesh.attributes.get("openmvg_sfm_data"),
                "colmap_path": mesh.attributes.get("colmap_path"),
            },
            indent=2,
        )
    )
    return 0


def _default_openmvs_command_template(interface: str) -> str:
    if interface == "colmap":
        return (
            'InterfaceCOLMAP -w "{scene_dir}" -i "{colmap_path}" -o "{scene_dir}/scene.mvs" '
            '--image-folder "{colmap_images_path}" && '
            'DensifyPointCloud -w "{scene_dir}" -i "{scene_dir}/scene.mvs" -o "{scene_dir}/scene_dense.mvs" '
            '--resolution-level 2 && '
            'ReconstructMesh -w "{scene_dir}" -i "{scene_dir}/scene_dense.mvs" '
            '-o "{output_mesh_path}" --export-type ply'
        )
    if interface == "openmvg":
        return (
            'InterfaceOpenMVG -w "{scene_dir}" -i "{sfm_data_path}" -o "{scene_dir}/scene.mvs" && '
            'DensifyPointCloud -w "{scene_dir}" -i "{scene_dir}/scene.mvs" -o "{scene_dir}/scene_dense.mvs" '
            '--resolution-level 2 && '
            'ReconstructMesh -w "{scene_dir}" -i "{scene_dir}/scene_dense.mvs" '
            '-o "{output_mesh_path}" --export-type ply'
        )
    raise ValueError(f"Unsupported OpenMVS interface: {interface}")


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
        print_every=args.print_every,
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


def _run_coarse_lap_oracle(args: argparse.Namespace) -> int:
    coarse_mesh = load_mesh(args.coarse_mesh)
    gt_mesh = load_mesh(args.gt_mesh)
    previous_mesh = load_mesh(args.previous_refined_mesh) if args.previous_refined_mesh else None
    previous_history = None
    if args.previous_history and args.previous_history.exists():
        with args.previous_history.open("r", encoding="utf-8") as handle:
            previous_history = json.load(handle)
    config = CoarseGraphOracleConfig(
        operator_type=args.operator,
        device=args.device,
        num_iters=args.iters,
        learning_rate=args.lr,
        lambda_lap=args.lambda_lap,
        lambda_anchor=args.lambda_anchor,
        lambda_pos=args.lambda_pos,
        lambda_edge=args.lambda_edge,
        normalized_eps=args.normalized_eps,
        log_every=args.log_every,
        print_every=args.print_every,
        chamfer_samples=args.chamfer_samples,
        seed=args.seed,
        reg_surface_loss=args.reg_surface_loss,
        reg_lambda_surface=args.reg_lambda_surface,
        reg_lambda_lap_smooth=args.reg_lambda_lap_smooth,
        reg_lambda_edge=args.reg_lambda_edge,
        reg_lambda_anchor=args.reg_lambda_anchor,
        reg_iters=args.reg_iters,
        reg_lr=args.reg_lr,
    )
    print(
        f"Loaded coarse mesh {args.coarse_mesh} ({coarse_mesh.num_vertices}v/{coarse_mesh.num_faces}f)",
        flush=True,
    )
    print(f"Loaded GT mesh {args.gt_mesh} ({gt_mesh.num_vertices}v/{gt_mesh.num_faces}f)", flush=True)
    print(f"Running coarse-graph Laplacian oracle experiments in {args.output_dir}", flush=True)
    comparison = run_coarse_graph_laplacian_oracles(
        coarse_mesh=coarse_mesh,
        gt_mesh=gt_mesh,
        output_dir=args.output_dir,
        config=config,
        previous_refined_mesh=previous_mesh,
        previous_history=previous_history,
    )
    print(json.dumps(comparison, indent=2))
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
