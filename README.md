# Multi-View Laplacian Refinement

This repository is a staged experiment scaffold for multi-view mesh reconstruction/refinement.
The goal is not to train a reconstruction network from scratch, but to:

1. generate or import a coarse mesh from an existing multi-view reconstruction method;
2. refine mesh vertices with Laplacian-coordinate supervision;
3. estimate a pseudo target surface from multi-view evidence;
4. convert the pseudo surface into a pseudo-Laplacian target;
5. run alternating visibility / pseudo-target / vertex-optimization loops.

The current implementation focuses on a small, testable framework:

- unified `Mesh` and `Camera` data structures;
- replaceable coarse mesh generator backends;
- fixed uniform and cotangent Laplacian operators;
- oracle Laplacian refinement baselines for known topology/correspondence;
- pseudo-surface interfaces with confidence weighting;
- an alternating refinement loop skeleton.

## Install

```bash
pip install -e .
```

For tests:

```bash
pip install -e ".[dev]"
python -m pytest
```

## Stage 1: Coarse Mesh Baseline

The coarse generator module exposes:

```python
generate_coarse_mesh(images, cameras, masks=None, method=...)
```

Every backend returns a normalized `Mesh`:

- `vertices`: `(N, 3)` float array;
- `faces`: `(M, 3)` int array;
- `normals`: `(N, 3)` vertex normals;
- `visibility`: optional vertex-view boolean cache.

Backends included now:

- `ExistingMeshGenerator`: import a precomputed mesh from OBJ/PLY-style text formats;
- `ExternalCommandMeshGenerator`: run an external reconstruction command that writes a mesh.
- `NvidiaInstantNGPMeshGenerator`: write an Instant-NGP/NeRF-style `transforms.json`, call a local NVIDIA Instant-NGP command or wrapper, then import the generated mesh.

That makes it easy to later plug in COLMAP/OpenMVS, NeuS, VolSDF, Instant-NGP marching cubes, or any other existing method without changing refinement code.

Example NVIDIA Instant-NGP adapter:

```python
from mlr.coarse import NvidiaInstantNGPMeshGenerator, generate_coarse_mesh

backend = NvidiaInstantNGPMeshGenerator(
    scene_dir="data/ngp_scene",
    output_mesh_path="runs/stage1/instant_ngp_coarse.obj",
    command_template='instant-ngp --scene "{scene_dir}" --save_mesh "{output_mesh_path}"',
)

mesh = generate_coarse_mesh(image_paths, cameras, masks=masks, method=backend)
```

`command_template` should match your local Instant-NGP build or wrapper. The adapter provides `{scene_dir}`, `{transforms_path}`, and `{output_mesh_path}` placeholders. Cameras are assumed to be CV-style world-to-camera internally; by default the adapter converts them to NeRF/OpenGL camera-to-world matrices.

## Stage 2: Oracle Laplacian Refinement

The oracle baseline assumes known topology/correspondence:

```python
from mlr.oracle import run_oracle_baselines

results = run_oracle_baselines(init_mesh, gt_vertices)
```

It compares:

- position-only refinement;
- Laplacian-only refinement;
- position + Laplacian refinement;
- zero-Laplacian smoothing;
- noisy GT Laplacian target.

The main refinement loss is:

```text
lambda_lap * robust(L(V) - delta_target)
+ lambda_anchor * ||V - V_anchor||^2
+ regularization
```

The default robust loss is Charbonnier, with Huber also available.

## Stage 3: Pseudo Target Surface

Pseudo targets are estimated as vertex positions first:

```python
P_star, confidence = estimate_pseudo_surface(mesh, images, cameras, masks, visibility)
delta_pseudo = compute_laplacian_target(P_star, mesh.faces, operator_type="uniform")
refined = refine_mesh_with_laplacian(mesh, delta_pseudo, confidence, anchors)
```

This keeps the pseudo-Laplacian integrable because it comes from a concrete pseudo surface instead of arbitrary per-vertex differential vectors.

## Stage 4 and 5

The `alternating.py` loop freezes/detaches `P_star` during each inner optimization round:

```python
for outer_iter in range(num_outer_iters):
    visibility = update_visibility(...)
    P_star, confidence = estimate_pseudo_surface(...)
    delta_pseudo = compute_laplacian_target(P_star, ...)
    mesh = refine_mesh_with_laplacian(...)
```

The initial implementation uses fixed uniform Laplacian by default. Fixed cotangent is available for comparison. Dynamic cotangent should be added later only after the fixed-operator experiments are stable.

## Minimal CLI

Generate synthetic multi-view experiment inputs from a mesh:

```bash
mlr synthetic --mesh path/to/gt.obj --out-dir runs/synthetic_gt --views 24 --width 512 --height 512 --mode lit
```

Generate inputs for every mesh in a directory:

```bash
mkdir meshes inputs
mlr synthetic --mesh-dir meshes --out-dir inputs --views 24 --width 512 --height 512 --mode lit
```

The default camera path is `orbit`: a turntable-style 360-degree azimuth sweep at one fixed elevation. For multi-view coverage across upper/lower viewpoints, use `sphere`:

```bash
mlr synthetic --mesh-dir meshes --out-dir inputs_sphere --views 48 --width 512 --height 512 --mode lit --trajectory sphere --min-elevation -60 --max-elevation 60
```

Use GPU rasterization through OpenGL/ModernGL:

```bash
pip install -e ".[gpu]"
mlr synthetic --mesh-dir meshes --out-dir inputs --views 48 --width 512 --height 512 --mode lit --trajectory sphere --backend opengl
```

The OpenGL backend is the recommended fast path for AMD GPUs such as the Radeon RX 7900 XTX. The default CPU backend remains available for portability:

```bash
mlr synthetic --mesh-dir meshes --out-dir inputs --backend cpu
```

For `meshes/bunny.obj` and `meshes/armadillo.obj`, this creates:

```text
inputs/
  bunny/
    images/
    masks/
    depth/
    cameras.json
    dataset.json
    mesh.obj
  armadillo/
    images/
    masks/
    depth/
    cameras.json
    dataset.json
    mesh.obj
```

This writes:

- `images/*.png`: rendered RGB views;
- `masks/*.png`: silhouette masks;
- `depth/*.npy`: z-buffer depth maps;
- `cameras.json`: intrinsics/extrinsics for every view;
- `dataset.json`: paths and render settings;
- `mesh.obj`: the normalized mesh used for rendering.

The renderer is intentionally simple and deterministic: orbit cameras, pinhole projection, NumPy z-buffer rasterization, and `lit` / `normal` / `depth` rendering modes. It is meant for controlled debugging before moving to real multi-view captures.

Run oracle baselines:

```bash
mlr oracle --init-mesh path/to/init.obj --gt-mesh path/to/gt.obj --out-dir runs/oracle
```

Import a coarse mesh baseline:

```bash
mlr coarse --mesh path/to/coarse.obj --out path/to/normalized_coarse.obj
```

Generate a coarse mesh from generated multi-view inputs with an Instant-NGP-style external command:

```bash
mlr coarse-ngp \
  --dataset inputs_sphere/bunny/dataset.json \
  --scene-dir runs/coarse_ngp/bunny_scene \
  --out runs/coarse_ngp/bunny_coarse.obj \
  --command-template 'instant-ngp --scene "{scene_dir}" --save_mesh "{output_mesh_path}"'
```

The command template receives:

- `{scene_dir}`: directory where `transforms.json` is written;
- `{transforms_path}`: full path to the generated `transforms.json`;
- `{output_mesh_path}`: path that the external method must write.

Adjust the template to match your local Instant-NGP build or wrapper. If you already generated a coarse mesh through another method, use `mlr coarse --mesh ...` to normalize/import it into the framework.

Generate a coarse mesh with NVIDIA nvdiffrec:

```bash
mlr coarse-nvdiffrec \
  --dataset inputs_sphere/bunny/dataset.json \
  --nvdiffrec-root path/to/nvdiffrec \
  --run-dir runs/nvdiffrec/bunny \
  --out runs/coarse/bunny_nvdiffrec.obj \
  --iters 1000 \
  --train-res 512 \
  --texture-res 1024 \
  --batch 4
```

This command converts the generated RGB/mask views into nvdiffrec's NeRF-style RGBA dataset, writes `transforms_train.json`, writes a nvdiffrec config, runs:

```bash
python train.py --config "{config_path}"
```

from `--nvdiffrec-root`, then imports the resulting OBJ as the framework coarse mesh. Official nvdiffrec depends on CUDA/nvdiffrast and is designed for NVIDIA GPUs; AMD GPUs can accelerate this repository's OpenGL synthetic rendering, but the upstream nvdiffrec training code typically needs a CUDA-capable NVIDIA environment.
