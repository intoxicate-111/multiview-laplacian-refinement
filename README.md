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
- `OpenMVSCommandMeshGenerator`: write an OpenMVG-style `sfm_data.json` or COLMAP text model, call a local OpenMVS command pipeline, then import the generated mesh.

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

Example OpenMVS adapter:

```python
from mlr.coarse import OpenMVSCommandMeshGenerator, generate_coarse_mesh

backend = OpenMVSCommandMeshGenerator(
  scene_dir="runs/openmvs",
  output_mesh_path="runs/openmvs/coarse.obj",
  interface_format="colmap",
  command_template='InterfaceCOLMAP -i "{colmap_path}" -o "{scene_dir}/scene.mvs" --image-folder "{colmap_images_path}" && DensifyPointCloud -i "{scene_dir}/scene.mvs" -o "{scene_dir}/scene_dense.mvs" --resolution-level 2 && ReconstructMesh -i "{scene_dir}/scene_dense.mvs" -o "{output_mesh_path}"',
)

mesh = generate_coarse_mesh(image_paths, cameras, masks=masks, method=backend)
```

The OpenMVS adapter defaults to COLMAP text-model export because vcpkg's OpenMVS
tools provide `InterfaceCOLMAP`. If you only want to prepare inputs, use the CLI
`--prepare-only` mode below.

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

For NVIDIA CUDA rasterization, install a CUDA-enabled PyTorch build that
matches your local driver, then enable the CUDA backend:

```bash
pip install -e ".[cuda]"
mlr synthetic --mesh-dir meshes --out-dir inputs_cuda --views 48 --width 512 --height 512 --mode lit --trajectory sphere --backend cuda
```

The CUDA backend uses PyTorch tensors on `cuda` for z-buffer rasterization and
falls back with a clear error if PyTorch is not installed with CUDA support.

Generate a COLMAP text model and preview the OpenMVS command (prepare-only):

```bash
mlr coarse-openmvs --dataset runs/synthetic_gt/dataset.json --scene-dir runs/openmvs --out runs/openmvs/coarse.obj --prepare-only
```

Run OpenMVS via the default `InterfaceCOLMAP -> DensifyPointCloud -> ReconstructMesh` template:

```bash
mlr coarse-openmvs --dataset runs/synthetic_gt/dataset.json --scene-dir runs/openmvs --out runs/openmvs/coarse.obj --interface colmap
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

Refine a generated coarse mesh with GT Laplacian supervision interpolated from the GT surface:

```bash
mlr gt-laplacian-refine \
  --coarse-mesh runs/coarse/bunny_coarse.obj \
  --gt-mesh inputs_sphere/bunny/mesh.obj \
  --out runs/refined/bunny_gt_laplacian.obj \
  --history-out runs/refined/bunny_gt_laplacian_history.json \
  --operator uniform \
  --iters 300 \
  --lr 0.005 \
  --lambda-lap 1.0 \
  --lambda-anchor 0.05
```

This projects each coarse vertex to the closest GT triangle, then computes the
target Laplacian from those projected positions using the coarse mesh graph.
It does not interpolate GT Laplacian vectors across different samplings. Use
`--distance-confidence-scale` to down-weight coarse vertices that project far
away from the GT surface.

Run the coarse-graph GT Laplacian oracle instead, where the GT surface is first
sampled at the coarse vertex set and the Laplacian target is recomputed with the
coarse mesh connectivity:

```bash
mlr coarse-lap-oracle \
  --coarse-mesh runs/openmvs/normalized_full/coarse.obj \
  --gt-mesh meshes/normalized.obj \
  --output-dir runs/oracle_coarse_lap \
  --operator uniform \
  --device cuda \
  --iters 100000 \
  --lr 0.001 \
  --lambda-lap 1.0 \
  --lambda-anchor 0.01 \
  --lambda-pos 0.01 \
  --lambda-edge 0.01
```

Use `--device cuda` to move the optimization loop to PyTorch CUDA. Projection
and final metrics still run on CPU.

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

## Single-Object Learned Laplacian Overfitting

This isolated subsystem is a single-object sanity test. It verifies that RGB
features can be projected onto one mesh graph and used to predict one 3D
Laplacian vector per vertex. It does **not** demonstrate generalisation.

Install the optional training dependencies without changing the geometry-only
installation:

```bash
pip install -e ".[train]"
```

First generate or obtain multi-view inputs, a prediction/coarse mesh, and a GT
surface in the same world coordinate system. Prepare one validated `.pt`
sample with:

```bash
python scripts/prepare_single_object_sample.py \
  --dataset inputs_sphere/bunny/dataset.json \
  --coarse-mesh runs/coarse/bunny_coarse.obj \
  --gt-mesh inputs_sphere/bunny/mesh.obj \
  --output inputs/learned_laplacian/bunny.pt \
  --image-size 128
```

For a controlled synthetic debugging case, `--coarse-noise-std 0.02` adds a
deterministic normal-direction perturbation to the supplied coarse mesh before
constructing the target. The sample stores images `[V,3,H,W]`, intrinsics,
world-to-camera extrinsics, mesh geometry, visibility, the initial Laplacian,
the target Laplacian, and target confidence. Shape and finite-value checks fail
early with field-specific errors.

The target is graph-compatible by construction:

```text
P_target = closest GT-surface positions for the prediction vertices
delta_target = L_prediction_graph P_target
```

Laplacian vectors are never transferred directly from a differently sampled
GT mesh. Sample preparation calls the repository's existing
`compute_coarse_graph_gt_laplacian_target` implementation.

Train repeatedly on that one object:

```bash
python scripts/overfit_single_object.py \
  --sample inputs/learned_laplacian/bunny.pt \
  --config configs/learned_laplacian/overfit_single_object.json \
  --output-dir runs/learned_laplacian/overfit_single_object
```

The model is intentionally small: a randomly initialized CNN produces
per-view feature maps; mesh vertices are projected and sampled with
`grid_sample`; a masked mean produces one image descriptor and valid-view ratio
per vertex; and three dependency-light `index_add_` graph blocks predict
`[N,3]` Laplacian vectors. Geometry inputs are position, normal, initial
Laplacian, and degree.

Camera convention: each 4x4 extrinsic transforms world coordinates to a
right-handed CV camera with `+X` right, `+Y` down, and `+Z` forward. Image
coordinates have a top-left origin. With `align_corners=True`, pixels `(0,0)`
and `(W-1,H-1)` map to grid coordinates `(-1,-1)` and `(1,1)`. Vertices behind
the camera, outside the image, or false in the optional visibility mask are
excluded. Vertices with zero valid views receive a zero aggregate without NaNs.

Available ablations are `--input-mode coarse_only`, `--input-mode
multiview_only`, and `--input-mode coarse_plus_multiview`. `--zero-images`
zeros encoded image features to test whether geometry alone explains the
result. Graph connectivity is still required in every mode.

The run writes `training_history.json`, `best.pt`, `delta_target.npy`,
`delta_pred.npy`, `coarse.obj`, `predicted_refined.obj`,
`oracle_refined.obj`, and `metrics.json`; it also writes `loss_curve.png` when
matplotlib is already installed. Reconstruction is evaluation-only and calls
the existing NumPy Laplacian solver without differentiating through it.

Success means a substantial one-object loss reduction, finite `[N,3]` output,
a non-collapsed reconstruction, improvement over the coarse mesh on at least
one reported geometry metric, and an oracle reconstruction that remains an
upper bound. Run all tests with:

```bash
python -m pytest
```

Current limitations: batch size is one, every sample has one fixed topology,
visibility uses the prepared mask rather than learned occlusion reasoning,
non-uniform and general different-topology target paths still use dense NumPy
Laplacians, and no claim is made about unseen objects. The next scaling step should be 10--20 prepared
objects with topology-aware batching or per-object gradient accumulation,
train/validation separation, and sparse Laplacian/reconstruction operators.

## Stanford Bunny Single-Object Overfitting

This experiment scales the preceding sanity test to a realistic Stanford Bunny
while retaining a controlled same-topology graph. Successful Bunny overfitting
does not demonstrate cross-object generalisation. It also does not by itself
prove that the network relies on multi-view evidence, because a fixed-geometry
network may memorise one object's target.

The local experiment uses the already normalised clean asset at
`inputs/stanford-bunny/mesh.obj` (35,947 vertices, 69,451 faces, bounding-box
diagonal approximately 2). This ignored asset was generated from the local
`meshes/stanford-bunny.obj` source recorded in its dataset metadata; the source
asset is not added to Git by this experiment. A second identical clean copy and
64-view sphere render are available under `inputs_sphere/stanford-bunny/`.

Install training and scalable surface-metric dependencies:

```bash
pip install -e ".[train,bunny]"
```

Preparation copies the clean topology, applies deterministic uniform
Laplacian smoothing and normal-direction Gaussian noise, resamples 24 clean
sphere views, constructs the target with the repository's sparse uniform
operator, and writes a projection overlay:

```bash
python scripts/prepare_bunny_overfit_experiment.py \
  --gt-mesh inputs/stanford-bunny/mesh.obj \
  --reuse-dataset inputs_sphere/stanford-bunny/dataset.json \
  --output-root runs/learned_laplacian/bunny_overfit \
  --views 24 \
  --image-size 256 \
  --noise-std 0.015 \
  --smoothing-iters 2 \
  --smoothing-strength 0.1 \
  --seed 7
```

The documented target is 256x256. On a CPU-only machine, use
`--image-size 128` for the validated diagnostic run. Without
`--reuse-dataset`, the wrapper invokes the repository's clean-GT sphere
renderer directly; select `--backend cpu`, `opengl`, or `cuda` as available.

Run the three matched-budget modes together:

```bash
python scripts/run_bunny_overfit_ablations.py \
  --sample runs/learned_laplacian/bunny_overfit/prepared_sample.pt \
  --config configs/learned_laplacian/overfit_bunny.json \
  --output-root runs/learned_laplacian/bunny_overfit \
  --device cuda
```

For the CPU diagnostic reported in this repository, append `--device cpu
--steps 300`. Equivalent individual commands are:

```bash
python scripts/overfit_single_object.py --sample runs/learned_laplacian/bunny_overfit/prepared_sample.pt --config configs/learned_laplacian/overfit_bunny.json --output-dir runs/learned_laplacian/bunny_overfit/coarse_only --input-mode coarse_only --device cpu --steps 300
python scripts/overfit_single_object.py --sample runs/learned_laplacian/bunny_overfit/prepared_sample.pt --config configs/learned_laplacian/overfit_bunny.json --output-dir runs/learned_laplacian/bunny_overfit/coarse_plus_multiview --input-mode coarse_plus_multiview --device cpu --steps 300
python scripts/overfit_single_object.py --sample runs/learned_laplacian/bunny_overfit/prepared_sample.pt --config configs/learned_laplacian/overfit_bunny.json --output-dir runs/learned_laplacian/bunny_overfit/zero_images --input-mode coarse_plus_multiview --zero-images --device cpu --steps 300
```

All modes share the same graph, target, seed, capacity, training budget,
reconstruction parameters, Chamfer sampling indices, and metric seed. The
large-mesh path caches graph edges, uses the existing sparse coarse-oracle
Laplacian loss/gradient for reconstruction, and uses `trimesh`/`rtree` for
exact point-to-surface distances. Outputs include the common GT/coarse/oracle
OBJ files, per-mode checkpoints and reconstructed meshes, loss and error
visualisations, `projection_debug.png`, `comparison_render.png`, and
`comparison.json`.

Point-to-surface and Chamfer measure surface agreement; target-position RMSE
uses the known same-topology correspondence; normal consistency approaches 1
as orientations agree; and the bounding-box ratio plus explicit
collapse/explosion flag checks reconstruction stability. The current
experiment remains one object with one corruption. If image and geometry-only
modes are similar, the next experiment should use multiple deterministic
corruptions of this same Bunny before moving to 10--20 different objects.
