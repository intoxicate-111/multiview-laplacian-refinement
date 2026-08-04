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

The single-object entry point remains batch size one and every prepared sample
has one internally fixed prediction/target topology. Visibility uses the
prepared mask rather than learned occlusion reasoning, non-uniform and general
different-topology target paths still use dense NumPy Laplacians, and no claim
is made about unseen objects. Shared training across multiple variable-size
prepared meshes is described below.

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

## Edge-Scale-Normalized Laplacian Target

The learned subsystem supports `target_mode` values `raw_laplacian` and
`edge_scale_normalized_laplacian`. For vertex `i`, `h_i` is the mean length of
its unique undirected incident edges on the input prediction mesh. The scale
is the square of that mean, not the mean of squared edge lengths:

```text
scale_i = h_i^2
delta_hat_target_i = delta_target_i / (h_i^2 + epsilon)
delta_pred_i = delta_hat_pred_i * h_i^2
```

The default `epsilon` is `1e-12`. Loss is evaluated in the selected target
space, while the existing reconstruction solver always receives denormalized
raw `delta_pred`. Clipping is disabled unless
`target_scaling.clip_max_norm` is explicitly configured, and its affected
vertex count is reported. Legacy prepared samples remain valid: scale fields
are derived on load and `laplacian_target` keeps its raw meaning. New samples
also store raw and normalized targets, `h`, `h^2`, the definition, source
graph, epsilon, and isolated-vertex count.

This transform is not globally scale invariant. Scaling coordinates by `a`
makes a first-order uniform Laplacian scale by `a`, but makes `h^2` scale by
`a^2`, so `delta_hat` scales by `1/a` on nonisolated vertices. Run the measured
global and same-surface cross-resolution diagnostic with:

```bash
python scripts/diagnose_edge_scale_normalization.py --fine-mesh inputs/stanford-bunny/mesh.obj --coarse-face-count 7000 --output-dir runs/learned_laplacian/edge_scale_diagnostics --epsilon 1e-12
```

The diagnostic uses Open3D quadric simplification to keep both resolutions on
the same Bunny surface, saves `diagnostics.json` and the underlying NPZ arrays,
and reports all-vertex and nonisolated statistics separately. This local Bunny
OBJ has 1,113 of 35,947 stored vertices absent from every face. Those vertices
receive `h=0`. Under the required formula their normalized targets are
epsilon-dominated (up to roughly `8.37e11` here), and multiplication by `h^2`
cannot recover a nonzero isolated-vertex raw target. No clipping or masking is
silently applied.

The matched 300-step CPU runs reuse the raw baseline's sample, views,
corruption, seed, architecture, optimizer, and reconstruction settings:

```bash
python scripts/overfit_single_object.py --sample runs/learned_laplacian/bunny_overfit/prepared_sample.pt --config configs/learned_laplacian/overfit_bunny_edge_normalized.json --output-dir runs/learned_laplacian/bunny_edge_normalized/coarse_only --input-mode coarse_only --device cpu --steps 300
python scripts/overfit_single_object.py --sample runs/learned_laplacian/bunny_overfit/prepared_sample.pt --config configs/learned_laplacian/overfit_bunny_edge_normalized.json --output-dir runs/learned_laplacian/bunny_edge_normalized/coarse_plus_multiview --input-mode coarse_plus_multiview --device cpu --steps 300
python scripts/compare_edge_scale_bunny.py --sample runs/learned_laplacian/bunny_overfit/prepared_sample.pt --raw-root runs/learned_laplacian/bunny_overfit --normalized-root runs/learned_laplacian/bunny_edge_normalized --output runs/learned_laplacian/bunny_edge_normalized/comparison.json --epsilon 1e-12
```

Each run prints and writes `pre_training_diagnostics.json` before optimisation.
It records vertex/face/unique-edge counts, degree distribution, local `h` and
`h^2` distributions, both target-magnitude distributions, and correlations
with `h^2`. Use `--diagnostics-only` to generate this file without training.

For this unmodified topology and budget, both normalized runs fail and are
marked exploded; both raw-target baselines remain stable and improve over the
coarse mesh. This is a negative experimental result, not evidence against
scale normalization on cleaned manifold graphs. A follow-up must state and
test an explicit isolated-vertex policy (topology cleanup, masking, or bounded
scaling) instead of introducing it implicitly into this matched comparison.

## Cleaned Bunny Edge-Scale-Normalised Experiment

The 1,113 isolated Bunny vertices are OBJ vertex records that are not
referenced by any triangle, rather than an edge-construction defect. The
cleaning path removes unreferenced vertices before normals, corruption, graph
edges, targets, projection, training, reconstruction, evaluation, or
visualisation are computed. Retained vertices keep their original order and
floating-point values, faces are remapped deterministically, and old-to-new,
new-to-old, and removed-index arrays are saved. Deleting vertices that are not
referenced by any face does not change the rendered triangle surface.

For every vertex with at least one incident edge, the normalised target is

```text
h_i = mean incident edge length on the coarse prediction mesh
s_i = h_i^2
delta_hat_target_i = (L P_target)_i / (s_i + epsilon)
delta_pred_i = delta_hat_pred_i * s_i
```

The scale is the square of the mean incident edge length, not the mean of
squared lengths. An isolated vertex has undefined `h_i`: it is marked false in
`valid_scale_mask`, receives zero target confidence, and is excluded from
normalised loss and metrics. Its uniform or zero-weight cotangent Laplacian row
is all zero, so its Laplacian vector is zero rather than its absolute position.
Epsilon remains a numerical guard for valid scales; it is not a substitute for
missing topology.

Diagnose the original OBJ and then prepare the cleaned 24-view, 128-pixel CPU
sample with the controlled corruption:

```bash
python scripts/diagnose_bunny_isolated_vertices.py \
  --mesh inputs/stanford-bunny/mesh.obj \
  --output runs/learned_laplacian/bunny_cleaned/isolated_vertex_diagnostics.json

python scripts/prepare_bunny_overfit_experiment.py \
  --gt-mesh inputs/stanford-bunny/mesh.obj \
  --reuse-dataset inputs_sphere/stanford-bunny/dataset.json \
  --output-root runs/learned_laplacian/bunny_cleaned \
  --views 24 \
  --image-size 128 \
  --noise-std 0.015 \
  --smoothing-iters 2 \
  --smoothing-strength 0.1 \
  --remove-unreferenced-vertices \
  --seed 7
```

The comparison runner first writes raw and normalised pre-training diagnostics,
then runs geometry-only and geometry-plus-RGB modes with identical capacity,
seed, reconstruction settings, metric sampling, and 300-step budget:

```bash
python scripts/run_bunny_normalization_comparison.py \
  --sample runs/learned_laplacian/bunny_cleaned/prepared_sample.pt \
  --raw-config configs/learned_laplacian/overfit_bunny.json \
  --normalized-config configs/learned_laplacian/overfit_bunny_edge_normalized.json \
  --output-root runs/learned_laplacian/bunny_cleaned \
  --device cpu \
  --steps 300
```

For a diagnostics-only run, invoke `scripts/overfit_single_object.py` with the
same sample and either config, an output directory, `--diagnostics-only`, and
`--diagnostics-output <path>`. Regenerate all fixed-camera mesh, shared-range
surface-error, recovered-raw-Laplacian-error, wireframe, histogram, scatter,
loss, and geometry-metric figures with:

```bash
python scripts/visualize_bunny_normalization.py \
  --sample runs/learned_laplacian/bunny_cleaned/prepared_sample.pt \
  --output-root runs/learned_laplacian/bunny_cleaned \
  --image-size 256
```

The output root contains cleaned/coarse/oracle/refined OBJ files and prediction
arrays, `comparison.json`/`.csv`, preparation and diagnostic JSON, plus
`renders/`, `errors/`, and `plots/`. Mesh grids reuse cameras derived once from
the cleaned GT bounds, so scale changes remain visible. Position-error panels
use exact distance to the GT triangle surface and one union 99th-percentile
range; Laplacian-error panels compare every model in recovered raw space.
Wireframe panels use shared GT-derived close-up cameras for the ears and feet.

This remains a single object, topology, corruption, view subset, and short CPU
overfit, so it establishes mathematical validity and numerical stability rather
than generalisation or superiority over the raw target. Edge-scale
normalisation is intended to reduce sampling-density sensitivity, not to
guarantee triangulation invariance. The next experiment should compare
edge-scale-normalised targets on multiple Bunny prediction graphs with
different sampling resolutions or subdivision levels.

## Multi-Mesh Shared Training

The multi-mesh path trains one shared CNN/GNN over prepared samples with
different vertex counts, face counts, graph connectivity, view counts, and
image sizes. Meshes are loaded lazily and forwarded one at a time; gradients
are divided by the number of meshes in each accumulation group before the
optimizer step. This gives every mesh equal weight regardless of its vertex
count and avoids padding large ragged graphs.

Create a JSON manifest whose paths are relative to the manifest file unless
absolute paths are used:

```json
{
  "samples": [
    {
      "sample_id": "bunny_low_01",
      "path": "prepared/bunny_low_01.pt",
      "split": "train"
    },
    {
      "sample_id": "bunny_mid_01",
      "path": "prepared/bunny_mid_01.pt",
      "split": "train"
    },
    {
      "sample_id": "bunny_high_validation",
      "path": "prepared/bunny_high_validation.pt",
      "split": "validation"
    }
  ]
}
```

Each file uses the existing prepared-sample schema. The coarse and target mesh
inside one sample must still share topology, but different samples may have
unrelated topologies and different numbers of views. Manifest `sample_id`
values are optional consistency checks; when supplied, they must match the ID
stored in the prepared file.

Run shared edge-scale-normalised training with:

```bash
python scripts/train_multi_mesh_laplacian.py \
  --manifest path/to/multi_mesh_manifest.json \
  --config configs/learned_laplacian/train_multi_mesh_edge_normalized.json \
  --output-dir runs/learned_laplacian/multi_mesh \
  --device cuda
```

The default example configuration uses 50 epochs and four meshes per gradient
accumulation group. `validation_every_epochs` controls validation frequency;
only validation epochs can replace the best checkpoint when a validation split
exists. Without a validation split, training loss is the selection criterion.
The command-line entry point requires both `train` and `validation` manifest
splits so accidental training-only experiments are explicit in Python.

Outputs include `best.pt`, optional epoch checkpoints,
`training_history.json`, `metrics.json`, and target-space plus recovered
raw-space prediction arrays under `predictions/train/` and
`predictions/validation/`. Metrics are reported separately for every object so
failures on a small mesh cannot be hidden by averaging with a large mesh.

This is gradient accumulation, not simultaneous packed-graph batching: it
trades some throughput for simple, memory-bounded support of ragged meshes.
Prepared samples are still expected to use consistent coordinate conventions,
and scale normalisation does not remove global-scale dependence. A meaningful
first dataset should contain multiple deterministic corruptions and multiple
Bunny sampling resolutions, with graph resolutions represented in both train
and held-out validation splits.
