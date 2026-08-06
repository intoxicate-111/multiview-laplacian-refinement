# Multi-View GT Laplacian Learning

Training guides: [English](docs/MULTI_MESH_TRAINING.md) |
[简体中文](docs/MULTI_MESH_TRAINING.zh-CN.md)

## Project objective

The learned-Laplacian pipeline has one primary objective:

```text
multi-view RGB + calibrated cameras + 3D query position + local graph context
    -> the ground-truth local Laplacian signal at that 3D location
```

Training uses ground-truth meshes because they provide the supervised field we
want the network to learn. It does **not** create a coarse mesh and does not
train a coarse-to-GT correction. At inference, the learned field is queried at
the vertices of an arbitrary input mesh, including an unseen coarse or
topology-expanded mesh. Generalisation to held-out objects and non-GT query
graphs is the final goal.

The current target is the edge-scale-normalised uniform Laplacian of the GT
mesh. For a GT vertex `i`,

```text
delta_gt_i = (L_gt V_gt)_i
h_i        = mean incident GT edge length at i
target_i   = delta_gt_i / (h_i^2 + epsilon)
```

The normalised target is predicted first. Reconstruction converts it back to
raw Laplacian coordinates with the query graph's local scale and solves the
existing Laplacian reconstruction problem.

## Training contract

For every training object:

1. render calibrated multi-view RGB observations from the GT mesh;
2. use GT vertices and GT connectivity as the training query graph;
3. retain some exact GT query positions;
4. perturb the other query positions by small normal and tangent offsets
   relative to local edge length `h_i`;
5. keep the target attached to the corresponding original GT vertex;
6. predict its edge-scale-normalised GT Laplacian.

In symbols:

```text
q_i = V_gt_i + small_normal_offset_i + small_tangent_offset_i

F(images, cameras, q_i, normal_i, h_i, graph)
    ~= edge_scale_normalized_laplacian_gt_i
```

The perturbation teaches a local 3D query field instead of a lookup table that
only works at exact GT coordinates. Exact and perturbed query losses are
recorded separately.

## No target leakage

GT Laplacian vectors are supervision only. They must never be copied into the
model input.

- `initial_laplacian` is zero in GT-query samples.
- The raw or normalised GT Laplacian is not an input feature.
- A GT Laplacian is never interpolated onto a coarse or expanded graph.
- An inference-only expanded sample may contain schema placeholders, but they
  are not training targets or oracle supervision.
- The input normal, local edge scale, degree, query coordinates, graph and
  image features provide context; none is the target itself.

## Query positional encoding

Fourier encoding is computed dynamically in the model after query
augmentation:

```text
query position
  -> normalize by per-object center and scale
  -> [q, sin(2^k pi q), cos(2^k pi q)]
  -> concatenate image feature, normal, relative local scale and degree
  -> graph predictor
```

It is intentionally not precomputed during dataset preparation. The encoded
coordinate must follow the actual perturbed training query or the actual
coarse/expanded inference query.

The production geometry mode is `query_fourier`. The historical CLI value
`coarse_plus_multiview` means “query-geometry context plus multi-view features”
in this mode; it does not mean that training constructs a coarse mesh or feeds
its raw Laplacian to the predictor.

## Inference contract

Inference is separate from supervised GT-query training:

```text
multi-view observations
  -> obtain any initial/coarse mesh
  -> optional topology expansion
  -> use its vertices as 3D queries
  -> project each query into all calibrated views
  -> aggregate CNN features
  -> apply Fourier query encoding and graph context
  -> predict normalised Laplacian
  -> recover raw query-graph Laplacian
  -> Laplacian reconstruction
```

No GT mesh is available or required in this path. Position normalisation at
inference must come from the observation/query coordinate frame, not from a
hidden GT mesh.

## Current Sofa50 dataset

The current full dataset is:

```text
/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/sofa_mesh/sofa50_refinement/multiview_960
```

It contains:

- 40 train, 5 validation and 5 held-out test objects;
- 14 calibrated 960 x 960 RGB views per object;
- lazy GT-query samples for supervised training;
- separate expanded-query samples for inference evaluation;
- variable topology and mesh size across objects.

Training manifest:

```text
.../multiview_960/gt_query_manifest.json
```

Inference-only expanded manifest:

```text
.../multiview_960/expanded_inference_manifest.json
```

Never pass the expanded inference manifest to the training loop.

## Full training

Install the package and training dependencies, then run:

```bash
pip install -e ".[train]"
bash scripts/train_sofa50_v8_960_5000.sh
```

The launcher uses:

```text
configs/learned_laplacian/train_gt_query_sofa50_v8_960_5000.json
```

The profile uses CUDA AMP, four lazy DataLoader workers, pinned memory,
non-blocking transfer, gradient accumulation over four meshes, validation every
five epochs and periodic checkpoints. It is capped at 5,000 epochs and 50,000
optimizer steps.

The current full output directory is:

```text
runs/learned_laplacian/sofa50_refinement_960_gt_query_5000_full
```

## What constitutes evidence

A lower GT-query validation loss is necessary but is not sufficient to prove
the final goal. A useful checkpoint must pass all of the following checks:

1. **Finite learning:** train and held-out validation losses improve over a
   zero predictor.
2. **Image dependence:** original RGB outperforms zero RGB, shuffled views and
   cross-object RGB with the query graph and target fixed.
3. **Non-collapsed amplitude:** `mean |prediction| / mean |GT|` is not near
   zero, especially in high-magnitude regions.
4. **Directional accuracy:** high-magnitude target regions have positive and
   improving cosine similarity.
5. **Cross-object generalisation:** held-out objects improve, not only the
   training meshes.
6. **Expanded-query transfer:** the same checkpoint works on real
   coarse/expanded queries that were never used as GT training graphs.
7. **Reconstruction:** predicted-Laplacian reconstruction improves Chamfer and
   normal consistency over the initial mesh.

Single-mesh overfitting only proves capacity. GT-query validation only proves
the supervised field is learnable. Neither alone proves expanded-query
reconstruction.

## Diagnostic commands

Image ablation and mesh-count scaling tools are available under `scripts/`:

```bash
python scripts/ablate_single_mesh_checkpoint_images.py --help
python scripts/run_mesh_count_scaling.py --help
python scripts/diagnose_laplacian_prediction.py --help
```

Supported image conditions include original RGB, zero RGB, shuffled view order
and cross-object RGB. Mesh-count scaling uses nested 1/2/4/8/16-object sets and
reports the zero-predictor baseline, prediction/target amplitude ratio,
high-10% cosine and per-object metrics.

## Data and precision path

Prepared RGB remains lazy on disk. A worker decodes only the requested views as
`uint8`; pinned CPU tensors are transferred to CUDA non-blockingly, converted
to floating point, divided by 255 and normalised on the GPU. CNN and GNN
forward passes use AMP. Target scaling, robust loss and numerical geometry
operations remain FP32.

The trainer records DataLoader wait, image decode, GPU transfer,
forward/backward, total epoch time, validation time and CPU/GPU memory.

## Legacy baselines

The repository still contains coarse-mesh generators, oracle refinement,
pseudo-surface experiments, single-object Bunny experiments and reconstruction
solvers. They are useful baselines and debugging tools, but they do not define
the current learned model's supervision contract.

In particular, historical coarse-graph targets or closest-point pseudo targets
must not be described as the production learned-Laplacian target. Production
training is direct GT-query supervision; coarse/expanded meshes appear only as
queries during downstream inference and evaluation.

## Tests

```bash
PYTHONPATH=src conda run --no-capture-output -n test pytest -q
```

Focused tests cover query perturbation bounds, zero initial-Laplacian leakage
protection, Fourier query encoding, lazy image loading, AMP training, image
ablation, mesh-count scaling and Sofa50 preparation.
