# Multi-Mesh GT-Query Training Guide

[简体中文](MULTI_MESH_TRAINING.zh-CN.md) | [Project README](../README.md)

## Purpose

This guide documents the production path for learning a shared local
Laplacian field from calibrated multi-view images:

```text
multi-view RGB + 3D query + local graph context
    -> GT edge-scale-normalised Laplacian at the query location
```

The network is trained on GT mesh graphs and is expected to generalise to
unseen objects and to vertices of arbitrary coarse/expanded inference meshes.
Training does not generate a coarse mesh, does not optimise a coarse-to-GT
residual and does not interpolate GT Laplacian vectors to another graph.

## Supervision and query construction

For each supervised object, sample preparation computes the uniform Laplacian
directly on the GT graph:

```text
raw_target_i        = (L_gt V_gt)_i
local_scale_i       = mean incident GT edge length
normalised_target_i = raw_target_i / (local_scale_i^2 + epsilon)
```

At training time, the query position is generated dynamically. Twenty percent
of vertices remain exact by default; the others receive bounded normal and
tangent offsets relative to local edge length. The target remains the direct
GT-graph target of the corresponding original vertex.

This trains a local query field around the surface while retaining exact-query
supervision. Exact and perturbed losses are reported separately.

## Leakage prevention

The following invariants are mandatory:

- GT-query samples store a zero `initial_laplacian`;
- raw and normalised GT Laplacian tensors are supervision, not input features;
- no GT Laplacian vector is transferred to a coarse or expanded graph;
- training geometry comes from GT vertices and GT faces;
- inference-only expanded samples never enter the training dataset;
- test objects are reserved for final evaluation.

The trainer validates the zero-initial-Laplacian invariant before use.

## Dynamic Fourier query encoding

Dataset preparation stores the coordinate-normalisation center and scale, but
does not precompute Fourier features. The model encodes the actual query after
augmentation:

```text
q_normalised = (q - center) / scale
PE(q) = [q, sin(2^k pi q), cos(2^k pi q)]
```

The production predictor concatenates:

- aggregated multi-view CNN features sampled at the query projection;
- valid-view ratio;
- Fourier-encoded query coordinates;
- query-graph vertex normal;
- relative local edge scale;
- graph degree.

`geometry_mode=query_fourier` excludes `initial_laplacian` from this feature
set. The config name `coarse_plus_multiview` is a legacy input-mode label; in
the production query-Fourier model it means graph/query context plus images,
not coarse-mesh supervision.

## Current Sofa50 contract

The checked dataset root is:

```text
/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/sofa_mesh/sofa50_refinement/multiview_960
```

It contains 50 objects split into 40 train, 5 validation and 5 held-out test
objects. Every object has 14 calibrated 960 x 960 RGB views and variable mesh
topology.

Use this manifest for training:

```text
/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/sofa_mesh/sofa50_refinement/multiview_960/gt_query_manifest.json
```

Use this manifest only for downstream inference evaluation:

```text
/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/sofa_mesh/sofa50_refinement/multiview_960/expanded_inference_manifest.json
```

The expanded manifest's schema-required target is not GT supervision. Passing
that manifest to the training loop would violate the project objective.

## Full launch

The production launcher is:

```bash
bash scripts/train_sofa50_v8_960_5000.sh
```

It uses:

```text
configs/learned_laplacian/train_gt_query_sofa50_v8_960_5000.json
```

The full run writes to:

```text
runs/learned_laplacian/sofa50_refinement_960_gt_query_5000_full
```

The current long-run policy is:

| Setting | Value |
|---|---:|
| Maximum epochs | 5,000 |
| Maximum optimizer steps | 50,000 |
| Gradient accumulation | 4 meshes |
| Optimizer steps per full epoch | 10 |
| Validation interval | 5 epochs |
| Checkpoint interval | 100 epochs |
| DataLoader workers | 4 |
| Prefetch factor | 2 |
| Pinned memory | enabled |
| Persistent workers | enabled |
| CUDA AMP | FP16 enabled |
| Primary loss | Huber, delta 0.01 |

The launcher uses the `test` Conda environment and requires CUDA. Split counts
are checked before training starts.

## Lazy data and precision path

`PreparedMeshDataset` remains lazy; it is not converted to a tuple or list.
The data path is:

```text
static GT graph and supervision metadata
  -> lazy DataLoader worker
  -> decode requested RGB views as uint8
  -> pinned CPU tensor
  -> non-blocking CUDA transfer
  -> convert to float, divide by 255 and normalise on GPU
  -> AMP CNN feature extraction and GNN prediction
  -> FP32 target scaling, Huber loss and metrics
```

Only requested and prefetched images are decoded. The full image dataset is
not cached in CPU or GPU memory. Meshes are forwarded one at a time and
gradients are accumulated across meshes, avoiding padded ragged-graph batches.

## Validation and model selection

Validation uses held-out objects, never training meshes. When query
augmentation is enabled for validation, the aggregate validation curve can be
noisy; exact-query and perturbed-query losses should therefore be inspected
separately as well as together.

A few worsening validation events are not enough to stop a run. A plateau
decision should require a window of validation events in which both:

- training loss no longer improves materially; and
- validation best no longer improves materially.

The best checkpoint is selected by validation loss. Periodic checkpoints are
kept independently so later image and expanded-query ablations can compare the
same training stage.

## Required evaluation

GT-query validation alone does not prove the final objective. Every candidate
checkpoint should report:

- loss and relative improvement versus a zero predictor;
- `mean |prediction| / mean |GT|`;
- magnitude-binned error;
- high-10% target-magnitude cosine similarity;
- per-object metrics, not only an aggregate mean.

Image dependence must be tested with the query, graph and target fixed:

1. original RGB;
2. zero RGB;
3. shuffled view order;
4. cross-object RGB.

If all four results are similar, the model is relying primarily on query/graph
context. If original RGB wins but amplitude remains near zero, the image branch
works and the next issue is target/loss calibration.

Mesh-count scaling should use nested 1/2/4/8/16-object subsets with comparable
per-object exposure. It identifies whether output amplitude collapses as object
diversity increases.

Finally, apply the same checkpoint to the expanded inference manifest and
report reconstruction Chamfer, normal consistency and visual results. This is
the step that tests transfer from GT training graphs to arbitrary inference
graphs.

## Diagnostics

```bash
python scripts/ablate_single_mesh_checkpoint_images.py --help
python scripts/run_mesh_count_scaling.py --help
python scripts/diagnose_laplacian_prediction.py --help
python scripts/render_image_ablation_reconstructions.py --help
```

The optional magnitude-weighted Huber experiment is a diagnostic alternative,
not the production objective. Its loss values are not directly comparable to
unweighted Huber because target-magnitude weighting changes the metric.

## Outputs and monitoring

The run directory contains:

- `best.pt`;
- `checkpoint_epoch_*.pt`;
- `config.json`, `run_config.json` and `dataset_manifest.json`;
- `training_history.json` and `metrics.json` after completion;
- per-object prediction arrays after final evaluation;
- the launcher/service log used for live monitoring.

Follow the current run with:

```bash
tail -f runs/learned_laplacian/sofa50_refinement_960_gt_query_5000_full/training.log
```

The trainer reports data wait, image decode, GPU transfer,
forward/backward, total epoch time, validation time, used views, decoded bytes
and CPU/GPU peak memory.

## Operational constraints

- Training is one ragged mesh forward at a time with gradient accumulation,
  not packed-graph batching.
- PNG decoding is currently the dominant steady-state cost at 960 pixels.
- Static graph preparation runs once and scales with mesh count and complexity.
- Automatic checkpoint resume is not implemented.
- A run must not start while dataset files are still being generated or moved.
- Coordinate and camera conventions must remain identical between GT training
  observations and coarse/expanded inference queries.

## Legacy code

Historical coarse-graph targets, closest-surface pseudo targets, oracle
refinement and single-object Bunny experiments remain useful tests. They are
not the production learned-Laplacian supervision contract and must not be used
as evidence of cross-object or expanded-query generalisation.

## Verification

```bash
PYTHONPATH=src conda run --no-capture-output -n test pytest -q
```

Focused learned-Laplacian tests cover lazy loading, GT-query leakage guards,
query perturbation bounds, Fourier encoding, image ablation, mesh-count
scaling, AMP and Sofa50 preparation.
