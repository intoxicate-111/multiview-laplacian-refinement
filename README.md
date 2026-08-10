# Multi-View Laplacian Refinement

[English](README.md) | [简体中文](README.zh-CN.md)

Method specification: [Canonical Sofa50 pipeline](docs/CANONICAL_SOFA50_PIPELINE.md)

Training guide: [Multi-mesh GT-query training](docs/MULTI_MESH_TRAINING.md)

Visibility and recovery: [Visibility-aware recovery report](docs/VISIBILITY_AWARE_RECOVERY_REPORT.md)

## Project status

Status date: 2026-08-10.

| Component | State | Conclusion |
|---|---|---|
| GT-query dataset and training pipeline | Implemented | Direct supervision of the absolute GT `h^2`-normalised Laplacian is operational. |
| Target-leakage controls | Implemented and tested | GT Laplacian values are excluded from model inputs. |
| Sofa50 960 image-resolution ablation | Complete | F2 has lower exact-query error than F0 and F1 at 50,000 optimiser steps. |
| Sofa50 960 C2F2 training | Complete | Three seeds completed at 50,000 optimiser steps. C2F2 is the current lowest-error exact-query configuration. |
| Sofa50 1920 C2F2 training | Complete | Three seeds completed at 20,000 optimiser steps. Mean endpoint error and recovery Chamfer are higher than the 960 result; mean cosine is higher. |
| Expanded-query recovery | Complete | Refinement increases Chamfer distance on all five validation meshes for the evaluated 960 and 1920 C2F2 checkpoints. |
| OpenMVS coarse-mesh recovery | Complete for 48 of 50 meshes | Mean Chamfer distance increases after refinement. Two objects have no OpenMVS coarse mesh. |
| Oracle residual expert | Closed | The 2,000-step diagnostic does not support this branch. |
| 14/28/56-view ablation | Blocked before training | Prepared samples lack `visibility_backface_and_occlusion`; expanded-inference manifests are absent. No checkpoint or result report was produced. |
| Automated tests | Passing | `216 passed, 3 skipped` in the `test` Conda environment. |

The implemented model learns the supervised differential field on GT-query
graphs and uses RGB information. Transfer from GT-query graphs to expanded or
OpenMVS query graphs has not produced a geometry improvement. The end-to-end
coarse-mesh refinement objective is not met by the current recovery path.

## Method

The model maps calibrated multi-view observations and a graph query to the GT
local differential signal:

```text
multi-view RGB + cameras + 3D query + local graph context
    -> absolute GT h^2-normalised Laplacian
```

For GT vertex `i`:

```text
delta_gt_i = (L_gt V_gt)_i
h_i        = mean incident GT edge length at i
target_i   = delta_gt_i / (h_i^2 + epsilon)
```

Training uses GT vertices and GT connectivity. A fixed fraction of queries
remain at exact GT positions; the other queries receive bounded normal and
tangent perturbations relative to `h_i`. The target remains attached to the
original GT vertex.

The current geometry mode is `query_fourier`. Fourier features are calculated
after query augmentation. Image features, query position, normal, relative
local scale, degree and graph connectivity are model inputs. The raw and
normalised GT Laplacians are supervision only. `initial_laplacian` is zero in
GT-query training samples.

Inference uses an independently produced coarse or topology-expanded mesh:

```text
coarse mesh vertices
  -> project into calibrated views
  -> aggregate image features
  -> predict normalised Laplacian
  -> denormalise with the current query graph scale
  -> confidence/visibility-weighted Laplacian recovery
```

No GT differential value is transferred to the inference graph. GT geometry is
used only for evaluation.

## Dataset contract

The Sofa50 data roots used on the HPC are:

```text
/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_960
/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_1920
```

Each dataset contains 40 training, 5 validation and 5 held-out test objects.
The standard 960 experiment uses 14 calibrated RGB views per object. Training
uses `gt_query_manifest.json`; expanded recovery uses
`expanded_inference_manifest.json`. Expanded manifests are inference-only.

RGB images remain on disk and are decoded lazily as `uint8`. CUDA training uses
pinned memory, non-blocking transfer and AMP for CNN/GNN forward passes.
Target scaling, loss accumulation and numerical geometry operations remain in
FP32.

## Experiment nomenclature

| Label | Definition |
|---|---|
| C0 | Image feature dimension 16; graph hidden dimension 64; 3 graph layers. |
| C2 | Image feature dimension 64; graph hidden dimension 256; 3 graph layers. |
| F0 | Encoder strides `2, 2`; 240 x 240 feature map for 960 input. |
| F1 | Encoder strides `2, 1`; 480 x 480 feature map for 960 input. |
| F2 | Encoder strides `1, 1`; feature-map resolution equals input resolution. |
| C2F2 | C2 capacity with the F2 image encoder. |

## Completed results

### Exact GT-query prediction at 960

| Run | Seed | All EPE ↓ | Top-10% EPE ↓ | Global cosine ↑ | Prediction/GT norm |
|---|---:|---:|---:|---:|---:|
| C0F0 | 7 | 9.4641 | 30.7221 | 0.7808 | 0.8020 |
| C0F1 | 7 | 9.3786 | 30.3095 | 0.7892 | 0.7938 |
| C0F2 | 7 | 9.1665 | 28.4751 | 0.8227 | 0.8180 |
| C2F2 | 7, 17, 27 | 2.8260 ± 0.0864 | 15.3614 ± 0.4036 | 0.8912 ± 0.0127 | 0.9348 ± 0.0160 |

Original RGB produces lower error than zero RGB in the completed resolution
ablation. The original-minus-zero global-cosine gaps are 0.2236, 0.3315 and
0.3724 for F0, F1 and F2, respectively.

### C2F2 at 960 and 1920

| Input | Training budget | Mean all EPE ↓ | Mean top-10% EPE ↓ | Mean cosine ↑ | Mean expanded Chamfer ↓ |
|---|---:|---:|---:|---:|---:|
| 960 | 50,000 steps, 3 seeds | 2.8282 | 15.3743 | 0.8911 | 0.0011624 |
| 1920 | 20,000 steps, 3 seeds | 3.0928 | 16.3299 | 0.8954 | 0.0012570 |

The two rows do not have the same optimiser-step budget. The 1920 runs do not
establish an improvement over the 960 runs.

### Expanded-query recovery

The shared five-object expanded-validation initial Chamfer is `0.000652884`.
The 960 C2F2 mean refined Chamfer is `0.00116202`; the 1920 C2F2 mean is
`0.00125704`. Each evaluated seed improves `0/5` meshes relative to its initial
mesh.

### OpenMVS coarse-mesh recovery

The test uses 48-view COLMAP/OpenMVS coarse meshes, the original 14 Sofa50 RGB
views for prediction, the three 960 C2F2 checkpoints, OpenGL visibility at 480,
and no GT differential transfer. Forty-eight meshes are evaluated; two coarse
meshes are absent.

| Recovery | Initial mean Chamfer | Ensemble refined mean Chamfer | Better meshes | Introduced flips |
|---|---:|---:|---:|---:|
| 200 iterations | 0.0212023 | 0.0213199 | 2/48 | 4,692 |
| 1,000 iterations | 0.0212023 | 0.0213198 | 2/48 | 4,734 |

Increasing recovery from 200 to 1,000 iterations does not change the aggregate
conclusion.

## Installation and verification

```bash
conda env create -f environment.yml
conda activate test
pip install -e ".[train]"
PYTHONPATH=src pytest -q
```

If the environment already exists:

```bash
PYTHONPATH=src conda run --no-capture-output -n test pytest -q
```

## HPC entry points

The checked-in Slurm files contain cluster-specific paths and resource
requests.

```bash
# 960 F0/F1/F2, 50,000 steps
bash scripts/slurm_jobs/submit_resolution_50k_parallel.sh

# 960 C2F2, seeds 7/17/27, 50,000 steps
sbatch scripts/HPC/sofa50_c2_f2_50k_3gpu.slurm

# 1920 C2F2, seeds 7/17/27, current 20,000-step output contract
sbatch scripts/HPC/sofa50_c2_f2_1920_50k_3gpu.slurm

# OpenMVS recovery with OpenGL visibility and 1,000 recovery iterations
sbatch scripts/HPC/test_sofa50_openmvs_coarse_14view_c2f2_48mesh_opengl_480_recovery1000.slurm
```

The 14/28/56-view job is not executable under its current data contract. It
requires renderer-visibility fields in the GT-query samples and matching
expanded-inference manifests.

## Result locations on the HPC

```text
runs/learned_laplacian/sofa50_image_resolution_ablation_50000step
runs/learned_laplacian/sofa50_c2_f2_50000step_3seed
runs/learned_laplacian/sofa50_c2_f2_1920_20000step_3seed
runs/learned_laplacian/sofa50_cf_c2f2_comparison_full
runs/learned_laplacian/sofa50_c2f2_960_vs_1920_full
runs/learned_laplacian/sofa50_openmvs_coarse_14view_c2f2_48mesh_opengl_480
runs/learned_laplacian/sofa50_openmvs_coarse_14view_c2f2_48mesh_opengl_480_recovery1000
```

Checkpoints, prepared datasets and HPC result directories are not distributed
with the source repository.
