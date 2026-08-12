# Multi-View Laplacian Refinement

[English](README.md) | [简体中文](README.zh-CN.md)

Method specification: [Canonical Sofa50 pipeline](docs/CANONICAL_SOFA50_PIPELINE.md)

Training guide: [Multi-mesh GT-query training](docs/MULTI_MESH_TRAINING.md)

Visibility and recovery: [Visibility-aware recovery report](docs/VISIBILITY_AWARE_RECOVERY_REPORT.md)

Experiment metrics and run status: [Experiment data summary](docs/EXPERIMENT_DATA_SUMMARY.md)

View-count and query-resolution results: [Ablation report](runs/learned_laplacian/sofa50_c2f2_view_query_resolution_ablation_20k_seed7/analysis/REPORT.md)

28-view current-graph target/loss-space results: [H2 ablation report](runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7/analysis/REPORT.md) | [25-case visual overview](runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7/analysis/comparison_images/B_direct_raw_laplacian/overview_25.png)

## Project status

Status date: 2026-08-12.

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
| 14/28/56-view ablation | Complete | Best validation losses are 0.0139316, 0.0130296 and 0.0138104 for 14, 28 and 56 views. |
| Query-graph resolution ablation | Complete | Best validation losses are 0.0139316 for the GT alias, 0.0614830 for GT-sub1 and 0.0145840 for GT-adaptive. GT-sub2 was excluded from training. |
| 28-view + GT-adaptive combination | Complete | Best validation loss is 0.0131095; raw EPE is 0.002879 on five matched validation meshes. |
| 28-view current-graph H2 ablation | Complete | Direct raw-Laplacian training is best in the unified test/recovery evaluation: raw EPE 0.00300525, refined Chamfer 0.00380671 and 19/25 improved samples. |
| Automated tests | Passing | `219 passed, 3 skipped` in the `test` Conda environment. |

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

## Mathematical specification

The equations below describe the implemented code path. Legacy terms are marked
explicitly.

### Uniform Laplacian and supervised target

Let `N(i)` be the one-ring neighbours of vertex `i`, and let
`d_i = |N(i)|`. The uniform graph Laplacian is

$$
(L X)_i = X_i - \frac{1}{d_i}\sum_{j\in N(i)}X_j,
\qquad
L_{ii}=1,\quad L_{ij}=-\frac{1}{d_i}.
$$

An isolated vertex has a zero Laplacian row. The local edge scale and the
absolute supervised target are

$$
h_i = \frac{1}{d_i}\sum_{j\in N(i)}\lVert V_i-V_j\rVert_2,
\qquad
\delta_i^{\mathrm{GT}}=(L_{\mathrm{GT}}V_{\mathrm{GT}})_i,
$$

$$
\widehat{\delta}_i^{\mathrm{GT}}
=\frac{\delta_i^{\mathrm{GT}}}{h_i^2+\varepsilon},
\qquad \varepsilon=10^{-12}.
$$

The network predicts the absolute normalised vector
`delta_hat_prediction`; it does not predict a displacement or Laplacian
residual. For a current inference graph,

$$
\delta_i^{\mathrm{pred}}
=\widehat{\delta}_i^{\mathrm{pred}}\left((h_i^{\mathrm{current}})^2+\varepsilon\right).
$$

This denormalisation is applied exactly once, using the current query graph.

### Query perturbation

For perturbed GT queries,

$$
q_i=V_i+h_i\left(\xi_i n_i+\zeta_i t_i\right),
\qquad
\xi_i\sim\mathcal N(0,\sigma_n^2),\quad
\zeta_i\sim\mathcal N(0,\sigma_t^2),
$$

where `n_i` is the vertex normal and `t_i` is a random unit tangent obtained by
removing the normal component from a Gaussian 3D direction. The displacement is
clamped to

$$
\lVert q_i-V_i\rVert_2\leq \kappa h_i.
$$

The canonical settings are
`sigma_n = sigma_t = 0.0003`, `kappa = 0.001`, and an exact-query fraction of
`0.2`. Exact queries set `q_i = V_i`. Perturbation changes the query position,
not the graph or target.

### Projection, renderer visibility and multi-view aggregation

For view `v`, world-to-camera projection is

$$
y_{vi}=E_v[q_i^\top,1]^\top,
\qquad
\widetilde p_{vi}=K_v y_{vi},
\qquad
(u_{vi},v_{vi})=
\left(\frac{\widetilde p_{vi,x}}{\widetilde p_{vi,z}},
      \frac{\widetilde p_{vi,y}}{\widetilde p_{vi,z}}\right).
$$

Let `f_vi` indicate positive depth and in-frame projection. Let `r_vi` be the
precomputed renderer-native back-face and occlusion result. The feature-sampling
mask is

$$
z_{vi}=f_{vi}r_{vi}\in\{0,1\}.
$$

If `F_v(u_vi,v_vi)` is the bilinearly sampled CNN feature, the implemented
masked mean and valid-view ratio are

$$
\overline F_i=
\frac{\sum_{v=1}^{M}z_{vi}F_v(u_{vi},v_{vi})}
     {\max\left(1,\sum_{v=1}^{M}z_{vi}\right)},
\qquad
\rho_i=\frac{1}{M}\sum_{v=1}^{M}z_{vi}.
$$

When no view is valid, `F_bar_i = 0` and `rho_i = 0`.

For C2F2, the image encoder is

$$
F_v=\mathrm{Conv}_{3\times3}^{64}\!\left(
\mathrm{ReLU}\!\left(
\mathrm{Conv}_{3\times3,s=1}^{64}\!\left(
\mathrm{ReLU}\!\left(
\mathrm{Conv}_{5\times5,s=1}^{32}(I_v)
\right)\right)\right)\right).
$$

Padding preserves the input spatial resolution in all three convolutions.

### Visibility, confidence and Gaussian gates

The canonical renderer gate is a strict any-view gate:

$$
m_i=\mathbf 1\!\left[\sum_{v=1}^{M}z_{vi}>0\right].
$$

The optional confidence head predicts a bounded reliability value

$$
c_i=\mathrm{sigmoid}(g_\theta(x_i))\in[0,1].
$$

The canonical recovery weight is

$$
w_i=m_i c_i,
$$

or `w_i = m_i` when the confidence head is disabled. Consequently, a vertex
that is invisible in every view has exactly zero learned-Laplacian weight.

The repository also implements the following Gaussian distance-confidence gate
in the legacy coarse/GT projection path:

$$
g_i=\mathrm{clip}\!\left(
\exp\!\left[-\left(\frac{d_i^{\mathrm{surface}}}{s}\right)^2\right],
g_{\min},1\right).
$$

Here `d_i^surface` is the distance from a coarse query to the GT surface and
`s` is `distance_confidence_scale`. This Gaussian gate is not renderer
visibility and is not used by the canonical GT-query training path. Canonical
training uses `z_vi` for image-feature sampling; canonical recovery uses
`m_i c_i`.

### Vertex representation and graph network

Let `c_obj` and `s_obj` be the object normalisation centre and scale. The
normalised query is

$$
\widetilde q_i=\frac{q_i-c_{\mathrm{obj}}}{s_{\mathrm{obj}}}.
$$

With `K = 6` frequencies, the dynamic Fourier encoding is

$$
\phi(\widetilde q_i)=
\left[
\widetilde q_i,
\left\{\sin(2^k\pi\widetilde q_i),
\cos(2^k\pi\widetilde q_i)\right\}_{k=0}^{K-1}
\right].
$$

The per-vertex input is

$$
x_i=\left[
\phi(\widetilde q_i),\ n_i,\
\log\!\left(\max(h_i/s_{\mathrm{obj}},10^{-8})\right),\
\log(1+d_i),\ \rho_i,\ \overline F_i
\right].
$$

For C2F2, `phi` has 39 channels and the complete vertex input has
`39 + 3 + 1 + 1 + 1 + 64 = 109` channels. The graph backbone is
`109 -> 256 -> 256`, followed by three 256-channel message-passing blocks and
an output MLP `256 -> 256 -> 3`. The confidence side head is
`109 -> 256 -> 1` with a final sigmoid.

After an input MLP, graph layer `l` computes

$$
\mu_i^{(l)}=\frac{1}{\max(1,d_i)}
\sum_{j\in N(i)}u_j^{(l)},
$$

$$
u_i^{(l+1)}=operatorname{ReLU}\!\left(
u_i^{(l)}+operatorname{MLP}_l
\left([u_i^{(l)},\mu_i^{(l)}]\right)
\right).
$$

The output MLP maps the final graph state to
`delta_hat_prediction in R^3`.

### Training objective

For component residual

$$
r_{ik}=\widehat\delta^{\mathrm{pred}}_{ik}
-\widehat\delta^{\mathrm{GT}}_{ik},
$$

the component-wise Huber function is

$$
H_\tau(r)=
\begin{cases}
\frac{1}{2}r^2, & |r|\leq\tau,\\
\tau\left(|r|-\frac{1}{2}\tau\right), & |r|>\tau,
\end{cases}
\qquad \tau=0.01.
$$

The per-vertex error and primary loss are

$$
e_i=\frac{1}{3}\sum_{k=1}^{3}H_\tau(r_{ik}),
\qquad
\mathcal L_{\mathrm{lap}}=
\frac{\sum_i a_i e_i}{\max(10^{-12},\sum_i a_i)},
$$

where `a_i` is the prepared target-confidence/valid-scale weight. The current
full-vertex GT-query contract assigns unit weight to valid non-isolated
vertices and zero weight to invalid local scales.

The confidence side head uses detached prediction error:

$$
\widetilde c_i=\mathrm{clip}(c_i,c_{\min},1),
$$

$$
\mathcal L_{\mathrm{conf}}=
\frac{\sum_i a_i
\left[\widetilde c_i\,\mathrm{stopgrad}(e_i)
-\beta\log\widetilde c_i\right]}
{\max(10^{-12},\sum_i a_i)}.
$$

The complete optimisation objective is

$$
\mathcal L_{\mathrm{train}}
=\mathcal L_{\mathrm{lap}}
+\lambda_{\mathrm{conf}}\mathcal L_{\mathrm{conf}},
$$

with `beta = 0.01`, `c_min = 10^-4`, and `lambda_conf = 1` in the canonical
configuration. Predicted confidence does not reweight
`L_lap`; this prevents the confidence head from suppressing the primary
supervision.

### Laplacian recovery objective

For the fixed current graph `(X_0, F)`, construct `L_current` and
`h_current`, then recover vertex positions `X` from `delta_pred`. The canonical
dense objective is

$$
\mathcal L_{\mathrm{rec}}(X)=
\lambda_{\mathrm{lap}}
\sum_{i,k}H_\tau\!\left(
\sqrt{w_i}\left[(L_{\mathrm{current}}X)_{ik}
-\delta_{ik}^{\mathrm{pred}}\right]\right)
+\frac{\lambda_{\mathrm{anchor}}}{2}\lVert X-X_0\rVert_F^2
+\mathcal L_{\mathrm{edge}}+\mathcal L_{\mathrm{unseen}}.
$$

The current canonical values are `lambda_lap = 1`,
`lambda_anchor = 0.01`, `lambda_edge = 0`, and
`lambda_unseen_anchor = 0`. The visibility/confidence weight applies to the
complete Laplacian equation row through `sqrt(w_i)`.

For large uniform-Laplacian meshes, the sparse solver uses the corresponding L2
form:

$$
\mathcal L_{\mathrm{sparse}}(X)=
\frac{\lambda_{\mathrm{lap}}}{N}
\left\lVert W^{1/2}(L_{\mathrm{current}}X-\delta^{\mathrm{pred}})\right\rVert_F^2
+\frac{\lambda_{\mathrm{anchor}}}{N}\lVert X-X_0\rVert_F^2,
\qquad W=\mathrm{diag}(w).
$$

### Reported metrics

For predicted and target normalised Laplacians `P` and `T`, the principal
prediction metrics are

$$
\mathrm{EPE}=\frac{1}{N}\sum_i\lVert P_i-T_i\rVert_2,
$$

$$
\mathrm{Cos}_{\mathrm{global}}=
\frac{\langle\mathrm{vec}(P),\mathrm{vec}(T)\rangle}
{\lVert P\rVert_F\lVert T\rVert_F},
\qquad
R_{\mathrm{norm}}=\frac{\lVert P\rVert_F}{\lVert T\rVert_F}.
$$

The reported bidirectional vertex-to-surface Chamfer value is

$$
D_{\mathrm{C}}(A,B)=\frac{1}{2}\left[
\frac{1}{|V_A|}\sum_{x\in V_A}d(x,S_B)
+\frac{1}{|V_B'|}\sum_{y\in V_B'}d(y,S_A)
\right],
$$

where `S_A` and `S_B` are triangle surfaces and `V_B'` is the evaluated or
subsampled GT vertex set.

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

### 28-view current-graph target and loss-space ablation

Three C2F2 arms use the same 28-view synthetic-current manifest, seed,
initialisation and 20,000-step budget. Local query jitter is disabled. Native
validation losses are reported in each arm's own loss space and are therefore
not comparable across rows.

| Arm | Output target | Native loss space | Best native val | Test raw EPE ↓ | Test raw cosine ↑ | Refined Chamfer ↓ | Improved |
|---|---|---|---:|---:|---:|---:|---:|
| A | `h^2`-normalised | Normalised output | 0.0184566 | 0.00769237 | 0.933526 | 0.00456011 | 3/25 |
| B | Raw Laplacian | Raw output | 1.58253e-6 | 0.00300525 | 0.998667 | 0.00380671 | 19/25 |
| C | `h^2`-normalised | Raw Laplacian | 2.16552e-6 | 0.00333673 | 0.997419 | 0.00383121 | 16/25 |

The shared initial Chamfer is `0.00391323`. Arm B is the primary result: it has
the lowest unified raw-space errors and recovery Chamfer, and improves 19 of 25
test samples. The very small B/C native losses reflect raw-Laplacian units;
they do not imply a four-order-of-magnitude advantage over A's normalised loss.
The contract audit passed, and the final evaluation ran as three L40 shards
(Slurm array 15686) followed by merge job 15687.

The local result bundle includes the [full report](runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7/analysis/REPORT.md),
[source JSON/CSV tables](runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7/analysis),
[75 comparison OBJ files](runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7/analysis/mesh_comparisons/B_direct_raw_laplacian)
and [25 fixed-camera GT/COARSE/REFINED images](runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7/analysis/comparison_images/B_direct_raw_laplacian).

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

# 14/28/56-view and query-resolution ablations, 20,000 steps per arm
sbatch scripts/HPC/c2f2_dataset_ablation_20k.slurm view 14
sbatch scripts/HPC/c2f2_dataset_ablation_20k.slurm query gt_sub1

# Three-L40 sharded H2 evaluation and dependent merge
bash scripts/HPC/submit_sofa50_synthetic_current_28view_h2_ablation_3gpu.sh
```

### Distributed multi-GPU training

`train_multi_mesh_laplacian.py` supports PyTorch DistributedDataParallel when
started with `torchrun`. Training meshes are sharded by rank with deterministic
padding, gradients and training metrics are reduced across ranks, and only rank
0 writes logs, predictions and checkpoints. Checkpoints retain canonical model
keys without a `module.` prefix and remain loadable by single-GPU evaluation.

The Slurm entry point defaults to one node with four L40 GPUs:

```bash
sbatch scripts/HPC/train_multi_mesh_ddp.slurm \
  /path/to/manifest.json \
  /path/to/config.json \
  /path/to/output_dir
```

The same script supports multiple nodes. This example launches eight ranks on
two nodes:

```bash
sbatch --nodes=2 --gres=gpu:L40:4 \
  scripts/HPC/train_multi_mesh_ddp.slurm \
  /path/to/manifest.json \
  /path/to/config.json \
  /path/to/output_dir
```

The global mesh batch is `world_size * gradient_accumulation_meshes`.
`max_optimizer_steps` counts synchronized global optimizer updates; increasing
the world size therefore increases the number of mesh exposures per update and
per fixed optimizer-step budget.

The generated 14/28/56-view dataset is located at:

```text
/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/sofa_mesh/sofa50_refinement/multiview_nested_14_28_56_cpu_v3
  gt_query_views_14_manifest.json
  gt_query_views_28_manifest.json
  gt_query_views_56_manifest.json
  expanded_inference_views_14_manifest.json
  expanded_inference_views_28_manifest.json
  expanded_inference_views_56_manifest.json
```

The generated query-graph resolution dataset is located at:

```text
/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/sofa_mesh/sofa50_refinement/multiview_960/query_resolution_ablation_v2
  gt_manifest.json
  gt_sub1_manifest.json
  gt_sub2_manifest.json
  gt_adaptive_manifest.json
```

The combined 28-view + GT-adaptive dataset is located at:

```text
/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/sofa_mesh/sofa50_refinement/view_query_combo_28_gt_adaptive_v1
  manifest.json
  summary.json
```

Each manifest contains 50 objects with a 40/5/5 train/validation/test split.
Prepared graphs contain CUDA-generated `visibility_backface_and_occlusion`.
The manifests passed the downstream training and inference loader checks.

### Local query-position jitter ablation

The ablation uses C2F2, 28 views, five fixed synthetic-current variants per
object, seed 7 and 20,000 optimizer steps. Arm A uses the stored current vertex
positions. Arm B applies training-only isotropic query jitter with component
standard deviation `0.003 h_i` and vector-norm limit `0.009 h_i`. The stored
proxy, normalized target, `h_i`, connectivity and target-construction operator
are unchanged. Validation and test do not apply jitter.

```text
Dataset: /networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_v1/manifest.json
Runs: runs/learned_laplacian/sofa50_synthetic_current_28view_jitter_ablation_seed7
Repository report: docs/SOFA50_LOCAL_QUERY_JITTER_ABLATION_REPORT.zh-CN.md
HPC report: runs/learned_laplacian/sofa50_synthetic_current_28view_jitter_ablation_seed7/analysis/REPORT.md
```

The report contains deterministic validation/test prediction metrics, the
original-RGB/zero-RGB comparison and OpenMVS48 current-mesh recovery metrics.
Arm B records higher best validation loss, test raw endpoint, test raw Top-10%
endpoint, test raw Top-1% endpoint, OpenMVS refined Chamfer and OpenMVS P2S than
Arm A. Arm B records lower test raw global cosine and OpenMVS refined normal
consistency. Neither arm improves OpenMVS Chamfer over the initial meshes.

## Result locations on the HPC

```text
runs/learned_laplacian/sofa50_image_resolution_ablation_50000step
runs/learned_laplacian/sofa50_c2_f2_50000step_3seed
runs/learned_laplacian/sofa50_c2_f2_1920_20000step_3seed
runs/learned_laplacian/sofa50_cf_c2f2_comparison_full
runs/learned_laplacian/sofa50_c2f2_960_vs_1920_full
runs/learned_laplacian/sofa50_c2f2_view_query_combo_28_gt_adaptive_20k_seed7_v1
runs/learned_laplacian/sofa50_synthetic_current_28view_jitter_ablation_seed7
runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7
runs/learned_laplacian/sofa50_openmvs_coarse_14view_c2f2_48mesh_opengl_480
runs/learned_laplacian/sofa50_openmvs_coarse_14view_c2f2_48mesh_opengl_480_recovery1000
```

Checkpoints, prepared datasets and HPC result directories are not distributed
with the source repository.
