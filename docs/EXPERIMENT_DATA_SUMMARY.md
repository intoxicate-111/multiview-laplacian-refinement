# Experiment data summary

[English](EXPERIMENT_DATA_SUMMARY.md) | [简体中文](EXPERIMENT_DATA_SUMMARY.zh-CN.md)

Status date: 2026-09-04 09:18 BST, Europe/London.

This document indexes the experiment data currently available in the local
workspace and on the HPC. A value marked `running snapshot` is not a final
result. Training losses are only compared within experiments that use the same
target, loss, split and evaluation path.

## Standard definitions

| Label | Implemented definition |
|---|---|
| C0 | Image feature dimension 16; graph hidden dimension 64; 3 graph layers. |
| C1 | Image feature dimension 32; graph hidden dimension 128; 3 graph layers. |
| C2 | Image feature dimension 64; graph hidden dimension 256; 3 graph layers. |
| F0 | Encoder strides `2,2`; 240 x 240 feature map for a 960 input. |
| F1 | Encoder strides `2,1`; 480 x 480 feature map for a 960 input. |
| F2 | Encoder strides `1,1`; feature-map resolution equals input resolution. |
| K0/K2/K4/K6 | Fourier position encoding with 0, 2, 4 or 6 frequencies. |

The historical canonical GT-query target is

$$
\delta_i=(LV)_i,
\qquad
\widehat{\delta}_i=\frac{\delta_i}{h_i^2+10^{-12}}.
$$

For the active synthetic-current experiment, the graph and raw target are both
defined on the current graph:

$$
\delta_i^{\mathrm{current}}=(L_cP_{\mathrm{proxy}})_i.
$$

The model directly predicts this raw field. `h_current` remains an input
geometry feature and audit quantity; it does not divide, clip or denormalize
the active target. The h-squared-normalized value is retained only for
historical comparisons and diagnostics.

## Dataset inventory

| Dataset | Objects and split | Views / resolution | State | Location |
|---|---|---|---|---|
| Sofa50 canonical GT-query | 50 objects; 40/5/5 | 14 / 960 | Complete | HPC: `sofa50_refinement/multiview_960` |
| Sofa50 1920 GT-query | 50 objects; 40/5/5 | 14 / 1920 | Complete | HPC: `sofa50_refinement/multiview_1920` |
| Nested view ablation | 50 objects; 40/5/5 | 14/28/56 / 960 | Complete | `sofa50_refinement/multiview_nested_14_28_56_cpu_v3` |
| Query-resolution ablation v2 | 50 objects; 40/5/5 | 14 / 960 | Complete | `multiview_960/query_resolution_ablation_v2` |
| Synthetic current-query, 14 views | 50 objects, 5 variants each; 200/25/25 variants | 14 / 960 | Complete and copied to HPC | `~/sofa_mesh/sofa50_synthetic_current` |
| Synthetic current-query, 28 views | 50 objects, 5 variants each; 200/25/25 variants | 28 / 960 | Complete | HPC: `sofa50_synthetic_current_28view_v1` |
| Synthetic current-query, native 1920 | Same 250 IDs and 200/25/25 split as 960 | 28 / 1920 | Complete; HF training/evaluation complete | HPC: `sofa50_synthetic_current_28view_native1920_v1` |
| Sofa50 multi-topology raw-Laplacian v1 | 50 objects, 10 variants each; 400/50/50 | 28 / 960 | Complete historical mild-smoothing dataset | HPC: `Sofa50MultiTopologyRawLap500_v1` |
| Sofa50 multi-topology raw-Laplacian v2 | Same objects, variants and split as v1 | 28 / 960 | 500/500 audited; 2×L40 20k training and unified v1-v2 test/recovery complete | HPC: `Sofa50MultiTopologyRawLap500_v2` |
| Future2000 GT-adaptive expanded current | 2,000 distinct objects, 5 frozen variants each; 8000/1000/1000 variants by object split | 28 / 960 | Formal Arm-B test and 200k Arm-E training complete; validation-only frozen B+E lambda sweep running | HPC: `future2000_gt_adaptive_synthetic_current_28view_v2` |
| OpenMVS coarse-query stress set | 48 available coarse meshes; 2 missing | Prediction uses the canonical 14 RGB views | Complete; diagnostic only, not a target | HPC: `openmvs_texture_test_v6_48view` |
| Thingi10K50 development set | 50 objects; 40/5/5 | 960 and 1920 variants | Development and smoke runs only | Local `thingi10k50` run directories |

The synthetic-current dataset contains 250 static samples. All samples pass the
target algebra check, current-graph `h^2` round trip, 14-view count, object-level
split and image-path checks.

## Completed canonical Sofa50 training

### Training metrics

`Best val loss` is the checkpoint-selection loss. It is not the endpoint error
in the next table.

| Run | Seed | Steps | Best epoch | Best val loss | Final train loss | Final val loss | Runtime h |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0F0, 960 | 7 | 50,000 | 5000 | 0.0365528 | 0.0361083 | 0.0365519 | 2.64 |
| C0F1, 960 | 7 | 50,000 | 4875 | 0.0358636 | 0.0350980 | 0.0358663 | 2.64 |
| C0F2, 960 | 7 | 50,000 | 5000 | 0.0349125 | 0.0348144 | 0.0349178 | 4.72 |
| C2F2, 960 | 7 | 50,000 | 4920 | 0.0126017 | 0.00571267 | 0.0126040 | 10.17 |
| C2F2, 960 | 17 | 50,000 | 4930 | 0.0132884 | 0.00646517 | 0.0132948 | 10.10 |
| C2F2, 960 | 27 | 50,000 | 4775 | 0.0133493 | 0.00710706 | 0.0133571 | 10.10 |
| C2F2, 1920 | 7 | 20,000 | 1995 | 0.0147794 | 0.00707957 | 0.0147794 | 14.82 |
| C2F2, 1920 | 17 | 20,000 | 2000 | 0.0145967 | 0.00613659 | 0.0146020 | 14.78 |
| C2F2, 1920 | 27 | 20,000 | 2000 | 0.0134383 | 0.00708745 | 0.0134342 | 14.79 |

HPC roots:

```text
runs/learned_laplacian/sofa50_image_resolution_ablation_50000step
runs/learned_laplacian/sofa50_c2_f2_50000step_3seed
runs/learned_laplacian/sofa50_c2_f2_1920_20000step_3seed
```

### Exact GT-query prediction

| Run | Seeds | All EPE ↓ | Top-10% EPE ↓ | Global cosine ↑ | Prediction/target norm |
|---|---|---:|---:|---:|---:|
| C0F0, 960 | 7 | 9.4641 | 30.7221 | 0.7808 | 0.8020 |
| C0F1, 960 | 7 | 9.3786 | 30.3095 | 0.7892 | 0.7938 |
| C0F2, 960 | 7 | 9.1665 | 28.4751 | 0.8227 | 0.8180 |
| C2F2, 960 | 7/17/27 | 2.82815 | 15.37434 | 0.89110 | 0.93480 |
| C2F2, 1920 | 7/17/27 | 3.09280 | 16.32997 | 0.89537 | 0.93118 |

The mean all-EPE and top-10% EPE are lower at 960. Mean global cosine is
higher at 1920. The budgets differ: 50,000 steps at 960 and 20,000 steps at
1920.

For the completed F0/F1/F2 experiment, the original-minus-zero RGB global
cosine gaps are `0.2236`, `0.3315` and `0.3724`. The models use RGB evidence.

## Recovery results

### Expanded-query Sofa50 validation

The shared initial Chamfer is `0.000652884`.

| Model | Seeds | Refined Chamfer ↓ | Point-to-surface ↓ | Normal consistency ↑ | Introduced flips | Improved meshes |
|---|---|---:|---:|---:|---:|---:|
| C2F2, 960 | 3 | 0.00116244 | 0.00118173 | 0.894626 | 4213.3 mean | 0/5 per seed |
| C2F2, 1920 | 3 | 0.00125695 | 0.00126905 | 0.892581 | 4264.0 mean | 0/5 per seed |

The same recovery configuration is recorded for both groups. Both groups
increase Chamfer relative to the initial expanded mesh.

### OpenMVS coarse meshes

These meshes have poor initial reconstruction quality and are no longer a
target or decision endpoint. They are retained only as labelled OOD stress
inputs; see [the OpenMVS input policy](OPENMVS_INPUT_POLICY.md). The historical
numbers below describe the executed diagnostic and do not rank target quality.

| Recovery iterations | Meshes | Initial Chamfer | Ensemble refined Chamfer | Better meshes | Ensemble introduced flips |
|---:|---:|---:|---:|---:|---:|
| 200 | 48 | 0.0212023 | 0.0213199 | 2/48 | 4,692 |
| 1,000 | 48 | 0.0212023 | 0.0213198 | 2/48 | 4,734 |

Objects `8ecad62d-fd41-4d86-87f0-5f640c46f238` and
`d7e2c96f-76cd-4699-bbe7-c65f7cb8b8cd` have no OpenMVS coarse mesh. Increasing
the recovery iterations from 200 to 1,000 does not change the aggregate result.

The later projected-GT failure decomposition reports unified Chamfer
`0.0469163` (initial), `0.0440446` (projected-GT position oracle), `0.0456376`
(projected-GT oracle Laplacian on the OpenMVS graph after frozen recovery) and
`0.0467913` (archived learned prediction). Recovery retains 44.53% of the
position-oracle gain; the learned arm realizes 4.36%. These are diagnostic
attributions on a poor OOD input and carry zero model-selection weight.

## Completed ablations

### Model capacity, C0/C1/C2, 2,000 steps

| Capacity | Best val loss ↓ | All EPE ↓ | Top-10% EPE ↓ | Global cosine ↑ | Pred/target norm |
|---|---:|---:|---:|---:|---:|
| C0 | 0.0478404 | 11.1322 | 39.2638 | 0.6700 | 0.6730 |
| C1 | 0.0446807 | 10.5003 | 35.9503 | 0.7137 | 0.7200 |
| C2 | 0.0428904 | 10.0454 | 35.4161 | 0.7193 | 0.7490 |

Detailed local report:
[capacity ablation](../runs/learned_laplacian/sofa50_capacity_ablation_2000step/analysis_v2/REPORT.md).

### Local position encoding, C1F1, 14 views, 960, 2,000 steps

| Encoding | Best epoch | Best/final val loss ↓ | Final train loss |
|---|---:|---:|---:|
| K0 | 195 | 0.0461422 | 0.0479796 |
| K2 | 195 | 0.0452735 | 0.0469340 |
| K4 | 170 | 0.0449300 | 0.0455100 |
| K6 | 170 | 0.0457297 | 0.0460866 |

K4 has the lowest validation loss in this local C1F1 screening. This result is
not a C2F2 comparison.

### Query-graph resolution, C2F2, 14 views, seed 7, 20,000 steps

| Query graph | State | Best epoch | Best val loss ↓ | Final train loss | Final val loss | Runtime h |
|---|---|---:|---:|---:|---:|---:|
| GT | Represented by the equivalent 14-view arm | 1995 | 0.0139316 | 0.00707592 | 0.0139314 | 3.15 |
| GT-sub1 | Complete | 1905 | 0.0614830 | 0.0580221 | 0.0614822 | 4.25 |
| GT-adaptive | Complete | 1790 | 0.0145840 | 0.00640667 | 0.0145877 | 3.95 |
| GT-sub2 | Excluded by experiment decision | — | — | — | — | — |

HPC root:
`runs/learned_laplacian/sofa50_c2f2_query_resolution_gt_sub1_adaptive_20k_seed7_v2`.

### View count, C2F2, seed 7, 20,000 steps

| Views | State | Best val loss ↓ | Final/current train loss | Final/current val loss | GPU memory MiB |
|---:|---|---:|---:|---:|---:|
| 14 | Complete | 0.0139316 | 0.00707592 | 0.0139314 | 9,095 |
| 28 | Complete | 0.0130296 | 0.00660375 | 0.0130341 | 18,130 |
| 56 | Complete | 0.0138104 | 0.006991 | 0.013812 | 31,692 |

All three view-count arms reached 20,000 optimizer steps. The 28-view arm has
the lowest best validation loss; the 56-view arm has the lowest unified raw
EPE and raw Top-10% EPE. HPC root:
`runs/learned_laplacian/sofa50_c2f2_views_14_28_56_20k_seed7_v4`.

## Synthetic current-query comparisons

### Frozen GT-query 50k versus current-query 20k

The final 14-view evaluation uses 25 matched synthetic-current test samples.
The training budgets differ and the comparison therefore does not isolate the
formulation from the budget.

| Metric | GT-query 50k | Current-query 20k |
|---|---:|---:|
| Evaluation loss | 0.0145788 | 0.0117459 |
| Vector L2 | 2.994356 | 2.391482 |
| Global cosine | 0.883605 | 0.895129 |
| Initial Chamfer | 0.00391323 | 0.00391323 |
| Refined Chamfer | 0.00551727 | 0.00417930 |
| Improved samples | 0/25 | 5/25 |

Current-query training improves the recorded prediction and recovery metrics
relative to the frozen GT-query checkpoint, but its mean refined Chamfer still
exceeds the shared initial Chamfer.

### 28-view current-graph H2 target/loss-space ablation

All three C2F2 arms use the same 28-view manifest, split IDs, seed,
initialisation, optimizer, scheduler, batching and 20,000-step budget. Local
query jitter is disabled and the contract audit passes. Native validation
losses use different spaces and must not be compared across arms.

| Arm | Output target | Native loss space | Best native val | Runtime h |
|---|---|---|---:|---:|
| A: canonical H2 | `h^2`-normalised | Output representation | 0.018456638 | 6.0416 |
| B: direct raw | Raw Laplacian | Output representation | 1.5825285e-6 | 6.1807 |
| C: normalised output/raw loss | `h^2`-normalised | Raw Laplacian | 2.1655217e-6 | 6.6896 |

Unified test raw-space prediction:

| Arm | Raw EPE ↓ | Top-1% EPE ↓ | Top-10% EPE ↓ | Raw cosine ↑ | Weighted raw RMS ↓ |
|---|---:|---:|---:|---:|---:|
| A | 0.00769237 | 0.253855 | 0.0557517 | 0.933526 | 0.0427999 |
| B | 0.00300525 | 0.0417512 | 0.0136982 | 0.998667 | 0.00611072 |
| C | 0.00333673 | 0.0547519 | 0.0159651 | 0.997419 | 0.00815502 |

Zero-replacement recovery uses a shared initial Chamfer of `0.00391323`:

| Arm | Refined Chamfer ↓ | P2S ↓ | Normal consistency ↑ | Flips | Improved/25 |
|---|---:|---:|---:|---:|---:|
| A | 0.00456011 | 0.00462286 | 0.934976 | 10,195 | 3/25 |
| B | 0.00380671 | 0.00380587 | 0.942406 | 6,566 | 19/25 |
| C | 0.00383121 | 0.00385409 | 0.941080 | 7,057 | 16/25 |

B is the primary result: it has the lowest unified raw-space error and refined
Chamfer and improves 19/25 samples. C is between B and A. The small B/C native
loss values result from raw-Laplacian units, not a directly comparable
four-order-of-magnitude loss reduction.

The three GPU shards ran as Slurm array 15686 on three L40 GPUs in parallel
(`00:19:06`–`00:19:25` per shard); merge job 15687 completed in `00:00:15`.
The local [report](../runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7/analysis/REPORT.md),
[JSON/CSV records](../runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7/analysis),
[75 OBJ meshes](../runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7/analysis/mesh_comparisons/B_direct_raw_laplacian)
and [25-case overview](../runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7/analysis/comparison_images/B_direct_raw_laplacian/overview_25.png)
are available in the workspace.

### Stage-2 distribution adaptation and Huber saturation

The three matched continuation arms add 20,000 steps to the same frozen Arm-B
checkpoint. None exceeds the original result:

| Arm, best checkpoint | Raw EPE | Chamfer | Normal | Improved | Retained | Gained | Lost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen stage-1 B | 0.00300521 | **0.00380687** | **0.942463** | **19/25** | 19/19 | 0/6 | 0/19 |
| Continue X0 | 0.00369851 | 0.00390257 | 0.931793 | 16/25 | 14/19 | 2/6 | 5/19 |
| Continue X1 | **0.00349257** | **0.00384032** | **0.936939** | 16/25 | 14/19 | 2/6 | 5/19 |
| Continue 50/50 | 0.00363284 | 0.00388119 | 0.934341 | 16/25 | 14/19 | 2/6 | 5/19 |

The X1 arm is the best continuation but remains worse than frozen stage-1 in
mean Chamfer, P2S, normal consistency and improved count. The result rejects
the current stage-2 recipe, not the general idea of distribution adaptation.

The Arm-B validation Huber diagnostic covers 243,000 vertices. GT raw-
Laplacian top-1% vertices have `13.071x` the bottom-90% mean raw error,
`66.049%` any-component saturation, `58.436%` gradient retention, `34.931%`
of Huber loss and `5.785%` of output-gradient L1. The compression is strongly
concentrated in the top 1%, rather than uniformly across the full top 10%.

### Huber versus raw MSE

The completed 25-sample unified test does not show a tail or mean-geometry
benefit from raw MSE:

| Loss | Raw EPE ↓ | Raw RMS ↓ | Top-10% ↓ | Top-1% ↓ | Chamfer ↓ | Normal ↑ | Flips | Improved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Huber, 0.01 | **0.00297478** | **0.00662604** | **0.0122438** | **0.0371716** | **0.00380692** | 0.942431 | 6,579 | **19/25** |
| Raw MSE | 0.00297688 | 0.00695997 | 0.0128078 | 0.0380294 | 0.00381317 | **0.943833** | **5,925** | 16/25 |

MSE uses global batch 6 and a 510-step validation interval, versus 2 and 500
for Huber. The audit therefore records a resource-driven non-strict training
comparison; the unified held-out evaluation remains directly comparable.

### Learned dynamic residual expert and gate causality

The learned final improves the jointly trained base from raw EPE `0.00450175`
to `0.00294740`, Chamfer `0.00416138` to `0.00377438`, normal consistency
`0.926613` to `0.944879`, and improved count `3/25` to `19/25`. The base branch
is not the frozen original Arm B, so this large within-model change must not be
reported as the gain over the original baseline.

Validation selects constant gate `alpha=0.16`. Base-to-constant improves raw
EPE, Chamfer, P2S and normal on `25/25`; constant-to-learned improves Chamfer
and P2S on `25/25` and raw EPE on `24/25`. Learned placement also beats every
one of five within-mesh gate shuffles on most paired samples. The causal result
supports both an effective residual expert and a smaller additional spatial-
placement contribution; gate/curvature correlations are observational only.

### 960 Gaussian and high-frequency image features

| Feature | Raw EPE ↓ | Raw RMS ↓ | Bottom 90% ↓ | Top 10% ↓ | Top 1% ↓ | Chamfer ↓ | Improved |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 0.00297471 | 0.00662531 | 0.00194494 | 0.0122427 | 0.0371654 | 0.00380683 | 19/25 |
| Gaussian | 0.00291322 | 0.00666042 | **0.00186460** | 0.0123509 | 0.0376811 | **0.00377507** | **21/25** |
| Original + HF | **0.00288627** | **0.00628246** | 0.00190114 | **0.0117524** | **0.0347902** | 0.00377832 | 20/25 |

Gaussian-only features improve mean downstream geometry but degrade the two
tail groups relative to original. Concatenating the original feature and
`F-Gaussian(F)` gives the strongest raw prediction and tail metrics without a
material Bottom-90% penalty.

### Native-1920 plus HF, completed

The native renderer reproduces the 960 HF sample IDs, 28 camera extrinsics,
split, graph, proxy, target and visibility contracts. Intrinsics are scaled for
1920, and the minimum native-versus-resized pixel MAE is `0.0205764`, rejecting
the resize-only path. Job 15854 completed a from-scratch four-L40 run with
20,000 global optimiser steps.

The 1920 run has global batch 4 versus the 960 baseline's 2. View chunking and
gradient checkpointing are tested as mathematically equivalent execution
changes, but the batch difference makes the training comparison non-strict.

| Resolution + HF | Raw EPE ↓ | Raw RMS ↓ | Bottom 90% ↓ | Top 10% ↓ | Top 1% ↓ | Chamfer ↓ | P2S ↓ | Normal ↑ | Flips | Improved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 960 | **0.00288618** | **0.00628203** | 0.00190107 | **0.0117522** | **0.0347895** | **0.00377857** | **0.00377999** | 0.942504 | 6303 | **20/25** |
| Native 1920 | 0.00290615 | 0.00690893 | **0.00183806** | 0.0125190 | 0.0389263 | 0.00378509 | 0.00378489 | **0.944522** | **5777** | 18/25 |

Native 1920 does not improve Top-10%, Top-1%, raw RMS, recovery-weighted RMS,
Chamfer or P2S. It improves normal consistency and reduces flips. Runtime rises
from 3.98 hours on two GPUs (`7.95` GPU-hours) to 22.35 hours on four GPUs
(`89.39` GPU-hours).

### GT-query direct-raw zero-shot transfer

A separate two-Blackwell, 20,000-step control trains on exact GT queries with
the raw target `L_gt @ V_gt`, then applies the frozen model to the current mesh.
The contract audit passes and GT is introduced only after prediction for
surface evaluation. Correct/zero/shuffled RGB controls confirm that image
features affect the held-out prediction.

| Arm | Current-mesh Chamfer ↓ | P2S ↓ | Normal ↑ | Flips | Improved |
|---|---:|---:|---:|---:|---:|
| Historical GT-query `h^2` normalized | 0.00581764 | 0.00606854 | 0.922509 | 11795 | 0/25 |
| GT-query direct raw + HF | 0.00400486 | 0.00401379 | **0.948067** | **3087** | 4/25 |
| Current-query direct raw + HF | **0.00377832** | **0.00377984** | 0.942475 | 6326 | **20/25** |

Removing normalization materially improves the historical GT-query transfer,
but does not close the query-distribution gap to supervised current-query
training. The historical arm predates HF and therefore is not a strict
single-variable normalization ablation.

### Strong-smoothing recovery diagnosis and recovery-aware A-E study

Exact target plus all-equation sparse integration establishes that the v2 raw
Laplacian is recoverable: mean oracle efficiency is `0.92366` with a component-
centroid translation gauge. In the frozen solver, hard visibility is the
largest incremental loss (`0.34258 -> 0.16875` mean eta; 44/50 worse).
Confidence is negligible, and 2,000 Adam steps reach only `0.18635`.

The completed A/B study uses all rows, `lambda=10^-2`, no confidence and no
recovery Huber/Adam. Arm B adds `beta=10^-2` same-index recovered-vertex MSE.

| Test metric | A Lap only | B Lap + vertex |
|---|---:|---:|
| Raw EPE | **0.00252641** | 0.00263986 |
| Raw RMS | 0.00737725 | **0.00683290** |
| Chamfer | 0.00395529 | **0.00358497** |
| Eta | 0.07206 | **0.13036** |
| P2S p95 | 0.0122582 | **0.0105581** |
| Normal | 0.954902 | **0.959366** |
| Vertex RMS | 0.0135181 | **0.0115532** |

B wins Chamfer on 32/50 and vertex RMS on 43/50, but raw EPE on only 10/50.
The matched A-E study is complete: weakening the recovery anchor in C/D worsens
geometry, while direct-vertex E reaches Chamfer `0.00334039`. Frozen B+E reaches
`0.00302983`; the later scalar-fusion control reaches `0.00318814`, confirming
that the operator hybrid is not explained by a single global vertex average.

Three formulation stress tests now narrow that interpretation. Direct-Lap
A+E is not meaningfully separated from B+E in paired surface distance at
either `lambda=0.03` (CD `0.00298590` versus `0.00302983`) or `lambda=0.01`
(`0.00314166` versus `0.00319840`); both CD confidence intervals include zero.
Changing the recovery-loss anchor during B_P training gives no same-anchor CD
separation from B_0 (`-0.00001703`, mesh/object CIs cross zero). Finally, the
sparse positional experiment shows a smooth density curve: fixed-lambda test
CD falls monotonically from `0.0330216` at 0% to `0.00302983` at 100%, while
the Song-scale 2% condition remains `243.70%` above dense and loses 0/50.
Together these results support dense learned anchoring as the measured
advantage, but not a claim that recovery-aware Arm-B training is necessary for
operator composition.

## Future2000 GT-adaptive scale-up

Status updated 2026-09-04 09:18 BST. The dataset contains 2,000 distinct 3D-FUTURE
source objects and five frozen deterministic current-mesh perturbation variants
per object. Object-level splits contain 8,000 train, 1,000 validation and 1,000
test meshes. Each object's variants share its 28 calibrated 960-pixel RGB
observations, while current geometry, connectivity, query graph and visibility
remain variant specific.

The archived old-structure job `16607` produced Chamfer `0.00522954770` and
959/1000 improvements. The formal current Arm B instead retains the established
mixed objective `L_raw-Laplacian-Huber + 10^-2 L_recovered-vertex` and uses the
validation-selected epoch-195 checkpoint (SHA-256
`fa934cd44c4009dd392c415fe2c5f731c8cf1b78cda6a31fab199d4c15510b82`).

| Full test system | Chamfer ↓ | P2S p95 ↓ | F-score ↑ | Normal ↑ | Improved |
|---|---:|---:|---:|---:|---:|
| Initial mesh | 0.00776417127 | — | — | 0.924252350 | — |
| Archived old-structure Ours | 0.00522954770 | — | — | 0.895907 | 959/1000 |
| **Formal mixed-loss Arm B** | **0.00476456546** | **0.0146282911** | **0.881035649** | **0.908597358** | **975/1000** |

Formal Arm B reduces Chamfer by `38.63%` from the initial mesh and `8.89%`
from the archived predictor. The paired difference is `-0.000464982242` with
882/1000 mesh wins and 185/200 object-mean wins; the object-bootstrap 95% CI is
`[-0.000580558,-0.000314545]`. Normal remains below the initial mesh.

On valid paired samples, formal Arm B wins 804/998 against NDS, 829/999 against
nvdiffrec and 974/996 against ExMesh. Two NDS metrics are invalid, one
nvdiffrec sample failed and four ExMesh outputs have invalid metrics or changed
topology. Chamfer equals the bidirectional P2S mean by definition here because
both directions use equal 3,000-sample sets; P2S p95 remains non-duplicate.
The replacement direct-vertex Arm-E job `17888` completed all 200,000 steps on
2026-09-04 with exit `0:0` and elapsed `2-16:05:51`. Validation selected epoch
160; checkpoint SHA-256 is
`5a6aaa32bec6edcdd2c30face02c4ae8bc139fef18d4d05b3394c987057cb50f`.
The frozen B+E comparison is deliberately two-stage: array `18673` is running
the declared lambda grid on all 1,000 validation meshes, `18677` will lock the
mean-CD lambda, `18678` will then open test once, and `18679` will generate the
baseline comparison report. No Future2000 Arm-E or B+E test result is reported
before those dependencies complete. Full Arm-B provenance is in the
[formal report](../reports/future2000_mixed_vs_old_external_20260831_v2/FINAL_REPORT.md).

## Sofa50 controlled same-initial external benchmark

Ours, NDS, nvdiffrec and ExMesh use the exact same 25 current/coarse meshes,
native-1920 28-view RGB observations and cameras. All four methods completed
`25/25`; the input identity and unified metric audits pass.

The first aggregation mixed method-native Chamfer values. The common initial
mesh then appeared as both `0.00391323` and `0.01707047`, proving that the table
was not metric-compatible. The corrected aggregation evaluates every archived
initial/final mesh with `evaluate_mesh_geometry`, 3,000 surface samples and
seed 7. Native metrics are provenance-only. See the bilingual
[incident report](CHAMFER_EVALUATION_INCIDENT_2026-08-21.md) and the tracked
[recent consolidated report](../reports/sofa50_multitopology_rawlap500_v2/recent_ablation_and_old_domain_comparison_v1/REPORT.md).

| Method | Initial Chamfer | Final Chamfer ↓ | Improvement | Improved | Normal ↑ |
|---|---:|---:|---:|---:|---:|
| Ours | 0.017070468 | 0.011347800 | 33.52% | **25/25** | **0.944514** |
| NDS | 0.017070468 | **0.011204992** | **34.36%** | 22/25 | 0.873805 |
| nvdiffrec | 0.017070468 | 0.013654660 | 20.01% | 18/25 | 0.848122 |
| ExMesh | 0.017070468 | 0.020170615 | -18.16% | 8/25 | 0.845337 |

This is a supplied-initial Sofa50 synthetic comparison, not the official DTU
ExMesh millimetre protocol.

## Other diagnostic experiments

| Diagnostic | Main recorded result | Detailed local report |
|---|---|---|
| 1,000-step image resolution | F2 EPE `12.2433` versus F0 `12.5780` and F1 `12.7205`; screening result only. | [report](../runs/learned_laplacian/sofa50_image_resolution_ablation_1000step/analysis/REPORT.md) |
| Geometry-aware sampling | High-Laplacian sampling increases all-EPE from `12.3002` to `16.7813`/`18.1458`; hypothesis not supported. | [report](../runs/learned_laplacian/sofa50_geometry_aware_sampling_1000step/analysis/REPORT.md) |
| Oracle residual expert | The 1,000-step result is inconclusive; the 2,000-step E0/E1 best val losses are `0.0454773` and `0.0454485`; no material separation is established. | [report](../runs/learned_laplacian/sofa50_oracle_residual_expert_1000step/analysis/REPORT.md) |
| Controlled screening | Tiny perturbation and support mismatch are inconclusive; high-Laplacian exposure is not supported. | [report](../runs/learned_laplacian/sofa50_controlled_screening_1000step/analysis/REPORT.md) |
| Counterfactual refinement | Direct, raw-Laplacian and normalized-Laplacian residual arms improve `0/8` validation cases. | [report](../runs/learned_laplacian/sofa50_counterfactual_refinement/REPORT.md) |
| Residual target comparison | All three target formulations improve `0/4` validation cases; raw residual introduces zero flips in the reported run. | [report](../runs/learned_laplacian/sofa50_residual_target_comparison/REPORT.md) |
| `h^2` normalization audit | Round trip passes; maximum relative L2 error `4.4331e-17`, maximum absolute error `5.55112e-17`. | [report](../runs/learned_laplacian/sofa50_h2_normalization_audit/REPORT.md) |
| Recovery identity/oracle | Identity and scale-zero gates pass. Exact same-topology oracle changes Chamfer from `0.00196814` to `0.00134768`; prediction/oracle cosine is `0.0554065`. | [report](../runs/learned_laplacian/sofa50_recovery_identity_oracle_diagnostic/REPORT.md) |
| Query transfer gap | Expanded-query mean distance is `0.0184h–0.0270h`; canonical training perturbations are capped at `0.001h`. | [report](../runs/learned_laplacian/sofa50_transfer_gap_diagnostics/REPORT.md) |
| Delta-scale sweep | Best global scale improves `0/5` meshes for both control and perturbed groups. | [report](../runs/learned_laplacian/sofa50_step2000_perturbed_scale_sweep_rejected_face_flips/REPORT.md) |
| Renderer visibility | All tested visibility definitions improve `0/5` expanded meshes. | [report](../runs/learned_laplacian/sofa50_renderer_visibility_expanded_fixed_checkpoint/REPORT.md) |
| Visibility-aware recovery | Hard masking reduces mean Chamfer from `0.120283` to `0.0146517`, but improves `0/5` meshes relative to the initial geometry. | [report](../runs/learned_laplacian/sofa50_visibility_recovery_expanded_fixed_checkpoint/REPORT.md) |

These diagnostics identify a query-distribution and recovery problem. They do
not establish end-to-end coarse-mesh refinement.

## Thingi10K50 development runs

These runs use different datasets or development contracts and are not compared
directly with the canonical Sofa50 results.

| Run | Steps | Best val loss | State |
|---|---:|---:|---|
| `thingi10k50_960_full` | 5,150 | 0.251656 | Development run |
| `thingi10k50_gt_query_960_full` | 1,100 | 0.170152 | Development run |
| `thingi10k50_gt_query_960_local001_20260806_0341` | 1,100 | 0.170279 | Development run |
| `thingi10k50_gt_query_960_weighted_lr1e4_20260806` | 950 | 0.862029 | Weighted development run |
| 960 optimized one-epoch smoke | 10 | 0.305584 | Smoke only |
| 1920 optimized workers-4 one-epoch smoke | 10 | 0.305584 | Smoke only |

## HPC execution record

| Job | Experiment | Final state | Recorded result |
|---:|---|---|---|
| 15625 | C2F2 56-view, 20k | Completed | 20,000 steps; best val `0.0138104`; elapsed `13:58:22`. |
| 15629 | C2F2 K2, 14-view, 20k | Completed | Elapsed `03:59:59`; retained in the position-encoding records. |
| 15630 | C2F2 K4, 14-view, 20k | Completed | Elapsed `04:04:08`; retained in the position-encoding records. |
| 15633 | Superseded current-query B run | Cancelled | Cancelled after `04:44:15`; not used by the H2 analysis. |
| 15634 | Superseded A/B evaluation | Cancelled | Dependency job never started; not used by the H2 analysis. |
| 15686 | H2 three-shard evaluation | Completed | Three L40 array tasks completed in about 19 minutes. |
| 15687 | H2 report merge | Completed | Final JSON/CSV/report merge completed in 15 seconds. |
| 15794 | Future2000 raw-Laplacian 200k | Failed, recoverable | Reached step 32,000; `Too many open files` caused a downstream NCCL timeout. |
| 15795 | Future2000 raw-Laplacian 200k resume | Failed, resumable | Reached step 64,000; a DataLoader worker exhausted `/dev/shm`. |
| 15791 | Future2000 external diagnostic array | Historical, non-formal | Incomplete high-failure diagnostic; superseded by the audited full-1000 comparison. |
| 15812/15813 | Raw MSE evaluation/report | Completed | Four evaluation shards completed in 75–85 seconds; report merge completed in 24 seconds. |
| 15844 | Gaussian feature, 20k | Completed | Two L40 GPUs; elapsed `03:31:05`. |
| 15845 | Original + HF feature, 20k | Completed | Two L40 GPUs; elapsed `03:58:50`. |
| 15846/15847 | Image-feature evaluation/report | Completed | Four shards completed in under two minutes; report merge took 11 seconds. |
| 15854 | Native-1920 + HF, 20k | Completed | Four L40 GPUs; from scratch, global batch 4; completed all 20,000 steps. |
| 15864/15865 | Native-1920 paired evaluation/report | Completed | Contract audit passed; 1920 does not improve Top-10%/Top-1% or mean Chamfer/P2S. |
| 16584 | GT-query direct-raw transfer, 20k | Completed | Two Blackwell GPUs; contract passed; current-mesh recovery `4/25`, below current-query HF `20/25`. |
| 16607 | Future2000 old-structure direct-raw + HF, 200k | Archived checkpoint complete | Seven Blackwell GPUs; produced the archived `0.00522955` full-1000 result, not the formal current-architecture result. |
| 16736 | Sofa50 same-initial unified report | Completed | Four methods at 25/25; deterministic unified evaluator; `contract_audit=true`. |
| 17082 | Sofa50 multi-topology strong-smoothing v2, 20k | Completed | Two L40 GPUs; effective global batch 8; final/best validation `2.26915e-6`. |
| 17110-17113 | Sofa50 v1-v2 test/recovery and unified merge | Completed | Contract true; v2 raw EPE `0.00276820` versus v1 `0.00840367`, but refined Chamfer `0.00451747` versus `0.00426879`. |
| 17274/17275/17278 | Sofa50 Arms C/D/E | Completed | Matched-v2 C/D/E results are complete; E reaches Chamfer `0.00334039`. |
| 17513/17515 | Old-domain native-1920 Arm B/E | Completed | Validation-selected specialists completed; test Chamfers are `0.00853777` and `0.00806580`. |
| 17805/17806/17807 | Future2000 formal smoke/evaluation/finalizer | Completed | Mixed-loss Arm-B full-1000 audit complete; Chamfer `0.00476457`, 975/1000 improved. |
| 17800/17883 | Superseded Future2000 Arm-E launches | Cancelled/superseded | The never-started 4-GPU job was replaced by a 2-GPU global-batch-8 run, which was later resumed at an epoch boundary. Neither ID is the final completed allocation. |
| 17888 | Future2000 direct-vertex Arm E, 200k | Completed | Four-Blackwell epoch-boundary resume preserving global batch 8; completed `0:0` on 2026-09-04 after `2-16:05:51`; validation selected epoch 160. |
| 18673/18677/18678/18679 | Future2000 frozen B+E validation/lock/test/report | Running/dependency-gated at 2026-09-04 09:18 | `18673` uses eight deterministic validation shards with `ArrayTaskThrottle=4` (4 running, 4 waiting at snapshot); dependent test `18678` has the same four-GPU cap, cannot start before lock job `18677`, and report `18679` requires successful test completion. |

Jobs 15631 and 15632 were cancelled before execution after the B budget changed
from 50,000 to 20,000 steps. Both recorded zero runtime and produced no model
or comparison result.

## Result interpretation

- The lowest completed exact GT-query prediction error is from C2F2 at 960.
- Increasing image feature resolution from F0 to F2 reduces exact-query error.
- Increasing input resolution from 960 to 1920 does not reduce mean endpoint
  error under the unequal 50k/20k budgets.
- Increasing canonical views from 14 to 28 reduces the completed 20k validation
  loss. The completed 56-view arm improves raw errors but not validation loss
  relative to 28 views, while requiring 2.085x its runtime.
- GT-sub1 has substantially higher validation loss than the GT and adaptive
  query-graph arms.
- Existing expanded-query and OpenMVS recovery experiments do not reduce mean
  Chamfer.
- The OpenMVS observation is diagnostic only: its low-quality input meshes are
  excluded from training targets, checkpoint/model selection and scale-up
  decisions. Synthetic-current and controlled same-initial evaluations remain
  the decision endpoints.
- Current-graph training narrows the synthetic-current recovery gap relative to
  the frozen GT-query baseline. In the controlled 28-view H2 ablation, direct
  raw-Laplacian training is the best arm and lowers mean Chamfer below the
  initial mesh while improving 19/25 test samples.
- Raw MSE does not improve the high-curvature tail or mean recovery over Huber.
- The learned residual expert is effective without spatial gating, while the
  learned gate adds a smaller placement-specific benefit beyond a validation-
  selected constant scale.
- At 960, original-plus-HF gives the best raw/tail prediction; Gaussian-only
  gives the best mean recovery. Native-1920+HF does not improve the tail or
  mean downstream distance, despite higher compute and better normals/flips.
- GT-query direct-raw training improves strongly over historical normalized
  GT-query transfer, but still does not match current-query supervision on the
  current-mesh recovery task.
- Same-initial cross-method Chamfer must come from the unified evaluator.
  Method-native geometry metrics are provenance-only and cannot define the
  primary ranking.

## Source hierarchy

1. Per-run `metrics.json`, `summary.json`, CSV files and checkpoints are the
   numerical source of record.
2. This document records an aggregate snapshot and does not replace per-object
   or per-variant files.
3. HPC completion rows record scheduler state; per-run analysis files remain the
   source for scientific metrics.
4. Rows marked as running are dated snapshots and must not be read as final
   experiment outcomes.
