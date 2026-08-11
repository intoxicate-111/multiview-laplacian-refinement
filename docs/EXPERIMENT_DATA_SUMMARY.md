# Experiment data summary

[English](EXPERIMENT_DATA_SUMMARY.md) | [简体中文](EXPERIMENT_DATA_SUMMARY.zh-CN.md)

Status date: 2026-08-11, Europe/London.

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

The canonical absolute target is

$$
\delta_i=(LV)_i,
\qquad
\widehat{\delta}_i=\frac{\delta_i}{h_i^2+10^{-12}}.
$$

For the synthetic-current experiment, the graph and target are both defined on
the current graph:

$$
\delta_i^{\mathrm{current}}=(L_cP_{\mathrm{proxy}})_i,
\qquad
\widehat{\delta}_i^{\mathrm{current}}
=\frac{\delta_i^{\mathrm{current}}}{(h_i^c)^2+10^{-12}}.
$$

## Dataset inventory

| Dataset | Objects and split | Views / resolution | State | Location |
|---|---|---|---|---|
| Sofa50 canonical GT-query | 50 objects; 40/5/5 | 14 / 960 | Complete | HPC: `sofa50_refinement/multiview_960` |
| Sofa50 1920 GT-query | 50 objects; 40/5/5 | 14 / 1920 | Complete | HPC: `sofa50_refinement/multiview_1920` |
| Nested view ablation | 50 objects; 40/5/5 | 14/28/56 / 960 | Complete | `sofa50_refinement/multiview_nested_14_28_56_cpu_v3` |
| Query-resolution ablation v2 | 50 objects; 40/5/5 | 14 / 960 | Complete | `multiview_960/query_resolution_ablation_v2` |
| Synthetic current-query B | 50 objects, 5 variants each; 200/25/25 variants | 14 / 960 | Complete and copied to HPC | `~/sofa_mesh/sofa50_synthetic_current` |
| OpenMVS coarse-query set | 48 available coarse meshes; 2 missing | Prediction uses the canonical 14 RGB views | Complete | HPC: `openmvs_texture_test_v6_48view` |
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

| Recovery iterations | Meshes | Initial Chamfer | Ensemble refined Chamfer | Better meshes | Ensemble introduced flips |
|---:|---:|---:|---:|---:|---:|
| 200 | 48 | 0.0212023 | 0.0213199 | 2/48 | 4,692 |
| 1,000 | 48 | 0.0212023 | 0.0213198 | 2/48 | 4,734 |

Objects `8ecad62d-fd41-4d86-87f0-5f640c46f238` and
`d7e2c96f-76cd-4699-bbe7-c65f7cb8b8cd` have no OpenMVS coarse mesh. Increasing
the recovery iterations from 200 to 1,000 does not change the aggregate result.

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
| 56 | Running snapshot, epoch 1941/2000 | 0.0138256 | 0.00699609 | 0.0138337 at epoch 1940 | not final |

The 56-view row is a status snapshot and is excluded from a final view-count
conclusion. HPC root:
`runs/learned_laplacian/sofa50_c2f2_views_14_28_56_20k_seed7_v4`.

## Synthetic current-query comparison

Experiment A is the existing frozen GT-query checkpoint:

```text
runs/learned_laplacian/sofa50_c2_f2_50000step_3seed/seed_7/best.pt
```

Experiment B is C2/F2/K6/14-view current-query training with seed 7 and a
20,000-step budget. A is not retrained. Both checkpoints will be evaluated on
the same 25 held-out synthetic-current variants.

The pre-training recovery oracle improves all five validation objects:

| Metric | Value |
|---|---:|
| Mean initial Chamfer | 0.00324172 |
| Mean oracle-recovered Chamfer | 0.00208458 |
| Improved validation objects | 5/5 |
| Introduced flipped faces | 553 total |

The final A/B prediction, RGB-ablation and reconstruction results are pending.
The configured outputs are:

```text
runs/learned_laplacian/sofa50_synthetic_current_c2f2_14view_20k_seed7
runs/learned_laplacian/sofa50_synthetic_current_ab_comparison_seed7
```

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

## HPC status snapshot

| Job | Experiment | State at snapshot | Current evidence |
|---:|---|---|---|
| 15625 | C2F2 56-view, 20k | Running | Epoch 1941/2000; best val `0.0138256`. |
| 15629 | C2F2 K2, 14-view, 20k | Running | Epoch 811/2000; best val `0.0166923`. |
| 15630 | C2F2 K4, 14-view, 20k | Running | Epoch 786/2000; best val `0.0159757`. |
| 15633 | Synthetic current-query B, 20k | Pending resources | Output directory uses the `20k` contract. |
| 15634 | Frozen-A versus B evaluation | Pending dependency | Starts only after job 15633 succeeds. |

Jobs 15631 and 15632 were cancelled before execution after the B budget changed
from 50,000 to 20,000 steps. Both recorded zero runtime and produced no model
or comparison result.

## Result interpretation

- The lowest completed exact GT-query prediction error is from C2F2 at 960.
- Increasing image feature resolution from F0 to F2 reduces exact-query error.
- Increasing input resolution from 960 to 1920 does not reduce mean endpoint
  error under the unequal 50k/20k budgets.
- Increasing canonical views from 14 to 28 reduces the completed 20k validation
  loss; the 56-view arm is not final at this snapshot.
- GT-sub1 has substantially higher validation loss than the GT and adaptive
  query-graph arms.
- Existing expanded-query and OpenMVS recovery experiments do not reduce mean
  Chamfer.
- The synthetic-current A/B comparison is the active experiment intended to
  test whether training on the current graph reduces this formulation gap.

## Source hierarchy

1. Per-run `metrics.json`, `summary.json`, CSV files and checkpoints are the
   numerical source of record.
2. This document records an aggregate snapshot and does not replace per-object
   or per-variant files.
3. Active-job rows must be replaced with final metrics after their `metrics.json`
   or `comparison.json` files are written.
