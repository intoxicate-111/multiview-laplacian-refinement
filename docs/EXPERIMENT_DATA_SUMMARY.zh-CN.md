# 实验数据汇总

[English](EXPERIMENT_DATA_SUMMARY.md) | [简体中文](EXPERIMENT_DATA_SUMMARY.zh-CN.md)

状态日期：2026-08-11，Europe/London。

本文档汇总当前本地工作区和 HPC 中已有的实验数据。标记为“运行中快照”的数值不是最终结果。只有目标、loss、数据划分和评估路径一致的实验，其训练 loss 才可直接比较。

## 标准定义

| 标记 | 实现定义 |
|---|---|
| C0 | 图像特征维度 16；图网络隐藏维度 64；3 层图网络。 |
| C1 | 图像特征维度 32；图网络隐藏维度 128；3 层图网络。 |
| C2 | 图像特征维度 64；图网络隐藏维度 256；3 层图网络。 |
| F0 | 编码器步长 `2,2`；960 输入对应 240 x 240 特征图。 |
| F1 | 编码器步长 `2,1`；960 输入对应 480 x 480 特征图。 |
| F2 | 编码器步长 `1,1`；特征图分辨率等于输入分辨率。 |
| K0/K2/K4/K6 | 使用 0、2、4 或 6 个频率的位置 Fourier 编码。 |

Canonical absolute target 为

$$
\delta_i=(LV)_i,
\qquad
\widehat{\delta}_i=\frac{\delta_i}{h_i^2+10^{-12}}.
$$

Synthetic-current 实验的图和目标均定义在 current graph 上：

$$
\delta_i^{\mathrm{current}}=(L_cP_{\mathrm{proxy}})_i,
\qquad
\widehat{\delta}_i^{\mathrm{current}}
=\frac{\delta_i^{\mathrm{current}}}{(h_i^c)^2+10^{-12}}.
$$

## 数据集清单

| 数据集 | 对象与划分 | 视图 / 分辨率 | 状态 | 位置 |
|---|---|---|---|---|
| Sofa50 canonical GT-query | 50 个对象；40/5/5 | 14 / 960 | 完成 | HPC：`sofa50_refinement/multiview_960` |
| Sofa50 1920 GT-query | 50 个对象；40/5/5 | 14 / 1920 | 完成 | HPC：`sofa50_refinement/multiview_1920` |
| 嵌套视图消融 | 50 个对象；40/5/5 | 14/28/56 / 960 | 完成 | `sofa50_refinement/multiview_nested_14_28_56_cpu_v3` |
| Query-resolution ablation v2 | 50 个对象；40/5/5 | 14 / 960 | 完成 | `multiview_960/query_resolution_ablation_v2` |
| Synthetic current-query B | 50 个对象，每个 5 个变体；变体划分 200/25/25 | 14 / 960 | 完成并已复制到 HPC | `~/sofa_mesh/sofa50_synthetic_current` |
| OpenMVS coarse-query | 48 个 coarse mesh 可用；2 个缺失 | 预测使用 canonical 14 个 RGB 视图 | 完成 | HPC：`openmvs_texture_test_v6_48view` |
| Thingi10K50 开发集 | 50 个对象；40/5/5 | 960 和 1920 变体 | 仅开发与 smoke run | 本地 `thingi10k50` 运行目录 |

Synthetic-current 数据集包含 250 个静态样本。全部样本通过目标代数、current-graph `h^2` 回算、14 视图数量、对象级划分和图像路径检查。

## 已完成的 canonical Sofa50 训练

### 训练指标

`Best val loss` 是 checkpoint 选择 loss，不是下一表中的 endpoint error。

| 实验 | Seed | Steps | Best epoch | Best val loss | Final train loss | Final val loss | 运行时 h |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0F0，960 | 7 | 50,000 | 5000 | 0.0365528 | 0.0361083 | 0.0365519 | 2.64 |
| C0F1，960 | 7 | 50,000 | 4875 | 0.0358636 | 0.0350980 | 0.0358663 | 2.64 |
| C0F2，960 | 7 | 50,000 | 5000 | 0.0349125 | 0.0348144 | 0.0349178 | 4.72 |
| C2F2，960 | 7 | 50,000 | 4920 | 0.0126017 | 0.00571267 | 0.0126040 | 10.17 |
| C2F2，960 | 17 | 50,000 | 4930 | 0.0132884 | 0.00646517 | 0.0132948 | 10.10 |
| C2F2，960 | 27 | 50,000 | 4775 | 0.0133493 | 0.00710706 | 0.0133571 | 10.10 |
| C2F2，1920 | 7 | 20,000 | 1995 | 0.0147794 | 0.00707957 | 0.0147794 | 14.82 |
| C2F2，1920 | 17 | 20,000 | 2000 | 0.0145967 | 0.00613659 | 0.0146020 | 14.78 |
| C2F2，1920 | 27 | 20,000 | 2000 | 0.0134383 | 0.00708745 | 0.0134342 | 14.79 |

HPC 结果目录：

```text
runs/learned_laplacian/sofa50_image_resolution_ablation_50000step
runs/learned_laplacian/sofa50_c2_f2_50000step_3seed
runs/learned_laplacian/sofa50_c2_f2_1920_20000step_3seed
```

### Exact GT-query 预测

| 实验 | Seeds | All EPE ↓ | Top-10% EPE ↓ | Global cosine ↑ | Prediction/target norm |
|---|---|---:|---:|---:|---:|
| C0F0，960 | 7 | 9.4641 | 30.7221 | 0.7808 | 0.8020 |
| C0F1，960 | 7 | 9.3786 | 30.3095 | 0.7892 | 0.7938 |
| C0F2，960 | 7 | 9.1665 | 28.4751 | 0.8227 | 0.8180 |
| C2F2，960 | 7/17/27 | 2.82815 | 15.37434 | 0.89110 | 0.93480 |
| C2F2，1920 | 7/17/27 | 3.09280 | 16.32997 | 0.89537 | 0.93118 |

960 的 mean all-EPE 和 top-10% EPE 更低；1920 的 mean global cosine 更高。两组预算不同：960 为 50,000 steps，1920 为 20,000 steps。

在已完成的 F0/F1/F2 实验中，original-minus-zero RGB global cosine gap 分别为 `0.2236`、`0.3315` 和 `0.3724`。模型使用了 RGB 信息。

## 重建结果

### Expanded-query Sofa50 validation

共同的 initial Chamfer 为 `0.000652884`。

| 模型 | Seeds | Refined Chamfer ↓ | Point-to-surface ↓ | Normal consistency ↑ | 新增翻转面 | 改善对象数 |
|---|---|---:|---:|---:|---:|---:|
| C2F2，960 | 3 | 0.00116244 | 0.00118173 | 0.894626 | 均值 4213.3 | 每个 seed 0/5 |
| C2F2，1920 | 3 | 0.00125695 | 0.00126905 | 0.892581 | 均值 4264.0 | 每个 seed 0/5 |

两组记录的 recovery 配置一致。两组 refined Chamfer 均高于 initial Chamfer。

### OpenMVS coarse mesh

| Recovery iterations | Mesh 数量 | Initial Chamfer | Ensemble refined Chamfer | 改善 mesh | Ensemble 新增翻转面 |
|---:|---:|---:|---:|---:|---:|
| 200 | 48 | 0.0212023 | 0.0213199 | 2/48 | 4,692 |
| 1,000 | 48 | 0.0212023 | 0.0213198 | 2/48 | 4,734 |

对象 `8ecad62d-fd41-4d86-87f0-5f640c46f238` 和 `d7e2c96f-76cd-4699-bbe7-c65f7cb8b8cd` 没有 OpenMVS coarse mesh。将 recovery iterations 从 200 增加到 1,000 不改变汇总结论。

## 已完成的消融实验

### 模型容量 C0/C1/C2，2,000 steps

| 容量 | Best val loss ↓ | All EPE ↓ | Top-10% EPE ↓ | Global cosine ↑ | Pred/target norm |
|---|---:|---:|---:|---:|---:|
| C0 | 0.0478404 | 11.1322 | 39.2638 | 0.6700 | 0.6730 |
| C1 | 0.0446807 | 10.5003 | 35.9503 | 0.7137 | 0.7200 |
| C2 | 0.0428904 | 10.0454 | 35.4161 | 0.7193 | 0.7490 |

详细本地报告：[capacity ablation](../runs/learned_laplacian/sofa50_capacity_ablation_2000step/analysis_v2/REPORT.md)。

### 本地位置编码，C1F1，14 views，960，2,000 steps

| 编码 | Best epoch | Best/final val loss ↓ | Final train loss |
|---|---:|---:|---:|
| K0 | 195 | 0.0461422 | 0.0479796 |
| K2 | 195 | 0.0452735 | 0.0469340 |
| K4 | 170 | 0.0449300 | 0.0455100 |
| K6 | 170 | 0.0457297 | 0.0460866 |

在本地 C1F1 screening 中，K4 的 validation loss 最低。该结果不是 C2F2 对比。

### Query-graph resolution，C2F2，14 views，seed 7，20,000 steps

| Query graph | 状态 | Best epoch | Best val loss ↓ | Final train loss | Final val loss | 运行时 h |
|---|---|---:|---:|---:|---:|---:|
| GT | 由等价 14-view arm 表示 | 1995 | 0.0139316 | 0.00707592 | 0.0139314 | 3.15 |
| GT-sub1 | 完成 | 1905 | 0.0614830 | 0.0580221 | 0.0614822 | 4.25 |
| GT-adaptive | 完成 | 1790 | 0.0145840 | 0.00640667 | 0.0145877 | 3.95 |
| GT-sub2 | 按实验决策排除 | — | — | — | — | — |

HPC 结果目录：`runs/learned_laplacian/sofa50_c2f2_query_resolution_gt_sub1_adaptive_20k_seed7_v2`。

### 视图数量，C2F2，seed 7，20,000 steps

| Views | 状态 | Best val loss ↓ | Final/current train loss | Final/current val loss | GPU memory MiB |
|---:|---|---:|---:|---:|---:|
| 14 | 完成 | 0.0139316 | 0.00707592 | 0.0139314 | 9,095 |
| 28 | 完成 | 0.0130296 | 0.00660375 | 0.0130341 | 18,130 |
| 56 | 运行中快照，epoch 1941/2000 | 0.0138256 | 0.00699609 | epoch 1940 为 0.0138337 | 非最终值 |

56-view 行是状态快照，不用于最终 view-count 结论。HPC 结果目录：`runs/learned_laplacian/sofa50_c2f2_views_14_28_56_20k_seed7_v4`。

## Synthetic current-query 对比

Experiment A 使用已有冻结 GT-query checkpoint：

```text
runs/learned_laplacian/sofa50_c2_f2_50000step_3seed/seed_7/best.pt
```

Experiment B 是 seed 7、20,000-step 的 C2/F2/K6/14-view current-query 训练。A 不重新训练。两个 checkpoint 将在相同的 25 个 held-out synthetic-current variants 上评估。

训练前 recovery oracle 在五个 validation 对象上均降低 Chamfer：

| 指标 | 数值 |
|---|---:|
| Mean initial Chamfer | 0.00324172 |
| Mean oracle-recovered Chamfer | 0.00208458 |
| 改善 validation 对象 | 5/5 |
| 新增翻转面 | 总计 553 |

最终 A/B 预测、RGB 消融和重建结果尚未生成。配置的输出位置为：

```text
runs/learned_laplacian/sofa50_synthetic_current_c2f2_14view_20k_seed7
runs/learned_laplacian/sofa50_synthetic_current_ab_comparison_seed7
```

## 其他诊断实验

| 诊断 | 主要记录结果 | 详细本地报告 |
|---|---|---|
| 1,000-step 图像分辨率 | F2 EPE `12.2433`，F0 为 `12.5780`，F1 为 `12.7205`；仅作为 screening。 | [报告](../runs/learned_laplacian/sofa50_image_resolution_ablation_1000step/analysis/REPORT.md) |
| Geometry-aware sampling | High-Laplacian sampling 将 all-EPE 从 `12.3002` 提高到 `16.7813`/`18.1458`；不支持该假设。 | [报告](../runs/learned_laplacian/sofa50_geometry_aware_sampling_1000step/analysis/REPORT.md) |
| Oracle residual expert | 1,000-step 结果不确定；2,000-step E0/E1 best val loss 为 `0.0454773` 和 `0.0454485`，未形成实质差异。 | [报告](../runs/learned_laplacian/sofa50_oracle_residual_expert_1000step/analysis/REPORT.md) |
| Controlled screening | Tiny perturbation 与 support mismatch 结果不确定；不支持 high-Laplacian exposure。 | [报告](../runs/learned_laplacian/sofa50_controlled_screening_1000step/analysis/REPORT.md) |
| Counterfactual refinement | Direct、raw-Laplacian 和 normalized-Laplacian residual 三组均改善 `0/8` validation cases。 | [报告](../runs/learned_laplacian/sofa50_counterfactual_refinement/REPORT.md) |
| Residual target comparison | 三种 target 均改善 `0/4` validation cases；所记录的 raw residual 运行新增翻转面为 0。 | [报告](../runs/learned_laplacian/sofa50_residual_target_comparison/REPORT.md) |
| `h^2` normalization audit | 回算通过；最大 relative L2 error 为 `4.4331e-17`，最大 absolute error 为 `5.55112e-17`。 | [报告](../runs/learned_laplacian/sofa50_h2_normalization_audit/REPORT.md) |
| Recovery identity/oracle | Identity 和 scale-zero gate 通过。Exact same-topology oracle 将 Chamfer 从 `0.00196814` 降至 `0.00134768`；prediction/oracle cosine 为 `0.0554065`。 | [报告](../runs/learned_laplacian/sofa50_recovery_identity_oracle_diagnostic/REPORT.md) |
| Query transfer gap | Expanded-query mean distance 为 `0.0184h–0.0270h`；canonical 训练扰动上限为 `0.001h`。 | [报告](../runs/learned_laplacian/sofa50_transfer_gap_diagnostics/REPORT.md) |
| Delta-scale sweep | Control 和 perturbed 两组的最佳 global scale 均改善 `0/5` meshes。 | [报告](../runs/learned_laplacian/sofa50_step2000_perturbed_scale_sweep_rejected_face_flips/REPORT.md) |
| Renderer visibility | 所有已测 visibility 定义均改善 `0/5` expanded meshes。 | [报告](../runs/learned_laplacian/sofa50_renderer_visibility_expanded_fixed_checkpoint/REPORT.md) |
| Visibility-aware recovery | Hard mask 将 mean Chamfer 从 `0.120283` 降至 `0.0146517`，但相对 initial geometry 改善 `0/5` meshes。 | [报告](../runs/learned_laplacian/sofa50_visibility_recovery_expanded_fixed_checkpoint/REPORT.md) |

这些诊断指向 query distribution 与 recovery 问题，尚未建立 end-to-end coarse-mesh refinement。

## Thingi10K50 开发实验

这些运行使用不同数据集或开发合同，不与 canonical Sofa50 结果直接比较。

| 实验 | Steps | Best val loss | 状态 |
|---|---:|---:|---|
| `thingi10k50_960_full` | 5,150 | 0.251656 | 开发运行 |
| `thingi10k50_gt_query_960_full` | 1,100 | 0.170152 | 开发运行 |
| `thingi10k50_gt_query_960_local001_20260806_0341` | 1,100 | 0.170279 | 开发运行 |
| `thingi10k50_gt_query_960_weighted_lr1e4_20260806` | 950 | 0.862029 | Weighted 开发运行 |
| 960 optimized one-epoch smoke | 10 | 0.305584 | 仅 smoke |
| 1920 optimized workers-4 one-epoch smoke | 10 | 0.305584 | 仅 smoke |

## HPC 状态快照

| Job | 实验 | 快照状态 | 当前记录 |
|---:|---|---|---|
| 15625 | C2F2 56-view，20k | 运行中 | Epoch 1941/2000；best val `0.0138256`。 |
| 15629 | C2F2 K2，14-view，20k | 运行中 | Epoch 811/2000；best val `0.0166923`。 |
| 15630 | C2F2 K4，14-view，20k | 运行中 | Epoch 786/2000；best val `0.0159757`。 |
| 15633 | Synthetic current-query B，20k | 等待资源 | 输出目录使用 `20k` 合同。 |
| 15634 | Frozen-A 与 B 统一评估 | 等待依赖 | 仅在 job 15633 成功完成后启动。 |

Jobs 15631 和 15632 在 B 的预算从 50,000 修改为 20,000 steps 后，于执行前取消。两项运行时间均为 0，未产生模型或对比结果。

## 结果解释

- 当前已完成实验中，C2F2 960 的 exact GT-query prediction error 最低。
- 从 F0 增加到 F2 可降低 exact-query error。
- 在 50k/20k 不等预算下，从 960 增加到 1920 不降低 mean endpoint error。
- Canonical views 从 14 增加到 28 可降低已完成 20k 训练的 validation loss；本快照中的 56-view arm 尚未完成。
- GT-sub1 的 validation loss 高于 GT 和 adaptive query-graph arms。
- 已有 expanded-query 和 OpenMVS recovery 实验未降低 mean Chamfer。
- 当前 synthetic-current A/B 实验用于检验 current-graph 训练是否缩小该 formulation gap。

## 数据来源优先级

1. 每个运行目录中的 `metrics.json`、`summary.json`、CSV 和 checkpoint 是数值记录源。
2. 本文档记录汇总快照，不替代 per-object 或 per-variant 文件。
3. 运行中表格应在对应 `metrics.json` 或 `comparison.json` 写入后更新为最终结果。
