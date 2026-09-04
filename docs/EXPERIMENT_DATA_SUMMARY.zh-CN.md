# 实验数据汇总

[English](EXPERIMENT_DATA_SUMMARY.md) | [简体中文](EXPERIMENT_DATA_SUMMARY.zh-CN.md)

状态日期：2026-09-04 09:18 BST，Europe/London。

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

历史 canonical GT-query target 为

$$
\delta_i=(LV)_i,
\qquad
\widehat{\delta}_i=\frac{\delta_i}{h_i^2+10^{-12}}.
$$

当前 synthetic-current 实验的图和 raw target 均定义在 current graph 上：

$$
\delta_i^{\mathrm{current}}=(L_cP_{\mathrm{proxy}})_i.
$$

模型直接预测这个 raw field。`h_current` 仍是输入 geometry feature 和 audit
quantity；它不会对当前 target 做除法、clip 或 denormalization。`h^2` normalized
数值只保留用于历史对比和 diagnostics。

## 数据集清单

| 数据集 | 对象与划分 | 视图 / 分辨率 | 状态 | 位置 |
|---|---|---|---|---|
| Sofa50 canonical GT-query | 50 个对象；40/5/5 | 14 / 960 | 完成 | HPC：`sofa50_refinement/multiview_960` |
| Sofa50 1920 GT-query | 50 个对象；40/5/5 | 14 / 1920 | 完成 | HPC：`sofa50_refinement/multiview_1920` |
| 嵌套视图消融 | 50 个对象；40/5/5 | 14/28/56 / 960 | 完成 | `sofa50_refinement/multiview_nested_14_28_56_cpu_v3` |
| Query-resolution ablation v2 | 50 个对象；40/5/5 | 14 / 960 | 完成 | `multiview_960/query_resolution_ablation_v2` |
| Synthetic current-query，14 views | 50 个对象，每个 5 个变体；变体划分 200/25/25 | 14 / 960 | 完成并已复制到 HPC | `~/sofa_mesh/sofa50_synthetic_current` |
| Synthetic current-query，28 views | 50 个对象，每个 5 个变体；变体划分 200/25/25 | 28 / 960 | 完成 | HPC：`sofa50_synthetic_current_28view_v1` |
| Synthetic current-query，native 1920 | 与 960 相同的 250 IDs 和 200/25/25 split | 28 / 1920 | 数据、HF 训练与评估均完成 | HPC：`sofa50_synthetic_current_28view_native1920_v1` |
| Sofa50 多拓扑 raw-Laplacian v1 | 50 个对象、每个 10 个 variants；400/50/50 | 28 / 960 | 历史弱 smoothing 数据集已完成 | HPC：`Sofa50MultiTopologyRawLap500_v1` |
| Sofa50 多拓扑 raw-Laplacian v2 | 与 v1 相同对象、variants 和 split | 28 / 960 | 500/500 审计通过；2×L40 20k 训练和统一 v1-v2 test/recovery 已完成 | HPC：`Sofa50MultiTopologyRawLap500_v2` |
| Future2000 GT-adaptive expanded current | 2,000 个不同对象，每个 5 个冻结变体；按对象划分 8000/1000/1000 | 28 / 960 | Formal Arm-B test 与 200k Arm-E 训练已完成；frozen B+E validation-only lambda sweep 运行中 | HPC：`future2000_gt_adaptive_synthetic_current_28view_v2` |
| OpenMVS coarse-query 压力测试集 | 48 个 coarse mesh 可用；2 个缺失 | 预测使用 canonical 14 个 RGB 视图 | 完成；仅诊断，不作为目标 | HPC：`openmvs_texture_test_v6_48view` |
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

这些 mesh 的 initial reconstruction quality 过差，不再作为 target 或决策端点；
只保留为明确标注的 OOD 压力输入。详见
[OpenMVS 输入使用政策](OPENMVS_INPUT_POLICY.zh-CN.md)。下列历史数值仅描述已执行
诊断，不能用于排列目标质量或选择模型。

| Recovery iterations | Mesh 数量 | Initial Chamfer | Ensemble refined Chamfer | 改善 mesh | Ensemble 新增翻转面 |
|---:|---:|---:|---:|---:|---:|
| 200 | 48 | 0.0212023 | 0.0213199 | 2/48 | 4,692 |
| 1,000 | 48 | 0.0212023 | 0.0213198 | 2/48 | 4,734 |

对象 `8ecad62d-fd41-4d86-87f0-5f640c46f238` 和 `d7e2c96f-76cd-4699-bbe7-c65f7cb8b8cd` 没有 OpenMVS coarse mesh。将 recovery iterations 从 200 增加到 1,000 不改变汇总结论。

后续 projected-GT failure decomposition 的统一 Chamfer 为：initial
`0.0469163`、projected-GT position oracle `0.0440446`、projected-GT oracle
Laplacian on the OpenMVS graph 经冻结 recovery 后 `0.0456376`、归档 learned
prediction `0.0467913`。Recovery 保留 position-oracle 增益的 44.53%，learned
arm 实现 4.36%。这些只是低质量 OOD 输入上的诊断归因，模型选择权重为零。

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
| 56 | 完成 | 0.0138104 | 0.006991 | 0.013812 | 31,692 |

三个 view-count arms 均完成 20,000 optimizer steps。28-view arm 的 best
validation loss 最低；56-view arm 的统一 raw EPE 和 raw Top-10% EPE 最低。
HPC 结果目录：`runs/learned_laplacian/sofa50_c2f2_views_14_28_56_20k_seed7_v4`。

## Synthetic current-query 对比

### Frozen GT-query 50k 与 current-query 20k

最终 14-view 评估使用 25 个 matched synthetic-current test samples。两组训练
预算不同，因此该对比不能将 formulation 与 budget 的影响分离。

| Metric | GT-query 50k | Current-query 20k |
|---|---:|---:|
| Evaluation loss | 0.0145788 | 0.0117459 |
| Vector L2 | 2.994356 | 2.391482 |
| Global cosine | 0.883605 | 0.895129 |
| Initial Chamfer | 0.00391323 | 0.00391323 |
| Refined Chamfer | 0.00551727 | 0.00417930 |
| 改善 samples | 0/25 | 5/25 |

Current-query training 相对 frozen GT-query checkpoint 改善了记录中的预测与
recovery 指标，但其 mean refined Chamfer 仍高于共享的 initial Chamfer。

### 28-view current-graph H2 target/loss-space 消融

三个 C2F2 arms 使用相同的 28-view manifest、split IDs、seed、初始化、optimizer、
scheduler、batching 和 20,000-step budget。Local query jitter 关闭，contract
audit 通过。Native validation loss 位于不同 loss space，不能跨 arm 直接比较。

| Arm | Output target | Native loss space | Best native val | Runtime h |
|---|---|---|---:|---:|
| A：canonical H2 | `h^2` normalized | Output representation | 0.018456638 | 6.0416 |
| B：direct raw | Raw Laplacian | Output representation | 1.5825285e-6 | 6.1807 |
| C：normalized output/raw loss | `h^2` normalized | Raw Laplacian | 2.1655217e-6 | 6.6896 |

统一 test raw-space prediction：

| Arm | Raw EPE ↓ | Top-1% EPE ↓ | Top-10% EPE ↓ | Raw cosine ↑ | Weighted raw RMS ↓ |
|---|---:|---:|---:|---:|---:|
| A | 0.00769237 | 0.253855 | 0.0557517 | 0.933526 | 0.0427999 |
| B | 0.00300525 | 0.0417512 | 0.0136982 | 0.998667 | 0.00611072 |
| C | 0.00333673 | 0.0547519 | 0.0159651 | 0.997419 | 0.00815502 |

Zero-replacement recovery 的共享 initial Chamfer 为 `0.00391323`：

| Arm | Refined Chamfer ↓ | P2S ↓ | Normal consistency ↑ | Flips | 改善数/25 |
|---|---:|---:|---:|---:|---:|
| A | 0.00456011 | 0.00462286 | 0.934976 | 10,195 | 3/25 |
| B | 0.00380671 | 0.00380587 | 0.942406 | 6,566 | 19/25 |
| C | 0.00383121 | 0.00385409 | 0.941080 | 7,057 | 16/25 |

B 是主要结果：统一 raw-space error 和 refined Chamfer 最低，并改善 19/25
samples；C 位于 B 与 A 之间。B/C 的小 native loss 来源于 raw-Laplacian 数值单位，
不能解释为相对 A 的 loss 直接下降四个数量级。

三个 GPU shards 通过 Slurm array 15686 在三张 L40 上并行运行，每个 shard 用时
`00:19:06`–`00:19:25`；merge job 15687 用时 `00:00:15`。本地已保存
[报告](../runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7/analysis/REPORT.md)、
[JSON/CSV 记录](../runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7/analysis)、
[75 个 OBJ meshes](../runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7/analysis/mesh_comparisons/B_direct_raw_laplacian)
和[25 组总览图](../runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7/analysis/comparison_images/B_direct_raw_laplacian/overview_25.png)。

### Stage-2 分布适配与 Huber 饱和诊断

三个 matched continuation arms 都从同一个冻结 Arm-B checkpoint 继续 20,000
steps，没有一组超过原结果：

| Arm，best checkpoint | Raw EPE | Chamfer | Normal | 改善数 | 保留 | 找回 | 丢失 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen stage-1 B | 0.00300521 | **0.00380687** | **0.942463** | **19/25** | 19/19 | 0/6 | 0/19 |
| Continue X0 | 0.00369851 | 0.00390257 | 0.931793 | 16/25 | 14/19 | 2/6 | 5/19 |
| Continue X1 | **0.00349257** | **0.00384032** | **0.936939** | 16/25 | 14/19 | 2/6 | 5/19 |
| Continue 50/50 | 0.00363284 | 0.00388119 | 0.934341 | 16/25 | 14/19 | 2/6 | 5/19 |

X1 是三个 continuation 中最好的一组，但 mean Chamfer、P2S、normal consistency
和改善数仍差于冻结 stage-1。该结果否定当前 stage-2 配方，不否定分布适配这一
一般方向。

Arm-B validation Huber 诊断覆盖 243,000 个 vertices。GT raw-Laplacian top 1%
的平均 raw error 是 bottom 90% 的 `13.071x`，any-component saturation 为
`66.049%`，gradient retention 为 `58.436%`；该组承担 `34.931%` Huber loss，
但只贡献 `5.785%` output-gradient L1。梯度压缩主要集中在 top 1%，并非整个
top 10% 全面饱和。

### Huber 与 raw MSE

25-sample 统一 test 不支持 raw MSE 能改善 tail 或 mean geometry：

| Loss | Raw EPE ↓ | Raw RMS ↓ | Top-10% ↓ | Top-1% ↓ | Chamfer ↓ | Normal ↑ | Flips | 改善数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Huber, 0.01 | **0.00297478** | **0.00662604** | **0.0122438** | **0.0371716** | **0.00380692** | 0.942431 | 6,579 | **19/25** |
| Raw MSE | 0.00297688 | 0.00695997 | 0.0128078 | 0.0380294 | 0.00381317 | **0.943833** | **5,925** | 16/25 |

MSE 使用 global batch 6 和 510-step validation interval，Huber 为 2 和 500；因此
audit 将其标为资源驱动的非严格训练对比，held-out 统一评估仍可直接比较。

### Learned dynamic residual expert 与 gate 因果消融

Learned final 将联合训练 base 的 raw EPE 从 `0.00450175` 降到 `0.00294740`，
Chamfer 从 `0.00416138` 降到 `0.00377438`，normal 从 `0.926613` 升到
`0.944879`，改善数从 `3/25` 升到 `19/25`。该 base 不是冻结的 original Arm B，
不能把这个大差异当作相对 original baseline 的收益。

Validation 选择 constant gate `alpha=0.16`。Base-to-constant 在 raw EPE、Chamfer、P2S
和 normal 上均为 `25/25`；constant-to-learned 在 Chamfer/P2S 上 `25/25`，raw EPE
上 `24/25`。Learned placement 在 5 个 mesh 内 shuffle seeds 上也多数胜出。结果同时
支持有效 residual expert 与较小但可测的 spatial-placement 增益；gate/curvature
correlation 只是观察证据。

### 960 Gaussian 与 high-frequency image features

| Feature | Raw EPE ↓ | Raw RMS ↓ | Bottom 90% ↓ | Top 10% ↓ | Top 1% ↓ | Chamfer ↓ | 改善数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 0.00297471 | 0.00662531 | 0.00194494 | 0.0122427 | 0.0371654 | 0.00380683 | 19/25 |
| Gaussian | 0.00291322 | 0.00666042 | **0.00186460** | 0.0123509 | 0.0376811 | **0.00377507** | **21/25** |
| Original + HF | **0.00288627** | **0.00628246** | 0.00190114 | **0.0117524** | **0.0347902** | 0.00377832 | 20/25 |

Gaussian-only 改善 mean downstream geometry，却相对 original 恶化两个 tail groups。
`F + (F-Gaussian(F))` 获得最强 raw prediction/tail metrics，且 Bottom-90%
没有实质退化。

### Native-1920 + HF（已完成）

Native renderer 复用 960 HF 的 sample IDs、28 个 camera extrinsics、split、graph、proxy、
target 和 visibility contracts。Intrinsics 按 1920 缩放，native 与 resized 的最小 pixel
MAE 为 `0.0205764`，排除 resize-only 路径。Job 15854 使用 4×L40 从零完成
20,000 global optimizer steps。

1920 global batch 为 4，960 基线为 2。View chunk 和 gradient checkpointing 已通过数学
等价测试，但 batch 差异使训练对比非严格。

| 分辨率 + HF | Raw EPE ↓ | Raw RMS ↓ | Bottom 90% ↓ | Top 10% ↓ | Top 1% ↓ | Chamfer ↓ | P2S ↓ | Normal ↑ | Flips | 改善数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 960 | **0.00288618** | **0.00628203** | 0.00190107 | **0.0117522** | **0.0347895** | **0.00377857** | **0.00377999** | 0.942504 | 6303 | **20/25** |
| Native 1920 | 0.00290615 | 0.00690893 | **0.00183806** | 0.0125190 | 0.0389263 | 0.00378509 | 0.00378489 | **0.944522** | **5777** | 18/25 |

Native 1920 未改善 Top-10%、Top-1%、raw RMS、recovery-weighted RMS、Chamfer
或 P2S。它改善 normal consistency 并减少 flips。Runtime 从 2 GPUs × 3.98 h
（`7.95` GPU-hours）增加到 4 GPUs × 22.35 h（`89.39` GPU-hours）。

### GT-query direct-raw zero-shot transfer

另一组 2×Blackwell、20,000-step control 在 exact GT query 上使用 raw target
`L_gt @ V_gt` 训练，再把 frozen model 应用于 current mesh。Contract audit 通过，GT
只在 prediction 之后用于 surface evaluation。Correct/zero/shuffled RGB controls
确认 image features 会影响 held-out prediction。

| Arm | Current-mesh Chamfer ↓ | P2S ↓ | Normal ↑ | Flips | 改善数 |
|---|---:|---:|---:|---:|---:|
| 历史 GT-query `h^2` normalized | 0.00581764 | 0.00606854 | 0.922509 | 11795 | 0/25 |
| GT-query direct raw + HF | 0.00400486 | 0.00401379 | **0.948067** | **3087** | 4/25 |
| Current-query direct raw + HF | **0.00377832** | **0.00377984** | 0.942475 | 6326 | **20/25** |

移除 normalization 相对历史 GT-query transfer 明显改善，但没有弥合与 supervised
current-query training 的 query-distribution gap。历史 arm 早于 HF，因此这不是严格
单变量 normalization ablation。

### Strong-smoothing recovery 诊断与 recovery-aware A-E 研究

Exact target 加全方程 sparse integration 证明 v2 raw Laplacian 可恢复：用 component-
centroid translation gauge 时 mean oracle efficiency 为 `0.92366`。冻结 solver 中，
hard visibility 是最大 incremental loss（mean eta `0.34258 -> 0.16875`，44/50
变差）；confidence 可忽略，2,000 Adam steps 也只达到 `0.18635`。

已完成 A/B 使用全部 rows、`lambda=10^-2`，不使用 confidence、recovery Huber 或
Adam。Arm B 加入 `beta=10^-2` same-index recovered-vertex MSE。

| Test metric | A：仅 Lap | B：Lap + vertex |
|---|---:|---:|
| Raw EPE | **0.00252641** | 0.00263986 |
| Raw RMS | 0.00737725 | **0.00683290** |
| Chamfer | 0.00395529 | **0.00358497** |
| Eta | 0.07206 | **0.13036** |
| P2S p95 | 0.0122582 | **0.0105581** |
| Normal | 0.954902 | **0.959366** |
| Vertex RMS | 0.0135181 | **0.0115532** |

B 的 Chamfer 胜 32/50、vertex RMS 胜 43/50，但 raw EPE 只胜 10/50。Matched A-E
study 已完成：减弱 C/D recovery anchor 会恶化 geometry，direct-vertex E 达到 Chamfer
`0.00334039`；Frozen B+E 达到 `0.00302983`，后续 scalar-fusion control 为
`0.00318814`，说明 operator hybrid 不能由单个全局 vertex average 解释。

三个 formulation stress tests 进一步收窄了该解释。Direct-Lap A+E 在
`lambda=0.03`（CD `0.00298590` 对 B+E `0.00302983`）与 `lambda=0.01`
（`0.00314166` 对 `0.00319840`）下的 paired surface-distance 均无显著区分，两个
CD CI 都含零。B_P 训练中改变 recovery-loss anchor，也没有相对 B_0 形成 same-anchor
CD separation（`-0.00001703`，mesh/object CI 均跨零）。Sparse positional 实验则给出
平滑 density curve：fixed-lambda test CD 从 0% 的 `0.0330216` 单调降到 100% 的
`0.00302983`，Song-scale 2% 仍比 dense 高 `243.70%`，paired 为 0/50 胜。综合来看，
证据支持 dense learned anchoring 的实测优势，但不支持 recovery-aware Arm-B training
是 operator composition 必要条件的主张。

## Future2000 GT-adaptive 扩展实验

状态更新：2026-09-04 09:18 BST。数据集包含 2,000 个不同的 3D-FUTURE source objects，
每个对象有 5 个冻结的确定性 current-mesh 扰动变体。Object-level split 为 8,000
train、1,000 validation、1,000 test meshes。同一对象的变体共享其 28 个标定
960-pixel RGB observations，但 current geometry、connectivity、query graph 与
visibility 均保持 variant-specific。

归档 old-structure job `16607` 产出 Chamfer `0.00522954770` 与 959/1000 改善。
Formal current Arm B 则保留既定 mixed objective
`L_raw-Laplacian-Huber + 10^-2 L_recovered-vertex`，使用 validation-selected
epoch-195 checkpoint（SHA-256
`fa934cd44c4009dd392c415fe2c5f731c8cf1b78cda6a31fab199d4c15510b82`）。

| Full test system | Chamfer ↓ | P2S p95 ↓ | F-score ↑ | Normal ↑ | 改善数 |
|---|---:|---:|---:|---:|---:|
| Initial mesh | 0.00776417127 | — | — | 0.924252350 | — |
| Archived old-structure Ours | 0.00522954770 | — | — | 0.895907 | 959/1000 |
| **Formal mixed-loss Arm B** | **0.00476456546** | **0.0146282911** | **0.881035649** | **0.908597358** | **975/1000** |

Formal Arm B 相对 initial mesh 降低 Chamfer `38.63%`，相对 archived predictor
再降低 `8.89%`。Paired difference 为 `-0.000464982242`，mesh 胜 882/1000，
object-mean 胜 185/200；object-bootstrap 95% CI 为
`[-0.000580558,-0.000314545]`。Normal 仍低于 initial mesh。

在 valid paired samples 上，formal Arm B 对 NDS、nvdiffrec、ExMesh 分别胜
804/998、829/999、974/996。2 个 NDS metrics invalid、1 个 nvdiffrec sample failed、
4 个 ExMesh outputs invalid 或改变 topology。由于正反方向使用相同数量的 3,000 个
samples，Chamfer 与 bidirectional P2S mean 在此定义上完全相同；P2S p95 不重复。
替代后的 direct-vertex Arm-E job `17888` 已于 2026-09-04 完成全部 200,000 steps，
exit `0:0`、elapsed `2-16:05:51`。Validation 选择 epoch 160；checkpoint SHA-256 为
`5a6aaa32bec6edcdd2c30face02c4ae8bc139fef18d4d05b3394c987057cb50f`。
Frozen B+E 对比严格分成两阶段：array `18673` 正在全部 1,000 validation meshes 上运行
预声明 lambda grid，`18677` 将按 mean CD 锁参，`18678` 随后只打开一次 test，`18679`
再生成 baseline comparison 报告。依赖完成前不报告 Future2000 Arm-E/B+E test 结果。
Formal Arm-B 完整 provenance 见
[formal report](../reports/future2000_mixed_vs_old_external_20260831_v2/FINAL_REPORT.md)。

## Sofa50 同初始网格外部 benchmark

Ours、NDS、nvdiffrec 和 ExMesh 使用完全相同的 25 个 current/coarse meshes、
native-1920 28-view RGB observations 与 cameras。四种方法均完成 `25/25`，input
identity 与 unified metric audits 通过。

第一次聚合混用了 method-native Chamfer，使 common initial mesh 同时出现
`0.00391323` 与 `0.01707047`，证明该表不具备 metric compatibility。修正聚合使用
`evaluate_mesh_geometry`、3,000 surface samples 和 seed 7，对全部归档 initial/final
mesh 统一重算。Native metrics 只作 provenance。详见双语
[事故报告](CHAMFER_EVALUATION_INCIDENT_2026-08-21.zh-CN.md)，以及已跟踪的
[近期汇总报告](../reports/sofa50_multitopology_rawlap500_v2/recent_ablation_and_old_domain_comparison_v1/REPORT.md)。

| 方法 | Initial Chamfer | Final Chamfer ↓ | Improvement | 改善数 | Normal ↑ |
|---|---:|---:|---:|---:|---:|
| Ours | 0.017070468 | 0.011347800 | 33.52% | **25/25** | **0.944514** |
| NDS | 0.017070468 | **0.011204992** | **34.36%** | 22/25 | 0.873805 |
| nvdiffrec | 0.017070468 | 0.013654660 | 20.01% | 18/25 | 0.848122 |
| ExMesh | 0.017070468 | 0.020170615 | -18.16% | 8/25 | 0.845337 |

这是 supplied-initial Sofa50 synthetic comparison，不是官方 DTU ExMesh 毫米制协议。

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

## HPC 执行记录

| Job | 实验 | 最终状态 | 记录结果 |
|---:|---|---|---|
| 15625 | C2F2 56-view，20k | 完成 | 20,000 steps；best val `0.0138104`；elapsed `13:58:22`。 |
| 15629 | C2F2 K2，14-view，20k | 完成 | Elapsed `03:59:59`；保留在 position-encoding 实验记录中。 |
| 15630 | C2F2 K4，14-view，20k | 完成 | Elapsed `04:04:08`；保留在 position-encoding 实验记录中。 |
| 15633 | 已替代的 current-query B run | 已取消 | 运行 `04:44:15` 后取消；H2 analysis 未使用该 run。 |
| 15634 | 已替代的 A/B evaluation | 已取消 | Dependency job 未启动；H2 analysis 未使用该 job。 |
| 15686 | H2 三分片评估 | 完成 | 三个 L40 array tasks 均约 19 分钟完成。 |
| 15687 | H2 report merge | 完成 | 最终 JSON/CSV/report 在 15 秒内合并完成。 |
| 15794 | Future2000 raw-Laplacian 200k | 失败，可恢复 | 到达 step 32,000；`Too many open files` 随后导致 NCCL timeout。 |
| 15795 | Future2000 raw-Laplacian 200k resume | 失败，可恢复 | 达到 step 64,000；DataLoader worker 耗尽 `/dev/shm`。 |
| 15791 | Future2000 外部方法诊断 array | 历史、非正式 | 未完成的高失败率 diagnostic；已被 audited full-1000 comparison 替代。 |
| 15812/15813 | Raw MSE 评估/报告 | 完成 | 4 个 shards 用时 75–85 秒，merge 用时 24 秒。 |
| 15844 | Gaussian feature，20k | 完成 | 2×L40；elapsed `03:31:05`。 |
| 15845 | Original + HF feature，20k | 完成 | 2×L40；elapsed `03:58:50`。 |
| 15846/15847 | Image-feature 评估/报告 | 完成 | 4 个 shards 都在 2 分钟内完成，merge 11 秒。 |
| 15854 | Native-1920 + HF，20k | 已完成 | 4×L40；从零训练，global batch 4；完成全部 20,000 steps。 |
| 15864/15865 | Native-1920 paired evaluation/report | 已完成 | Contract audit 通过；1920 未改善 Top-10%/Top-1% 或 mean Chamfer/P2S。 |
| 16584 | GT-query direct-raw transfer，20k | 已完成 | 2×Blackwell；contract 通过；current-mesh recovery `4/25`，低于 current-query HF 的 `20/25`。 |
| 16607 | Future2000 old-structure direct-raw + HF，200k | 归档 checkpoint 已完成 | 7×Blackwell；产出归档 `0.00522955` full-1000 结果，不是 formal current-architecture 结果。 |
| 16736 | Sofa50 same-initial unified report | 已完成 | 四种方法均 25/25；使用 deterministic unified evaluator；`contract_audit=true`。 |
| 17082 | Sofa50 多拓扑 strong-smoothing v2，20k | 已完成 | 2×L40；effective global batch 8；final/best validation `2.26915e-6`。 |
| 17110–17113 | Sofa50 v1-v2 test/recovery 与 unified merge | 已完成 | Contract true；v2 raw EPE `0.00276820` 对 v1 `0.00840367`，但 refined Chamfer `0.00451747` 对 `0.00426879`。 |
| 17274/17275/17278 | Sofa50 Arms C/D/E | 已完成 | Matched-v2 C/D/E 结果已完成；E 的 Chamfer 为 `0.00334039`。 |
| 17513/17515 | Old-domain native-1920 Arm B/E | 已完成 | Validation-selected specialists 完成；test Chamfer 分别为 `0.00853777`、`0.00806580`。 |
| 17805/17806/17807 | Future2000 formal smoke/evaluation/finalizer | 已完成 | Mixed-loss Arm-B full-1000 audit 完成；Chamfer `0.00476457`，975/1000 改善。 |
| 17800/17883 | 已替代的 Future2000 Arm-E launches | 已取消/替代 | 未启动的 4-GPU job 被 2-GPU global-batch-8 run 替代，后者再于 epoch boundary 恢复；两者都不是最终完成 allocation。 |
| 17888 | Future2000 direct-vertex Arm E，200k | 已完成 | 4×Blackwell epoch-boundary resume 保持 global batch 8；2026-09-04 以 `0:0` 完成，elapsed `2-16:05:51`；validation 选择 epoch 160。 |
| 18673/18677/18678/18679 | Future2000 frozen B+E validation/锁参/test/report | 2026-09-04 09:18 运行中/依赖门控 | `18673` 保留 8 个确定性 validation shards，但设置 `ArrayTaskThrottle=4`；快照时 4 个运行、4 个等待。依赖 test `18678` 使用相同 4-GPU 上限，必须等待 `18677` 锁参成功；report `18679` 必须等待 test 成功。 |

Jobs 15631 和 15632 在 B 的预算从 50,000 修改为 20,000 steps 后，于执行前取消。两项运行时间均为 0，未产生模型或对比结果。

## 结果解释

- 当前已完成实验中，C2F2 960 的 exact GT-query prediction error 最低。
- 从 F0 增加到 F2 可降低 exact-query error。
- 在 50k/20k 不等预算下，从 960 增加到 1920 不降低 mean endpoint error。
- Canonical views 从 14 增加到 28 可降低已完成 20k 训练的 validation loss。
  已完成的 56-view arm 相对 28 views 改善 raw errors，但 validation loss 更高，
  runtime 为其 2.085 倍。
- GT-sub1 的 validation loss 高于 GT 和 adaptive query-graph arms。
- 已有 expanded-query 和 OpenMVS recovery 实验未降低 mean Chamfer。
- OpenMVS 观察仅用于诊断：其低质量输入明确排除在 training target、checkpoint/
  model selection 和 scale-up 决策之外。决策端点仍是 synthetic-current 与受控
  same-initial evaluation。
- Current-graph training 相对 frozen GT-query baseline 缩小了 synthetic-current
  recovery gap。在受控 28-view H2 消融中，direct raw-Laplacian training 最优，
  mean Chamfer 低于 initial mesh，并改善 19/25 test samples。
- Raw MSE 未改善 high-curvature tail 或 mean recovery。
- Learned residual expert 在无 spatial gate 时已有效；learned gate 在 validation-selected
  constant scale 之上还提供较小的 placement-specific 增益。
- 960 original+HF 的 raw/tail prediction 最好，Gaussian-only 的 mean recovery 最好。
  Native-1920+HF 未改善 tail 或 mean downstream distance，尽管 normal/flips 更好且
  使用更多计算资源。
- GT-query direct-raw training 相对历史 normalized GT-query transfer 明显改善，但在
  current-mesh recovery 上仍未达到 current-query supervision。
- Same-initial 跨方法 Chamfer 必须来自 unified evaluator。Method-native geometry
  metrics 只作为 provenance，不能定义 primary ranking。

## 数据来源优先级

1. 每个运行目录中的 `metrics.json`、`summary.json`、CSV 和 checkpoint 是数值记录源。
2. 本文档记录汇总快照，不替代 per-object 或 per-variant 文件。
3. HPC 完成记录反映 scheduler 终态；科学指标仍以各 run 的 analysis 文件为准。
4. 标记为运行中的记录是带日期快照，不应解释为最终实验结果。
