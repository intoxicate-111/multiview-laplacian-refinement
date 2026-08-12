# Sofa50 C2F2 受控消融报告

状态时间：2026-08-12 06:01 BST

训练设备：NVIDIA L40

随机种子：7

## 1. 范围

本报告汇总以下实验：

1. 14、28、56 canonical views 消融；
2. GT、GT-sub1、GT-adaptive query graph resolution 消融；
3. `views_28 + GT-adaptive` 组合 arm；
4. GT-query 与 synthetic current-query 训练对比；
5. synthetic current-query 从 20,000 steps 延长到 50,000 steps；
6. 28-view synthetic current-query local jitter 对照。

已完成实验使用最终 `metrics.json` 或分析产物。Local jitter 使用状态时间对应的训练快照，不作为最终对照结果。

## 2. 指标定义

- Validation loss：训练目标空间中的 validation Huber loss。
- Target-space EPE：h²-normalized Laplacian prediction 与 target 的逐顶点向量距离均值。
- Raw EPE：按当前 query graph 的局部尺度恢复后计算的逐顶点向量距离均值，恢复关系为
  `delta_raw = delta_hat * (h_current² + 1e-12)`。
- Top-10% 和 Top-1%：按各自 target magnitude 排序后的最高 10% 和 1% 顶点。
- Global cosine：展平后的 prediction 与 target 的余弦相似度。
- Pred/target norm：prediction 全局 L2 norm 与 target 全局 L2 norm 的比值。

不同 query graph 上的 raw EPE、Top-10% 和 Top-1% 使用各自的顶点集合、局部 `h` 和 target 排序。它们不是共同物理点上的 paired metric。

## 3. 实验契约

### 3.1 View-count 消融

- 模型：C2F2；
- 训练预算：20,000 optimizer steps；
- 训练/validation mesh：40/5；
- views：14、28、56；
- `views_14` 是 `views_28` 的前缀，`views_28` 是 `views_56` 的前缀；
- 三个 arm 使用相同 GT graph 和 target；
- 基础 14 个 camera poses 完全复用；
- 56-view observations 使用同一数据契约生成。

### 3.2 Query-resolution 消融

- 模型：C2F2；
- views：14；
- 训练预算：20,000 optimizer steps；
- RGB 与 cameras 在各 query-resolution arm 间复用；
- target 在各自 current graph 上重新计算；
- 不执行 cross-graph target interpolation；
- GT arm 使用 14-view run 作为契约 alias；
- GT-sub2 为 data-only arm，未训练。

| Query graph | Train vertices | Validation vertices | 状态 |
|---|---:|---:|---|
| GT | 5,962–15,650 | 8,421–10,849 | 完成，使用 views_14 alias |
| GT-sub1 | 23,800–60,978 | 33,580–43,246 | 完成 |
| GT-sub2 | 95,152–240,632 | 134,211–172,831 | 未训练 |
| GT-adaptive | 6,430–16,119 | 8,639–11,272 | 完成 |

GT-adaptive 对每个 sample 匹配 GT-sub2 的 maximum represented-area threshold。GT-adaptive 相对 GT-sub2 的最大 vertex-count ratio 为 `0.07841051`。

### 3.3 组合 arm

- observations：复用 views_28；
- query graph 与 targets：复用 GT-adaptive；
- renderer visibility：在 GT-adaptive graph 上对 28 views 重新计算；
- visibility backend：CUDA；
- CPU visibility fallback：关闭；
- validation sample IDs：与 views_28 和 GT-adaptive 相同；
- 训练预算：20,000 optimizer steps。

### 3.4 Local query jitter 对照

两个 arm 使用相同 C2F2、28 views、200/25/25 train/validation/test samples、5 个 stored current variants、seed、loss、optimizer、scheduler、recovery 和 20,000-step budget。

Arm B 的训练时 query position 为：

`V_tilde = V + eta`

其中 `eta` 为逐顶点 isotropic Gaussian，`std = 0.003 h`，L2 offset 截断在 `0.009 h`。Runtime jitter 只用于训练。Proxy、normalized target、raw target、`h_current`、graph、connectivity 和 target operator 保持不变。Validation 和 test 不使用 jitter。

实现中的 contract checks 覆盖：两个 epoch 的 runtime jitter positions 不相同；对应 target tensors、graph tensors、`h_current` 和 proxy positions 完全相同；同一 sample/seed/epoch 的 jitter 可重复；jitter scale 上限为 `0.009 h`。对应测试位于 `tests/learned_laplacian/test_local_query_jitter.py`。Final evaluator 还检查两个 arm 的配置除 jitter enablement 和 arm label 外一致。

## 4. View-count 结果

五个 matched validation meshes 的 macro mean：

| Metric | 14 views | 28 views | 56 views |
|---|---:|---:|---:|
| Best validation loss | 0.0139316 | 0.0130296 | 0.0138104 |
| Target-space EPE | 2.916554 | 2.788490 | 2.922096 |
| Raw EPE | 0.003203 | 0.003119 | 0.003016 |
| Raw Top-10% EPE | 0.022291 | 0.021110 | 0.019517 |
| Raw Top-1% EPE | 0.125458 | 0.105166 | 0.101320 |
| Raw global cosine | 0.938055 | 0.972197 | 0.975880 |
| Raw pred/target norm | 1.141279 | 1.057517 | 1.088511 |
| Runtime | 3.146 h | 6.699 h | 13.968 h |
| Peak GPU memory | 9,095 MiB | 18,130 MiB | 31,692 MiB |

相对变化：

| Comparison | Best-val improvement | Raw EPE improvement | Raw Top-10% improvement | Raw cosine change | Runtime multiplier |
|---|---:|---:|---:|---:|---:|
| 28 vs 14 | +6.47% | +2.63% | +5.30% | +0.034142 | 2.129x |
| 56 vs 14 | +0.87% | +5.85% | +12.44% | +0.037825 | 4.440x |
| 56 vs 28 | -5.99% | +3.30% | +7.55% | +0.003682 | 2.085x |

结论：

- 28 views 的 best validation loss 低于 14 和 56 views。
- 56 views 的 raw EPE、raw Top-10% EPE 和 raw Top-1% EPE 低于 28 views，raw cosine 高于 28 views；best validation loss 高于 28 views。
- 28 views 相对 14 views 的 runtime multiplier 为 2.129；56 views 相对 14 views 为 4.440。

## 5. Query-resolution 结果

| Metric | GT | GT-sub1 | GT-adaptive |
|---|---:|---:|---:|
| Best validation loss | 0.0139316 | 0.0614830 | 0.0145840 |
| Target-space EPE | 2.916554 | 15.919201 | 3.082665 |
| Raw EPE | 0.003203 | 0.006359 | 0.002917 |
| Raw Top-10% EPE | 0.022291 | 0.043869 | 0.018364 |
| Raw Top-1% EPE | 0.125458 | 0.197677 | 0.093426 |
| Raw global cosine | 0.938055 | 0.328798 | 0.949597 |
| Raw pred/target norm | 1.141279 | 0.419126 | 1.036761 |
| Runtime | 3.146 h | 4.255 h | 3.954 h |

GT-adaptive 相对 GT：

| Metric | GT | GT-adaptive | Change |
|---|---:|---:|---:|
| Best validation loss | 0.0139316 | 0.0145840 | +4.68% loss |
| Target-space EPE | 2.916554 | 3.082665 | +5.69% error |
| Target-space Top-10% EPE | 16.035759 | 17.208148 | +7.31% error |
| Target-space Top-1% EPE | 63.886231 | 78.285706 | +22.54% error |
| Raw EPE | 0.003203 | 0.002917 | -8.92% error |
| Raw Top-10% EPE | 0.022291 | 0.018364 | -17.61% error |
| Raw Top-1% EPE | 0.125458 | 0.093426 | -25.53% error |
| Raw Top-10% cosine | 0.997907 | 0.989986 | -0.007920 |
| Raw Top-1% cosine | 0.996282 | 0.976694 | -0.019588 |

按各 graph 自身 target magnitude 分位拆分 raw EPE：

| 区间 | GT | GT-adaptive | Adaptive change |
|---|---:|---:|---:|
| Bottom 90% | 0.001082 | 0.001201 | +11.0% error |
| 90–99% | 0.010852 | 0.010007 | -7.8% error |
| Top 10% | 0.022291 | 0.018364 | -17.6% error |
| Top 1% | 0.125458 | 0.093426 | -25.5% error |

五个 validation meshes 上，GT-adaptive 的 raw EPE、raw Top-10% EPE 和 raw Top-1% EPE 均低于 GT，计数为 5/5。

结论：

- GT-sub1 的 validation loss、target-space EPE 和 raw EPE 均高于 GT 与 GT-adaptive。
- GT-adaptive 的 raw absolute EPE 在自身 graph 上低于 GT，差异集中在 target magnitude 的上分位。
- GT-adaptive 的 normalized validation loss、target-space EPE、target-space Top-10% EPE、target-space Top-1% EPE、raw Top-10% cosine 和 raw Top-1% cosine 未显示同方向变化。
- Adaptive graph 改变局部 `h²`、顶点集合和 target 分布。现有 raw EPE 结果不能单独区分模型预测变化与 graph/denormalization 尺度变化。
- 当前证据支持 GT-adaptive pipeline 在未见 Sofa meshes 上产生一致的 raw-space metric 变化；不构成共同物理点上的预测能力对照。

## 6. `views_28 + GT-adaptive` 组合结果

| Metric | views_28 | GT-adaptive | Combination |
|---|---:|---:|---:|
| Best validation loss | 0.0130296 | 0.0145840 | 0.0131095 |
| Target-space EPE | 2.788490 | 3.082665 | 2.766282 |
| Raw EPE | 0.003119 | 0.002917 | 0.002879 |
| Raw Top-10% EPE | 0.021110 | 0.018364 | 0.018652 |
| Raw Top-1% EPE | 0.105166 | 0.093426 | 0.092543 |
| Raw global cosine | 0.972197 | 0.949597 | 0.962147 |
| Runtime | 6.699 h | 3.954 h | 7.700 h |

组合相对单因素 arm：

| Comparison | Best-val improvement | Raw EPE improvement | Raw Top-10% improvement | Raw Top-1% improvement | Raw cosine change |
|---|---:|---:|---:|---:|---:|
| Combination vs views_28 | -0.61% | +7.70% | +11.64% | +12.00% | -0.010050 |
| Combination vs GT-adaptive | +10.11% | +1.33% | -1.57% | +0.94% | +0.012550 |

预设判据：

| Condition | Result |
|---|---|
| Raw Top-10% EPE no higher than GT-adaptive | False |
| Raw Top-1% EPE no higher than GT-adaptive | True |
| Best validation loss no higher than GT-adaptive | True |
| Raw global cosine no lower than GT-adaptive | True |
| All four conditions | False |

结论：

- 组合 arm 的 best validation loss 位于 views_28 与 GT-adaptive 之间。
- 组合 arm 的 raw EPE 和 raw Top-1% EPE 低于两个单因素 arm。
- 组合 arm 的 raw Top-10% EPE 比 GT-adaptive 高 1.57%。
- 组合 arm 未满足全部预设判据。

## 7. Synthetic current-query

### 7.1 20k current-query 与既有 GT-query checkpoint

该对比在同一 synthetic-current test split 上评估 25 samples、5 objects。GT-query checkpoint 使用既有 50k 训练；current-query checkpoint 使用 20k 训练，因此不是同预算 paired training。

| Metric | GT-query 50k | Current-query 20k |
|---|---:|---:|
| Evaluation loss | 0.0145788 | 0.0117459 |
| Vector L2 | 2.994356 | 2.391482 |
| Global cosine | 0.883605 | 0.895129 |
| High-10% cosine | 0.966991 | 0.974085 |
| Pred/target norm | 1.006624 | 0.914124 |
| Zero-RGB loss | 0.0243689 | 0.0362948 |
| Correct-zero loss gap | 0.0097902 | 0.0245489 |
| Initial Chamfer | 0.00391323 | 0.00391323 |
| Reconstruction Chamfer | 0.00551727 | 0.00417930 |
| Point-to-surface | 0.00567265 | 0.00423201 |
| Normal consistency | 0.922748 | 0.940047 |
| Introduced flipped faces | 11,369 | 8,434 |
| Samples improved over initial | 0/25 | 5/25 |

结论：

- Current-query 20k 在该 synthetic-current test split 上的 loss、vector L2、reconstruction Chamfer、point-to-surface 和 introduced flipped faces 数值低于 GT-query 50k。
- Current-query 20k 的 global cosine、high-10% cosine 和 normal consistency 高于 GT-query 50k。
- Current-query 20k 的 reconstruction Chamfer 仍高于 initial Chamfer；25 个 samples 中 5 个低于各自 initial Chamfer。
- 该结果比较了不同训练 formulation 和不同训练预算，不能分离 formulation 与预算效应。

### 7.2 Current-query 50k continuation

Job 15664 从 20k checkpoint 恢复，使用 2×L40 完成到 50,000 optimizer steps。

| Metric | 20k | 50k | Change |
|---|---:|---:|---:|
| Best validation loss | 0.0151933 | 0.0139379 | -8.26% |
| Final validation loss | 0.0151914 | 0.0139382 | -8.25% |
| Final train loss | 0.0127891 | 0.0109742 | -14.19% |
| Best epoch | 400 | 975 | — |
| Optimizer steps | 20,000 | 50,000 | — |

50k continuation 的最终学习率为 `1e-6`，best epoch 为 975，final epoch 为 1000。最后 25 个 epochs 的 validation loss 位于 `0.0139379–0.0139546` 附近。

结论：

- 从 20k 延长到 50k 后，native validation loss 和 train loss 均下降。
- 50k checkpoint 尚未完成与 20k 相同的 synthetic-current test、zero-RGB 和 reconstruction 评估，不能从 native validation loss 推导下游 reconstruction 变化。

## 8. Local query jitter 训练快照

状态时间：2026-08-12 06:01 BST。单 GPU 下每个 epoch 对应 100 optimizer steps，20,000-step 上限对应 200 epochs。

| Arm | Job | Epoch | Approx. steps | Latest train loss | Latest validation loss | Best validation loss | LR |
|---|---:|---:|---:|---:|---:|---:|---:|
| A: no jitter | 15662_0 | 90/200 | 9,000/20,000 | 0.0250877 | 0.0217246 | 0.0217145 | 1e-3 |
| B: local jitter | 15662_1 | 72/200 | 7,200/20,000 | 0.0269448 | 0.0243396 | 0.0243396 | 1e-3 |

在相同 epoch 70：

| Arm | Validation loss |
|---|---:|
| A: no jitter | 0.0217145 |
| B: local jitter | 0.0243396 |

该快照中 B 比 A 高 `12.1%`。两个 arm 的 scheduler 均未降低学习率。两个 stderr 为空。

实时资源快照：

| Arm | GPU | GPU memory | GPU utilization snapshot |
|---|---|---:|---:|
| A | L40 | 30,387 MiB | 97% |
| B | L40 | 30,387 MiB | 34% |

GPU utilization 是单次采样。每个 epoch 同时包含约 55–64 秒 image decode/data 阶段和约 62–63 秒 forward/backward 阶段。

结论：

- 当前 epoch-aligned validation loss 未显示 local jitter 相对 no-jitter 的下降。
- 两个 arm 尚未完成，当前差异不作为最终结果。
- Job 15663 依赖两个训练 arm，完成后执行 deterministic synthetic validation/test、zero-RGB、OpenMVS48 recovery 和汇总分析。

## 9. 假设状态

| Hypothesis | Status | Recorded result |
|---|---|---|
| 28 views 相对 14 views 降低 validation loss | Supported in completed seed-7 run | -6.47% best validation loss |
| 56 views 相对 28 views 降低 validation loss | Not supported in completed seed-7 run | +5.99% best validation loss |
| GT-sub1 uniform subdivision 降低 validation/raw error | Not supported | Validation loss 0.0614830；raw EPE 0.006359 |
| GT-adaptive 降低自身 graph 上的 raw tail EPE | Supported as a pipeline metric | Raw Top-10% -17.61%；Top-1% -25.53%；5/5 meshes |
| GT-adaptive 提升共同物理位置上的预测能力 | Not tested | 缺少 common-surface paired evaluation |
| 28 views 与 GT-adaptive 的指标变化全部叠加 | Not supported under four-condition rule | Top-10% retention condition false |
| Current-query 20k 相对既有 GT-query 50k 改变 synthetic-current test metrics | Supported for recorded comparison | Loss、EPE 和 reconstruction metrics 见第 7.1 节 |
| Current-query 50k 相对 current-query 20k 降低 native validation loss | Supported | -8.26% |
| Local query jitter 改善最终 synthetic 与 OpenMVS recovery | Pending | Jobs 15662_0、15662_1、15663 |

## 10. 尚需完成的判定

1. 对 GT 与 GT-adaptive prediction 做 common-surface paired evaluation：映射到相同 GT vertices 或固定表面采样点，使用同一 target、同一 curvature bins、同一 EPE/cosine 定义。
2. 对 current-query 50k checkpoint 执行与 20k 相同的 synthetic-current test、zero-RGB 和 reconstruction evaluation。
3. 等待 15662 两个 arm 完成以及 15663 输出 final local-jitter report。
4. 在 adaptive 的 common-surface 指标与 local-jitter downstream endpoint 完成前，不从 native validation loss 或 graph-specific raw EPE 推导 Sofa50 主训练配置。

## 11. 产物位置

- View/query-resolution analysis：
  `runs/learned_laplacian/sofa50_c2f2_view_query_resolution_ablation_20k_seed7/analysis/`
- Combination analysis：
  `runs/learned_laplacian/sofa50_c2f2_view_query_combo_28_gt_adaptive_20k_seed7_v1/analysis/`
- Synthetic-current 20k：
  `runs/learned_laplacian/sofa50_synthetic_current_c2f2_14view_20k_seed7/`
- Synthetic-current 20k A/B comparison：
  `runs/learned_laplacian/sofa50_synthetic_current_ab_comparison_seed7/`
- Synthetic-current 50k continuation：
  `runs/learned_laplacian/sofa50_synthetic_current_c2f2_14view_50k_resume_from_20k_seed7/`
- Local-jitter runs：
  `runs/learned_laplacian/sofa50_synthetic_current_28view_jitter_ablation_seed7/`
