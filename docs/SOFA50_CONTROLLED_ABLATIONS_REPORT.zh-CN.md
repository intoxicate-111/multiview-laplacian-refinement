# Sofa50 C2F2 受控消融报告

状态时间：2026-08-12 10:03 BST

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

已完成实验使用最终 `metrics.json` 或分析产物。

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

### 7.3 Current-query 50k downstream evaluation

三个 checkpoint 使用旧 A/B comparison 的相同 synthetic-current evaluator、manifest、25 test samples、target、zero-RGB 条件和 recovery contract。旧 GT-query 50k/current-query 20k 结果通过 regression check。

| Metric | GT-query 50k | Current-query 20k | Current-query 50k |
|---|---:|---:|---:|
| Evaluation loss | 0.0145785 | 0.0117459 | 0.0112148 |
| Vector L2 | 2.994202 | 2.391465 | 2.285737 |
| Global cosine | 0.883648 | 0.895147 | 0.896569 |
| High-10% cosine | 0.967048 | 0.974082 | 0.975401 |
| Pred/target norm | 1.006536 | 0.914148 | 0.946351 |
| Zero-RGB loss | 0.0243697 | 0.0362934 | 0.0346068 |
| Correct-zero gap | 0.0097912 | 0.0245475 | 0.0233920 |
| Refined Chamfer | 0.00551716 | 0.00417940 | 0.00422413 |
| Refined P2S | 0.00567282 | 0.00423266 | 0.00424708 |
| Normal consistency | 0.922792 | 0.940034 | 0.939325 |
| Introduced flips | 11,384 | 8,421 | 8,488 |
| Improved over initial | 0/25 | 5/25 | 3/25 |

Current-query 50k 相对 current-query 20k：

- Evaluation loss 下降 4.52%，vector L2 下降 4.42%。
- Global cosine 增加 0.001423，high-10% cosine 增加 0.001320。
- Correct-zero gap 下降 4.71%，relative correct-vs-zero improvement 从 67.6362% 变为 67.5937%。
- Refined Chamfer 增加 1.07%，P2S 增加 0.34%，normal consistency 下降 0.000709，introduced flips 增加 67。
- Improved-over-initial sample count 从 5/25 变为 3/25；`v00` 和 `v04` 不再低于各自 initial Chamfer。

Current-query 50k 相对 GT-query 50k：

- Evaluation loss 与 vector L2 分别下降 23.07% 和 23.66%。
- Refined Chamfer 与 P2S 分别下降 23.44% 和 25.13%。
- Normal consistency 增加 0.016533，introduced flips 减少 2,896，improved-over-initial count 为 3/25 对 0/25。
- Correct-RGB loss 低于 zero-RGB loss，image dependence 保留。

结论：

- 20k 延长到 50k 的 native validation 与 synthetic prediction metrics 同时下降。
- 20k 延长到 50k 未降低 reconstruction Chamfer 或 P2S，且 improved-over-initial count 从 5 降为 3。
- 在相同 50k budget 下，current-query 的记录 prediction 与 reconstruction endpoints 均处于 GT-query 对应值的指定方向。

### 7.4 Current-graph exact-target oracle recovery

Job 15675 在固定 manifest 的 25 个 test variants 上比较 current-query 20k、current-query 50k 和 current-graph exact-target oracle。Oracle 使用 current-query 50k 的 predicted confidence、保存的 visibility 和相同 recovery weight，只将 recovery 输入从 `delta_pred_hat` 替换为保存的 `delta_target_hat`。未执行训练、数据生成、target 生成、graph 修改或 solver 修改。

Manifest SHA-256 为 `b28e133c277032cceee05ac10115d11ee3007bbd2c3983c31cfa41992159eba3`。20k/50k learned replay、`L_current @ P_proxy` target 公式和 oracle 两次重复恢复均通过检查。两次 oracle recovery 的 recorded metrics 最大绝对差为 0，25 个 OBJ 文件的 SHA-256 相同。Raw target round-trip 最大绝对误差为 `5.96e-8`；current-graph proxy raw target 与 normalized formula 的最大绝对误差为 0。

| Metric | Current-query 20k | Current-query 50k | Exact-target oracle |
|---|---:|---:|---:|
| Evaluation loss | 0.0117434 | 0.0112163 | 0 |
| Target EPE | 2.390879 | 2.286041 | 0 |
| Global cosine | 0.895213 | 0.896632 | 1.000021 |
| High-10% cosine | 0.974112 | 0.975400 | 1.000000 |
| Pred/target norm | 0.914106 | 0.946364 | 1.000000 |
| Initial Chamfer | 0.00391323 | 0.00391323 | 0.00391323 |
| Refined Chamfer | 0.00417977 | 0.00422430 | 0.00317485 |
| Initial P2S | 0.00393459 | 0.00393459 | 0.00393459 |
| Refined P2S | 0.00423260 | 0.00424771 | 0.00317849 |
| Initial normal consistency | 0.955191 | 0.955191 | 0.955191 |
| Refined normal consistency | 0.940028 | 0.939283 | 0.963383 |
| Introduced flips | 8,424 | 8,495 | 3,242 |
| Improved over initial | 5/25 | 3/25 | 25/25 |

Exact-target oracle 相对 initial 的 mean Chamfer 变化为 `-18.87%`，P2S 变化为 `-19.22%`。Oracle refined Chamfer 相对 20k 和 50k 分别为 `-24.04%` 和 `-24.84%`；P2S 分别为 `-24.90%` 和 `-25.17%`；introduced flips 分别为 `-61.51%` 和 `-61.84%`。五个 objects 各有 5/5 variants 的 oracle Chamfer 低于 initial。

Lost-success samples：

| Sample | Checkpoint | Normalized EPE | Top-10% normalized residual | Shared-weight normalized RMS | Raw residual RMS | Raw residual max | Shared-weight raw RMS | Refined Chamfer | Flips |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `43bd...__v00` | 20k | 1.824392 | 7.293408 | 2.404215 | 0.0172963 | 1.185689 | 0.0104527 | 0.00436591 | 263 |
| `43bd...__v00` | 50k | 1.679789 | 6.691750 | 2.243700 | 0.0229498 | 1.696373 | 0.0127689 | 0.00467236 | 299 |
| `43bd...__v00` | oracle | 0 | 0 | 0 | 0 | 0 | 0 | 0.00355531 | 106 |
| `43bd...__v04` | 20k | 1.767634 | 6.858504 | 2.350372 | 0.0177445 | 1.213729 | 0.0188273 | 0.00442270 | 254 |
| `43bd...__v04` | 50k | 1.670313 | 6.434657 | 2.237047 | 0.0237216 | 1.724809 | 0.0253384 | 0.00451079 | 272 |
| `43bd...__v04` | oracle | 0 | 0 | 0 | 0 | 0 | 0 | 0.00344048 | 90 |

结论：

- Exact-target oracle 在 25/25 samples 和 5/5 objects 上降低 Chamfer；固定 `P_proxy`/target 与 recovery objective 在 zero-prediction-error 输入下产生低于 initial 的结果。
- 该对照把当前 synthetic-current downstream 差异定位到 learned Laplacian prediction error 与 recovery 的交互。
- 20k 与 50k 的两个非零 error endpoints 不能给出因果 prediction-error threshold。
- v00/v04 中，50k 的 normalized EPE、top-10% normalized residual 和共享权重 normalized RMS 均低于 20k；raw residual RMS、raw maximum 和共享 50k recovery weight 下的 raw RMS 均高于 20k，且 refined Chamfer 高于 20k。两个 samples 的 solver-input raw-tail pattern 均为 true。
- Oracle 在五个 objects 上均为 5/5，learned recovery 的单 object success pattern 未保留。

### 7.5 Raw-residual Top-k oracle replacement

Job 15677 使用与第 7.4 节相同的 manifest、25 个 synthetic-current test samples、checkpoint、target、confidence、visibility 和 recovery solver。每个 checkpoint/sample 按 recovery 实际接收的 raw solver-input residual
`||delta_pred_raw[i] - delta_target_raw[i]||_2` 降序排列；normalized residual 仅作为对照字段。0/1/10/20/50/100% replacement 形成嵌套集合，正比例顶点数使用 `ceil`，并列值按 vertex index 升序处理。

Manifest、50 个 checkpoint/sample 的 Top-k 选择、两个 0% learned endpoints 和 current-query 50k 的 100% exact-target endpoint 均通过契约检查。Job 状态为 `COMPLETED (0:0)`，运行时间为 2:44:22。

| Checkpoint | Replacement | Raw residual energy replaced | Refined Chamfer | Chamfer oracle gap closed | Refined P2S | P2S oracle gap closed | Normal consistency | Introduced flips | Improved/25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Current-query 20k | 0% | 0.0000% | 0.00417912 | 0.00% | 0.00423241 | 0.00% | 0.940038 | 8,421 | 5 |
| Current-query 20k | 1% | 83.9786% | 0.00384648 | 33.20% | 0.00384780 | 36.57% | 0.942978 | 7,133 | 16 |
| Current-query 20k | 10% | 98.5787% | 0.00358803 | 58.99% | 0.00359401 | 60.70% | 0.951773 | 4,629 | 25 |
| Current-query 20k | 20% | 99.3648% | 0.00345724 | 72.04% | 0.00346345 | 73.11% | 0.955379 | 4,031 | 25 |
| Current-query 20k | 50% | 99.8679% | 0.00323774 | 93.95% | 0.00324234 | 94.14% | 0.960823 | 3,503 | 25 |
| Current-query 20k | 100% | 100.0000% | 0.00317707 | 100.00% | 0.00318069 | 100.00% | 0.963353 | 3,243 | 25 |
| Current-query 50k | 0% | 0.0000% | 0.00422421 | 0.00% | 0.00424747 | 0.00% | 0.939335 | 8,486 | 3 |
| Current-query 50k | 1% | 85.1477% | 0.00383752 | 36.85% | 0.00384135 | 37.99% | 0.942262 | 7,153 | 17 |
| Current-query 50k | 10% | 98.8987% | 0.00356691 | 62.64% | 0.00357169 | 63.22% | 0.951897 | 4,490 | 25 |
| Current-query 50k | 20% | 99.5244% | 0.00343960 | 74.77% | 0.00344401 | 75.16% | 0.955357 | 3,952 | 25 |
| Current-query 50k | 50% | 99.9058% | 0.00322792 | 94.94% | 0.00323261 | 94.94% | 0.960826 | 3,516 | 25 |
| Current-query 50k | 100% | 100.0000% | 0.00317485 | 100.00% | 0.00317849 | 100.00% | 0.963383 | 3,242 | 25 |

两组 checkpoint 的 mean Chamfer 在 1% replacement 时低于 mean initial Chamfer；首次关闭至少 90% Chamfer oracle gap 的记录比例均为 50%。20k 的 sample-level first-improvement 分布为 0%: 5、1%: 11、10%: 9；50k 为 0%: 3、1%: 14、10%: 8。25/25 samples 均在 10% replacement 时低于各自 initial Chamfer。50 个 checkpoint/sample 的 Chamfer 和 P2S 在六个 replacement arms 上均单调不增。

Baseline recovery 的 raw-residual percentile groups：

| Checkpoint | Percentile group | Raw residual mean | Normalized residual mean | Initial vertex-to-GT-surface distance | Recovered vertex-to-GT-surface distance |
|---|---|---:|---:|---:|---:|
| Current-query 20k | Top 0–1% | 0.172977 | 0.475026 | 0.0294214 | 0.0661856 |
| Current-query 20k | 1–10% | 0.0214065 | 0.796804 | 0.0114638 | 0.0133091 |
| Current-query 20k | 10–20% | 0.00515213 | 1.823238 | 0.00691102 | 0.00715702 |
| Current-query 20k | 20–50% | 0.00232813 | 2.807241 | 0.00423863 | 0.00422278 |
| Current-query 20k | Bottom 50% | 0.000904194 | 2.580521 | 0.00229025 | 0.00231084 |
| Current-query 50k | Top 0–1% | 0.169111 | 0.462460 | 0.0293045 | 0.0721756 |
| Current-query 50k | 1–10% | 0.0205310 | 0.781526 | 0.0115668 | 0.0135119 |
| Current-query 50k | 10–20% | 0.00490782 | 1.832673 | 0.00690671 | 0.00710504 |
| Current-query 50k | 20–50% | 0.00223119 | 2.681124 | 0.00424841 | 0.00425367 |
| Current-query 50k | Bottom 50% | 0.000849648 | 2.446157 | 0.00226908 | 0.00227088 |

Top 1% raw-residual group 的 baseline recovered surface distance 分别是 bottom 50% 的 28.64 倍和 31.78 倍。两组 checkpoint 的 mean vertex-wise raw residual 与 recovered surface distance 的 baseline Spearman 相关系数分别为 0.4256 和 0.4357。

结论：

- High raw-residual vertices 对应较高的 recovered vertex-to-GT-surface distance；该关系在两组 checkpoint 上保留。
- Top 1% vertices 包含 83.98–85.15% raw residual energy，replacement 后关闭 33.20–36.85% Chamfer oracle gap，并将 improved count 增加到 16/25 和 17/25。
- Top 10% replacement 使 25/25 samples 的 Chamfer 低于 initial，但只关闭 58.99–62.64% mean Chamfer oracle gap。
- 达到至少 90% mean Chamfer 和 P2S oracle-gap closure 需要 50% replacement。当前结果不支持 downstream gap 仅由 Top 1% 或 Top 10% raw-residual vertices 构成。
- Raw residual percentile 与 normalized residual percentile 不等价；Top 1% raw-residual group 的 normalized residual mean 低于其余记录 groups，normalized residual 未用于该实验的 Top-k 选择。

## 8. Local query jitter 最终结果

Jobs 15662_0、15662_1 和 15663 均以 exit code `0:0` 完成。两个训练 arm 均达到 20,000 optimizer steps。

| Metric | A: no jitter | B: local jitter | B − A |
|---|---:|---:|---:|
| Best validation loss | 0.018456638 | 0.018836601 | +0.000379964 |
| Test raw endpoint | 0.007681539 | 0.007804981 | +0.000123442 |
| Test raw Top-10% endpoint | 0.053443110 | 0.054163195 | +0.000720084 |
| Test raw Top-1% endpoint | 0.202496550 | 0.225826787 | +0.023330238 |
| Test raw global cosine | 0.957896502 | 0.928684516 | -0.029211986 |
| Runtime | 6.0416 h | 6.1807 h | +0.1391 h |
| Runtime ratio | 1.0000 | 1.0230 | +2.3023% |

OpenMVS48 current-mesh recovery 使用 5 个 paired meshes：

| Metric | A: no jitter | B: local jitter | B − A |
|---|---:|---:|---:|
| Mean initial Chamfer | 0.024729284 | 0.024729284 | 0 |
| Mean refined Chamfer | 0.025067426 | 0.025249771 | +0.000182345 |
| Mean refined P2S | 0.024900180 | 0.025077573 | +0.000177393 |
| Mean refined normal consistency | 0.819698948 | 0.819023295 | -0.000675653 |
| Improved-over-initial meshes | 0/5 | 0/5 | 0 |
| Introduced flipped faces | 329 | 370 | +41 |

结论：

- Arm B 的 best validation loss、test raw endpoint、raw Top-10% endpoint 和 raw Top-1% endpoint 均高于 Arm A，test raw global cosine 低于 Arm A。
- Arm B 的 OpenMVS refined Chamfer 在 5/5 paired meshes 上高于 Arm A；两个 arm 均为 0/5 meshes 低于各自 initial Chamfer。
- 当前记录不支持在该 contract 下启用 training-only local query jitter。
- 独立报告位于 `docs/SOFA50_LOCAL_QUERY_JITTER_ABLATION_REPORT.zh-CN.md`。

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
| Current-query 50k 相对 current-query 20k 降低 synthetic prediction loss/EPE | Supported | Loss -4.52%；EPE -4.42% |
| Current-query 50k 相对 current-query 20k 降低 reconstruction Chamfer/P2S | Not supported | Chamfer +1.07%；P2S +0.34% |
| Current-query 50k 相对 GT-query 50k 改变 matched-budget downstream endpoints | Supported for recorded protocol | Chamfer -23.44%；P2S -25.13%；flips -25.44% |
| Exact current-graph target 在固定 recovery contract 下改善 synthetic-current geometry | Supported | Chamfer 25/25；mean 相对 initial -18.87% |
| 50k lost-success samples 的 raw solver-input tail 低于 20k | Not supported | v00/v04 的共享权重 raw RMS 均升高 |
| Top 1% raw-residual vertices 单独构成主要 downstream oracle gap | Not supported | 20k/50k Chamfer gap closure 33.20%/36.85% |
| Top 10% raw-residual replacement 使每个 sample 低于 initial | Supported | 20k/50k 均为 25/25 |
| Top-k replacement 关闭至少 90% mean Chamfer oracle gap | Supported at 50% replacement | 20k/50k 为 93.95%/94.94% |
| High raw-residual vertices 对应 high recovered geometry error | Supported for recorded percentile analysis | Top 1%/bottom 50% surface-distance ratio 28.64/31.78 |
| Local query jitter 降低最终 synthetic 与 OpenMVS recovery error | Not supported | Test raw EPE +0.000123442；OpenMVS Chamfer +0.000182345；5/5 paired meshes |

## 10. 尚需完成的判定

1. 对 GT 与 GT-adaptive prediction 做 common-surface paired evaluation：映射到相同 GT vertices 或固定表面采样点，使用同一 target、同一 curvature bins、同一 EPE/cosine 定义。
2. 在 adaptive 的 common-surface 指标完成前，不从 graph-specific raw EPE 推导 Sofa50 主训练配置。

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
- Synthetic-current 50k downstream evaluation：
  `runs/learned_laplacian/sofa50_synthetic_current_50k_downstream_evaluation_seed7/`
- Synthetic-current exact-target oracle recovery：
  `runs/learned_laplacian/sofa50_synthetic_current_50k_downstream_evaluation_seed7/oracle_recovery_comparison/`
- Synthetic-current Top-k raw-residual oracle replacement：
  `runs/learned_laplacian/sofa50_synthetic_current_50k_downstream_evaluation_seed7/oracle_recovery_comparison/topk_prediction_error_recovery_comparison/`
- Local-jitter runs：
  `runs/learned_laplacian/sofa50_synthetic_current_28view_jitter_ablation_seed7/`
- Local-jitter 独立报告：
  `docs/SOFA50_LOCAL_QUERY_JITTER_ABLATION_REPORT.zh-CN.md`
