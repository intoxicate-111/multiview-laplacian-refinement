# 多视图 Laplacian 网格细化

[English](README.md) | [简体中文](README.zh-CN.md)

方法定义：[Sofa50 标准流程](docs/CANONICAL_SOFA50_PIPELINE.md)

训练说明：[多网格训练](docs/MULTI_MESH_TRAINING.zh-CN.md)

可见性与恢复：[可见性感知恢复报告](docs/VISIBILITY_AWARE_RECOVERY_REPORT.md)

当前 recovery-aware 研究：[中文](docs/SOFA50_RECOVERY_AWARE_STUDY.zh-CN.md) | [English](docs/SOFA50_RECOVERY_AWARE_STUDY.md)

实验指标与运行状态：[实验数据汇总](docs/EXPERIMENT_DATA_SUMMARY.zh-CN.md)

Chamfer evaluator 事故：[中文报告](docs/CHAMFER_EVALUATION_INCIDENT_2026-08-21.zh-CN.md) | [English report](docs/CHAMFER_EVALUATION_INCIDENT_2026-08-21.md)

当前 Sofa50 受控消融：[Direct-raw/loss/expert/image-feature 报告](docs/SOFA50_CONTROLLED_ABLATIONS_REPORT.zh-CN.md)

Sofa50 更强 coarse-mesh smoothing：[v2 中文说明](docs/SOFA50_STRONG_SMOOTHING_V2.zh-CN.md) | [English](docs/SOFA50_STRONG_SMOOTHING_V2.md)

Future2000 本地对比任务：[本地任务说明](docs/FUTURE2000_LOCAL_COMPARISON_TASKS.md)

Future2000 formal mixed-loss 结果：[完整 2,000-object / 1,000-test-mesh 报告](reports/future2000_mixed_vs_old_external_20260831_v2/FINAL_REPORT.md)

当前 Sofa50 formulation 压力测试：[综合报告](reports/sofa50_multitopology_rawlap500_v2/recent_ablation_and_old_domain_comparison_v1/REPORT.md)

近期 commit 与实验记录：[8 月 4–15 日报告与补充记录](docs/RECENT_COMMIT_AND_EXPERIMENT_REPORT_2026-08-04_2026-08-14.zh-CN.md)

View-count 与 query-resolution 结果：[消融报告](runs/learned_laplacian/sofa50_c2f2_view_query_resolution_ablation_20k_seed7/analysis/REPORT.md)

28-view current-graph target/loss-space 结果：[H2 消融报告](runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7/analysis/REPORT.md) | [25 组可视化总览](runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7/analysis/comparison_images/B_direct_raw_laplacian/overview_25.png)

## 方法贡献的适用范围

> **Contribution（正式定义）.** A frequency-aware, operator-guided mesh refinement
> framework that converts multiview high-frequency visual evidence into
> complementary differential and positional geometric constraints, and
> reconciles them through an explicit differentiable linear geometric
> operator.

中文表述：这是一个**频率感知、算子引导的网格细化框架**；它把多视图高频视觉证据
转化为互补的差分几何约束与位置几何约束，再通过显式、可微的线性几何算子协调两者。

具体而言，高频视觉分支同时向两种几何表示提供 encoder feature `F` 和残差
`F-G(F)`。Arm B 将视觉证据转化为 raw differential constraints，Arm E 将其转化为
direct positional prior。冻结 B+E 及其可训练扩展通过下式协调两者：

```text
(L_U^T L_U + lambda I) V_H = L_U^T delta_B + lambda V_E.
```

其中 `L_U=I-D^-1 A` 是显式几何算子，求解过程可微。“频率感知”指高频视觉证据和
实测 graph-frequency 行为，并不表示网络或恢复过程显式执行 mesh 特征分解。

当前证据不能证明 recovery-aware Arm-B training 是该组合所必需的。使用相同 Arm-E
anchor 的 matched direct-Laplacian Arm-A，在 `lambda=0.03` 与 `lambda=0.01` 下与
B+E 的 surface-distance 差异均无统计区分；dense B+E 同时是 positional-constraint
density family 平滑变化的 100% endpoint。因此可辩护的贡献应限定为 learned dense
positional/differential operator composition 及其实测 trade-off，而不是声称该公式
或 Arm-B training principle 没有前身。

## Frozen B/E 图频率分析

选定的 frozen Arm B、Arm E 与 B+E 在完全相同的 matched-v2 meshes 上评估。误差
诊断使用 symmetric-normalized graph Laplacian
`Lsym=I-D^-1/2 A D^-1/2`，并以 Chebyshev--Jackson 近似划分
`low=[0,2/3)`、`mid=[2/3,4/3)`、`high=[4/3,2]`。下表给出绝对 XYZ error
energy；主要证据是绝对能量，而不是 normalized fraction。

| Split | 方法 | Total | Low | Mid | High |
|---|---|---:|---:|---:|---:|
| validation | Arm B | 59.81110 | 47.81169 | 9.05632 | 2.94309 |
| validation | Arm E | 22.56830 | 8.53185 | 9.99271 | 4.04375 |
| validation | Frozen B+E | 30.17443 | 18.60651 | 8.72829 | 2.83963 |
| test | Arm B | 102.25649 | 74.69651 | 22.59283 | 4.96715 |
| test | Arm E | 55.86585 | 24.49914 | 24.49353 | 6.87319 |
| test | Frozen B+E | 67.31840 | 40.48458 | 21.97737 | 4.85644 |

这是一个有边界的频谱互补结论，并不表示每个频段都形成了独立 specialist。Arm B 的
误差明显由低频主导；Arm E 的 total 和 low-frequency error 最低，但其绝对 mid/high
error 均高于 B 与 Hybrid。B+E 的 low error 位于 B/E 之间，同时把 mid/high error
略微降到两个分支以下。test 上 `Hybrid-B` 变化的绝对能量有 99.11% 位于 low band；
Hybrid 的 component-translation RMS（`0.0071013`）也几乎复现 E（`0.0070970`），
而不是 B（`0.0118360`）。因此 E 提供 positional/Laplacian-nullspace anchor，B 保留
differential structure，并略微降低 mid/high-frequency residual。

恢复本身还具有严格的算子频谱刻画。令
`A_R=L_U^T L_U=Q Lambda Q^T`，并以
`A_R V_B_dagger=L_U^T delta_B` 定义 `V_B_dagger`；仅为消除顶点空间比较中的
歧义，其逐连通分量常数 gauge 取自 `V_E`。则每个 recovery mode 都严格满足：

```text
v_H,k = Lambda_k/(Lambda_k+lambda) v_B_dagger,k
      + lambda/(Lambda_k+lambda) v_E,k.
```

这个等式是严格的；只有用于汇总频段能量的 Chebyshev--Jackson projector 是近似的。
归档 Arm-B 网格并不是 `V_B_dagger`：它包含独立的 `1e-2 V_input` anchor，因此仍作为
不同的经验比较项报告。实际恢复仍只执行稀疏矩阵方程，并不显式做 eigendecomposition。

直接 `A_R` 审计的 100 个 validation/test meshes 全部通过。normal-equation residual
最大为 `2.839e-12`；独立计算再相加 B/E transfer contribution 后，重建 Hybrid 的最大
vertex RMS 为 `1.005e-11`。采用逐 mesh `Lambda/Lambda_max` 频段后，tight float64
reference energy 为：

| Split | Signal | Total | Low | Mid | High |
|---|---|---:|---:|---:|---:|
| validation | Archived B | 59.81110 | 51.23842 | 4.75454 | 3.81814 |
| validation | E | 22.56830 | 11.41362 | 5.74857 | 5.40611 |
| validation | Hybrid | 30.18638 | 21.73557 | 4.67255 | 3.77826 |
| test | Archived B | 102.25649 | 84.72261 | 11.48790 | 6.04598 |
| test | E | 55.86585 | 33.71023 | 13.76141 | 8.39422 |
| test | Hybrid | 67.33614 | 49.97247 | 11.36961 | 5.99406 |

更关键的是，test 上 `Hybrid-V_B_dagger` 变化能量的 `99.862%` 位于严格定义的
E-dominant 区间 `Lambda<lambda/2`。对实际 archived B 比较，结论仍然很强：
`80.932%` 位于 E-dominant、`13.173%` 位于 transition，只有 `5.896%` 位于
B-dominant 的 `Lambda>=2lambda`；按 mesh-relative 划分，最低三分之一占
`99.930%`。因此，真实 recovery operator 直接确认：E 主要通过低 `Lambda` 的
positional/nullspace modes 改变 B。反过来，`Hybrid-E` 变化能量的 `73.240%` 位于
B-dominant，只有 `9.577%` 位于 E-dominant，直接说明 B 向 E 提供响应更高的
differential correction。无 anchor 的 `V_B_dagger` 存在巨大的低模误差能量，它是
理论端点而不是竞争模型输出。

完整逐样本 energy、exactness audit、图和 protocol 见
[真实 recovery-operator 报告](reports/sofa50_multitopology_rawlap500_v2/recovery_operator_spectrum_v1/REPORT.md)、
[Frozen B+E 报告](reports/sofa50_multitopology_rawlap500_v2/frozen_hybrid_recovery_v1/FINAL_REPORT.md)
和
[Frozen vs joint 机制报告](reports/sofa50_multitopology_rawlap500_v2/frozen_vs_joint_mechanism_analysis_v1/FINAL_REPORT.md)。

## 项目状态

状态日期：2026-09-04 08:14 BST。

OpenMVS 使用政策：现有 Sofa50 OpenMVS mesh 是低质量外部重建，只保留为
分布外压力测试；它不是 training target、pseudo-GT、模型选择端点或期望质量
ceiling。详见 [OpenMVS 输入使用政策](docs/OPENMVS_INPUT_POLICY.zh-CN.md)。

| 组件 | 状态 | 结论 |
|---|---|---|
| Current-query/current-graph 训练流程 | 当前主线 | 模型直接预测 raw target `L_current @ P_proxy`；不进行 `h^2` target normalization 或 output denormalization。 |
| 历史 GT-query 流程 | 保留用于对照 | 绝对 GT `h^2`-normalized Laplacian 监督仍可复现，但不再是当前训练主线。 |
| Target 泄漏控制 | 已实现并测试 | 模型输入不包含 proxy positions 或监督 raw/normalized Laplacian 数值。 |
| Sofa50 960 图像分辨率消融 | 已完成 | 在 50,000 个 optimizer steps 下，F2 的 exact-query error 低于 F0 和 F1。 |
| Sofa50 960 C2F2 训练 | 已完成 | 三个 seed 均完成 50,000 个 optimizer steps。C2F2 是当前 exact-query error 最低的配置。 |
| Sofa50 1920 C2F2 训练 | 已完成 | 三个 seed 均完成 20,000 个 optimizer steps。平均 endpoint error 和 recovery Chamfer 高于 960 结果；平均 cosine 更高。 |
| Expanded-query recovery | 已完成 | 对已评估的 960 和 1920 C2F2 checkpoint，五个 validation mesh 的 Chamfer 均增加。 |
| OpenMVS coarse-mesh recovery | 50 个物体中完成 48 个；仅诊断 | 低质量重建只作为 OOD 压力输入，不作为目标或决策端点。历史 refinement metrics 保留；两个物体缺少 OpenMVS coarse mesh。 |
| OpenMVS projected-GT oracle 分解 | 已完成；仅诊断 | Position projection 令统一 Chamfer 改善 6.12%；冻结 recovery 保留其中 44.53%，learned prediction 最终实现 4.36%。该结果仅诊断低质量 OOD 输入上的失败，模型选择/scale-up 权重为零。 |
| Oracle residual expert | 已关闭 | 2,000-step diagnostic 不支持该分支。 |
| 14/28/56-view 消融 | 已完成 | 14、28、56 views 的 best validation loss 分别为 0.0139316、0.0130296、0.0138104。 |
| Query-graph resolution 消融 | 已完成 | GT alias、GT-sub1、GT-adaptive 的 best validation loss 分别为 0.0139316、0.0614830、0.0145840；GT-sub2 未训练。 |
| 28-view + GT-adaptive 组合 | 已完成 | Best validation loss 为 0.0131095；五个 matched validation meshes 上的 raw EPE 为 0.002879。 |
| 28-view current-graph H2 消融 | 已完成 | 统一 test/recovery 评估中 direct raw-Laplacian training 最优：raw EPE 0.00300525、refined Chamfer 0.00380671，改善 19/25 samples。 |
| 冻结模型三轮递归 | 已完成 | 改善数从 Arm-B 基线 `19/25` 降为 `12/25`、`7/25`、`2/25`；重复 inference 不是有效提升路径。 |
| Stage-2 分布适配 | 已完成 | X1 训练分支的最好结果为 `16/25`、Chamfer `0.00384032`，未超过冻结 Arm-B 基线。 |
| Arm-B Huber 饱和诊断 | 已完成 | GT 曲率 top 1% 中，66.049% vertex 至少一个分量饱和，gradient retention 为 58.436%。 |
| Raw MSE 与 Huber | 已完成 | Raw MSE 未降低 test Top-10%/Top-1% error，也未降低 mean Chamfer/P2S。MSE 使用 global batch 6，基线为 2，因此不是严格单变量训练对比。 |
| Learned dynamic residual expert 与 gate | 已完成 | Learned final 在 25/25 test samples 上的 raw EPE、Chamfer 和 P2S 均优于联合训练的 base。Validation-selected constant gate 与 5 个 mesh 内 shuffle 干预表明：expert 是主要贡献，vertex-level placement 还提供较小但可测的额外增益。 |
| 960 image-feature 消融 | 已完成 | `F + (F-Gaussian(F))` 的 test raw EPE、RMS、Top-10% 和 Top-1% 最低；Gaussian-only 的 mean Chamfer、normal consistency 和改善数（`21/25`）最好。 |
| Native-1920 + high-frequency residual | 已完成；非严格 resolution ablation | 4×L40、20,000-step 实验降低 Bottom-90% error，但相对 960+HF 恶化 test raw EPE/RMS、Top-10%/Top-1%、Chamfer 和 P2S；normal consistency 改善，flips 减少。Global batch 为 4 对 2。 |
| Sofa50 多拓扑 coarse smoothing v2 | 20k 训练与受控 test/recovery 已完成 | Job `17082` 已在 2×L40 上从零完成，final/best validation 为 `2.26915e-6`。在相同 50 个 strong-smoothing test inputs 上，v2 将 raw EPE 从 v1 的 `0.00840367` 降到 `0.00276820`，但统一 refined Chamfer 更差（`0.00451747` vs `0.00426879`），normal 更低、flips 更多，改善数仅 26/50 对 38/50。更强 smoothing 改善 target prediction，但收益没有通过冻结 recovery contract 传递到 geometry。 |
| Exact-target sparse-recovery 诊断 | 已完成 | 只用 component-centroid gauge 的全方程 sparse solve 在 v2 达到 mean oracle efficiency `0.92366`。在 `0.01` anchor 后加入 hard visibility，会把 mean efficiency 从 `0.34258` 降到 `0.16875`，并令 44/50 samples 变差；confidence 可忽略，2,000 Adam steps 也未消除坍塌。 |
| Recovery-aware Arm A/B | 已完成 | 使用相同 `lambda=beta=10^-2` sparse recovery 时，Arm B 将 test Chamfer 从 A 的 `0.00395529` 降至 `0.00358497`，recovered vertex RMS 从 `0.0135181` 降至 `0.0115532`，尽管 raw EPE 更差。这支持 geometric-utility supervision，而不是 raw regression 改善。 |
| Recovery-aware Arm C/D | 已完成 | 减弱 positional regularization 未改善 recovered geometry。C（`lambda=10^-3`）与 D（`10^-4`）的 test Chamfer 分别为 `0.00414926`、`0.00653139`，均差于 B（`10^-2`）的 `0.00358497`。两组使用已记录的 float64 PCG，objective 未改变。 |
| Direct-vertex Arm E | 已完成 | 826,115 参数 C2F2+HF 仅以 same-index vertex MSE 训练 `Delta V`。Test Chamfer 为 `0.00334039`、vertex RMS 为 `0.00822130`，45/50 inputs 改善；E 不含 Laplacian target、sparse solver 或 recovery gate。 |
| 冻结 B+E hybrid recovery | 已完成；只读 | 使用冻结 B Laplacian、冻结 E positions 作为唯一 anchor，并用 validation 选择 `lambda=3e-2`；test Chamfer 为 `0.00302983`，49/50 inputs 改善。它支持后续联合 hybrid 训练，但本身没有重训，也不授权 scaling。 |
| Direct-Lap A+E matched 对照 | 已完成 | `lambda=0.03` 时 A+E CD 为 `0.00298590`、B+E 为 `0.00302983`；`lambda=0.01` 时分别为 `0.00314166` 与 `0.00319840`。两组 paired CD CI 均含零，不能据此声称 Arm-B recovery-aware training 是 operator composition 的必要条件。 |
| Arm-B anchor-conditioning 消融 | 已完成 | B_P@V_P 相对 B_0@V_P 的 CD 差为 `-0.00001703`，mesh/object CI 均跨零；interaction 虽显著，但没有形成 final same-anchor gain 证据。 |
| Sparse positional-density 消融 | 已完成；只读 | Dense B+E 是平滑 densified-anchor family 的 endpoint。Fixed-lambda test CD 从 0% 的 `0.0330216` 单调降到 100% 的 `0.00302983`；Song-scale 2% 仍比 dense 高 `243.70%`，paired 结果为 0/50 胜。 |
| End-to-end direct–Laplacian hybrid | 已完成 | 单个 892,678 参数共享模型以 final-hybrid-only supervision 预测 latent raw-Laplacian 与 direct-displacement fields。Matched-v2 test Chamfer 为 `0.00341857`，不及 frozen B+E 的 `0.00302983`；机制审计为 MECH5，不支持“分开训练普遍更优”的泛化结论。 |
| Matched-v2 continuous pretrained B+E | 已完成；validation 选择 | 两个完整独立 specialist 只通过最终 Hybrid geometry 继续联合训练。Validation 选择 step 9,400；matched-test Chamfer 从 geometry-equivalent step-0 的 `0.00302691` 改善到 `0.00288357`，但 legacy/unseen OOD 仍未成功。 |
| Old-domain native-1920 specialists 与 frozen B+E | 已完成 | Validation-selected Arm B、Arm E、post-hoc scalar blend 与 Frozen B+E 已在完全相同的 25 个 inputs 上评估；Chamfer 分别为 `0.00853777`、`0.00806580`、`0.00756219`、`0.00670460`。Frozen 改善 25/25，并以 22/25、23/25、21/25 分别胜 B、E、scalar blend；Normal 与 same-index vertex RMS 则由 scalar blend 占优，因此保留 metric trade-off。 |
| Future2000 formal mixed-loss Arm B | 已完成 | 正式 200,000-step checkpoint 保留 `L_raw-Laplacian-Huber + 10^-2 L_recovered-vertex`。在 200 个 held-out objects × 5 个冻结变体上，Chamfer 从 `0.00776417` 降至 `0.00476457`，975/1000 samples 改善；相对 archived old-structure predictor 的 paired 改善为 `0.000464982`（882/1000 meshes、185/200 object means，object-bootstrap CI 不含零）。 |
| Future2000 direct-vertex Arm E | 已完成 | Job `17888` 于 2026-09-04 完成 200,000 steps（`0:0`，elapsed `2-16:05:51`）。Validation-selected epoch-160 checkpoint SHA-256 为 `5a6aaa32bec6edcdd2c30face02c4ae8bc139fef18d4d05b3394c987057cb50f`；fusion lambda 锁定前 test metrics 仍保持封存。 |
| Future2000 frozen B+E 对比 | Validation 运行中；test 由依赖门控 | Validation array `18673` 正在全部 1,000 validation variants 上扫描预声明 lambda grid；`18677` 将按 mean CD 锁参，`18678` 随后一次性评估 Arm-E/B+E test，`18679` 再与 Arm-B、old structure、NDS、nvdiffrec、ExMesh 生成综合报告。目前不声明 Future2000 Arm-E/B+E test metric。 |
| GT-query direct-raw zero-shot transfer | 已完成 | 去掉 `h^2` normalization 相对历史 GT-query arm 明显改善，但 current-mesh recovery 仅达到 Chamfer `0.00400486`、`4/25`，仍低于 supervised current-query HF 的 `0.00377832`、`20/25`。 |
| Future2000 GT-adaptive 扩展 | Formal 200k Arm-B test 已完成 | 数据由 2,000 个不同的 3D-FUTURE source objects 构成，每个对象有 5 个冻结的确定性扰动变体，按对象划分为 `8000/1000/1000`。Formal Arm B 达到 Chamfer `0.00476457`、P2S p95 `0.01462829`、F-score `0.88103565`、975/1000 改善；normal consistency 从 `0.92425235` 降至 `0.90859736`。 |
| Sofa50 同初始网格外部 benchmark | 已完成并修正 evaluator | Ours、NDS、nvdiffrec 和 ExMesh 均从相同 current mesh/observations 完成 25/25。Native-metric 聚合问题已通过对全部归档 mesh 使用同一 deterministic evaluator 修复；`contract_audit=true`。 |
| Future2000 外部基线 | Full-1000 对比已完成，invalid metrics 显式保留 | Formal Arm B 的 Chamfer 为 `0.00476457`；在 valid paired samples 上，对 NDS、nvdiffrec、ExMesh 分别胜 804/998、829/999、974/996。Input-contract audit 通过；2 个 NDS invalid metrics、1 个 nvdiffrec failure 与 4 个 ExMesh invalid/topology-changing outputs 均显式保留，因此 strict/full metric completeness 仍为 false。 |
| 自动化测试 | 当前文档改动对应检查通过 | External adapters、same-initial aggregation、raw loss、dynamic expert/gate、image feature、native-1920 和分布式训练相关 targeted tests 通过；下文命令仍是新 checkout 的最终依据。 |

已确定的 representation 主线是在 Sofa50 上建立的 synthetic-current、current-query/
current-graph、direct-raw formulation。最新受控扩展加入 direct-displacement head，
两个 latent branches 只通过最终 hybrid geometry 训练。全部 predicted Laplacian rows
都会被集成；hard visibility、confidence、recovery Huber 和 Adam mesh optimization
均关闭。Clean geometry 只用于 loss，不会输入任一 branch 或 recovery solve。已完成的
formal Future2000 Arm-B scale-up 使用 960 high-frequency construction、28 views、
C2F2、既定 mixed objective 与固定 200,000-step checkpoint。Full-1000 same-initial
comparison 已完成，invalid external outputs 显式保留。Direct-vertex Arm E 已完成，
B+E validation-only lambda sweep 正在运行；lambda lock 写入前 test 仍由依赖门控。

早期 GT-query、`h^2`-normalized formulation 仍作为历史背景保留。该路径迁移到
expanded/OpenMVS query graph 后未改善 geometry，因此不再作为下文数学定义的主线。
此外，由于 initial mesh 质量过差，OpenMVS 明确排除在 target 定义、模型选择与后续
scale-up 决策之外。

## Loss contract：formal mixed Arm B 与独立 ablations

多个实验复用相同 Arm-B predictor 和 sparse operator，但 optimisation objective
并不相同，不能笼统写成同一个“Arm-B loss”。

已完成的 Sofa50 recovery-aware Arm B 与 formal Future2000 Arm B 使用相同双项目标：

$$
\mathcal L_{B,\mathrm{formal}}
=\mathcal L_{\mathrm{lap}}+10^{-2}\mathcal L_{\mathrm{vertex}}.
$$

其中 `L_lap` 是 raw-Laplacian Huber，`L_vertex` 是通过以下 differentiable solve
得到的 recovered-vertex MSE：

$$
(L^\top L+\lambda I)V_B
=L^\top\widehat\delta+\lambda V_{\mathrm{input}},
\qquad \lambda=10^{-2},
$$

Pure recovered-vertex objective 只保留为独立 ablation。它在 matched-v2 上降低
same-index vertex RMS，却恶化 Chamfer 与 raw EPE，且 frozen E 无法挽救。因此不能把
它写成 formal Future2000 checkpoint 的 objective。

Arm E 不使用 sparse solve，只训练 direct-displacement MSE。Continuous B+E 使用两个
完整 specialist，计算

$$
V_H=(L^\top L+\lambda I)^{-1}
\left(L^\top\widehat\delta_B
+\lambda(V_{\mathrm{input}}+\Delta V_E)\right),
$$

同样只优化最终 vertex MSE，不加入 raw-Laplacian 或 direct-displacement auxiliary
loss。Continuous B+E 的训练量是 final-vertex MSE，checkpoint selector 则是
validation-only unified surface Chamfer；锁定 selection 前 test 保持 sealed。

## 当前训练方法

已建立的 A-D 模型根据 28 个标定视图和 current mesh graph，直接预测 raw target
Laplacian：

```text
28-view RGB + 相机 + current vertices/connectivity + 局部几何
    -> direct raw current-graph Laplacian target
```

对于存储的 current mesh `P_current`、faces `F_current` 和配对 proxy positions
`P_proxy`：

```text
L_current       = uniform_laplacian(P_current, F_current)
target_raw      = L_current @ P_proxy
prediction_raw  = model(images, cameras, P_current, F_current)
```

这里不除以 `h_current^2`，不 clip target，也不反归一化 output。配置中的
`target_scaling` 仍声明 edge-scale 定义，因为 `h_current` 用作局部几何 feature 和
validity metadata；当 `target_mode = raw_laplacian` 时，它不会改变 raw target。

当前 geometry mode 为 `query_fourier`，local query jitter 关闭，query 就是 current
vertex position。Image feature、Fourier-encoded query position、current vertex
normal、relative local edge scale、degree、valid-view ratio 和 current connectivity
属于 inference 输入。`P_proxy`、raw target 与 normalized target 都不是模型输入。

当前 high-frequency image branch 使用 encoder feature `F`、固定 Gaussian blur
`G(F)`（kernel 5、sigma 1.0），并采样拼接结果 `[F, F-G(F)]`。Recovery 直接使用
同一 raw 单位的 prediction：

```text
current mesh vertices
  -> 投影到 28 个标定视图
  -> 聚合 original + high-frequency image features
  -> 直接预测 delta_pred_raw
  -> 全方程 regularized sparse Laplacian integration
```

GT geometry 只用于构造监督与评估。Recovery-aware arms 只在 training-side vertex
loss 使用 clean vertices，clean geometry 不进入 model 或 sparse solve。Dynamic
residual expert、gate 和 raw-MSE loss 是已完成受控消融；direct displacement 是独立
audit 的 Arm E 对照。当前 end-to-end hybrid 实验复用相同 shared features，并预测
两个 latent outputs：

```text
28-view RGB + 相机 + current vertices/connectivity + 局部几何
    -> shared C2F2 + HF features
    -> latent raw Laplacian delta_hat
    -> latent direct displacement Delta V_direct
    -> differentiable hybrid solve
    -> final recovered mesh V_H
```

该实验不对任一 latent branch 添加 auxiliary target；clean vertices 只监督 `V_H`。

## 数学定义

以下公式对应当前 direct-raw + high-frequency 训练路径。历史 normalized
formulation 会在独立小节中说明。

### Current-graph uniform Laplacian 与 direct-raw target

令 `N(i)` 为 vertex `i` 的 one-ring neighbours，`d_i = |N(i)|`。Uniform graph
Laplacian 为

$$
(L X)_i = X_i - \frac{1}{d_i}\sum_{j\in N(i)}X_j,
\qquad
L_{ii}=1,\quad L_{ij}=-\frac{1}{d_i}.
$$

变量说明：`i` 是目标 vertex，`j` 遍历其 one-ring 集合 `N(i)`，`d_i` 是该集合的
neighbour 数量。`X in R^{N x C}` 表示定义在 `N` 个 vertices 上的 `C`-channel
signal（这里通常是 3D positions），`L in R^{N x N}` 是 row-normalized uniform
Laplacian，`(LX)_i` 是第 `i` 行输出。

Isolated vertex 对应零 Laplacian row。令训练 query mesh 为
`X_0 = P_current`，`P_proxy` 与其具有完全相同的 vertex ordering。局部边尺度和
当前监督 target 为

$$
h_i^{\mathrm{current}}=
\frac{1}{d_i}\sum_{j\in N(i)}\lVert X_{0,i}-X_{0,j}\rVert_2,
\qquad
\delta_i^*=(L_{\mathrm{current}}P_{\mathrm{proxy}})_i.
$$

变量说明：`X_0,i in R^3` 是 current vertex `i`；`h_i^current` 是其 mean
incident-edge length；`P_proxy in R^{N x 3}` 是具有相同 vertex ordering 的 paired
proxy；`L_current` 只由 current faces 构建；`delta_i* in R^3` 是监督 raw
Laplacian vector。`j`、`N(i)` 和 `d_i` 沿用上面的 one-ring 定义，`||.||_2` 是
Euclidean length。

网络直接预测与 `delta_target_raw = delta*` 单位相同的 `delta_pred_raw`：

$$
f_\theta(I_{1:M},K_{1:M},E_{1:M},X_0,F)_i
=\delta_i^{\mathrm{pred,raw}}\approx\delta_i^*.
$$

变量说明：`f_theta` 是参数为 `theta` 的 learned predictor；`I_1:M`、`K_1:M` 和
`E_1:M` 分别是 `M=28` 个 RGB images、intrinsics 与 world-to-camera extrinsics；
`F` 是 current face array；`delta_i^(pred,raw)` 是 vertex `i` 的三分量输出。

两侧都不乘除 `(h_i^current)^2`。当前配置中，
`target_scaling.method = square_of_mean_incident_edge_length` 只定义可用的尺度
metadata；`target_mode = raw_laplacian` 会在 loss 前直接选择 raw tensor。
`clip_max_norm = null`，因此 target 也不会被 clipping。

### Current-query 合同

主线 query 就是 current vertex：

$$
q_i=X_{0,i}.
$$

变量说明：`q_i in R^3` 是 vertex `i` 用于 image projection 与 positional encoding
的 query，`X_0,i` 是该 vertex 存储的 current position。两者相等表示不添加 query
offset 或 jitter。

`query_training.enabled` 与 `local_query_jitter.enabled` 都是 false。因此 current
connectivity、`P_proxy`、target、local scale 与 Laplacian operator 不会被训练期
query augmentation 改变。

### 投影、renderer visibility 与多视图聚合

对 view `v`，world-to-camera projection 为

$$
y_{vi}=E_v[q_i^\top,1]^\top,
\qquad
\widetilde p_{vi}=K_v y_{vi},
\qquad
(u_{vi},v_{vi})=
\left(\frac{\widetilde p_{vi,x}}{\widetilde p_{vi,z}},
      \frac{\widetilde p_{vi,y}}{\widetilde p_{vi,z}}\right).
$$

变量说明：`v in {1,...,M}` 是 camera index；`[q_i^T,1]^T in R^4` 是 homogeneous
world point；`E_v in R^{3 x 4}` 生成 camera coordinates `y_vi`；
`K_v in R^{3 x 3}` 生成 homogeneous pixel `p_tilde_vi`；`(u_vi,v_vi)` 是除以
camera depth `p_tilde_vi,z` 后的非齐次 pixel coordinates。

令 `f_vi` 表示正深度且投影位于图像范围内，`r_vi` 表示预计算的 renderer-native
back-face 与 occlusion 结果。Feature-sampling mask 为

$$
z_{vi}=f_{vi}r_{vi}\in\{0,1\}.
$$

变量说明：`f_vi` 仅在 positive-depth 且 in-frame 时为 1；`r_vi` 仅在 renderer 的
front-face 与 occlusion tests 接受 view `v` 中的 vertex `i` 时为 1；`z_vi` 是两者
用于 feature sampling 的 binary conjunction。

若 `F_v(u_vi,v_vi)` 是 bilinear sampled CNN feature，则 masked mean 与 valid-view
ratio 为

$$
\overline F_i=
\frac{\sum_{v=1}^{M}z_{vi}F_v(u_{vi},v_{vi})}
     {\max\left(1,\sum_{v=1}^{M}z_{vi}\right)},
\qquad
\rho_i=\frac{1}{M}\sum_{v=1}^{M}z_{vi}.
$$

变量说明：`F_v(u_vi,v_vi) in R^C` 是 view `v` 的 bilinear sampled feature；
`F_bar_i in R^C` 是 vertex `i` 的 masked mean；`rho_i in [0,1]` 是 valid-view
fraction；`max(1,...)` denominator 在所有 view 不可见时避免除零。

没有有效 view 时，`F_bar_i = 0` 且 `rho_i = 0`。

C2F2 image encoder 为

$$
F_v=\mathrm{Conv}_{3\times3}^{64}\!\left(
\mathrm{ReLU}\!\left(
\mathrm{Conv}_{3\times3,s=1}^{64}\!\left(
\mathrm{ReLU}\!\left(
\mathrm{Conv}_{5\times5,s=1}^{32}(I_v)
\right)\right)\right)\right).
$$

变量说明：`I_v in R^{H x W x 3}` 是 view `v`；每个
`Conv_(k x k,s)^Cout` 表示 kernel size `k`、stride `s`、输出 channels `Cout` 的
convolution；`ReLU` 逐元素执行；所有 active stride 为 1，因此
`F_v in R^{H x W x 64}` 是保持输入分辨率的 C2F2 encoder output。

三个 convolution 均通过 padding 保持 input spatial resolution。
当前 high-frequency construction 为

$$
F_v^{\mathrm{blur}}=G_{5,1.0}(F_v),
\qquad
F_v^{\mathrm{HF}}=F_v-F_v^{\mathrm{blur}},
\qquad
F_v^{\mathrm{out}}=[F_v,F_v^{\mathrm{HF}}].
$$

变量说明：`G_(5,1.0)` 是固定的 depthwise 5x5 Gaussian blur，sigma 为 1.0；
`F_v^blur` 是 low-frequency result；`F_v^HF` 是 residual high-frequency map；
`[.,.]` 表示 channel concatenation，因此 `F_v^out` 有 128 channels。

`G` 是使用 reflect padding 的固定 depthwise Gaussian operation，不增加 learned
parameters。`F_v` 有 64 channels，因此 `F_v^out` 和 aggregated image feature 均为
128 channels。Native-1920 训练以 4-view chunks 加 gradient checkpointing 处理
视图；这些是执行/显存策略，不改变上述 feature 数学定义。因此在前面的 masked
aggregation 中，当前分支实际采样 `F_v^out`，而不是未变换的 `F_v`。

### Feature visibility 与历史 recovery gates

历史 recovery 路径保留的 any-view mask 为：

$$
m_i=\mathbf 1\!\left[\sum_{v=1}^{M}z_{vi}>0\right].
$$

变量说明：`1[.]` 是 indicator function，`z_vi` 是前述 per-view binary visibility，
`m_i in {0,1}` 表示 vertex `i` 是否至少在 `M` 个 views 中的一个可见。

可选 confidence head 预测有界 reliability：

$$
c_i=\mathrm{sigmoid}(g_\theta(x_i))\in[0,1].
$$

变量说明：`x_i` 是下文定义的完整 vertex representation；`g_theta` 是 confidence
side head；`c_i` 是 vertex `i` 的有界 predicted reliability。

其历史 recovery weight 为

$$
w_i=m_i c_i.
$$

变量说明：`w_i in [0,1]` 是 Laplacian row `i` 的 recovery weight，`m_i` 是 hard
any-view visibility gate，`c_i` 是 learned confidence。关闭 confidence head 时，
implementation 等价于令 `c_i=1`。

关闭 confidence head 时使用 `w_i = m_i`。所有 view 均不可见的 vertex，其
learned-Laplacian weight 严格为零。

仓库中的历史 coarse/GT projection 路径实现了以下 Gaussian distance-confidence
gate：

$$
g_i=\mathrm{clip}\!\left(
\exp\!\left[-\left(\frac{d_i^{\mathrm{surface}}}{s}\right)^2\right],
g_{\min},1\right).
$$

变量说明：`d_i^surface` 是 coarse query 到 GT surface 的距离，`s>0` 是 configured
distance scale，`g_min` 是 lower clamp，`g_i` 是 legacy Gaussian confidence
weight。这里的 `g_i` 与上面的 confidence-head function `g_theta` 无关。

其中 `d_i^surface` 是 coarse query 到 GT surface 的距离，`s` 为
`distance_confidence_scale`。该 Gaussian gate 不是 renderer visibility，也不用于
当前 synthetic-current training。当前 recovery-aware 研究只用 `z_vi` 做
image-feature sampling；`L_current` 的每一行都以单位权重进入 sparse solve。
`m_i`、`c_i` 与 `w_i=m_i c_i` 记录冻结历史 baseline，不描述 A-D solver。

### Vertex representation 与 graph network

令 `c_obj` 和 `s_obj` 为 object normalization center 与 scale。Normalized query
为

$$
\widetilde q_i=\frac{q_i-c_{\mathrm{obj}}}{s_{\mathrm{obj}}}.
$$

变量说明：`q_i in R^3` 是 current query，`c_obj in R^3` 是 object normalization
center，`s_obj>0` 是 scalar normalization scale，`q_tilde_i in R^3` 是得到的
dimensionless coordinate。

使用 `K = 6` 个 frequencies 时，dynamic Fourier encoding 为

$$
\phi(\widetilde q_i)=
\left[
\widetilde q_i,
\left\{\sin(2^k\pi\widetilde q_i),
\cos(2^k\pi\widetilde q_i)\right\}_{k=0}^{K-1}
\right].
$$

变量说明：`phi` 是 positional encoder，`k` 遍历 `K=6` 个 octave frequencies；
幂、sine 与 cosine 都逐坐标执行；brackets 将原始 3 个 coordinates 与 `2K` 个
三坐标 sinusoidal blocks 拼接，输出 39 channels。

每个 vertex 的输入为

$$
x_i=\left[
\phi(\widetilde q_i),\ n_i,\
\log\!\left(\max(h_i/s_{\mathrm{obj}},10^{-8})\right),\
\log(1+d_i),\ \rho_i,\ \overline F_i
\right].
$$

变量说明：`x_i` 拼接 39D positional encoding、current unit normal
`n_i in R^3`、relative edge scale `h_i/s_obj`、degree `d_i`、visible-view ratio
`rho_i` 和 aggregated image feature `F_bar_i in R^128`。`max` 防止零值进入
logarithm，每个 scalar term 占一个 channel。

对于当前 HF construction 下的 C2F2，`phi` 为 39 channels，完整 vertex input 为
`39 + 3 + 1 + 1 + 1 + 128 = 173` channels，分别对应 position encoding、normal、
log relative edge scale、log degree、valid-view ratio 和 aggregated image feature。
Graph backbone 为 `173 -> 256 -> 256`，后接 3 个 256-channel message-passing
blocks 和 output MLP `256 -> 256 -> 3`。历史 confidence-enabled 配置额外使用
`173 -> 256 -> 1` sigmoid side head；Arms A-E 关闭该 head，参数量为 826,115。

经过 input MLP 后，第 `l` 个 graph layer 计算

$$
\mu_i^{(l)}=\frac{1}{\max(1,d_i)}
\sum_{j\in N(i)}u_j^{(l)},
$$

变量说明：`l` 是 graph-layer index，`u_j^(l) in R^256` 是 neighbour `j` 的 hidden
state，`mu_i^(l) in R^256` 是其 degree-normalized mean；denominator 同时定义了
isolated vertex 的 zero-safe case。

$$
u_i^{(l+1)}=\mathrm{ReLU}\!\left(
u_i^{(l)}+operatorname{MLP}_l
\left([u_i^{(l)},\mu_i^{(l)}]\right)
\right).
$$

变量说明：`u_i^(l)` 与 `u_i^(l+1)` 是 vertex `i` 的输入/输出 hidden states；
`MLP_l` 是作用于 concatenation `[u_i^(l),mu_i^(l)]` 的 layer-specific learned
update；outer residual addition 保留旧 state；`ReLU` 逐元素执行。

Output MLP 将最终 graph state 直接映射为 `delta_pred_raw in R^3`。Python result
field 名为 `predicted_laplacian`；历史 `delta_hat_prediction` accessor 在
`target_mode = raw_laplacian` 时不表示该 tensor 做过 normalization。

### 训练目标

分量 residual 为

$$
r_{ik}=\delta^{\mathrm{pred,raw}}_{ik}-\delta^*_{ik}.
$$

变量说明：`i` 遍历 vertices，`k in {1,2,3}` 遍历 Cartesian components，`r_ik`
是单个分量在 raw space 中的 signed prediction residual。

逐分量 Huber function 为

$$
H_\tau(r)=
\begin{cases}
\frac{1}{2}r^2, & |r|\leq\tau,\\
\tau\left(|r|-\frac{1}{2}\tau\right), & |r|>\tau,
\end{cases}
\qquad \tau=0.01.
$$

变量说明：`H_tau` 是 scalar Huber penalty，`r` 是一个 signed residual，`tau` 是
quadratic 与 linear branches 之间的 transition magnitude；当前训练值为 raw-
Laplacian 单位下的 `0.01`。

Per-vertex error 与 primary loss 为

$$
e_i=\frac{1}{3}\sum_{k=1}^{3}H_\tau(r_{ik}),
\qquad
\mathcal L_{\mathrm{lap}}=
\frac{\sum_i a_i e_i}{\max(10^{-12},\sum_i a_i)},
$$

变量说明：`e_i` 是三个 component Huber penalties 的均值；`a_i>=0` 是 prepared
validity/target-confidence weight；`L_lap` 是其跨 vertices 的 normalized weighted
mean；`10^-12` 用于保护 empty valid set。

其中 `a_i` 是 prepared target-confidence/valid-scale weight。当前 full-vertex
contract 对有效的非 isolated vertices 使用单位权重，对无效 local scale 使用零
权重。这里没有 curvature weighting，predicted confidence 也不进入 primary loss。

Arm A 只使用该 raw-field objective：

$$
\mathcal L_A=\mathcal L_{\mathrm{lap}}.
$$

变量说明：`L_A` 是完整 Arm-A training objective，`L_lap` 是上面的 raw-Laplacian
Huber loss。Arms B-D 加入下一节定义的 recovery-aware vertex term；它们关闭
confidence head 与 confidence loss。历史 confidence-enabled runs 使用 detached
error-calibration side loss `L_conf`，该实现只保留用于复现，不属于当前 A-E 研究。

### Regularized sparse integration 与 recovery-aware supervision

Arms A-D 从全部 predicted Laplacian equations 恢复 positions：

$$
\widehat X_\lambda=
\arg\min_X
\left\lVert L_{\mathrm{current}}X-\delta^{\mathrm{pred,raw}}\right\rVert_F^2
+\lambda\left\lVert X-X_0\right\rVert_F^2.
$$

变量说明：`X in R^(N x 3)` 是待求 mesh，`X_0 in R^(N x 3)` 是 input mesh，
`L_current in R^(N x N)` 是其固定 row-normalized uniform Laplacian，
`delta^(pred,raw) in R^(N x 3)` 是 network output，`X_hat_lambda` 是唯一 regularized
solution，`lambda>0` 控制对 input positions 的 fidelity。这里没有 row mask 或
learned row weight。

等价求解为

$$
(L_{\mathrm{current}}^\top L_{\mathrm{current}}+\lambda I)
\widehat X_\lambda
=L_{\mathrm{current}}^\top\delta^{\mathrm{pred,raw}}+\lambda X_0.
$$

变量说明：`L_current^T` 是 transpose operator，`I` 是 `N x N` identity。Evaluation
使用 sparse LSMR；recovery-aware training 使用同一系统的 differentiable sparse
PCG solve。

利用 same-index clean vertices，Arms B-D 加入

$$
\mathcal L_{\mathrm{vertex}}=
\frac{1}{N}\sum_{i=1}^{N}
\left\lVert\widehat X_{\lambda,i}-X_i^*\right\rVert_2^2,
\qquad
\mathcal L_{B,C,D}=\mathcal L_{\mathrm{lap}}
+\beta\mathcal L_{\mathrm{vertex}},
\quad \beta=10^{-2}.
$$

变量说明：`X_i*` 是与 current vertex `i` 精确对应的 clean position，`L_vertex` 是
mean squared 3D position error，`beta` 是其 training coefficient。Clean vertices
只用于 loss，不进入 model 或 solve。Arm B 使用 `lambda=10^-2`；C/D 测试
`10^-3` 与 `10^-4`。

Arm E 不走 Laplacian path，而是直接预测 residual：

$$
\Delta X^{\mathrm{pred}}=f_\theta(I,K,E,X_0,F),
\qquad
X^{\mathrm{refined}}=X_0+\Delta X^{\mathrm{pred}},
$$

$$
\Delta X^*=X^*-X_0,
\qquad
\mathcal L_E=\frac{1}{N}\sum_{i=1}^{N}
\left\lVert\Delta X_i^{\mathrm{pred}}-\Delta X_i^*\right\rVert_2^2.
$$

变量说明：`Delta X_pred` 是 Arm E 的 `N x 3` output，`Delta X*` 是 exact clean
displacement，`L_E` 是 direct vertex-space MSE。Arm E 使用相同 encoder/backbone/
head width，但不使用 `L`、sparse solver、lambda 或 post-process。

### End-to-end direct–Laplacian hybrid 与隐式反向传播

当前受控 hybrid 使用一个 shared encoder 与两个 latent geometry heads：

$$
\widehat\delta=h_{\mathrm{lap}}(\Phi_\theta),
\qquad
\Delta V_{\mathrm{direct}}=h_{\mathrm{direct}}(\Phi_\theta),
\qquad
V_{\mathrm{direct}}=V_{\mathrm{input}}+\Delta V_{\mathrm{direct}}.
$$

变量说明：<code>Phi_theta</code> 是 shared C2F2+HF feature field；
<code>delta_hat in R^(N x 3)</code> 与
<code>Delta V_direct in R^(N x 3)</code> 是 latent outputs，二者都没有直接监督。
双 head 模型共有 892,678 个参数，比 Arm B 或 Arm E 多 66,563 个。

使用 validation 选择并固定的 <code>lambda=3e-2</code>，唯一 recovery 为

$$
V_H=
\arg\min_V
\left\lVert LV-\widehat\delta\right\rVert_F^2
+\lambda\left\lVert V-V_{\mathrm{direct}}\right\rVert_F^2.
$$

这里没有额外的 <code>V_input</code> anchor。定义

$$
A=L^\top L+\lambda I,
\qquad
b=L^\top\widehat\delta+\lambda V_{\mathrm{direct}},
$$

differentiable forward solve 为

$$
A V_H=b,
\qquad
V_H=A^{-1}\left(L^\top\widehat\delta+\lambda V_{\mathrm{direct}}\right).
$$

完整 training objective 只有最终 geometry supervision：

$$
\mathcal L_{\mathrm{hybrid}}
=\frac{1}{N}\sum_{i=1}^{N}
\left\lVert V_{H,i}-V_{\mathrm{clean},i}\right\rVert_2^2.
$$

这里不加入 raw-Laplacian、direct-displacement、confidence、spectral、normal 或
Chamfer auxiliary loss。只有在两个 predictions 与 solve 均已建立之后才读取
<code>V_clean</code>；它从不进入 model 或 recovery input。

对于精确 implicit backward，令

$$
G=\nabla_{V_H}\mathcal L_{\mathrm{hybrid}}
=\frac{2}{N}\left(V_H-V_{\mathrm{clean}}\right),
\qquad
A^\top Z=G.
$$

由于 <code>lambda&gt;0</code> 时 <code>A</code> 是 symmetric positive definite，
实际实现求解 <code>AZ=G</code>。两个 branch 的梯度为

$$
\boxed{
\nabla_{\widehat\delta}\mathcal L_{\mathrm{hybrid}}=LZ,
\qquad
\nabla_{V_{\mathrm{direct}}}\mathcal L_{\mathrm{hybrid}}=\lambda Z,
\qquad
\nabla_{\Delta V_{\mathrm{direct}}}\mathcal L_{\mathrm{hybrid}}=\lambda Z
}.
$$

等价 forward Jacobians 为

$$
\frac{\partial V_H}{\partial\widehat\delta}=A^{-1}L^\top,
\qquad
\frac{\partial V_H}{\partial V_{\mathrm{direct}}}=\lambda A^{-1}.
$$

因此 shared parameters 会通过两个 heads 的 Jacobian 同时接收梯度；无需展开 PCG
iterations。该实验固定 <code>L</code> 与 <code>lambda</code>。Forward 与 adjoint
solves 使用 float64 PCG、tolerance <code>1e-8</code>、最多 2,048 iterations。
Preflight 在代表性 meshes 上相对 trusted LSMR 的最大 vertex RMS difference 为
<code>6.10e-9</code>，finite-difference 最大 relative error 为
<code>9.68e-11</code>；两个 branch 的梯度均 finite 且 non-zero。

旧 visibility/confidence-weighted Huber/Adam recovery 仍作为冻结历史 baseline。
Exact-target matched-domain diagnostics 表明 hard visibility 是已测试的最大 recovery-
efficiency 损失，因此不能再把它描述为 active recovery design。

### 报告指标

对于 raw prediction `P = delta_pred_raw` 和 raw target `T = delta*`，主要
prediction metrics 为

$$
\mathrm{EPE}=\frac{1}{N}\sum_i\lVert P_i-T_i\rVert_2,
$$

变量说明：`P,T in R^{N x 3}` 分别是 raw predicted/target Laplacian fields，
`P_i-T_i` 是 vertex `i` 的 vector residual，`N` 是 evaluated vertex count，`EPE`
是 residual Euclidean magnitude 的均值。

$$
\mathrm{Cos}_{\mathrm{global}}=
\frac{\langle\mathrm{vec}(P),\mathrm{vec}(T)\rangle}
{\lVert P\rVert_F\lVert T\rVert_F},
\qquad
R_{\mathrm{norm}}=\frac{\lVert P\rVert_F}{\lVert T\rVert_F}.
$$

变量说明：`vec(.)` 将 field 的全部 components 展平，`<.,.>` 是 Euclidean inner
product，`||.||_F` 是 Frobenius norm，`Cos_global` 是整个 field 的单一 cosine，
`R_norm` 是 predicted-to-target field-magnitude ratio。

Raw RMS 和 maximum residual 为

$$
\mathrm{RMS}_{\mathrm{raw}}=
\sqrt{\frac{1}{N}\sum_i\lVert P_i-T_i\rVert_2^2},
\qquad
\mathrm{Max}_{\mathrm{raw}}=\max_i\lVert P_i-T_i\rVert_2.
$$

变量说明：`RMS_raw` 是 per-vertex vector residual magnitude 的 root mean square，
`Max_raw` 是其最大值；`P`、`T`、`N` 与 `i` 的定义和 EPE 中相同。

Bottom-90%、Top-10% 和 Top-1% group 按全局 `||delta_i*||_2` 定义，而不是按
prediction 定义。Recovery-weighted raw RMS 使用固定 evaluation recovery weights
和同一组 raw residual。

报告中的 bidirectional sampled-surface Chamfer 为

$$
Q_A=\mathrm{SampleSurface}(S_A,n,s),\qquad
Q_B=\mathrm{SampleSurface}(S_B,n,s+1),
\qquad
D_{\mathrm{C}}(A,B)=\frac{1}{2}\left[
\frac{1}{|Q_A|}\sum_{x\in Q_A}d(x,S_B)
+\frac{1}{|Q_B|}\sum_{y\in Q_B}d(y,S_A)
\right],
$$

变量说明：`A` 是 evaluated mesh，`B` 是 GT mesh；`S_A` 与 `S_B` 是各自 triangle
surfaces；`Q_A` 与 `Q_B` 是分别使用 base seeds `s` 和 `s+1` 生成的 `n` 个
deterministic area-weighted surface samples；`d(x,S)` 是 point 到 triangle surface
的最短距离；`|.|` 表示 set cardinality。修正后的 same-initial benchmark 固定
`n=3000`、`s=7`；其他报告会分别序列化自己的 sample count 与 base seed。

对于 same-topology recovered geometry，报告 vertex RMS：

$$
\mathrm{RMS}_{\mathrm{vertex}}=
\sqrt{\frac{1}{N}\sum_{i=1}^{N}
\left\lVert X_i^{\mathrm{refined}}-X_i^*\right\rVert_2^2}.
$$

变量说明：`X_i^refined` 与 `X_i*` 是相同 vertex index 的 recovered/clean positions，
`N` 是 mesh vertex count。该 correspondence metric 补充而不替代 surface Chamfer。

### 历史 `h^2`-normalized formulation

早期 GT-query 实验使用

$$
\delta_i^{\mathrm{GT}}=(L_{\mathrm{GT}}V_{\mathrm{GT}})_i,
\qquad
\widehat\delta_i^{\mathrm{GT}}=
\frac{\delta_i^{\mathrm{GT}}}{h_i^2+\varepsilon},
$$

变量说明：`V_GT in R^{N x 3}` 与 `L_GT` 分别是历史 GT vertices 及其 graph
Laplacian；`delta_i^GT` 是 raw GT Laplacian；`h_i` 是 mean incident-edge length；
`epsilon=10^-12` 防止除零；`delta_hat_i^GT` 是历史 h-squared-normalized target。

并通过 `delta_pred_raw = delta_hat_prediction * (h_current^2 + epsilon)` 将
normalized prediction 转回 raw space。该路径仍保留用于复现，但它不描述当前
Sofa50 direct-raw + HF 训练。历史 normalized representation 的 native loss 不能与
当前 raw-space loss 直接比较。

## 数据契约

HPC 上当前主线使用的 Sofa50 manifests 为：

```text
/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_v1/manifest.json
/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_native1920_v1/manifest.json
```

两者使用相同的 250 个 sample IDs、object-level split 和 28 个 camera poses：50 个
objects、每个 5 个 variants，划分为 200 train、25 validation 和 25 held-out test。
Native-1920 observations 由 renderer 直接以 1920 x 1920 生成，不是将 960 images
resize 得到。两套数据中的 current graph、proxy positions、raw targets 和 renderer
visibility 遵循同一合同。

历史 GT-query 数据仍位于 `sofa50_refinement/multiview_960` 与
`sofa50_refinement/multiview_1920`。它们使用 40/5/5 objects 和
`gt_query_manifest.json`；expanded manifests 仅用于 inference，不是当前训练源。

RGB 图像保存在磁盘上，并以 `uint8` 形式 lazy decode。CUDA 训练使用 pinned
memory、non-blocking transfer，以及用于 CNN/GNN forward 的 AMP。Target scaling、
loss accumulation 和数值几何操作使用 FP32。

## 实验命名

| 标签 | 定义 |
|---|---|
| C0 | Image feature dimension 16；graph hidden dimension 64；3 个 graph layers。 |
| C2 | Image feature dimension 64；graph hidden dimension 256；3 个 graph layers。 |
| F0 | Encoder strides 为 `2, 2`；960 输入对应 240 x 240 feature map。 |
| F1 | Encoder strides 为 `2, 1`；960 输入对应 480 x 480 feature map。 |
| F2 | Encoder strides 为 `1, 1`；feature-map resolution 等于 input resolution。 |
| C2F2 | C2 capacity 与 F2 image encoder 的组合。 |

## 已完成结果

### 960 exact GT-query prediction

| Run | Seed | All EPE ↓ | Top-10% EPE ↓ | Global cosine ↑ | Prediction/GT norm |
|---|---:|---:|---:|---:|---:|
| C0F0 | 7 | 9.4641 | 30.7221 | 0.7808 | 0.8020 |
| C0F1 | 7 | 9.3786 | 30.3095 | 0.7892 | 0.7938 |
| C0F2 | 7 | 9.1665 | 28.4751 | 0.8227 | 0.8180 |
| C2F2 | 7, 17, 27 | 2.8260 ± 0.0864 | 15.3614 ± 0.4036 | 0.8912 ± 0.0127 | 0.9348 ± 0.0160 |

在已完成的分辨率消融中，original RGB 的 error 低于 zero RGB。F0、F1 和 F2
的 original-minus-zero global-cosine gap 分别为 0.2236、0.3315 和 0.3724。

### 960 与 1920 C2F2

| 输入 | 训练预算 | Mean all EPE ↓ | Mean top-10% EPE ↓ | Mean cosine ↑ | Mean expanded Chamfer ↓ |
|---|---:|---:|---:|---:|---:|
| 960 | 50,000 steps，3 seeds | 2.8282 | 15.3743 | 0.8911 | 0.0011624 |
| 1920 | 20,000 steps，3 seeds | 3.0928 | 16.3299 | 0.8954 | 0.0012570 |

两组实验的 optimizer-step budget 不相同。1920 结果未证明相对 960 的改善。

### Expanded-query recovery

五个共享 expanded-validation objects 的 initial Chamfer 为 `0.000652884`。960
C2F2 的平均 refined Chamfer 为 `0.00116202`；1920 C2F2 为 `0.00125704`。每个
已评估 seed 相对 initial mesh 的改善数量均为 `0/5`。

### OpenMVS coarse-mesh recovery

**仅诊断警告。** 这些 OpenMVS mesh 是低质量外部输入，不是期望输出、target
topology、pseudo-GT 或主要方法端点。下表仅保留为 OOD failure/robustness 记录，
不得用于 checkpoint 选择或 architecture 结论。

该测试使用由 48 views 经 COLMAP/OpenMVS 生成的 coarse meshes、原始 14 个
Sofa50 RGB views、三个 960 C2F2 checkpoints、480 分辨率 OpenGL visibility，且
不传递 GT differential。共评估 48 个 meshes；两个 coarse meshes 不存在。

| Recovery | Initial mean Chamfer | Ensemble refined mean Chamfer | 改善 mesh 数 | Introduced flips |
|---|---:|---:|---:|---:|
| 200 iterations | 0.0212023 | 0.0213199 | 2/48 | 4,692 |
| 1,000 iterations | 0.0212023 | 0.0213198 | 2/48 | 4,734 |

Recovery 从 200 增加到 1,000 iterations 未改变聚合结论。

后续 48-case projected-GT 诊断的统一 Chamfer 为：OpenMVS initial `0.0469163`、
projected positions `0.0440446`、projected-position raw Laplacian 经冻结 recovery
后 `0.0456376`、归档 learned prediction `0.0467913`。Recovery 保留 position
projection 潜在增益的 44.53%，learned prediction 最终实现 4.36%。这些数值只
分解低质量 OOD 输入上的 failure，不会把 OpenMVS 变成 target 或模型质量端点。

### 28-view current-graph target 与 loss-space 消融

三个 C2F2 arms 使用相同的 28-view synthetic-current manifest、seed、初始化和
20,000-step budget，并关闭 local query jitter。Native validation loss 位于各自
arm 的 loss space，因此不能跨行直接比较。

| Arm | Output target | Native loss space | Best native val | Test raw EPE ↓ | Test raw cosine ↑ | Refined Chamfer ↓ | 改善数 |
|---|---|---|---:|---:|---:|---:|---:|
| A | `h^2` normalized | Normalized output | 0.0184566 | 0.00769237 | 0.933526 | 0.00456011 | 3/25 |
| B | Raw Laplacian | Raw output | 1.58253e-6 | 0.00300525 | 0.998667 | 0.00380671 | 19/25 |
| C | `h^2` normalized | Raw Laplacian | 2.16552e-6 | 0.00333673 | 0.997419 | 0.00383121 | 16/25 |

三组共享的 initial Chamfer 为 `0.00391323`。Arm B 是当前主要结果：统一
raw-space error 和 recovery Chamfer 最低，并改善 25 个 test samples 中的 19 个。
B/C 极小的 native loss 来源于 raw-Laplacian 数值单位，不表示其相对 A 的
normalized loss 存在四个数量级优势。Contract audit 已通过；最终评估由三个 L40
shards（Slurm array 15686）并行完成，随后由 job 15687 合并。

本地产物包括[完整报告](runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7/analysis/REPORT.md)、
[原始 JSON/CSV 表](runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7/analysis)、
[75 个对比 OBJ](runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7/analysis/mesh_comparisons/B_direct_raw_laplacian)
和[25 张固定相机 GT/COARSE/REFINED 对比图](runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7/analysis/comparison_images/B_direct_raw_laplacian)。

### Stage-2 适配与 Huber 尾部诊断

三个严格配对的 continuation arms 均从同一个 20k Arm-B checkpoint 出发，再训练
20,000 steps。继续使用原 X0、recovered X1 或 X0/X1 各 50% 的三组最终都只改善
`16/25` samples。X1 训练分支的 best checkpoint Chamfer 为 `0.00384032`，高于
冻结 stage-1 的 `0.00380687`；它找回原来 6 个失败样本中的 2 个，却丢失原来
19 个成功样本中的 5 个。因此额外训练和当前 X1 分布适配都没有超过 stage-1。

本地 validation 诊断按 243,000 个 vertex 的 GT raw-Laplacian magnitude 分组。
Top 1% 的平均 raw error 是 bottom 90% 的 `13.071x`，其中 `66.049%` 的 vertex
至少一个 Huber 分量饱和。该组承担 `34.931%` 的 Huber loss，却只贡献 `5.785%`
的 output-gradient L1，gradient retention 为 `58.436%`。这证明极端尾部存在集中
梯度压缩；若要建立其与 Chamfer 的完整因果关系，仍需 surface-sensitivity 对照。

### Raw loss、dynamic expert 与 image-feature 消融

Raw-MSE control 不支持替换 Huber。共享 25-sample test 上，Huber/MSE 的
Top-10% EPE 为 `0.0122438/0.0128078`，Top-1% EPE 为
`0.0371716/0.0380294`，Chamfer 为 `0.00380692/0.00381317`，改善数为
`19/25` 与 `16/25`。MSE 使用 global batch 6，Huber 基线为 2，因此不做严格
单变量训练声称。

从零训练的 learned dynamic residual expert 的 test raw EPE 为
`0.00294740`、Chamfer `0.00377438`、normal consistency `0.944879`，新增
5,699 个 flips，改善 `19/25`。Inference-time causal ablation 在 validation 上选得
`alpha=0.16`。Constant gate 在 25/25 test samples 上优于联合 base；learned gate
又在 Chamfer/P2S 上 `25/25` 优于 constant gate，并在多数 samples 上优于 5 个
mesh 内 gate shuffle。Chamfer attribution diagnostic 中 expert/gate 分别占
`90.32%/9.68%`；这只是诊断，不是严格独立因果分解。

| 960 image feature | Test raw EPE ↓ | Top-10% ↓ | Top-1% ↓ | Chamfer ↓ | Normal ↑ | 改善数 |
|---|---:|---:|---:|---:|---:|---:|
| Original Arm B | 0.00297471 | 0.0122427 | 0.0371654 | 0.00380683 | 0.942470 | 19/25 |
| Gaussian only | 0.00291322 | 0.0123509 | 0.0376811 | **0.00377507** | **0.944459** | **21/25** |
| Original + HF residual | **0.00288627** | **0.0117524** | **0.0347902** | 0.00377832 | 0.942475 | 20/25 |

Gaussian-only 略微恶化 high-curvature tail，却获得最好的 mean downstream geometry。
`F + (F-Gaussian(F))` 获得最好的 prediction/tail metrics，且 mean
Chamfer/P2S 仍优于 original Arm B。

### Strong-smoothing sparse recovery 与 recovery-aware training

使用全部 Laplacian rows 并仅用 component centroid 固定 translation 时，v2
exact-target sparse oracle 的 mean efficiency 为 `0.92366`。旧冻结 recovery 中，
hard visibility 是最大的已测试 efficiency 损失；confidence 基本为常数，增加 Adam
steps 也未关闭差距。因此当前研究采用全方程 regularized sparse recovery。

已完成 A/B test 结果为：

| Arm | Raw EPE ↓ | Raw RMS ↓ | Chamfer ↓ | Eta ↑ | P2S p95 ↓ | Normal ↑ | Vertex RMS ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| A：仅 Lap | **0.00252641** | 0.00737725 | 0.00395529 | 0.07206 | 0.0122582 | 0.954902 | 0.0135181 |
| B：Lap + recovery-aware vertex | 0.00263986 | **0.00683290** | **0.00358497** | **0.13036** | **0.0105581** | **0.959366** | **0.0115532** |

Arm B 在 32/50 上取得更低 paired Chamfer，在 43/50 上取得更低 vertex RMS，但
raw EPE 仅胜 10/50。C（`lambda=10^-3`）与 D（`10^-4`）的 recovered test geometry
均未超过 B。Direct-vertex E 达到 Chamfer `0.00334039`、vertex RMS `0.00822130`、
normal consistency `0.970112`。随后，只读冻结 B+E hybrid 使用 validation-selected
`lambda=3e-2` 达到 Chamfer `0.00302983`，并改善 49/50 test inputs。正在运行的
end-to-end 实验检验：当两个 latent branches 只接收上文最终 hybrid-geometry loss
时，能否学出同样的互补性。

### Native 1920 + high-frequency residual（已完成）

Native-1920 数据使用相同的 250 个 sample IDs、`200/25/25` split、28 个
camera extrinsics、current graph、proxy、raw target 和 visibility tensor。Intrinsics 按
native 1920 渲染缩放，不是 960 resize；native 与 resized 的最小 pixel MAE 为
`0.0205764`。

4×L40 job 15854 已从零完成 20,000 steps。View chunk=4 和 gradient checkpointing
通过 forward/gradient equivalence tests。实际 global batch 为 4，960 HF 基线为 2，
因此这不是严格单变量训练对比。

| 分辨率 + HF | Raw EPE ↓ | Raw RMS ↓ | Bottom 90% ↓ | Top 10% ↓ | Top 1% ↓ | Chamfer ↓ | Normal ↑ | Flips | 改善数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 960 | **0.00288618** | **0.00628203** | 0.00190107 | **0.0117522** | **0.0347895** | **0.00377857** | 0.942504 | 6303 | **20/25** |
| Native 1920 | 0.00290615 | 0.00690893 | **0.00183806** | 0.0125190 | 0.0389263 | 0.00378509 | **0.944522** | **5777** | 18/25 |

Native 1920 没有改善 high-curvature tail 或 mean downstream distance。它改善
normal consistency 并减少 flips，但计算成本为 4 GPUs × 22.35 h（`89.39`
GPU-hours），而 960 为 2 GPUs × 3.98 h（`7.95` GPU-hours）。

### Future2000 GT-adaptive 扩展实验

扩展实验包含 2,000 个不同的 3D-FUTURE source objects。每个对象生成 5 个冻结的
确定性 current-mesh 扰动变体，共 10,000 个 meshes，按对象执行 80/10/10 split
（`8000/1000/1000`）。同一对象的 5 个变体共享相同的 28 个标定 960-pixel RGB
observations，但 current vertices、connectivity、query graph 与 visibility 均按变体
重新确定。

归档 job `16607` 给出 old-structure full-scale 结果（Chamfer `0.00522955`，
959/1000 改善）。它与失败的 jobs `15794`、`15795` 一起保留为基础设施历史，但不再
代表 formal current-architecture 结果。Formal Arm B 保留既定 mixed objective
`L_raw-Laplacian-Huber + 10^-2 L_recovered-vertex`，使用 validation-selected
epoch-195 checkpoint（SHA-256
`fa934cd44c4009dd392c415fe2c5f731c8cf1b78cda6a31fab199d4c15510b82`）。

| Full 1,000-mesh test system | Chamfer ↓ | P2S p95 ↓ | F-score ↑ | Normal ↑ | 改善数 |
|---|---:|---:|---:|---:|---:|
| Initial mesh | 0.00776417127 | — | — | 0.924252350 | — |
| Archived old-structure Ours | 0.00522954770 | — | — | 0.895907 | 959/1000 |
| **Formal mixed-loss Arm B** | **0.00476456546** | **0.0146282911** | **0.881035649** | **0.908597358** | **975/1000** |

Formal Arm B 相对 initial mesh 将 Chamfer 降低 `38.63%`，相对 archived predictor
再降低 `8.89%`。Formal-minus-archived paired difference 为 `-0.000464982242`；
formal 在 882/1000 meshes 与 185/200 object means 上获胜，10,000 次 object
bootstrap CI 为 `[-0.000580558,-0.000314545]`。Normal 仍低于 initial mesh，因此
distance/normal trade-off 被明确保留。

对 external methods，formal Arm B 在 valid pairs 上分别以 804/998、829/999、
974/996 胜 NDS、nvdiffrec、ExMesh。2 个 NDS metrics invalid、1 个 nvdiffrec sample
failed、4 个 ExMesh 结果 invalid 或改变 topology，均显式保留。由于 evaluator 的正反
方向各使用相同数量的 3,000 个 samples，Chamfer 与 bidirectional P2S mean 定义上完全
相同，不能把重复的 P2S mean 当作独立证据；P2S p95 仍是不同统计量。

替代后的 direct-vertex Arm-E job `17888` 已于 2026-09-04 完成全部 200,000 steps。
Validation array `18673` 正在不打开 test 的条件下选择 frozen B+E lambda；jobs
`18677`、`18678`、`18679` 通过 `afterok` 依次锁参、评估 test、生成综合报告。目前
尚不声明 Future2000 Arm-E 或 B+E test metric。冻结合同、paired samples、invalid
output audit 与 provenance 见
[formal Future2000 report](reports/future2000_mixed_vs_old_external_20260831_v2/FINAL_REPORT.md)。

### Sofa50 同初始网格外部对比

Ours、NDS、nvdiffrec 与 ExMesh 在相同的 25 个 native-1920 Sofa50 test inputs 上
运行：完全相同的 current/coarse mesh、28 个 RGB observations 和 cameras。GT 只由
common evaluator 使用。四种方法均完成 `25/25`，input identity audit 通过。

初步聚合错误地混用了各方法原生 Chamfer implementation，使同一个 initial mesh
同时出现 `0.00391323` 和 `0.01707047` 两个分数，该表因此失效。修正报告对 common
initial 与全部 final mesh 使用同一个 deterministic 3,000-surface-point evaluator
（seed 7）重算；native 数值仅保留为 provenance，`contract_audit=true`。详见双语
[事故报告](docs/CHAMFER_EVALUATION_INCIDENT_2026-08-21.zh-CN.md)，以及已跟踪的
[近期 Sofa50 汇总报告](reports/sofa50_multitopology_rawlap500_v2/recent_ablation_and_old_domain_comparison_v1/REPORT.md)。

| 方法 | Unified final Chamfer ↓ | Improvement | 改善数 | Normal ↑ |
|---|---:|---:|---:|---:|
| Ours | 0.011347800 | 33.52% | **25/25** | **0.944514** |
| NDS | **0.011204992** | **34.36%** | 22/25 | 0.873805 |
| nvdiffrec | 0.013654660 | 20.01% | 18/25 | 0.848122 |
| ExMesh | 0.020170615 | -18.16% | 8/25 | 0.845337 |

NDS 的 mean Chamfer 略低；ours 的逐 sample 一致性更好，normal 也明显更高。这些
synthetic-protocol 数值不是官方 DTU ExMesh 毫米制指标。

## 安装与验证

```bash
conda env create -f environment.yml
conda activate test
pip install -e ".[train]"
PYTHONPATH=src pytest -q
```

如果环境已经存在：

```bash
PYTHONPATH=src conda run --no-capture-output -n test pytest -q
```

## HPC 入口

仓库中的 Slurm 文件包含当前集群的路径和资源配置。

```bash
# 960 F0/F1/F2，50,000 steps
bash scripts/slurm_jobs/submit_resolution_50k_parallel.sh

# 960 C2F2，seeds 7/17/27，50,000 steps
sbatch scripts/HPC/sofa50_c2_f2_50k_3gpu.slurm

# 1920 C2F2，seeds 7/17/27，当前输出契约为 20,000 steps
sbatch scripts/HPC/sofa50_c2_f2_1920_50k_3gpu.slurm

# OpenMVS recovery，OpenGL visibility，1,000 recovery iterations
sbatch scripts/HPC/test_sofa50_openmvs_coarse_14view_c2f2_48mesh_opengl_480_recovery1000.slurm

# 14/28/56-view 与 query-resolution 消融，每组 20,000 steps
sbatch scripts/HPC/c2f2_dataset_ablation_20k.slurm view 14
sbatch scripts/HPC/c2f2_dataset_ablation_20k.slurm query gt_sub1

# 三张 L40 分片执行 H2 评估，并提交依赖合并作业
bash scripts/HPC/submit_sofa50_synthetic_current_28view_h2_ablation_3gpu.sh

# Raw MSE 与 Huber 训练/评估
bash scripts/HPC/submit_sofa50_synthetic_current_28view_loss_ablation_3gpu.sh

# Learned dynamic residual expert 与 inference-time gate 消融
bash scripts/HPC/submit_sofa50_dynamic_residual_expert_from_scratch_4gpu.sh

# Gaussian 与 original-plus-high-frequency image-feature arms
bash scripts/HPC/submit_sofa50_image_feature_ablation_2x2gpu.sh

# Native-1920 original-plus-high-frequency 数据、训练和评估链
bash scripts/HPC/submit_sofa50_hf1920_4gpu.sh

# Future2000 200k from-scratch smoke 与 7×Blackwell 训练
sbatch scripts/HPC/smoke_future2000_current_28view_hf_7gpu_blackwell.slurm
sbatch scripts/HPC/train_future2000_current_28view_hf_200k_7gpu_blackwell.slurm
```

### 分布式多 GPU 训练

`train_multi_mesh_laplacian.py` 在通过 `torchrun` 启动时使用 PyTorch
DistributedDataParallel。Training meshes 按 rank 分片，不能整除时执行确定性补齐；
gradient 和 training metrics 在 ranks 间归约；仅 rank 0 写入日志、prediction 和
checkpoint。Checkpoint 使用不含 `module.` 前缀的标准 model keys，可直接用于单卡
evaluation。

Slurm 入口默认申请单节点四张 L40：

```bash
sbatch scripts/HPC/train_multi_mesh_ddp.slurm \
  /path/to/manifest.json \
  /path/to/config.json \
  /path/to/output_dir
```

同一脚本支持多节点。以下命令在两个节点上启动八个 ranks：

```bash
sbatch --nodes=2 --gres=gpu:L40:4 \
  scripts/HPC/train_multi_mesh_ddp.slurm \
  /path/to/manifest.json \
  /path/to/config.json \
  /path/to/output_dir
```

Global mesh batch 为 `world_size * gradient_accumulation_meshes`。
`max_optimizer_steps` 统计同步后的 global optimizer updates；world size 增大后，
每次 update 以及固定 optimizer-step budget 对应的 mesh exposures 数量会增加。

使用 worker 的 lazy-image 训练支持
`data_loading.multiprocessing_sharing_strategy`。历史 Future2000 recovery run 在
descriptor exhaustion 后使用 `file_system` 和 non-persistent workers；已完成的
7×Blackwell run 改用 0 个 DataLoader workers，并把 RGB observations stage 到
node-local storage，从而同时规避 descriptor 与 shared-memory failure。外部方法对比
由[本地任务说明](docs/FUTURE2000_LOCAL_COMPARISON_TASKS.md)
管理，不应重新通过 Slurm 提交。

已生成的 14/28/56-view 数据集位于：

```text
/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/sofa_mesh/sofa50_refinement/multiview_nested_14_28_56_cpu_v3
  gt_query_views_14_manifest.json
  gt_query_views_28_manifest.json
  gt_query_views_56_manifest.json
  expanded_inference_views_14_manifest.json
  expanded_inference_views_28_manifest.json
  expanded_inference_views_56_manifest.json
```

已生成的 query-graph resolution 数据集位于：

```text
/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/sofa_mesh/sofa50_refinement/multiview_960/query_resolution_ablation_v2
  gt_manifest.json
  gt_sub1_manifest.json
  gt_sub2_manifest.json
  gt_adaptive_manifest.json
```

已生成的 28-view + GT-adaptive 组合数据集位于：

```text
/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/sofa_mesh/sofa50_refinement/view_query_combo_28_gt_adaptive_v1
  manifest.json
  summary.json
```

每份 manifest 包含 50 个物体，train/validation/test 为 40/5/5。Prepared
graph 包含 CUDA 生成的 `visibility_backface_and_occlusion`。Manifest 已通过
下游 training 和 inference loader 检查。

### 局部 query-position jitter 消融

该消融使用 C2F2、28 views、每个物体 5 个固定 synthetic-current variants、
seed 7 和 20,000 optimizer steps。Arm A 使用存储的 current vertex positions。
Arm B 仅在训练时加入各分量标准差为 `0.003 h_i`、向量范数上限为
`0.009 h_i` 的 isotropic query jitter。存储的 proxy、normalized target、
`h_i`、connectivity 和 target-construction operator 不变。Validation 和 test
不加入 jitter。

```text
Dataset: /networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_v1/manifest.json
Runs: runs/learned_laplacian/sofa50_synthetic_current_28view_jitter_ablation_seed7
Report: runs/learned_laplacian/sofa50_synthetic_current_28view_jitter_ablation_seed7/analysis/REPORT.md
```

Report 包含 deterministic validation/test prediction metrics、
original-RGB/zero-RGB comparison 和 OpenMVS48 current-mesh recovery metrics。
其中 OpenMVS 部分只作为次要诊断证据；该消融结论由 synthetic-current endpoints
决定。

## HPC 结果目录

```text
runs/learned_laplacian/sofa50_image_resolution_ablation_50000step
runs/learned_laplacian/sofa50_c2_f2_50000step_3seed
runs/learned_laplacian/sofa50_c2_f2_1920_20000step_3seed
runs/learned_laplacian/sofa50_cf_c2f2_comparison_full
runs/learned_laplacian/sofa50_c2f2_960_vs_1920_full
runs/learned_laplacian/sofa50_c2f2_view_query_combo_28_gt_adaptive_20k_seed7_v1
runs/learned_laplacian/sofa50_synthetic_current_28view_jitter_ablation_seed7
runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7
runs/learned_laplacian/sofa50_synthetic_current_28view_b_stage2_adaptation_20k_seed7
runs/learned_laplacian/future2000_gt_adaptive_2000mesh_expanded_current_28view_direct_raw_20k_seed7
runs/learned_laplacian/sofa50_openmvs_coarse_14view_c2f2_48mesh_opengl_480
runs/learned_laplacian/sofa50_openmvs_coarse_14view_c2f2_48mesh_opengl_480_recovery1000
```

Source repository 不包含 checkpoints、prepared datasets 和 HPC result
directories。

### 独立 ExMesh 官方协议 benchmark

ExMesh 官方 DTU 对比是一个完全独立的 external benchmark，不复用上面列出的
synthetic 数据、相机、renderer 或重建 mesh。官方源码版本、15-scene 复现 gate、
common contract 抽取、失败记录规则以及六方法单场景 sanity gate 见
[ExMesh baseline suite 说明](docs/EXMESH_BASELINE_SUITE.md)。发布版 ExMesh 的
15-scene 复现 gate 已通过（复现 mean CD 0.60484 mm，论文 0.58 mm）。官方
six-method DTU benchmark 与上面的 Sofa50 same-initial comparison 相互独立；其完整
执行仍由 scan-24 shared-coordinate-frame audit 控制。Learned-method 预期使用的
DTU scan-24 current mesh 经本地 lineage audit 确认从未生成，且明确禁止把 ExMesh
PGSR mesh 静默替代为 primary input。生成的 provenance report 保留在被忽略的本地
`reports/` 目录中。
