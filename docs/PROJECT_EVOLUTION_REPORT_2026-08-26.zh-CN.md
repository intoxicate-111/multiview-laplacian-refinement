# Multiview Laplacian Refinement 项目转向与演化报告

状态日期：**2026-08-26（Europe/London）**

观察区间：**2026-08-04 至今**

性质：基于 Git 历史、冻结实验报告、当前配置和运行日志的项目级复盘。

## 摘要

这个项目最初的问题可以概括为：**从多视图 RGB 与一个粗网格预测 Laplacian
坐标，再通过几何恢复得到更好的网格**。到目前为止，它已经转向为：

> 在严格的数据、输入和评估契约下，联合研究差分表示与直接位置表示的互补性，
> 并把“恢复后的最终几何质量”而不是中间场预测误差作为主要决策对象。

这不是一次单独的架构升级，而是由一系列负结果推动的四次核心转向：

1. **GT-query → current-query/current-graph**：训练时在 GT 顶点查询得到的好预测，
   不能自动迁移到实际粗网格查询与恢复条件。
2. **normalized H2 → direct raw Laplacian**：直接预测求解器真正接收的 raw field，
   明显优于归一化表示；增加分辨率、视角或训练步数不是主要解法。
3. **预测误差 → 恢复后几何效用**：更低 raw EPE 不保证更低 Chamfer；loss、
   regularization 与线性恢复必须作为一个系统设计。
4. **单一 Laplacian 分支 → Laplacian B + direct-vertex E 双表示**：冻结 specialist
   融合成为 matched-v2 最强已完成方案，但共享联合训练没有复现该优势，域外迁移也
   明显失败。

因此，项目当前最重要的结论不是“某个网络已经普适地优于其他方法”，而是：

- matched-v2 上已经建立了可靠的 B/E 互补收益；
- recovery、representation、domain contract 和 evaluator 都是决定结果的一等变量；
- 预测指标、训练 loss、验证 CD 和域外 test 必须分开解释；
- 当前仍没有证据支持把 matched-v2 的优势直接推广到旧 native-1920 域、任意
  topology 或 Cotangent operator。

## 证据等级与仓库状态

本报告按以下优先级使用证据：

1. `FINAL_REPORT.md`、冻结 checkpoint SHA 和通过的 contract audit；
2. 已完成的 preflight、checkpoint trajectory 和只读诊断；
3. 当前工作树中的配置、代码和 HPC 日志；
4. 运行中的 loss 只用于健康检查，不作为科学结论。

当前 Git HEAD 的最新提交为 `9d8ffae`（2026-08-25，Cotangent pilot L40 resource
修复）。工作树在此基础上还有大量未提交的 S1、continuous B+E、old-domain
native-1920、机制分析和 PCG 稳定性修改。因此，“仓库已提交主线”与“当前实验
工作树”必须区分；本报告不会把运行中或未封板结果写成最终结论。

## 演化时间线

| 阶段 | 核心问题 | 关键证据 | 项目决策 |
|---|---|---|---|
| 08-04：单物体闭环 | 网络能否学习 Laplacian 场并完成恢复？ | Bunny 暴露 1,113 个未引用顶点；`h=0` 使 normalized target 爆炸 | 先做确定性 topology cleanup、isolated-vertex 审计和一致 evaluator |
| 08-06–08-07：canonical GT-query/H2 | RGB、visibility 与 confidence 是否足以改善恢复？ | canonical Sofa50 能学到 field，但 main+confidence CD `0.00299063`，远差于 initial `0.000652884`，5/5 全部变差 | visibility 是必要防护但不是根因；开始质疑 query contract 与 recovery |
| 08-07–08-10：规模因素搜索 | 更大模型、更高分辨率、更多视角能否解决问题？ | C2 容量较好；28 views 是折中；1920 未优于 960，且成本显著更高 | 固定 28×960 为主线，不再默认“更大输入=更好几何” |
| 08-11–08-14：current-query 与 raw 表示 | 训练目标是否与部署求解器一致？ | direct-raw B：raw EPE `0.00300525`、CD `0.00380671`、19/25；优于 normalized arms | 主线从 GT-query normalized H2 转向 current-query/current-graph direct raw |
| 08-12–08-15：尾部、递归与 loss 专家 | 是否可靠更长训练、top-k、递归或新 loss 修补？ | 20k→50k validation loss 降但 CD 变差；递归从 19/25 降到 12、7、2/25；Top-1% 不能解释全部 CD gap | 放弃把局部 tail 或训练 loss 当作唯一瓶颈，转向 recovery-aware objective |
| 08-14–08-23：多拓扑、Future2000 与外部方法 | 改善能否跨 topology、规模和方法成立？ | v1 multi-topology prediction 改善但 recovery/legacy preservation 失败，判定 NO-GO；Future2000 full-1000 则 959/1000 改善 | 建立 scale/OOD gate；“预测更准”不再等同于“可扩大训练” |
| 08-21：评估事故修复 | 跨方法 CD 是否真的可比？ | 同一个 initial mesh 在两条路径得到 `0.003913228` 与 `0.017070468` | 所有跨方法 primary metric 必须对归档 mesh 用同一个 evaluator 重算 |
| 08-23–08-24：strong smoothing 与 recovery-aware | field 已很好时，为什么 final geometry 仍不好？ | v2 raw EPE 从 `0.00840367` 降至 `0.00276820`，CD 却从 `0.00426879` 恶化到 `0.00451747` | recovery 被提升为核心研究对象；引入全方程 regularized sparse solve 和 geometry loss |
| 08-24–08-25：B/E 双表示与融合 | 差分和直接位置表示是否互补？ | B test CD `0.00358497`；E `0.00334039`；冻结 B+E `0.00302983`、49/50 | 冻结 specialist fusion 成为 matched-v2 当前最强已完成路线 |
| 08-25 至今：联合训练、拆分架构与域匹配 | specialist 优势来自哪里，能否联合学习并迁移旧域？ | shared joint test CD `0.00341857`；机制分类 MECH5；S1 与 old-domain 仍未完成正式封板 | 不把“分开训练更好”泛化为定律；当前转向受控架构诊断和 native-1920 域内重训 |

完整的 08-04–08-14 原始实验链见
[近期提交与实验报告](RECENT_COMMIT_AND_EXPERIMENT_REPORT_2026-08-04_2026-08-14.zh-CN.md)。

## 一、问题定义发生了什么变化

### 1. 从“能否拟合 Laplacian”转向“部署条件下能否改善网格”

早期 canonical 管线使用 clean/GT geometry 上的查询位置或尺度来定义监督。这对分析
网络是否能学习 RGB-conditioned differential field 很有用，但训练与部署之间存在
结构性差异：部署时输入是受扰动的 current mesh，查询位置、graph、局部尺度和
recovery RHS 全部发生变化。

诊断显示 query transfer gap 约为 `0.0184h–0.0270h`，而训练 perturbation 不超过
`0.001h`。这解释了为什么 field loss 可以看起来正常，最终恢复仍会显著变差。
后续主线因此统一为：

- 在 current vertices 查询 image features；
- 在 current connectivity 上构造 operator；
- target、prediction 和 recovery 使用同一个 raw coordinate contract；
- clean mesh 只在 loss/evaluation side 出现，不进入 model input 或 solver。

### 2. 从 normalized H2 转向 direct raw solver input

受控三臂实验中：

| 表示 | Test raw EPE | Refined CD | Improved/test |
|---|---:|---:|---:|
| normalized target / normalized output | `0.00769237` | `0.00456011` | 3/25 |
| **raw target / raw output** | **`0.00300525`** | **`0.00380671`** | **19/25** |
| normalized output / raw-space loss | `0.00333673` | `0.00383121` | 16/25 |

direct raw 的关键优势是目标与 recovery 输入同义，而不只是 loss 数值较小。不同
表示下的 loss 单位也不同，所以 raw loss 与 normalized loss 不能按数量级直接比较。

### 3. 从“单个固定 topology”转向显式 topology/domain contract

multi-topology v1 证明网络可以把 raw prediction 指标改善到新 topology，但 new-test
recovery 未改善，legacy accuracy 也未保持在 5% 内，最终判定 FUTURE-20K **NO-GO**。
这促使数据 contract 从“sample 数量和 split”扩展为：

- object/variant identity；
- input mesh 顶点、faces、ordering 与 connectivity；
- render resolution、view count、camera mapping；
- smoothing/perturbation recipe；
- train/validation/test 与 OOD domain 的明确分离。

相关证据见
[multi-topology v1 最终报告](../reports/sofa50_multitopology_rawlap500_v1/final_unified_v2/FINAL_REPORT.md)。

## 二、模型结构与 loss 的转向

### 1. 早期主线：单个 Laplacian predictor

早期网络由多视图 image encoder、projected image field、per-vertex positional/query
features 和 graph/message-passing backbone 组成，输出每顶点 3D Laplacian field。
实验先后测试了容量 C0/C1/C2、14/28/56 views、960/1920 输入、HF features、局部
jitter、loss expert 和 residual weighting。

这些实验的共同结果是：容量和输入条件会影响 field prediction，但无法单独消除
prediction-to-recovery gap。特别是：

- 28 views 的 validation loss 优于 14/56，而 56 views 运行时间约为 14 views 的
  `2.085×`；
- native-1920+HF 未在 CD 上超过 960，却消耗约 `89.39` 对 `7.95` GPU-hours；
- 20k→50k continuation 的 validation loss 降约 8.26%，CD 反而恶化约 1.07%；
- recursive refinement 会累积偏差和 flips，而不是自动收敛。

### 2. Arm B：保留差分表示，但把恢复后几何写入训练目标

Arm B 预测 `delta_hat`，再求解

```text
(L^T L + lambda I) V_hat = L^T delta_hat + lambda V_input,
lambda = 1e-2.
```

训练目标为 Laplacian prediction loss 加 recovered-vertex loss：

```text
L_B = L_lap + beta * mean_i ||V_hat_i - V_clean_i||_2^2,
beta = 1e-2.
```

Arm A 的 raw EPE 更低（`0.00252641` 对 `0.00263986`），但 Arm B 的 test CD 更好
（`0.00358497` 对 `0.00395529`）。这成为项目最关键的方法学证据之一：**训练应优化
field 的几何效用，而不是只优化 field 的逐点均值误差。**

详见 [recovery-aware 研究记录](SOFA50_RECOVERY_AWARE_STUDY.zh-CN.md) 与
[A–E 最终对照](../reports/sofa50_multitopology_rawlap500_v2/direct_vertex_arm_e_extension/final/FINAL_REPORT.md)。

### 3. Arm E：去掉 Laplacian 与 solver 的直接位置基线

Arm E 使用相同 predictor family 和相同参数量（826,115），但输出
`Delta V_pred`，并直接计算

```text
V_direct = V_input + Delta V_pred.
```

它不使用 Laplacian target、PCG/LSMR、visibility/confidence gate 或任何 post-process。
其 matched-v2 test CD 为 `0.00334039`，优于 Arm B 的 `0.00358497`；同时 VRMS 和
normal consistency 也更好。这说明 direct positional representation 是一个强基线，
而不是只用于给 Laplacian nullspace 定位的辅助项。

### 4. Frozen B+E：目前最成功的 matched-v2 结构

冻结融合求解：

```text
(L_U^T L_U + 0.03 I) V_H
    = L_U^T delta_B + 0.03 V_direct.
```

其中 B 提供 differential constraint，E 提供 direct positional anchor。validation
只选择 `lambda=3e-2`，test 不参与选择。

| 方法 | Validation CD | Test CD | Test improved/worsened |
|---|---:|---:|---:|
| Arm B | `0.00320962` | `0.00358497` | 36/14 |
| Arm E | `0.00285065` | `0.00334039` | 45/5 |
| **Frozen B+E** | **`0.00244917`** | **`0.00302983`** | **49/1** |

它同时优于两个 standalone branch，但不是所有指标都优于 E：例如 Hybrid 的 VRMS
和 normal consistency 仍可能介于 B/E 之间。频谱诊断表明，它主要继承 E 的 component
translation modes，同时把 mid/high-frequency error 略降到两个分支以下。

详见 [Frozen B+E 最终报告](../reports/sofa50_multitopology_rawlap500_v2/frozen_hybrid_recovery_v1/FINAL_REPORT.md)。

### 5. Shared joint 与 S1：从“融合有效”追问“如何共同学习”

from-scratch shared joint 用共享 encoder、image field 和 graph backbone，分出 Laplacian
与 direct output heads，只用最终 `V_H` 的 geometry loss。它的 matched-v2 test CD
为 `0.00341857`，明显不及 Frozen B+E 的 `0.00302983`；legacy/unseen OOD 也分别恶化
到 `0.00726879` 和 `0.00710387`。

机制分析没有找到足以单独解释差距的强梯度冲突或强 complementarity gap，最终分类
为 **MECH5**：只有 specialist gap 明确成立，不能据此宣称“分开训练普遍优于联合
训练”。详见 [机制分析](../reports/sofa50_multitopology_rawlap500_v2/frozen_vs_joint_mechanism_analysis_v1/FINAL_REPORT.md)。

S1 随后把 fork 提前到 graph input MLP 之前：只共享 visual/pre-graph frontend，两个
branch 各自拥有独立 input MLP、3 个 graph blocks 和 output MLP。参数量从 S0 的
892,678 增至 1,594,374。它检验的是“共享 geometry tower 是否限制 specialist
形成”，而不是修改 loss 或 recovery。S1 的 preflight、implicit-gradient 和
PCG↔LSMR gate 均通过；已有训练日志给出 best validation Hybrid CD 约
`0.00312273`，但正式 test/report 尚未封板，因此不能宣称 S1 成功或失败。

详见 [S1 preflight](../reports/sofa50_multitopology_rawlap500_v2/s1_split_geometry_hybrid/PREFLIGHT_REPORT.md)。

## 三、恢复层从后处理变成了模型的一部分

### 1. 旧恢复路线为什么被放弃

早期恢复依赖 visibility/confidence gating、input anchoring 和 Adam vertex
optimization。后续精确审计发现：

- exact target 的全方程 sparse solve oracle efficiency 在 v2 可达 `0.92366`；
- `lambda=0.01` 后再加 hard visibility 会把 efficiency 从 `0.34258` 降到
  `0.16875`，44/50 变差；
- confidence 的影响接近数值噪声；
- Adam 从 200 增至 2,000 steps 仍未解决主要差距。

因此主线改为全方程 regularized least squares，不再默认删除“不可信”方程。

### 2. 当前稀疏求解方法

训练中的恢复采用 matrix-free **Jacobi-preconditioned conjugate gradient（PCG）**，
求解 SPD normal equations；XYZ 被视为一个 flattened block-diagonal system。反向
传播使用 custom implicit autograd，而不是展开每一步迭代。standalone audit 使用
float64 LSMR/等价 least-squares 作为参考。

当前实现还增加了一个重要数值修复：float32 CG 的递归 residual 触达 tolerance 后，
重新计算真实 `b-Ax`；若仍未收敛，则从真实 residual 重启 Krylov recurrence。这是
针对旧域 Arm-B 先前出现的“递归 residual 假收敛”故障，不改变目标方程。

实现见
[differentiable_sparse_recovery.py](../src/mlr/learned_laplacian/differentiable_sparse_recovery.py)。

### 3. regularization 不是普通超参数

`lambda` 同时决定低频/nullspace anchoring、condition number、branch 混合比例和最终
几何偏差。v2 regularized sweep 中，predicted raw 的 `lambda=1e-2` 是诊断最优，
但只保留 exact-target oracle 效率的一小部分；B/E frozen fusion 则在 validation 选择
`3e-2`。更弱的 B-style anchoring并未带来更自由的恢复：C (`1e-3`) test CD
`0.00414926`，D (`1e-4`) 为 `0.00653139`，均说明系统会变得不稳定或放大 field error。

### 4. Cotangent 是数值方法失败，不是性能输赢

受控 Uniform/Cotangent ablation 保持 loss、solver 和模型不变，只替换 operator。
Cotangent stiffness 的代表性 operator norm 中位数约 `1428.351`，Uniform 为
`1.578`；在 `lambda=3e-2` 时，估计 condition number 中位数约 `6.801e7` 对
`84.02`。所有预声明 lambda 都在第一个 optimizer step 前未通过 float64 PCG gate。

因此分类为 **COT5**：当前 mesh/operator/solver contract 下数值不适用。没有有效的
Cotangent CD 曲线，不能把它表述成“Cotangent 几何效果较差”。若继续，需要单独声明
scale-aware operator、mass normalization 或更强 preconditioner 的数值方法实验。

详见 [Uniform vs Cotangent 报告](../reports/sofa50_multitopology_rawlap500_v2/uniform_vs_cotangent_single_loss/FINAL_REPORT.md)。

## 四、评估方法的转向

### 1. 从 native metric 汇总转向统一重评估

2026-08-21 的 Chamfer 事故是项目治理上的分水岭。原横向表直接聚合不同方法的 native
metric；同一个 common initial mesh 因 evaluator 不同，竟得到 `0.003913228` 和
`0.017070468` 两个分数。修复后，所有 output mesh 统一调用：

```text
mlr.learned_laplacian.evaluation.evaluate_mesh_geometry
area-weighted surface sampling; samples=3000; seed=7;
bidirectional sampled-surface-to-exact-triangle-surface;
fscore_threshold=0.01; no ICP.
```

native metrics 只保留作 provenance，不能参与 primary ranking。修正后的同初始 25
sample 对照为：Ours CD `0.01134780`、25/25 改善、normal `0.944514`；NDS 的 mean
CD 略低至 `0.01120499`，但只改善 22/25，normal 为 `0.873805`。

详见 [Chamfer 评估事故报告](CHAMFER_EVALUATION_INCIDENT_2026-08-21.zh-CN.md)。

### 2. 从单一 CD 转向多维几何质量

当前标准报告同时包含：

- surface Chamfer、P2S p95、F-score；
- normal consistency、introduced flips、new degenerate faces；
- same-index VRMS、component translation 与 centered deformation；
- graph-frequency low/mid/high error energy；
- cotangent twice-mean-curvature、dihedral、face-normal、edge/area log error。

这使项目能够识别“CD 变好但 normal/curvature/topology 变差”的真实 trade-off。比如
Future2000 full-1000 中，Ours 将 CD 从 initial `0.00776417` 降至 `0.00522955`，
959/1000 改善，但 normal 从 `0.924252` 降至 `0.895907`；这不是可以被 CD 掩盖的细节。

### 3. 从 test 看最好 checkpoint 转向 validation-only selection

continuous pretrained B+E 的只读 test trajectory 显示 test CD 从 step 0 的
`0.00302694` 降至 step 7500 的最低观测值 `0.00284108`，到 step 15000 回升至
`0.00286524`。但正式流程没有按这条 test 曲线选 checkpoint：matched validation
选择了 epoch 188（step 9,400），validation CD 从 `0.00244956` 降至
`0.00223262`，随后一次性得到 matched-test CD `0.00288357`，相对 geometry-equivalent
step 0 改善 `0.00014334`。该改变量超过重复执行噪声 envelope，最终分类为 **CT2**；
legacy/unseen OOD CD 仍为 `0.00717772` / `0.00678227`，没有解决域外失败。

这说明 continuation 有真实 matched-domain 收益，也有过训练与 validation/test 最优点
不一致的风险；**step 7500 不能因为 test 最低而被直接选中**。

详见 [continuous checkpoint trajectory](../reports/sofa50_multitopology_rawlap500_v2/continuous_b_e_hybrid/checkpoint_diagnostics/TEST_TRAJECTORY_REPORT.md)
与 [validation-selected 最终报告](../reports/sofa50_multitopology_rawlap500_v2/continuous_b_e_hybrid/final_evaluation/FINAL_REPORT.md)。

## 五、规模化与工程路线的变化

项目从单 GPU、单 mesh 的研究原型逐步发展为可审计的多 GPU pipeline：

- lazy image-path dataset，避免把所有高分辨率图像常驻内存；
- 960/1920 memory-safe profiles、chunked projection 与 gradient checkpointing；
- DDP、CUDA prefetch、node-local staging 和明确的 effective global batch；
- 针对 file descriptor、`/dev/shm`、DataLoader worker 与共享文件系统故障的降级策略；
- 每个 experiment 的 preflight、checkpoint SHA、dependency gate、resume 和 fail-closed
  report；
- FP16 CUDA nondeterminism 不再要求不现实的 latent bitwise equality，而用 checkpoint
  identity、重复噪声 envelope 和 recovered-geometry tolerance 审计。

Future2000 full-1000 证明 pipeline 可以在大规模多拓扑数据上工作：Ours 完成
1000/1000，aggregate CD `0.00522955`，并在有效 paired samples 上分别击败 NDS
742/998、NDS-28V-full 632/999、nvdiffrec 799/999、ExMesh 955/996。严格 contract
仍记录外部方法的 invalid/missing outputs，而不静默丢弃。

详见 [Future2000 full-1000 报告](../reports/future2000_same_initial_full1000_20260822/full_report/FINAL_REPORT.md)。

近期 HPC 故障进一步带来两项工程修正：

- Arm-B 的 float32 PCG 增加 true-residual recomputation/restart；
- Arm-E 从不兼容 L40 driver 的 Blackwell CUDA 环境切换到 PyTorch 2.5.1+cu124 环境。

这些属于执行可靠性修复，不应被解释为模型方法变化。

## 六、域外失败带来的最新转向

把 v2 Frozen B+E 零样本应用到旧 native-1920 `v00–v04` 25 meshes 时，结果从
initial CD `0.01707047` 恶化到 `0.03438491`，0/25 改善，并输给所有同输入
comparator。25 个 float64 PCG solve 全部收敛，且没有新退化面，所以问题不是线性
求解器失败，而是明确的 domain shift：render resolution、perturbation/topology
distribution 与训练域不同。

详见 [旧域 zero-shot 报告](../reports/sofa50_multitopology_rawlap500_v2/frozen_b_e_same_initial_v00_v04/REPORT.md)。

这直接推动了当前路线：在旧域 native-1920 contract 上分别训练新的 Arm B 和 Arm E，
先做 validation-only specialist/lambda 选择，再做一次 sealed test。当前替代训练作业
已能正常运行，但尚无最终 validation/frozen-fusion/test 报告，因此没有资格与 NDS、
nvdiffrec、ExMesh 下结论。

## 七、哪些假设已被否定，哪些仍然成立

| 假设 | 当前判定 | 依据 |
|---|---|---|
| 更高分辨率自然提高最终几何 | 不支持 | 1920 未稳定优于 960，成本大幅增加 |
| 更多 views 一定更好 | 不支持 | 28 views validation 优于 56；56 主要改善部分 tail |
| 更长训练自然降低 CD | 不支持 | 20k→50k、continuous 后段均出现 loss/CD 脱钩 |
| raw EPE 越低，最终 CD 越低 | 明确不成立 | Arm A/B、strong-smoothing v1/v2 均给出反例 |
| visibility/confidence 是主要 recovery 解法 | 不支持 | hard visibility 可显著降低 oracle efficiency；confidence 接近零影响 |
| recursive refinement 会逐步收敛 | 不支持 | R1/R2/R3 的 improved count 持续下降并积累 flips |
| direct vertex 只是弱 baseline | 不成立 | Arm E 在 matched-v2 上强于 standalone B |
| B/E 表示在 matched-v2 有互补性 | 支持 | Frozen fusion validation/test 均优于 B 和 E |
| 共享联合训练能够自动学习同样的互补性 | 当前不支持 | shared joint 明显落后于 frozen specialists |
| 差距主要由强梯度冲突造成 | 不支持强结论 | MECH5：all-shared gradient gate 未通过 |
| Cotangent 只需替换 Uniform 即可公平训练 | 当前不成立 | condition number 与 PCG gate 在 step 0 前失败 |
| matched-v2 优势可零样本迁移旧 native-1920 域 | 明确不成立 | Frozen B+E 旧域 0/25，aggregate CD 翻倍 |

## 八、当前项目状态

### 已完成且可作为结论

- current-query direct-raw 表示优于 canonical normalized H2；
- recovery-aware Arm B、direct Arm E 与 frozen B+E 的 matched-v2 对照；
- frozen vs shared-joint 机制诊断（MECH5）；
- unified Chamfer 修复后的 Sofa50/Future2000 外部对照；
- Uniform/Cotangent 数值 gate（COT5）；
- continuous B+E validation-selected continuation 最终评估（CT2）及 checkpoint
  trajectory；
- old-domain frozen zero-shot 失败归因。

### 已完成训练但仍缺正式科学封板

- S1 split-geometry 20k：训练与 validation selection 已完成，正式 selected-checkpoint
  test、paired statistics 与最终 ARCH 分类仍待生成。

### 正在进行

- old-domain native-1920 Arm-B replacement：使用 current-query raw Laplacian、
  `lambda=1e-2`、recovery-aware geometry loss 与修复后的 float32 PCG；
- old-domain native-1920 Arm-E replacement：直接 displacement MSE，不使用 sparse
  recovery；
- 后续 frozen fusion、lambda selection 和 sealed test 必须等待两个 specialist 完成。

运行中的 loss 下降只能说明训练数值健康，不能说明最终 Chamfer、curvature 或外部
方法排名。按照当前运行纪律，除非明确要求，不应在 HPC 上并行挂大量评估任务。

## 九、对当前方向的判断

项目已经从“learned Laplacian predictor”演化为一个更准确但也更克制的研究问题：

> 如何在 domain-matched input contract 下，让 differential constraint 与 direct
> positional prior 形成可训练、可恢复、可跨 topology 验证的几何系统？

目前最可靠的答案是 **独立训练 B/E + validation-selected frozen fusion**，但它只是
matched-v2 内成立的强基线，不是最终通用架构。shared joint、S1、continuous
fine-tuning 和 old-domain retraining 分别在测试四种解释：共享参数干扰、geometry
tower 容量、初始化/优化路径、domain mismatch。它们不应混成一次无边界的架构搜索。

下一阶段最有信息量的封板顺序应为：

1. 完成 old-domain B/E 的 validation-only selection，冻结 checkpoint 与 SHA；
2. 在 validation 上选择 frozen-fusion lambda，随后只运行一次 sealed 25-sample test；
3. 用统一 evaluator 与 exact common input 对照 NDS、nvdiffrec、ExMesh 和 Previous
   Ours，同时报告 curvature/normal/topology；
4. 完成 S1 selected checkpoint 的 matched test，再与 frozen、shared S0 做相同 paired
   comparison；
5. 只有在上述结果表明 geometry tower 或 domain matching 是稳定增益来源后，才决定
   是否继续 continuous joint 或更大规模训练；
6. Cotangent 若重启，应作为独立数值线性代数研究，而不是普通 operator ablation。

## 结论

项目到现在最重要的变化，是研究判断标准已经从“网络是否能把某个 target loss 降低”
转向“整个输入—预测—恢复—评估链是否在冻结契约下产生更好的几何”。这使许多早期
看似正面的结果被正确降级：更高分辨率、更多训练、较低 raw EPE、不同 native CD，
都不再足以证明方法进步。

与此同时，项目也得到了一条更可信的主线：current-query direct-raw、全方程
regularized recovery、recovery-aware supervision、direct-vertex specialist 和冻结
B/E 融合。下一步的关键不在于继续无条件扩大模型，而在于完成 S1 与 old-domain 的
严格封板，确认 specialist 形成和 domain matching 中哪一项真正可复现、可迁移。
