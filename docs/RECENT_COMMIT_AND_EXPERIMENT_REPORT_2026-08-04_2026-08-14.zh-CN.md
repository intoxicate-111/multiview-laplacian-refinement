# 近期 Commit、Sofa50 与 Future2000 实验对比报告

报告更新时间：2026-08-15 18:47 BST（含 5.4 节 addendum）

实验统计窗口：2026-08-04 00:00 BST 至 2026-08-14 09:49 BST

已提交统计范围：`0081f80` 至父提交 `bd0f2aa`

统计基准 HEAD：`bd0f2aaf909af0cea288d551b2857fbe7a9f877a`

说明：Commit 统计仍固定在上述 8 月 4–14 日窗口，不包含承载本报告的发布
commit。5.4 节单独记录 8 月 14–15 日后续实验，不回写历史 commit 统计。

## 1. 摘要

这一阶段从建立独立 learned-Laplacian package、单物体 Bunny sanity check 和
multi-mesh pipeline 开始，经过 Sofa50 canonical H2、renderer visibility、模型容量、
图像/query 分辨率与 OpenMVS recovery，逐步收敛到 synthetic-current 的主要误差
来源。最重要的已完成结果是：

- 统计窗口内共有 **91 个 commits**，涉及 **257 个独立文件**；Git numstat
  记录约 **625,899 行新增、562,754 行删除**。该口径包含 tracked research assets
  和大规模工作区变更，不应解释为纯手写源代码行数。其中 `bde55ef` 添加约
  567,485 行，随后 `95c4061` 为移出 research assets/generated runs 删除约
  559,287 行；两条提交解释了绝大多数 gross churn。
- 8 月 4 日的 Bunny 实验暴露了 1,113 个未被 face 引用的孤立顶点；未清理图上的
  edge-scale normalized runs 爆炸，而 raw-target baselines 稳定。确定性清理后建立
  了 normalized pipeline 的数学/数值有效性，但不构成跨对象泛化结论。
- 早期 canonical Sofa50 H2 模型在 2,000 epochs 时学到了 RGB 与 confidence 信号，
  但 expanded recovery 仍为 `0/5`；renderer hard visibility mask 大幅减小错误，却
  仍未超过 initial geometry。
- 28-view current-graph 三组受控实验中，**B direct raw-Laplacian** 是当前最佳
  已完成方案：raw EPE `0.00300525`、refined Chamfer `0.00380671`，并使
  `19/25` 个 test samples 优于原始 mesh。
- B 相对共享 initial Chamfer `0.00391323` 降低 **2.72%**；相对 canonical H2
  Arm A 降低 **16.52%**，相对 Arm C 降低 **0.64%**。
- 单纯增加训练步数并不保证 geometry 改善：旧 current-query 从 20k 延长到
  50k 后 prediction loss/EPE 下降，但 Chamfer 增加 `1.07%`，成功数从
  `5/25` 降为 `3/25`。
- 精确 current-graph target oracle 能使 `25/25` samples 改善，mean Chamfer
  相对 initial 降低 `18.87%`，说明当前 proxy/recovery 合同存在有效上限，主要
  缺口仍与 learned Laplacian prediction error 及其 recovery 交互有关。
- 冻结 B 的直接三轮递归 inference 不成立：OpenGL visibility 策略下改善数从
  `19/25` 依次降至 `12/25`、`7/25`、`2/25`。
- Stage-2 distribution-adaptation 三分支已完成。最好 continuation 是 X1 分支，
  但只改善 `16/25`，Chamfer `0.00384032`，没有超过冻结 B 的 `19/25` 与
  `0.00380687`。
- Arm-B Huber 诊断显示梯度压缩集中于 GT raw-Laplacian top 1%：`66.049%`
  vertices 至少一个分量饱和，gradient retention `58.436%`。
- Future2000 已形成 2,000×5、28-view、GT-adaptive、C2F2 current-graph 训练合同。
  主训练首次在 32k 因文件描述符耗尽失败，恢复后到 64k 又因 `/dev/shm` 耗尽失败；
  checkpoint 可恢复，当前不报告最终 geometry 结论。

## 2. Commit 统计

### 2.1 按日期

| 日期 | Commit 数 | 主要工作 |
|---|---:|---|
| 2026-08-04 | 17 | learned-Laplacian package、单物体/Bunny、归一化与拓扑清理 |
| 2026-08-05 | 6 | multi-mesh、可视化、lazy image/OpenGL 数据合同 |
| 2026-08-06 | 7 | lazy training 优化、Sofa50 GT-query、诊断、renderer visibility |
| 2026-08-07 | 9 | canonical H2、residual/identity/resolution 诊断、HPC 环境 |
| 2026-08-08 | 14 | 模型容量、C1F2、C2F2 50k、HPC 脚本与可视化 |
| 2026-08-09 | 14 | 1920 C2F2 分析、960/OpenMVS inference 与 recovery |
| 2026-08-10 | 5 | 14/28/56 views 消融、双语 README、数学定义 |
| 2026-08-11 | 2 | Sofa50 query/current-mesh 消融、current-query 50k continuation |
| 2026-08-12 | 17 | 50k downstream、oracle/top-k、jitter、H2/raw loss 与递归评估 |
| 2026-08-13 | 0 | 父提交后工作尚未发布；stage-2、Future2000 与外部基线实现 |
| 2026-08-14 | 0 | 父提交后工作尚未发布；训练恢复与文档汇总 |
| **总计** | **91** | — |

### 2.2 按 Git author identity

| Author identity | Commits | 占比 |
|---|---:|---:|
| `intoxicate-111 <intoxicate-111@users.noreply.github.com>` | 47 | 51.6% |
| `Chuanlin Zhou <zhou_c@wmgubws30.wmgds.warwick.ac.uk>` | 44 | 48.4% |

这里按 Git author 字段统计，不推断两个 identity 是否属于同一自然人。

### 2.3 关键里程碑 commits

| Commit | 日期 | 内容 | 产出/意义 |
|---|---|---|---|
| `0081f80`–`5783e5b` | 08-04 | learned package、sample、trainer、evaluation、tests | 建立隔离实现和单物体闭环 |
| `5b1645e`–`ac0d023` | 08-04 | Stanford Bunny preparation/ablation | 将 sanity check 扩展到真实 Bunny |
| `cfe4d42`–`a79062b` | 08-04 | edge-scale normalization、孤立点清理、可视化 | 定位 1,113 个孤立顶点并修复实验合同 |
| `bde55ef` / `c147238` | 08-05 | multi-mesh trainer 与 prediction visualizer | 从单物体扩展到多 mesh |
| `95c4061`–`a778146` | 08-05 | lazy sample、OpenGL render、1920 profile | 建立内存安全数据路径 |
| `ecf328b`–`ceb5012` | 08-06 | lazy trainer、GT-query Sofa50、diagnostics | 建立 Sofa50 训练/诊断管线 |
| `6f9e3d6`–`dd150ed` | 08-06 | 文档、双语 README、renderer visibility/recovery | 定位 visibility 必要但不充分 |
| `c656684` | 08-07 | canonical Sofa50 H2 pipeline | 固化 target、confidence 和 recovery 合同 |
| `32e4738` | 08-07 | residual/identity/transfer diagnostics | 批量排查 query/recovery 瓶颈 |
| `b1c1853` / `941720c` | 08-07 | F0/F1/F2 resolution screening | 建立 feature-resolution 对照 |
| `690994b`–`08f1b96` | 08-07 | Conda 与 Slurm tasks | HPC 运行环境和提交入口 |
| `a9f56e2` / `4935c31` | 08-08 | model capacity experiment | C0/C1/C2 受控容量基线 |
| `f9bd206`–`bad3616` | 08-08 | C2F2 50k 与分析 | canonical 长训练与分析入口 |
| `f91372b` / `68f4949` | 08-08 | 1920 Slurm 与可视化 | 分辨率实验和结果展示 |
| `1a4f5ce`–`51e5e1b` | 08-09 | 1920 C2F2 analysis | 三 seed 1920 汇总 |
| `1eee021`–`450ec29` | 08-09 | OpenMVS/960 prediction | coarse-mesh recovery 与 OpenGL pipeline |
| `9722afa` / `39aa6d5` | 08-10 | 14/28/56-view ablation | view-count 对照 |
| `4647e1e`–`24f3169` | 08-10 | 双语文档与公式 | 固化项目状态和数学合同 |
| `7a2da2f` | 08-11 | query/current-mesh ablations | current-query 对照框架 |
| `1ae2ca8` | 08-11 | 50k continuation job | 20k→50k 额外训练对照 |
| `34c17ee`–`299368b` | 08-12 | 50k downstream | 证明 loss 降低未转化为 geometry 改善 |
| `555e904`–`ff3a6b1` | 08-12 | exact-target oracle | 定位 learned prediction/recovery gap |
| `4b04eca` / `660e441` | 08-12 | raw-residual Top-k | 定量定位高残差顶点贡献 |
| `a9414ed`–`5032670` | 08-12 | H2/raw 三组消融 | 得到 B direct-raw 主结果 |
| `5d7ea9b` | 08-12 | local jitter report | 排除当前 jitter 配置 |
| `bd0f2aa` | 08-12 | H2 + recursive evaluations | 固化 19/25 和递归退化结论 |

## 3. 实验结果对比

### 3.1 早期 Bunny 与 Sofa50 pipeline 验证

#### 单物体与 Stanford Bunny（8 月 4 日）

- 初始 package 建立了 image encoder、多视图投影/聚合、graph layers、loss、trainer、
  evaluation 和 Laplacian reconstruction 的隔离闭环。
- Stanford Bunny 使用 35,947 个 OBJ vertices、69,451 faces；其中 1,113 个 vertex
  records 未被任何 face 引用。未清理 topology 时这些点的 `h=0`，normalized target
  可达到约 `8.37e11`，两个 normalized 300-step runs 均被判定 exploded；两个
  raw-target baselines 保持稳定并改善 coarse mesh。
- 后续确定性删除 unreferenced vertices，并在 normals、corruption、graph、target、
  projection、training、recovery 和 evaluation 前统一应用 remap。清理后的实验验证
  normalized formulation 可稳定运行；但它只有一个对象、一个 corruption 和短 CPU
  overfit，只证明合同有效，不证明 normalized target 优于 raw 或可以跨对象泛化。

#### Canonical Sofa50 H2，2,000 epochs（8 月 7 日固化）

| Metric | Epoch 100 | Epoch 1,000 | Epoch 2,000 |
|---|---:|---:|---:|
| Train loss ↓ | 0.0615358 | 0.0367423 | **0.0365896** |
| Validation loss ↓ | 0.0567680 | 0.0371126 | **0.0369713** |
| Validation cosine ↑ | 0.618441 | 0.772255 | **0.773831** |
| High-10% cosine ↑ | 0.662561 | 0.798558 | **0.799548** |

Epoch 2,000 的 correct-RGB/zero-RGB normalized MSE 为 `100.646/124.313`，模型
确实使用 RGB；confidence 与负 normalized error 的相关系数为 `0.487002`。但
expanded validation 仍未成功：

| Recovery | Initial Chamfer | Refined Chamfer | P2S | Normal | Flips | Improved |
|---|---:|---:|---:|---:|---:|---:|
| Main + confidence | 0.000652884 | **0.00299063** | **0.00288043** | **0.877039** | 6,078 | 0/5 |
| Hard visibility only | 0.000652884 | 0.00334968 | 0.00313189 | 0.873276 | 6,055 | 0/5 |
| Zero RGB | 0.000652884 | 0.00714054 | 0.00664820 | 0.859963 | 9,017 | 0/5 |

#### Renderer visibility / recovery gating（8 月 6 日）

| Variant | Chamfer ↓ | Normal ↑ | Visible displacement ↓ | Invisible displacement ↓ | vs initial |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.120283 | 0.5026 | 0.34983 | 0.40937 | 0/5 |
| Hard mask | **0.0146517** | 0.7089 | **0.08129** | 0.06813 | 0/5 |
| Hard mask + unseen anchor | 0.0198938 | **0.7196** | 0.12845 | **0.004389** | 0/5 |

Hard mask 将 Chamfer 降低约 `87.8%`，说明 visibility gating 必要；但它仍约为
initial Chamfer `0.000652884` 的 22 倍。强 unseen anchor 虽冻结不可见区域，却让
Chamfer 反弹。证据把瓶颈进一步指向 visible/low-view prediction error、graph coupling
和 query distribution，而不是只归因于不可见顶点。

#### 8 月 7 日诊断筛选

| 诊断 | 结果 | 判定 |
|---|---|---|
| 1,000-step feature resolution | F2 EPE `12.2433`；F0 `12.5780`；F1 `12.7205` | F2 screening 最低 |
| Geometry-aware sampling | all-EPE `12.3002→16.7813/18.1458` | 不支持 high-Laplacian oversampling |
| Oracle residual expert | 2k best val `0.0454773/0.0454485` | 无实质分离 |
| Counterfactual refinement | 三 target arms 均 `0/8` 改善 | 不支持 |
| Residual target comparison | 三种 target 均 `0/4` 改善 | 不支持 end-to-end |
| H2 normalization audit | max relative L2 `4.4331e-17` | 回算合同通过 |
| Identity/oracle recovery | Chamfer `0.00196814→0.00134768` | solver 有 oracle 改善空间 |
| Query transfer gap | expanded query 距 GT surface `0.0184h–0.0270h`；训练扰动 ≤`0.001h` | 明显分布外 |
| Delta-scale sweep | control/perturbed 均 `0/5` | 单纯缩放不解决问题 |

这些早期筛选共同促成了后续 current-query/current-graph 实验，而不是继续盲目延长
原 canonical H2 训练。

### 3.2 模型容量与输入规模

#### 模型容量，2,000 steps

| 容量 | Best val loss ↓ | All EPE ↓ | Top-10% EPE ↓ | Global cosine ↑ |
|---|---:|---:|---:|---:|
| C0 | 0.0478404 | 11.1322 | 39.2638 | 0.6700 |
| C1 | 0.0446807 | 10.5003 | 35.9503 | 0.7137 |
| C2 | **0.0428904** | **10.0454** | **35.4161** | **0.7193** |

C2 在该 screening 中统一优于 C0/C1，因此后续主实验使用 C2F2。

#### View count，C2F2、seed 7、20k

| Views | Best val loss ↓ | Raw EPE ↓ | Raw Top-10% ↓ | Runtime | Peak GPU memory |
|---:|---:|---:|---:|---:|---:|
| 14 | 0.0139316 | 0.003203 | 0.022291 | 3.146 h | 9,095 MiB |
| 28 | **0.0130296** | 0.003119 | 0.021110 | 6.699 h | 18,130 MiB |
| 56 | 0.0138104 | **0.003016** | **0.019517** | 13.968 h | 31,692 MiB |

28 views 的 checkpoint-selection loss 最低；56 views 的 raw tail error 最低，但
运行时间是 28 views 的 `2.085×`，validation loss 反而高 `5.99%`。因此 28 views
是后续受控实验的性价比选择，而不是声称所有指标都最优。

#### 960 与 1920 C2F2

| 输入 | 预算 | Mean all EPE ↓ | Mean Top-10% EPE ↓ | Mean cosine ↑ | Expanded Chamfer ↓ |
|---|---:|---:|---:|---:|---:|
| 960 | 50k | **2.82815** | **15.37434** | 0.89110 | **0.00116244** |
| 1920 | 20k | 3.09280 | 16.32997 | **0.89537** | 0.00125695 |

两组预算不同，不能把差异单独归因于分辨率。已有记录不支持“升至 1920 就降低
endpoint error 或 recovery Chamfer”。

### 3.3 Query graph 与组合条件

| Query graph | Best val loss ↓ | Raw EPE ↓ | Raw Top-10% ↓ | Raw Top-1% ↓ |
|---|---:|---:|---:|---:|
| GT | **0.0139316** | 0.003203 | 0.022291 | 0.125458 |
| GT-sub1 | 0.0614830 | 0.006359 | 0.043869 | 0.197677 |
| GT-adaptive | 0.0145840 | **0.002917** | **0.018364** | **0.093426** |

GT-sub1 明显退化。GT-adaptive 在自身 graph 上降低 raw tail EPE，但 graph、局部
`h²`、顶点集合和 target 分布同时改变，因此不能把这些 raw 指标直接解释为共同
物理位置上的纯模型能力提升。

`28 views + GT-adaptive` 的 raw EPE 为 `0.002879`，低于两个单因素 arm；但其
Top-10% EPE 比 GT-adaptive 高 `1.57%`，未通过预设的全部四项 retention 判据。

### 3.4 Recovery 基线与额外训练

#### Expanded-query / OpenMVS

| 场景 | Initial Chamfer | Refined Chamfer | 改善数 | 结论 |
|---|---:|---:|---:|---|
| Expanded C2F2 960 | 0.000652884 | 0.00116244 | 每 seed 0/5 | 退化 |
| Expanded C2F2 1920 | 0.000652884 | 0.00125695 | 每 seed 0/5 | 退化 |
| OpenMVS，200 iters | 0.0212023 | 0.0213199 | 2/48 | mean 退化 |
| OpenMVS，1,000 iters | 0.0212023 | 0.0213198 | 2/48 | 增加迭代无实质变化 |

#### Current-query 20k→50k

| Metric | 20k | 50k | 变化 |
|---|---:|---:|---:|
| Best validation loss | 0.0151933 | 0.0139379 | **-8.26%** |
| Evaluation loss | 0.0117459 | 0.0112148 | **-4.52%** |
| Vector L2 | 2.391465 | 2.285737 | **-4.42%** |
| Refined Chamfer | **0.00417940** | 0.00422413 | **+1.07%（退化）** |
| Refined P2S | **0.00423266** | 0.00424708 | **+0.34%（退化）** |
| Improved over initial | **5/25** | 3/25 | 丢失 2 个成功样本 |

这是本阶段最重要的控制结论之一：native/prediction loss 持续下降，不保证 recovery
geometry 同步改善。后续实验必须保留 Chamfer、P2S、flip 和 sample-transition
判据，不能只按 validation loss 宣称成功。

### 3.5 Oracle 与 raw-residual Top-k

| Endpoint | Refined Chamfer ↓ | P2S ↓ | Normal ↑ | Flips ↓ | Improved/25 |
|---|---:|---:|---:|---:|---:|
| Current-query 20k | 0.00417977 | 0.00423260 | 0.940028 | 8,424 | 5 |
| Current-query 50k | 0.00422430 | 0.00424771 | 0.939283 | 8,495 | 3 |
| Exact-target oracle | **0.00317485** | **0.00317849** | **0.963383** | **3,242** | **25** |

Exact-target oracle 相对 initial Chamfer/P2S 分别降低 `18.87%`/`19.22%`，证明
固定 `P_proxy`、current-graph target 和 recovery objective 在零预测误差时可以
产生有效 geometry。

Top-k replacement 的核心结果：

- Top 1% raw-residual vertices 包含约 `83.98%–85.15%` residual energy，但只关闭
  `33.20%–36.85%` mean Chamfer oracle gap。
- Top 10% replacement 已使 `25/25` samples 优于 initial，但仅关闭
  `58.99%–62.64%` mean Chamfer gap。
- 达到至少 90% mean Chamfer gap closure 需要替换 50% 顶点。
- 因而 high-residual tail 很重要，但当前证据不支持“只修 Top 1% 或 Top 10% 就能
  解释全部 downstream gap”。

### 3.6 Local query jitter

| Metric | No jitter | Local jitter | Jitter − control |
|---|---:|---:|---:|
| Test raw EPE | 0.007681539 | 0.007804981 | +0.000123442 |
| Test raw Top-10% | 0.053443110 | 0.054163195 | +0.000720084 |
| OpenMVS refined Chamfer | 0.025067426 | 0.025249771 | +0.000182345 |
| OpenMVS refined P2S | 0.024900180 | 0.025077573 | +0.000177393 |

Jitter arm 在 5/5 paired OpenMVS meshes 上 Chamfer 更高；当前合同不支持启用
`std=0.003h`、cap=`0.009h` 的 training-only local query jitter。

### 3.7 H2 / raw-Laplacian 三组主对照

三组模型、split、seed、optimizer、scheduler、batching、28 views 和 20k budget
一致。Native validation loss 位于不同数值空间，不能横向比较。

| Arm | Output/loss | Raw EPE ↓ | Top-1% ↓ | Top-10% ↓ | Weighted RMS ↓ | Chamfer ↓ | P2S ↓ | Flips ↓ | Improved |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A canonical H2 | normalized/output | 0.00769237 | 0.253855 | 0.0557517 | 0.0427999 | 0.00456011 | 0.00462286 | 10,195 | 3/25 |
| **B direct raw** | **raw/output** | **0.00300525** | **0.0417512** | **0.0136982** | **0.00611072** | **0.00380671** | **0.00380587** | **6,566** | **19/25** |
| C normalized/raw loss | normalized/raw | 0.00333673 | 0.0547519 | 0.0159651 | 0.00815502 | 0.00383121 | 0.00385409 | 7,057 | 16/25 |

B 的 raw EPE 相对 A/C 分别降低 `60.93%`/`9.93%`；Chamfer 相对 A/C 分别降低
`16.52%`/`0.64%`。B 是当前统一协议下的主结果。

B/C 的约 `1e-6` native loss 很低是 raw-Laplacian 数值单位导致，并不表示比 A 的
normalized loss 小四个数量级；判定必须使用统一 raw-space 和 geometry endpoints。

### 3.8 冻结模型三轮递归 refinement

主策略每轮从上一轮 mesh 重新构造 positions、normals、Laplacian、local `h`、
normalization、RGB projection 和 OpenGL-960 visibility，但模型权重保持冻结。

| 结果 | B baseline/R0 | R1 | R2 | R3 |
|---|---:|---:|---:|---:|
| Raw EPE ↓ | **0.00300525** | 0.00382410 | 0.00470070 | 0.00526386 |
| Chamfer ↓ | **0.00380671** | 0.00399224 | 0.00428061 | 0.00455340 |
| P2S ↓ | **0.00380587** | 0.00398244 | 0.00426096 | 0.00452720 |
| Improved/original | **19/25** | 12/25 | 7/25 | 2/25 |
| Retained original 19 | **19/19** | 12/19 | 7/19 | 2/19 |
| Gained from failed 6 | 0/6 | 0/6 | 0/6 | 0/6 |
| Cumulative flips | **6,566** | 12,419 | 17,327 | 20,964 |

R1/R3 Chamfer 已分别比 B baseline 高 `4.87%`/`19.61%`。固定 prepared visibility
的 sensitivity 路径更差，R3 为 `0/25`。因此不能把冻结 B 直接递归当作提升方法。

## 4. 综合判断

| 假设 | 状态 | 证据 |
|---|---|---|
| learned-Laplacian 单物体闭环可运行 | 支持为 sanity check | Bunny raw-target baselines 稳定并改善 coarse |
| 未清理 Bunny 可直接用于 H2 normalized target | 不支持 | 1,113 个孤立顶点导致 normalized runs exploded |
| Renderer hard visibility 足以解决 expanded recovery | 不支持 | Chamfer 降 87.8%，但仍为 0/5 |
| Query transfer gap 可忽略 | 不支持 | expanded query 距离 0.0184h–0.0270h，训练上限 0.001h |
| High-Laplacian oversampling 改善整体 EPE | 不支持 | all-EPE 从 12.3002 升至 16.7813/18.1458 |
| 增加模型容量能改善 screening 指标 | 支持 | C2 优于 C0/C1 |
| 28 views 相对 14 views 降低 val loss | 支持 | `-6.47%` |
| 56 views 在所有指标上优于 28 views | 不支持 | raw error 更低，但 val loss `+5.99%`、runtime `2.085×` |
| 1920 自动优于 960 | 不支持 | 不等预算下 mean EPE/Chamfer 更高 |
| 增加 recovery iterations 解决 OpenMVS gap | 不支持 | 200→1,000 iterations 几乎不变 |
| 额外训练 steps 必然改善 geometry | 不支持 | 20k→50k loss 降而 Chamfer/P2S 升 |
| Exact current-graph target 可改善 geometry | 支持 | oracle `25/25`，Chamfer 相对 initial `-18.87%` |
| Top 1% residual 单独解释全部 gap | 不支持 | 仅关闭 33.20%–36.85% Chamfer gap |
| 当前 local jitter 配置有效 | 不支持 | prediction 与 OpenMVS geometry 均退化 |
| Direct raw-Laplacian 优于 A/C | 支持 | 最低 raw error/Chamfer，`19/25` |
| 冻结 B 递归三轮继续提升 | 不支持 | `19→12→7→2/25` |
| 当前 X1 stage-2 适配超过冻结 B | 不支持 | `16/25`、Chamfer `0.00384032`，且丢失 5 个原成功样本 |
| Huber 梯度压缩集中于最高曲率尾部 | 支持 | Top 1% any-component saturation `66.049%`，retention `58.436%` |

总体路径已经从“增加分辨率、views、steps、重复 inference 或当前 X1 stage-2
配方”转向两个更具体的问题：一是调整 loss 对最高曲率尾部的梯度分配并验证其
surface sensitivity；二是在 Future2000 规模上比较 current-graph Laplacian 与
direct displacement 的严格配对 geometry 结果。

## 5. 新完成诊断与运行中工作

### 5.1 Stage-2 distribution adaptation（已完成）

Stage-2 实现包括数据生成、三分支训练、评估、Slurm 编排和测试。它们不计入
本报告基于父提交统计的 91 个 commits，但随本报告的发布 commit 一并提交。

实验合同：从同一 B checkpoint
`ba1c77c3ce4c91ef70ba4b70570664d3ffa2c1c41a3f9f342778149ead0526e8`
出发，运行：

1. `continue_original`：继续在 X0 训练，控制“只增加 20k steps”的效果；
2. `continue_B_result`：在冻结 B 一次生成的 X1 上训练；
3. `continue_mix_50_50`：X0/X1 各 50%，检查遗忘。

最终三组均完成到 step 40,000；validation loss 来自不同输入分布，仍不横向比较。

| Arm，best checkpoint | Raw EPE | Chamfer | P2S | Normal | 改善数 | Retained/Gained/Lost |
|---|---:|---:|---:|---:|---:|---:|
| frozen stage-1 B | 0.00300521 | **0.00380687** | **0.00380594** | **0.942463** | **19/25** | 19/0/0 |
| continue_original | 0.00369851 | 0.00390257 | 0.00389548 | 0.931793 | 16/25 | 14/2/5 |
| continue_B_result | **0.00349257** | **0.00384032** | **0.00383760** | **0.936939** | 16/25 | 14/2/5 |
| continue_mix_50_50 | 0.00363284 | 0.00388119 | 0.00387446 | 0.934341 | 16/25 | 14/2/5 |

X1 分支虽然是 continuation 中最好的一组，但没有超过冻结 stage-1，也没有优于
matched extra-training control 到足以改变结论的程度。它找回 2 个失败样本，同时
丢失 5 个成功样本。

### 5.2 Arm-B Huber 饱和诊断（已完成）

本地 float32 诊断覆盖 validation split 的 25 samples、243,000 个有效 vertices，
训练 Huber `delta=0.01`。主要结果：

| Group | Mean raw error | P(any component saturated) | Gradient share | Gradient retention | Huber-loss share |
|---|---:|---:|---:|---:|---:|
| bottom 90% | 0.00149443 | 0.114% | 72.910% | 99.716% | 32.145% |
| top 10% | 0.00614772 | 12.185% | 27.090% | 83.308% | 67.855% |
| top 1% | 0.0195337 | 66.049% | 5.785% | 58.436% | 34.931% |

结论应限定为：极端 top 1% 曲率 vertex 存在明确的 Huber 梯度压缩；不能表述为
整个 top 10% 全面饱和。要证明它与 reconstruction objective 的完整因果不一致，
仍需将同一 vertex 与 surface displacement/Chamfer sensitivity 配对。

### 5.3 Future2000 GT-adaptive 扩展（停在 64k，可恢复）

数据合同为 2,000 个源 meshes × 5 个固定 current variants，object-level split
为 `8000/1000/1000`，使用 28 views、GT-adaptive subdivision、C2F2、raw
current-graph target；配对实验直接预测 vertex displacement。

四张 L40 的 200k 主训练 job 15794 在 step 32,000 失败。训练本身此前正常：

| Step | Train loss | Validation loss |
|---:|---:|---:|
| 20,000 | 5.30099e-6 | 4.99851e-6 |
| 30,000 | 4.82400e-6 | **4.19731e-6** |
| 40,000 | 4.62000e-6 | **3.88000e-6** |
| 50,000 | 4.26000e-6 | 4.23000e-6 |
| 60,000 | 4.19000e-6 | 5.27000e-6 |
| 64,000 | **3.99000e-6** | — |

首个异常是 DataLoader worker 的 `Too many open files (24)`；随后其他 DDP ranks
在 `ALLREDUCE` 等待 30 分钟并触发 NCCL watchdog。修复使用 `file_system`
multiprocessing sharing strategy 和每个 rank 四个 non-persistent workers。
Job 15795 从完整 step-32k checkpoint 恢复并达到 step 64,000，后因
DataLoader worker 耗尽 `/dev/shm` 而失败（`Bus error` / `No space left on
device`）。Step-64k checkpoint 仍完整可恢复。15759/15760 配对 displacement jobs
已取消，因此这一规模实验尚无最终 geometry 结论。

现有 external diagnostic array 15791 在快照时 shard 0 为 `181/334`，其中
64 completed、117 failed。作业本身无 Traceback，但样本失败率过高且其余 shards
未完成，因此不形成外部方法结论。新的 external comparison 只使用本地 runner。

### 5.4 8 月 14–15 日后续 Sofa50 结果

本报告的 commit 计数边界仍是原文档的 8 月 4–14 日历史窗口；下列后续实验作为
结果 addendum，不追溯修改原 commit 统计。

- Raw MSE 没有降低 test Top-10%/Top-1% 或 mean Chamfer/P2S；Huber/MSE 改善数为
  `19/25` 与 `16/25`。MSE global batch 6 与基线 2 不同。
- Learned dynamic residual expert 的 learned final 相对 joint base 在 raw EPE、Chamfer、P2S
  上均 `25/25` 胜出。Validation-selected `alpha=0.16` 和 5 个 within-mesh shuffles
  支持：expert 是主要贡献，learned gate placement 有较小额外贡献。
- 960 image features 中，Gaussian-only 的 Chamfer `0.00377507`、改善数 `21/25`
  最好；original+HF 的 raw EPE `0.00288627`、Top-10% `0.0117524`、Top-1%
  `0.0347902` 最好。
- Native-1920+HF 数据 audit 已通过，4×L40 job 15854 从零训练中。它的 global
  batch 为 4，960 HF 为 2，因此最终只能作为带 batch caveat 的分辨率消融。

## 6. 数据来源与复核规则

主要来源：

- [实验数据汇总](EXPERIMENT_DATA_SUMMARY.zh-CN.md)
- [Sofa50 受控消融报告](SOFA50_CONTROLLED_ABLATIONS_REPORT.zh-CN.md)
- [Local query jitter 报告](SOFA50_LOCAL_QUERY_JITTER_ABLATION_REPORT.zh-CN.md)
- [Renderer visibility 与 recovery 报告](VISIBILITY_AWARE_RECOVERY_REPORT.md)
- [Canonical Sofa50 H2 报告](../runs/learned_laplacian/sofa50_50mesh_2000epoch_absolute_h2_confidence/REPORT.md)
- [H2 三组最终报告](../runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7/analysis/REPORT.md)
- [三轮递归最终报告](../runs/learned_laplacian/sofa50_synthetic_current_28view_b_recursive_refinement_3round_seed7/REPORT.md)
- [Future2000 本地对比任务](FUTURE2000_LOCAL_COMPARISON_TASKS.md)
- Git history：`git log --since='2026-08-04 00:00:00 +0100' HEAD`

复核原则：

- 已完成实验使用最终 JSON/CSV/report 的记录值；
- 不跨 loss space 比较 native loss 数量级；
- 不把不同 graph 上的 raw metric 当成共同物理点 paired metric；
- 不把中间 checkpoint 或运行中 validation loss写成最终 geometry 结论；
- 取消或被替代的 jobs 不纳入主要结果。

## 7. 完整 commit 清单

### 2026-08-12（17）

- `bd0f2aa` Add H2 and recursive refinement evaluations
- `5d7ea9b` Report local query jitter ablation
- `5032670` Tighten normalization ablation audit
- `1db92ff` Add three-arm normalization evaluator
- `a9414ed` Add raw-loss normalization ablation arms
- `660e441` Report top-k recovery comparison
- `4b04eca` Add raw-residual top-k recovery comparison
- `ff3a6b1` Record synthetic current oracle recovery result
- `f02adad` Report solver-input raw tail diagnostics
- `03795ff` Bound oracle learned replay AMP drift
- `77e2bcb` Validate oracle target against stored proxy
- `f916d6f` Add HPC oracle recovery evaluation job
- `555e904` Add synthetic current oracle recovery comparison
- `299368b` Record current 50k downstream results
- `4abe457` Bound replay flip-count tolerance
- `ca586ef` Allow bounded evaluation replay drift
- `34c17ee` Add synthetic current 50k downstream evaluation

### 2026-08-11（2）

- `1ae2ca8` Add 50k synthetic-current continuation job
- `7a2da2f` Add Sofa50 query and current-mesh ablations

### 2026-08-10（5）

- `24f3169` Replace unsupported GitHub math macros
- `9cc79d9` Document model and loss equations
- `4647e1e` Update bilingual README with current project status
- `39aa6d5` Add C2F2 14-28-56 view ablation experiment
- `9722afa` Add C2F2 14-28-56 view ablation experiment

### 2026-08-09（14）

- `450ec29` Add 960 OpenMVS predict OpenGL pipeline with 1,000 Laplacian iterations
- `ec87c21` Add 960 OpenMVS predict OpenGL pipeline
- `166042d` Add 960 OpenMVS prediction
- `722176a` Add 960 OpenMVS prediction
- `4cd8621` Add 960 OpenMVS prediction
- `51e5e1b` Add 1920 C2F2 analysis
- `84519ac` Add 1920 C2F2 analysis
- `9b4ceb6` Add 1920 C2F2 analysis
- `e77ca30` Add 1920 C2F2 analysis
- `7e0598e` Add 1920 C2F2 analysis
- `af77a83` Add 1920 C2F2 analysis
- `1a4f5ce` Add 1920 C2F2 analysis
- `77db161` OpenMVS coarse mesh refinement test
- `1eee021` OpenMVS coarse mesh refinement test

### 2026-08-08（14）

- `68f4949` Visualization
- `f91372b` Add C2F2 Slurm script for 1920
- `0309275` Add C2F2 Slurm script for 1920
- `12f2fab` Add full C2F2 Slurm script
- `58c0254` Add full C2F2 Slurm script
- `ffc9651` Add C2F2 Slurm script
- `bad3616` Add C2F2 analysis
- `0784cad` Add C2F2 analysis
- `f9bd206` Add C2F2 50,000-step experiment
- `27127d4` Add C2F2 50,000-step experiment
- `c0b3434` Add C2F2 50,000-step experiment
- `e5751ff` Add C1F2 HPC negative/positive experiment
- `4935c31` Model capacity experiment
- `a9f56e2` Add model capacity experiment

### 2026-08-07（9）

- `08f1b96` Build Slurm task
- `1ff5ae5` Build Slurm task
- `09c8c6a` Build Slurm task
- `690994b` Add Conda environment
- `941720c` Add resolution F2 test
- `b962d63` Minor update
- `b1c1853` Resolution comparison
- `32e4738` Residual experiment
- `c656684` feat: canonicalize Sofa50 h2 Laplacian pipeline

### 2026-08-06（7）

- `dd150ed` Add renderer visibility and recovery gating
- `9540a30` Add Chinese README
- `6f9e3d6` Align documentation with GT Laplacian objective
- `ceb5012` Add learned Laplacian diagnostics and Sofa50 training
- `d543fd3` Add GT-query Laplacian training pipeline
- `bebe923` Document optimized multi-mesh training
- `ecf328b` Optimize lazy multi-mesh training pipeline

### 2026-08-05（6）

- `a778146` Add memory-safe 1920px lazy training profile
- `f180920` Support lazy image-path prepared samples downstream
- `a26b195` Add fixed cube-surface OpenGL render contract
- `95c4061` Keep research assets and generated runs out of main
- `c147238` new version
- `bde55ef` new version

### 2026-08-04（17）

- `4304db7` Update workspace: save changes
- `a79062b` Document cleaned normalized Bunny experiment
- `2bfd317` Add Bunny mesh and error visualizations
- `225a813` Add cleaned Bunny comparison runner
- `71f2eea` Handle isolated Laplacian vertices explicitly
- `58e25c1` Add deterministic Bunny mesh cleaning
- `d985cb7` Add pre-training graph diagnostics
- `cfe4d42` Add edge-scale-normalized Laplacian experiment
- `ac0d023` Document Stanford Bunny overfitting experiment
- `fe7e10b` Add Bunny ablation runner and configuration
- `f4689ed` Scale learned Laplacian evaluation to Bunny meshes
- `5b1645e` Add reproducible Bunny experiment preparation
- `78a1d5d` Use consistent geometry evaluation sampling
- `5783e5b` Add learned Laplacian tests and documentation
- `5e2227a` Add single-object overfitting and reconstruction
- `c6aa635` Add single-object sample preparation
- `0081f80` Add isolated learned Laplacian package
