# 多视图 Laplacian 网格细化

[English](README.md) | [简体中文](README.zh-CN.md)

方法定义：[Sofa50 标准流程](docs/CANONICAL_SOFA50_PIPELINE.md)

训练说明：[多网格 GT-query 训练](docs/MULTI_MESH_TRAINING.zh-CN.md)

可见性与恢复：[可见性感知恢复报告](docs/VISIBILITY_AWARE_RECOVERY_REPORT.md)

实验指标与运行状态：[实验数据汇总](docs/EXPERIMENT_DATA_SUMMARY.zh-CN.md)

当前 Sofa50 受控消融：[Direct-raw/loss/expert/image-feature 报告](docs/SOFA50_CONTROLLED_ABLATIONS_REPORT.zh-CN.md)

Future2000 本地对比任务：[本地任务说明](docs/FUTURE2000_LOCAL_COMPARISON_TASKS.md)

近期 commit 与实验记录：[8 月 4–15 日报告与补充记录](docs/RECENT_COMMIT_AND_EXPERIMENT_REPORT_2026-08-04_2026-08-14.zh-CN.md)

View-count 与 query-resolution 结果：[消融报告](runs/learned_laplacian/sofa50_c2f2_view_query_resolution_ablation_20k_seed7/analysis/REPORT.md)

28-view current-graph target/loss-space 结果：[H2 消融报告](runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7/analysis/REPORT.md) | [25 组可视化总览](runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7/analysis/comparison_images/B_direct_raw_laplacian/overview_25.png)

## 项目状态

状态日期：2026-08-15 18:47 BST。

| 组件 | 状态 | 结论 |
|---|---|---|
| GT-query 数据与训练流程 | 已实现 | 绝对 GT `h^2` 归一化 Laplacian 的直接监督训练可以运行。 |
| Target 泄漏控制 | 已实现并测试 | 模型输入不包含 GT Laplacian。 |
| Sofa50 960 图像分辨率消融 | 已完成 | 在 50,000 个 optimizer steps 下，F2 的 exact-query error 低于 F0 和 F1。 |
| Sofa50 960 C2F2 训练 | 已完成 | 三个 seed 均完成 50,000 个 optimizer steps。C2F2 是当前 exact-query error 最低的配置。 |
| Sofa50 1920 C2F2 训练 | 已完成 | 三个 seed 均完成 20,000 个 optimizer steps。平均 endpoint error 和 recovery Chamfer 高于 960 结果；平均 cosine 更高。 |
| Expanded-query recovery | 已完成 | 对已评估的 960 和 1920 C2F2 checkpoint，五个 validation mesh 的 Chamfer 均增加。 |
| OpenMVS coarse-mesh recovery | 50 个物体中完成 48 个 | 细化后的平均 Chamfer 增加。两个物体缺少 OpenMVS coarse mesh。 |
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
| Native-1920 + high-frequency residual | 运行中，尚无最终结论 | Native renderer 观测已通过 camera/split/graph/target/visibility audit。4×L40 job 15854 从零训练 20,000 steps；global batch 4 与 960 基线的 2 不同。 |
| Future2000 GT-adaptive 扩展 | 停在 step 64,000，可恢复 | Job 15795 从 32,000 继续到 64,000，后因 DataLoader worker 耗尽 shared memory 失败。Checkpoint 完整，尚无最终 geometry 结果。 |
| Future2000 外部基线 | 未完成，不报告结论 | 现有分片诊断的样本级失败数较高；新对比任务仍仅使用本地脚本。 |
| 自动化测试 | 当前文档改动对应检查通过 | Raw loss、dynamic expert/gate、image feature、native-1920 数据准备和分布式训练相关测试通过；下文验证命令是新 checkout 的最终依据。 |

当前模型能够在 GT-query graph 上学习监督微分场，并使用 RGB 信息。从 GT-query
graph 到 expanded 或 OpenMVS query graph 的迁移未产生几何改善。当前 recovery
流程未达到端到端 coarse-mesh refinement 目标。

当前扩展实验采用另一个明确的 current-graph 合同：query mesh 与 connectivity
属于模型输入，监督 raw target 为 `L_current @ P_proxy`；配对分支改为直接预测
vertex displacement。这些 target 是监督信号，不是额外的 inference 输入。

## 方法

模型根据标定后的多视图观测和图查询预测 GT 局部微分信号：

```text
多视图 RGB + 相机 + 3D query + 局部图上下文
    -> 绝对 GT h^2 归一化 Laplacian
```

对于 GT vertex `i`：

```text
delta_gt_i = (L_gt V_gt)_i
h_i        = vertex i 的平均相邻 GT 边长
target_i   = delta_gt_i / (h_i^2 + epsilon)
```

训练使用 GT vertices 和 GT connectivity。一部分 query 保持精确 GT 位置，其余
query 按照 `h_i` 加入有界的法向与切向扰动。Target 始终绑定到原始 GT vertex。

当前 geometry mode 为 `query_fourier`。Fourier feature 在 query augmentation
之后计算。Image feature、query position、normal、相对局部尺度、degree 和 graph
connectivity 是模型输入。Raw 和 normalized GT Laplacian 仅用于监督。GT-query
training sample 中的 `initial_laplacian` 为零。

推理使用独立生成的 coarse mesh 或 topology-expanded mesh：

```text
coarse mesh vertices
  -> 投影到标定视图
  -> 聚合 image features
  -> 预测 normalized Laplacian
  -> 使用当前 query graph 尺度反归一化
  -> confidence/visibility 加权 Laplacian recovery
```

推理 graph 不接收任何 GT differential value。GT geometry 仅用于评估。

## 数学定义

以下公式对应当前实现。历史路径会单独标记。

### Uniform Laplacian 与监督目标

令 `N(i)` 为 vertex `i` 的 one-ring neighbours，`d_i = |N(i)|`。Uniform graph
Laplacian 为

$$
(L X)_i = X_i - \frac{1}{d_i}\sum_{j\in N(i)}X_j,
\qquad
L_{ii}=1,\quad L_{ij}=-\frac{1}{d_i}.
$$

Isolated vertex 对应零 Laplacian row。局部边尺度与绝对监督目标为

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

网络预测绝对 normalized vector `delta_hat_prediction`，不预测 displacement 或
Laplacian residual。对于当前 inference graph：

$$
\delta_i^{\mathrm{pred}}
=\widehat{\delta}_i^{\mathrm{pred}}\left((h_i^{\mathrm{current}})^2+\varepsilon\right).
$$

反归一化只执行一次，并使用当前 query graph 的尺度。

### Query 扰动

对于 perturbed GT query：

$$
q_i=V_i+h_i\left(\xi_i n_i+\zeta_i t_i\right),
\qquad
\xi_i\sim\mathcal N(0,\sigma_n^2),\quad
\zeta_i\sim\mathcal N(0,\sigma_t^2),
$$

其中 `n_i` 是 vertex normal；`t_i` 是从 Gaussian 3D direction 中去除法向分量
后得到的随机单位切向量。位移满足

$$
\lVert q_i-V_i\rVert_2\leq \kappa h_i.
$$

Canonical settings 为 `sigma_n = sigma_t = 0.0003`、`kappa = 0.001`，
exact-query fraction 为 `0.2`。Exact query 使用 `q_i = V_i`。扰动只改变 query
position，不改变 graph 或 target。

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

令 `f_vi` 表示正深度且投影位于图像范围内，`r_vi` 表示预计算的 renderer-native
back-face 与 occlusion 结果。Feature-sampling mask 为

$$
z_{vi}=f_{vi}r_{vi}\in\{0,1\}.
$$

若 `F_v(u_vi,v_vi)` 是 bilinear sampled CNN feature，则 masked mean 与 valid-view
ratio 为

$$
\overline F_i=
\frac{\sum_{v=1}^{M}z_{vi}F_v(u_{vi},v_{vi})}
     {\max\left(1,\sum_{v=1}^{M}z_{vi}\right)},
\qquad
\rho_i=\frac{1}{M}\sum_{v=1}^{M}z_{vi}.
$$

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

三个 convolution 均通过 padding 保持 input spatial resolution。

### Visibility、confidence 与 Gaussian gates

Canonical renderer gate 是严格的 any-view gate：

$$
m_i=\mathbf 1\!\left[\sum_{v=1}^{M}z_{vi}>0\right].
$$

可选 confidence head 预测有界 reliability：

$$
c_i=\mathrm{sigmoid}(g_\theta(x_i))\in[0,1].
$$

Canonical recovery weight 为

$$
w_i=m_i c_i.
$$

关闭 confidence head 时使用 `w_i = m_i`。所有 view 均不可见的 vertex，其
learned-Laplacian weight 严格为零。

仓库中的历史 coarse/GT projection 路径实现了以下 Gaussian distance-confidence
gate：

$$
g_i=\mathrm{clip}\!\left(
\exp\!\left[-\left(\frac{d_i^{\mathrm{surface}}}{s}\right)^2\right],
g_{\min},1\right).
$$

其中 `d_i^surface` 是 coarse query 到 GT surface 的距离，`s` 为
`distance_confidence_scale`。该 Gaussian gate 不是 renderer visibility，也不用于
canonical GT-query training。Canonical training 使用 `z_vi` 进行 image-feature
sampling；canonical recovery 使用 `m_i c_i`。

### Vertex representation 与 graph network

令 `c_obj` 和 `s_obj` 为 object normalization center 与 scale。Normalized query
为

$$
\widetilde q_i=\frac{q_i-c_{\mathrm{obj}}}{s_{\mathrm{obj}}}.
$$

使用 `K = 6` 个 frequencies 时，dynamic Fourier encoding 为

$$
\phi(\widetilde q_i)=
\left[
\widetilde q_i,
\left\{\sin(2^k\pi\widetilde q_i),
\cos(2^k\pi\widetilde q_i)\right\}_{k=0}^{K-1}
\right].
$$

每个 vertex 的输入为

$$
x_i=\left[
\phi(\widetilde q_i),\ n_i,\
\log\!\left(\max(h_i/s_{\mathrm{obj}},10^{-8})\right),\
\log(1+d_i),\ \rho_i,\ \overline F_i
\right].
$$

对于 C2F2，`phi` 为 39 channels，完整 vertex input 为
`39 + 3 + 1 + 1 + 1 + 64 = 109` channels。Graph backbone 为
`109 -> 256 -> 256`，后接 3 个 256-channel message-passing blocks 和 output
MLP `256 -> 256 -> 3`。Confidence side head 为 `109 -> 256 -> 1`，末端使用
sigmoid。

经过 input MLP 后，第 `l` 个 graph layer 计算

$$
\mu_i^{(l)}=\frac{1}{\max(1,d_i)}
\sum_{j\in N(i)}u_j^{(l)},
$$

$$
u_i^{(l+1)}=\mathrm{ReLU}\!\left(
u_i^{(l)}+operatorname{MLP}_l
\left([u_i^{(l)},\mu_i^{(l)}]\right)
\right).
$$

Output MLP 将最终 graph state 映射为
`delta_hat_prediction in R^3`。

### 训练目标

分量 residual 为

$$
r_{ik}=\widehat\delta^{\mathrm{pred}}_{ik}
-\widehat\delta^{\mathrm{GT}}_{ik}.
$$

逐分量 Huber function 为

$$
H_\tau(r)=
\begin{cases}
\frac{1}{2}r^2, & |r|\leq\tau,\\
\tau\left(|r|-\frac{1}{2}\tau\right), & |r|>\tau,
\end{cases}
\qquad \tau=0.01.
$$

Per-vertex error 与 primary loss 为

$$
e_i=\frac{1}{3}\sum_{k=1}^{3}H_\tau(r_{ik}),
\qquad
\mathcal L_{\mathrm{lap}}=
\frac{\sum_i a_i e_i}{\max(10^{-12},\sum_i a_i)},
$$

其中 `a_i` 是 prepared target-confidence/valid-scale weight。当前 full-vertex
GT-query contract 对有效的非 isolated vertices 使用单位权重，对无效 local scale
使用零权重。

Confidence side head 使用 detached prediction error：

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

完整训练目标为

$$
\mathcal L_{\mathrm{train}}
=\mathcal L_{\mathrm{lap}}
+\lambda_{\mathrm{conf}}\mathcal L_{\mathrm{conf}}.
$$

Canonical config 使用 `beta = 0.01`、`c_min = 10^-4` 和
`lambda_conf = 1`。Predicted confidence 不会重新加权 `L_lap`，因此 confidence
head 不能通过降低 confidence 来抑制 primary supervision。

### Laplacian recovery 目标

对固定 current graph `(X_0, F)` 构建 `L_current` 与 `h_current`，然后根据
`delta_pred` 恢复 vertex positions `X`。Canonical dense objective 为

$$
\mathcal L_{\mathrm{rec}}(X)=
\lambda_{\mathrm{lap}}
\sum_{i,k}H_\tau\!\left(
\sqrt{w_i}\left[(L_{\mathrm{current}}X)_{ik}
-\delta_{ik}^{\mathrm{pred}}\right]\right)
+\frac{\lambda_{\mathrm{anchor}}}{2}\lVert X-X_0\rVert_F^2
+\mathcal L_{\mathrm{edge}}+\mathcal L_{\mathrm{unseen}}.
$$

当前 canonical values 为 `lambda_lap = 1`、`lambda_anchor = 0.01`、
`lambda_edge = 0` 和 `lambda_unseen_anchor = 0`。Visibility/confidence weight
通过 `sqrt(w_i)` 作用于完整 Laplacian equation row。

对于较大的 uniform-Laplacian meshes，sparse solver 使用对应的 L2 form：

$$
\mathcal L_{\mathrm{sparse}}(X)=
\frac{\lambda_{\mathrm{lap}}}{N}
\left\lVert W^{1/2}(L_{\mathrm{current}}X-\delta^{\mathrm{pred}})\right\rVert_F^2
+\frac{\lambda_{\mathrm{anchor}}}{N}\lVert X-X_0\rVert_F^2,
\qquad W=\mathrm{diag}(w).
$$

### 报告指标

对于 predicted 与 target normalized Laplacians `P` 和 `T`，主要 prediction
metrics 为

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

报告中的 bidirectional vertex-to-surface Chamfer 为

$$
D_{\mathrm{C}}(A,B)=\frac{1}{2}\left[
\frac{1}{|V_A|}\sum_{x\in V_A}d(x,S_B)
+\frac{1}{|V_B'|}\sum_{y\in V_B'}d(y,S_A)
\right],
$$

其中 `S_A` 和 `S_B` 是 triangle surfaces，`V_B'` 是实际评估或 subsampled GT
vertex set。

## 数据契约

HPC 上使用的 Sofa50 数据目录为：

```text
/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_960
/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_refinement/multiview_1920
```

每个数据集包含 40 个 training、5 个 validation 和 5 个 held-out test objects。
标准 960 实验对每个物体使用 14 个标定 RGB views。训练使用
`gt_query_manifest.json`；expanded recovery 使用
`expanded_inference_manifest.json`。Expanded manifest 仅用于推理。

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

该测试使用由 48 views 经 COLMAP/OpenMVS 生成的 coarse meshes、原始 14 个
Sofa50 RGB views、三个 960 C2F2 checkpoints、480 分辨率 OpenGL visibility，且
不传递 GT differential。共评估 48 个 meshes；两个 coarse meshes 不存在。

| Recovery | Initial mean Chamfer | Ensemble refined mean Chamfer | 改善 mesh 数 | Introduced flips |
|---|---:|---:|---:|---:|
| 200 iterations | 0.0212023 | 0.0213199 | 2/48 | 4,692 |
| 1,000 iterations | 0.0212023 | 0.0213198 | 2/48 | 4,734 |

Recovery 从 200 增加到 1,000 iterations 未改变聚合结论。

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

### Native 1920 + high-frequency residual（运行中）

Native-1920 数据使用相同的 250 个 sample IDs、`200/25/25` split、28 个
camera extrinsics、current graph、proxy、raw target 和 visibility tensor。Intrinsics 按
native 1920 渲染缩放，不是 960 resize；native 与 resized 的最小 pixel MAE 为
`0.0205764`。

4×L40 job 15854 从零训练 20,000 steps。View chunk=4 和 gradient checkpointing
已通过 forward/gradient equivalence tests。实际 global batch 为 4，960 HF 基线为 2，
因此最终对比会标记为非严格单变量。8 月 15 日 18:47 快照的完整 checkpoint
为 step 900；step-500 validation loss 为 `6.34615e-5`，同期 960 HF 为
`9.61499e-5`。这一 early loss 差异不是 Top-10%/Top-1% 或 downstream 结论。
Jobs 15864/15865 将在训练后执行 paired evaluation 和 report merge。

### Future2000 GT-adaptive 扩展实验

当前扩展实验包含 2,000 个上游 meshes，每个物体生成 5 个确定性 current-mesh
variants，按物体执行 80/10/10 split（`8000/1000/1000` samples），使用 28 个标定
views、GT-adaptive subdivision、C2F2 和 current-graph direct-raw target。HPC 启动
参数把配置中的开发预算覆盖为 200,000 个 global optimizer steps，并使用四张 L40。

| Step | Rolling train loss | Validation loss |
|---:|---:|---:|
| 20,000 | 5.30099e-6 | 4.99851e-6 |
| 30,000 | 4.82400e-6 | **4.19731e-6** |
| 40,000 | 4.62000e-6 | **3.88000e-6** |
| 50,000 | 4.26000e-6 | 4.23000e-6 |
| 60,000 | 4.19000e-6 | 5.27000e-6 |
| 64,000 | **3.99000e-6** | — |

Job 15794 在 32,000 step 因文件描述符耗尽停止。替代 job 15795 使用
`file_system` sharing strategy 和 non-persistent workers 恢复，到 step 64,000 后因
DataLoader worker 耗尽 `/dev/shm` 失败（`Bus error` / `No space left on device`）。
Step-64k checkpoint 完整可恢复。配对 direct-displacement jobs 已取消，尚无最终
prediction 或 geometry 对比。

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

# Future2000 合同审计与四张 L40 current-graph 训练
sbatch scripts/HPC/audit_future2000_gt_adaptive_2000mesh.slurm
sbatch scripts/HPC/train_future2000_gt_adaptive_fast_io.slurm \
  configs/learned_laplacian/train_future2000_gt_adaptive_2000mesh_expanded_current_28view_direct_raw_20k.json \
  runs/learned_laplacian/future2000_gt_adaptive_2000mesh_expanded_current_28view_direct_raw_20k_seed7
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
`data_loading.multiprocessing_sharing_strategy`。Future2000 使用 `file_system`
和 non-persistent workers，避免 persistent workers 传递 tensor 时持续累积文件
描述符。外部方法对比由[本地任务说明](docs/FUTURE2000_LOCAL_COMPARISON_TASKS.md)
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
