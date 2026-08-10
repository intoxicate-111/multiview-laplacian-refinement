# 多视图 Laplacian 网格细化

[English](README.md) | [简体中文](README.zh-CN.md)

方法定义：[Sofa50 标准流程](docs/CANONICAL_SOFA50_PIPELINE.md)

训练说明：[多网格 GT-query 训练](docs/MULTI_MESH_TRAINING.zh-CN.md)

可见性与恢复：[可见性感知恢复报告](docs/VISIBILITY_AWARE_RECOVERY_REPORT.md)

## 项目状态

状态日期：2026-08-10。

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
| 14/28/56-view 消融 | 数据准备契约已实现 | 数据准备为每个 view count 写出 GT-query training manifest、expanded-inference manifest 和对应 graph 的 `visibility_backface_and_occlusion`。尚未生成 checkpoint 或结果报告。 |
| Query-graph resolution 消融 | 数据准备契约已实现 | 数据准备写出 GT、GT-sub1、GT-sub2 和 adaptive represented-vertex-area manifests。尚未生成 checkpoint 或结果报告。 |
| 自动化测试 | 通过 | `test` Conda 环境中为 `216 passed, 3 skipped`。 |

当前模型能够在 GT-query graph 上学习监督微分场，并使用 RGB 信息。从 GT-query
graph 到 expanded 或 OpenMVS query graph 的迁移未产生几何改善。当前 recovery
流程未达到端到端 coarse-mesh refinement 目标。

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
```

14/28/56-view 数据准备会分别写出 14、28 和 56 views 的 GT-query 与
expanded-inference manifests。每个 prepared graph 包含 renderer-native
`visibility_backface_and_occlusion`。Training job 直接读取 GT-query manifests，
不需要额外执行 visibility attach。

## HPC 结果目录

```text
runs/learned_laplacian/sofa50_image_resolution_ablation_50000step
runs/learned_laplacian/sofa50_c2_f2_50000step_3seed
runs/learned_laplacian/sofa50_c2_f2_1920_20000step_3seed
runs/learned_laplacian/sofa50_cf_c2f2_comparison_full
runs/learned_laplacian/sofa50_c2f2_960_vs_1920_full
runs/learned_laplacian/sofa50_openmvs_coarse_14view_c2f2_48mesh_opengl_480
runs/learned_laplacian/sofa50_openmvs_coarse_14view_c2f2_48mesh_opengl_480_recovery1000
```

Source repository 不包含 checkpoints、prepared datasets 和 HPC result
directories。
