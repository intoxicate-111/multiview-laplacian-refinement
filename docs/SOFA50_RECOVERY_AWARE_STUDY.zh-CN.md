# Sofa50 v2 recovery-aware 训练研究

状态日期：2026-08-24 BST。

本文记录 `Sofa50MultiTopologyRawLap500_v2` 上当前 matched-domain 研究。它把
全方程 regularized sparse integration 作为当前实验 recovery 主线；历史
visibility/confidence/Adam recovery 仍可复现，但不覆盖任何冻结 benchmark 结果。

## 为什么修改 recovery 主线

Exact native target 为

$$
\delta^*=L V_{\mathrm{clean}},
$$

其中 `L` 是 current graph 上 row-normalized uniform Laplacian，`V_clean` 与 input
mesh 保持相同 vertex ordering。仅使用 component centroid 固定 translation
nullspace 的 direct sparse least-squares，在 `legacy_v1` 和 `strong_smooth_v2` 上的
mean oracle efficiency 分别达到 `0.94293` 和 `0.92366`。因此 representation 本身
不是 exact-target recovery 的主要 ceiling。

在冻结 v2 recovery 中，`lambda_anchor=0.01` 后加入 hard any-view visibility，会
把 mean efficiency 从 `0.34258` 降到 `0.16875`，并令 44/50 samples 变差；confidence
影响接近数值噪声。Production sparse route 实际执行 L2，而不是配置中声明的 recovery
Huber；Adam 从 200 增加到 2,000 steps，也只把 mean v2 efficiency 从 `0.16876`
变为 `0.18635`。

因此当前研究使用所有 Laplacian equations 的 regularized sparse solve：

$$
\widehat V_\lambda=
\arg\min_V
\left\lVert L V-\widehat\delta\right\rVert_F^2+
\lambda\left\lVert V-V_{\mathrm{input}}\right\rVert_F^2.
$$

变量说明：`V in R^(N x 3)` 是待求 refined mesh，`V_input in R^(N x 3)` 是 initial
mesh，`L in R^(N x N)` 由其 connectivity 固定，`delta_hat in R^(N x 3)` 是 predicted
raw Laplacian，`lambda>0` 是 positional regularization coefficient。该求解使用全部
equation rows，不使用 visibility、confidence、recovery Huber 或 Adam。

等价 normal equations 为

$$
(L^\top L+\lambda I)\widehat V_\lambda
=L^\top\widehat\delta+\lambda V_{\mathrm{input}}.
$$

变量说明：`L^T` 是 `L` 的 transpose，`I` 是 `N x N` identity。Standalone
evaluation 使用 sparse LSMR/LSQR-equivalent least squares；训练使用同一系统的
differentiable sparse PCG 实现。

## 已完成 Arm A 与 Arm B

两组使用相同 400/50/50 split、28 个 native-960 views、C2F2+HF、826,115 个参数、
seed 7、effective global batch 8 和 20,000 optimizer steps；confidence head 关闭。

Arm A 只训练 direct raw-Laplacian Huber objective：

$$
\mathcal L_A=\mathcal L_{\mathrm{lap}}.
$$

Arm B 使用 `lambda=10^-2` integration，并加入 same-index vertex supervision：

$$
\mathcal L_{\mathrm{vertex}}=
\frac{1}{N}\sum_{i=1}^{N}
\left\lVert \widehat V_{\lambda,i}-V_{\mathrm{clean},i}\right\rVert_2^2,
\qquad
\mathcal L_B=\mathcal L_{\mathrm{lap}}+\beta\mathcal L_{\mathrm{vertex}},
\quad \beta=10^{-2}.
$$

变量说明：`V_hat_lambda,i` 与 `V_clean,i` 分别是 vertex `i` 的 recovered/clean 3D
position；`L_vertex` 是 squared 3D Euclidean norm 的 vertex mean，而不是互不相关
coordinate samples 的均值；`beta` 是 geometry supervision 权重。Clean vertices
只存在于 loss side，不进入 model input 或 sparse solve。

| Test metric | Arm A：仅 Lap | Arm B：Lap + vertex |
|---|---:|---:|
| Raw EPE | **0.00252641** | 0.00263986 |
| Raw RMS | 0.00737725 | **0.00683290** |
| Top-10% EPE | 0.00751175 | **0.00737282** |
| Top-1% EPE | 0.0182152 | **0.0159263** |
| Refined Chamfer | 0.00395529 | **0.00358497** |
| Relative Chamfer gain | 7.21% | **13.04%** |
| Mean recovery efficiency | 0.07206 | **0.13036** |
| P2S p95 | 0.0122582 | **0.0105581** |
| F-score | 0.917435 | **0.935013** |
| Normal consistency | 0.954902 | **0.959366** |
| Introduced flips | 53,838 | **52,338** |
| Recovered vertex RMS | 0.0135181 | **0.0115532** |

Arm B 在 32/50 test samples 上有更低 Chamfer，在 43/50 上有更低 recovered vertex
RMS，但只有 10/50 的 raw EPE 更低。因此已完成证据支持一个窄结论：recovery-aware
supervision 改善 predicted differential field 的 geometric utility；它并不意味着所有
raw prediction metrics 都改善。

Strict hardware/sharding contract 为 false，因为已完成 A/B 都在 epoch boundary 从
2×L40 转到 8×RTX PRO 6000 Blackwell；effective global batch、optimizer-step budget
与 executable mathematical contract 保持不变。

## 运行中的 lambda extension：Arm C 与 Arm D

C/D 保持 Arm B 不变，只修改 differentiable/evaluation regularization：

| Arm | `lambda` | `beta` | 2026-08-24 状态快照 |
|---|---:|---:|---|
| C | `10^-3` | `10^-2` | Job `17274` 在 8×Blackwell 上运行；step 3,200/20,000，failed solves 与 NaN/Inf 均为 0。 |
| D | `10^-4` | `10^-2` | Job `17275` 依赖 C 排队。 |

PCG tolerance 仍为 `10^-4`。Preflight 发现较小 lambda 下 float32 stagnation，因此
C/D 使用 float64 PCG、最多 2,048 iterations。这是已记录的 numerical execution
change；lambda 和 objective 均未被静默修改。Dependent validation/test evaluation 与
report merge 完成前，不得给出 C/D 科学结论。

## Direct-vertex 对照：Arm E

Arm E 保留相同 C2F2+HF encoder、graph network、`N x 3` output width 和 826,115 个
参数，但将 output semantics 改为 residual vertex displacement：

$$
\Delta V_{\mathrm{pred}}=
f_\theta(I_{1:M},K_{1:M},E_{1:M},V_{\mathrm{input}},F),
\qquad
V_{\mathrm{refined}}=V_{\mathrm{input}}+\Delta V_{\mathrm{pred}}.
$$

变量说明：`f_theta` 是共享 predictor family；`I`、`K`、`E` 分别是 28 个 RGB
views、intrinsics 与 extrinsics；`F` 是 input connectivity；
`Delta V_pred in R^(N x 3)` 是 direct displacement。Arm E 不使用 `L`、Laplacian
target、sparse integration、PCG/LSMR、lambda、visibility、confidence、recovery
Huber、Adam 或 post-processing。

它的 target 与 loss 为

$$
\Delta V^*=V_{\mathrm{clean}}-V_{\mathrm{input}},
\qquad
\mathcal L_E=\frac{1}{N}\sum_{i=1}^{N}
\left\lVert\Delta V_{\mathrm{pred},i}-\Delta V_i^*\right\rVert_2^2.
$$

变量说明：`Delta V*` 是 exact same-index clean displacement，`L_E` 是 mean squared
3D displacement error；GT vertices 仅用于 loss。500 个 prepared samples 的
implementation audit 已通过。Job `17278` 在 D 后排队，evaluation `17279`、A-E
merge `17280` 与 matched visualization `17281` 均由 dependency gate 控制。在这些
任务完成前，H1（vertex loss 足以解释提升）与 H2（differential representation +
structured integration 提供额外 inductive bias）仍未判定。

## 判定边界

- Raw EPE/RMS/tail 只用于预测 Laplacian 的 A-D，Arm E 不报告 raw Laplacian EPE。
- Recovery-aware Laplacian arm 必须按 validation recovered geometry 选择，再冻结后
  解释 test。
- Arm E 必须分别与 B、以及 validation-selected B/C/D 最优组做 paired Chamfer、
  vertex RMS、P2S p95、F-score、normal 和 flips 对比。
- 不得根据中间结果自动启动 2,000-mesh `strong_smooth_v2` scaling。
