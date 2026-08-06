# 多视图 GT Laplacian 学习

[English](README.md) | [简体中文](README.zh-CN.md)

训练指南：[English](docs/MULTI_MESH_TRAINING.md) |
[简体中文](docs/MULTI_MESH_TRAINING.zh-CN.md)

## 项目目标

Learned-Laplacian pipeline 只有一个核心目标：

```text
多视图 RGB + 标定相机 + 3D query position + 局部图上下文
    -> 该 3D 位置对应的 GT 局部 Laplacian signal
```

训练使用 GT mesh，因为 GT mesh 提供了希望网络学习的监督场。训练阶段**不生成
coarse mesh**，也不学习 coarse-to-GT correction。推理时，在任意输入 mesh 的
顶点上查询这个学习到的场，包括未见物体的 coarse mesh 或经过 topology expansion
的 mesh。最终目标是泛化到 held-out objects 和非 GT query graphs。

当前 target 是 GT mesh 的 edge-scale-normalized uniform Laplacian。对于 GT vertex
`i`：

```text
delta_gt_i = (L_gt V_gt)_i
h_i        = vertex i 的平均相邻 GT 边长
target_i   = delta_gt_i / (h_i^2 + epsilon)
```

网络首先预测 normalized target。Reconstruction 使用 query graph 的局部尺度将其
恢复为 raw Laplacian coordinates，然后求解现有的 Laplacian reconstruction 问题。

## 训练契约

对每个训练物体：

1. 从 GT mesh 渲染带标定信息的多视图 RGB；
2. 使用 GT vertices 和 GT connectivity 构建训练 query graph；
3. 保留一部分未扰动的精确 GT query positions；
4. 按照局部边长 `h_i` 对其余 query 加入小尺度法向和切向扰动；
5. target 始终绑定到对应的原始 GT vertex；
6. 预测该点的 edge-scale-normalized GT Laplacian。

形式化表示为：

```text
q_i = V_gt_i + small_normal_offset_i + small_tangent_offset_i

F(images, cameras, q_i, normal_i, h_i, graph)
    ~= edge_scale_normalized_laplacian_gt_i
```

Query 扰动让模型学习表面附近的局部 3D query field，而不是只能在精确 GT
coordinates 上工作的 lookup table。训练分别记录 exact-query loss 和
perturbed-query loss。

## 防止 target 泄漏

GT Laplacian vector 只能作为监督，绝不能复制到模型输入中。

- GT-query sample 中的 `initial_laplacian` 为零；
- raw 或 normalized GT Laplacian 不能作为输入特征；
- 不允许把 GT Laplacian 插值到 coarse 或 expanded graph；
- inference-only expanded sample 可能包含满足 schema 所需的 placeholder，但它们
  不是 training target 或 oracle supervision；
- 输入的 normal、局部边长、degree、query coordinate、graph 和 image feature
  提供预测上下文，但其中任何一个都不是 target 本身。

## Query 位置编码

Fourier encoding 在 query augmentation 之后由模型动态计算：

```text
query position
  -> 使用每个物体的 center 和 scale 归一化
  -> [q, sin(2^k pi q), cos(2^k pi q)]
  -> 拼接 image feature、normal、相对局部边长和 degree
  -> graph predictor
```

Fourier feature 不会在数据准备阶段预计算。编码后的坐标必须对应实际扰动后的训练
query，或实际 coarse/expanded inference query。

正式 geometry mode 是 `query_fourier`。历史 CLI 参数
`coarse_plus_multiview` 在该模式中表示“query geometry context 加多视图特征”；
它不表示训练阶段会构造 coarse mesh，也不表示会把 coarse mesh 的 raw
Laplacian 输入 predictor。

## 推理契约

推理与有监督的 GT-query training 是两个不同阶段：

```text
多视图观测
  -> 获得任意 initial/coarse mesh
  -> 可选 topology expansion
  -> 将其 vertices 作为 3D queries
  -> 把每个 query 投影到所有标定视图
  -> 聚合 CNN features
  -> 使用 Fourier query encoding 和 graph context
  -> 预测 normalized Laplacian
  -> 恢复 query graph 的 raw Laplacian
  -> Laplacian reconstruction
```

这条推理路径不使用也不要求 GT mesh。推理时的位置归一化必须来自 observation/query
coordinate frame，不能依赖隐藏的 GT mesh。

## 当前 Sofa50 数据集

当前完整数据集为：

```text
/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/sofa_mesh/sofa50_refinement/multiview_960
```

数据集包含：

- 40 个 train、5 个 validation 和 5 个 held-out test objects；
- 每个物体 14 个带标定信息的 960 x 960 RGB views；
- 用于有监督训练的 lazy GT-query samples；
- 用于 inference evaluation 的独立 expanded-query samples；
- 不同物体允许使用不同 topology 和 mesh size。

训练 manifest：

```text
.../multiview_960/gt_query_manifest.json
```

仅用于推理的 expanded manifest：

```text
.../multiview_960/expanded_inference_manifest.json
```

不要把 expanded inference manifest 传给训练循环。

## 完整训练

安装 package 和训练依赖后运行：

```bash
pip install -e ".[train]"
bash scripts/train_sofa50_v8_960_5000.sh
```

启动脚本使用：

```text
configs/learned_laplacian/train_gt_query_sofa50_v8_960_5000.json
```

该配置使用 CUDA AMP、4 个 lazy DataLoader workers、pinned memory、non-blocking
transfer、跨 4 个 meshes 的 gradient accumulation、每 5 epochs validation，以及
周期 checkpoint。训练上限为 5,000 epochs 和 50,000 optimizer steps。

当前完整输出目录为：

```text
runs/learned_laplacian/sofa50_refinement_960_gt_query_5000_full
```

## 什么才算有效证据

GT-query validation loss 降低是必要条件，但不足以证明最终目标已经实现。有效的
checkpoint 必须通过以下检查：

1. **有效学习：** train 和 held-out validation loss 都优于 zero predictor；
2. **依赖图像：** 固定 query graph 和 target 时，original RGB 优于 zero RGB、
   shuffled views 和 cross-object RGB；
3. **预测幅值未塌缩：** `mean |prediction| / mean |GT|` 不接近零，尤其是在
   high-magnitude regions；
4. **方向准确：** high-magnitude target regions 的 cosine similarity 为正且持续改善；
5. **跨物体泛化：** held-out objects 得到改善，而不只是 training meshes；
6. **Expanded-query 迁移：** 同一个 checkpoint 能用于从未作为 GT training graph
   出现的真实 coarse/expanded queries；
7. **重建有效：** predicted-Laplacian reconstruction 相比 initial mesh 改善
   Chamfer 和 normal consistency。

Single-mesh overfitting 只能证明模型容量足够。GT-query validation 只能证明监督场
可以学习。两者都不能单独证明 expanded-query reconstruction 有效。

## 诊断命令

Image ablation 和 mesh-count scaling 工具位于 `scripts/`：

```bash
python scripts/ablate_single_mesh_checkpoint_images.py --help
python scripts/run_mesh_count_scaling.py --help
python scripts/diagnose_laplacian_prediction.py --help
```

支持的图像条件包括 original RGB、zero RGB、shuffled view order 和 cross-object
RGB。Mesh-count scaling 使用嵌套的 1/2/4/8/16-object sets，并报告 zero-predictor
baseline、prediction/target amplitude ratio、high-10% cosine 和逐物体指标。

## 数据和精度路径

Prepared RGB 以 lazy 形式保留在磁盘上。Worker 只将当前请求的 views 解码为
`uint8`；pinned CPU tensor 通过 non-blocking transfer 传到 CUDA，在 GPU 上转换为
浮点数、除以 255 并进行 normalization。CNN 和 GNN forward 使用 AMP。Target
scaling、robust loss 和数值几何操作保留 FP32。

Trainer 会记录 DataLoader wait、image decode、GPU transfer、forward/backward、
总 epoch 时间、validation 时间，以及 CPU/GPU memory。

## 历史基线

仓库仍包含 coarse-mesh generator、oracle refinement、pseudo-surface experiment、
single-object Bunny experiment 和 reconstruction solver。它们是有用的基线和调试
工具，但不定义当前 learned model 的 supervision contract。

特别是，历史 coarse-graph target 或 closest-point pseudo target 不能描述为正式的
learned-Laplacian target。正式训练使用直接 GT-query supervision；coarse/expanded
mesh 只在 downstream inference 和 evaluation 阶段作为 query 出现。

Renderer-native visibility 与 hard any-view Laplacian recovery 的定义、实验结果和
复现命令见[可见性感知恢复报告](docs/VISIBILITY_AWARE_RECOVERY_REPORT.md)。该功能
无需 depth image，也不修改网络；它会从 recovery 的 Laplacian objective 中移除
所有 view 均不可见 vertex 自身的预测方程。

## 测试

```bash
PYTHONPATH=src conda run --no-capture-output -n test pytest -q
```

相关测试覆盖 query perturbation bounds、zero initial-Laplacian leakage protection、
Fourier query encoding、lazy image loading、AMP training、image ablation、mesh-count
scaling 和 Sofa50 preparation。
