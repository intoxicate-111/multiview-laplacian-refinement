# 多网格 GT-Query 训练指南

[English](MULTI_MESH_TRAINING.md) | [项目 README](../README.md)

## 项目目标

本文档说明当前正式训练流程。项目的唯一核心目标是从带标定信息的多视图图像中，
学习一个可跨物体泛化的局部 Laplacian 场：

```text
多视图 RGB + 3D query + 局部图上下文
    -> query 位置对应的 GT edge-scale-normalized Laplacian
```

网络在 GT mesh 的图结构上接受监督，最终要泛化到未见物体，以及任意
coarse/expanded inference mesh 的顶点。训练阶段不生成 coarse mesh，不学习
coarse-to-GT residual，也不把 GT Laplacian 插值到另一个图上。

这里的“从图像学习”并不表示网络只接收 RGB。3D query 和局部图上下文用于定义
预测位置与局部尺度，多视图 RGB 则提供跨物体恢复真实局部几何信号所需的观测证据。
必须通过 image ablation 验证模型确实使用了图像，而不是只记住坐标或图结构。

## 监督目标与 query 构造

对每个监督物体，数据准备直接在 GT graph 上计算 uniform Laplacian：

```text
raw_target_i        = (L_gt V_gt)_i
local_scale_i       = GT vertex i 的平均相邻边长
normalized_target_i = raw_target_i / (local_scale_i^2 + epsilon)
```

训练时动态生成 query position。默认保留 20% 未扰动的精确 GT vertices，其余顶点
按照局部边长施加有界的法向与切向小扰动。监督 target 始终是原 GT vertex 在
GT graph 上直接计算的 target，不随 query 扰动，也不来自 coarse mesh。

这个设计同时学习表面上的精确 query 和表面邻域中的局部 query 场。训练日志分别
记录 exact-query loss 与 perturbed-query loss。

## 防止 target 泄漏

以下约束必须始终成立：

- GT-query sample 中的 `initial_laplacian` 必须为零；
- raw/normalized GT Laplacian tensor 只能作为监督，不能作为输入特征；
- 不允许把 GT Laplacian vector 转移或插值到 coarse/expanded graph；
- 训练几何必须来自 GT vertices 和 GT faces；
- inference-only expanded sample 不得进入训练 dataset；
- test objects 只用于最终 held-out evaluation。

Trainer 会在使用 sample 前检查 zero-initial-Laplacian 约束。任何依赖 GT raw
Laplacian、oracle correspondence 或 coarse-to-GT target transfer 的结果，都不能
证明当前目标已经实现。

## 动态 Fourier 位置编码

数据准备只保存坐标归一化所需的 center 和 scale，不预计算 Fourier feature。
模型在 query 扰动完成后，对实际输入的 query 动态编码：

```text
q_normalized = (q - center) / scale
PE(q) = [q, sin(2^k pi q), cos(2^k pi q)]
```

正式 predictor 拼接以下特征：

- 投影到 query 后聚合得到的多视图 CNN feature；
- valid-view ratio；
- Fourier 编码后的 query coordinate；
- query graph vertex normal；
- 相对局部边长；
- graph degree。

`geometry_mode=query_fourier` 不包含 `initial_laplacian`。配置中的
`coarse_plus_multiview` 是历史遗留的 input-mode 名称；在正式 query-Fourier
模型中，它表示 query/graph context 加多视图图像，并不表示使用 coarse mesh
进行监督。

## 当前 Sofa50 数据契约

已检查的数据根目录为：

```text
/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/sofa_mesh/sofa50_refinement/multiview_960
```

数据包含 50 个物体，划分为 40 train、5 validation、5 held-out test。每个物体有
14 个带标定信息的 960 x 960 RGB 视图，mesh topology 和顶点数可以不同。

训练必须使用：

```text
/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/sofa_mesh/sofa50_refinement/multiview_960/gt_query_manifest.json
```

下面的 manifest 只用于 downstream inference evaluation：

```text
/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/sofa_mesh/sofa50_refinement/multiview_960/expanded_inference_manifest.json
```

Expanded manifest 中为了满足 schema 而存在的 target 不是 GT supervision。将它传给
训练循环会违反项目目标。

## Synthetic-current direct-raw 合同

28-view Sofa50 Arm B 使用 50 个物体、每个物体 5 个固定 current-mesh
variants，并按物体划分 `200/25/25` samples。与上述 GT-query 合同不同，
它直接预测 current-graph raw target：

```text
delta_target_raw = L_current @ P_proxy
target_mode = raw_laplacian
prediction_loss_space = output_representation
```

`target_scaling` 仍保留在公共 schema 中，因为数据准备同时保存
`local_edge_length`、`valid_scale_mask`、`raw_laplacian_target` 和并行的
normalized target。它不会将 Arm-B target 除以 `h^2`；只有
`target_mode=edge_scale_normalized_laplacian` 才选择 normalized representation。
`clip_max_norm=null` 时 direct raw target 也不会被裁剪。

Image encoder 支持三种受控 feature construction：

- `original`：采样 encoder output `F`；
- `gaussian`：使用固定 kernel/sigma 采样 `Gaussian(F)`；
- `original_plus_high_frequency`：采样 128-channel 拼接
  `[F, F-Gaussian(F)]`。

Native 1920 训练可以用固定 view chunk 运行 encoder，并对整个
encoder/feature-construction/sampling 路径执行 gradient checkpointing。这些开关只降低
activation memory，不改变数学输出或梯度；full/chunked 路径已通过等价测试。
它们不允许改变 optimizer-step budget，也不能隐藏 global-batch 差异。

## Recovery-aware 与 direct-vertex 训练模式

当前 `Sofa50MultiTopologyRawLap500_v2` 研究关闭 confidence head 与历史 recovery
gates。Laplacian Arms A-D 统一用以下公式评估：

```text
min_V ||L @ V - delta_pred||_F^2 + lambda ||V - V_input||_F^2
```

Arm A 只训练既有 raw-Laplacian Huber loss。Arms B-D 在训练中执行同一系统的
differentiable PCG，并加入：

```text
L_vertex = mean_i ||V_refine[i] - V_clean[i]||_2^2
L_total = L_lap + beta * L_vertex
```

`V_clean` 只保存在 loss-side prepared object，并从传给 model 的 mapping 中删除。
Solve 只能看到 `L`、`delta_pred` 与 `V_input`。Arm B 使用
`lambda=beta=1e-2`；C/D 保持 `beta=1e-2`，只把 lambda 改为 `1e-3/1e-4`。
Runtime diagnostics 记录 PCG iterations/residual/failures、gradient norms、NaN/Inf
和 peak GPU memory。

Arm E 复用相同 `N x 3` predictor head，但修改 target semantics：

```text
delta_v_target = V_clean - V_input
V_refined = V_input + delta_v_pred
L_E = mean_i ||delta_v_pred[i] - delta_v_target[i]||_2^2
```

E 不允许使用 Laplacian target/operator、sparse solver、lambda、visibility、
confidence、recovery Huber、Adam 或 post-processing。完整合同见
[A-E 研究说明](SOFA50_RECOVERY_AWARE_STUDY.zh-CN.md)。

## 启动完整训练

正式启动脚本为：

```bash
bash scripts/train_sofa50_v8_960_5000.sh
```

使用的配置为：

```text
configs/learned_laplacian/train_gt_query_sofa50_v8_960_5000.json
```

完整 run 输出到：

```text
runs/learned_laplacian/sofa50_refinement_960_gt_query_5000_full
```

当前长训练策略如下：

| 设置 | 数值 |
|---|---:|
| 最大 epochs | 5,000 |
| 最大 optimizer steps | 50,000 |
| 梯度累积 | 4 meshes |
| 每个完整 epoch 的 optimizer steps | 10 |
| Validation 间隔 | 5 epochs |
| Checkpoint 间隔 | 100 epochs |
| DataLoader workers | 4 |
| Prefetch factor | 2 |
| Pinned memory | 开启 |
| Persistent workers | 开启 |
| CUDA transfer prefetch | 开启 |
| CUDA AMP | FP16 开启 |
| 主 loss | Huber，delta 0.01 |

启动脚本使用 `test` Conda 环境并要求 CUDA 可用。训练开始前会检查 split 数量。

## Lazy 数据和精度路径

`PreparedMeshDataset` 保持 lazy，不会转换为 `tuple` 或 `list`。数据路径为：

```text
静态 GT graph 和监督 metadata
  -> lazy DataLoader worker
  -> 将本次请求的 RGB view 解码为 uint8
  -> pinned CPU tensor
  -> 在独立 prefetch stream 上执行 non-blocking CUDA transfer
  -> 在 GPU 上转换为 float、除以 255 并 normalization
  -> AMP CNN feature extraction 与 GNN prediction
  -> FP32 target scaling、Huber loss 和 metrics
```

系统只解码当前请求和预取的图像，不把完整图像数据集缓存到 CPU 或 GPU。每次前向
处理一个 ragged mesh，再跨 mesh 累积梯度，避免为不同拓扑构造大量 padding。
DataLoader workers 在 GPU 计算期间并行解码并 pin 后续 samples。在每个
accumulation group 内，mesh `i+1` 的传输与 mesh `i` 的 forward/backward 重叠。
该 CUDA overlap 要求 `pin_memory=true`、`cuda_prefetch=true`，并关闭 device cache。

## Validation 与模型选择

Validation 使用 held-out objects，绝不使用训练物体。Validation 启用 query
augmentation 时，总体曲线可能有噪声，因此除了 aggregate validation loss，还要
分别查看 exact-query 和 perturbed-query loss。

少数几次 validation 变差不足以判断训练失败。只有在一个 validation window 内，
下列两项都没有实质改善时，才应认为进入平台期：

- training loss 不再明显下降；
- validation best 不再明显改善。

最佳 checkpoint 按 validation loss 选择。周期 checkpoint 独立保留，便于之后在
同一训练阶段执行 image ablation 和 expanded-query evaluation。

## 必须完成的评估

仅有 GT-query validation 改善不能证明最终目标已经实现。每个候选 checkpoint
至少应报告：

- 相对 zero predictor 的 loss 与改善幅度；
- `mean |prediction| / mean |GT|`；
- 按 GT magnitude 分桶后的误差；
- GT magnitude high-10% 区域的 cosine similarity；
- 每个物体的独立指标，而不只是 aggregate mean。

验证图像依赖性时，必须固定 query、graph 和 target，只替换图像输入：

1. original RGB；
2. zero RGB；
3. shuffled view order；
4. cross-object RGB。

如果四组结果相近，说明模型主要依赖 query/graph context。如果 original RGB 明显
更好，但预测幅值仍接近零，则说明图像分支有效，下一步应检查 target distribution
和 loss calibration，而不是先大改网络。

Mesh-count scaling 应使用嵌套的 1/2/4/8/16-object subsets，并使每个物体获得的
训练 exposure 大致相同。这一实验用于定位物体多样性增加到什么规模时，输出开始
出现 amplitude collapse。

最后必须将同一个 checkpoint 应用于 expanded inference manifest，并报告
reconstruction Chamfer、normal consistency 和可视化结果。只有这一步才能检验模型
是否从 GT training graph 泛化到任意 inference graph。

## 诊断脚本

```bash
python scripts/ablate_single_mesh_checkpoint_images.py --help
python scripts/run_mesh_count_scaling.py --help
python scripts/diagnose_laplacian_prediction.py --help
python scripts/render_image_ablation_reconstructions.py --help
```

Magnitude-weighted Huber 是可选诊断实验，不是当前正式训练目标。因为 target
magnitude weighting 改变了优化指标，它的 loss 数值不能直接与 unweighted Huber
比较。

## 输出与监控

Run 目录包含：

- `best.pt`；
- `checkpoint_epoch_*.pt`；
- `config.json`、`run_config.json` 和 `dataset_manifest.json`；
- 训练完成后的 `training_history.json` 和 `metrics.json`；
- 最终评估生成的逐物体 prediction array；
- 用于实时监控的 launcher/service log。

查看当前训练日志：

```bash
tail -f runs/learned_laplacian/sofa50_refinement_960_gt_query_5000_full/training.log
```

Trainer 会报告 DataLoader wait、image decode、GPU transfer、forward/backward、
总 epoch 时间、validation 时间、实际使用的视角数、解码字节数，以及 CPU/GPU
峰值内存。

## 运行约束

- 训练仍然是每次前向一个 ragged mesh，再进行梯度累积，不是 packed-graph batching；
- 960 像素下 PNG decode 是当前稳态训练的主要耗时；
- Native 1920 F2+HF 使用 view chunk 和 gradient checkpointing 将完整 28-view
  feature 路径保持在 L40 显存内；每个 sample 仍使用全部 28 张 RGB；
- 静态 graph preparation 只运行一次，其耗时随 mesh 数量和复杂度增长；
- 支持显式 checkpoint resume，launcher 不会自动判断应恢复哪个 checkpoint；
- DDP 每个 rank 至少处理一个 mesh。四个 ranks 且 accumulation=1 时 global batch
  为 4，不修改 batching implementation 就无法复现两个 ranks 的 global batch 2；
- 数据集文件仍在生成或移动时不得启动训练；
- GT training observation 与 coarse/expanded inference query 必须保持一致的坐标和相机约定。

## 历史代码

历史 coarse-graph target、closest-surface pseudo target、oracle refinement 和
single-object Bunny 实验仍可作为测试或对照，但它们不是当前正式的
learned-Laplacian supervision contract，也不能作为跨物体或 expanded-query 泛化的
证据。

## 验证

```bash
PYTHONPATH=src conda run --no-capture-output -n test pytest -q
```

Learned-Laplacian 相关测试覆盖 lazy loading、GT-query leakage guard、query
perturbation bounds、Fourier encoding、image ablation、mesh-count scaling、AMP、
differentiable sparse recovery、direct-displacement semantics 和 Sofa50 preparation。
