# 多网格优化训练指南

[English](MULTI_MESH_TRAINING.md) | [项目 README](../README.md)

本文档说明现有 CNN + 图网络的优化训练路径。模型结构没有改变，也没有实现
稀疏 vertex-view patch。

## 当前正式数据集

已检查的本地正式 manifest 为：

```text
/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/thingi10k50/sample_50_960/prepared_manifest.json
```

数据契约如下：

- 共 50 个 prepared mesh：40 train、5 validation、5 test；
- 每个 mesh 有 14 个视角；
- prepared 图像尺寸为 960 x 960；
- 所有样本均使用 `lazy_image_paths_v1` 存储；
- 不同样本允许拥有不同的网格拓扑和规模。

正式配置会在创建 run 之前校验这些 split 数量：

```text
configs/learned_laplacian/train_multi_mesh_edge_normalized_50_960.json
```

## 一条命令启动

可以从任意目录运行：

```bash
bash /home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/multiview-laplacian-refinement/scripts/train_thingi10k50_960_full.sh
```

启动脚本会：

1. 激活 `test` Conda 环境；
2. 检查 manifest 和 JSON 配置；
3. 确认 CUDA 设备可用；
4. 拒绝覆盖非空的输出目录；
5. 启动训练，并将终端输出同步写入 `console.log`。

固定输出目录为：

```text
runs/learned_laplacian/thingi10k50_960_full
```

只检查启动条件、不开始训练：

```bash
bash scripts/train_thingi10k50_960_full.sh --check
```

长时间前台运行时应保持终端开启，或者在 `tmux` 中启动。当前 trainer
尚不支持从中断的 checkpoint 自动续训；非空目录保护用于避免两个 run
意外混合。

## 正式训练策略

50-mesh、960px 配置使用以下参数：

| 设置 | 数值 |
|---|---:|
| 最大 epochs | 5,000 |
| 最大 optimizer steps | 50,000 |
| 梯度累积 | 4 meshes |
| 每个完整 epoch 的 optimizer steps | 10 |
| Validation 间隔 | 5 epochs |
| Checkpoint 间隔 | 10 epochs |
| Early-stopping patience | 15 次 validation |
| Early-stopping minimum delta | 0.0001 |
| DataLoader workers | 4 |
| Prefetch factor | 2 |
| Pinned memory | 开启 |
| Persistent workers | 开启 |
| CUDA AMP | FP16 开启 |

训练会在任一终止条件满足时结束：达到最大 epochs、达到最大 optimizer
steps，或者 early stopping。Validation 每 5 epochs 执行一次，因此 15 次
validation patience 对应连续 75 epochs 没有足够改善。ReduceLROnPlateau
scheduler 按 validation 次数而不是原始 epoch 数计数。

## Lazy 数据和精度路径

命令行入口直接把 `PreparedMeshDataset` 传给 trainer，不再把 dataset 转换成
`tuple` 或 `list`。

数据路径为：

```text
prepared 静态 mesh tensors
  -> lazy DataLoader worker
  -> 将当前图像解码为 uint8
  -> pinned CPU memory
  -> non-blocking CUDA transfer
  -> GPU 上转换 FP32、除以 255，并执行配置的 normalization
  -> FP16 autocast CNN + 图网络前向
  -> FP32 Laplacian target、robust loss 和 metrics
```

只有当前请求和预取的图像会被解码。系统不会缓存整个数据集的图像，因此
数据集数量不会直接乘到 GPU 显存上。所有 train/validation mesh 的静态图
结构和 target tensor 仍会在启动时准备一次。

配置中的图像 normalization 在 `[0,1]` 缩放后保持恒等：

```json
{
  "mean": [0.0, 0.0, 0.0],
  "std": [1.0, 1.0, 1.0]
}
```

这保持了原始训练路径的输入语义。由于图像编码器从头训练，因此没有使用
ImageNet normalization。

## 可用配置

| 配置 | 用途 |
|---|---|
| `train_multi_mesh_edge_normalized_50_960.json` | 完整 40/5/5 正式训练 |
| `train_multi_mesh_edge_normalized_960_epoch1.json` | 960px、1 epoch CUDA smoke test |
| `train_multi_mesh_edge_normalized_1920_epoch1.json` | 1920px、1 epoch CUDA smoke test |
| `train_multi_mesh_edge_normalized_1000_1920.json` | 800/100/100、250 epochs、50k steps |

实际图像尺寸记录在 prepared sample 中。配置文件名表示预期使用的数据 profile，
loader 则从每个样本读取 `prepared_image_size`。

1000-sample 配置会拒绝不是严格 800 train、100 validation、100 test 的
manifest。Test split 保留用于最终 held-out evaluation，不会进入训练循环。

## 数据加载控制

Lazy sample 进入 DataLoader worker 前会先裁剪。保留 forward 字段、相机 tensor、
confidence、局部尺度和选定的 training target；GT mesh、faces、target positions、
重复的 raw/normalized targets、`local_edge_scale` 和 metadata 不再进入 worker IPC
或 GPU。Raw target 和 face count 只保留在主进程，并仅在 validation 和最终预测指标
阶段显式关联。

可在 `data_loading` 中配置视角采样：

```json
{
  "train_views_per_sample": null,
  "validation_views_per_sample": null
}
```

`null` 保持原有全部视角语义。正整数会使用同一组索引同步选择 image paths、
intrinsics、extrinsics 和 visibility。训练视角由 seed、sample ID 和 epoch 决定，
不同 epoch 可变化且可复现；validation 视角固定且可复现。请求数不小于已有视角数
时使用全部视角。
CLI 可通过 `--train-views-per-sample 4` 和
`--validation-views-per-sample 4` 覆盖配置，无需直接修改源配置文件。

`coarse_only` 和 `--zero-images` 都不会打开或 resize 图像。`coarse_only` 还会省略
相机 tensor，因为该 ablation 的图像特征与 valid-view ratio 都为零；
`--zero-images` 仍保留相机投影，以维持历史 valid-view-ratio 输入语义。

启用 profiling 后，每个 epoch 会记录 `sample_wait_seconds`、worker 内部
`image_decode_resize_seconds`、`pin_or_transfer_seconds`、
`forward_backward_seconds`、平均实际视角数和 uint8 解码字节数。由于 DataLoader
等待、prefetch 与 IPC 无法可靠拆分，因此不单独伪造 worker-to-main 时间。

## 输出与监控

训练期间每个 epoch 会输出：

```text
epoch、train loss、validation loss、best loss、learning rate
DataLoader wait、GPU transfer、forward/backward、total step、validation time
```

Run 目录包含：

- `console.log`：启动脚本的实时输出；
- `best.pt`：最佳 validation checkpoint；
- `checkpoint_epoch_*.pt`：定期 checkpoint；
- `training_history.json`：epoch loss、learning rate、steps 和 timing；
- `metrics.json`：最终 loss、停止原因、性能和逐对象指标；
- `config.json`、`run_config.json`、`dataset_manifest.json`：复现实验所需元数据；
- `predictions/train/` 和 `predictions/validation/`：target-space 与 recovered raw prediction。

查看正在运行的日志：

```bash
tail -f runs/learned_laplacian/thingi10k50_960_full/console.log
```

`metrics.json` 至少记录：

- 初始/静态准备时间；
- 平均 DataLoader 等待时间；
- 平均 GPU transfer 时间；
- 平均 forward/backward 时间；
- 平均 optimizer step 总时间；
- validation 时间；
- GPU 峰值分配显存；
- 主进程 CPU 峰值内存；
- 完成 epochs、optimizer steps、AMP 状态和停止原因。

CPU 峰值是主进程的 high-water mark，不是所有 persistent worker RSS 的
严格总和。

## 实测性能

测试使用相同的 40/5/5 数据契约和 Quadro RTX 5000：

| 指标 | 优化后 1920 | 优化后 960 |
|---|---:|---:|
| 单个训练 epoch | 10.85 s | 4.78 s |
| Validation pass | 2.24 s | 1.06 s |
| DataLoader wait | 5.45 s | 2.52 s |
| GPU transfer | 1.89 s | 0.54 s |
| Forward/backward | 3.17 s | 1.52 s |
| GPU 峰值分配 | 3.01 GiB | 1.00 GiB |
| 主进程 CPU 峰值 | 5.06 GiB | 2.83 GiB |
| 完整 1-epoch smoke runtime | 48.86 s | 33.94 s |

优化后的 960 稳态训练 epoch 比优化后的 1920 约快 2.27 倍。与原始 eager
1920 路径相比，完整 1920 smoke runtime 从 187.96 秒下降到 48.86 秒。

960 与 1920 的 1-epoch loss 几乎一致，但 1 epoch 不足以证明最终精度完全
相同。在把 960 视为精度等价方案之前，应执行更长的受控 A/B 训练。

## Loss 解读

Target 使用 edge-scale-normalized Laplacian coordinates，Huber loss 的
`delta=0.01`。Normalized target magnitude 是重尾分布，因此即使 checkpoint
持续改善，loss 数字也可能下降较慢。在当前 960 数据集上，全零预测基线约为
train 0.281828、validation 0.305586。正式训练在 epoch 40 已达到约
0.298575 validation loss，说明模型已经明显低于零预测基线。

不要在正在运行的实验中途修改 target clipping、Huber delta 或 target
standardization。应当使用新的输出目录启动新实验来比较这些选择。

## 验证

激活相同环境后运行：

```bash
source /home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/miniconda3/etc/profile.d/conda.sh
conda activate test
PYTHONPATH=src pytest -q
```

优化实现通过了 108 项测试，包括 lazy manifest loading、CPU uint8 图像、
persistent workers、max-step stopping、early stopping、按 epoch 对齐视角采样、
无图像 ablation、CUDA transfer 路径和有限 CUDA AMP loss。

## 当前限制

- 训练仍然是每次前向一个 ragged mesh，再做梯度累积，不是 packed-graph batching。
- 960px 下 PNG 解码仍是稳态训练的最大耗时部分。
- 静态 mesh/graph preparation 会随 train/validation mesh 数量和复杂度增长，
  但只在启动时执行一次。
- Trainer 会在训练结束后再次评估全部 train 和 validation 以写入最终指标。
- 尚未实现自动 checkpoint resume。
- 1000-sample profile 已准备，但必须先提供真实匹配的 800/100/100 prepared
  manifest 才能启动。
