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
| 14/28/56-view 消融 | 训练前阻断 | Prepared sample 缺少 `visibility_backface_and_occlusion`；expanded-inference manifest 不存在。未生成 checkpoint 或结果报告。 |
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

当前数据契约下不能执行 14/28/56-view job。该任务需要 GT-query sample 中的
renderer-visibility fields，以及对应的 expanded-inference manifests。

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
