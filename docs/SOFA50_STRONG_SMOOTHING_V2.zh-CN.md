# Sofa50 更强 coarse-mesh smoothing v2

状态日期：2026-08-23 BST。

历史 `legacy_v1` 多拓扑 Sofa50 coarse meshes 继续保留且可以复现，不会被覆盖。
新的准备流程默认使用版本化的 `strong_smooth_v2`，输出到
`Sofa50MultiTopologyRawLap500_v2`。

## 受控改动

唯一改变是最后的 uniform-Laplacian mesh smoothing。拓扑 recipe、法向/切向
扰动幅度、扰动场 smoothing、随机 seed namespace、28 views、960 分辨率、clean
reference 和 native raw-Laplacian target 全部保持不变。

| 档位 | legacy_v1 | strong_smooth_v2 | attenuation proxy：旧 -> 新 |
|---|---:|---:|---:|
| mild | 2 iterations × 0.08 | 6 × 0.12 | 0.1536 -> 0.5356 |
| strong | 4 × 0.12 | 10 × 0.15 | 0.4003 -> 0.8031 |
| unseen intermediate | 3 × 0.10 | 8 × 0.135 | 0.2710 -> 0.6866 |

Attenuation proxy 定义为 `1 - (1 - strength)^iterations`。它只用于审计 smoothing
预算，不表示每个 mesh 上的精确频谱衰减。

## 本地预检

在五个 held-out Sofa object 的原始拓扑上，保持相同扰动设置和 seed：

| 档位 | Mean displacement / bbox diagonal：旧 -> 新 | P95：旧 -> 新 | 同索引翻面比例：旧 -> 新 |
|---|---:|---:|---:|
| mild | 0.00273 -> 0.00909 | 0.01247 -> 0.04361 | 0.329% -> 1.511% |
| strong | 0.00686 -> 0.01503 | 0.03243 -> 0.06893 | 1.022% -> 2.674% |

另一次预检覆盖两个 held-out object 的全部十种 topology variants，共 20 cases：
没有 degenerate faces，在归一化 `1e-10` 阈值下没有 near-zero faces，最大同索引
翻面比例为 1.972%。这只是预检；开始训练前仍必须通过完整 500-sample v2 merge
audit。

## 安全性与启动状态

- v1 Slurm 脚本显式锁定 `legacy_v1`。
- v2 preparation、merge 和从零训练使用独立输出路径；profile 不匹配时拒绝 resume。
- 每个 sample 的 audit 新增 bbox-normalized displacement、实际 smoothing
  displacement、smoothing budget 和 invalidity 指标。
- Preparation `17077` 与 merge/full audit `17079` 已成功完成 500/500；
  `contract_audit=true`，全部 strong-smoothing budget 检查通过。
- 从零训练 job `17082` 已在 2×L40 上成功完成，每个 rank 累积四个 mesh，effective
  global batch 保持为 8，完整执行 20,000 optimizer steps。总训练时间为 15.279
  小时；final train loss 为 `1.83395e-6`，step 20,000 的 best/final selection
  validation loss 为 `2.26915e-6`。
- Jobs `17110`–`17113` 已在 v2 prepared meshes 上完成受控 v1-v2
  validation/test 与 downstream recovery。两个 20k checkpoints 使用相同 samples
  和 initial meshes；primary geometry 使用共享 prepared 坐标系、无 ICP/测试时
  alignment 的统一 area-weighted surface evaluator 重新计算。
- 被替代的 8×Blackwell job `17080` 和 4×L40 job `17081` 均在启动前取消，运行
  时间为零。

## 受控 test 与 downstream recovery

Contract audit：**true**。两个 20k 模型都在相同 50 个 v2 strong-smoothing test
meshes 上评估，因此 initial mesh、vertex ordering、target、confidence/visibility
inputs 与 recovery solver 均逐样本配对。Primary geometry 使用共享 prepared
坐标系中的 area-weighted triangle-surface sampling 和 exact bidirectional
point-to-triangle-surface distance；不使用 ICP 或测试时 alignment。

| 指标 | v1 model on v2 inputs | v2 strong-smoothing model |
|---|---:|---:|
| Raw EPE | 0.00840367 | **0.00276820** |
| Raw RMS | 0.0234761 | **0.00843035** |
| Top-10% EPE | 0.0447991 | **0.00812695** |
| Top-1% EPE | 0.143090 | **0.0206836** |
| 统一 refined Chamfer | **0.00426879** | 0.00451747 |
| 统一 P2S | **0.00426879** | 0.00451747 |
| Normal consistency | **0.960320** | 0.952386 |
| Introduced flips | **12,813** | 46,339 |
| 相对 common initial 改善 | **38/50** | 26/50 |

Common initial Chamfer 为 `0.00438635`。V1 model 虽然 raw prediction error
明显更大，但平均 geometry 小幅改善；v2 虽显著改善 raw EPE 和 high-curvature
tail，平均 geometry 却略微恶化。这直接表明在更强 smoothing 下，prediction
improvement 没有通过冻结 recovery configuration 传递；本实验不引入其他 recovery
方法。
