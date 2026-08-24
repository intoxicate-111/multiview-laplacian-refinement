# Sofa50 28-view Local Query Jitter 消融报告

状态时间：2026-08-12 10:03 BST

## 1. 实验问题

本实验比较训练时使用 stored current vertex position 与在其邻域内加入 local query-position jitter。主要判定端点为 best validation loss、synthetic-current test raw endpoint、raw Top-10% endpoint、raw Top-1% endpoint、raw cosine 和 runtime。OpenMVS48 current-mesh recovery 仅保留为低质量 OOD 输入压力测试，不参与模型选择或目标质量判断。

## 2. 实验契约

| 条目 | 设定 |
|---|---|
| Model | C2F2 |
| Views | 28 |
| Dataset | Sofa50 synthetic current-graph |
| Split | 200/25/25 train/validation/test samples |
| Stored current variants | 每个 GT object 5 个 |
| Seed | 7 |
| Budget | 每个 arm 20,000 optimizer steps |
| Validation/test jitter | 关闭 |
| Arm A | stored current vertex position |
| Arm B | `q_i = c_i + eta_i` |
| Arm B jitter | `eta_i = h_i * clip_l2(N(0, 0.003^2 I), 0.009)` |

两个 arm 复用 proxy、normalized target、raw target、`h_current`、graph connectivity、Laplacian operator、RGB、cameras、visibility、confidence branch 和 recovery settings。配置比较仅允许 local jitter enablement 与 arm label 不同。

审计记录：

- 两个 arm 的 seeded initial parameter SHA-256 均为 `2463bbf7e31d635d48808c011d91a8e625a1fb5598d17632cf9c0c733a0ef616`。
- 两个 run 内 dataset manifest SHA-256 均为 `d6ba842741a3fb2921e1ebdcbdb80ef5b12e3985c3ff7b8ddcdcd4f9ff84c674`。
- Runtime jitter 的测试覆盖跨 epoch position 变化、固定 seed/epoch 可重复、target/graph/`h_current`/proxy 不变和 `0.009 h` offset 上限。

## 3. 作业终态

| Job | Arm/阶段 | State | Exit code | Elapsed |
|---:|---|---|---|---:|
| 15662_0 | A: no jitter training | COMPLETED | 0:0 | 06:02:49 |
| 15662_1 | B: local jitter training | COMPLETED | 0:0 | 06:11:10 |
| 15663 | deterministic evaluation and recovery | COMPLETED | 0:0 | 00:09:09 |

## 4. Training

| Metric | A: no jitter | B: local jitter | B − A |
|---|---:|---:|---:|
| Best validation loss | 0.018456638 | 0.018836601 | +0.000379964 |
| Best epoch | 195 | 160 | -35 |
| Optimizer steps | 20,000 | 20,000 | 0 |
| Runtime | 21,749.697 s | 22,250.444 s | +500.747 s |
| Runtime | 6.0416 h | 6.1807 h | +0.1391 h |
| Runtime ratio | 1.0000 | 1.0230 | +2.3023% |
| Peak GPU memory | 18,128.202 MiB | 18,128.561 MiB | +0.358 MiB |

Arm B 的 best validation loss 比 Arm A 高 2.0587%。

## 5. Deterministic synthetic-current prediction

### 5.1 Test，correct RGB

25 个 test samples 的 macro mean：

| Metric | A: no jitter | B: local jitter | B − A | B 较低/较高的 paired samples |
|---|---:|---:|---:|---:|
| Evaluation loss | 0.014160124 | 0.014530453 | +0.000370330 | — |
| Raw endpoint | 0.007681539 | 0.007804981 | +0.000123442 | 10/15 较低/较高 |
| Raw Top-10% endpoint | 0.053443110 | 0.054163195 | +0.000720084 | 13/12 较低/较高 |
| Raw Top-1% endpoint | 0.202496550 | 0.225826787 | +0.023330238 | 9/16 较低/较高 |
| Raw global cosine | 0.957896502 | 0.928684516 | -0.029211986 | 1/24 较高/较低 |
| Raw pred/target norm ratio | 1.232569814 | 1.278238244 | +0.045668430 | — |
| Raw Top-10% cosine | 0.995099745 | 0.992019522 | -0.003080223 | — |
| Raw Top-1% cosine | 0.989990728 | 0.980677154 | -0.009313574 | — |

Paired median `B − A` 为：raw endpoint `+0.000105437`、raw Top-10% endpoint `-0.000084665`、raw Top-1% endpoint `+0.009985954`、raw global cosine `-0.014830828`。

### 5.2 Validation，correct RGB

| Metric | A: no jitter | B: local jitter | B − A |
|---|---:|---:|---:|
| Evaluation loss | 0.018453153 | 0.018833027 | +0.000379874 |
| Raw endpoint | 0.004203440 | 0.004435406 | +0.000231966 |
| Raw Top-10% endpoint | 0.026191274 | 0.027933967 | +0.001742693 |
| Raw Top-1% endpoint | 0.123439917 | 0.134013718 | +0.010573801 |
| Raw global cosine | 0.977699494 | 0.970374954 | -0.007324541 |

### 5.3 Test，zero RGB

| Metric | A: no jitter | B: local jitter | B − A |
|---|---:|---:|---:|
| Evaluation loss | 0.014142160 | 0.014576221 | +0.000434061 |
| Raw endpoint | 0.007951096 | 0.008111844 | +0.000160749 |
| Raw Top-10% endpoint | 0.055987176 | 0.057046108 | +0.001058932 |
| Raw Top-1% endpoint | 0.226911303 | 0.251707654 | +0.024796351 |
| Raw global cosine | 0.928834684 | 0.909044228 | -0.019790456 |

Correct RGB 到 zero RGB 的 raw endpoint 变化为 Arm A `+0.000269557`，Arm B `+0.000306863`。

## 6. OpenMVS48 current-mesh recovery

该阶段使用 5 个 OpenMVS meshes、相同 inputs 和 `sparse_uniform_oracle_core` recovery solver。

**2026-08-22 解释更新：仅诊断。** OpenMVS initial mesh 质量过差，不能作为
target、pseudo-GT 或本消融的判定端点。下表是历史压力测试记录；Arm A/B 的正式
选择只依据受控 synthetic-current 指标。

| Metric | A: no jitter | B: local jitter | B − A |
|---|---:|---:|---:|
| Mean initial Chamfer | 0.024729284 | 0.024729284 | 0 |
| Mean refined Chamfer | 0.025067426 | 0.025249771 | +0.000182345 |
| Median refined Chamfer | 0.022496150 | 0.022578672 | +0.000082522 |
| Mean initial P2S | 0.024559215 | 0.024559215 | 0 |
| Mean refined P2S | 0.024900180 | 0.025077573 | +0.000177393 |
| Mean initial normal consistency | 0.819898524 | 0.819898524 | 0 |
| Mean refined normal consistency | 0.819698948 | 0.819023295 | -0.000675653 |
| Improved-over-initial meshes | 0/5 | 0/5 | 0 |
| Introduced flipped faces | 329 | 370 | +41 |
| New degeneracies | 0 | 0 | 0 |
| Mean vertex displacement | 0.000527020 | 0.000593928 | +0.000066909 |

逐 mesh paired refined Chamfer：

| Mesh | A: no jitter | B: local jitter | B − A |
|---|---:|---:|---:|
| `038b62bb-b277-4d92-8afe-3ff115add02c` | 0.046464817 | 0.046873286 | +0.000408468 |
| `43bd0910-1dd1-4b1e-9ba2-e9801e6b5761` | 0.022496150 | 0.022578672 | +0.000082522 |
| `5ac05fe8-b550-4786-8e8c-c5a43c30d112` | 0.016170162 | 0.016204344 | +0.000034182 |
| `5c226f2b-aad3-4371-a3f9-ea2ee9a63327` | 0.023058260 | 0.023401206 | +0.000342946 |
| `653efc24-a5c5-4f94-86d9-1256dcf4bc28` | 0.017147743 | 0.017191347 | +0.000043604 |

Arm B 的 refined Chamfer 在 5/5 paired meshes 上高于 Arm A。两个 arm 的 mean refined Chamfer 均高于 mean initial Chamfer。

## 7. 结论

- 在 seed 7、28 views、20,000 steps 和本报告契约下，Arm B 的 best validation loss 比 Arm A 高 2.0587%。
- Arm B 的 test raw endpoint、raw Top-10% endpoint 和 raw Top-1% endpoint 分别比 Arm A 高 `0.000123442`、`0.000720084` 和 `0.023330238`；raw global cosine 低 `0.029211986`。
- OpenMVS 压力测试中，Arm B 的 mean refined Chamfer 和 P2S 分别比 Arm A 高 `0.000182345` 和 `0.000177393`；该结果仅作 OOD 失败记录，不参与 arm 选择。
- 两个 arm 均为 0/5 OpenMVS meshes 低于各自 initial Chamfer；这不把 OpenMVS 提升为期望目标。
- Arm B 的 runtime 是 Arm A 的 1.0230 倍。
- 当前记录不支持在该 contract 下启用 `std = 0.003 h`、L2 cap `0.009 h` 的 training-only local query jitter。

## 8. 产物

HPC run root：

`runs/learned_laplacian/sofa50_synthetic_current_28view_jitter_ablation_seed7/`

其中：

- `analysis/REPORT.md`：HPC 汇总；
- `analysis/prediction/prediction_summary.json`：aggregate、paired 和 per-sample prediction metrics；
- `analysis/prediction/prediction_per_sample.csv`：validation/test、correct-RGB/zero-RGB 逐样本记录；
- `analysis/openmvs48/openmvs_summary.json`：OpenMVS aggregate 和 per-mesh records；
- `analysis/openmvs48/openmvs_aggregate.csv`：OpenMVS arm 汇总；
- `analysis/openmvs48/openmvs_paired.csv`：OpenMVS paired comparison；
- `analysis/openmvs48/openmvs_per_mesh.csv`：OpenMVS 逐 mesh recovery metrics。
