# Chamfer 评估事故：Sofa50 同初始网格 benchmark

[English](CHAMFER_EVALUATION_INCIDENT_2026-08-21.md) | [简体中文](CHAMFER_EVALUATION_INCIDENT_2026-08-21.zh-CN.md)

状态：已于 2026-08-21 修复。修正后的 25-sample 报告为
`contract_audit: true`。

## 摘要

最初的 Sofa50 同初始网格对比把各方法原生计算的 Chamfer 放进了同一张横向表。
这些数值来自不同评估路径，因此不可直接比较。共同 initial mesh 暴露了问题：
learned-method 路径给出的 mean initial Chamfer 是 `0.003913228`，external-method
路径则是 `0.017070468`，但逐 sample 的输入 mesh 身份审计已经通过。

这是评估与聚合错误，不代表某个方法拿到了更好的初始网格。受影响的初步表格
不能作为科学结论。

## 影响范围

事故影响 `ours`、NDS、nvdiffrec 与 ExMesh 的初步 Sofa50 横向对比，但不改变：

- 各方法已经生成的 mesh output；
- 25 个 canonical test sample IDs；
- 输入的 current/coarse mesh、RGB observations 或 cameras；
- 独立报告的 DTU/ExMesh 官方毫米制 evaluator；
- Sofa50 learned-Laplacian prediction metrics。

事故会使统一重评估前任何混用 `native_*` Chamfer、P2S 或 normal 的跨方法排名失效。

## 发现过程与根因

所有方法都应从同一 prepared current mesh 出发。输入审计检查了文件 lineage 和逐
sample mesh identity。同一个 initial mesh 在同一 deterministic evaluator 下必须只有
一个分数，因此 initial score 的巨大差异构成明确的协议失败。

原聚合器直接信任了各 adapter 的 native metric fields。Native fields 可以保留为
provenance，但其 sampling、GT loading 与 metric semantics 不保证完全相同。原聚合层
没有对归档后的 mesh 强制执行同一个 evaluator。

## 修复方式

`scripts/aggregate_sofa50_same_initial_benchmark.py` 现在会加载 common initial mesh
与每个已完成的 output mesh，并通过唯一实现重新计算全部主要 geometry metrics：

```text
mlr.learned_laplacian.evaluation.evaluate_mesh_geometry
surface_samples = 3000
seed = 7
fscore_threshold = 0.01
```

Primary Chamfer 是两个“采样表面点到三角形表面”方向均值的平均。在两个方向使用
相同 sample count 时，报告的 bidirectional P2S mean 与它数值相同。方法原生指标只以
`native_*` provenance fields 保留在 per-sample 文件中，绝不参与 primary ranking。

修正后的 contract 要求：

- 每个 Group-A 方法完成全部 25 个预期 samples；
- 各方法逐 sample 的 common initial metric 完全一致；
- 每个 completed row 都有 `unified_metric_audit: true`；
- 每个 completed row 都记录 `metric_protocol`；
- topology-dependent 指标不可比较时必须明确为 null。

任一条件不满足都会令 `contract_audit` 为 false。

## 修正结果

四种方法均完成 `25/25`；统一 mean initial Chamfer 为 `0.017070468`。

| 方法 | Mean final Chamfer ↓ | Aggregate improvement | 改善 samples | Normal consistency ↑ |
|---|---:|---:|---:|---:|
| Ours | 0.011347800 | 33.52% | 25/25 | **0.944514** |
| NDS | **0.011204992** | **34.36%** | 22/25 | 0.873805 |
| nvdiffrec | 0.013654660 | 20.01% | 18/25 | 0.848122 |
| ExMesh | 0.020170615 | -18.16% | 8/25 | 0.845337 |

本次运行中，NDS 的 mean Chamfer 比 ours 低 `1.26%`。Ours 是唯一改善全部 samples
的方法，并具有明显更高的 normal consistency。ExMesh 会改变 topology，且在这个
supplied-initial synthetic protocol 下 aggregate Chamfer 变差。

这些数值只适用于 Sofa50 same-initial protocol。禁止将其与官方 DTU ExMesh
`overall`、已废弃的 `ours_exmesh_initial_zero_shot = 0.616526 mm` 探索结果，或此前
learned-recovery 的 native Chamfer 表混用。

## 预防规则

1. 跨方法 primary table 只能对归档 mesh 调用同一个 evaluator 生成。
2. Common initial score 是 evaluator checksum，而不仅是描述字段。
3. Native metrics 只用于调试与 provenance。
4. Metric implementation、sample count、seed 与 threshold 必须写入
   `summary.json` 和每条 per-sample row。
5. 官方 benchmark 指标与项目内部 synthetic 指标必须分表，并明确单位和 semantics。

## 产物

- [修正后的最终报告](../reports/synthetic_same_initial_benchmark_20260820/full_report/FINAL_REPORT.md)
- [Summary JSON](../reports/synthetic_same_initial_benchmark_20260820/full_report/summary.json)
- [Per-sample CSV](../reports/synthetic_same_initial_benchmark_20260820/full_report/per_sample.csv)
- [方法/输入 contract audit](../reports/synthetic_same_initial_benchmark_20260820/METHOD_INPUT_CONTRACT_AUDIT.md)
