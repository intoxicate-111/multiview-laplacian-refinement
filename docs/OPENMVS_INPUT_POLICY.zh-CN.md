# OpenMVS 输入使用政策

状态：自 2026-08-22 起生效。

Sofa50 OpenMVS mesh 是包含明显细节缺失与重建伪影的低质量外部重建结果，且显著偏离受控 synthetic-current 输入分布。它不能被当成期望 mesh target。

## 必须遵守的解释规则

OpenMVS mesh 只能保留为：

- 分布外、低质量输入压力测试；
- 鲁棒性和失败模式分析样本；
- 在同时报告其 initial quality 时使用的外部重建基线。

OpenMVS mesh 不得用作：

- training target、pseudo-GT、supervision proxy 或 target-topology template；
- checkpoint 或 hyperparameter 选择端点；
- learned refinement 方法的质量上限；
- 判断模型、loss 或 architecture 优劣的主要证据；
- 后续数据扩展或方法设计所追逐的目标分布。

所有保留的 OpenMVS 表格都必须标注为 `仅诊断 / 不参与决策`，并在 refined result 旁同时报告 initial mesh quality。历史 OpenMVS 数值仍然是已执行压力测试的有效记录，但后续结论必须建立在受控 GT-derived current mesh、same-initial benchmark，或通过明确 mesh-quality gate 的外部输入上。

Projected-GT oracle 实验仍可用于分解该低质量输入上的 representation、recovery 与 prediction failure，但它不会把 OpenMVS 变成目标，不能建立通用方法 ceiling，也不能成为针对 OpenMVS 特定伪影优化 pipeline 的理由。
