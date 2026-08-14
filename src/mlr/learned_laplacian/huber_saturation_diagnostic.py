from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


GROUPS = ("bottom_90_percent", "top_10_percent", "top_1_percent")


def summarize_huber_saturation(
    prediction_raw: np.ndarray,
    target_raw: np.ndarray,
    target_weight: np.ndarray,
    sample_index: np.ndarray,
    *,
    huber_delta: float,
) -> dict[str, Any]:
    """Summarize output-space Huber saturation by GT target magnitude.

    The weighted gradient matches the derivative of the repository's validation
    prediction loss: each sample is normalized by its target-weight sum, xyz is
    averaged, and then samples are averaged.  It is the gradient with respect to
    the model's raw-Laplacian output, before the network Jacobian.
    """

    prediction = np.asarray(prediction_raw, dtype=np.float64)
    target = np.asarray(target_raw, dtype=np.float64)
    weight = np.asarray(target_weight, dtype=np.float64)
    samples = np.asarray(sample_index, dtype=np.int64)
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 3:
        raise ValueError("prediction_raw and target_raw must both have shape [N, 3].")
    if weight.shape != (len(prediction),) or samples.shape != (len(prediction),):
        raise ValueError("target_weight and sample_index must both have shape [N].")
    if len(prediction) == 0:
        raise ValueError("At least one valid vertex is required.")
    if huber_delta <= 0:
        raise ValueError("huber_delta must be positive.")
    if np.any(weight < 0) or not np.all(np.isfinite(weight)):
        raise ValueError("target_weight must be finite and non-negative.")
    if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(target)):
        raise ValueError("prediction_raw and target_raw must be finite.")

    unique_samples, inverse = np.unique(samples, return_inverse=True)
    sample_weight_sum = np.bincount(inverse, weights=weight)
    if np.any(sample_weight_sum <= 0):
        raise ValueError("Every sample must have positive total target weight.")
    sample_count = len(unique_samples)
    normalized_weight = weight / sample_weight_sum[inverse] / sample_count

    residual = prediction - target
    absolute_error = np.abs(residual)
    clipped_error = np.minimum(absolute_error, huber_delta)
    saturated = absolute_error > huber_delta
    vector_error = np.linalg.norm(residual, axis=1)
    target_magnitude = np.linalg.norm(target, axis=1)
    huber_component = np.where(
        saturated,
        huber_delta * (absolute_error - 0.5 * huber_delta),
        0.5 * absolute_error**2,
    )
    # The implementation averages xyz before applying the per-sample weighted mean.
    output_gradient_abs = normalized_weight[:, None] * clipped_error / 3.0
    unclipped_output_gradient_abs = normalized_weight[:, None] * absolute_error / 3.0
    weighted_huber = normalized_weight[:, None] * huber_component / 3.0

    vertex_count = len(prediction)
    descending = np.argsort(-target_magnitude, kind="stable")
    top_10_count = max(1, int(math.ceil(0.10 * vertex_count)))
    top_1_count = max(1, int(math.ceil(0.01 * vertex_count)))
    top_10 = np.zeros(vertex_count, dtype=bool)
    top_1 = np.zeros(vertex_count, dtype=bool)
    top_10[descending[:top_10_count]] = True
    top_1[descending[:top_1_count]] = True
    masks = {
        "bottom_90_percent": ~top_10,
        "top_10_percent": top_10,
        "top_1_percent": top_1,
    }

    total_gradient = float(output_gradient_abs.sum())
    total_huber = float(weighted_huber.sum())
    rows: list[dict[str, Any]] = []
    for name in GROUPS:
        mask = masks[name]
        group_error = absolute_error[mask]
        group_saturated = saturated[mask]
        group_gradient = output_gradient_abs[mask]
        group_unclipped_gradient = unclipped_output_gradient_abs[mask]
        group_huber = weighted_huber[mask]
        gradient_total = float(group_gradient.sum())
        unclipped_gradient_total = float(group_unclipped_gradient.sum())
        huber_total = float(group_huber.sum())
        saturated_gradient = float(group_gradient[group_saturated].sum())
        rows.append(
            {
                "group": name,
                "vertex_count": int(mask.sum()),
                "vertex_fraction": float(mask.mean()),
                "target_magnitude_min": float(target_magnitude[mask].min()),
                "target_magnitude_mean": float(target_magnitude[mask].mean()),
                "target_magnitude_max": float(target_magnitude[mask].max()),
                "raw_error_magnitude_mean": float(vector_error[mask].mean()),
                "raw_error_magnitude_rms": float(
                    np.sqrt(np.mean(np.square(vector_error[mask])))
                ),
                # Exact saturation rate for the component-wise Huber implementation.
                "component_saturation_probability": float(group_saturated.mean()),
                "vertex_any_component_saturated_probability": float(
                    group_saturated.any(axis=1).mean()
                ),
                # Literal vector-magnitude interpretation of P(|e| > delta | group).
                "vector_error_exceeds_delta_probability": float(
                    (vector_error[mask] > huber_delta).mean()
                ),
                "mean_huber_gradient_l1_per_vertex_unweighted": float(
                    clipped_error[mask].sum(axis=1).mean()
                ),
                "mean_huber_gradient_l2_per_vertex_unweighted": float(
                    np.linalg.norm(clipped_error[mask], axis=1).mean()
                ),
                "weighted_output_gradient_l1_total": gradient_total,
                "weighted_output_gradient_l1_share": (
                    gradient_total / total_gradient if total_gradient else 0.0
                ),
                "weighted_unclipped_l1_total": unclipped_gradient_total,
                "huber_gradient_retention_vs_unclipped_l1": (
                    gradient_total / unclipped_gradient_total
                    if unclipped_gradient_total
                    else 1.0
                ),
                "gradient_from_saturated_components_fraction": (
                    saturated_gradient / gradient_total if gradient_total else 0.0
                ),
                "weighted_huber_loss_total": huber_total,
                "weighted_huber_loss_share": (
                    huber_total / total_huber if total_huber else 0.0
                ),
            }
        )

    by_name = {str(row["group"]): row for row in rows}
    bottom = by_name["bottom_90_percent"]
    top_10_row = by_name["top_10_percent"]
    top_1_row = by_name["top_1_percent"]

    def ratio(numerator: float, denominator: float) -> float | None:
        return numerator / denominator if denominator else None

    return {
        "huber_delta": float(huber_delta),
        "valid_vertex_count": vertex_count,
        "sample_count": sample_count,
        "group_definition": {
            "ranking": "global descending GT raw-Laplacian vector magnitude",
            "bottom_90_and_top_10_partition": True,
            "top_1_is_nested_in_top_10": True,
            "ties": "stable vertex order",
        },
        "gradient_definition": (
            "absolute output-space d(mean weighted component-wise Huber)/d(prediction); "
            "includes per-sample target-weight normalization and split mean"
        ),
        "overall": {
            "component_saturation_probability": float(saturated.mean()),
            "vertex_any_component_saturated_probability": float(
                saturated.any(axis=1).mean()
            ),
            "vector_error_exceeds_delta_probability": float(
                (vector_error > huber_delta).mean()
            ),
            "weighted_output_gradient_l1_total": total_gradient,
            "weighted_huber_loss_total": total_huber,
        },
        "groups": rows,
        "contrasts": {
            "top10_to_bottom90_mean_raw_error_ratio": ratio(
                float(top_10_row["raw_error_magnitude_mean"]),
                float(bottom["raw_error_magnitude_mean"]),
            ),
            "top1_to_bottom90_mean_raw_error_ratio": ratio(
                float(top_1_row["raw_error_magnitude_mean"]),
                float(bottom["raw_error_magnitude_mean"]),
            ),
            "top10_to_bottom90_component_saturation_ratio": ratio(
                float(top_10_row["component_saturation_probability"]),
                float(bottom["component_saturation_probability"]),
            ),
            "top1_to_bottom90_component_saturation_ratio": ratio(
                float(top_1_row["component_saturation_probability"]),
                float(bottom["component_saturation_probability"]),
            ),
            "top10_gradient_share_over_vertex_share": ratio(
                float(top_10_row["weighted_output_gradient_l1_share"]),
                float(top_10_row["vertex_fraction"]),
            ),
            "top1_gradient_share_over_vertex_share": ratio(
                float(top_1_row["weighted_output_gradient_l1_share"]),
                float(top_1_row["vertex_fraction"]),
            ),
        },
    }


def write_huber_saturation_outputs(
    output_dir: str | Path,
    summary: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any],
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": dict(metadata), **dict(summary)}
    (output / "huber_saturation_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = list(summary["groups"])
    with (output / "huber_saturation_groups.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "REPORT.md").write_text(
        _markdown_report(payload), encoding="utf-8"
    )


def _markdown_report(payload: Mapping[str, Any]) -> str:
    metadata = payload["metadata"]
    rows = {str(row["group"]): row for row in payload["groups"]}
    contrasts = payload["contrasts"]

    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{100.0 * float(value):.3f}%"

    def num(value: float | None) -> str:
        return "n/a" if value is None else f"{float(value):.6g}"

    lines = [
        "# Sofa50 Arm B：GT raw-Laplacian 分组 Huber 饱和诊断",
        "",
        "## 口径",
        "",
        f"- Split：`{metadata['split']}`；有效 vertex：`{payload['valid_vertex_count']}`；样本：`{payload['sample_count']}`。",
        f"- Checkpoint：`{metadata['checkpoint']}`（optimizer steps `{metadata['optimizer_steps']}`）。",
        f"- Huber `delta={payload['huber_delta']}`，与 Arm B 训练配置一致。",
        f"- 本地推理精度：`{metadata.get('local_inference_precision')}`；本地重算 validation Huber：`{num(metadata.get('recomputed_validation_loss'))}`；原运行记录：`{num(metadata.get('recorded_final_validation_loss'))}`（相对差异 `{pct(metadata.get('relative_loss_difference_from_recorded', 0.0))}`）。",
        "- 分组按全部有效 vertex 的 GT raw-Laplacian 向量模长全局排序；top 1% 包含在 top 10% 中。",
        "- `P(component saturated)` 是与训练实现严格对应的逐 xyz 分量饱和概率；`P(any component saturated)` 是 vertex 至少一个分量饱和；`P(||e||₂>δ)` 是问题中字面向量误差口径。",
        "- gradient 是 loss 对模型 raw-Laplacian 输出的梯度，不包含网络 Jacobian；贡献统计保留 target-confidence、逐样本归一化、xyz 平均和 split 平均。",
        "",
        "## 结果",
        "",
        "| Group | Vertices | Mean ||target||₂ | Mean ||error||₂ | P(component saturated) | P(any component saturated) | P(||e||₂>δ) | Gradient share | Gradient retention | Huber-loss share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in GROUPS:
        row = rows[name]
        lines.append(
            f"| {name} | {row['vertex_count']} ({pct(row['vertex_fraction'])}) | "
            f"{num(row['target_magnitude_mean'])} | {num(row['raw_error_magnitude_mean'])} | "
            f"{pct(row['component_saturation_probability'])} | "
            f"{pct(row['vertex_any_component_saturated_probability'])} | "
            f"{pct(row['vector_error_exceeds_delta_probability'])} | "
            f"{pct(row['weighted_output_gradient_l1_share'])} | "
            f"{pct(row['huber_gradient_retention_vs_unclipped_l1'])} | "
            f"{pct(row['weighted_huber_loss_share'])} |"
        )
    top_10 = rows["top_10_percent"]
    top_1 = rows["top_1_percent"]
    lines.extend(
        [
            "",
            "`Gradient retention` 是实际 Huber gradient L1 与未裁剪 L1-error gradient 的比值；越低表示 Huber 饱和抑制越强。",
            "",
            "## 对比与判读",
            "",
            f"- Top 10% / bottom 90% 平均 raw error 比：`{num(contrasts['top10_to_bottom90_mean_raw_error_ratio'])}×`。",
            f"- Top 1% / bottom 90% 平均 raw error 比：`{num(contrasts['top1_to_bottom90_mean_raw_error_ratio'])}×`。",
            f"- Top 10% 只占 `{pct(top_10['vertex_fraction'])}` vertex，但贡献 `{pct(top_10['weighted_output_gradient_l1_share'])}` 的实际 output-gradient（富集倍数 `{num(contrasts['top10_gradient_share_over_vertex_share'])}×`）。",
            f"- Top 1% 只占 `{pct(top_1['vertex_fraction'])}` vertex，但贡献 `{pct(top_1['weighted_output_gradient_l1_share'])}` 的实际 output-gradient（富集倍数 `{num(contrasts['top1_gradient_share_over_vertex_share'])}×`）。",
            f"- Top 10% 中饱和分量贡献该组 `{pct(top_10['gradient_from_saturated_components_fraction'])}` 的 gradient；gradient retention 为 `{pct(top_10['huber_gradient_retention_vs_unclipped_l1'])}`。",
            f"- Top 1% 中饱和分量贡献该组 `{pct(top_1['gradient_from_saturated_components_fraction'])}` 的 gradient；gradient retention 为 `{pct(top_1['huber_gradient_retention_vs_unclipped_l1'])}`。",
            "",
            "## 结论",
            "",
            f"Top 10% 不是整体进入饱和区：逐分量饱和率为 `{pct(top_10['component_saturation_probability'])}`，至少一个分量饱和的 vertex 为 `{pct(top_10['vertex_any_component_saturated_probability'])}`。但饱和高度集中在最极端的 top 1%：其平均 raw error 是 bottom 90% 的 `{num(contrasts['top1_to_bottom90_mean_raw_error_ratio'])}×`，`{pct(top_1['vertex_any_component_saturated_probability'])}` 的 vertex 至少一个分量饱和，Huber gradient 只保留未裁剪 L1-error gradient 的 `{pct(top_1['huber_gradient_retention_vs_unclipped_l1'])}`。",
            "",
            f"因此，数据支持更精确的表述：当前 `delta=0.01` Huber loss 对最高曲率 1% vertex 存在明显梯度压缩，而不是整个 top 10% 全面饱和。top 1% 占全部 vertex 的 1%，承担 `{pct(top_1['weighted_huber_loss_share'])}` 的 Huber loss，却只占 `{pct(top_1['weighted_output_gradient_l1_share'])}` 的 output-gradient L1。这个结果直接验证了所怀疑的 loss inductive bias；要把它提升为“与 reconstruction objective 不一致”的完整因果结论，还需把同一批 vertex 与 surface displacement / Chamfer sensitivity 配对。",
            "",
            "## 可复现产物",
            "",
            "- `huber_saturation_summary.json`：完整统计与 checkpoint 元数据。",
            "- `huber_saturation_groups.csv`：三组汇总数据。",
        ]
    )
    return "\n".join(lines) + "\n"
