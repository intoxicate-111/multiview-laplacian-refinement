# Canonical Sofa50 learned-Laplacian pipeline

This document distinguishes two explicit, non-interchangeable contracts. The
historical canonical GT-query model predicts an absolute `h^2`-normalised
Laplacian. The current synthetic-current Arm B and its loss/image-feature
ablations predict the direct raw current-graph Laplacian. A configuration must
declare `target_mode`; native loss values from the two representations are not
numerically comparable.

## Training

### Canonical GT-query contract

For every GT-query Sofa mesh, construct the uniform graph Laplacian on the GT
vertices and faces:

```text
delta_gt_raw = L_gt @ V_gt
h_gt[i] = arithmetic mean length of unique undirected one-ring edges at i
delta_gt_hat[i] = delta_gt_raw[i] / (h_gt[i]^2 + 1e-12)
```

The network is supervised to predict the absolute `delta_gt_hat`. It does not
predict displacement, a raw Laplacian residual, a normalized residual, or a
current-to-target correction. A small optional side head predicts point
confidence; it is trained as heteroscedastic reliability and receives no GT
quantity as an inference input.

### Synthetic-current direct-raw contract

For each stored current graph `P_current, F_current`, the paired proxy and
supervised target are fixed during training:

```text
delta_target_raw = L_current @ P_proxy
target_mode = raw_laplacian
prediction_loss_space = output_representation
```

The trainer initially selects `raw_laplacian_target` and switches to the
normalised copy only when `target_mode=edge_scale_normalized_laplacian`.
Consequently Arm B compares `delta_pred_raw` directly with
`delta_target_raw`; it performs no division by `h_current^2`. The shared
`target_scaling` object supplies edge-scale metadata, a valid-topology mask and
the optional normalised copy. With `clip_max_norm=null`, it does not rescale or
clip the raw target.

## Inference and recovery

For the canonical normalised-output path, given the current expanded mesh
`X0, F_e`, recompute the uniform `L_e` and
per-vertex `h_current` from that mesh. Convert once:

```text
delta_pred_raw = delta_hat_prediction * (h_current^2 + 1e-12)
weight = renderer_visible_any * confidence_prediction
```

With confidence disabled, `weight = renderer_visible_any`. The strict renderer
gate means an all-view-invisible vertex always has exactly zero learned-target
weight. Recovery minimizes the weighted current-graph Laplacian mismatch plus
the established global position anchor to `X0` (`lambda_anchor=0.01`). The
canonical baseline uses `unseen_anchor_weight=0.0`; the historical value `1.0`
is retained only in legacy/ablation configurations. No GT scale, GT
differential vector, or expanded placeholder target enters inference.

For direct-raw Arm B and its HF variants, `delta_pred_raw` is already in solver
units and is passed to the same confidence/visibility-weighted recovery without
an `h^2` conversion. Evaluation saves raw predictions and checks the current-
graph target formula and raw round trip explicitly.

The explicit reference implementation is
`mlr.learned_laplacian.canonical_pipeline.canonical_current_graph_recovery_inputs`.
It exposes the unambiguous names `delta_hat_prediction`, `delta_pred_raw`,
`h_current`, and `delta_current_raw` and is covered by round-trip, current-scale,
visibility-gate, and identity regression tests.

## Canonical Sofa50 experiment

- Dataset: Sofa50 only, 40 train / 5 validation / 5 test.
- GT-query manifest:
  `/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/sofa_mesh/sofa50_refinement/multiview_960/gt_query_manifest.json`
- Expanded inference manifest:
  `/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/sofa_mesh/sofa50_refinement/multiview_960/expanded_inference_manifest.json`
- Configuration:
  `configs/learned_laplacian/train_sofa50_50mesh_2000epoch_absolute_h2_confidence.json`
- Run directory:
  `runs/learned_laplacian/sofa50_50mesh_2000epoch_absolute_h2_confidence/`
- Budget: exactly 2000 epochs; no Thingi10K and no 5000-epoch continuation.

Run with:

```bash
bash scripts/train_sofa50_50mesh_2000epoch_absolute_h2_confidence.sh
```

The run writes `checkpoint_latest.pt`, `checkpoint_best.pt`, requested periodic
epoch checkpoints, normalized prediction metrics, confidence diagnostics, and
the complete final report/artifact tables.

## Current direct-raw Sofa50 baseline

The current controlled baseline uses C2F2, 28 native 960 observations,
current-query/current-graph inputs, seed 7, Huber with `delta=0.01`, no local
jitter and 20,000 optimiser steps. Its output is `delta_pred_raw`. Completed
controlled extensions include raw MSE, a learned dynamic residual expert with
inference-time gate interventions, Gaussian/HF feature construction, and the
running native-1920 HF resolution experiment. All scientific comparisons use
the same 25 test samples and unified raw prediction/recovery metrics; early
training loss alone is not a decision metric.

## Smooth-region diagnostic convention

Small `||delta_hat||` is used only as a relative smooth/low-local-variation
grouping for error analysis. The canonical diagnostic compares magnitude
percentiles (for example, smooth bottom 90% against high-curvature top 10%)
rather than assigning an arbitrary absolute flatness threshold. A near-zero
uniform-Laplacian magnitude is not a strict plane detector because it remains
sensitive to graph connectivity, sampling irregularity, and discretization.
This grouping does not change training loss, confidence, or recovery weights.
