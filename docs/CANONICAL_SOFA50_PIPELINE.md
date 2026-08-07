# Canonical Sofa50 learned-Laplacian pipeline

This is the single production formulation. Files and reports for residual,
direct-displacement, step-sweep, and oracle experiments are legacy diagnostics;
they are not alternate meanings of the main model output.

## Training

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

## Inference and recovery

Given the current expanded mesh `X0, F_e`, recompute the uniform `L_e` and
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

## Smooth-region diagnostic convention

Small `||delta_hat||` is used only as a relative smooth/low-local-variation
grouping for error analysis. The canonical diagnostic compares magnitude
percentiles (for example, smooth bottom 90% against high-curvature top 10%)
rather than assigning an arbitrary absolute flatness threshold. A near-zero
uniform-Laplacian magnitude is not a strict plane detector because it remains
sensitive to graph connectivity, sampling irregularity, and discretization.
This grouping does not change training loss, confidence, or recovery weights.
