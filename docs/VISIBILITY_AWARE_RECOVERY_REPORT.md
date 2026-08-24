# Sofa50 renderer visibility and visibility-aware recovery report

Date: 2026-08-06

## 2026-08-24 scope correction

The decision below is a historical expanded-query/GT-query result. It showed
that a hard mask was less catastrophic than fitting every unreliable equation
from that out-of-distribution frozen checkpoint; it does **not** establish hard
visibility as the current matched-domain recovery design.

The later `Sofa50MultiTopologyRawLap500_v2` exact-target ablation holds the
current graph, exact raw target, L2 solve and `0.01` anchor fixed. Adding hard
any-view visibility lowers mean recovery efficiency from `0.34258` to
`0.16875` and worsens 44/50 samples. Confidence is nearly constant and adds no
material recovery. The current A-D study consequently uses every Laplacian row
with regularised sparse integration and no visibility/confidence/recovery
Huber/Adam. See [the current recovery-aware study](SOFA50_RECOVERY_AWARE_STUDY.md).

The original experiment, numbers and conclusion remain below for provenance;
they must be read within their 2026-08-06 expanded-query scope.

## Decision

Hard any-view visibility gating is necessary but not sufficient. On the five real
expanded validation meshes it reduced frozen-checkpoint reconstruction Chamfer from
`0.120283` to `0.0146517` (87.8%) and raised nearest-surface absolute normal
consistency from `0.5026` to `0.7089`. It still did not beat the initial expanded
mesh (`0.000652884` mean Chamfer): all three recovery variants worsened all five
meshes relative to initial.

A strong unseen anchor is not recommended as the default. It reduced all-view-
invisible displacement from `0.06813` to `0.004389`, but worsened Chamfer from
`0.0146517` to `0.0198938`. The remaining failure therefore cannot be attributed
only to unseen vertices' own equations. Incorrect predictions on visible—especially
low-view-count—vertices and their graph coupling remain the dominant problem.

Do not resume the formal 5000-epoch Sofa50 training yet.

## Input and stopped-run record

The existing expanded inputs were found; no coarse or expanded mesh was regenerated.

- GT-query manifest: `/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/sofa_mesh/sofa50_refinement/multiview_960/gt_query_manifest.json`
- Expanded manifest: `/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/sofa_mesh/sofa50_refinement/multiview_960/expanded_inference_manifest.json`
- Expanded split counts: train/validation/test = 40/5/5.
- Frozen checkpoint: `runs/learned_laplacian/sofa50_refinement_960_gt_query_5000_full/best.pt`, epoch 700.
- The expanded target is an identity-placeholder required by the schema; it was not
  reported as an expanded-graph GT-delta oracle.

The previous formal run was stopped before these changes:

- systemd unit: `mlr-sofa50-refinement-960-5000-full.service`
- service PID: 130192; trainer PID: 130260
- last complete log epoch: 731; approximate optimizer step: 7310
- last periodic checkpoint: `checkpoint_epoch_000700.pt`
- best checkpoint: `best.pt`, epoch 700, validation `0.03812928088`
- output: `runs/learned_laplacian/sofa50_refinement_960_gt_query_5000_full`
- verified inactive/dead, MainPID 0, no matching process, and GPU compute process released

No formal long training was restarted.

## Visibility definition and recovery equation

Prepared tensors use `[views, vertices]`, not `[vertices, views]`. For expanded
validation the shape is `[14, N]` and dtype is `bool`.

`visibility_backface_and_occlusion[v, i]` is true only when the expanded vertex is
inside the frustum and an incident expanded-mesh face survives explicit CCW OpenGL
back-face culling and wins the face-ID depth test at the projected pixel or its 3×3
neighborhood. No depth image is loaded, passed through the DataLoader, used as model
input, or compared with query depth. Expanded visibility comes from each expanded
mesh's own vertices and faces, never GT visibility or correspondence.

The hard mask is exact:

```text
visibility_count[i] = sum_v visibility[v, i]
visible_any[i]      = visibility_count[i] > 0
laplacian_weight[i] = float(visible_any[i])
```

There is no epsilon. Recovery applies the weight to the complete equation:

```text
sqrt(weight[:, None]) * (L @ X - delta_pred)
```

It never changes an unseen target to zero while retaining the left-hand `L @ X`
row. A zero-weight vertex can still move through the global position anchor and
neighboring visible Laplacian rows.

## Renderer diagnostics

Across all 50 GT-query meshes:

| statistic | mesh mean |
|---|---:|
| frustum-valid ratio | 100.00% |
| rejected by projected back-face test | 40.08% |
| rejected by depth-tested face-ID occlusion | 82.78% |
| final visible vertex-view ratio | 17.17% |
| vertices with zero visible views | 31.57% |
| pixels removed by culling | 0.729% (max 9.457%) |

On the five expanded validation meshes, mean all-view-invisible ratio is 42.23%,
with per-mesh values from 34.72% to 57.39%. Mean visible views per vertex is 1.857.

Orientation diagnostics found boundary edges in 39/50 meshes and non-manifold edges
in 31/50. Mesh `2b3562c2-03a7-4a81-92d2-e658f31b11b9` has 38 inconsistent-winding
edges among 40,303 manifold shared edges (0.0943%). It was reported, not repaired.

Important limitation: existing saved Sofa RGB observations were rendered two-sided
with MSAA4. The face-ID pass shares cameras, projection and depth convention, but
culling variants explicitly enable culling and the integer ID pass is non-MSAA.
`occlusion_only` matches the original two-sided culling state most closely. Exact
RGB/ID pixel parity for culling variants would require rerendering RGB, which this
task deliberately did not do.

## Frozen-checkpoint and short-training diagnostics

The old checkpoint behaves normally only under its original frustum distribution:

| visibility | original loss | zero RGB | shuffled RGB | cross RGB | pred/GT magnitude | High-10% cosine |
|---|---:|---:|---:|---:|---:|---:|
| frustum only | 0.038118 | 0.053616 | 0.041356 | 0.038857 | 0.739 | 0.778 |
| backface only | 0.074531 | 0.059811 | 0.072459 | 0.073767 | 1.220 | 0.710 |
| occlusion only | 0.485045 | 0.471220 | 0.484747 | 0.483877 | 6.218 | 0.510 |
| backface + occlusion | 0.484685 | 0.470818 | 0.484392 | 0.483502 | 6.214 | 0.510 |

This is an input-distribution failure, not evidence against training with correct
visibility. Six paired short runs used the same seed, Adam settings, Huber delta,
14 views and 100 optimizer steps. Full visibility stayed finite, but its image-
ablation gaps remained very small. At 16 meshes, full visibility had exact-query
loss `0.085056`, pred/GT magnitude `0.144`, High-10% cosine `0.253`, and original-
versus-zero RGB loss gap only `0.000369`. The corresponding frustum gap was
`0.004831`. A longer formal run is not justified by these short results.

## Visibility-aware recovery experiment

All variants use one identical frozen prediction per mesh and unchanged operator,
scale, solver, learning rate, 200 iterations, global anchor and edge settings. Only
the recovery gate and optional unseen anchor differ. The unseen anchor weight is
1.0; the existing global anchor weight remains 0.01.

Initial expanded mean Chamfer is `0.000652884`.

| variant | Chamfer mean | Chamfer median | bidirectional P2S mean | normal consistency | visible displacement | invisible displacement | vs initial |
|---|---:|---:|---:|---:|---:|---:|---|
| baseline | 0.120283 | 0.118436 | 0.119475 | 0.5026 | 0.34983 | 0.40937 | 0/5 improve |
| hard mask | 0.0146517 | 0.0143050 | 0.0146196 | 0.7089 | 0.08129 | 0.06813 | 0/5 improve |
| hard mask + unseen anchor | 0.0198938 | 0.0206468 | 0.0195751 | 0.7196 | 0.12845 | 0.004389 | 0/5 improve |

Hard masking lowers displacement in both regions:

- visible: 0.34983 → 0.08129 (76.8% reduction);
- all-view-invisible: 0.40937 → 0.06813 (83.4% reduction);
- low-view-count (1–2): 0.37953 → 0.09855;
- well-observed (3+): 0.33297 → 0.07169.

The nonzero unseen displacement under hard masking is expected graph coupling, not
an unseen vertex's own predicted equation. The strong unseen anchor almost freezes
that region but increases visible displacement and worsens Chamfer, so it is not
necessary for the best current result. Because pure hard masking removes all unseen
rows yet remains about 22× worse than initial Chamfer, visible/low-view prediction
errors are still the main reconstruction bottleneck.

Nearest-surface normal consistency is mean absolute cosine in both directions and
supports different GT and expanded topology. Chamfer and point-to-surface metrics
are bidirectional; forward and reverse distances are also saved separately.

## Tests

```bash
conda run --no-capture-output -n test pytest -q \
  tests/learned_laplacian/test_visibility_recovery.py \
  tests/learned_laplacian/test_bunny_support.py
```

Result: `8 passed`. Renderer/projection/aggregation focused tests: `19 passed`;
short-training control tests: `14 passed`. The complete learned-Laplacian test
directory passed (`110 passed`), and the remaining repository tests also passed
(`36 passed`).

## Reproduction and artifacts

Prepare expanded visibility without regenerating geometry:

```bash
PYTHONPATH=src conda run --no-capture-output -n test \
  python scripts/prepare_renderer_visibility.py \
  --manifest ~/sofa_mesh/sofa50_refinement/multiview_960/expanded_inference_manifest.json \
  --backend opengl --front-face-winding ccw --neighborhood-radius 1 \
  --output-dir ~/sofa_mesh/sofa50_refinement/multiview_960/renderer_visibility_expanded_validation \
  --split validation --attach --overwrite
```

Run recovery ablation:

```bash
PYTHONPATH=src conda run --no-capture-output -n test \
  python scripts/run_visibility_recovery_ablation.py \
  --run-dir runs/learned_laplacian/sofa50_refinement_960_gt_query_5000_full \
  --expanded-manifest ~/sofa_mesh/sofa50_refinement/multiview_960/expanded_inference_manifest.json \
  --config configs/learned_laplacian/visibility_recovery_sofa50_expanded.json \
  --output-dir runs/learned_laplacian/sofa50_visibility_recovery_expanded_fixed_checkpoint \
  --split validation --device cuda
```

The output root contains `summary.json`, `per_mesh.csv`, and `REPORT.md`. Each mesh
directory contains the requested four OBJ files, two colored visibility PLY files,
`visibility_diagnostics.json`, `recovery_metrics.json`, and
`per_vertex_diagnostics.npz`.

## Next decision gate

Keep hard visibility gating available for future expanded recovery, but do not treat
it as proof that learned reconstruction is useful. Before formal retraining, improve
valid image dependence under renderer visibility, investigate the 1–2-view group and
zero-view coverage, and define a real expanded-graph oracle. Learned Gaussian
uncertainty can then be compared against this hard-mask baseline; renderer visibility
must remain a strict zero-precision gate.
