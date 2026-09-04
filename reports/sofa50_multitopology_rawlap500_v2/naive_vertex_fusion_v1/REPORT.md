# Naive scalar vertex fusion — matched_v2

Contract audit: **true**.

The predictors and reconstruction contracts are frozen. The scalar coefficient was selected only on validation by minimum macro mean CD, then evaluated once on test.

Validation-selected `alpha* = 0.31`.

The selected validation point and the complete 101-point multi-metric curve are archived in `selection_lock.json` and `validation_alpha_sweep.csv`.

![Validation alpha sweep](validation_alpha_cd.png)

| Method | CD | P2S p95 | F-score | Normal | Vertex RMS | Improved/worsened |
|---|---:|---:|---:|---:|---:|---:|
| Initial mesh | 0.0043863516 | 0.0146957304 | 0.901781516 | 0.969623498 | 0.0425195806 | 0/0 |
| Operator-Mediated Differential | 0.0035849702 | 0.0105580821 | 0.935012989 | 0.959365744 | 0.0115531855 | 36/14 |
| Direct Positional | 0.0033403882 | 0.0103976753 | 0.943048517 | 0.970111650 | 0.0082212991 | 45/5 |
| Naive scalar fusion | 0.0031881427 | 0.0097184416 | 0.951832862 | 0.972382154 | 0.0081156234 | 48/2 |
| Proposed operator Hybrid | 0.0030298326 | 0.0093658805 | 0.956291439 | 0.962734888 | 0.0092334079 | 49/1 |

## Paired CD comparisons

Differences are candidate minus reference; negative values favor the candidate.

| Candidate vs reference | Mean difference [mesh 95% CI] | Object-cluster 95% CI | W/L/T |
|---|---:|---:|---:|
| Naive scalar fusion vs Direct Positional | -0.0001522455 [-0.0002113306, -0.0000903153] | [-0.0002176225, -0.0000830808] | 41/9/0 |
| Naive scalar fusion vs Operator-Mediated Differential | -0.0003968275 [-0.0005955541, -0.0002312840] | [-0.0006071973, -0.0002042889] | 37/13/0 |
| Proposed operator Hybrid vs Naive scalar fusion | -0.0001583101 [-0.0002334686, -0.0000747634] | [-0.0002575031, -0.0000783528] | 43/7/0 |

On mean paired test CD, the proposed Hybrid **outperforms** the locked naive scalar fusion by `-0.0001583101` (Hybrid minus naive).

Metric protocol: `mlr.learned_laplacian.evaluation.evaluate_mesh_geometry;area_weighted_triangle_surface_sampling;bidirectional_sampled_surface_to_exact_triangle_surface;surface_samples=3000;seed=7;fscore_threshold=0.01;alignment=shared_prepared_coordinate_frame_no_ICP`.
