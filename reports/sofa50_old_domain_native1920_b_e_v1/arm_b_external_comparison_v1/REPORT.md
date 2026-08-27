# Old-domain native-1920 Arm-B versus external methods

Contract audit: **true**.

This is a user-authorized Arm-B-only test opening, not the sealed final B/E/fusion evaluation. The Arm-B checkpoint was selected only by validation objective. NDS, nvdiffrec, and ExMesh rows are read from the existing same-input archive and were already recomputed with the identical unified evaluator.

Arm-B selected checkpoint: `/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/runs/learned_laplacian/sofa50_old_domain_native1920_arm_b_20k_seed7_v1/checkpoint_best.pt`; SHA-256 `c250afad2b63828a4b8ae3d692dca22757e6477a50d13e882461035a7a35522a`; selected epoch `744` / optimizer step `18600`.

## Unified same-input comparison

| Method | CD | P2S p95 | F-score | Normal | Improved/worsened |
|---|---:|---:|---:|---:|---:|
| Old-domain Arm B | 0.0085343303 | 0.0271235767 | 0.716657383 | 0.948320643 | 25/0 |
| NDS | 0.0112049924 | 0.0398475607 | 0.652827299 | 0.873805125 | 22/3 |
| nvdiffrec | 0.0136546593 | 0.0457457720 | 0.558673128 | 0.848122276 | 18/7 |
| ExMesh | 0.0201706152 | 0.0696287606 | 0.478513280 | 0.845337056 | 8/17 |
| Initial mesh | 0.0170704685 | 0.0724794854 | 0.577250432 | 0.955190949 | 0/0 |

## Paired Arm-B comparisons

Differences are Arm B minus comparator. Negative CD/P2S differences and positive F-score/normal differences favor Arm B.

| Comparator | CD difference [95% CI] | CD W/L/T | P2S-p95 difference | F-score difference | Normal difference |
|---|---:|---:|---:|---:|---:|
| NDS | -0.0026706620 [-0.0032912672, -0.0020934088] | 25/0/0 | -0.0127239840 | 0.063830084 | 0.074515518 |
| nvdiffrec | -0.0051203290 [-0.0058605752, -0.0043072116] | 24/1/0 | -0.0186221954 | 0.157984255 | 0.100198368 |
| ExMesh | -0.0116362849 [-0.0152515573, -0.0084157784] | 25/0/0 | -0.0425051839 | 0.238144103 | 0.102983587 |

## Audit

- Samples: `25` exact common native-1920 `v00`--`v04` inputs.
- Arm-B recovery: Uniform random-walk Laplacian, `lambda=1e-2`, float64 PCG, tolerance `1e-8`; maximum residual `9.834e-09`.
- Archived comparator reproduction: `true`.
- Metric protocol: `mlr.learned_laplacian.evaluation.evaluate_mesh_geometry;area_weighted_triangle_surface_sampling;bidirectional_sampled_surface_to_exact_triangle_surface;surface_samples=3000;seed=7;fscore_threshold=0.01;alignment=shared_prepared_coordinate_frame_no_ICP`.
- Test access occurred before old-domain Arm-E/fusion/continuous final selection; these results must not be described as a sealed full-model final test.
