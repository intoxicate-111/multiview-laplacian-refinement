# Matched Direct-Lap Arm-A + Direct-Positional Arm-E fusion

## 1. EXECUTION STATUS

**completed**. The evaluation reused saved frozen predictions on local CPU; no model was trained, no network inference ran, and no HPC job was submitted.

## 2. MATCH VALIDATION

Contract audit: **true with one provenance warning**. A+E and the locally reproduced B+E use the identical Arm-E displacement array, sample-specific Uniform random-walk Laplacian, `lambda=0.03`, float64 PCG implementation (`tol=1e-4`, maximum 2048 iterations), same ordered 50 meshes, same GT, and the same geometry evaluator.
The operator is `L_U=I-D^{-1}A`, and both systems solve `(L_U^T L_U+0.03 I)V=L_U^T delta+0.03 V_P` on each sample's fixed topology.
The evaluator contract is `mlr.learned_laplacian.evaluation.evaluate_mesh_geometry;area_weighted_triangle_surface_sampling;bidirectional_sampled_surface_to_exact_triangle_surface;surface_samples=3000;seed=7;fscore_threshold=0.01;alignment=shared_prepared_coordinate_frame_no_ICP`. The archived B+E result was independently reproduced; maximum CD discrepancy was `3.980e-09` and topology counts matched exactly.
The Arm-A checkpoint file itself is no longer present locally, so it could not be directly rehashed; the reused Arm-A prediction archive is tied to archived metadata declaring the required checkpoint SHA, its IDs/order match the manifest exactly, its raw target is byte-identical to Arm B's, and its raw EPE reproduces the archived value.

## 3. MAIN TABLE

| Method | CD | P2S p95 | F-score | Normal | VRMS |
|---|---:|---:|---:|---:|---:|
| Arm A standalone | 0.00395528689 | 0.01225819285 | 0.9174354413 | 0.9549020034 | 0.01351805373 |
| Arm B standalone | 0.003584970226 | 0.01055808211 | 0.9350129892 | 0.9593657435 | 0.01155318553 |
| Arm E standalone | 0.003340388174 | 0.01039767527 | 0.9430485173 | 0.9701116497 | 0.008221299063 |
| Direct-Lap A+E, lambda=0.03 | 0.00298590286 | 0.009274695137 | 0.9563943178 | 0.9638249491 | 0.009563517302 |
| Proposed B+E, lambda=0.03 | 0.003029832983 | 0.009365880535 | 0.9562914388 | 0.9627348883 | 0.009233407942 |
| Pure-Vertex+E, lambda=0.03 | 0.003925840933 | 0.01193782633 | 0.9193390933 | 0.9583599319 | 0.01058685599 |
| Scalar fusion, alpha=0.31 | 0.00318814268 | 0.009718441593 | 0.9518328617 | 0.9723821544 | 0.008115623445 |

Direct-Lap A+E introduced `51035` fixed-connectivity flips in total (normalized rate `0.03014612`), with `0` new degenerate faces. Mean local runtime was `0.661987` s/mesh (PCG plus evaluator only).
Arm-A raw differential EPE is `0.0025264054`; Arm-B raw differential EPE is `0.0026398567`. These are predictor diagnostics, not fused-output metrics.

## 4. DIRECT A+E VS B+E

All differences are A+E minus B+E. Positive CD/P2S/VRMS and negative F-score/Normal favor Proposed B+E.

| Metric | Mean difference | Median difference | A+E W/T/L | Mesh bootstrap 95% CI | Object-cluster bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| refined_chamfer | -4.393012333e-05 | -2.50130814e-05 | 27/0/23 | [-0.0001327670009, 4.549019158e-05] | [-9.815857697e-05, 1.029833031e-05] |
| p2s_p95 | -9.118539786e-05 | 3.549124255e-05 | 24/0/26 | [-0.000738732811, 0.0005656579541] | [-0.0007913884574, 0.0005363085257] |
| fscore | 0.0001028790168 | -4.897169327e-05 | 21/4/25 | [-0.004981925737, 0.005211279525] | [-0.002208901128, 0.002414659161] |
| normal_consistency | 0.001090060791 | 0.001116559153 | 34/0/16 | [0.0003470880423, 0.001854184594] | [0.0003884927195, 0.002311368661] |
| same_index_recovered_vertex_rms | 0.0003301093601 | 4.787689856e-05 | 22/0/28 | [-6.672044593e-05, 0.0007477787827] | [-0.0004176602953, 0.001317904113] |

Per-object aggregates are in `per_object_metrics.csv`; all per-mesh values are in `per_mesh_metrics.csv`.

## 5. VERDICT

**NO MEANINGFUL DIFFERENCE**

## 6. PAPER IMPLICATION

For the primary surface-distance comparison, the matched experiment does not distinguish Direct-Lap A+E from Proposed B+E with meaningful confidence.
A+E is numerically lower in mean CD and P2S p95 and slightly higher in F-score, while its Normal advantage is statistically positive under both bootstrap units; B+E is numerically better only in VRMS among the reported metrics.
Therefore the paper cannot use this baseline to argue that Arm B's training is necessary for operator composition.
The simple-combination criticism becomes substantially stronger and the methodological novelty claim must be narrowed.
This conclusion remains scoped to matched Sofa50-v2, frozen single-pass fusion at lambda 0.03.

## Reproducibility

- Git HEAD: `1b70e5a7162e18d6f8ac00eebc3ca11bce6ec6e9`.
- Dataset manifest: `/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/multiview-laplacian-refinement/.external/Sofa50MultiTopologyRawLap500_v2/manifest.json`; SHA-256 `6924e28f7e0845e65e670b447261fc9d1541cf24bf49861fe615c31399667fb0`.
- Evaluation command: `PYTHONPATH=src:scripts conda run --no-capture-output -n test python scripts/evaluate_sofa50_direct_lap_positional_matched_fusion.py --manifest .external/Sofa50MultiTopologyRawLap500_v2/manifest.json --arm-ab-report reports/sofa50_multitopology_rawlap500_v2/recovery_aware_two_arm_ablation/final --arm-e-report reports/sofa50_multitopology_rawlap500_v2/direct_vertex_arm_e_extension/final --hybrid-report reports/sofa50_multitopology_rawlap500_v2/frozen_hybrid_recovery_v1 --pure-fusion-report reports/sofa50_multitopology_rawlap500_v2/pure_vertex_b_e_fusion_ablation_v1 --scalar-fusion-report reports/sofa50_multitopology_rawlap500_v2/naive_vertex_fusion_v1 --arm-b-checkpoint runs/learned_laplacian/sofa50_v2_sparse_recovery_arm_b_recovery_aware_20k_seed7/checkpoint_best.pt --arm-e-checkpoint runs/learned_laplacian/sofa50_v2_direct_vertex_arm_e_20k_seed7/checkpoint_best.pt --output-dir reports/sofa50_multitopology_rawlap500_v2/direct_lap_positional_matched_fusion_v1 --device cpu`.
- Bootstrap: `10000` replicates, seed `7`; 50 mesh units and five object-cluster units.
- Total wall time: `144.632` seconds.
- Checkpoint and prediction-artifact paths/hashes are recorded in `contract_audit.json`.
- Warning: Arm-A checkpoint direct rehash was unavailable locally; see the match-validation qualification above.
