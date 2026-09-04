# Matched Direct-Lap A+E versus Proposed B+E at lambda=0.01

## Execution status

**completed**. Saved frozen predictions were evaluated on local CPU; no training, network inference, or HPC submission occurred.

## Match validation

Contract audit: **true with one provenance warning**. Both systems use the identical E array, ordered 50-mesh test set, GT, topology, Uniform random-walk operator, float64 PCG, `lambda=0.01`, and evaluator.
Both solve `(L_U^T L_U+0.01 I)V=L_U^T delta+0.01 V_P`, differing only in whether `delta` comes from frozen Arm A or Arm B.
The archived B+E lambda=0.01 result was independently reproduced; maximum CD discrepancy was `0.000e+00` and topology counts matched exactly.
Arm-A checkpoint direct rehash remains unavailable locally; its archived prediction metadata declares the required SHA and all array/ID/target/raw-EPE checks pass.

## Aggregate results

| Method | CD | P2S p95 | F-score | Normal | VRMS | Flips | Improved/worsened | Runtime s/mesh |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct-Lap A+E, lambda=0.01 | 0.003141655253 | 0.009566528534 | 0.9516877376 | 0.9585014819 | 0.01089578817 | 55753 | 45/5 | 0.720338 |
| Proposed B+E, lambda=0.01 | 0.00319840408 | 0.009761256567 | 0.9494629161 | 0.9559537973 | 0.01104777405 | 63863 | 44/6 | 0.706350 |

Arm-A raw EPE is `0.0025264054` and Arm-B raw EPE is `0.0026398567`; these are differential-field diagnostics, not fused-output metrics.

The exact paired outputs are indexed by the [result-mesh bundle README](result_meshes/README.md) and its SHA-256 manifest. The 100 OBJ payloads remain reproducible local artifacts and are intentionally excluded from Git because the bundle is approximately 252 MiB.

## Paired A+E versus B+E

Differences are A+E minus B+E. Positive CD/P2S/VRMS and negative F-score/Normal favor B+E.

| Metric | Mean difference | Median difference | A+E W/T/L | Mesh 95% CI | Object-cluster 95% CI |
|---|---:|---:|---:|---:|---:|
| refined_chamfer | -5.674882668e-05 | -0.0001171323761 | 34/0/16 | [-0.0001786321572, 6.984533747e-05] | [-0.0001501461962, 3.481065987e-05] |
| p2s_p95 | -0.0001947280332 | -0.0001536433667 | 28/0/22 | [-0.000987921987, 0.0006435609988] | [-0.0009682619258, 0.0007487735613] |
| fscore | 0.002224821489 | 0.001257646857 | 28/2/20 | [-0.003678111392, 0.008225876406] | [-0.0007471624055, 0.004560961355] |
| normal_consistency | 0.002547684594 | 0.002318685069 | 41/0/9 | [0.001554478885, 0.003578440386] | [0.002002809995, 0.003350146746] |
| same_index_recovered_vertex_rms | -0.0001519858778 | -0.0001750969481 | 32/0/18 | [-0.0006186206907, 0.0003268094408] | [-0.001005390002, 0.001038186684] |

## Verdict

**NO MEANINGFUL DIFFERENCE**

## Result mesh exports

All 50 paired outputs are available under [`result_meshes/`](result_meshes/README.md). Each sample directory contains `A_plus_E_lambda1e2.obj` and `B_plus_E_lambda1e2.obj` with the original input connectivity. The export reproduced every archived per-sample VRMS exactly; all face arrays survived the OBJ round trip unchanged, and the maximum vertex-coordinate round-trip error was `5.000e-09`. Exact paths, SHA-256 hashes, topology counts, and solver audits are recorded in [`result_meshes/MANIFEST.json`](result_meshes/MANIFEST.json).

## Reproducibility

- Git HEAD: `1b70e5a7162e18d6f8ac00eebc3ca11bce6ec6e9`.
- Manifest: `/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/multiview-laplacian-refinement/.external/Sofa50MultiTopologyRawLap500_v2/manifest.json`; SHA-256 `6924e28f7e0845e65e670b447261fc9d1541cf24bf49861fe615c31399667fb0`.
- Command: `PYTHONPATH=src:scripts conda run --no-capture-output -n test python scripts/evaluate_sofa50_direct_lap_positional_matched_fusion_lambda1e2.py --manifest .external/Sofa50MultiTopologyRawLap500_v2/manifest.json --arm-ab-report reports/sofa50_multitopology_rawlap500_v2/recovery_aware_two_arm_ablation/final --arm-e-report reports/sofa50_multitopology_rawlap500_v2/direct_vertex_arm_e_extension/final --b-e-reference-report reports/sofa50_multitopology_rawlap500_v2/original_b_e_fusion_lambda1e2_v1 --arm-b-checkpoint runs/learned_laplacian/sofa50_v2_sparse_recovery_arm_b_recovery_aware_20k_seed7/checkpoint_best.pt --arm-e-checkpoint runs/learned_laplacian/sofa50_v2_direct_vertex_arm_e_20k_seed7/checkpoint_best.pt --output-dir reports/sofa50_multitopology_rawlap500_v2/direct_lap_positional_matched_fusion_lambda1e2_v1 --device cpu`.
- Bootstrap: `10000` replicates, seed `7`.
- Total wall time: `152.164` seconds.
- Full per-mesh, per-object, solver, checkpoint, and artifact audits are stored beside this report.
