# Sofa50 anchor-conditioning ablation

## 1. EXECUTION STATUS

**completed**. Exactly one new model, Arm B_P, was trained; A, B_0 and E remained frozen.

## 2. CONTRACT AUDIT

Contract audit: **true**. B_P copies B_0's architecture, raw target, Huber loss, beta=0.01, lambda_train=0.01, optimizer, schedule, 20k-step budget, split, views, resolution and seed. The only methodological change is the recovery-loss anchor from V0 to the cached detached frozen Arm-E prediction V_P. B_P used four L40 ranks with accumulation two (global batch eight); this execution layout differs from B_0's historical allocation and is therefore recorded rather than described as a perfectly identical hardware execution.

## 3. TRAINING SUMMARY

- Selected checkpoint: `/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/runs/learned_laplacian/sofa50_v2_sparse_recovery_arm_bp_positional_anchor_20k_seed7/checkpoint_best.pt` (SHA-256 `44e36cd8fc7df98b734af6e7883ffa33b4c0ed70ab4590a9768e0ec9e8f9df19`).
- Selected epoch/step: `400` / `None`.
- Gradient audit passed: `True`; delta gradient norm `0.007224015892`; E gradients present `False`.
- Validation B_P@V_P lambda=0.01: CD `0.002556890762`, P2S p95 `0.007465585555`, F-score `0.9766339378`, Normal `0.9708400954`, VRMS `0.006219008543`.
- Evaluation PCG: failed solves `0`; maximum relative residual `9.994806826e-05`.

## 4. CROSS-ANCHOR MATRIX (test, lambda=0.01)

| Field source | V0: CD / P2S p95 / F / Normal / VRMS | V_P: CD / P2S p95 / F / Normal / VRMS |
|---|---:|---:|
| A | 0.003957559733 / 0.01225215352 / 0.9172248817 / 0.9548748061 / 0.01352081646 | 0.003141655253 / 0.009566528534 / 0.9516877376 / 0.9585014819 / 0.01089578817 |
| B_0 | 0.003596496885 / 0.01057761424 / 0.9344499136 / 0.959314934 / 0.01156872866 | 0.00319840408 / 0.009761256567 / 0.9494629161 / 0.9559537973 / 0.01104777405 |
| B_P | 0.003952952343 / 0.01228242028 / 0.9152487925 / 0.9586577095 / 0.01278357865 | 0.003181370468 / 0.009811770689 / 0.9498324068 / 0.9628328986 / 0.009868669002 |

Secondary V_P-column results at lambda=0.03 are recorded in `aggregate_metrics.csv`.

## 5. PRIMARY PAIRED RESULTS

Differences are candidate minus reference.

| Comparison | CD mean [mesh CI] [object CI] | W/T/L |
|---|---:|---:|
| B_P@V_P vs B_0@V_P | -1.703361162e-05 [-0.0001463599357, 0.0001160982458] [-9.729385426e-05, 5.942411055e-05] | 30/0/20 |
| B_P@V0 vs B_0@V0 | 0.0003564554578 [0.0002240721166, 0.0004961669704] [0.0001129674736, 0.0007087710751] | 14/0/36 |
| B_P@V_P vs A@V_P | 3.971521506e-05 [-4.592922275e-05, 0.0001242312426] [-1.557614011e-05, 9.844661194e-05] | 23/0/27 |

## 6. ANCHOR-INTERACTION RESULT

For CD, `interaction=(B_P@V_P-B_0@V_P)-(B_P@V0-B_0@V0)` is `-0.0003734890694` with mesh CI `[-0.0005399646818, -0.0002258978035]` and object-cluster CI `[-0.0007209731421, -0.0001391325594]`. Negative values favor anchor-specific conditioning.

## 7. VERDICT

**NO EVIDENCE FOR ANCHOR CONDITIONING**

## 8. PAPER IMPLICATION

This experiment isolates the recovery anchor used during reconstruction-mediated differential training. The paired same-anchor comparison and the paired interaction determine whether any B_P gain is specifically associated with V_P rather than a general improvement. Raw differential metrics are reported separately and do not determine the recovery verdict. The result is scoped to frozen single-pass Sofa50-v2 recovery at the tested lambdas; no paper file was modified.

## Reproducibility

- Evaluator: `mlr.learned_laplacian.evaluation.evaluate_mesh_geometry;area_weighted_triangle_surface_sampling;bidirectional_sampled_surface_to_exact_triangle_surface;surface_samples=3000;seed=7;fscore_threshold=0.01;alignment=shared_prepared_coordinate_frame_no_ICP`.
- Bootstrap: `10000` replicates, seed `7`.
- Git HEAD: `39aa6d5920e0eae0ea9b0d380403da9d78a9754d`.
- Evaluation command: `scripts/evaluate_sofa50_anchor_conditioning_ablation.py --manifest /networkhome/WMGDS/zhou_c/Sofa50MultiTopologyRawLap500_v2/manifest.json --arm-ab-report /networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/reports/sofa50_multitopology_rawlap500_v2/recovery_aware_two_arm_ablation/final --arm-e-report /networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/reports/sofa50_multitopology_rawlap500_v2/direct_vertex_arm_e_extension/final --bp-run /networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/runs/learned_laplacian/sofa50_v2_sparse_recovery_arm_bp_positional_anchor_20k_seed7 --bp-checkpoint /networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/runs/learned_laplacian/sofa50_v2_sparse_recovery_arm_bp_positional_anchor_20k_seed7/checkpoint_best.pt --output-dir /networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/reports/sofa50_multitopology_rawlap500_v2/anchor_conditioning_ablation_v1 --device cpu --bootstrap-replicates 10000 --seed 7`.
