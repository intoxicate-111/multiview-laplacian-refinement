# Sofa50 v2 Pure-Vertex Arm-B + frozen Arm-E fusion at lambda=1e-2

Contract audit: **true**. This is a read-only, non-recursive fixed-lambda fusion test on the exact same 50 validation and 50 test meshes. No model is retrained and no lambda search is run.

## Fixed contract

- Pure-Vertex Arm-B: `/tmp/mlr_sofa50_v2_bvonly_recursive_20260827/runs_root/sofa50_v2_sparse_recovery_arm_b_recovery_aware_20k_seed7/checkpoint_best.pt`; epoch `312`, optimizer step `15600`, SHA-256 `3f29d66302f30a487e3aac9c7c09a5875328602cbcc715f3780aa24ba5b6367a`.
- Frozen Arm-E: `/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/runs/learned_laplacian/sofa50_v2_direct_vertex_arm_e_20k_seed7/checkpoint_best.pt`; SHA-256 `6ed27da8759b7bd752ffa75ea8dac3977dd4ced358b5282e0c1c68f750dbade1`.
- Candidate solve: `min_V ||L_U V-delta_PureB||^2 + 0.01 ||V-V_E||^2`.
- Reference solve: the already evaluated Pure-B+E system with the same inputs and `lambda=0.03`.
- Uniform random-walk operator, frozen arrays, float64 PCG, evaluator, meshes, cameras, ordering, and all other settings are unchanged.
- GT enters neither predictor nor fusion solve. Test is not used for checkpoint or lambda selection.

## Aggregate results

CD gain is the macro mean of per-mesh relative improvement over initial CD. Raw EPE is the frozen Pure-B field diagnostic and is unchanged by fusion.

| Split | System | Initial CD | Refined CD | CD gain | P2S mean | P2S p95 | F-score | Normal | Raw EPE | Vertex RMS | Improved/worsened |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | Initial mesh | 0.00381765892 | 0.00381765892 | +0.00% | 0.00381765892 | 0.0138686196 | 0.918392483 | 0.976308527 | n/a | 0.031591482 | 0/0 |
| validation | Pure-Vertex Arm-B | 0.00381765892 | 0.00345892192 | +3.40% | 0.00345892192 | 0.0107392229 | 0.936345803 | 0.965681501 | 0.0076741744 | 0.00622738242 | 24/26 |
| validation | Arm-E | 0.00381765892 | 0.00285065224 | +20.05% | 0.00285065224 | 0.00891131992 | 0.960090448 | 0.975651504 | n/a | 0.00472553239 | 48/2 |
| validation | Pure-Vertex B+E (lambda=3e-2) | 0.00381765892 | 0.00336727057 | +6.96% | 0.00336727057 | 0.0104846605 | 0.940200292 | 0.963813809 | 0.0076741744 | 0.00634253795 | 31/19 |
| validation | Pure-Vertex B+E (lambda=1e-2) | 0.00381765892 | 0.00413467896 | -11.46% | 0.00413467896 | 0.0131315703 | 0.901890269 | 0.956543083 | 0.0076741744 | 0.00909805861 | 13/37 |
| test | Initial mesh | 0.00438635163 | 0.00438635163 | +0.00% | 0.00438635163 | 0.0146957304 | 0.901781516 | 0.969623498 | n/a | 0.0425195806 | 0/0 |
| test | Pure-Vertex Arm-B | 0.00438635163 | 0.00397816927 | +1.34% | 0.00397816927 | 0.0117444276 | 0.91756792 | 0.959623804 | 0.00857208259 | 0.0105424394 | 27/23 |
| test | Arm-E | 0.00438635163 | 0.00334038817 | +18.59% | 0.00334038817 | 0.0103976753 | 0.943048517 | 0.97011165 | n/a | 0.00822129906 | 45/5 |
| test | Pure-Vertex B+E (lambda=3e-2) | 0.00438635163 | 0.00392584093 | +3.93% | 0.00392584093 | 0.0119378263 | 0.919339093 | 0.958359932 | 0.00857208259 | 0.010586856 | 28/22 |
| test | Pure-Vertex B+E (lambda=1e-2) | 0.00438635163 | 0.00469272235 | -14.95% | 0.00469272235 | 0.0140863565 | 0.887183555 | 0.949796227 | 0.00857208259 | 0.0139116846 | 16/34 |

## Paired comparisons

Differences are candidate minus reference. Negative CD/P2S/raw-EPE/vertex-RMS and positive F-score/normal favor the candidate. Confidence intervals bootstrap meshes.

| Split | Comparison | Metric | Mean difference [95% CI] | Candidate W/L/T |
|---|---|---|---:|---:|
| validation | Pure-Vertex B+E (lambda=1e-2) vs Pure-Vertex B+E (lambda=3e-2) | Refined CD | 0.000767408394 [0.000597651336, 0.000961247894] | 0/50/0 |
| validation | Pure-Vertex B+E (lambda=1e-2) vs Pure-Vertex B+E (lambda=3e-2) | P2S p95 | 0.00264690979 [0.00182091382, 0.00361160324] | 1/49/0 |
| validation | Pure-Vertex B+E (lambda=1e-2) vs Pure-Vertex B+E (lambda=3e-2) | F-score | -0.038310023 [-0.0506702996, -0.0273264415] | 0/50/0 |
| validation | Pure-Vertex B+E (lambda=1e-2) vs Pure-Vertex B+E (lambda=3e-2) | Normal | -0.00727072685 [-0.00805895471, -0.00648733444] | 0/50/0 |
| validation | Pure-Vertex B+E (lambda=1e-2) vs Pure-Vertex B+E (lambda=3e-2) | Raw EPE | 0 [0, 0] | 0/0/50 |
| validation | Pure-Vertex B+E (lambda=1e-2) vs Pure-Vertex B+E (lambda=3e-2) | Vertex RMS | 0.00275552066 [0.00247441553, 0.00304000865] | 0/50/0 |
| test | Pure-Vertex B+E (lambda=1e-2) vs Pure-Vertex B+E (lambda=3e-2) | Refined CD | 0.000766881413 [0.000609606024, 0.000966560277] | 0/50/0 |
| test | Pure-Vertex B+E (lambda=1e-2) vs Pure-Vertex B+E (lambda=3e-2) | P2S p95 | 0.0021485302 [0.00163041426, 0.00272889629] | 3/47/0 |
| test | Pure-Vertex B+E (lambda=1e-2) vs Pure-Vertex B+E (lambda=3e-2) | F-score | -0.0321555378 [-0.0413032758, -0.0244528413] | 2/48/0 |
| test | Pure-Vertex B+E (lambda=1e-2) vs Pure-Vertex B+E (lambda=3e-2) | Normal | -0.00856370477 [-0.00963962628, -0.00753836597] | 0/50/0 |
| test | Pure-Vertex B+E (lambda=1e-2) vs Pure-Vertex B+E (lambda=3e-2) | Raw EPE | 0 [0, 0] | 0/0/50 |
| test | Pure-Vertex B+E (lambda=1e-2) vs Pure-Vertex B+E (lambda=3e-2) | Vertex RMS | 0.0033248286 [0.00285097548, 0.00385426127] | 0/50/0 |
| validation | Pure-Vertex B+E (lambda=1e-2) vs Pure-Vertex Arm-B | Refined CD | 0.00067575704 [0.000466842479, 0.000919372955] | 1/49/0 |
| validation | Pure-Vertex B+E (lambda=1e-2) vs Pure-Vertex Arm-B | P2S p95 | 0.00239234737 [0.00142884142, 0.00353955699] | 9/41/0 |
| validation | Pure-Vertex B+E (lambda=1e-2) vs Pure-Vertex Arm-B | F-score | -0.0344555346 [-0.0481573433, -0.0221523782] | 5/45/0 |
| validation | Pure-Vertex B+E (lambda=1e-2) vs Pure-Vertex Arm-B | Normal | -0.00913841859 [-0.0104391607, -0.00790660954] | 1/49/0 |
| validation | Pure-Vertex B+E (lambda=1e-2) vs Pure-Vertex Arm-B | Raw EPE | 0 [0, 0] | 0/0/50 |
| validation | Pure-Vertex B+E (lambda=1e-2) vs Pure-Vertex Arm-B | Vertex RMS | 0.00287067618 [0.00245406278, 0.0033048675] | 0/50/0 |
| test | Pure-Vertex B+E (lambda=1e-2) vs Pure-Vertex Arm-B | Refined CD | 0.000714553073 [0.000538051493, 0.000935153276] | 1/49/0 |
| test | Pure-Vertex B+E (lambda=1e-2) vs Pure-Vertex Arm-B | P2S p95 | 0.00234192891 [0.00169373745, 0.00304608393] | 6/44/0 |
| test | Pure-Vertex B+E (lambda=1e-2) vs Pure-Vertex Arm-B | F-score | -0.0303843646 [-0.0413703512, -0.0213625468] | 6/44/0 |
| test | Pure-Vertex B+E (lambda=1e-2) vs Pure-Vertex Arm-B | Normal | -0.00982757701 [-0.0114699175, -0.00827487231] | 0/50/0 |
| test | Pure-Vertex B+E (lambda=1e-2) vs Pure-Vertex Arm-B | Raw EPE | 0 [0, 0] | 0/0/50 |
| test | Pure-Vertex B+E (lambda=1e-2) vs Pure-Vertex Arm-B | Vertex RMS | 0.0033692452 [0.00265556743, 0.00414051316] | 3/47/0 |
| validation | Pure-Vertex B+E (lambda=3e-2) vs Pure-Vertex Arm-B | Refined CD | -9.16513542e-05 [-0.00015279871, -2.00907837e-05] | 41/9/0 |
| validation | Pure-Vertex B+E (lambda=3e-2) vs Pure-Vertex Arm-B | P2S p95 | -0.00025456242 [-0.000475619801, 1.23882343e-05] | 36/14/0 |
| validation | Pure-Vertex B+E (lambda=3e-2) vs Pure-Vertex Arm-B | F-score | 0.00385448839 [-0.0005820607, 0.00767516999] | 35/15/0 |
| validation | Pure-Vertex B+E (lambda=3e-2) vs Pure-Vertex Arm-B | Normal | -0.00186769174 [-0.0027432728, -0.00104150176] | 17/33/0 |
| validation | Pure-Vertex B+E (lambda=3e-2) vs Pure-Vertex Arm-B | Raw EPE | 0 [0, 0] | 0/0/50 |
| validation | Pure-Vertex B+E (lambda=3e-2) vs Pure-Vertex Arm-B | Vertex RMS | 0.000115155526 [-0.000106514701, 0.000334448967] | 21/29/0 |
| test | Pure-Vertex B+E (lambda=3e-2) vs Pure-Vertex Arm-B | Refined CD | -5.23283401e-05 [-0.000119921627, 1.54754304e-05] | 33/17/0 |
| test | Pure-Vertex B+E (lambda=3e-2) vs Pure-Vertex Arm-B | P2S p95 | 0.000193398711 [-0.000147266123, 0.000548681086] | 27/23/0 |
| test | Pure-Vertex B+E (lambda=3e-2) vs Pure-Vertex Arm-B | F-score | 0.00177117321 [-0.00219090404, 0.0059217377] | 33/17/0 |
| test | Pure-Vertex B+E (lambda=3e-2) vs Pure-Vertex Arm-B | Normal | -0.00126387224 [-0.00207906072, -0.000494973918] | 17/33/0 |
| test | Pure-Vertex B+E (lambda=3e-2) vs Pure-Vertex Arm-B | Raw EPE | 0 [0, 0] | 0/0/50 |
| test | Pure-Vertex B+E (lambda=3e-2) vs Pure-Vertex Arm-B | Vertex RMS | 4.4416607e-05 [-0.000250198157, 0.000337823412] | 28/22/0 |
| validation | Pure-Vertex B+E (lambda=1e-2) vs Arm-E | Refined CD | 0.00128402672 [0.00104336233, 0.00155152815] | 0/50/0 |
| validation | Pure-Vertex B+E (lambda=1e-2) vs Arm-E | P2S p95 | 0.0042202504 [0.0032637859, 0.00532756855] | 0/50/0 |
| validation | Pure-Vertex B+E (lambda=1e-2) vs Arm-E | F-score | -0.0582001795 [-0.0736714573, -0.0439580685] | 2/48/0 |
| validation | Pure-Vertex B+E (lambda=1e-2) vs Arm-E | Normal | -0.0191084216 [-0.0207051323, -0.0175421212] | 0/50/0 |
| validation | Pure-Vertex B+E (lambda=1e-2) vs Arm-E | Vertex RMS | 0.00437252621 [0.00391842015, 0.00484577395] | 0/50/0 |
| test | Pure-Vertex B+E (lambda=1e-2) vs Arm-E | Refined CD | 0.00135233417 [0.00112733129, 0.00161764699] | 0/50/0 |
| test | Pure-Vertex B+E (lambda=1e-2) vs Arm-E | P2S p95 | 0.00368868127 [0.00290083455, 0.00450023024] | 1/49/0 |
| test | Pure-Vertex B+E (lambda=1e-2) vs Arm-E | F-score | -0.0558649618 [-0.0687260383, -0.0441744178] | 3/47/0 |
| test | Pure-Vertex B+E (lambda=1e-2) vs Arm-E | Normal | -0.0203154226 [-0.0222365698, -0.0184915353] | 0/50/0 |
| test | Pure-Vertex B+E (lambda=1e-2) vs Arm-E | Vertex RMS | 0.00569038553 [0.00483893852, 0.00662773912] | 0/50/0 |
| validation | Pure-Vertex B+E (lambda=3e-2) vs Arm-E | Refined CD | 0.000516618327 [0.000426189551, 0.000616872295] | 1/49/0 |
| validation | Pure-Vertex B+E (lambda=3e-2) vs Arm-E | P2S p95 | 0.00157334061 [0.00122307509, 0.00195149657] | 3/47/0 |
| validation | Pure-Vertex B+E (lambda=3e-2) vs Arm-E | F-score | -0.0198901565 [-0.0255457381, -0.0148467461] | 3/47/0 |
| validation | Pure-Vertex B+E (lambda=3e-2) vs Arm-E | Normal | -0.0118376948 [-0.0130214845, -0.0107206252] | 0/50/0 |
| validation | Pure-Vertex B+E (lambda=3e-2) vs Arm-E | Vertex RMS | 0.00161700556 [0.00136859988, 0.00188878856] | 0/50/0 |
| test | Pure-Vertex B+E (lambda=3e-2) vs Arm-E | Refined CD | 0.000585452759 [0.000476926777, 0.00070171078] | 1/49/0 |
| test | Pure-Vertex B+E (lambda=3e-2) vs Arm-E | P2S p95 | 0.00154015106 [0.00103134663, 0.00207021732] | 9/41/0 |
| test | Pure-Vertex B+E (lambda=3e-2) vs Arm-E | F-score | -0.023709424 [-0.0308376663, -0.0170315157] | 7/43/0 |
| test | Pure-Vertex B+E (lambda=3e-2) vs Arm-E | Normal | -0.0117517178 [-0.0131102193, -0.0104334982] | 0/50/0 |
| test | Pure-Vertex B+E (lambda=3e-2) vs Arm-E | Vertex RMS | 0.00236555693 [0.00182063364, 0.00298465378] | 4/46/0 |

## Fusion gain and lambda effect

Positive gain favors fusion over standalone Pure-B. The gain ratio is interpreted only when the `0.03` fusion gain is positive.

| Split | Metric | Gain at 0.01 | Gain at 0.03 | 0.01/0.03 gain ratio | Value(0.01)-Value(0.03) |
|---|---|---:|---:|---:|---:|
| validation | Refined CD | -0.00067575704 | 9.16513542e-05 | -737.31% | 0.000767408394 |
| validation | P2S p95 | -0.00239234737 | 0.00025456242 | -939.79% | 0.00264690979 |
| validation | F-score | -0.0344555346 | 0.00385448839 | -893.91% | -0.038310023 |
| validation | Normal | -0.00913841859 | -0.00186769174 | n/a | -0.00727072685 |
| test | Refined CD | -0.000714553073 | 5.23283401e-05 | -1365.52% | 0.000766881413 |
| test | P2S p95 | -0.00234192891 | -0.000193398711 | n/a | 0.0021485302 |
| test | F-score | -0.0303843646 | 0.00177117321 | -1715.49% | -0.0321555378 |
| test | Normal | -0.00982757701 | -0.00126387224 | n/a | -0.00856370477 |

## Decision

Classification: **PURE_LAMBDA_1E2_WORSE_THAN_3E2**.

Test CD is `0.00469272235` at Pure-B+E `lambda=0.01` versus `0.00392584093` at `lambda=0.03`. The paired difference is `0.000766881413` with 95% CI `[0.000609606024, 0.000966560277]` and W/L/T `0/50/0`.

This fixed-lambda diagnostic does not change any formal selected system and makes no claim about Original B, recursion, Future2000, or old native-1920.

## Numerical audit

- Maximum initial-metric discrepancy: `0.000e+00`.
- Exact sample identity/order and checkpoint hashes passed for Pure-B, Arm-E, the lambda=0.03 reference, and the manifest.
- All 100 new float64 PCG solves converged at tolerance `1e-4`, maximum `2048` iterations.
- Maximum relative residual: `9.973e-05`; iterations mean/max: `41.70/48`; new degenerate faces: `0`.
- Execution device: `cpu`; no inference or fusion hyperparameter other than the declared fixed lambda changed.
