# Sofa50 v2 Pure-Vertex Arm-B + frozen Arm-E fusion ablation

Contract audit: **true**. This is a read-only, non-recursive, exact paired comparison on 50 validation and 50 test meshes. No model is retrained and no Pure-B-specific hyperparameter is selected.

## Fixed contract

- Pure-Vertex Arm-B: `/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/runs/learned_laplacian/sofa50_v2_arm_b_recovery_only_lambda1e-2_20k_seed7_bw4_v1/checkpoint_best.pt`; epoch `312`, optimizer step `15600`, SHA-256 `3f29d66302f30a487e3aac9c7c09a5875328602cbcc715f3780aa24ba5b6367a`.
- Original Arm-B: `/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/runs/learned_laplacian/sofa50_v2_sparse_recovery_arm_b_recovery_aware_20k_seed7/checkpoint_best.pt`; SHA-256 `a483e2212f568e771873594cf1e37d13d62cbd2e1e72244baded7dd15573970c`.
- Frozen Arm-E: `/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/runs/learned_laplacian/sofa50_v2_direct_vertex_arm_e_20k_seed7/checkpoint_best.pt`; SHA-256 `6ed27da8759b7bd752ffa75ea8dac3977dd4ced358b5282e0c1c68f750dbade1`; its archived predictions are reused unchanged.
- Standalone B recovery retains `lambda=1e-2`. Fusion retains the existing Original B+E validation-selected `lambda=3e-2`; no lambda sweep is run for Pure-B.
- Fusion is the unchanged solve `min_V ||L_U V-delta_B||^2 + 0.03 ||V-V_E||^2`, using the same Uniform random-walk operator and float64 PCG implementation.
- The existing Original B+E per-sample artifacts are reused read-only. GT enters neither predictor nor fusion solve.

The two lambda values are distinct established roles: `1e-2` is the standalone Arm-B input-anchor recovery, while `3e-2` is the frozen B+E positional-anchor fusion weight. Using `1e-2` for fusion would change the existing Original B+E system and violate the requested checkpoint-only ablation.

## Aggregate results

CD gain is the macro mean of each mesh's relative improvement over its initial CD. Raw EPE is the B-field diagnostic and is therefore inherited unchanged by the corresponding B+E row.

| Split | System | Initial CD | Refined CD | CD gain | P2S mean | P2S p95 | F-score | Normal | Raw EPE | Vertex RMS | Improved/worsened |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | Initial mesh | 0.00381765892 | 0.00381765892 | +0.00% | 0.00381765892 | 0.0138686196 | 0.918392483 | 0.976308527 | n/a | 0.031591482 | 0/0 |
| validation | Original Arm-B | 0.00381765892 | 0.00320962349 | +15.46% | 0.00320962349 | 0.00999485296 | 0.950224425 | 0.96877577 | 0.0020152807 | 0.00767549059 | 46/4 |
| validation | Pure-Vertex Arm-B | 0.00381765892 | 0.00345892192 | +3.40% | 0.00345892192 | 0.0107392229 | 0.936345803 | 0.965681501 | 0.0076741744 | 0.00622738242 | 24/26 |
| validation | Original Arm-B + Arm-E | 0.00381765892 | 0.00244916745 | +31.80% | 0.00244916745 | 0.00713319482 | 0.980307047 | 0.97009371 | 0.0020152807 | 0.00542431207 | 50/0 |
| validation | Pure-Vertex Arm-B + Arm-E | 0.00381765892 | 0.00336727057 | +6.96% | 0.00336727057 | 0.0104846605 | 0.940200292 | 0.963813809 | 0.0076741744 | 0.00634253795 | 31/19 |
| test | Initial mesh | 0.00438635163 | 0.00438635163 | +0.00% | 0.00438635163 | 0.0146957304 | 0.901781516 | 0.969623498 | n/a | 0.0425195806 | 0/0 |
| test | Original Arm-B | 0.00438635163 | 0.00358497023 | +13.04% | 0.00358497023 | 0.0105580821 | 0.935012989 | 0.959365744 | 0.00263985669 | 0.0115531855 | 36/14 |
| test | Pure-Vertex Arm-B | 0.00438635163 | 0.00397816927 | +1.34% | 0.00397816927 | 0.0117444276 | 0.91756792 | 0.959623804 | 0.00857208259 | 0.0105424394 | 27/23 |
| test | Original Arm-B + Arm-E | 0.00438635163 | 0.00302983298 | +26.73% | 0.00302983298 | 0.00936588053 | 0.956291439 | 0.962734888 | 0.00263985669 | 0.00923340794 | 49/1 |
| test | Pure-Vertex Arm-B + Arm-E | 0.00438635163 | 0.00392584093 | +3.93% | 0.00392584093 | 0.0119378263 | 0.919339093 | 0.958359932 | 0.00857208259 | 0.010586856 | 28/22 |

## Paired comparisons

Differences are candidate minus reference. Negative CD/P2S/raw-EPE/vertex-RMS and positive F-score/normal favor the candidate. Confidence intervals bootstrap meshes.

| Split | Comparison | Metric | Mean difference [95% CI] | Candidate W/L/T |
|---|---|---|---:|---:|
| validation | Pure-Vertex Arm-B + Arm-E vs Original Arm-B + Arm-E | Refined CD | 0.000918103114 [0.000811959145, 0.00102991461] | 0/50/0 |
| validation | Pure-Vertex Arm-B + Arm-E vs Original Arm-B + Arm-E | P2S p95 | 0.00335146571 [0.0028671687, 0.00389775521] | 0/50/0 |
| validation | Pure-Vertex Arm-B + Arm-E vs Original Arm-B + Arm-E | F-score | -0.0401067556 [-0.0481587889, -0.0324772438] | 0/50/0 |
| validation | Pure-Vertex Arm-B + Arm-E vs Original Arm-B + Arm-E | Normal | -0.00627990055 [-0.0074766797, -0.00508772524] | 5/45/0 |
| validation | Pure-Vertex Arm-B + Arm-E vs Original Arm-B + Arm-E | Raw EPE | 0.00561309645 [0.00548097855, 0.00574025929] | 0/50/0 |
| validation | Pure-Vertex Arm-B + Arm-E vs Original Arm-B + Arm-E | Vertex RMS | 0.000918225882 [0.000682415071, 0.00119125046] | 5/45/0 |
| test | Pure-Vertex Arm-B + Arm-E vs Original Arm-B + Arm-E | Refined CD | 0.000896007949 [0.000778623463, 0.00101479267] | 0/50/0 |
| test | Pure-Vertex Arm-B + Arm-E vs Original Arm-B + Arm-E | P2S p95 | 0.00257194579 [0.00211835004, 0.00304965991] | 1/49/0 |
| test | Pure-Vertex Arm-B + Arm-E vs Original Arm-B + Arm-E | F-score | -0.0369523456 [-0.045832823, -0.0284328479] | 2/48/0 |
| test | Pure-Vertex Arm-B + Arm-E vs Original Arm-B + Arm-E | Normal | -0.00437495642 [-0.00516928259, -0.00355782146] | 6/44/0 |
| test | Pure-Vertex Arm-B + Arm-E vs Original Arm-B + Arm-E | Raw EPE | 0.00587576449 [0.00569190362, 0.00605583645] | 0/50/0 |
| test | Pure-Vertex Arm-B + Arm-E vs Original Arm-B + Arm-E | Vertex RMS | 0.00135344805 [0.000832404914, 0.00194242057] | 10/40/0 |
| validation | Pure-Vertex Arm-B + Arm-E vs Pure-Vertex Arm-B | Refined CD | -9.16513542e-05 [-0.00015279871, -2.00907837e-05] | 41/9/0 |
| validation | Pure-Vertex Arm-B + Arm-E vs Pure-Vertex Arm-B | P2S p95 | -0.00025456242 [-0.000475619801, 1.23882343e-05] | 36/14/0 |
| validation | Pure-Vertex Arm-B + Arm-E vs Pure-Vertex Arm-B | F-score | 0.00385448839 [-0.0005820607, 0.00767516999] | 35/15/0 |
| validation | Pure-Vertex Arm-B + Arm-E vs Pure-Vertex Arm-B | Normal | -0.00186769174 [-0.0027432728, -0.00104150176] | 17/33/0 |
| validation | Pure-Vertex Arm-B + Arm-E vs Pure-Vertex Arm-B | Raw EPE | 0 [0, 0] | 0/0/50 |
| validation | Pure-Vertex Arm-B + Arm-E vs Pure-Vertex Arm-B | Vertex RMS | 0.000115155526 [-0.000106514701, 0.000334448967] | 21/29/0 |
| test | Pure-Vertex Arm-B + Arm-E vs Pure-Vertex Arm-B | Refined CD | -5.23283401e-05 [-0.000119921627, 1.54754304e-05] | 33/17/0 |
| test | Pure-Vertex Arm-B + Arm-E vs Pure-Vertex Arm-B | P2S p95 | 0.000193398711 [-0.000147266123, 0.000548681086] | 27/23/0 |
| test | Pure-Vertex Arm-B + Arm-E vs Pure-Vertex Arm-B | F-score | 0.00177117321 [-0.00219090404, 0.0059217377] | 33/17/0 |
| test | Pure-Vertex Arm-B + Arm-E vs Pure-Vertex Arm-B | Normal | -0.00126387224 [-0.00207906072, -0.000494973918] | 17/33/0 |
| test | Pure-Vertex Arm-B + Arm-E vs Pure-Vertex Arm-B | Raw EPE | 0 [0, 0] | 0/0/50 |
| test | Pure-Vertex Arm-B + Arm-E vs Pure-Vertex Arm-B | Vertex RMS | 4.4416607e-05 [-0.000250198157, 0.000337823412] | 28/22/0 |
| validation | Original Arm-B + Arm-E vs Original Arm-B | Refined CD | -0.000760456038 [-0.000993065439, -0.000551295553] | 46/4/0 |
| validation | Original Arm-B + Arm-E vs Original Arm-B | P2S p95 | -0.00286165814 [-0.00403732565, -0.00184883116] | 41/9/0 |
| validation | Original Arm-B + Arm-E vs Original Arm-B | F-score | 0.030082622 [0.0194242822, 0.0415868213] | 40/7/3 |
| validation | Original Arm-B + Arm-E vs Original Arm-B | Normal | 0.00131794032 [0.000657416151, 0.00194533134] | 39/11/0 |
| validation | Original Arm-B + Arm-E vs Original Arm-B | Raw EPE | 0 [0, 0] | 0/0/50 |
| validation | Original Arm-B + Arm-E vs Original Arm-B | Vertex RMS | -0.00225117852 [-0.00262194088, -0.00191663411] | 50/0/0 |
| test | Original Arm-B + Arm-E vs Original Arm-B | Refined CD | -0.000555137243 [-0.000763311898, -0.000389500022] | 45/5/0 |
| test | Original Arm-B + Arm-E vs Original Arm-B | P2S p95 | -0.00119220157 [-0.00185739715, -0.000563323329] | 39/11/0 |
| test | Original Arm-B + Arm-E vs Original Arm-B | F-score | 0.0212784496 [0.011454818, 0.0326810571] | 43/6/1 |
| test | Original Arm-B + Arm-E vs Original Arm-B | Normal | 0.00336914475 [0.00282216791, 0.00396167392] | 47/3/0 |
| test | Original Arm-B + Arm-E vs Original Arm-B | Raw EPE | 0 [0, 0] | 0/0/50 |
| test | Original Arm-B + Arm-E vs Original Arm-B | Vertex RMS | -0.00231977758 [-0.00265423569, -0.00201003507] | 50/0/0 |

## Arm-E compensation

All quantities below use a positive-is-better directional convention. `D_B` is the degradation from Original B to Pure-B; `D_BE` is the residual degradation after adding the same E; compensation is `1-D_BE/D_B` when `D_B>0`.

| Split | Metric | D_B | D_BE | Compensation | E gain on Original B | E gain on Pure-B |
|---|---|---:|---:|---:|---:|---:|
| validation | Refined CD | 0.00024929843 | 0.000918103114 | -268.27% | 0.000760456038 | 9.16513542e-05 |
| validation | P2S p95 | 0.000744369992 | 0.00335146571 | -350.24% | 0.00286165814 | 0.00025456242 |
| validation | F-score | 0.013878622 | 0.0401067556 | -188.98% | 0.030082622 | 0.00385448839 |
| test | Refined CD | 0.000393199046 | 0.000896007949 | -127.88% | 0.000555137243 | 5.23283401e-05 |
| test | P2S p95 | 0.00118634551 | 0.00257194579 | -116.80% | 0.00119220157 | -0.000193398711 |
| test | F-score | 0.0174450691 | 0.0369523456 | -111.82% | 0.0212784496 | 0.00177117321 |

## Decision

Classification: **PURE_B_E_DOES_NOT_RECOVER**.

On test, Pure-B causes CD degradation `D_B=0.000393199046`. With the same frozen E and fusion rule, residual degradation is `D_BE=0.000896007949`, giving a compensation ratio of `-127.88%`.
The primary Pure-B+E minus Original-B+E paired CD difference is `0.000896007949` with 95% CI `[0.000778623463, 0.00101479267]` and W/L/T `0/50/0`.

Arm-E provides a smaller absolute test-CD gain on Pure-B (`5.23283401e-05`) than on Original B (`0.000555137243`). Relative gains are `1.32%` and `15.49%`, respectively.

This decision concerns only the matched Sofa50 v2, single-pass frozen fusion contract. It makes no claim about recursion, Future2000, old native-1920, or any other configuration.

## Numerical audit

- Maximum initial-metric discrepancy: `0.000e+00`.
- Exact sample identity and ordering passed for Pure-B, Original B, Arm-E, and the prepared manifest.
- All 100 new float64 PCG fusion solves converged at the existing tolerance `1e-4` and maximum `2048` iterations.
- Maximum relative residual: `9.966e-05`; PCG iterations mean/max: `24.80/28`; new degenerate faces: `0`.
- Execution device: `cpu`. Device choice changes neither the float64 equation nor the frozen fusion parameters.
- Test was evaluated only after all checkpoints and the Original B+E validation-selected fusion lambda were frozen.
