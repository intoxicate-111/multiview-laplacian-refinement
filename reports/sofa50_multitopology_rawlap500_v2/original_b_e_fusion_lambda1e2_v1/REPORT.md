# Sofa50 v2 Original Arm-B + frozen Arm-E fusion at lambda=1e-2

Contract audit: **true**. This is a read-only, non-recursive fixed-lambda fusion test on the exact same 50 validation and 50 test meshes. No model is retrained and no lambda search is run.

## Fixed contract

- Original Arm-B: `/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/runs/learned_laplacian/sofa50_v2_sparse_recovery_arm_b_recovery_aware_20k_seed7/checkpoint_best.pt`; SHA-256 `a483e2212f568e771873594cf1e37d13d62cbd2e1e72244baded7dd15573970c`.
- Frozen Arm-E: `/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/runs/learned_laplacian/sofa50_v2_direct_vertex_arm_e_20k_seed7/checkpoint_best.pt`; SHA-256 `6ed27da8759b7bd752ffa75ea8dac3977dd4ced358b5282e0c1c68f750dbade1`.
- Candidate solve: `min_V ||L_U V-delta_B||^2 + 0.01 ||V-V_E||^2`.
- Reference solve: the existing validation-selected Original B+E system with `lambda=0.03`.
- Uniform random-walk operator, frozen B/E arrays, float64 PCG, evaluator, meshes, cameras, sample ordering, and all other settings are unchanged.
- GT enters neither predictor nor fusion solve. Test is not used to select either checkpoint or lambda.

## Aggregate results

CD gain is the macro mean of per-mesh relative improvement over the initial CD. Raw EPE is the frozen B-field diagnostic and is unchanged by fusion.

| Split | System | Initial CD | Refined CD | CD gain | P2S mean | P2S p95 | F-score | Normal | Raw EPE | Vertex RMS | Improved/worsened |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | Initial mesh | 0.00381765892 | 0.00381765892 | +0.00% | 0.00381765892 | 0.0138686196 | 0.918392483 | 0.976308527 | n/a | 0.031591482 | 0/0 |
| validation | Original Arm-B | 0.00381765892 | 0.00320962349 | +15.46% | 0.00320962349 | 0.00999485296 | 0.950224425 | 0.96877577 | 0.0020152807 | 0.00767549059 | 46/4 |
| validation | Arm-E | 0.00381765892 | 0.00285065224 | +20.05% | 0.00285065224 | 0.00891131992 | 0.960090448 | 0.975651504 | n/a | 0.00472553239 | 48/2 |
| validation | Original B+E (lambda=3e-2) | 0.00381765892 | 0.00244916745 | +31.80% | 0.00244916745 | 0.00713319482 | 0.980307047 | 0.97009371 | 0.0020152807 | 0.00542431207 | 50/0 |
| validation | Original B+E (lambda=1e-2) | 0.00381765892 | 0.00256976129 | +29.24% | 0.00256976129 | 0.00748929034 | 0.975889281 | 0.96400505 | 0.0020152807 | 0.0065800354 | 50/0 |
| test | Initial mesh | 0.00438635163 | 0.00438635163 | +0.00% | 0.00438635163 | 0.0146957304 | 0.901781516 | 0.969623498 | n/a | 0.0425195806 | 0/0 |
| test | Original Arm-B | 0.00438635163 | 0.00358497023 | +13.04% | 0.00358497023 | 0.0105580821 | 0.935012989 | 0.959365744 | 0.00263985669 | 0.0115531855 | 36/14 |
| test | Arm-E | 0.00438635163 | 0.00334038817 | +18.59% | 0.00334038817 | 0.0103976753 | 0.943048517 | 0.97011165 | n/a | 0.00822129906 | 45/5 |
| test | Original B+E (lambda=3e-2) | 0.00438635163 | 0.00302983298 | +26.73% | 0.00302983298 | 0.00936588053 | 0.956291439 | 0.962734888 | 0.00263985669 | 0.00923340794 | 49/1 |
| test | Original B+E (lambda=1e-2) | 0.00438635163 | 0.00319840408 | +22.41% | 0.00319840408 | 0.00976125657 | 0.949462916 | 0.955953797 | 0.00263985669 | 0.011047774 | 44/6 |

## Paired comparisons

Differences are candidate minus reference. Negative CD/P2S/raw-EPE/vertex-RMS and positive F-score/normal favor the candidate. Confidence intervals bootstrap meshes.

| Split | Comparison | Metric | Mean difference [95% CI] | Candidate W/L/T |
|---|---|---|---:|---:|
| validation | Original B+E (lambda=1e-2) vs Original B+E (lambda=3e-2) | Refined CD | 0.000120593832 [5.04061948e-05, 0.000202415568] | 16/34/0 |
| validation | Original B+E (lambda=1e-2) vs Original B+E (lambda=3e-2) | P2S p95 | 0.000356095519 [4.21503388e-05, 0.000744750702] | 20/30/0 |
| validation | Original B+E (lambda=1e-2) vs Original B+E (lambda=3e-2) | F-score | -0.00441776582 [-0.00855110077, -0.000756678015] | 12/37/1 |
| validation | Original B+E (lambda=1e-2) vs Original B+E (lambda=3e-2) | Normal | -0.00608866016 [-0.00661409817, -0.00558565276] | 0/50/0 |
| validation | Original B+E (lambda=1e-2) vs Original B+E (lambda=3e-2) | Raw EPE | 0 [0, 0] | 0/0/50 |
| validation | Original B+E (lambda=1e-2) vs Original B+E (lambda=3e-2) | Vertex RMS | 0.00115572334 [0.00102874375, 0.00128642236] | 0/50/0 |
| test | Original B+E (lambda=1e-2) vs Original B+E (lambda=3e-2) | Refined CD | 0.000168571097 [0.000113261301, 0.000227099794] | 9/41/0 |
| test | Original B+E (lambda=1e-2) vs Original B+E (lambda=3e-2) | P2S p95 | 0.000395376032 [0.000138754392, 0.000641774771] | 12/38/0 |
| test | Original B+E (lambda=1e-2) vs Original B+E (lambda=3e-2) | F-score | -0.00682852275 [-0.00978830411, -0.00411994903] | 5/44/1 |
| test | Original B+E (lambda=1e-2) vs Original B+E (lambda=3e-2) | Normal | -0.00678109102 [-0.00751648215, -0.00611652243] | 0/50/0 |
| test | Original B+E (lambda=1e-2) vs Original B+E (lambda=3e-2) | Raw EPE | 0 [0, 0] | 0/0/50 |
| test | Original B+E (lambda=1e-2) vs Original B+E (lambda=3e-2) | Vertex RMS | 0.00181436611 [0.0015697056, 0.0020814247] | 0/50/0 |
| validation | Original B+E (lambda=1e-2) vs Original Arm-B | Refined CD | -0.000639862206 [-0.000837200965, -0.000461540825] | 44/6/0 |
| validation | Original B+E (lambda=1e-2) vs Original Arm-B | P2S p95 | -0.00250556262 [-0.00338639343, -0.00172102828] | 43/7/0 |
| validation | Original B+E (lambda=1e-2) vs Original Arm-B | F-score | 0.0256648561 [0.0170885374, 0.034818235] | 35/14/1 |
| validation | Original B+E (lambda=1e-2) vs Original Arm-B | Normal | -0.00477071984 [-0.00580110397, -0.00383267372] | 2/48/0 |
| validation | Original B+E (lambda=1e-2) vs Original Arm-B | Raw EPE | 0 [0, 0] | 0/0/50 |
| validation | Original B+E (lambda=1e-2) vs Original Arm-B | Vertex RMS | -0.00109545518 [-0.00144407823, -0.000781207316] | 38/12/0 |
| test | Original B+E (lambda=1e-2) vs Original Arm-B | Refined CD | -0.000386566146 [-0.000571850978, -0.000236979712] | 40/10/0 |
| test | Original B+E (lambda=1e-2) vs Original Arm-B | P2S p95 | -0.000796825542 [-0.00138652504, -0.000261253084] | 30/20/0 |
| test | Original B+E (lambda=1e-2) vs Original Arm-B | F-score | 0.0144499269 [0.00527079452, 0.0249527936] | 31/19/0 |
| test | Original B+E (lambda=1e-2) vs Original Arm-B | Normal | -0.00341194627 [-0.00440438618, -0.00246233336] | 4/46/0 |
| test | Original B+E (lambda=1e-2) vs Original Arm-B | Raw EPE | 0 [0, 0] | 0/0/50 |
| test | Original B+E (lambda=1e-2) vs Original Arm-B | Vertex RMS | -0.000505411478 [-0.000786378325, -0.000250351807] | 36/14/0 |
| validation | Original B+E (lambda=3e-2) vs Original Arm-B | Refined CD | -0.000760456038 [-0.000993065439, -0.000551295553] | 46/4/0 |
| validation | Original B+E (lambda=3e-2) vs Original Arm-B | P2S p95 | -0.00286165814 [-0.00403732565, -0.00184883116] | 41/9/0 |
| validation | Original B+E (lambda=3e-2) vs Original Arm-B | F-score | 0.030082622 [0.0194242822, 0.0415868213] | 40/7/3 |
| validation | Original B+E (lambda=3e-2) vs Original Arm-B | Normal | 0.00131794032 [0.000657416151, 0.00194533134] | 39/11/0 |
| validation | Original B+E (lambda=3e-2) vs Original Arm-B | Raw EPE | 0 [0, 0] | 0/0/50 |
| validation | Original B+E (lambda=3e-2) vs Original Arm-B | Vertex RMS | -0.00225117852 [-0.00262194088, -0.00191663411] | 50/0/0 |
| test | Original B+E (lambda=3e-2) vs Original Arm-B | Refined CD | -0.000555137243 [-0.000763311898, -0.000389500022] | 45/5/0 |
| test | Original B+E (lambda=3e-2) vs Original Arm-B | P2S p95 | -0.00119220157 [-0.00185739715, -0.000563323329] | 39/11/0 |
| test | Original B+E (lambda=3e-2) vs Original Arm-B | F-score | 0.0212784496 [0.011454818, 0.0326810571] | 43/6/1 |
| test | Original B+E (lambda=3e-2) vs Original Arm-B | Normal | 0.00336914475 [0.00282216791, 0.00396167392] | 47/3/0 |
| test | Original B+E (lambda=3e-2) vs Original Arm-B | Raw EPE | 0 [0, 0] | 0/0/50 |
| test | Original B+E (lambda=3e-2) vs Original Arm-B | Vertex RMS | -0.00231977758 [-0.00265423569, -0.00201003507] | 50/0/0 |
| validation | Original B+E (lambda=1e-2) vs Arm-E | Refined CD | -0.000280890955 [-0.00041917117, -0.000125721449] | 41/9/0 |
| validation | Original B+E (lambda=1e-2) vs Arm-E | P2S p95 | -0.00142202958 [-0.00201386237, -0.000776365145] | 41/9/0 |
| validation | Original B+E (lambda=1e-2) vs Arm-E | F-score | 0.0157988332 [0.00703057992, 0.0246043042] | 35/15/0 |
| validation | Original B+E (lambda=1e-2) vs Arm-E | Normal | -0.0116464544 [-0.0127657964, -0.0106038825] | 0/50/0 |
| validation | Original B+E (lambda=1e-2) vs Arm-E | Vertex RMS | 0.00185450301 [0.00162529677, 0.00208866241] | 0/50/0 |
| test | Original B+E (lambda=1e-2) vs Arm-E | Refined CD | -0.000141984094 [-0.000279063269, 1.62340242e-05] | 38/12/0 |
| test | Original B+E (lambda=1e-2) vs Arm-E | P2S p95 | -0.000636418699 [-0.00140123926, 0.000120412042] | 37/13/0 |
| test | Original B+E (lambda=1e-2) vs Arm-E | F-score | 0.00641439879 [0.000287558575, 0.0121024694] | 39/10/1 |
| test | Original B+E (lambda=1e-2) vs Arm-E | Normal | -0.0141578524 [-0.0155940876, -0.0128584445] | 0/50/0 |
| test | Original B+E (lambda=1e-2) vs Arm-E | Vertex RMS | 0.00282647499 [0.00226670464, 0.00342743893] | 2/48/0 |
| validation | Original B+E (lambda=3e-2) vs Arm-E | Refined CD | -0.000401484787 [-0.000490777912, -0.000308028037] | 48/2/0 |
| validation | Original B+E (lambda=3e-2) vs Arm-E | P2S p95 | -0.0017781251 [-0.00214208712, -0.00141127503] | 46/4/0 |
| validation | Original B+E (lambda=3e-2) vs Arm-E | F-score | 0.0202165991 [0.0139849559, 0.0266952276] | 41/9/0 |
| validation | Original B+E (lambda=3e-2) vs Arm-E | Normal | -0.0055577942 [-0.00624212113, -0.00493570404] | 0/50/0 |
| validation | Original B+E (lambda=3e-2) vs Arm-E | Vertex RMS | 0.000698779676 [0.000556917548, 0.000845320559] | 2/48/0 |
| test | Original B+E (lambda=3e-2) vs Arm-E | Refined CD | -0.00031055519 [-0.000419057311, -0.000187327397] | 46/4/0 |
| test | Original B+E (lambda=3e-2) vs Arm-E | P2S p95 | -0.00103179473 [-0.00170836447, -0.000340126447] | 43/7/0 |
| test | Original B+E (lambda=3e-2) vs Arm-E | F-score | 0.0132429215 [0.00807757857, 0.0184555938] | 43/5/2 |
| test | Original B+E (lambda=3e-2) vs Arm-E | Normal | -0.00737676141 [-0.00839094582, -0.0064388529] | 0/50/0 |
| test | Original B+E (lambda=3e-2) vs Arm-E | Vertex RMS | 0.00101210888 [0.000569259339, 0.00145623765] | 12/38/0 |

## Fusion gain and lambda effect

Positive gain favors fusion over standalone Original B. The gain ratio is interpreted only when the selected `0.03` fusion gain is positive.

| Split | Metric | Gain at 0.01 | Gain at 0.03 | 0.01/0.03 gain ratio | Value(0.01)-Value(0.03) |
|---|---|---:|---:|---:|---:|
| validation | Refined CD | 0.000639862206 | 0.000760456038 | 84.14% | 0.000120593832 |
| validation | P2S p95 | 0.00250556262 | 0.00286165814 | 87.56% | 0.000356095519 |
| validation | F-score | 0.0256648561 | 0.030082622 | 85.31% | -0.00441776582 |
| validation | Normal | -0.00477071984 | 0.00131794032 | -361.98% | -0.00608866016 |
| test | Refined CD | 0.000386566146 | 0.000555137243 | 69.63% | 0.000168571097 |
| test | P2S p95 | 0.000796825542 | 0.00119220157 | 66.84% | 0.000395376032 |
| test | F-score | 0.0144499269 | 0.0212784496 | 67.91% | -0.00682852275 |
| test | Normal | -0.00341194627 | 0.00336914475 | -101.27% | -0.00678109102 |

## Decision

Classification: **LAMBDA_1E2_WORSE_THAN_SELECTED_3E2**.

Test CD is `0.00319840408` at fusion `lambda=0.01` versus `0.00302983298` at the existing validation-selected `lambda=0.03`. The paired candidate-minus-reference difference is `0.000168571097` with 95% CI `[0.000113261301, 0.000227099794]` and W/L/T `9/41/0`.

This is a fixed-lambda diagnostic of the Original B+E system. It does not alter the validation-selected formal system and makes no claim about Pure-B, recursion, Future2000, or old native-1920.

## Numerical audit

- Maximum initial-metric discrepancy: `0.000e+00`.
- Exact sample identity/order and checkpoint hashes passed for Original B, Arm-E, and the prepared manifest.
- All 100 new float64 PCG solves converged at tolerance `1e-4`, maximum `2048` iterations.
- Maximum relative residual: `9.989e-05`; iterations mean/max: `37.24/47`; new degenerate faces: `0`.
- Execution device: `cpu`; no inference or fusion hyperparameter changed.
