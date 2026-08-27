# Sofa50 curvature-conditioned differential-branch benefit

Contract audit: **true**. Read-only test-set analysis over **50** meshes.

GT curvature is the magnitude of the standard cotangent discrete twice-mean-curvature vector `2Hn`. Vertices are ranked independently inside each mesh and split into `0–25%`, `25–50%`, `50–75%`, `75–90%`, and `90–100%` bins. This controls mesh scale and vertex-count imbalance.

Hybrid is reproduced with the established frozen solve (`lambda=3e-2`, float64 PCG, `tol=1e-4`, maximum 2048 iterations). GT is loaded only after the frozen B/E predictions and recovery inputs are fixed.

## Curvature-conditioned local error

Positive gain means adding Arm B to E reduces error. Values are macro-averages of per-mesh bin means; confidence intervals bootstrap the 50 meshes.

| GT curvature bin | E exact P2S | Hybrid exact P2S | Surface gain [95% CI] | E vertex | Hybrid vertex | Vertex gain [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| p00_p25 | 0.00226622496 | 0.00223870988 | 2.75150834e-05 [-8.77778463e-06, 6.17838953e-05] | 0.00457195813 | 0.00538448779 | -0.000812529659 [-0.000992422883, -0.000650248851] |
| p25_p50 | 0.00178297885 | 0.00187861681 | -9.56379617e-05 [-0.000133454953, -6.01216181e-05] | 0.00373382581 | 0.00426882793 | -0.000535002121 [-0.000661050488, -0.000418279118] |
| p50_p75 | 0.00135181496 | 0.00150026303 | -0.000148448073 [-0.000180790814, -0.000117322755] | 0.00325145656 | 0.00383839012 | -0.000586933561 [-0.00069082638, -0.000488731285] |
| p75_p90 | 0.00095637358 | 0.0012522641 | -0.00029589052 [-0.000357419295, -0.000240704134] | 0.00335264076 | 0.00432069769 | -0.000968056936 [-0.00117324558, -0.000789181332] |
| p90_p100 | 0.000759131891 | 0.0011526675 | -0.000393535613 [-0.000486870749, -0.000305884694] | 0.00308856708 | 0.00448845457 | -0.00139988749 [-0.00177580373, -0.00107728879] |

Exact P2S is the distance from each E/Hybrid vertex to the clean GT triangle surface, with the query vertex retaining its corresponding GT-vertex curvature bin. The normal error (reported in CSV/JSON and below) is the absolute displacement along the GT area-weighted vertex normal; vertex error is the full same-index Euclidean distance.

## High-curvature benefit test

Highest-10%-minus-lowest-25% exact-surface-gain difference: `-0.000421050696` (bootstrap 95% CI `[-0.00050607406, -0.000343324696]`; high larger on `0/50` meshes).

Highest-10%-minus-lowest-25% vertex-gain difference: `-0.00058735783` (bootstrap 95% CI `[-0.000841602742, -0.00037405462]`; high larger on `7/50` meshes).

Highest-10%-minus-lowest-25% normal-gain difference: `-0.000504737455` (bootstrap 95% CI `[-0.000622456554, -0.000410902525]`; high larger on `0/50` meshes).

Per-mesh curvature-versus-local-gain Spearman: exact surface macro mean `-0.09906` (median `-0.09562`); vertex macro mean `-0.04860` (median `-0.04438`); normal macro mean `-0.11329` (median `-0.11096`).

Predeclared support gate (exact-surface high-minus-low bootstrap lower bound is positive and high-curvature gain is larger on a majority of meshes): **false**.

## Main finding

The proposed curvature-localization hypothesis is not supported; the measured effect is the opposite. Hybrid is statistically indistinguishable from a small improvement in the lowest-curvature quartile for exact P2S, but is significantly worse from the 25th percentile upward. In the highest-curvature 10%, exact P2S rises from `0.000759132` for E to `0.00115267` for Hybrid (51.8% worse), same-index vertex error rises by 45.3%, and GT-normal error rises by 51.5%. The high-minus-low exact-surface gain is negative on all 50 meshes.

This does not contradict Hybrid's lower global surface Chamfer (`0.00302983` versus E's `0.00334039`). Global Chamfer is area-weighted and bidirectional, whereas this audit conditions forward vertex-to-GT-surface errors on GT-vertex curvature rank. The global gain can therefore arise from surface-area weighting, the reverse GT-to-prediction direction, and error redistribution rather than preferential correction of high-curvature vertices.

The weak cotangent-curvature correlation should not be assigned solely to predictor failure: even the exact clean uniform-Laplacian field has similarly weak magnitude correlation and top-region recall. Under the current operator contract, Arm B is best described as an operator-guided differential constraint, not as a cotangent-curvature predictor. The paper should retain the exact recovery-spectrum result but avoid claiming that B specifically improves regions where cotangent curvature is high.

![Curvature-conditioned local error](curvature_bin_local_error.png)

![Curvature-conditioned gain](curvature_bin_gain.png)

## Differential field versus cotangent curvature

`predicted_b` is frozen Arm-B's raw uniform-Laplacian prediction. `gt_uniform` is the clean mesh's exact uniform-Laplacian field and is included as an operator-mismatch reference; neither is expected to numerically equal cotangent `2Hn`.

| Signal | Magnitude Pearson | Magnitude Spearman | Direction cosine | Abs. direction cosine | Top-10% recall | Top-25% recall |
|---|---:|---:|---:|---:|---:|---:|
| predicted_b | 0.02692 | -0.11404 | 0.18811 | 0.25568 | 0.14343 | 0.25386 |
| gt_uniform | 0.02912 | -0.09126 | 0.22672 | 0.23872 | 0.14715 | 0.26143 |

Random top-set recall baselines are 0.10 and 0.25. Direction cosine uses the signed `2Hn` convention in the protocol; absolute cosine is also reported to expose orientation agreement independently of sign.

## Protocol and scope

Curvature protocol: `same-index cotangent discrete twice-mean-curvature vector 2Hn=(2A_bary)^-1 sum_j(cot_alpha+cot_beta)(v_i-v_j); compare magnitudes; eligible vertices require positive refined/clean barycentric area; dihedral and face-normal angles use acos(abs(dot)) to ignore winding sign; edge-length and triangle-area distortion use absolute log ratios`

This analysis establishes where the already-selected frozen fusion changes local geometry. It does not use test curvature to select a checkpoint, lambda, model, or recovery setting.
