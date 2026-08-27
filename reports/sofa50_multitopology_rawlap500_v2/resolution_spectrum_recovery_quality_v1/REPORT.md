# Sofa50 mesh resolution, recovery spectrum, and Hybrid gain

Contract audit: **true**. Read-only local analysis of 50 validation and 50 test meshes. No model, checkpoint, mesh, recovery setting, or prior result was modified; no HPC job was submitted.

Primary gain is frozen `E CD - Hybrid CD`, so positive values favor adding the differential branch. The fixed recovery gate is `g_B(Lambda)=Lambda/(Lambda+0.03)`.

## Resolution and gain

Each cell is Pearson [mesh-bootstrap 95% CI] / Spearman [95% CI].

| Split | Predictor | CD gain | P2S-p95 gain | VRMS gain |
|---|---|---:|---:|---:|
| validation | log vertices | 0.3458 [0.0624, 0.5413] / 0.2699 [-0.0047, 0.5096] | 0.1751 [-0.1667, 0.4519] / 0.0642 [-0.2407, 0.3542] | 0.1626 [-0.1746, 0.4659] / 0.0841 [-0.2121, 0.3816] |
| validation | log median edge | -0.3782 [-0.5825, -0.1593] / -0.3760 [-0.5917, -0.1106] | -0.2764 [-0.5330, 0.0384] / -0.1845 [-0.4548, 0.1077] | -0.3690 [-0.5737, -0.1410] / -0.3297 [-0.6028, -0.0247] |
| validation | log vertices/area | 0.3643 [0.0621, 0.5721] / 0.2883 [0.0019, 0.5360] | 0.2027 [-0.1518, 0.4803] / 0.0911 [-0.2151, 0.3817] | 0.2228 [-0.1125, 0.5107] / 0.1606 [-0.1460, 0.4492] |
| test | log vertices | 0.2840 [-0.1142, 0.5503] / 0.1572 [-0.1644, 0.4562] | 0.1693 [-0.2767, 0.5622] / 0.2427 [-0.0781, 0.5338] | 0.0577 [-0.2456, 0.3802] / 0.1149 [-0.1641, 0.3844] |
| test | log median edge | -0.2665 [-0.5244, -0.0444] / -0.2913 [-0.5474, -0.0093] | -0.0760 [-0.4007, 0.1582] / -0.2586 [-0.5158, 0.0336] | 0.1416 [-0.1304, 0.3850] / 0.1465 [-0.1353, 0.4125] |
| test | log vertices/area | 0.2540 [-0.2264, 0.5482] / 0.0573 [-0.2618, 0.3617] | 0.2112 [-0.2597, 0.6024] / 0.1922 [-0.1258, 0.4853] | 0.1659 [-0.1410, 0.4742] / 0.1980 [-0.0862, 0.4570] |

![Resolution and recovery gain](resolution_gain.png)

## Spectrum relative to fixed lambda

### validation

| Statistic | Median | p10 / p90 | Minimum / maximum |
|---|---:|---:|---:|
| E-dominant fraction | 0.034839 | 0.033564 / 0.038367 | 0.032664 / 0.040992 |
| Transition fraction | 0.037210 | 0.036082 / 0.037875 | 0.032122 / 0.038727 |
| B-dominant fraction | 0.927834 | 0.926086 / 0.929306 | 0.924328 / 0.930607 |
| Median Lambda/lambda | 43.036497 | 42.779989 / 43.300054 | 42.730322 / 43.426661 |
| Mean effective B weight | 0.913117 | 0.910661 / 0.914579 | 0.909067 / 0.915606 |

### test

| Statistic | Median | p10 / p90 | Minimum / maximum |
|---|---:|---:|---:|
| E-dominant fraction | 0.034649 | 0.033320 / 0.035643 | 0.032752 / 0.036167 |
| Transition fraction | 0.037217 | 0.035964 / 0.037902 | 0.032900 / 0.038690 |
| B-dominant fraction | 0.928153 | 0.927123 / 0.930019 | 0.926847 / 0.933734 |
| Median Lambda/lambda | 43.137652 | 42.939905 / 43.293319 | 42.818364 / 43.381058 |
| Mean effective B weight | 0.913541 | 0.912604 / 0.915019 | 0.912171 / 0.915759 |

![Resolution and spectral regimes](resolution_spectral_fractions.png)

## Resolution to spectrum to gain

### Test correlations

| Link | Predictor | Outcome | Pearson / Spearman with 95% CI |
|---|---|---|---:|
| resolution_to_spectrum | log_vertices | e_dominant_fraction | 0.4511 [0.2293, 0.6368] / 0.4479 [0.2033, 0.6399] |
| resolution_to_spectrum | log_vertices | transition_fraction | 0.4252 [0.1907, 0.6196] / 0.4633 [0.1837, 0.6952] |
| resolution_to_spectrum | log_vertices | b_dominant_fraction | -0.6264 [-0.7621, -0.4672] / -0.6994 [-0.8396, -0.4834] |
| resolution_to_spectrum | log_vertex_density | b_dominant_fraction | -0.5173 [-0.6643, -0.3557] / -0.5815 [-0.7431, -0.3517] |
| spectrum_to_gain | e_dominant_fraction | cd_gain_e_minus_h | 0.2184 [-0.0553, 0.4478] / 0.2059 [-0.0719, 0.4572] |
| spectrum_to_gain | transition_fraction | cd_gain_e_minus_h | 0.0885 [-0.1330, 0.3176] / 0.1365 [-0.1491, 0.4071] |
| spectrum_to_gain | b_dominant_fraction | cd_gain_e_minus_h | -0.2067 [-0.4510, 0.0515] / -0.2124 [-0.4809, 0.0882] |
| spectrum_to_gain | lambda_q50_over_anchor | cd_gain_e_minus_h | 0.2159 [-0.1282, 0.4916] / 0.1673 [-0.1378, 0.4560] |

![Spectrum and recovery gain](spectrum_gain.png)

## Adjusted resolution effect

Standardized OLS coefficients use mesh bootstrap. They are conditional associations, not causal effects.

| Split | Model | Predictor | Controls | beta [95% CI] |
|---|---|---|---|---:|
| validation | difficulty_geometry | log_vertices | initial_chamfer;log_surface_area;log_median_edge_length | 0.2842 [-0.2524, 0.8560] |
| validation | difficulty_geometry_spectrum | log_vertices | initial_chamfer;log_surface_area;log_median_edge_length;transition_fraction;b_dominant_fraction;lambda_q50_over_anchor | 0.7057 [-0.1889, 1.3733] |
| validation | e_error_geometry | log_vertices | e_chamfer;log_surface_area;log_median_edge_length | 0.5499 [-0.0355, 0.9713] |
| validation | alternative_density | log_median_edge_length | initial_chamfer;log_surface_area | -0.3778 [-0.5955, -0.1280] |
| validation | alternative_density | log_vertex_density | initial_chamfer;log_surface_area | 0.4307 [0.1366, 0.6680] |
| test | difficulty_geometry | log_vertices | initial_chamfer;log_surface_area;log_median_edge_length | 0.1921 [-0.5638, 1.1794] |
| test | difficulty_geometry_spectrum | log_vertices | initial_chamfer;log_surface_area;log_median_edge_length;transition_fraction;b_dominant_fraction;lambda_q50_over_anchor | 0.0669 [-1.0616, 0.9608] |
| test | e_error_geometry | log_vertices | e_chamfer;log_surface_area;log_median_edge_length | 0.6944 [-0.1070, 1.0710] |
| test | alternative_density | log_median_edge_length | initial_chamfer;log_surface_area | -0.2543 [-0.5344, -0.0048] |
| test | alternative_density | log_vertex_density | initial_chamfer;log_surface_area | 0.3288 [-0.0933, 0.7867] |

## Decision

Classification: **NO_RELIABLE_VERTEX_COUNT_OR_SPECTRUM_MEDIATED_RELATIONSHIP**.

Predeclared interpretation: a reliable raw resolution relationship requires positive validation and test Spearman lower bounds; an adjusted relationship requires the test standardized log-vertex coefficient CI to exclude zero after initial error, area, and edge length. A spectrum-mediated pattern additionally requires resolution-to-spectrum and spectrum-to-gain links with CIs excluding zero plus at least 25% attenuation of the adjusted log-vertex coefficient after spectral variables. A raw relationship that disappears after controls is classified as confounding; absent raw replication is no reliable relationship.

The primary vertex-count/spectrum-mediation gate fails, but this is not evidence that every sampling statistic is unrelated to gain. Shorter median edges have a reproducible observational association with larger CD gain in both splits, including after controlling initial CD and area on test. This secondary result is reported as an edge-length association, not as proof that resolution causally drives recovery.

## Answers to the six questions

1. **No reliable raw vertex-count effect.** Test Pearson is `0.2840` (95% CI `[-0.1142, 0.5503]`) and Spearman is `0.1572` (`[-0.1644, 0.4562]`); both include zero, and validation Spearman also includes zero.
2. **No adjusted vertex-count effect.** After initial error, area, and edge length, test standardized log-vertex beta is `0.1921` (95% CI `[-0.5638, 1.1794]`).
3. **Median edge length is the better replicated predictor.** Its CD-gain Spearman is `-0.3760` on validation (`[-0.5917, -0.1106]`) and `-0.2913` on test (`[-0.5474, -0.0093]`). Test adjusted beta is `-0.2543` (`[-0.5344, -0.0048]`). Vertex-density evidence is less stable on test.
4. **Vertex count changes the estimated regime proportions statistically, but only slightly in absolute terms.** On test, log-vertex Spearman is positive for E-dominant mass and negative for B-dominant mass (see chain table). Across all meshes, E-dominant mass spans `0.0327`--`0.0410` and B-dominant mass `0.9243`--`0.9337`. The fractions count all estimated non-null modes, not only representative eigenmodes.
5. **No supported spectral mediation.** Every test spectrum-to-CD-gain Spearman interval includes zero; for example E-dominant `0.2059` (`[-0.0719, 0.4572]`) and B-dominant `-0.2124` (`[-0.4809, 0.0882]`). Adding spectral variables changes test log-vertex beta from `0.1921` to `0.0669`, but attenuation without a spectrum-to-gain link is not mediation evidence.
6. **The fixed crossover is similar, not substantially different, across these meshes.** Median `Lambda/lambda` ranges only `42.730`--`43.427`; about 92.4--93.4% of estimated non-null modes are B-dominant. Thus `lambda=0.03` induces measurable but modest cross-mesh gate shifts.

The requested mesh bootstrap treats the 50 meshes in each split as sampling units. Each split contains five base objects with ten topology/perturbation variants, so variants are not a substitute for 50 independent object identities; causal or population-level resolution claims require more base shapes or a controlled remeshing experiment.

## Spectrum-estimator audit

The full-spectrum fractions use nullspace-projected Hutchinson traces with Chebyshev--Jackson order `384` and `16` Rademacher probes. Maximum full-vs-half-order band-fraction difference: `0.000462`; maximum first-half-vs-second-half probe difference: `0.005277`. Maximum raw band partition error: `2.220e-16`. Component constants are explicitly projected out before trace estimation.

Surface area and edge statistics use the frozen input vertices/faces. Global geometry errors come unchanged from `frozen_hybrid_recovery_v1/matched_per_sample.csv`. The operator uses connectivity only; no cotangent operator, image, GT geometry, checkpoint inference, or new recovery solve is involved.
