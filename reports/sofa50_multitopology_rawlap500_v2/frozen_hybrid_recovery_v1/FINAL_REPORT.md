# Sofa50 v2 frozen Arm-B/Arm-E hybrid recovery diagnostic

Contract audit: **true**. This is a zero-retraining, read-only recombination of the frozen selected Arm B and Arm E outputs.

The primary solve is `min ||L V - delta_B||² + lambda ||V - V_direct||²`, with no additional input anchor. PCG is float64, tolerance `1e-4`, maximum `2048` iterations. GT is evaluation-only.

## Implementation/read-only audit

- `delta_B` is read from the selected frozen Arm B prediction archive/checkpoint `/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/runs/learned_laplacian/sofa50_v2_sparse_recovery_arm_b_recovery_aware_20k_seed7/checkpoint_best.pt` (SHA-256 `a483e2212f568e771873594cf1e37d13d62cbd2e1e72244baded7dd15573970c`).
- `V_direct = V_input + delta_v_E` uses the selected frozen Arm E archive/checkpoint `/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/runs/learned_laplacian/sofa50_v2_direct_vertex_arm_e_20k_seed7/checkpoint_best.pt` (SHA-256 `6ed27da8759b7bd752ffa75ea8dac3977dd4ced358b5282e0c1c68f750dbade1`).
- Matched B/E arrays are checked against the exact same manifest sample IDs, ordering, input mesh and connectivity. The archived predictors used the same 28 native-960 images and cameras; OOD re-inference additionally audits the actual common model-input mapping for every sample.
- No GT field enters either predictor or the recovery solve; clean vertices are loaded only after predictions for evaluation.
- No network parameter, prediction, image, camera, mesh, topology or benchmark output is modified. No fine-tuning occurs.
- Relative to Arm B, the only primary recovery change is the positional anchor target: `V_input` is replaced by frozen `V_direct`.

## Validation-only lambda selection

Selected by validation mean Chamfer: **lambda = 3e-02**. Diagnostic VRMS optimum: `1e+00`; diagnostic P2S-p95 optimum: `3e-02`.

| Lambda | CD | CD gain | VRMS | P2S p95 | F-score | Normal | Flip rate | Improved/worsened | PCG iter mean/max | Hybrid→E VRMS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e-04 | 0.0091752713 | -149.05% | 0.02201258 | 0.028063352 | 0.70581513 | 0.93830664 | 4.297% | 0/50 | 200.62/247 | 0.021693688 |
| 3e-04 | 0.0063595275 | -70.44% | 0.01630372 | 0.019887258 | 0.81793273 | 0.94345663 | 4.194% | 1/49 | 140.68/165 | 0.015899952 |
| 1e-03 | 0.0042077058 | -12.26% | 0.011665613 | 0.013077111 | 0.91207561 | 0.94982615 | 3.895% | 17/33 | 90.78/103 | 0.011131361 |
| 3e-03 | 0.0031257153 | +15.57% | 0.0087175696 | 0.0094527257 | 0.95639312 | 0.95664166 | 3.455% | 41/9 | 59.08/66 | 0.0080251244 |
| 1e-02 | 0.0025697611 | +29.24% | 0.0065800354 | 0.0074893061 | 0.97588928 | 0.96400505 | 2.898% | 50/0 | 36.36/40 | 0.0056680345 |
| 3e-02 | 0.0024491677 | +31.80% | 0.0054243121 | 0.0071331948 | 0.98030705 | 0.97009371 | 2.443% | 50/0 | 22.48/25 | 0.0042832757 |
| 1e-01 | 0.0025090846 | +29.67% | 0.004746635 | 0.0075505882 | 0.9758176 | 0.97528701 | 2.066% | 50/0 | 12.18/14 | 0.0033180716 |
| 3e-01 | 0.0026309428 | +26.24% | 0.0044311691 | 0.0080787646 | 0.97029411 | 0.97851153 | 1.836% | 50/0 | 6.54/8 | 0.0026055467 |
| 1e+00 | 0.0027378447 | +23.25% | 0.0043226429 | 0.0084856147 | 0.96568638 | 0.98002833 | 1.736% | 48/2 | 3.22/4 | 0.0017495249 |
| 3e+00 | 0.0027951256 | +21.60% | 0.004420898 | 0.0086904769 | 0.96297683 | 0.97912645 | 1.904% | 48/2 | 1.96/2 | 0.00096402844 |

Forward audit against tight float64 LSMR: maximum PCG↔LSMR vertex RMS `0.00508694`, maximum coordinate difference `0.0164427`; all PCG and LSMR checks converged.

## Matched validation and test

| Split | Arm | Initial CD | Refined CD | Gain / eta | P2S p95 | F-score | Normal | Flips / rate | New deg. | Improved/worsened | VRMS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | initial | 0.0038176589 | 0.0038176589 | +0.00% / 0 | 0.01386862 | 0.91839248 | 0.97630853 | 0 / 0.000% | 0 | 0/0 | 0.031591482 |
| validation | B_lap_plus_refine | 0.0038176589 | 0.0032096235 | +15.46% / 0.154608 | 0.009994853 | 0.95022443 | 0.96877577 | 49677 / 2.147% | 0 | 46/4 | 0.0076754906 |
| validation | E_direct_vertex_residual | 0.0038176589 | 0.0028506522 | +20.05% / 0.200538 | 0.0089113199 | 0.96009045 | 0.9756515 | 56763 / 2.453% | 0 | 48/2 | 0.0047255324 |
| validation | Hybrid_B_laplacian_E_anchor | 0.0038176589 | 0.0024491675 | +31.80% / 0.318011 | 0.0071331948 | 0.98030705 | 0.97009371 | 56528 / 2.443% | 0 | 50/0 | 0.0054243121 |
| test | initial | 0.0043863516 | 0.0043863516 | +0.00% / 0 | 0.01469573 | 0.90178152 | 0.9696235 | 0 / 0.000% | 0 | 0/0 | 0.042519581 |
| test | B_lap_plus_refine | 0.0043863516 | 0.0035849702 | +13.04% / 0.13036 | 0.010558082 | 0.93501299 | 0.95936574 | 52338 / 3.092% | 0 | 36/14 | 0.011553186 |
| test | E_direct_vertex_residual | 0.0043863516 | 0.0033403882 | +18.59% / 0.18589 | 0.010397675 | 0.94304852 | 0.97011165 | 52582 / 3.106% | 0 | 45/5 | 0.0082212991 |
| test | Hybrid_B_laplacian_E_anchor | 0.0043863516 | 0.003029833 | +26.73% / 0.267293 | 0.0093658805 | 0.95629144 | 0.96273489 | 55247 / 3.263% | 0 | 49/1 | 0.0092334079 |

## Paired comparisons and bootstrap intervals

| Split | Comparison | H lower CD | H lower VRMS | H lower P95 | H higher F | H higher normal | H fewer flips |
|---|---|---:|---:|---:|---:|---:|---:|
| validation | Hybrid_B_laplacian_E_anchor_vs_B_lap_plus_refine | 46/50 | 50/50 | 41/50 | 40/50 | 39/50 | 9/50 |
| validation | Hybrid_B_laplacian_E_anchor_vs_E_direct_vertex_residual | 48/50 | 2/50 | 46/50 | 41/50 | 0/50 | 22/50 |
| test | Hybrid_B_laplacian_E_anchor_vs_B_lap_plus_refine | 45/50 | 50/50 | 39/50 | 43/50 | 47/50 | 18/50 |
| test | Hybrid_B_laplacian_E_anchor_vs_E_direct_vertex_residual | 46/50 | 12/50 | 43/50 | 43/50 | 0/50 | 14/50 |

| Split | Quantity | Mean | Median | Paired bootstrap 95% CI |
|---|---|---:|---:|---:|
| validation | refined_chamfer_hybrid_minus_B_lap_plus_refine | -0.00076045604 | -0.00035890561 | [-0.00099306544, -0.00055129555] |
| validation | same_index_recovered_vertex_rms_hybrid_minus_B_lap_plus_refine | -0.0022511785 | -0.0021062079 | [-0.0026081579, -0.0019117845] |
| validation | p2s_p95_hybrid_minus_B_lap_plus_refine | -0.0028616581 | -0.0011384172 | [-0.0040053918, -0.001848785] |
| validation | normal_consistency_hybrid_minus_B_lap_plus_refine | +0.0013179403 | +0.0018381763 | [+0.00064871915, +0.0019367842] |
| validation | refined_chamfer_hybrid_minus_E_direct_vertex_residual | -0.00040148479 | -0.00040981562 | [-0.00049077791, -0.00030802804] |
| validation | same_index_recovered_vertex_rms_hybrid_minus_E_direct_vertex_residual | +0.00069877968 | +0.00070198077 | [+0.00055640022, +0.00084674588] |
| validation | p2s_p95_hybrid_minus_E_direct_vertex_residual | -0.0017781251 | -0.00161469 | [-0.0021405821, -0.0014182365] |
| validation | normal_consistency_hybrid_minus_E_direct_vertex_residual | -0.0055577942 | -0.0054051273 | [-0.0062342838, -0.0049442701] |
| test | refined_chamfer_hybrid_minus_B_lap_plus_refine | -0.00055513724 | -0.00043226127 | [-0.0007633119, -0.00038950002] |
| test | same_index_recovered_vertex_rms_hybrid_minus_B_lap_plus_refine | -0.0023197776 | -0.0018734503 | [-0.002644747, -0.0020088994] |
| test | p2s_p95_hybrid_minus_B_lap_plus_refine | -0.0011922016 | -0.00090449801 | [-0.0018429774, -0.00056497673] |
| test | normal_consistency_hybrid_minus_B_lap_plus_refine | +0.0033691447 | +0.0033969599 | [+0.0028282444, +0.0039596529] |
| test | refined_chamfer_hybrid_minus_E_direct_vertex_residual | -0.00031055519 | -0.0002396988 | [-0.00041905731, -0.0001873274] |
| test | same_index_recovered_vertex_rms_hybrid_minus_E_direct_vertex_residual | +0.0010121089 | +0.0010458648 | [+0.00057208195, +0.0014538849] |
| test | p2s_p95_hybrid_minus_E_direct_vertex_residual | -0.0010317947 | -0.0010333001 | [-0.0017087079, -0.00037268157] |
| test | normal_consistency_hybrid_minus_E_direct_vertex_residual | -0.0073767614 | -0.006668927 | [-0.0083877166, -0.0064177429] |

## Test recipe and generation-family breakdown

| Group | Arm | CD | VRMS | P2S p95 | Normal | Flip rate | Improved/worsened |
|---|---|---:|---:|---:|---:|---:|---:|
| A1 | B_lap_plus_refine | 0.0044165734 | 0.010350723 | 0.013071591 | 0.96214778 | 3.542% | 2/3 |
| A1 | E_direct_vertex_residual | 0.0031942353 | 0.0069173496 | 0.0096606812 | 0.97468976 | 3.559% | 5/0 |
| A1 | Hybrid_B_laplacian_E_anchor | 0.0033305818 | 0.0082528541 | 0.010499529 | 0.96748188 | 3.496% | 5/0 |
| A2 | B_lap_plus_refine | 0.0062579633 | 0.021705026 | 0.018019486 | 0.9488916 | 3.929% | 3/2 |
| A2 | E_direct_vertex_residual | 0.0049113907 | 0.016972355 | 0.015975959 | 0.96000596 | 3.578% | 4/1 |
| A2 | Hybrid_B_laplacian_E_anchor | 0.0048906465 | 0.018458076 | 0.014998655 | 0.95311343 | 4.056% | 5/0 |
| B1 | B_lap_plus_refine | 0.0023205226 | 0.0067194751 | 0.0069687781 | 0.96134348 | 3.222% | 4/1 |
| B1 | E_direct_vertex_residual | 0.0025614886 | 0.0044915606 | 0.0084428966 | 0.97224303 | 3.389% | 4/1 |
| B1 | Hybrid_B_laplacian_E_anchor | 0.0021400631 | 0.0055727869 | 0.0067165557 | 0.96409824 | 3.396% | 4/1 |
| B2 | B_lap_plus_refine | 0.0032857223 | 0.0088257563 | 0.010394206 | 0.95486003 | 2.559% | 4/1 |
| B2 | E_direct_vertex_residual | 0.0035401882 | 0.0059749646 | 0.011000983 | 0.96619477 | 2.644% | 4/1 |
| B2 | Hybrid_B_laplacian_E_anchor | 0.0030721183 | 0.0064209624 | 0.0093565838 | 0.95603888 | 3.041% | 5/0 |
| C1 | B_lap_plus_refine | 0.0025554582 | 0.0076224975 | 0.0079605346 | 0.96871574 | 2.777% | 3/2 |
| C1 | E_direct_vertex_residual | 0.002434867 | 0.0049000081 | 0.0077649828 | 0.97813249 | 2.807% | 4/1 |
| C1 | Hybrid_B_laplacian_E_anchor | 0.0021111454 | 0.0063037218 | 0.0063791415 | 0.97214124 | 2.742% | 5/0 |
| C2 | B_lap_plus_refine | 0.003632492 | 0.011042564 | 0.010937561 | 0.96603163 | 2.413% | 4/1 |
| C2 | E_direct_vertex_residual | 0.0037994822 | 0.0082004249 | 0.011666021 | 0.97369803 | 2.349% | 5/0 |
| C2 | Hybrid_B_laplacian_E_anchor | 0.0033035922 | 0.0083728112 | 0.010799983 | 0.9682976 | 2.610% | 5/0 |
| C3 | B_lap_plus_refine | 0.0027242542 | 0.011706743 | 0.0077967314 | 0.96734598 | 3.146% | 4/1 |
| C3 | E_direct_vertex_residual | 0.0025344041 | 0.0072067128 | 0.0082567139 | 0.97827378 | 3.099% | 5/0 |
| C3 | Hybrid_B_laplacian_E_anchor | 0.0022456622 | 0.009674927 | 0.0069032946 | 0.97107218 | 3.090% | 5/0 |
| C4 | B_lap_plus_refine | 0.004259927 | 0.015528239 | 0.011639931 | 0.9543715 | 3.427% | 4/1 |
| C4 | E_direct_vertex_residual | 0.0039943342 | 0.011763215 | 0.011664193 | 0.96512472 | 3.346% | 5/0 |
| C4 | Hybrid_B_laplacian_E_anchor | 0.0036015907 | 0.012011909 | 0.010398682 | 0.9582014 | 3.664% | 5/0 |
| D1 | B_lap_plus_refine | 0.0023907253 | 0.008539623 | 0.0068847807 | 0.96087838 | 3.646% | 3/2 |
| D1 | E_direct_vertex_residual | 0.002392779 | 0.0056352223 | 0.0075526176 | 0.97307874 | 3.594% | 5/0 |
| D1 | Hybrid_B_laplacian_E_anchor | 0.0020777624 | 0.0071023228 | 0.0062287225 | 0.96517601 | 3.674% | 5/0 |
| D2 | B_lap_plus_refine | 0.004006064 | 0.013491209 | 0.011907222 | 0.94907132 | 3.746% | 5/0 |
| D2 | E_direct_vertex_residual | 0.0040407124 | 0.010151178 | 0.011991704 | 0.95967522 | 3.799% | 4/1 |
| D2 | Hybrid_B_laplacian_E_anchor | 0.0035251671 | 0.010163708 | 0.011377658 | 0.95172802 | 4.145% | 5/0 |
| mild | B_lap_plus_refine | 0.0028815067 | 0.0089878123 | 0.0085364833 | 0.96408627 | 3.206% | 16/9 |
| mild | E_direct_vertex_residual | 0.0026235548 | 0.0058301707 | 0.0083355784 | 0.97528356 | 3.248% | 23/2 |
| mild | Hybrid_B_laplacian_E_anchor | 0.002381043 | 0.0073813225 | 0.0073454486 | 0.96799391 | 3.244% | 24/1 |
| strong | B_lap_plus_refine | 0.0042884337 | 0.014118559 | 0.012579681 | 0.95464522 | 2.954% | 20/5 |
| strong | E_direct_vertex_residual | 0.0040572216 | 0.010612427 | 0.012459772 | 0.96493974 | 2.935% | 22/3 |
| strong | Hybrid_B_laplacian_E_anchor | 0.003678623 | 0.011085493 | 0.011386313 | 0.95747586 | 3.286% | 25/0 |
| original_topology | B_lap_plus_refine | 0.0053372683 | 0.016027874 | 0.015545539 | 0.95551969 | 3.735% | 5/5 |
| original_topology | E_direct_vertex_residual | 0.004052813 | 0.011944852 | 0.01281832 | 0.96734786 | 3.569% | 9/1 |
| original_topology | Hybrid_B_laplacian_E_anchor | 0.0041106142 | 0.013355465 | 0.012749092 | 0.96029766 | 3.776% | 10/0 |
| subdivided | B_lap_plus_refine | 0.0031468957 | 0.010434513 | 0.0093112179 | 0.96032726 | 3.034% | 31/9 |
| subdivided | E_direct_vertex_residual | 0.003162282 | 0.0072904108 | 0.009792514 | 0.9708026 | 3.064% | 36/4 |
| subdivided | Hybrid_B_laplacian_E_anchor | 0.0027596377 | 0.0082028936 | 0.0085200777 | 0.9633442 | 3.217% | 39/1 |
| adaptive_topology | B_lap_plus_refine | 0.0032614868 | 0.011321813 | 0.0095211266 | 0.96106909 | 3.114% | 23/7 |
| adaptive_topology | E_direct_vertex_residual | 0.0031994298 | 0.0079761268 | 0.0098160387 | 0.9713305 | 3.091% | 28/2 |
| adaptive_topology | Hybrid_B_laplacian_E_anchor | 0.00281082 | 0.0089382333 | 0.008681247 | 0.96443608 | 3.217% | 30/0 |

Per-recipe Hybrid-vs-B/E paired wins are in `recipe_paired_wins.csv`.

## Graph-frequency analysis

uniform-undirected-graph symmetric-normalized Laplacian Lsym=I-D^-1/2 A D^-1/2; eigenvalue range [0,2]; Chebyshev-Jackson hard-band approximation; low=[0,2/3), mid=[2/3,4/3), high=[4/3,2]; xyz energy summed.

| Split | Signal | Total absolute energy | Low energy / fraction | Mid energy / fraction | High energy / fraction |
|---|---|---:|---:|---:|---:|
| validation | gt_displacement | 1014.8297 | 545.05636 / 53.71% | 397.73453 / 39.19% | 72.038836 / 7.10% |
| validation | b_error | 59.811096 | 47.811692 / 79.94% | 9.0563159 / 15.14% | 2.9430882 / 4.92% |
| validation | e_error | 22.568302 | 8.531847 / 37.80% | 9.9927079 / 44.28% | 4.0437466 / 17.92% |
| validation | hybrid_error | 30.174426 | 18.60651 / 61.66% | 8.7282877 / 28.93% | 2.8396275 / 9.41% |
| validation | hybrid_minus_b | 34.520693 | 34.277589 / 99.30% | 0.17094441 / 0.50% | 0.072159037 / 0.21% |
| validation | hybrid_minus_e | 18.803203 | 10.279446 / 54.67% | 5.9229428 / 31.50% | 2.6008146 / 13.83% |
| test | gt_displacement | 1371.4334 | 746.13134 / 54.41% | 528.3618 / 38.53% | 96.940248 / 7.07% |
| test | b_error | 102.25649 | 74.696507 / 73.05% | 22.592834 / 22.09% | 4.9671479 / 4.86% |
| test | e_error | 55.865855 | 24.499142 / 43.85% | 24.493527 / 43.84% | 6.8731859 / 12.30% |
| test | hybrid_error | 67.318395 | 40.484578 / 60.14% | 21.977374 / 32.65% | 4.8564427 / 7.21% |
| test | hybrid_minus_b | 40.054368 | 39.696263 / 99.11% | 0.26505919 / 0.66% | 0.093046567 / 0.23% |
| test | hybrid_minus_e | 39.485805 | 21.04304 / 53.29% | 13.375161 / 33.87% | 5.0676039 / 12.83% |

Absolute test error energy shows a qualified spectral fusion: Hybrid low-frequency error is between E and B, while Hybrid mid/high error is slightly below both. It does not beat E in total error energy, so the conclusion is not inferred from normalized fractions alone.

## Connected components

| Split | Arm | Components | Translation error mean / RMS / median / p95 | Centered deformation VRMS |
|---|---|---:|---:|---:|
| validation | B_lap_plus_refine | 890 | 0.00030684951 / 0.00064820268 / 0.00014724282 / 0.0010782473 | 0.0076702482 |
| validation | E_direct_vertex_residual | 890 | 0.00076661429 / 0.001196047 / 0.00057688484 / 0.0018515329 | 0.0046820803 |
| validation | Hybrid_B_laplacian_E_anchor | 890 | 0.00076666544 / 0.0011959672 / 0.00057685994 / 0.0018515528 | 0.0053861742 |
| test | B_lap_plus_refine | 880 | 0.0032666726 / 0.011835974 / 0.00029567993 / 0.027015688 | 0.011421331 |
| test | E_direct_vertex_residual | 880 | 0.0019960755 / 0.0070969782 / 0.00061282815 / 0.0064982101 | 0.0081316111 |
| test | Hybrid_B_laplacian_E_anchor | 880 | 0.0019957212 / 0.0071013264 / 0.00061287946 / 0.0064996252 | 0.0091505727 |

The component translation modes of Hybrid reproduce E almost exactly, as expected because the random-walk Laplacian cannot constrain per-component constants and the direct anchor fixes them. Within-component centered error is between B and E, not better than E.

## Frozen OOD (lambda fixed from matched validation)

| Domain | Arm | Initial CD | Refined CD | Mean gain | P2S p95 | F-score | Normal | Flip rate | VRMS | Improved/worsened |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy_v1 | B_lap_plus_refine | 0.0044145842 | 0.0070599892 | -64.22% | 0.025372027 | 0.84299257 | 0.90306823 | 11.716% | 0.025490568 | 9/41 |
| legacy_v1 | E_direct_vertex_residual | 0.0044145842 | 0.0082270787 | -93.40% | 0.031432919 | 0.82988271 | 0.9085034 | 11.780% | 0.027324822 | 0/50 |
| legacy_v1 | Hybrid_B_laplacian_E_anchor | 0.0044145842 | 0.0072971122 | -70.18% | 0.026852501 | 0.84228964 | 0.90392891 | 11.909% | 0.026693539 | 6/44 |
| unseen_recipes_v1 | B_lap_plus_refine | 0.0044120083 | 0.0067774225 | -46.37% | 0.022922457 | 0.84141477 | 0.89928854 | 9.077% | 0.027216646 | 6/19 |
| unseen_recipes_v1 | E_direct_vertex_residual | 0.0044120083 | 0.0076851937 | -66.93% | 0.02805435 | 0.83265162 | 0.90557811 | 9.145% | 0.029109683 | 1/24 |
| unseen_recipes_v1 | Hybrid_B_laplacian_E_anchor | 0.0044120083 | 0.0069731843 | -50.47% | 0.024548011 | 0.84353266 | 0.90087832 | 9.185% | 0.028490968 | 4/21 |

| Domain | Comparison | Right lower CD | Right lower VRMS | Right lower P95 | Right higher F | Right higher normal | Right fewer flips |
|---|---|---:|---:|---:|---:|---:|---:|
| legacy_v1 | Hybrid_B_laplacian_E_anchor_vs_B_lap_plus_refine | 12/50 | 0/50 | 11/50 | 16/50 | 35/50 | 13/50 |
| legacy_v1 | Hybrid_B_laplacian_E_anchor_vs_E_direct_vertex_residual | 46/50 | 31/50 | 44/50 | 40/50 | 6/50 | 24/50 |
| unseen_recipes_v1 | Hybrid_B_laplacian_E_anchor_vs_B_lap_plus_refine | 7/25 | 0/25 | 5/25 | 9/25 | 23/25 | 12/25 |
| unseen_recipes_v1 | Hybrid_B_laplacian_E_anchor_vs_E_direct_vertex_residual | 23/25 | 14/25 | 23/25 | 19/25 | 3/25 | 11/25 |

Relative OOD improvements are not called successful refinement unless the Hybrid aggregate gain is positive.

## Lambda sensitivity and endpoint audit

Small lambda collapses toward the unstable unanchored Laplacian inverse: validation CD is worst at `1e-4`. The useful CD basin is centered around `1e-2`–`1e-1`, with the validation optimum at `3e-2`. Increasing lambda monotonically reduces Hybrid→E vertex distance over the tested range; lambda `3` is already close to E but is not the mathematical infinity endpoint.

The recovery admits an exact mode-wise characterization, provided the
differential endpoint is defined correctly. Let
`A_R=L_U^T L_U=Q Lambda Q^T` and choose `V_B_dagger` such that
`A_R V_B_dagger=L_U^T delta_B` (with its component-nullspace gauge copied from
`V_E` only to make vertex-space comparisons unambiguous). Then
`v_H,k=Lambda_k/(Lambda_k+lambda) v_B_dagger,k + lambda/(Lambda_k+lambda) v_E,k`
holds exactly. The archived Arm-B recovered mesh is a separate comparator
because it contains its own `1e-2 V_input` anchor and cannot be substituted for
`V_B_dagger` in this identity. Recovery itself still performs no explicit
eigendecomposition; only the diagnostic band projectors are approximated.

## Exact recovery-operator spectral addendum

The direct `A_R=L_U^T L_U` audit passed all 100 validation/test meshes. Maximum
normal-equation relative residual was `2.839e-12`; independently solving and
summing the B and E transfer terms reproduced the tight Hybrid with maximum
vertex RMS `1.005e-11`.

| Split | Change | Lambda<lambda/2 | lambda/2<=Lambda<2lambda | Lambda>=2lambda |
|---|---|---:|---:|---:|
| validation | Hybrid minus V_B_dagger | 99.944% | 0.049% | 0.007% |
| validation | Hybrid minus archived B | 80.606% | 14.598% | 4.797% |
| validation | Hybrid minus E | 11.666% | 17.816% | 70.518% |
| test | Hybrid minus V_B_dagger | 99.862% | 0.125% | 0.012% |
| test | Hybrid minus archived B | 80.932% | 13.173% | 5.896% |
| test | Hybrid minus E | 9.577% | 17.183% | 73.240% |

This directly supports the low-mode hypothesis under the actual recovery
operator. On test, `99.930%` of Hybrid-minus-archived-B energy also lies in the
lowest mesh-relative third of `Lambda/Lambda_max`. The tight Hybrid spectra are
`Hybrid-E` is complementary: `73.240%` of its test change energy lies in the
B-dominant interval. The tight Hybrid spectra are reference solves for the
operator identity; the primary frozen table above retains its established
`tol=1e-4` execution.

Full per-sample energies, exactness audits and plots are in
[`../recovery_operator_spectrum_v1/REPORT.md`](../recovery_operator_spectrum_v1/REPORT.md).

## Decision

Classification: **HBR3**.

Hybrid improves matched geometry and moves OOD behavior from E toward B, but does not fully retain B's OOD Chamfer.

Recommendation: the frozen fusion is strong enough to justify a later, separately controlled jointly trained hybrid ablation, but it does not authorize scaling or retraining automatically.

Metric protocol: `mlr.learned_laplacian.evaluation.evaluate_mesh_geometry;area_weighted_triangle_surface_sampling;bidirectional_sampled_surface_to_exact_triangle_surface;surface_samples=3000;seed=7;fscore_threshold=0.01;alignment=shared_prepared_coordinate_frame_no_ICP`.
