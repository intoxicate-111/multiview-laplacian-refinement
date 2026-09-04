# Sofa50 pre-fusion modal error sanity check

Contract audit: **true**.

This analysis deliberately does **not** inspect `H-B` or `H-E`. It projects only the two pre-fusion errors

```text
e_B = V_B^dagger - V_GT
e_E = V_E        - V_GT
Delta E(k) = ||e_E(k)||_2^2 - ||e_B(k)||_2^2.
```

The full eigendecomposition is not materialized. Each plotted value is the sum of the exact requested per-mode quantity over a narrow recovery-response bin, approximated with Jackson-damped Chebyshev projectors of `A_R=L_U^T L_U`. Thus negative values mean Arm-E has lower pre-fusion error energy; positive values mean the unanchored Arm-B solution has lower error energy.

![Pre-fusion modal error advantage](pre_fusion_modal_delta_energy.png)

## Main finding

The pre-fusion check supports a **selective**, not blanket, division of labor. Arm-E has lower error through almost the entire response coordinate. The advantage reverses robustly only in the strongest-response bin `w_B in [35/36,1]`: paired local contrast is `+0.1591` `[+0.1334, +0.1839]` on validation and `+0.1698` `[+0.1300, +0.2089]` on test. `V_B^dagger` is better on `48/50` validation and `45/50` test meshes in that bin.

Conversely, integrating the entire nominal B-dominant interval `w_B>=2/3` still favors Arm-E. Therefore the defensible statement is that the differential branch has a reproducible advantage in the **highest recovery-response modes**; the broad B-dominant interval is not uniformly more accurate for `V_B^dagger`.

## Fusion-response regime totals

The horizontal coordinate is `w_B=Lambda/(Lambda+0.03)` only to label the operator response. No Hybrid vertices enter the calculation. Confidence intervals bootstrap paired meshes.

| Split | Regime | sum Delta E | Aggregate contrast | Paired contrast [95% CI] | B better / E better / tie |
|---|---|---:|---:|---:|---:|
| validation | E-dominant | -86244.409 | -0.9999 | -0.9989 [-0.9993, -0.9985] | 0 / 50 / 0 |
| validation | transition | -17.176828 | -0.8462 | -0.8222 [-0.8421, -0.8010] | 0 / 50 / 0 |
| validation | B-dominant | -4.6067619 | -0.1103 | -0.1055 [-0.1365, -0.0744] | 9 / 41 / 0 |
| test | E-dominant | -11616.256 | -0.9992 | -0.9980 [-0.9988, -0.9971] | 0 / 50 / 0 |
| test | transition | -29.859853 | -0.8531 | -0.8430 [-0.8593, -0.8256] | 0 / 50 / 0 |
| test | B-dominant | -4.2044357 | -0.0414 | -0.0822 [-0.1280, -0.0361] | 19 / 31 / 0 |

## Numerical audit

- Meshes: `100` (50 validation + 50 test).
- Maximum relative spectral-partition residual: `1.993e-13`.
- Maximum component-gauge mismatch between `V_B^dagger` and `V_E`: `4.552e-15`.
- Maximum exact-solve normal-equation residual: `2.839e-12`.
- GT is used only after both frozen branch outputs and the operator are fixed; it is not used in prediction, recovery, bin construction or model selection.

Machine-readable outputs: `modal_delta_energy_bins.csv`, `modal_delta_energy_per_sample.csv`, `modal_delta_energy_regimes.csv`, `exactness_audit.csv`, and `summary.json`.
