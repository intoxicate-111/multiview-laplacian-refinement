# Sofa50 exact recovery-operator spectral characterization

Contract audit: **true**. All 100 validation/test meshes passed.

Read-only analysis of the real frozen-hybrid operator `A=L_U^T L_U`, with `L_U=I-D^-1 A_adj` and `lambda=3e-2`.

## Exact characterization

Let `b=L_U^T delta_B` and choose `V_B_dagger` such that `A V_B_dagger=b`, with its component-nullspace gauge copied from `V_E`. For `A=Q Lambda Q^T`, the recovery is exactly

```text
v_H,k = Lambda_k/(Lambda_k+lambda) v_B_dagger,k
      + lambda/(Lambda_k+lambda) v_E,k.
```

This exact identity does not use the archived Arm-B recovered mesh, which has its own `1e-2 V_input` anchor and is reported separately.
The reported operator spectra use tight float64 reference solves; the original frozen-Hybrid table used its established `tol=1e-4` execution.

Maximum normal-equation residual: `2.839e-12`. Maximum transfer-decomposition VRMS: `1.005e-11`.

## Relative operator-spectrum error energy

Each mesh uses its own `Lambda/Lambda_max` coordinate with low `[0,1/3)`, mid `[1/3,2/3)` and high `[2/3,1]`. Values are absolute XYZ energies from Chebyshev--Jackson projectors of `L_U^T L_U`.

| Split | Signal | Total | Low | Mid | High |
|---|---|---:|---:|---:|---:|
| validation | b_dagger_error | 86288.76 | 86280.068 | 4.8310533 | 3.8614646 |
| validation | archived_b_error | 59.811096 | 51.238422 | 4.7545377 | 3.8181361 |
| validation | e_error | 22.568302 | 11.413624 | 5.7485694 | 5.4061085 |
| validation | hybrid_error | 30.186378 | 21.735567 | 4.6725546 | 3.7782568 |
| test | b_dagger_error | 11706.186 | 11688.601 | 11.521795 | 6.0636897 |
| test | archived_b_error | 102.25649 | 84.722613 | 11.487895 | 6.0459809 |
| test | e_error | 55.865855 | 33.710228 | 13.761411 | 8.3942165 |
| test | hybrid_error | 67.336138 | 49.972465 | 11.369614 | 5.9940588 |

## Fusion-regime decomposition

The operator-defined bands are E-dominant `Lambda<lambda/2`, transition `lambda/2<=Lambda<2lambda`, and B-dominant `Lambda>=2lambda`. They correspond to differential transfer weight below 1/3, between 1/3 and 2/3, and above 2/3.

| Split | Change | E-dominant | Transition | B-dominant |
|---|---|---:|---:|---:|
| validation | hybrid_minus_b_dagger | 99.944% | 0.049% | 0.007% |
| validation | hybrid_minus_archived_b | 80.606% | 14.598% | 4.797% |
| validation | hybrid_minus_e | 11.666% | 17.816% | 70.518% |
| test | hybrid_minus_b_dagger | 99.862% | 0.125% | 0.012% |
| test | hybrid_minus_archived_b | 80.932% | 13.173% | 5.896% |
| test | hybrid_minus_e | 9.577% | 17.183% | 73.240% |

## Main finding

The low-mode hypothesis is strongly supported by the real recovery operator. On test, `99.862%` of `Hybrid-V_B_dagger` energy and `80.932%` of `Hybrid-archived-B` energy lie in the E-dominant interval `Lambda<lambda/2`. Only `5.896%` of the latter lies in `Lambda>=2lambda`. Under the mesh-relative partition, `99.930%` of `Hybrid-archived-B` energy is in the lowest third of the spectrum. Conversely, `73.240%` of `Hybrid-E` energy lies in the B-dominant interval, directly showing that B supplies the higher-response correction to E.

![Operator error energy](recovery_operator_error_energy.png)

![Operator hybrid change](recovery_operator_hybrid_change.png)

The analysis uses no GT in prediction or recovery. Clean vertices enter only when defining error signals after all B/E/operator states are fixed.
