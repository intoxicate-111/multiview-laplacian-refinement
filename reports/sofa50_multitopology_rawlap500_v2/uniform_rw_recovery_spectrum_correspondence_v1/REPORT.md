# Sofa50 uniform random-walk versus recovery spectrum correspondence

Contract audit: **true**. Read-only local analysis of **100** frozen validation/test meshes; no checkpoint, mesh, recovery setting, or prior result was modified, and no HPC job was submitted.

## Exact operators and non-symmetric treatment

```text
L_rw = I - D^-1 A_adj,                 A_U = L_rw^T L_rw
L_sym = D^1/2 L_rw D^-1/2
L_sym u_k = lambda_k u_k,              phi_k = D^-1/2 u_k
A_U q_j = Lambda_j q_j.
```

`L_rw` was never passed to a symmetric eigensolver. Its real right modes were obtained through the exact similarity to `L_sym`; recovery modes are the right singular modes of `L_rw`. Connected-component constants were constructed explicitly and excluded.

Two identities are separated from the correspondence test:

```text
r_A(phi_k) = phi_k^T A_U phi_k / phi_k^T phi_k = lambda_k^2
sqrt(Lambda_j) = ||L_rw q_j||_2 / ||q_j||_2.
```

The nontrivial reverse measure is the D-consistent frequency centroid `lambda_eff(q)=q^T D L_rw q / q^T D q`; correspondence tests compare `Lambda` with `lambda_eff(q)^2`. Each operator contributes 8 lowest non-null, 8 middle, and 8 largest modes per mesh. Cross-basis bands use one recovery-response coordinate: recovery middle modes are sampled near `0.5 Lambda_max`, and Laplacian middle modes near `lambda=sqrt(0.5 Lambda_max)` so that `lambda^2` targets the same response.

## Bidirectional correlation

| Split | Direction | Pearson [95% CI] | Spearman [95% CI] | Log-Pearson [95% CI] | Band diagonal [95% CI] |
|---|---|---:|---:|---:|---:|
| validation | laplacian_to_recovery | 1.00000 [1.00000, 1.00000] | 0.99995 [0.99988, 1.00000] | 1.00000 [1.00000, 1.00000] | 100.00% [100.00%, 100.00%] |
| validation | recovery_to_laplacian | 0.98884 [0.98611, 0.99145] | 0.92297 [0.91576, 0.92993] | 0.99991 [0.99988, 0.99993] | 100.00% [100.00%, 100.00%] |
| test | laplacian_to_recovery | 1.00000 [1.00000, 1.00000] | 0.99977 [0.99960, 0.99990] | 1.00000 [1.00000, 1.00000] | 100.00% [100.00%, 100.00%] |
| test | recovery_to_laplacian | 0.98745 [0.98478, 0.99001] | 0.93090 [0.92445, 0.93696] | 0.99990 [0.99988, 0.99993] | 99.92% [99.75%, 100.00%] |

![Laplacian frequency versus recovery response](uniform_rw_recovery_response.png)

## Low/mid/high band correspondence

### validation: `laplacian_to_recovery`

| Source band | Target low | Target mid | Target high |
|---|---:|---:|---:|
| low | 100.00% [100.00%, 100.00%] | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] |
| mid | 0.00% [0.00%, 0.00%] | 100.00% [100.00%, 100.00%] | 0.00% [0.00%, 0.00%] |
| high | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 100.00% [100.00%, 100.00%] |

### validation: `recovery_to_laplacian`

| Source band | Target low | Target mid | Target high |
|---|---:|---:|---:|
| low | 100.00% [100.00%, 100.00%] | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] |
| mid | 0.00% [0.00%, 0.00%] | 100.00% [100.00%, 100.00%] | 0.00% [0.00%, 0.00%] |
| high | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 100.00% [100.00%, 100.00%] |

### test: `laplacian_to_recovery`

| Source band | Target low | Target mid | Target high |
|---|---:|---:|---:|
| low | 100.00% [100.00%, 100.00%] | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] |
| mid | 0.00% [0.00%, 0.00%] | 100.00% [100.00%, 100.00%] | 0.00% [0.00%, 0.00%] |
| high | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] | 100.00% [100.00%, 100.00%] |

### test: `recovery_to_laplacian`

| Source band | Target low | Target mid | Target high |
|---|---:|---:|---:|
| low | 100.00% [100.00%, 100.00%] | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] |
| mid | 0.00% [0.00%, 0.00%] | 100.00% [100.00%, 100.00%] | 0.00% [0.00%, 0.00%] |
| high | 0.00% [0.00%, 0.00%] | 0.25% [0.00%, 0.75%] | 99.75% [99.25%, 100.00%] |

## Mode and subspace overlap

All modes are Euclidean-normalized because `A_U` is Euclidean symmetric and the right modes of `L_rw` are not Euclidean-orthogonal. Band subspaces are independently QR-orthonormalized before principal-angle overlap.

| Split | Pairwise same/off band | Subspace same/off band | Difference [95% CI] |
|---|---:|---:|---:|
| validation | 0.058809 / 0.000001 | 0.470561 / 0.000010 | 0.470551 [0.445113, 0.496086] |
| test | 0.067666 / 0.000003 | 0.541105 / 0.000023 | 0.541082 [0.520204, 0.561896] |

![Mode and band-subspace overlap](uniform_rw_recovery_overlap.png)

## Decision

Classification: **PARTIAL_CORRESPONDENCE**.

Strong correspondence was predeclared to require the test reverse-direction Spearman bootstrap lower bound above 0.8, same-band subspace overlap above 0.75, and substantially diagonal band mapping. Partial correspondence requires a positive reverse Spearman lower bound above 0.3 and same-band subspace overlap above off-band overlap. Otherwise the result is weak.

## Main finding

1. **Ordering:** yes at coarse scale. The nontrivial test direction has Spearman `0.93090` (95% CI `[0.92445, 0.93696]`) and 99.92% low/mid/high band agreement. However, within-band test Spearman is `0.99333` low, `0.18381` mid, and `-0.06905` high; the strong overall value is mainly a between-band result.

2. **Modes:** they are strongly band-aligned but substantially rotated within bands. Test pairwise same-band squared cosine is only `0.06767`; the sampled eight-dimensional same-band subspaces retain `0.54110` overlap, while cross-band overlap is only `0.00002`. Per band, the sampled test subspace overlap is `0.982` low, `0.014` mid, and `0.627` high. The mid-spectrum is dense, so an eight-mode local basis there is not a stable or complete eigenspace; its very low sampled overlap should be read as absence of reliable one-to-one mode identity, not as a claim that the full mid-frequency spaces are orthogonal.

3. **Squared law:** for exact `L_rw` right modes, `r_A(phi)=lambda^2` holds to worst relative error `6.318e-08` by algebra, not approximately. For recovery modes, the nontrivial comparison `Lambda` versus `lambda_eff(q)^2` has pooled test median relative deviation `1.58%`, 90th percentile `14.81%`, and maximum `40.53%`. Thus the sampled recovery response is close in scale but not an exact eigenvalue-square spectrum.

4. **Hybrid gate:** `Lambda/(Lambda+lambda_anchor)` is exact only in the recovery/singular basis. It can be interpreted as a coarse gate inherited from the original uniform spectrum because bands align, but not as exact mode-wise `lambda^2/(lambda^2+lambda_anchor)` gating because the bases rotate and mid/high within-band order is weak.

## Numerical and contract audit

ARPACK tolerance: `1.0e-06`; maximum iterations: `20000`; modes per band: `8`. Maximum operator-scale backward residual: `L_sym` `4.916e-07`, recovery `4.984e-07`. Maximum explicit component-null overlap: `L_sym` `1.879e-07`, recovery `9.805e-13`. Maximum within-band orthogonality error: `L_sym` `9.416e-13`, recovery `7.734e-13`. Mean relative non-normality `||L^T L-LL^T||_F/||L^T L||_F`: `0.075221`. All 100 face hashes, vertex/face counts, component counts, degree statistics, residuals, and nullspace overlaps are stored in `eigensolver_audit.csv`.

Only faces/connectivity from the frozen Sofa50-v2 static samples enter this analysis. No images, clean geometry, cotangent operator, checkpoint prediction, or GT signal is used.
