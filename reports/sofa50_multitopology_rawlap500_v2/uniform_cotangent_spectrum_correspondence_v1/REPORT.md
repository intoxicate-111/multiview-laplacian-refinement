# Sofa50 recovery versus cotangent operator-spectrum correspondence

Contract audit: **true**. Read-only analysis of **100** frozen validation/test meshes; no checkpoint, mesh, recovery setting, or prior result was modified.

## Operators and sampled modes

The actual frozen recovery operator is

```text
L_rw = I - D^-1 A_adj,        A_U = L_rw^T L_rw,
A_U q_k = Lambda_k q_k.
```

Intrinsic frequency is defined independently on the clean GT geometry with the standard symmetric cotangent stiffness `C` and lumped barycentric mass `M`:

```text
C phi_i = mu_i M phi_i,
mu_cot(q_k) = (q_k^T C q_k) / (q_k^T M q_k),
r_U(phi_i) = (phi_i^T A_U phi_i) / (phi_i^T phi_i).
```

Because meshes contain 5,716–43,246 vertices, full spectra are not numerically practical. Per operator and mesh, the audit extracts **8 lowest non-null, 8 middle, and 8 largest** sparse eigenmodes. Middle modes are nearest half the measured maximum eigenvalue. Formulas above are evaluated exactly on these modes; eigenpair tolerances and residual gates are reported below.

Connected-component constant modes are represented explicitly and removed before correlation. Recovery and cotangent nullspaces are never included as data points.

## Bidirectional monotonic correspondence

Correlations are computed per mesh over its 24 sampled non-null modes, then macro-averaged. Confidence intervals bootstrap meshes.

| Split | Direction | Pearson [95% CI] | Spearman [95% CI] | Log-Pearson | Band diagonal |
|---|---|---:|---:|---:|---:|
| validation | recovery_to_cotangent | 0.53133 [0.48298, 0.57878] | 0.80049 [0.77504, 0.82487] | 0.96694 | 33.333% [33.333%, 33.333%] |
| validation | cotangent_to_recovery | 0.62765 [0.56558, 0.68794] | 0.64344 [0.61907, 0.66871] | 0.94104 | 66.167% [65.583%, 66.750%] |
| test | recovery_to_cotangent | 0.53922 [0.48337, 0.59277] | 0.74094 [0.70243, 0.77741] | 0.90296 | 33.333% [33.333%, 33.333%] |
| test | cotangent_to_recovery | 0.37702 [0.32124, 0.43778] | 0.65287 [0.62392, 0.68136] | 0.87739 | 64.833% [63.500%, 66.000%] |

`recovery_to_cotangent` orders `A_U` eigenmodes by `Lambda_k` and measures their cotangent Rayleigh frequency. `cotangent_to_recovery` orders generalized cotangent modes by `mu_i` and measures the recovery response. Band diagonal is the fraction whose normalized target response falls in the same low/mid/high third as its source sample band.

![Recovery modes versus cotangent frequency](recovery_modes_vs_cotangent_frequency.png)

![Cotangent modes versus recovery response](cotangent_modes_vs_recovery_response.png)

## Low/mid/high band correspondence

Each cell is the macro fraction of eight source-band modes whose normalized target response falls in the indicated target third; brackets are the 95% mesh-bootstrap interval.

### validation: `recovery_to_cotangent`

| Source band | Target low | Target mid | Target high |
|---|---:|---:|---:|
| low | 100.00% [100.00%, 100.00%] | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] |
| mid | 100.00% [100.00%, 100.00%] | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] |
| high | 100.00% [100.00%, 100.00%] | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] |

### validation: `cotangent_to_recovery`

| Source band | Target low | Target mid | Target high |
|---|---:|---:|---:|
| low | 99.50% [98.75%, 100.00%] | 0.50% [0.00%, 1.25%] | 0.00% [0.00%, 0.00%] |
| mid | 1.00% [0.00%, 2.50%] | 86.00% [80.50%, 91.25%] | 13.00% [8.00%, 18.50%] |
| high | 0.50% [0.00%, 1.25%] | 86.50% [81.50%, 91.25%] | 13.00% [8.50%, 17.75%] |

### test: `recovery_to_cotangent`

| Source band | Target low | Target mid | Target high |
|---|---:|---:|---:|
| low | 100.00% [100.00%, 100.00%] | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] |
| mid | 100.00% [100.00%, 100.00%] | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] |
| high | 100.00% [100.00%, 100.00%] | 0.00% [0.00%, 0.00%] | 0.00% [0.00%, 0.00%] |

### test: `cotangent_to_recovery`

| Source band | Target low | Target mid | Target high |
|---|---:|---:|---:|
| low | 97.50% [93.75%, 100.00%] | 2.50% [0.00%, 6.25%] | 0.00% [0.00%, 0.00%] |
| mid | 3.25% [1.25%, 5.50%] | 92.50% [89.50%, 95.25%] | 4.25% [2.25%, 6.25%] |
| high | 2.25% [1.00%, 3.75%] | 93.25% [90.00%, 96.25%] | 4.50% [2.50%, 7.00%] |

## Cross-basis overlap

Recovery modes are renormalized under the barycentric `M` inner product; cotangent modes are already `M`-orthonormal. Each heatmap entry is the squared pairwise `M`-cosine, so sign ambiguity does not matter.

| Split | Same-band overlap | Off-band overlap | Difference [95% CI] | Sampled-basis capture |
|---|---:|---:|---:|---:|
| validation | 0.00909152 | 8.37541e-05 | 0.00900776 [0.00622567, 0.0118049] | 0.0740722 |
| test | 0.00931942 | 0.000253394 | 0.00906602 [0.00704136, 0.0111376] | 0.0786097 |

![Cross-basis overlap](cross_basis_overlap_heatmap.png)

## Decision

Classification: **PARTIAL_PROXY**.

Predeclared rule: strong proxy requires both directions' bootstrap Spearman lower bounds above 0.5 and mean band-diagonal fraction above 60%; partial proxy requires both lower bounds above 0 and both mean Spearman correlations above 0.3. Observed result: the two orderings have positive but incomplete correspondence.

Recovery-operator response and intrinsic cotangent frequency remain distinct quantities even when correlated. This report does not relabel `A_U` eigenvalues as Laplace–Beltrami frequencies.

## Main finding

The answer to the main question is **yes only as a coarse, partial ordering; no
as a calibrated or mode-wise substitute for intrinsic cotangent frequency**.
On test, the overall Spearman correlation is `0.74094` from recovery modes to
cotangent Rayleigh frequency and `0.65287` in the reverse direction, with both
bootstrap intervals strictly positive. This supports a broad progression from
smoother to more oscillatory modes.

The correspondence largely comes from separation between the sampled spectral
regions, not reliable ordering within them. Test within-band Spearman values
for recovery-to-cotangent are `0.55810` (low), `-0.08619` (mid), and `0.03048`
(high); reverse values are `0.06952`, `0.01476`, and `-0.09381`. Moreover, all
sampled recovery low/mid/high modes fall in the lowest absolute third of the
cotangent spectrum. Even recovery-high modes have mean cotangent response only
`0.002056 mu_max`, and the largest observed response is `0.025398 mu_max`.
Conversely, `93.25%` of cotangent-high test modes produce only a mid-band
recovery response, while `4.50%` reach the recovery-high third.

The cross-basis result is similarly qualified. Same-band squared `M`-cosine is
higher than off-band overlap, but the visible alignment is concentrated in the
low--low block; the 24 sampled cotangent modes capture only `7.86%` of a sampled
recovery mode on average. Thus `A_U` provides a meaningful operator-specific
coarse spectral ordering for the exact Hybrid transfer analysis, but its modes
should not be described as cotangent Laplace--Beltrami modes or its eigenvalues
as intrinsic geometric frequencies.

## Numerical audit

ARPACK tolerance: `1.0e-06`; maximum iterations: `20000`. Maximum operator-scale backward eigen-residual: recovery `4.984e-07`, cotangent transformed `4.081e-07`. Maximum within-band orthogonality error: `1.430e-12`; maximum overlap with an explicitly removed component-null basis: `4.231e-06` (gate `<1.0e-05`). Protected cotangent triangles: `0`.

The cotangent operator and mass matrix use clean GT vertices only for this read-only intrinsic-frequency analysis. They do not enter any frozen prediction or recovery solve.
