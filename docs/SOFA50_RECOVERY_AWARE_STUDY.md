# Sofa50 v2 recovery-aware training study

Status date: 2026-08-24 BST.

This document records the current matched-domain study on
`Sofa50MultiTopologyRawLap500_v2`. It supersedes the historical
visibility/confidence/Adam recovery as the active experimental recovery line,
but does not rewrite or overwrite any frozen benchmark result.

## Why the recovery line changed

The exact native target is

$$
\delta^*=L V_{\mathrm{clean}},
$$

where `L` is the row-normalised uniform current-graph Laplacian and
`V_clean` has the same vertex ordering as the input mesh. A direct sparse
least-squares solve with only a per-component centroid gauge reconstructs the
clean geometry closely: mean oracle efficiency is `0.94293` on `legacy_v1`
and `0.92366` on `strong_smooth_v2`. The representation is therefore not the
main exact-target ceiling.

For the frozen v2 recovery, adding the hard any-view visibility mask after the
`lambda_anchor=0.01` arm reduces mean recovery efficiency from `0.34258` to
`0.16875`; 44/50 samples become worse. Confidence changes this result only at
numerical noise scale. The production sparse route also executes L2 rather
than the configured recovery Huber, and increasing Adam from 200 to 2,000
steps changes mean v2 efficiency only from `0.16876` to `0.18635`.

These observations motivate an all-equation regularised sparse solve:

$$
\widehat V_\lambda=
\arg\min_V
\left\lVert L V-\widehat\delta\right\rVert_F^2+
\lambda\left\lVert V-V_{\mathrm{input}}\right\rVert_F^2.
$$

Variables: `V in R^(N x 3)` is the unknown refined mesh,
`V_input in R^(N x 3)` is the initial mesh, `L in R^(N x N)` is fixed by its
connectivity, `delta_hat in R^(N x 3)` is the predicted raw Laplacian, and
`lambda>0` is the positional regularisation coefficient. Every Laplacian row
is used; visibility, confidence, recovery Huber and Adam are absent.

The equivalent normal equations are

$$
(L^\top L+\lambda I)\widehat V_\lambda
=L^\top\widehat\delta+\lambda V_{\mathrm{input}}.
$$

Variables: `L^T` is the transpose of `L`, and `I` is the `N x N` identity.
Standalone evaluation uses sparse LSMR/LSQR-equivalent least squares. Training
uses a differentiable sparse PCG implementation of the same system.

## Completed Arms A and B

Both arms use the same 400/50/50 split, 28 native-960 views, C2F2+HF network,
826,115 parameters, seed 7, effective global batch 8 and 20,000 optimiser
steps. The confidence head is disabled.

Arm A trains only the direct raw-Laplacian Huber objective:

$$
\mathcal L_A=\mathcal L_{\mathrm{lap}}.
$$

Arm B integrates the prediction with `lambda=10^-2` and adds same-index
vertex supervision:

$$
\mathcal L_{\mathrm{vertex}}=
\frac{1}{N}\sum_{i=1}^{N}
\left\lVert \widehat V_{\lambda,i}-V_{\mathrm{clean},i}\right\rVert_2^2,
\qquad
\mathcal L_B=\mathcal L_{\mathrm{lap}}+\beta\mathcal L_{\mathrm{vertex}},
\quad \beta=10^{-2}.
$$

Variables: `V_hat_lambda,i` and `V_clean,i` are the recovered and clean 3D
positions of vertex `i`; `L_vertex` is a mean of squared 3D Euclidean norms,
not a mean over unrelated coordinate samples; and `beta` weights geometric
supervision. Clean vertices are available only on the loss side and never
enter model inputs or the sparse solve.

| Test metric | Arm A: Lap only | Arm B: Lap + vertex |
|---|---:|---:|
| Raw EPE | **0.00252641** | 0.00263986 |
| Raw RMS | 0.00737725 | **0.00683290** |
| Top-10% EPE | 0.00751175 | **0.00737282** |
| Top-1% EPE | 0.0182152 | **0.0159263** |
| Refined Chamfer | 0.00395529 | **0.00358497** |
| Relative Chamfer gain | 7.21% | **13.04%** |
| Mean recovery efficiency | 0.07206 | **0.13036** |
| P2S p95 | 0.0122582 | **0.0105581** |
| F-score | 0.917435 | **0.935013** |
| Normal consistency | 0.954902 | **0.959366** |
| Introduced flips | 53,838 | **52,338** |
| Recovered vertex RMS | 0.0135181 | **0.0115532** |

Arm B has lower test Chamfer on 32/50 samples and lower recovered vertex RMS
on 43/50, while it has lower raw EPE on only 10/50. The completed evidence
therefore supports the narrow claim that recovery-aware supervision improves
the geometric utility of the predicted differential field; it does not claim
that all raw prediction metrics improve.

The strict hardware/sharding contract is false because the completed A/B jobs
moved from two L40s to eight RTX PRO 6000 Blackwell GPUs at epoch boundaries.
The effective global batch, optimiser-step budget and executable mathematical
contract remained fixed.

## Running lambda extension: Arms C and D

Arms C and D preserve Arm B and change only the differentiable/evaluation
regularisation:

| Arm | `lambda` | `beta` | Status at 2026-08-24 snapshot |
|---|---:|---:|---|
| C | `10^-3` | `10^-2` | Job `17274` running on 8 Blackwell GPUs; step 3,200/20,000, zero failed solves and zero NaN/Inf. |
| D | `10^-4` | `10^-2` | Job `17275` queued after C. |

The PCG tolerance remains `10^-4`. Preflight showed float32 stagnation at the
smaller lambdas, so C/D use float64 PCG with at most 2,048 iterations. This is
a documented numerical execution change; neither lambda nor the objective was
silently modified. No C/D scientific conclusion is valid until the dependent
validation/test evaluation and report merge complete.

## Direct-vertex control: Arm E

Arm E keeps the same C2F2+HF encoder, graph network, `N x 3` output width and
826,115 parameters, but changes the output semantics to a residual vertex
displacement:

$$
\Delta V_{\mathrm{pred}}=
f_\theta(I_{1:M},K_{1:M},E_{1:M},V_{\mathrm{input}},F),
\qquad
V_{\mathrm{refined}}=V_{\mathrm{input}}+\Delta V_{\mathrm{pred}}.
$$

Variables: `f_theta` is the shared predictor family; `I`, `K` and `E` are the
28 RGB views, intrinsics and extrinsics; `F` is input connectivity; and
`Delta V_pred in R^(N x 3)` is the direct displacement. Arm E does not use
`L`, a Laplacian target, sparse integration, PCG/LSMR, lambda, visibility,
confidence, recovery Huber, Adam or post-processing.

Its target and loss are

$$
\Delta V^*=V_{\mathrm{clean}}-V_{\mathrm{input}},
\qquad
\mathcal L_E=\frac{1}{N}\sum_{i=1}^{N}
\left\lVert\Delta V_{\mathrm{pred},i}-\Delta V_i^*\right\rVert_2^2.
$$

Variables: `Delta V*` is the exact same-index clean displacement and `L_E` is
the mean squared 3D displacement error. GT vertices are loss-only. The
implementation audit passes on all 500 prepared samples. Job `17278` is queued
after D; evaluation `17279`, A-E merge `17280` and matched visualisation
`17281` are dependency-gated. H1 (vertex loss alone) versus H2 (differential
representation plus structured integration) remains undecided until those
jobs finish.

## Decision boundary

- Use raw EPE/RMS/tail metrics only for A-D, which predict Laplacians. Do not
  report Arm E as raw Laplacian EPE.
- Select among recovery-aware Laplacian arms by validation recovered geometry,
  then freeze selection before test interpretation.
- Compare Arm E to Arm B and to the validation-selected best of B/C/D with
  paired Chamfer, vertex RMS, P2S p95, F-score, normal and flips.
- Do not start 2,000-mesh `strong_smooth_v2` scaling from these intermediate
  results.
