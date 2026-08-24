# Multi-View Laplacian Refinement

[English](README.md) | [简体中文](README.zh-CN.md)

Method specification: [Canonical Sofa50 pipeline](docs/CANONICAL_SOFA50_PIPELINE.md)

Training guide: [Multi-mesh training](docs/MULTI_MESH_TRAINING.md)

Visibility and recovery: [Visibility-aware recovery report](docs/VISIBILITY_AWARE_RECOVERY_REPORT.md)

Current recovery-aware study: [English](docs/SOFA50_RECOVERY_AWARE_STUDY.md) | [中文](docs/SOFA50_RECOVERY_AWARE_STUDY.zh-CN.md)

Experiment metrics and run status: [Experiment data summary](docs/EXPERIMENT_DATA_SUMMARY.md)

Chamfer evaluator incident: [English report](docs/CHAMFER_EVALUATION_INCIDENT_2026-08-21.md) | [中文报告](docs/CHAMFER_EVALUATION_INCIDENT_2026-08-21.zh-CN.md)

Current Sofa50 controlled ablations: [Direct-raw/loss/expert/image-feature report](docs/SOFA50_CONTROLLED_ABLATIONS_REPORT.zh-CN.md)

Sofa50 stronger coarse-mesh smoothing: [v2 contract](docs/SOFA50_STRONG_SMOOTHING_V2.md) | [中文](docs/SOFA50_STRONG_SMOOTHING_V2.zh-CN.md)

Future2000 local comparisons: [Local task guide](docs/FUTURE2000_LOCAL_COMPARISON_TASKS.md)

Recent commit and experiment record: [4–15 August report and addendum](docs/RECENT_COMMIT_AND_EXPERIMENT_REPORT_2026-08-04_2026-08-14.zh-CN.md)

View-count and query-resolution results: [Ablation report](runs/learned_laplacian/sofa50_c2f2_view_query_resolution_ablation_20k_seed7/analysis/REPORT.md)

28-view current-graph target/loss-space results: [H2 ablation report](runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7/analysis/REPORT.md) | [25-case visual overview](runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7/analysis/comparison_images/B_direct_raw_laplacian/overview_25.png)

## Project status

Status date: 2026-08-24 BST.

OpenMVS policy: the existing Sofa50 OpenMVS meshes are low-quality external
reconstructions and are retained only as out-of-distribution stress tests.
They are not training targets, pseudo-GT, model-selection endpoints or a
desired quality ceiling. See [the OpenMVS input policy](docs/OPENMVS_INPUT_POLICY.md).

| Component | State | Conclusion |
|---|---|---|
| Current-query/current-graph training pipeline | Current mainline | The model directly predicts the raw target `L_current @ P_proxy`; no `h^2` target normalisation or output denormalisation is used. |
| Historical GT-query pipeline | Retained for comparison | Absolute GT `h^2`-normalised Laplacian supervision remains implemented, but it is not the current training mainline. |
| Target-leakage controls | Implemented and tested | Proxy positions and supervised raw/normalised Laplacian values are excluded from model inputs. |
| Sofa50 960 image-resolution ablation | Complete | F2 has lower exact-query error than F0 and F1 at 50,000 optimiser steps. |
| Sofa50 960 C2F2 training | Complete | Three seeds completed at 50,000 optimiser steps. C2F2 is the current lowest-error exact-query configuration. |
| Sofa50 1920 C2F2 training | Complete | Three seeds completed at 20,000 optimiser steps. Mean endpoint error and recovery Chamfer are higher than the 960 result; mean cosine is higher. |
| Expanded-query recovery | Complete | Refinement increases Chamfer distance on all five validation meshes for the evaluated 960 and 1920 C2F2 checkpoints. |
| OpenMVS coarse-mesh recovery | Complete for 48 of 50 meshes; diagnostic only | The low-quality reconstructions are retained as OOD stress inputs, not targets or decision endpoints. Historical refinement metrics remain recorded; two objects have no OpenMVS coarse mesh. |
| OpenMVS projected-GT oracle decomposition | Complete; diagnostic only | Position projection improves unified Chamfer by 6.12%; frozen recovery retains 44.53% of that gain and learned prediction realizes 4.36%. This diagnoses failure on poor OOD inputs and has zero model-selection/scale-up weight. |
| Oracle residual expert | Closed | The 2,000-step diagnostic does not support this branch. |
| 14/28/56-view ablation | Complete | Best validation losses are 0.0139316, 0.0130296 and 0.0138104 for 14, 28 and 56 views. |
| Query-graph resolution ablation | Complete | Best validation losses are 0.0139316 for the GT alias, 0.0614830 for GT-sub1 and 0.0145840 for GT-adaptive. GT-sub2 was excluded from training. |
| 28-view + GT-adaptive combination | Complete | Best validation loss is 0.0131095; raw EPE is 0.002879 on five matched validation meshes. |
| 28-view current-graph H2 ablation | Complete | Direct raw-Laplacian training is best in the unified test/recovery evaluation: raw EPE 0.00300525, refined Chamfer 0.00380671 and 19/25 improved samples. |
| Three-round frozen-model recursion | Complete | Improvement count falls from the Arm-B baseline `19/25` to `12/25`, `7/25` and `2/25`; repeated inference is not a valid improvement path. |
| Stage-2 distribution adaptation | Complete | The best X1-trained arm reaches `16/25` and Chamfer `0.00384032`; it does not exceed the frozen Arm-B baseline. |
| Arm-B Huber saturation diagnostic | Complete | In the top 1% GT-curvature group, 66.049% of vertices have at least one saturated component and gradient retention is 58.436%. |
| Raw MSE versus Huber | Complete | Raw MSE does not lower test Top-10%/Top-1% error or mean Chamfer/P2S. The MSE run used global batch 6 versus 2, so it is not a strict single-variable training comparison. |
| Learned dynamic residual expert and gate | Complete | The learned final improves the jointly trained base on every test sample for raw EPE, Chamfer and P2S. Validation-selected constant-gate and five within-mesh shuffle interventions show that the residual expert is the main contribution and vertex-level gate placement adds a smaller, measurable gain. |
| 960 image-feature ablation | Complete | `F + (F-Gaussian(F))` has the lowest test raw EPE, RMS, Top-10% and Top-1% errors. Gaussian-only features have the best mean Chamfer, normal consistency and improved count (`21/25`). |
| Native-1920 plus high-frequency residual | Complete; non-strict resolution ablation | The 20,000-step four-L40 run lowers Bottom-90% error but worsens test raw EPE/RMS, Top-10%/Top-1%, Chamfer and P2S versus 960+HF. Normal consistency improves and flips decrease. Global batch is 4 versus 2. |
| Sofa50 multi-topology coarse smoothing v2 | 20k training and controlled test/recovery complete | Job `17082` completed from scratch on 2×L40 with final/best validation `2.26915e-6`. On the same 50 strong-smoothing test inputs, v2 lowers raw EPE from v1's `0.00840367` to `0.00276820`, but unified refined Chamfer is worse (`0.00451747` vs `0.00426879`), normal consistency is lower, flips are higher and only 26/50 improve versus 38/50. Stronger smoothing improves target prediction but does not transfer through the frozen recovery contract. |
| Exact-target sparse-recovery diagnosis | Complete | A centroid-gauged all-equation sparse solve reaches mean oracle efficiency `0.92366` on v2. Adding hard visibility after the `0.01` anchor lowers mean efficiency from `0.34258` to `0.16875` and worsens 44/50 samples; confidence is negligible and 2,000 Adam steps do not remove the collapse. |
| Recovery-aware Arm A/B | Complete | With the same `lambda=beta=10^-2` sparse recovery, Arm B lowers test Chamfer from Arm A's `0.00395529` to `0.00358497` and recovered vertex RMS from `0.0135181` to `0.0115532`, despite worse raw EPE. This supports geometric-utility supervision rather than raw-regression improvement. |
| Recovery-aware Arms C/D | Running/queued | Arm C (`lambda=10^-3`) job `17274` is running on 8×Blackwell; Arm D (`10^-4`) job `17275` follows it. The objective and `10^-4` PCG tolerance are fixed; float64/2,048-iteration PCG is a documented stability execution change. No result is claimed yet. |
| Direct-vertex Arm E | Audited and queued | The 826,115-parameter C2F2+HF model predicts `Delta V` and uses only same-index vertex MSE; it contains no Laplacian target, sparse solver or recovery gate. Job `17278` follows D, so the H1-versus-H2 representation conclusion remains open. |
| GT-query direct-raw zero-shot transfer | Complete | Removing `h^2` normalization improves strongly over the historical GT-query arm, but current-mesh recovery reaches only Chamfer `0.00400486` and `4/25`, versus `0.00377832` and `20/25` for supervised current-query HF. |
| Future2000 GT-adaptive scale-up | 200k training and learned-method test complete | The fixed 200,000-step checkpoint was evaluated on 200 held-out objects × 5 variants. Unified Chamfer changes from `0.00776417` to `0.00522955`, with 959/1000 improved samples; mean normal consistency decreases from `0.924252` to `0.895907`, so the distance/normal trade-off remains explicit. |
| Sofa50 same-initial external benchmark | Complete, corrected evaluator | Ours, NDS, nvdiffrec and ExMesh completed 25/25 from the same current mesh and observations. A native-metric aggregation bug was corrected by re-evaluating every archived mesh with one deterministic evaluator; `contract_audit=true`. |
| Future2000 external baselines | Full-1000 comparison complete with explicit invalid metrics | Ours has the lowest aggregate valid-sample Chamfer (`0.00522955`) and improves 959/1000 inputs. Paired Chamfer wins are 742/998 vs NDS, 632/999 vs NDS-28V-full, 799/999 vs nvdiffrec and 955/996 vs ExMesh. Input-contract audit passes; strict/full metric completeness remains false because invalid outputs and ExMesh topology changes are preserved rather than repaired. |
| Automated tests | Passing for the documented changes | Targeted external-adapter, same-initial aggregation, raw-loss, dynamic-expert/gate, image-feature, native-1920 and distributed-training tests pass; the verification commands below remain the source of truth for a fresh checkout. |

The active representation is the synthetic-current, current-query/current-
graph, direct-raw formulation established on Sofa50. The current recovery
study integrates every predicted Laplacian row with a regularised sparse solve;
hard visibility, confidence, recovery Huber and Adam mesh optimisation are
disabled. The supervised field `L_current @ P_proxy` is a target only and is
never passed to the model as an inference feature. The completed Future2000
scale-up uses the 960 high-frequency construction, 28 views, C2F2 and a fixed
200,000-step checkpoint; its full-1000 same-initial external comparison is now
complete, with invalid method outputs retained explicitly.

The earlier GT-query, `h^2`-normalised formulation remains useful historical
context. Its transfer to expanded and OpenMVS query graphs did not improve
geometry, and it is no longer the mathematical mainline described below.
OpenMVS is additionally excluded from the target definition and from future
model-selection or scaling decisions because its initial mesh quality is too
poor for that role.

## Current training method

The model maps 28 calibrated views and the current mesh graph directly to a raw
target Laplacian field:

```text
28-view RGB + cameras + current vertices/connectivity + local geometry
    -> direct raw current-graph Laplacian target
```

For the stored current mesh `P_current`, its faces `F_current`, and the paired
proxy positions `P_proxy`:

```text
L_current       = uniform_laplacian(P_current, F_current)
target_raw      = L_current @ P_proxy
prediction_raw  = model(images, cameras, P_current, F_current)
```

There is no division by `h_current^2`, no target clipping, and no output
denormalisation. `target_scaling` retains the edge-scale definition because
`h_current` is used as a local geometry feature and for validity bookkeeping;
it does not change the raw target when `target_mode = raw_laplacian`.

The current geometry mode is `query_fourier`, local query jitter is disabled,
and the query is exactly the current vertex position. Image features, Fourier-
encoded query position, current vertex normal, relative local edge scale,
degree, valid-view ratio and current connectivity are inference inputs. Neither
`P_proxy` nor either raw/normalised target is a model input.

The current high-frequency image branch uses the encoder feature `F`, a fixed
Gaussian blur `G(F)` with kernel size 5 and sigma 1.0, and samples the
concatenation `[F, F-G(F)]`. Recovery then uses the prediction in the same raw
units:

```text
current mesh vertices
  -> project into 28 calibrated views
  -> aggregate original + high-frequency image features
  -> predict delta_pred_raw directly
  -> all-equation regularised sparse Laplacian integration
```

GT geometry is used only for constructing supervision and evaluation. The
recovery-aware arms use clean vertices only in the training-side vertex loss;
they never enter the model or sparse solve. The dynamic residual expert, gate
and raw-MSE loss remain completed controlled ablations. Direct displacement is
now the separately audited Arm E control, not part of Arms A-D.

## Mathematical specification

The equations below describe the active direct-raw + high-frequency training
path. The historical normalised formulation is isolated in its own subsection.

### Current-graph uniform Laplacian and direct-raw target

Let `N(i)` be the one-ring neighbours of vertex `i`, and let
`d_i = |N(i)|`. The uniform graph Laplacian is

$$
(L X)_i = X_i - \frac{1}{d_i}\sum_{j\in N(i)}X_j,
\qquad
L_{ii}=1,\quad L_{ij}=-\frac{1}{d_i}.
$$

Variables: `i` is the destination vertex, `j` indexes its one-ring set
`N(i)`, and `d_i` is the number of those neighbours. `X in R^{N x C}` stores
a `C`-channel signal on `N` vertices (normally 3D positions), `L in R^{N x N}`
is the row-normalised uniform Laplacian, and `(LX)_i` is its `i`-th output row.

An isolated vertex has a zero Laplacian row. Let the training query mesh be
`X_0 = P_current`, and let `P_proxy` be the paired proxy positions with the same
vertex ordering. The local edge scale and active supervised target are

$$
h_i^{\mathrm{current}}=
\frac{1}{d_i}\sum_{j\in N(i)}\lVert X_{0,i}-X_{0,j}\rVert_2,
\qquad
\delta_i^*=(L_{\mathrm{current}}P_{\mathrm{proxy}})_i.
$$

Variables: `X_0,i in R^3` is current vertex `i`; `h_i^current` is its mean
incident-edge length; `P_proxy in R^{N x 3}` is the paired proxy with the same
vertex ordering; `L_current` is built only from the current faces; and
`delta_i* in R^3` is the supervised raw Laplacian vector. `j`, `N(i)` and
`d_i` retain the one-ring definitions above, and `||.||_2` is Euclidean length.

The network directly predicts `delta_pred_raw` in the same units as
`delta_target_raw = delta*`:

$$
f_\theta(I_{1:M},K_{1:M},E_{1:M},X_0,F)_i
=\delta_i^{\mathrm{pred,raw}}\approx\delta_i^*.
$$

Variables: `f_theta` is the learned predictor with parameters `theta`;
`I_1:M`, `K_1:M` and `E_1:M` are the `M=28` RGB images, intrinsics and
world-to-camera extrinsics; `F` is the current face array; and
`delta_i^(pred,raw)` is the three-component output at vertex `i`.

No factor of `(h_i^current)^2` is applied to either side. In the active config,
`target_scaling.method = square_of_mean_incident_edge_length` defines available
scale metadata, while `target_mode = raw_laplacian` selects the raw tensor before
the loss. With `clip_max_norm = null`, the target is also not clipped.

### Current-query contract

The mainline query is the current vertex itself:

$$
q_i=X_{0,i}.
$$

Variables: `q_i in R^3` is the image-projection and positional-encoding query
for vertex `i`, and `X_0,i` is that vertex's stored current position. Equality
means no query offset or jitter is applied.

Both `query_training.enabled` and `local_query_jitter.enabled` are false. The
current connectivity, `P_proxy`, target, local scales and Laplacian operator are
therefore unchanged by training-time query augmentation.

### Projection, renderer visibility and multi-view aggregation

For view `v`, world-to-camera projection is

$$
y_{vi}=E_v[q_i^\top,1]^\top,
\qquad
\widetilde p_{vi}=K_v y_{vi},
\qquad
(u_{vi},v_{vi})=
\left(\frac{\widetilde p_{vi,x}}{\widetilde p_{vi,z}},
      \frac{\widetilde p_{vi,y}}{\widetilde p_{vi,z}}\right).
$$

Variables: `v in {1,...,M}` indexes a camera; `[q_i^T,1]^T in R^4` is the
homogeneous world point; `E_v in R^{3 x 4}` produces camera coordinates
`y_vi`; `K_v in R^{3 x 3}` produces homogeneous pixels `p_tilde_vi`; and
`(u_vi,v_vi)` are the inhomogeneous pixel coordinates obtained by dividing by
camera depth `p_tilde_vi,z`.

Let `f_vi` indicate positive depth and in-frame projection. Let `r_vi` be the
precomputed renderer-native back-face and occlusion result. The feature-sampling
mask is

$$
z_{vi}=f_{vi}r_{vi}\in\{0,1\}.
$$

Variables: `f_vi` is 1 only for positive-depth, in-frame projections; `r_vi`
is 1 only when the renderer's front-face and occlusion tests accept vertex `i`
in view `v`; and `z_vi` is their binary conjunction used for feature sampling.

If `F_v(u_vi,v_vi)` is the bilinearly sampled CNN feature, the implemented
masked mean and valid-view ratio are

$$
\overline F_i=
\frac{\sum_{v=1}^{M}z_{vi}F_v(u_{vi},v_{vi})}
     {\max\left(1,\sum_{v=1}^{M}z_{vi}\right)},
\qquad
\rho_i=\frac{1}{M}\sum_{v=1}^{M}z_{vi}.
$$

Variables: `F_v(u_vi,v_vi) in R^C` is the bilinearly sampled feature in view
`v`; `F_bar_i in R^C` is the masked mean for vertex `i`; `rho_i in [0,1]` is
its valid-view fraction; and the `max(1,...)` denominator defines the all-
invisible case without division by zero.

When no view is valid, `F_bar_i = 0` and `rho_i = 0`.

For C2F2, the image encoder is

$$
F_v=\mathrm{Conv}_{3\times3}^{64}\!\left(
\mathrm{ReLU}\!\left(
\mathrm{Conv}_{3\times3,s=1}^{64}\!\left(
\mathrm{ReLU}\!\left(
\mathrm{Conv}_{5\times5,s=1}^{32}(I_v)
\right)\right)\right)\right).
$$

Variables: `I_v in R^{H x W x 3}` is view `v`; each `Conv_(k x k,s)^Cout`
denotes a convolution with kernel size `k`, stride `s` and `Cout` output
channels; `ReLU` is applied elementwise; and `F_v in R^{H x W x 64}` is the
C2F2 encoder output because every active stride is one.

Padding preserves the input spatial resolution in all three convolutions.
The active high-frequency construction is

$$
F_v^{\mathrm{blur}}=G_{5,1.0}(F_v),
\qquad
F_v^{\mathrm{HF}}=F_v-F_v^{\mathrm{blur}},
\qquad
F_v^{\mathrm{out}}=[F_v,F_v^{\mathrm{HF}}].
$$

Variables: `G_(5,1.0)` is a fixed depthwise 5x5 Gaussian blur with sigma 1.0;
`F_v^blur` is its low-frequency result; `F_v^HF` is the residual high-frequency
map; and `[.,.]` concatenates channels, so `F_v^out` has 128 channels.

`G` is a fixed depthwise Gaussian operation with reflect padding and adds no
learned parameters. Since `F_v` has 64 channels, `F_v^out` and the aggregated
image feature have 128 channels. The native-1920 run processes views in chunks
of four with gradient checkpointing; these are execution choices, not a change
to the feature definition. In the masked aggregation above, the active branch
therefore samples `F_v^out` rather than the untransformed `F_v`.

### Feature visibility and historical recovery gates

The any-view mask retained by the historical recovery path is

$$
m_i=\mathbf 1\!\left[\sum_{v=1}^{M}z_{vi}>0\right].
$$

Variables: `1[.]` is the indicator function, `z_vi` is the binary per-view
visibility above, and `m_i in {0,1}` states whether vertex `i` is visible in at
least one of the `M` views.

The optional confidence head predicts a bounded reliability value

$$
c_i=\mathrm{sigmoid}(g_\theta(x_i))\in[0,1].
$$

Variables: `x_i` is the complete vertex representation defined below;
`g_theta` is the confidence side head sharing the training parameter set; and
`c_i` is its bounded predicted reliability for vertex `i`.

Its historical recovery weight is

$$
w_i=m_i c_i.
$$

Variables: `w_i in [0,1]` is the recovery weight for Laplacian row `i`; `m_i`
is the hard any-view visibility gate and `c_i` is learned confidence. With no
confidence head, the implementation substitutes `c_i=1`.

or `w_i = m_i` when the confidence head is disabled. Consequently, a vertex
that is invisible in every view has exactly zero learned-Laplacian weight.

The repository also implements the following Gaussian distance-confidence gate
in the legacy coarse/GT projection path:

$$
g_i=\mathrm{clip}\!\left(
\exp\!\left[-\left(\frac{d_i^{\mathrm{surface}}}{s}\right)^2\right],
g_{\min},1\right).
$$

Variables: `d_i^surface` is the coarse-query-to-GT-surface distance, `s>0` is
the configured distance scale, `g_min` is the lower clamp, and `g_i` is the
legacy Gaussian confidence weight. This `g_i` is unrelated to the confidence
head function `g_theta` above.

Here `d_i^surface` is the distance from a coarse query to the GT surface and
`s` is `distance_confidence_scale`. This Gaussian gate is not renderer
visibility and is not used by the current synthetic-current training path.
The active recovery-aware study uses `z_vi` only for image-feature sampling.
Every row of `L_current` enters its sparse solve with unit weight. The formulas
`m_i`, `c_i` and `w_i=m_i c_i` document the frozen historical baseline, not
the A-D solver.

### Vertex representation and graph network

Let `c_obj` and `s_obj` be the object normalisation centre and scale. The
normalised query is

$$
\widetilde q_i=\frac{q_i-c_{\mathrm{obj}}}{s_{\mathrm{obj}}}.
$$

Variables: `q_i in R^3` is the current query, `c_obj in R^3` is the object's
normalisation centre, `s_obj>0` is its scalar normalisation scale, and
`q_tilde_i in R^3` is the resulting dimensionless coordinate.

With `K = 6` frequencies, the dynamic Fourier encoding is

$$
\phi(\widetilde q_i)=
\left[
\widetilde q_i,
\left\{\sin(2^k\pi\widetilde q_i),
\cos(2^k\pi\widetilde q_i)\right\}_{k=0}^{K-1}
\right].
$$

Variables: `phi` is the positional encoder; `k` indexes the `K=6` octave
frequencies; powers, sine and cosine act componentwise on `q_tilde_i`; and
brackets concatenate the original 3 coordinates with `2K` three-coordinate
sinusoidal blocks, producing 39 channels.

The per-vertex input is

$$
x_i=\left[
\phi(\widetilde q_i),\ n_i,\
\log\!\left(\max(h_i/s_{\mathrm{obj}},10^{-8})\right),\
\log(1+d_i),\ \rho_i,\ \overline F_i
\right].
$$

Variables: `x_i` concatenates the 39D positional encoding, current unit normal
`n_i in R^3`, relative edge scale `h_i/s_obj`, degree `d_i`, visible-view ratio
`rho_i`, and aggregated image feature `F_bar_i in R^128`. `max` protects the
logarithm at zero and all scalar terms contribute one channel each.

For C2F2 with the active HF construction, `phi` has 39 channels and the complete
vertex input has `39 + 3 + 1 + 1 + 1 + 128 = 173` channels. These terms are
position encoding, normal, log relative edge scale, log degree, valid-view
ratio and aggregated image feature. The graph backbone is
`173 -> 256 -> 256`, followed by three 256-channel message-passing blocks and
an output MLP `256 -> 256 -> 3`. Historical confidence-enabled configurations
add a `173 -> 256 -> 1` sigmoid side head; Arms A-E disable it and retain
826,115 parameters.

After an input MLP, graph layer `l` computes

$$
\mu_i^{(l)}=\frac{1}{\max(1,d_i)}
\sum_{j\in N(i)}u_j^{(l)},
$$

Variables: `l` is the graph-layer index, `u_j^(l) in R^256` is neighbour `j`'s
hidden state, and `mu_i^(l) in R^256` is their degree-normalised mean; the
denominator also defines a zero-safe isolated-vertex case.

$$
u_i^{(l+1)}=\mathrm{ReLU}\!\left(
u_i^{(l)}+operatorname{MLP}_l
\left([u_i^{(l)},\mu_i^{(l)}]\right)
\right).
$$

Variables: `u_i^(l)` and `u_i^(l+1)` are vertex `i`'s input and output hidden
states; `MLP_l` is the learned layer-specific update applied to the
concatenation `[u_i^(l),mu_i^(l)]`; the outer residual addition preserves the
previous state; and `ReLU` is elementwise.

The output MLP maps the final graph state directly to
`delta_pred_raw in R^3`. The Python result field is named
`predicted_laplacian`; the legacy `delta_hat_prediction` accessor does not imply
normalisation when `target_mode = raw_laplacian`.

### Training objective

For component residual

$$
r_{ik}=\delta^{\mathrm{pred,raw}}_{ik}-\delta^*_{ik}.
$$

Variables: `i` indexes vertices, `k in {1,2,3}` indexes Cartesian components,
and `r_ik` is the signed raw-space prediction residual for one component.

the component-wise Huber function is

$$
H_\tau(r)=
\begin{cases}
\frac{1}{2}r^2, & |r|\leq\tau,\\
\tau\left(|r|-\frac{1}{2}\tau\right), & |r|>\tau,
\end{cases}
\qquad \tau=0.01.
$$

Variables: `H_tau` is the scalar Huber penalty, `r` is one signed residual,
and `tau` is the transition magnitude between its quadratic and linear
branches; the active training value is `0.01` in raw-Laplacian units.

The per-vertex error and primary loss are

$$
e_i=\frac{1}{3}\sum_{k=1}^{3}H_\tau(r_{ik}),
\qquad
\mathcal L_{\mathrm{lap}}=
\frac{\sum_i a_i e_i}{\max(10^{-12},\sum_i a_i)},
$$

Variables: `e_i` is the mean of the three component Huber penalties; `a_i>=0`
is the prepared validity/target-confidence weight; `L_lap` is their normalised
weighted mean over vertices; and `10^-12` protects an empty valid set.

where `a_i` is the prepared target-confidence/valid-scale weight. The current
full-vertex contract assigns unit weight to valid non-isolated vertices and
zero weight to invalid local scales. There is no curvature weighting, and the
predicted confidence does not enter this primary loss.

Arm A uses only this raw-field objective:

$$
\mathcal L_A=\mathcal L_{\mathrm{lap}}.
$$

Variables: `L_A` is the full Arm-A training objective and `L_lap` is the raw-
Laplacian Huber loss above. Arms B-D add the recovery-aware vertex term defined
in the next subsection. Their confidence head and confidence loss are disabled.

Historical confidence-enabled runs used a detached error-calibration side loss
`L_conf`; that formula remains implemented for reproduction but is not part of
the current A-E study.

### Regularised sparse integration and recovery-aware supervision

For Arms A-D, recover positions from every predicted Laplacian equation:

$$
\widehat X_\lambda=
\arg\min_X
\left\lVert L_{\mathrm{current}}X-\delta^{\mathrm{pred,raw}}\right\rVert_F^2
+\lambda\left\lVert X-X_0\right\rVert_F^2.
$$

Variables: `X in R^(N x 3)` is the unknown mesh, `X_0 in R^(N x 3)` is the
input mesh, `L_current in R^(N x N)` is its fixed row-normalised uniform
Laplacian, `delta^(pred,raw) in R^(N x 3)` is the network output,
`X_hat_lambda` is the unique regularised solution, and `lambda>0` controls
fidelity to the input positions. There is no row mask or learned row weight.

The solve is equivalently

$$
(L_{\mathrm{current}}^\top L_{\mathrm{current}}+\lambda I)
\widehat X_\lambda
=L_{\mathrm{current}}^\top\delta^{\mathrm{pred,raw}}+\lambda X_0.
$$

Variables: `L_current^T` is the transpose operator and `I` is the `N x N`
identity. Evaluation uses sparse LSMR; recovery-aware training uses a
differentiable sparse PCG solve of the same system.

With same-index clean vertices, Arms B-D add

$$
\mathcal L_{\mathrm{vertex}}=
\frac{1}{N}\sum_{i=1}^{N}
\left\lVert\widehat X_{\lambda,i}-X_i^*\right\rVert_2^2,
\qquad
\mathcal L_{B,C,D}=\mathcal L_{\mathrm{lap}}
+\beta\mathcal L_{\mathrm{vertex}},
\quad \beta=10^{-2}.
$$

Variables: `X_i*` is the clean position corresponding exactly to current
vertex `i`, `L_vertex` is the mean squared 3D position error, and `beta` is its
training coefficient. Clean vertices are loss-only and never enter the model
or solve. Arm B uses `lambda=10^-2`; C and D test `10^-3` and `10^-4`.

Arm E instead predicts a direct residual with no Laplacian path:

$$
\Delta X^{\mathrm{pred}}=f_\theta(I,K,E,X_0,F),
\qquad
X^{\mathrm{refined}}=X_0+\Delta X^{\mathrm{pred}},
$$

$$
\Delta X^*=X^*-X_0,
\qquad
\mathcal L_E=\frac{1}{N}\sum_{i=1}^{N}
\left\lVert\Delta X_i^{\mathrm{pred}}-\Delta X_i^*\right\rVert_2^2.
$$

Variables: `Delta X_pred` is Arm E's `N x 3` output, `Delta X*` is the exact
clean displacement and `L_E` is direct vertex-space MSE. Arm E uses the same
encoder/backbone/head width but no `L`, sparse solver, lambda or post-process.

The old visibility/confidence-weighted Huber/Adam recovery remains a frozen
historical baseline. Exact-target matched-domain diagnostics show that its hard
visibility intervention is the largest tested recovery-efficiency loss, so it
must not be described as the active recovery design.

### Reported metrics

For the raw prediction `P = delta_pred_raw` and raw target `T = delta*`, the
principal prediction metrics are

$$
\mathrm{EPE}=\frac{1}{N}\sum_i\lVert P_i-T_i\rVert_2,
$$

Variables: `P,T in R^{N x 3}` are the raw predicted and target Laplacian
fields, `P_i-T_i` is vertex `i`'s vector residual, `N` is the evaluated vertex
count, and `EPE` is its mean Euclidean magnitude.

$$
\mathrm{Cos}_{\mathrm{global}}=
\frac{\langle\mathrm{vec}(P),\mathrm{vec}(T)\rangle}
{\lVert P\rVert_F\lVert T\rVert_F},
\qquad
R_{\mathrm{norm}}=\frac{\lVert P\rVert_F}{\lVert T\rVert_F}.
$$

Variables: `vec(.)` stacks all field components, `<.,.>` is the Euclidean inner
product, `||.||_F` is the Frobenius norm, `Cos_global` is one cosine for the
whole field, and `R_norm` is the predicted-to-target field-magnitude ratio.

Raw RMS and maximum residual are

$$
\mathrm{RMS}_{\mathrm{raw}}=
\sqrt{\frac{1}{N}\sum_i\lVert P_i-T_i\rVert_2^2},
\qquad
\mathrm{Max}_{\mathrm{raw}}=\max_i\lVert P_i-T_i\rVert_2.
$$

Variables: `RMS_raw` is the root mean square of per-vertex vector residual
magnitudes and `Max_raw` is their maximum; `P`, `T`, `N` and `i` have the same
definitions as in EPE.

Bottom-90%, Top-10% and Top-1% groups are defined globally by
`||delta_i*||_2`, not by the prediction. Recovery-weighted raw RMS uses the
fixed evaluation recovery weights and the same raw residual.

The reported bidirectional sampled-surface Chamfer value is

$$
Q_A=\mathrm{SampleSurface}(S_A,n,s),\qquad
Q_B=\mathrm{SampleSurface}(S_B,n,s+1),
\qquad
D_{\mathrm{C}}(A,B)=\frac{1}{2}\left[
\frac{1}{|Q_A|}\sum_{x\in Q_A}d(x,S_B)
+\frac{1}{|Q_B|}\sum_{y\in Q_B}d(y,S_A)
\right],
$$

Variables: `A` is the evaluated mesh and `B` the GT mesh; `S_A` and `S_B` are
their triangle surfaces; `Q_A` and `Q_B` are `n` deterministic area-weighted
surface samples generated with base seeds `s` and `s+1`; `d(x,S)` is the
shortest point-to-triangle-surface distance; and `|.|` denotes set cardinality.
The corrected same-initial benchmark fixes `n=3000` and `s=7`; other reports
serialize their own sample count and base seed.

For same-topology recovered geometry, the reported vertex RMS is

$$
\mathrm{RMS}_{\mathrm{vertex}}=
\sqrt{\frac{1}{N}\sum_{i=1}^{N}
\left\lVert X_i^{\mathrm{refined}}-X_i^*\right\rVert_2^2}.
$$

Variables: `X_i^refined` and `X_i*` are the recovered and clean positions with
the same vertex index, and `N` is the mesh vertex count. This correspondence
metric complements, rather than replaces, the surface Chamfer metric.

### Historical `h^2`-normalised formulation

Older GT-query experiments used

$$
\delta_i^{\mathrm{GT}}=(L_{\mathrm{GT}}V_{\mathrm{GT}})_i,
\qquad
\widehat\delta_i^{\mathrm{GT}}=
\frac{\delta_i^{\mathrm{GT}}}{h_i^2+\varepsilon},
$$

Variables: `V_GT in R^{N x 3}` and `L_GT` are the historical GT vertices and
their graph Laplacian; `delta_i^GT` is the raw GT Laplacian; `h_i` is mean
incident-edge length; `epsilon=10^-12` prevents division by zero; and
`delta_hat_i^GT` is the historical h-squared-normalised target.

and converted a normalised prediction back with
`delta_pred_raw = delta_hat_prediction * (h_current^2 + epsilon)`. That path is
still implemented for reproducibility, but it does not describe the active
Sofa50 direct-raw + HF training run. Native losses from the historical
normalised representation must not be compared numerically with current
raw-space loss values.

## Dataset contract

The active Sofa50 manifests on the HPC are:

```text
/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_v1/manifest.json
/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_native1920_v1/manifest.json
```

Both use the same 250 sample IDs, object-level split and 28 camera poses:
200 training, 25 validation and 25 held-out test variants from 50 objects with
five variants each. The native-1920 observations are rendered at 1920 x 1920;
they are not resized 960 images. Current graphs, proxy positions, raw targets
and renderer visibility follow the same contract in both datasets.

The historical GT-query data roots remain under
`sofa50_refinement/multiview_960` and `sofa50_refinement/multiview_1920`.
Those datasets contain 40/5/5 objects and use `gt_query_manifest.json`; their
expanded manifests are inference-only and are not the active training source.

RGB images remain on disk and are decoded lazily as `uint8`. CUDA training uses
pinned memory, non-blocking transfer and AMP for CNN/GNN forward passes.
Target scaling, loss accumulation and numerical geometry operations remain in
FP32.

## Experiment nomenclature

| Label | Definition |
|---|---|
| C0 | Image feature dimension 16; graph hidden dimension 64; 3 graph layers. |
| C2 | Image feature dimension 64; graph hidden dimension 256; 3 graph layers. |
| F0 | Encoder strides `2, 2`; 240 x 240 feature map for 960 input. |
| F1 | Encoder strides `2, 1`; 480 x 480 feature map for 960 input. |
| F2 | Encoder strides `1, 1`; feature-map resolution equals input resolution. |
| C2F2 | C2 capacity with the F2 image encoder. |

## Completed results

### Exact GT-query prediction at 960

| Run | Seed | All EPE ↓ | Top-10% EPE ↓ | Global cosine ↑ | Prediction/GT norm |
|---|---:|---:|---:|---:|---:|
| C0F0 | 7 | 9.4641 | 30.7221 | 0.7808 | 0.8020 |
| C0F1 | 7 | 9.3786 | 30.3095 | 0.7892 | 0.7938 |
| C0F2 | 7 | 9.1665 | 28.4751 | 0.8227 | 0.8180 |
| C2F2 | 7, 17, 27 | 2.8260 ± 0.0864 | 15.3614 ± 0.4036 | 0.8912 ± 0.0127 | 0.9348 ± 0.0160 |

Original RGB produces lower error than zero RGB in the completed resolution
ablation. The original-minus-zero global-cosine gaps are 0.2236, 0.3315 and
0.3724 for F0, F1 and F2, respectively.

### C2F2 at 960 and 1920

| Input | Training budget | Mean all EPE ↓ | Mean top-10% EPE ↓ | Mean cosine ↑ | Mean expanded Chamfer ↓ |
|---|---:|---:|---:|---:|---:|
| 960 | 50,000 steps, 3 seeds | 2.8282 | 15.3743 | 0.8911 | 0.0011624 |
| 1920 | 20,000 steps, 3 seeds | 3.0928 | 16.3299 | 0.8954 | 0.0012570 |

The two rows do not have the same optimiser-step budget. The 1920 runs do not
establish an improvement over the 960 runs.

### Expanded-query recovery

The shared five-object expanded-validation initial Chamfer is `0.000652884`.
The 960 C2F2 mean refined Chamfer is `0.00116202`; the 1920 C2F2 mean is
`0.00125704`. Each evaluated seed improves `0/5` meshes relative to its initial
mesh.

### OpenMVS coarse-mesh recovery

**Diagnostic-only warning.** These OpenMVS meshes are low-quality external
inputs, not the desired output, target topology, pseudo-GT or a primary method
endpoint. The table below is preserved as an OOD failure/robustness record and
must not drive checkpoint selection or architecture conclusions.

The test uses 48-view COLMAP/OpenMVS coarse meshes, the original 14 Sofa50 RGB
views for prediction, the three 960 C2F2 checkpoints, OpenGL visibility at 480,
and no GT differential transfer. Forty-eight meshes are evaluated; two coarse
meshes are absent.

| Recovery | Initial mean Chamfer | Ensemble refined mean Chamfer | Better meshes | Introduced flips |
|---|---:|---:|---:|---:|
| 200 iterations | 0.0212023 | 0.0213199 | 2/48 | 4,692 |
| 1,000 iterations | 0.0212023 | 0.0213198 | 2/48 | 4,734 |

Increasing recovery from 200 to 1,000 iterations does not change the aggregate
conclusion.

A later 48-case projected-GT diagnostic gives unified Chamfer `0.0469163` for
the initial OpenMVS meshes, `0.0440446` for projected positions, `0.0456376`
after the projected-position raw Laplacian is passed through frozen recovery,
and `0.0467913` for the archived learned prediction. Recovery retains 44.53%
of the projected-position gain and learned prediction realizes 4.36%. These
numbers decompose failure on poor OOD inputs; they do not make OpenMVS a target
or a model-quality endpoint.

### 28-view current-graph target and loss-space ablation

Three C2F2 arms use the same 28-view synthetic-current manifest, seed,
initialisation and 20,000-step budget. Local query jitter is disabled. Native
validation losses are reported in each arm's own loss space and are therefore
not comparable across rows.

| Arm | Output target | Native loss space | Best native val | Test raw EPE ↓ | Test raw cosine ↑ | Refined Chamfer ↓ | Improved |
|---|---|---|---:|---:|---:|---:|---:|
| A | `h^2`-normalised | Normalised output | 0.0184566 | 0.00769237 | 0.933526 | 0.00456011 | 3/25 |
| B | Raw Laplacian | Raw output | 1.58253e-6 | 0.00300525 | 0.998667 | 0.00380671 | 19/25 |
| C | `h^2`-normalised | Raw Laplacian | 2.16552e-6 | 0.00333673 | 0.997419 | 0.00383121 | 16/25 |

The shared initial Chamfer is `0.00391323`. Arm B is the primary result: it has
the lowest unified raw-space errors and recovery Chamfer, and improves 19 of 25
test samples. The very small B/C native losses reflect raw-Laplacian units;
they do not imply a four-order-of-magnitude advantage over A's normalised loss.
The contract audit passed, and the final evaluation ran as three L40 shards
(Slurm array 15686) followed by merge job 15687.

The local result bundle includes the [full report](runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7/analysis/REPORT.md),
[source JSON/CSV tables](runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7/analysis),
[75 comparison OBJ files](runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7/analysis/mesh_comparisons/B_direct_raw_laplacian)
and [25 fixed-camera GT/COARSE/REFINED images](runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7/analysis/comparison_images/B_direct_raw_laplacian).

### Stage-2 adaptation and Huber-tail diagnostics

Three matched 20,000-step continuation arms started from the same 20k Arm-B
checkpoint. Continuing on original X0 inputs, recovered X1 inputs, or a 50/50
mixture all finish at `16/25` improved samples. The best X1-trained checkpoint
has Chamfer `0.00384032`, versus `0.00380687` for the frozen stage-1 model; it
gains two of the six original failures but loses five of the original nineteen
successes. Extra training and X1 distribution adaptation therefore do not beat
the stage-1 result.

A local validation diagnostic over 243,000 vertices groups errors by GT raw
Laplacian magnitude. The top 1% has `13.071x` the bottom-90% mean raw error;
`66.049%` of those vertices have at least one Huber-saturated component. This
group contributes `34.931%` of Huber loss but only `5.785%` of output-gradient
L1, with `58.436%` gradient retention. The result identifies concentrated
tail-gradient compression; linking it causally to Chamfer still requires a
surface-sensitivity experiment.

### Raw loss, dynamic expert and image-feature ablations

The raw-MSE control does not support replacing Huber. On the shared 25-sample
test set, Huber/MSE Top-10% EPE is `0.0122438/0.0128078`, Top-1% EPE is
`0.0371716/0.0380294`, Chamfer is `0.00380692/0.00381317`, and improved count is
`19/25` versus `16/25`. The MSE run used three L40 ranks with global batch 6,
whereas the Huber baseline used global batch 2; this resource-driven difference
is recorded by the audit and prevents a strict single-variable claim.

The from-scratch learned dynamic residual expert produces test raw EPE
`0.00294740`, Chamfer `0.00377438`, normal consistency `0.944879`, 5,699
introduced flips and `19/25` improved samples. Its inference-time causal
ablation selects constant `alpha=0.16` on validation. Constant gating already
improves the jointly trained base on all 25 test samples; the learned spatial
gate then beats the constant gate on Chamfer and P2S for `25/25`, and beats each
of five within-mesh gate shuffles on most samples. For Chamfer, the attribution
diagnostic assigns `90.32%` of the total base-to-learned change to the expert
without spatial modulation and `9.68%` to learned gating. These ratios are
diagnostics, not an independent causal decomposition.

The 960 image-feature experiment keeps the direct-raw C2F2 contract fixed:

| Image feature | Test raw EPE ↓ | Top-10% ↓ | Top-1% ↓ | Chamfer ↓ | Normal ↑ | Improved |
|---|---:|---:|---:|---:|---:|---:|
| Original Arm B | 0.00297471 | 0.0122427 | 0.0371654 | 0.00380683 | 0.942470 | 19/25 |
| Gaussian only | 0.00291322 | 0.0123509 | 0.0376811 | **0.00377507** | **0.944459** | **21/25** |
| Original + HF residual | **0.00288627** | **0.0117524** | **0.0347902** | 0.00377832 | 0.942475 | 20/25 |

Gaussian-only sampling slightly degrades the high-curvature tail while
improving mean downstream geometry. Adding `F-Gaussian(F)` to the original
feature gives the best prediction and tail metrics and still improves mean
Chamfer/P2S over original Arm B.

### Strong-smoothing sparse recovery and recovery-aware training

The v2 exact-target sparse oracle reaches mean efficiency `0.92366` when every
Laplacian row is used and component centroids fix only translation. Under the
old frozen recovery, hard visibility is the largest tested efficiency loss;
confidence is effectively constant and additional Adam steps do not close the
gap. The active study therefore uses regularised all-equation sparse recovery.

Completed A/B test results are:

| Arm | Raw EPE ↓ | Raw RMS ↓ | Chamfer ↓ | Eta ↑ | P2S p95 ↓ | Normal ↑ | Vertex RMS ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| A: Lap only | **0.00252641** | 0.00737725 | 0.00395529 | 0.07206 | 0.0122582 | 0.954902 | 0.0135181 |
| B: Lap + recovery-aware vertex | 0.00263986 | **0.00683290** | **0.00358497** | **0.13036** | **0.0105581** | **0.959366** | **0.0115532** |

Arm B wins paired Chamfer on 32/50 and vertex RMS on 43/50, while winning raw
EPE on only 10/50. Arms C/D (`lambda=10^-3/10^-4`) and the direct-vertex Arm E
are still dependency-gated; see the [current study contract and live-status
snapshot](docs/SOFA50_RECOVERY_AWARE_STUDY.md). No H1/H2 conclusion is drawn
before the matched A-E evaluation.

### Native 1920 plus high-frequency residual, complete

The native-1920 dataset contains the same 250 sample IDs, object-level
`200/25/25` split, 28 camera extrinsics, current graphs, proxy positions,
raw-Laplacian targets and renderer visibility tensors as the 960 HF run.
Intrinsics are scaled for native 1920 rendering; the observations are not
resized 960 images. The minimum native-versus-resized pixel MAE across the
audit is `0.0205764`.

Four-L40 job 15854 completed 20,000 optimiser steps from scratch. View chunks
of four and gradient checkpointing are execution-only memory controls covered
by forward/gradient equivalence tests. The actual global batch is 4 versus 2
for the 960 HF baseline, so this is not a strict single-variable training
comparison.

| Resolution + HF | Raw EPE ↓ | Raw RMS ↓ | Bottom 90% ↓ | Top 10% ↓ | Top 1% ↓ | Chamfer ↓ | Normal ↑ | Flips | Improved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 960 | **0.00288618** | **0.00628203** | 0.00190107 | **0.0117522** | **0.0347895** | **0.00377857** | 0.942504 | 6303 | **20/25** |
| Native 1920 | 0.00290615 | 0.00690893 | **0.00183806** | 0.0125190 | 0.0389263 | 0.00378509 | **0.944522** | **5777** | 18/25 |

Native 1920 does not improve the high-curvature tail or mean downstream
distance. It improves normal consistency and reduces flips, but costs 22.35
hours on four GPUs (`89.39` GPU-hours) versus 3.98 hours on two GPUs (`7.95`
GPU-hours) for 960.

### Future2000 GT-adaptive scale-up

The scale-up uses 2,000 upstream meshes, five deterministic current-mesh
variants per object, an object-level 80/10/10 split (`8000/1000/1000` samples),
28 calibrated views, GT-adaptive subdivision, C2F2, original-plus-HF image
features and the current-graph direct-raw target. Job 16607 starts from a fresh
initialization and runs 200,000 global optimiser steps on seven NVIDIA RTX PRO
6000 Blackwell Server Edition GPUs with global batch 7.

| Step | Rolling train loss | Validation loss |
|---:|---:|---:|
| 20,000 | 5.30099e-6 | 4.99851e-6 |
| 30,000 | 4.82400e-6 | **4.19731e-6** |
| 40,000 | 4.62000e-6 | **3.88000e-6** |
| 50,000 | 4.26000e-6 | 4.23000e-6 |
| 60,000 | 4.19000e-6 | 5.27000e-6 |
| 64,000 | **3.99000e-6** | — |
| 132,000 | 2.10e-6 | — |
| 188,000 | **1.72e-6** | 2.87e-6 at the latest completed validation near step 182,880 |

Historical job 15794 stopped at step 32,000 after a DataLoader worker exhausted 51,200
file descriptors; the other DDP ranks then reached the 30-minute NCCL watchdog
timeout. Replacement job 15795 resumed the intact checkpoint with PyTorch's
`file_system` sharing strategy and non-persistent workers, reached step 64,000,
then failed when a DataLoader worker exhausted `/dev/shm` (`Bus error` and `No
space left on device`). These runs are retained as infrastructure history only.
Job 16607 does not reuse either checkpoint: RGB files are staged to node-local
storage, workers are disabled, and overwrite/resume guards enforce the
from-scratch contract. It completed the fixed 200,000-step checkpoint. The
full 1,000-sample learned-method evaluation changes unified Chamfer from
`0.00776417` to `0.00522955`, with 959/1000 samples improved.

The same-initial external comparison is also complete. Ours has the lowest
aggregate valid-sample Chamfer; paired wins are 742/998 against NDS, 632/999
against NDS-28V-full, 799/999 against nvdiffrec and 955/996 against ExMesh.
The input contract passes, while strict/full metric completeness stays false:
invalid method outputs and ExMesh connectivity changes remain explicit and are
not repaired or silently imputed.

### Sofa50 same-initial external comparison

Ours, NDS, nvdiffrec and ExMesh were run on the same 25 native-1920 Sofa50 test
inputs: identical current/coarse mesh, 28 RGB observations and cameras. GT is
used only by the common evaluator. All four methods completed `25/25` and the
input identity audit passed.

The preliminary aggregation mixed method-native Chamfer implementations. The
shared initial mesh consequently appeared as both `0.00391323` and
`0.01707047`, which invalidated that table. The corrected report recomputes the
common initial and every final mesh using one deterministic 3,000-surface-point
evaluator (seed 7). Native numbers are provenance-only and
`contract_audit=true`. See the bilingual [incident report](docs/CHAMFER_EVALUATION_INCIDENT_2026-08-21.md)
for the corrected protocol. The generated final report remains a local artifact
under the ignored `reports/` tree and is not distributed with the source repository.

| Method | Unified final Chamfer ↓ | Improvement | Improved | Normal ↑ |
|---|---:|---:|---:|---:|
| Ours | 0.011347800 | 33.52% | **25/25** | **0.944514** |
| NDS | **0.011204992** | **34.36%** | 22/25 | 0.873805 |
| nvdiffrec | 0.013654660 | 20.01% | 18/25 | 0.848122 |
| ExMesh | 0.020170615 | -18.16% | 8/25 | 0.845337 |

NDS is marginally lower in mean Chamfer; ours is more consistent across
samples and preserves substantially better normals. These synthetic-protocol
values are not the official DTU ExMesh millimetre metric.

## Installation and verification

```bash
conda env create -f environment.yml
conda activate test
pip install -e ".[train]"
PYTHONPATH=src pytest -q
```

If the environment already exists:

```bash
PYTHONPATH=src conda run --no-capture-output -n test pytest -q
```

## HPC entry points

The checked-in Slurm files contain cluster-specific paths and resource
requests.

```bash
# 960 F0/F1/F2, 50,000 steps
bash scripts/slurm_jobs/submit_resolution_50k_parallel.sh

# 960 C2F2, seeds 7/17/27, 50,000 steps
sbatch scripts/HPC/sofa50_c2_f2_50k_3gpu.slurm

# 1920 C2F2, seeds 7/17/27, current 20,000-step output contract
sbatch scripts/HPC/sofa50_c2_f2_1920_50k_3gpu.slurm

# OpenMVS recovery with OpenGL visibility and 1,000 recovery iterations
sbatch scripts/HPC/test_sofa50_openmvs_coarse_14view_c2f2_48mesh_opengl_480_recovery1000.slurm

# 14/28/56-view and query-resolution ablations, 20,000 steps per arm
sbatch scripts/HPC/c2f2_dataset_ablation_20k.slurm view 14
sbatch scripts/HPC/c2f2_dataset_ablation_20k.slurm query gt_sub1

# Three-L40 sharded H2 evaluation and dependent merge
bash scripts/HPC/submit_sofa50_synthetic_current_28view_h2_ablation_3gpu.sh

# Raw MSE versus Huber training/evaluation
bash scripts/HPC/submit_sofa50_synthetic_current_28view_loss_ablation_3gpu.sh

# Learned dynamic residual expert and inference-time gate ablation
bash scripts/HPC/submit_sofa50_dynamic_residual_expert_from_scratch_4gpu.sh

# Gaussian and original-plus-high-frequency image-feature arms
bash scripts/HPC/submit_sofa50_image_feature_ablation_2x2gpu.sh

# Native-1920 original-plus-high-frequency data, training and evaluation chain
bash scripts/HPC/submit_sofa50_hf1920_4gpu.sh

# Future2000 200k from-scratch smoke and seven-Blackwell training
sbatch scripts/HPC/smoke_future2000_current_28view_hf_7gpu_blackwell.slurm
sbatch scripts/HPC/train_future2000_current_28view_hf_200k_7gpu_blackwell.slurm
```

### Distributed multi-GPU training

`train_multi_mesh_laplacian.py` supports PyTorch DistributedDataParallel when
started with `torchrun`. Training meshes are sharded by rank with deterministic
padding, gradients and training metrics are reduced across ranks, and only rank
0 writes logs, predictions and checkpoints. Checkpoints retain canonical model
keys without a `module.` prefix and remain loadable by single-GPU evaluation.

The Slurm entry point defaults to one node with four L40 GPUs:

```bash
sbatch scripts/HPC/train_multi_mesh_ddp.slurm \
  /path/to/manifest.json \
  /path/to/config.json \
  /path/to/output_dir
```

The same script supports multiple nodes. This example launches eight ranks on
two nodes:

```bash
sbatch --nodes=2 --gres=gpu:L40:4 \
  scripts/HPC/train_multi_mesh_ddp.slurm \
  /path/to/manifest.json \
  /path/to/config.json \
  /path/to/output_dir
```

The global mesh batch is `world_size * gradient_accumulation_meshes`.
`max_optimizer_steps` counts synchronized global optimizer updates; increasing
the world size therefore increases the number of mesh exposures per update and
per fixed optimizer-step budget.

Worker-backed lazy-image training accepts
`data_loading.multiprocessing_sharing_strategy`. Historical Future2000 recovery
runs used `file_system` and non-persistent workers after descriptor exhaustion.
The completed seven-Blackwell run instead used zero DataLoader workers and
staged RGB observations to node-local storage, avoiding both descriptor and
shared-memory failure modes. External-method comparison jobs are
documented in the [local runner guide](docs/FUTURE2000_LOCAL_COMPARISON_TASKS.md)
and should not be resubmitted through Slurm.

The generated 14/28/56-view dataset is located at:

```text
/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/sofa_mesh/sofa50_refinement/multiview_nested_14_28_56_cpu_v3
  gt_query_views_14_manifest.json
  gt_query_views_28_manifest.json
  gt_query_views_56_manifest.json
  expanded_inference_views_14_manifest.json
  expanded_inference_views_28_manifest.json
  expanded_inference_views_56_manifest.json
```

The generated query-graph resolution dataset is located at:

```text
/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/sofa_mesh/sofa50_refinement/multiview_960/query_resolution_ablation_v2
  gt_manifest.json
  gt_sub1_manifest.json
  gt_sub2_manifest.json
  gt_adaptive_manifest.json
```

The combined 28-view + GT-adaptive dataset is located at:

```text
/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/sofa_mesh/sofa50_refinement/view_query_combo_28_gt_adaptive_v1
  manifest.json
  summary.json
```

Each manifest contains 50 objects with a 40/5/5 train/validation/test split.
Prepared graphs contain CUDA-generated `visibility_backface_and_occlusion`.
The manifests passed the downstream training and inference loader checks.

### Local query-position jitter ablation

The ablation uses C2F2, 28 views, five fixed synthetic-current variants per
object, seed 7 and 20,000 optimizer steps. Arm A uses the stored current vertex
positions. Arm B applies training-only isotropic query jitter with component
standard deviation `0.003 h_i` and vector-norm limit `0.009 h_i`. The stored
proxy, normalized target, `h_i`, connectivity and target-construction operator
are unchanged. Validation and test do not apply jitter.

```text
Dataset: /networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_v1/manifest.json
Runs: runs/learned_laplacian/sofa50_synthetic_current_28view_jitter_ablation_seed7
Repository report: docs/SOFA50_LOCAL_QUERY_JITTER_ABLATION_REPORT.zh-CN.md
HPC report: runs/learned_laplacian/sofa50_synthetic_current_28view_jitter_ablation_seed7/analysis/REPORT.md
```

The report contains deterministic validation/test prediction metrics, the
original-RGB/zero-RGB comparison and OpenMVS48 current-mesh recovery metrics.
The OpenMVS portion is secondary, diagnostic-only evidence under the policy
above; the synthetic-current endpoints determine the ablation conclusion.
Arm B records higher best validation loss, test raw endpoint, test raw Top-10%
endpoint and test raw Top-1% endpoint, and lower test raw global cosine; these
controlled synthetic-current values determine the conclusion. In the separate
non-decisional OpenMVS stress record, Arm B also has higher refined Chamfer/P2S
and lower normal consistency, and neither arm improves over its poor initial
OpenMVS mesh.

## Result locations on the HPC

```text
runs/learned_laplacian/sofa50_image_resolution_ablation_50000step
runs/learned_laplacian/sofa50_c2_f2_50000step_3seed
runs/learned_laplacian/sofa50_c2_f2_1920_20000step_3seed
runs/learned_laplacian/sofa50_cf_c2f2_comparison_full
runs/learned_laplacian/sofa50_c2f2_960_vs_1920_full
runs/learned_laplacian/sofa50_c2f2_view_query_combo_28_gt_adaptive_20k_seed7_v1
runs/learned_laplacian/sofa50_synthetic_current_28view_jitter_ablation_seed7
runs/learned_laplacian/sofa50_synthetic_current_28view_h2_normalization_ablation_20k_seed7
runs/learned_laplacian/sofa50_synthetic_current_28view_b_stage2_adaptation_20k_seed7
runs/learned_laplacian/future2000_gt_adaptive_2000mesh_expanded_current_28view_direct_raw_20k_seed7
runs/learned_laplacian/sofa50_openmvs_coarse_14view_c2f2_48mesh_opengl_480
runs/learned_laplacian/sofa50_openmvs_coarse_14view_c2f2_48mesh_opengl_480_recovery1000
```

Checkpoints, prepared datasets and HPC result directories are not distributed
with the source repository.

### Independent ExMesh-protocol benchmark

The official ExMesh DTU comparison is a separate external benchmark. It does
not reuse the synthetic datasets, cameras, renderer, or reconstructed meshes
listed above. Its official-source pins, reproduction gate, common-contract
extractor, failure policy, and six-method sanity gate are documented in
[the ExMesh baseline suite guide](docs/EXMESH_BASELINE_SUITE.md). Its full
execution is distinct from the Sofa50 same-initial comparison above. The
released ExMesh 15-scene reproduction gate has passed (0.60484 mm reproduced
mean CD versus 0.58 mm in the paper); the complete official six-method DTU
benchmark remains gated by the scan-24 shared-coordinate-frame audit. The
local provenance audit established that the intended learned-method DTU
scan-24 current mesh was never generated; the ExMesh PGSR mesh is explicitly
rejected as a silent substitute. Generated provenance reports remain under the
ignored `reports/` tree.
