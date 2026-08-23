# Sofa50 stronger coarse-mesh smoothing v2

Status date: 2026-08-23 BST.

The `legacy_v1` multi-topology Sofa50 coarse meshes remain reproducible and are
not overwritten. New preparation defaults to the versioned
`strong_smooth_v2` profile and writes `Sofa50MultiTopologyRawLap500_v2`.

## Controlled change

Only the final uniform Laplacian mesh-smoothing pass changes. Topology recipes,
normal/tangent perturbation magnitudes, perturbation-field smoothing, random
seed namespace, 28 views, 960 resolution, clean references and native raw
Laplacian targets remain unchanged.

| Regime | legacy_v1 | strong_smooth_v2 | attenuation proxy: old -> new |
|---|---:|---:|---:|
| mild | 2 iterations × 0.08 | 6 × 0.12 | 0.1536 -> 0.5356 |
| strong | 4 × 0.12 | 10 × 0.15 | 0.4003 -> 0.8031 |
| unseen intermediate | 3 × 0.10 | 8 × 0.135 | 0.2710 -> 0.6866 |

The attenuation proxy is `1 - (1 - strength)^iterations`. It is an audit of
the smoothing budget, not a claim about exact spectral attenuation on every
mesh.

## Local preflight

On the original topology of five held-out Sofa objects, using identical
perturbation settings and seeds:

| Regime | Mean displacement / bbox diagonal: old -> new | P95: old -> new | Flipped-face fraction: old -> new |
|---|---:|---:|---:|
| mild | 0.00273 -> 0.00909 | 0.01247 -> 0.04361 | 0.329% -> 1.511% |
| strong | 0.00686 -> 0.01503 | 0.03243 -> 0.06893 | 1.022% -> 2.674% |

A separate 20-case preflight covering all ten topology variants on two held-out
objects found zero degenerate faces, zero near-zero faces at the normalized
`1e-10` threshold and a maximum same-index flip fraction of 1.972%. These are
preflight observations; the merged 500-sample v2 audit remains mandatory before
training.

## Safety and launch status

- v1 Slurm scripts explicitly request `legacy_v1`.
- v2 preparation, merge and from-scratch training scripts use separate output
  paths and refuse to reuse mismatched profile audits.
- Per-sample audits now record bbox-normalized displacement, the actual
  smoothing displacement, smoothing budget and invalidity indicators.
- Preparation `17077` and merge/full audit `17079` completed successfully for
  500/500 samples; `contract_audit=true` and all strong-smoothing budget checks
  passed.
- From-scratch job `17082` completed successfully on two L40 GPUs with four
  local meshes accumulated per rank. The effective global batch was eight and
  the full 20,000-step budget was preserved. Runtime was 15.279 hours; the
  final train loss was `1.83395e-6`, and the best/final selection validation
  loss was `2.26915e-6` at step 20,000.
- Jobs `17110`-`17113` completed the controlled v1-versus-v2 validation/test and
  downstream recovery on the v2 prepared meshes. Both 20k checkpoints see the
  same samples and initial meshes. Primary geometry is recomputed with the
  unified area-weighted surface evaluator in the shared prepared frame, with
  no ICP or test-time alignment.
- Superseded 8×Blackwell job `17080` and 4×L40 job `17081` were both cancelled
  before starting and consumed zero runtime.

## Controlled test and downstream recovery

Contract audit: **true**. Both 20k models were evaluated on the same 50 v2
strong-smoothing test meshes, so the initial mesh, vertex ordering, target,
confidence/visibility inputs and recovery solver are paired. Primary geometry
uses area-weighted triangle-surface sampling and exact bidirectional
point-to-triangle-surface distances in the shared prepared frame; no ICP or
test-time alignment is used.

| Metric | v1 model on v2 inputs | v2 strong-smoothing model |
|---|---:|---:|
| Raw EPE | 0.00840367 | **0.00276820** |
| Raw RMS | 0.0234761 | **0.00843035** |
| Top-10% EPE | 0.0447991 | **0.00812695** |
| Top-1% EPE | 0.143090 | **0.0206836** |
| Unified refined Chamfer | **0.00426879** | 0.00451747 |
| Unified P2S | **0.00426879** | 0.00451747 |
| Normal consistency | **0.960320** | 0.952386 |
| Introduced flips | **12,813** | 46,339 |
| Improved over common initial | **38/50** | 26/50 |

The common initial Chamfer is `0.00438635`. The v1 model gives a small mean
geometry improvement despite its much larger raw prediction error, whereas v2
slightly worsens mean geometry despite strongly improving raw EPE and the
high-curvature tail. This is direct evidence that prediction improvement does
not transfer through the frozen recovery configuration under stronger
smoothing; no alternative recovery method is introduced in this experiment.
