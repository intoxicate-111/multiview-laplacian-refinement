# Future2000 formal Arm-B full-test comparison and Arm-E preparation

Report date: **2026-08-31**.

Status addendum, **2026-09-04 08:14 BST**: the replacement Arm-E training job
`17888` completed all 200,000 steps with exit `0:0`; validation selected epoch
160 and checkpoint SHA-256
`5a6aaa32bec6edcdd2c30face02c4ae8bc139fef18d4d05b3394c987057cb50f`.
Frozen B+E validation array `18673` is now running, followed by dependency-gated
lambda lock `18677`, sealed test `18678` and comprehensive report `18679`.
No Arm-E/B+E test metric is claimed in this historical Arm-B report; the
original 2026-08-31 snapshot text below is retained as execution provenance.

Input-contract audit: **true**. Formal Arm-B completion: **true**. Overall
contract audit: **false**. Metric completeness: **false**.

The two negative top-level flags come only from invalid outputs in the frozen
external archive: NDS has two invalid Chamfer values, nvdiffrec has one failed
mesh, and ExMesh has four invalid Chamfer values. The formal Arm-B and archived
old-structure Ours rows are complete on all 1,000 test inputs, and every method
receives the exact same current mesh, 28 native-960 RGB observations and camera
matrices for each sample. Ground truth is used only by the evaluator.

## Executive findings

- The formal mixed-loss Future2000 Arm-B reaches Chamfer `0.00476456546`, a
  `38.63%` reduction from the common initial meshes, and improves `975/1000`
  test variants.
- Against the archived old-structure Ours predictor, formal Arm-B lowers mean
  Chamfer by `0.000464982242` (`8.89%` relative), wins `882/1000` variants and
  wins on the object-mean Chamfer for `185/200` independent test objects. The
  10,000-resample object-cluster bootstrap interval is
  `[-0.000580558,-0.000314545]`, excluding zero.
- Formal Arm-B also improves P2S p95, F-score and normal consistency over the
  old structure, while reducing introduced flips by `61.56` faces per sample
  on average and creating no new degenerate faces.
- It has the lowest valid-sample Chamfer among the compared methods and wins
  paired Chamfer on `804/998` valid NDS outputs, `829/999` nvdiffrec outputs and
  `974/996` ExMesh outputs.
- Chamfer and the previously displayed bidirectional P2S mean are exactly the
  same statistic under this evaluator. P2S mean is therefore removed from the
  report as duplicate evidence; P2S p95 remains a distinct tail statistic.
- Future2000 Arm-E has been prepared by reusing the established Sofa50 Arm-E
  architecture and direct-vertex objective for subsequent frozen B+E fusion.
  HPC job `17800` was still `PENDING (Resources)` at this report snapshot, so
  no Arm-E training result or B+E result is claimed here.

## Dataset and independence contract

This dataset is not one mesh with 2,000 random perturbations. Its source
manifest contains **2,000 distinct 3D-FUTURE objects** with 2,000 unique sample
IDs and source paths. Each object is converted into one fixed GT-adaptive query
graph and then into five deterministic synthetic-current variants:

```text
2,000 independent source objects x 5 frozen variants = 10,000 samples

train:      1,600 objects x 5 = 8,000 samples
validation:   200 objects x 5 = 1,000 samples
test:         200 objects x 5 = 1,000 samples
```

Splitting is performed at object level, so variants of one object cannot cross
train, validation and test. The present benchmark covers the complete test
split: 200 independent objects and all five variants per object. Variant-level
statistics use all 1,000 inputs; the additional cluster bootstrap resamples the
200 objects and keeps their five variants together.

Before perturbation, each source GT graph is adaptively subdivided until it
meets the represented-vertex-area threshold of its GT-sub2 reference. For each
variant, a SHA-256-derived seed from object ID, base seed `7` and variant index
drives a Gaussian scalar field. The field is graph-smoothed for five iterations
with neighbor weight `0.65`, normalized and clipped at three standard
deviations, then applied along GT normals at scale `0.15h`. Local damping avoids
new flips and degeneracies while preserving connectivity. These currents are
prepared once and frozen; training-time query and local jitter are disabled.

The five variants of an object share the same 28 RGB observations and cameras.
The first 14 images are the upstream observations and views 15--28 are rendered
once from the clean object with the nested camera layout. Visibility is then
recomputed for each current graph; GT depth, visibility and correspondence are
not model inputs.

## Full 1,000-sample results

`CD gain` is the macro mean relative reduction from the common initial mesh.
Invalid Chamfer outputs are excluded only from that method's Chamfer mean and
paired denominator; they are not repaired or silently replaced.

| Method | Complete | Valid CD | Chamfer | CD gain | P2S p95 | F-score | Normal | Improved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Initial mesh | 1000/1000 | 1000/1000 | 0.00776417127 | +0.00% | 0.0272955594 | 0.764002732 | 0.924252350 | 0/1000 |
| **Formal mixed-loss Arm-B** | **1000/1000** | **1000/1000** | **0.00476456546** | **+38.63%** | **0.0146282911** | **0.881035649** | **0.908597358** | **975/1000** |
| Archived old-structure Ours | 1000/1000 | 1000/1000 | 0.00522954770 | +32.65% | 0.0163648574 | 0.857139334 | 0.895906909 | 959/1000 |
| NDS | 1000/1000 | 998/1000 | 0.00806632098 | -3.89% | 0.0263812819 | 0.769122488 | 0.780901076 | 543/1000 |
| nvdiffrec | 999/1000 | 999/1000 | 0.0131264340 | -69.03% | 0.0508395046 | 0.791414232 | 0.755936584 | 565/1000 |
| ExMesh | 1000/1000 | 996/1000 | 0.0162620513 | -109.45% | 0.0624136772 | 0.648993101 | 0.833134666 | 191/1000 |

The initial meshes have higher normal consistency than formal Arm-B, so the
formal row should not be described as improving every metric relative to the
input. The result is strongest on the declared surface-distance and F-score
objectives.

### Why Chamfer and P2S mean were identical

For equal-sized forward and reverse arrays `d_f` and `d_r`, the evaluator uses

```text
Chamfer = 0.5 * (mean(d_f) + mean(d_r))
P2S bidirectional mean = mean(concat(d_f, d_r))
```

The two expressions are mathematically identical because both arrays contain
3,000 samples. Their equality was not an empirical coincidence and does not
provide two independent confirmations. The raw CSV retains both compatibility
fields, but the report presents Chamfer once and retains only P2S p95 as the
separate tail-distance metric.

## Paired Chamfer comparisons

Differences are formal Arm-B minus comparator, so negative values favor formal
Arm-B. W/L/T is from formal Arm-B's perspective.

| Comparator | Valid pairs | Variant W/L/T | Mean CD difference |
|---|---:|---:|---:|
| Archived old-structure Ours | 1000/1000 | 882/118/0 | -0.000464982242 |
| NDS | 998/1000 | 804/194/0 | -0.00330183437 |
| nvdiffrec | 999/1000 | 829/170/0 | -0.00836142520 |
| ExMesh | 996/1000 | 974/22/0 | -0.0114867244 |

### Object-cluster robustness

The following calculation averages each object's valid variant differences,
then resamples objects 10,000 times with seed `7`. This is the appropriate
independence check because the five variants of an object share its source
shape, images and cameras.

| Comparator | Objects represented | Object W/L/T | Object-mean CD difference | 95% cluster-bootstrap CI |
|---|---:|---:|---:|---:|
| Archived old-structure Ours | 200 | 185/15/0 | -0.000464982242 | [-0.000580558, -0.000314545] |
| NDS | 200 | 159/41/0 | -0.00329943456 | [-0.004242780, -0.002472711] |
| nvdiffrec | 200 | 169/31/0 | -0.00835330952 | [-0.018176520, -0.002863842] |
| ExMesh | 200 | 196/4/0 | -0.0114636925 | [-0.013416080, -0.009649682] |

All four object-cluster intervals remain strictly negative. The strongest
within-family conclusion is therefore not merely an effect of treating five
correlated variants as 1,000 independent objects: formal Arm-B improves the
object mean on 185 of 200 held-out source objects.

## Other paired effects against the old structure

| Metric | Formal Arm-B minus old structure | Formal Arm-B W/L/T |
|---|---:|---:|
| P2S p95 | -0.00173656631 | 871/129/0 |
| F-score | +0.0238963151 | 838/147/15 |
| Normal consistency | +0.0126904491 | 877/123/0 |
| Introduced flipped faces | -61.56 | 691/300/9 |
| New degenerate faces | 0 | 0/0/1000 |

## Topology and compute

| Method | Connectivity preserved | Introduced flips | New degenerates | Runtime/mesh | Peak GPU memory |
|---|---:|---:|---:|---:|---:|
| Formal mixed-loss Arm-B | 1000/1000 | 472697 | 0 | 6.90847283 s | 3076.29644 MiB |
| Archived old-structure Ours | 1000/1000 | 534257 | 0 | 19.8725343 s | 3076.38944 MiB |
| NDS | 1000/1000 | 1584502 | 3041 | 26.8886510 s | 4601.19000 MiB |
| nvdiffrec | 999/999 | 3004124 | 0 | 459.585490 s | 2567.43243 MiB |
| ExMesh | 0/1000 | n/a | n/a | 268.512651 s | 2524.03200 MiB |

Formal Arm-B is about `2.88x` faster than the archived old-structure Ours row
under these recorded runs. Runtime includes the model forward and the declared
sparse recovery but excludes the common evaluator.

## Formal Arm-B protocol and evaluation correction

The selected model is the validation-best checkpoint at epoch `195`:

```text
checkpoint_best.pt
SHA-256: fa934cd44c4009dd392c415fe2c5f731c8cf1b78cda6a31fab199d4c15510b82
```

Training uses the Sofa50 Arm-B architecture with 28-view original-plus-HF image
features and the mixed objective
`raw-Laplacian Huber + 1e-2 recovered-vertex MSE`. Its formal standalone solve
is

```text
min_V ||L_U V - delta_B||^2 + 0.01 ||V - V_input||^2,
```

with the Uniform current-graph random-walk operator, all vertex equations, no
visibility gate, no confidence weighting and a float64 SciPy LSMR augmented
system. All 1,000 formal solves converged.

The first evaluation attempt (`17796`) incorrectly reused an H2-ablation helper
that required a confidence head, although the formal checkpoint has
`confidence.enabled=false`; its dependent finalizer (`17801`) was cancelled.
The runner was corrected to support explicit no-confidence inference and to
dispatch the exact formal all-vertex LSMR recovery. Smoke job `17805` passed
with the frozen checkpoint, the eight-shard evaluation `17806` completed all
1,000 samples, and finalizer `17807` completed successfully. No checkpoint,
recovery lambda, evaluator or test sample was changed in response to the failed
launch.

The old Ours, NDS, nvdiffrec and ExMesh rows were staged read-only from the
completed same-input archive; no external method was rerun. Exact source hashes
are recorded in [the frozen-comparator provenance](FROZEN_COMPARATOR_PROVENANCE.json).

## Failures and invalid external metrics

- NDS: `1000/1000` executions completed; `2` Chamfer outputs are invalid.
- nvdiffrec: `999/1000` completed; one mesh failed because surface sampling
  requires positive finite mesh area.
- ExMesh: `1000/1000` executions completed; `4` Chamfer outputs are invalid and
  topology is changed, so same-index flip/degeneracy counts are not comparable.

These failures explain `contract_audit=false` and `metric_completeness=false`.
They do not alter the all-valid `1000/1000` formal-versus-old paired comparison.

## Future2000 Arm-E preparation for frozen B+E fusion

Arm-E is configured as a separate direct-vertex specialist, reusing the
established Sofa50 Arm-E architecture, observation path and objective rather
than modifying Arm-B. It predicts a residual directly:

```text
delta_V = f_E(images, cameras, V_input, mesh_features)
V_E = V_input + delta_V
```

The training contract is from-scratch initialization, `200,000` optimizer
steps, four Blackwell GPUs, effective global batch `8`, mean same-index vertex
residual squared L2, and checkpoint selection by validation direct-vertex MSE
only. It has `826,115` parameters, performs no Laplacian target loss or analytic
recovery, and keeps the test split sealed. Its intended downstream role is the
frozen vertex anchor `V_E` in the same operator B+E fusion used for Sofa50.

The builder and Slurm launch script are complete, but job `17800` remains
`PENDING (Resources)` with zero elapsed runtime and no output files as of this
report snapshot. Consequently, Arm-E checkpoint selection, fusion-lambda
selection and final Future2000 B+E evaluation remain future work.

## Reproducibility sources

- [Formal Future2000 Arm-B config](../../configs/learned_laplacian/train_future2000_current28view_arm_b_mixed_loss_200k_4blackwell.json)
- [Future2000 dataset preparation](../../scripts/prepare_future2000_synthetic_current_28view.py)
- [Formal Ours evaluation runner](../../scripts/run_future2000_same_initial_ours.py)
- [Frozen-comparator staging](../../scripts/stage_future2000_frozen_comparators.py)
- [Comparison launch](../../scripts/HPC/evaluate_future2000_mixed_vs_old_external_8blackwell.slurm)
- [Finalizer](../../scripts/HPC/finalize_future2000_mixed_vs_old_external.slurm)
- [Report generator](../../scripts/generate_future2000_same_initial_subset_report.py)
- [Arm-E config builder](../../scripts/build_future2000_current_arm_e_config.py)
- [Arm-E training launch](../../scripts/HPC/train_future2000_current_arm_e_200k_4blackwell.slurm)

## Final conclusion

The completed result supports a stronger and correctly scoped claim: on 200
held-out Future2000 objects with five frozen current-mesh perturbations each,
the formal mixed-loss Arm-B consistently improves over the archived old
structure and all three frozen external baselines in surface-distance metrics.
The conclusion survives object-level clustering, while the report keeps the
normal-consistency trade-off against the initial mesh and all invalid external
outputs explicit. Future2000 Arm-E and B+E fusion are prepared but are not yet
results.
