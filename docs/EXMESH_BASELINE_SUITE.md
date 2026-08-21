# ExMesh-protocol external benchmark

This benchmark is intentionally independent from the project's synthetic-data
experiments. Its source of truth is the official ExMesh DTU release: official
RGBA observations (whose alpha channel is the released training mask), cameras,
normalization, PGSR initialization, DA3
priors, DTU ground truth, and released evaluator.

## Pinned implementations

| Method | Venue | Official repository | Commit |
|---|---:|---|---|
| ExMesh | CVPR 2026 | `Fan-Treasure/ExMesh` | `09950d283fc5372a09079e30c88d998f1c40b2d0` |
| Neural Deferred Shading | CVPR 2022 | `fraunhoferhhi/neural-deferred-shading` | `760e4549f59adaed9adf1bd705599786a00ba6b8` |
| nvdiffrec | CVPR 2022 | `NVlabs/nvdiffrec` | `abf3a34b1eb6e782abffefc2462c7e9bcd89f9bb` |
| Neuralangelo | CVPR 2023 | `NVlabs/neuralangelo` | `94390b64683c067c620d9e075224ccfe582647d0` |
| MAtCha Gaussians | CVPR 2025 | `Anttwo/MAtCha` | `b119fd96e484fc81eb40623c1ea92ad3dbd3c21e` |

The executable contract and paper reference values are pinned in
`configs/baselines/exmesh_official_suite.json`.

## Current gate status

The 15-scene released-code reproduction has passed: reproduced mean official
CD is 0.60484 mm versus the paper's 0.58 mm, with 0.02484 mm absolute mean
difference and 0.02194 mm per-scene MAE. The scan-24 sanity gate is still in
progress, so the full 15-scene comparison remains locked. Neural Deferred
Shading has completed scan 24 at 6.97350 mm official CD (51,740 vertices,
103,480 faces, 579 seconds, 45,087 MiB peak GPU memory). This is retained as a
valid result, not replaced by a simplified approximation.

This official DTU suite is separate from the completed Sofa50 same-initial
synthetic benchmark. In that benchmark, ours, ExMesh, NDS and nvdiffrec all
completed 25/25 from the exact same supplied current mesh, RGB observations and
cameras. Its first aggregate incorrectly mixed method-native Chamfer values.
The corrected aggregate re-evaluates every archived initial/final mesh through
one deterministic project evaluator and has `contract_audit: true`. See the
[bilingual Chamfer incident report](CHAMFER_EVALUATION_INCIDENT_2026-08-21.md)
and [corrected benchmark artifact](../reports/synthetic_same_initial_benchmark_20260820/full_report/FINAL_REPORT.md).

The corrected Sofa50 final Chamfer values are ours `0.011347800`, NDS
`0.011204992`, nvdiffrec `0.013654660` and ExMesh `0.020170615`, from a shared
initial `0.017070468`. These unitless normalized-scene values must not be mixed
with the official DTU millimetre `overall` metric documented below.

The separate DTU scan-24 learned-method input audit found that the intended
prepared/current mesh was never generated. Its cancelled-job placeholder was a
copy of the ExMesh PGSR output and is rejected. The file lineage, searched
locations and exact preparation entry point are recorded in the
[DTU scan-24 provenance report](../reports/DTU_SCAN24_PREPARED_CURRENT_PROVENANCE.md).

## Mandatory execution gates

The order is enforced scientifically, even though setup code can be prepared
in advance:

1. Reproduce the released ExMesh pipeline on all 15 official DTU scenes.
2. Compare the official evaluator's per-scene `overall` values with ExMesh
   Table 1. The configured gate requires all scenes, an absolute mean
   difference no larger than 0.10 mm, and a per-scene MAE no larger than
   0.15 mm.
3. Extract `runs/exmesh_baselines/common_contract.json` from the materialized
   official data and PGSR initial meshes.
4. Run all six reconstruction methods on scan 24 and pass a fixed-camera frame,
   handedness, scale, mask, export, evaluator, and GT-leakage audit.
5. Only then launch the full external benchmark.

The suite never silently replaces a failed official method with an in-house
approximation. A missing or failed per-scene `status.json` becomes an explicit
failed row in the aggregate outputs.

## Released-code reproduction

Download the exact data linked by ExMesh and build a separate Python 3.11,
CUDA 12.1 environment:

```bash
sbatch scripts/HPC/download_exmesh_official_data.slurm
sbatch scripts/HPC/setup_exmesh_official_env.slurm
```

After the downloaded archives have been audited and organized into the
official `workdir/DTU` layout, submit the 15-scene reproduction:

```bash
bash scripts/HPC/submit_exmesh_official_reproduction.sh
```

Each scene uses the bundled PGSR 5000-step initialization and the released
ExMesh 10000-step optimization with the scene-specific smoothness/depth weights
from `scripts/run_dtu.py`. Configs, commands, logs, initial/final meshes,
runtime, peak GPU-memory samples, environment metadata, source commit, and
official evaluation JSON are retained under
`runs/exmesh_baselines/exmesh_official/`.

There is one upstream protocol discrepancy that must remain visible: Section
4.1 of the paper describes a 7000-step 2DGS initialization and 256³ TSDF
extraction, while the released README and bundled runner specify 5000-step
PGSR. This suite first reproduces the released repository and does not hide the
difference.

The linked 2DGS archive contains 1554×1162 RGBA observations. The released
ExMesh loader defaults to `resolution=-1` and only auto-resizes images wider
than 1600 pixels, so the ExMesh stage consumes 1554×1162 while its PGSR command
uses `-r2`. The paper states 800×600. The suite follows the released command and
records native and actual execution resolutions in the common contract rather
than silently claiming the paper resolution.

The tarball also contains 485 macOS AppleDouble files named `._NNN.png` inside
mask directories. They are filesystem metadata rather than images, but the
released culling evaluator blindly globs every `*.png` and OpenCV cannot decode
them. Materialization removes only these sidecars from the extracted working
copy. The downloaded archives remain untouched, all five archive SHA-256 values
are retained, and `data_audit.json` records the exact cleanup count.

The released evaluation wrapper invokes its bundled `eval.py` through the bare
command `python`. Batch jobs therefore prepend the pinned ExMesh environment to
`PATH`; no evaluator source or metric definition is changed. The runner also
checks for `results.json` immediately after each wrapper call because the
upstream wrapper does not propagate its `os.system` child exit code.

## Common contract extraction

After all initial meshes exist:

```bash
PYTHONPATH=src python scripts/extract_exmesh_common_contract.py \
  --config configs/baselines/exmesh_official_suite.json \
  --dtu-root /networkhome/WMGDS/zhou_c/external_baselines/ExMesh/workdir/DTU \
  --output runs/exmesh_baselines/common_contract.json
```

The contract records every RGBA path and its alpha-mask semantics, image
resolution, raw ExMesh
`world_mat_i` and `scale_mat_i`, decomposed intrinsics/extrinsics, normalization
transform, DA3 prior paths, initial-mesh hash, GT point cloud, observation mask,
ground plane, and exact official evaluator settings. The audit rejects legacy
dataset/reconstruction paths and GT geometry as a method input.

The archive's separate 1600×1200 `mask/` PNGs are not consumed by the released
ExMesh training loader; that loader reads the 1554×1162 RGBA alpha channel.
The separate mask directory remains part of the released DTU culling/evaluation
pipeline. External method adapters therefore preserve the RGBA files
byte-for-byte and use alpha where the official method supports masks.

The NDS adapter writes only OpenCV K/R/t sidecars and uses the official `vh32`
visual-hull initialization in the normalized scene box. The nvdiffrec adapter
retains exact per-view fx/fy/cx/cy through a data-loader overlay because its
stock NeRF loader supports only a centered single-FOV camera; DMTet, material,
lighting, and loss optimization remain official. Neuralangelo's native DTU
transforms adapter is also audited; its official loader consumes RGB and ignores
alpha, which is recorded as a method capability difference.

## Metric semantics

The released ExMesh evaluator reports:

- `mean_d2s`: reconstructed surface samples to DTU STL points (accuracy);
- `mean_s2d`: DTU STL points to reconstructed surface samples (completeness);
- `overall`: their arithmetic mean, reported as CD in the ExMesh paper.

It does not implement normal consistency, F-score, or a triangle
point-to-surface metric. Those requested aggregate columns remain null unless a
separately labeled official implementation is added. They are never fabricated
or relabeled. The released evaluator also calls `default_rng()` without a seed
before radius downsampling; exact seed control is therefore unavailable without
changing official evaluation code. This limitation is recorded in the contract
and report.

Generate the auditable aggregate at any point with:

```bash
PYTHONPATH=src python scripts/aggregate_exmesh_baselines.py \
  --config configs/baselines/exmesh_official_suite.json \
  --output-root runs/exmesh_baselines
```

This writes `summary.csv`, `summary.json`, and `BASELINE_REPORT.md` and states
whether each gate has passed.

## Learned-method training blocker

ExMesh is a per-scene optimization protocol and supplies no non-evaluation
supervised split for learning a current-graph raw Laplacian predictor. The 15
DTU evaluation scenes and their GT point clouds cannot be used to tune or train
the learned method. A valid non-evaluation training-source contract must be
chosen before the learned adapter can be scientifically finalized. Reusing a
checkpoint trained on a prohibited data source, training on the DTU evaluation
GT, or distilling the test-scene ExMesh result is not done implicitly.
