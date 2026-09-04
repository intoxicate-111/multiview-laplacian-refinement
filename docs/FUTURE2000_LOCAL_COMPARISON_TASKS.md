# Future2000 local comparison tasks

These scripts run the comparison and reporting pipeline directly on the local
machine. They contain no Slurm submission commands.

Status update: 2026-09-04 12:07 BST. This local launcher is retained for
reproducibility but is not the source of the formal comparison. The frozen
full-1000 report now evaluates the formal mixed-loss current Arm B, the
archived old-structure predictor and the external methods. Formal Arm B reaches
Chamfer `0.00476456546`, improves 975/1000 meshes and beats the archived
predictor on 882/1000 meshes and 185/200 object means. Its valid paired wins
are 804/998 versus NDS, 829/999 versus nvdiffrec and 974/996 versus ExMesh;
invalid external outputs remain explicit. Do not replace the
[formal frozen report](../reports/future2000_mixed_vs_old_external_20260831_v2/FINAL_REPORT.md)
with the archived `0.00522955` result, the incomplete step-64k checkpoint or
outputs from this local workflow.

The older job 15791 and the local workflow remain historical diagnostic paths.
Their partial outputs must not be merged with the completed full-1000 report.

## Before running

The formal Arm-B/external comparison remains independently reproducible. The
replacement Future2000 direct-vertex Arm-E job `17888` has now completed all
200,000 steps, and validation selected epoch 160 (checkpoint SHA-256
`5a6aaa32bec6edcdd2c30face02c4ae8bc139fef18d4d05b3394c987057cb50f`).
The new frozen B+E evaluation remains separate and uses an HPC-only sealed
sequence. Job `18673` completed the validation sweep, and `18677` locked
`lambda=0.1` at validation mean CD `0.00295644415` without test access. Test
shards 0–3 completed under `18678`; only unfinished shards 4–7 were resubmitted
as four-GPU-capped array `18780` after the original pending tasks stalled at a
dynamically reduced array throttle. The replacement was pending for `Priority`
at the status snapshot. Job `18679` now depends on `18780_*` and
will write the comprehensive baseline report after successful completion. No
aggregate Arm-E/B+E test metric is claimed yet. To reproduce the existing Arm-B comparison,
download only the test data and completed frozen artifacts:

```bash
bash scripts/local/sync_future2000_comparison_inputs.sh all
```

The local bundle contains approximately 1.47 GB of prepared test tensors and
642 MB of shared rendered RGB observations. Training samples are not copied.

Install each external method in an isolated environment after a CUDA-capable
GPU and driver are available:

```bash
bash scripts/local/setup_future2000_comparison_envs.sh all
```

The setup pins the same official commits as
`configs/baselines/future2000_external_baselines.json`. OpenMVS is built with
hermetic vcpkg FFmpeg; NeRF2Mesh installs a pkg_resources-compatible
setuptools; ExMesh installs CUDA runtime headers and Ninja.

OpenMVS RefineMesh is retained here only as an external low-quality-input
baseline/stress arm. Its mesh must never become a training target, pseudo-GT,
checkpoint-selection endpoint or desired output topology. Reports must show its
initial quality and must not use its result alone to rank or scale the learned
method; see [the OpenMVS input policy](OPENMVS_INPUT_POLICY.md).

## Run tasks

List the available local tasks:

```bash
bash scripts/local/run_future2000_comparisons.sh list
```

Select local GPUs with a comma-separated list. The default is GPU 0 and one
shard. For three local GPUs:

```bash
F2K_GPUS=0,1,2 bash scripts/local/run_future2000_comparisons.sh preflight
F2K_GPUS=0,1,2 bash scripts/local/run_future2000_comparisons.sh all
```

`all` runs this strict sequence locally:

1. learned Laplacian versus direct displacement;
2. OpenMVS RefineMesh (diagnostic external stress arm only);
3. NDS;
4. NeRF2Mesh;
5. RGB-only DA3 priors and ExMesh;
6. four qualitative comparison figures and the unified report.

Every GPU shard writes a separate log under
`.external/future2000_local/logs`. Completed external-method samples and DA3
priors are reused when a local run is resumed.

The final report must distinguish infrastructure/sample failures from valid
method outputs and include a denominator for every aggregate. A partial shard
or a run with missing methods must remain explicitly incomplete. Because the
evaluator uses equal 3,000-sample forward and reverse sets, Chamfer is exactly
the bidirectional P2S mean; report P2S p95 as the distinct tail statistic and
do not count P2S mean as independent evidence.
