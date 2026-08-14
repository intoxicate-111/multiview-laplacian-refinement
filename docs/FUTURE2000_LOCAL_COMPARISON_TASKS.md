# Future2000 local comparison tasks

These scripts run the comparison and reporting pipeline directly on the local
machine. They contain no Slurm submission commands.

Status snapshot: 2026-08-14 09:49 BST. The learned raw-Laplacian model is still
training on HPC as job 15795 from the intact step-32k checkpoint. The paired
direct-displacement run waits on that job. Do not start the learned comparison
until both checkpoints are complete.

An older external-method diagnostic array, job 15791, was already running on
HPC when the workflow was moved local. Its partial outputs are not final and
its high sample-level failure count must not be merged into the local report.
Do not submit another external comparison through Slurm; this document is the
supported launch path for new comparison work.

## Before running

Wait until the learned-Laplacian and direct-displacement training runs have
finished, then download only the test data and completed run artifacts:

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
2. OpenMVS RefineMesh;
3. NDS;
4. NeRF2Mesh;
5. RGB-only DA3 priors and ExMesh;
6. four qualitative comparison figures and the unified report.

Every GPU shard writes a separate log under
`.external/future2000_local/logs`. Completed external-method samples and DA3
priors are reused when a local run is resumed.

The final report must distinguish infrastructure/sample failures from valid
method outputs and include a denominator for every aggregate. A partial shard
or a run with missing methods must remain explicitly incomplete.
