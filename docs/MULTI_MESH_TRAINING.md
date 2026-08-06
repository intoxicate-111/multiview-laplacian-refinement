# Optimized Multi-Mesh Training Guide

[简体中文](MULTI_MESH_TRAINING.zh-CN.md) | [Project README](../README.md)

This guide documents the optimized training path for the existing CNN + graph
network. It does not change the model architecture and does not implement
sparse vertex-view patches.

## Current production dataset

The checked local production manifest is:

```text
/home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/thingi10k50/sample_50_960/prepared_manifest.json
```

Its contract is:

- 50 prepared meshes: 40 train, 5 validation, and 5 test;
- 14 views per mesh;
- 960 x 960 prepared images;
- `lazy_image_paths_v1` storage for every sample;
- variable mesh topology and size across samples.

The production config validates these split counts before creating a run:

```text
configs/learned_laplacian/train_multi_mesh_edge_normalized_50_960.json
```

## One-command launch

Run from any directory:

```bash
bash /home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/multiview-laplacian-refinement/scripts/train_thingi10k50_960_full.sh
```

The launcher:

1. activates the `test` Conda environment;
2. checks the manifest and JSON config;
3. requires a visible CUDA device;
4. refuses to overwrite a non-empty output directory;
5. starts training and tees the console to `console.log`.

The fixed output directory is:

```text
runs/learned_laplacian/thingi10k50_960_full
```

Validate the launch contract without training:

```bash
bash scripts/train_thingi10k50_960_full.sh --check
```

For a long foreground run, keep the terminal open or invoke the launcher from
`tmux`. The current trainer does not resume from an interrupted checkpoint;
the non-empty-output guard prevents accidental mixing of two runs.

## Production training policy

The 50-mesh 960 profile uses:

| Setting | Value |
|---|---:|
| Maximum epochs | 5,000 |
| Maximum optimizer steps | 50,000 |
| Gradient accumulation | 4 meshes |
| Optimizer steps per complete epoch | 10 |
| Validation interval | 5 epochs |
| Checkpoint interval | 10 epochs |
| Early-stopping patience | 15 validation events |
| Early-stopping minimum delta | 0.0001 |
| DataLoader workers | 4 |
| Prefetch factor | 2 |
| Pinned memory | enabled |
| Persistent workers | enabled |
| CUDA AMP | FP16 enabled |

Training stops when any configured terminal condition is reached: maximum
epochs, maximum optimizer steps, or early stopping. With validation every five
epochs, an early-stopping patience of 15 corresponds to 75 epochs without a
sufficient validation improvement. The ReduceLROnPlateau scheduler uses
validation events rather than raw epochs.

## Lazy data and precision path

The command-line entry point passes `PreparedMeshDataset` directly into the
trainer. It does not convert the dataset to `tuple` or `list`.

The data path is:

```text
prepared static mesh tensors
  -> lazy DataLoader worker
  -> decode current images as uint8
  -> pinned CPU memory
  -> non-blocking CUDA transfer
  -> FP32 conversion / 255 and configured normalization on GPU
  -> FP16 autocast CNN + graph-network forward
  -> FP32 Laplacian target, robust loss, and metrics
```

Only currently requested and prefetched images are decoded. Images are not
cached for all meshes, so dataset size does not directly multiply GPU memory.
Static graph/target tensors are still prepared once for all train and
validation meshes.

The configured image normalization is identity after `[0,1]` scaling:

```json
{
  "mean": [0.0, 0.0, 0.0],
  "std": [1.0, 1.0, 1.0]
}
```

This preserves the input semantics of the original training path. It is not
ImageNet normalization because the image encoder is trained from scratch.

## Available configs

| Config | Purpose |
|---|---|
| `train_multi_mesh_edge_normalized_50_960.json` | Full 40/5/5 production run |
| `train_multi_mesh_edge_normalized_960_epoch1.json` | 960-pixel one-epoch CUDA smoke test |
| `train_multi_mesh_edge_normalized_1920_epoch1.json` | 1920-pixel one-epoch CUDA smoke test |
| `train_multi_mesh_edge_normalized_1000_1920.json` | 800/100/100, 250-epoch, 50k-step profile |

The prepared sample records the actual image size. Config filenames document
the intended manifest profile, while the loader reads `prepared_image_size`
from each sample.

The 1,000-sample config rejects manifests that do not contain exactly 800
train, 100 validation, and 100 test entries. The test split is reserved for
held-out evaluation and is not used by the training loop.

## Data loading controls

Lazy samples are pruned before entering DataLoader workers. Forward fields,
camera tensors, confidence, local scale, and the selected training target are
retained. GT meshes, faces, target positions, duplicate raw/normalized targets,
`local_edge_scale`, and metadata stay out of worker IPC and GPU transfer. The
raw target and face count remain in the main process and are attached only for
validation and final prediction metrics.

Optional view sampling is configured under `data_loading`:

```json
{
  "train_views_per_sample": null,
  "validation_views_per_sample": null
}
```

`null` preserves the original all-view behavior. A positive integer selects
that many aligned image paths, intrinsics, extrinsics, and visibility rows.
Training selection changes deterministically by epoch and sample ID;
validation selection is fixed and reproducible. Values at least as large as
the available view count use all views.
The CLI can override these values with `--train-views-per-sample 4` and
`--validation-views-per-sample 4` without editing the source config.

`coarse_only` and `--zero-images` do not open or resize image files. The former
also omits camera tensors because image features and valid-view ratio are both
zero in that ablation. The latter retains camera projection so its historical
valid-view-ratio input remains unchanged.

With profiling enabled, each epoch records `sample_wait_seconds`, worker-side
`image_decode_resize_seconds`, `pin_or_transfer_seconds`,
`forward_backward_seconds`, mean selected views, and decoded uint8 image bytes.
Worker-to-main IPC time is not reported separately because it cannot be
isolated reliably from DataLoader waiting and prefetching.

## Outputs and monitoring

During training, each epoch prints:

```text
epoch, train loss, validation loss, best loss, learning rate
DataLoader wait, GPU transfer, forward/backward, total step, validation time
```

The run directory contains:

- `console.log`: live launcher output;
- `best.pt`: best validation checkpoint;
- `checkpoint_epoch_*.pt`: periodic checkpoints;
- `training_history.json`: epoch losses, learning rates, steps, and timing;
- `metrics.json`: final losses, stop reason, performance, and per-object metrics;
- `config.json`, `run_config.json`, and `dataset_manifest.json`: reproducibility metadata;
- `predictions/train/` and `predictions/validation/`: target-space and recovered raw predictions.

Follow a running job with:

```bash
tail -f runs/learned_laplacian/thingi10k50_960_full/console.log
```

`metrics.json` records at least:

- initial/static preparation time;
- mean DataLoader wait time;
- mean GPU transfer time;
- mean forward/backward time;
- mean total optimizer-step time;
- validation time;
- peak allocated GPU memory;
- peak main-process CPU memory;
- completed epochs, optimizer steps, AMP state, and stop reason.

The CPU peak is the main-process high-water mark, not a strict aggregate of
all persistent worker RSS values.

## Measured performance

Measurements used the same 40/5/5 dataset contract and a Quadro RTX 5000:

| Metric | Optimized 1920 | Optimized 960 |
|---|---:|---:|
| One training epoch | 10.85 s | 4.78 s |
| Validation pass | 2.24 s | 1.06 s |
| DataLoader wait | 5.45 s | 2.52 s |
| GPU transfer | 1.89 s | 0.54 s |
| Forward/backward | 3.17 s | 1.52 s |
| Peak GPU allocation | 3.01 GiB | 1.00 GiB |
| Peak main-process CPU memory | 5.06 GiB | 2.83 GiB |
| Complete one-epoch smoke runtime | 48.86 s | 33.94 s |

The optimized 960 steady-state training epoch was approximately 2.27 times
faster than optimized 1920. Compared with the original eager 1920 path, the
complete smoke runtime improved from 187.96 seconds to 48.86 seconds at 1920.

The one-epoch 960 and 1920 losses were almost identical, but one epoch is not
enough to establish equal final accuracy. Use a longer controlled A/B run
before treating 960 as accuracy-equivalent.

## Loss interpretation

The target uses edge-scale-normalized Laplacian coordinates and Huber loss with
`delta=0.01`. Normalized targets have a heavy-tailed magnitude distribution,
so loss values can move slowly even while checkpoints improve. On the current
960 dataset, the zero-prediction baselines were approximately 0.281828 train
and 0.305586 validation. The live production run reached a validation loss of
approximately 0.298575 by epoch 40, so it had moved materially below the zero
baseline.

Do not change target clipping, Huber delta, or target standardization during a
running experiment. Compare those choices in a new run with a new output
directory.

## Verification

Activate the same environment and run:

```bash
source /home/zhou_c_WMGDS.WMG.WARWICK.AC.UK/miniconda3/etc/profile.d/conda.sh
conda activate test
PYTHONPATH=src pytest -q
```

The optimized implementation was validated with 108 passing tests, including
lazy manifest loading, uint8 CPU images, persistent workers, max-step stopping,
early stopping, aligned epoch-aware view sampling, image-free ablations, CUDA
transfer paths, and finite CUDA AMP loss.

## Remaining constraints

- Training is one ragged mesh forward at a time with gradient accumulation,
  not packed-graph batching.
- PNG decoding remains the largest steady-state component at 960 pixels.
- Static mesh/graph preparation scales with the number and complexity of
  train/validation meshes, but runs once at startup.
- The trainer writes final metrics by evaluating train and validation again.
- Automatic checkpoint resume is not implemented.
- The 1,000-sample profile is configured, but it requires a real matching
  800/100/100 prepared manifest before launch.
