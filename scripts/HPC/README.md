# Sofa50 C1 + F2 ± oracle residual expert, 2000 steps

Files:
- `sofa50_c1_f2_expert_2000.slurm`: Slurm array job (`0=E0`, `1=E1`).
- `sofa50_c1_f2_expert_2000_analyze.slurm`: strict contract audit + paired analysis.
- `submit_sofa50_c1_f2_expert_2000.sh`: submits training array and dependent analysis.

Copy them into the repository:

```bash
mkdir -p hpc
cp sofa50_c1_f2_expert_2000.slurm hpc/
cp sofa50_c1_f2_expert_2000_analyze.slurm hpc/
cp submit_sofa50_c1_f2_expert_2000.sh hpc/
chmod +x hpc/submit_sofa50_c1_f2_expert_2000.sh
```

Typical submission:

```bash
cd ~/multiview-laplacian-refinement
PROJECT_ROOT="$PWD" \
MANIFEST="$HOME/sofa_mesh/sofa50_refinement/multiview_960/gt_query_manifest.json" \
CONDA_ENV=test \
bash hpc/submit_sofa50_c1_f2_expert_2000.sh
```

If the HPC requires a GPU partition/account/module, add only those cluster-specific
SBATCH/module lines to the two `.slurm` files. Do not change the experiment config.

Experiment contract:
- C1: image feature dim 32, graph hidden dim 128, 3 graph layers.
- F2: image encoder first stride 1, second stride 1.
- 2000 optimizer steps.
- seed 7.
- full-vertex training.
- exact GT-query validation.
- same canonical target/confidence/query contract.
- E0: no residual expert.
- E1: oracle clean-GT normalized-Laplacian top-10% gate, residual MLP 128->32->3.
- E0/E1 must differ only by `model.oracle_residual_expert` and the screening arm name.

The analyzer writes:
- `analysis/c1_f2_contract_audit.json`
- `analysis/summary.json`
- `analysis/REPORT.md`

Primary decision metrics:
- top10 endpoint improvement
- top1 endpoint improvement
- smooth90 degradation
- overall degradation
- global cosine change
- per-mesh top10 improvement count

## Future2000 GT-adaptive scale-up

The Future2000 training path uses 2,000 objects, five deterministic current-mesh
variants per object, 28 views, GT-adaptive subdivision and an object-level
`8000/1000/1000` sample split. The main arm predicts the raw current-graph
Laplacian; a dependent paired arm predicts direct vertex displacement.

Relevant entry points:

- `audit_future2000_gt_adaptive_2000mesh.slurm`: contract and split audit.
- `train_future2000_gt_adaptive_fast_io.slurm`: four-L40 node-local RGB staging
  and resumable DDP training.
- `smoke_future2000_gt_adaptive_2000mesh.slurm`: dependent displacement smoke.
- `evaluate_future2000_laplacian_vs_displacement_3gpu.slurm`: sharded learned
  comparison after both models are complete.

The 200,000-step launch overrides the development budget stored in the JSON
configuration. Job 15794 reached step 32,000 and then failed because a
persistent DataLoader worker exhausted the 51,200 file-descriptor limit; the
remaining ranks later hit the NCCL watchdog. Job 15795 resumes the preserved
checkpoint with:

```text
workers_per_rank=4
persistent_workers=false
multiprocessing_sharing_strategy=file_system
```

These settings preserve parallel image loading while recreating workers at
epoch boundaries. Job 15795 reached step 64,000, then failed because a worker
exhausted `/dev/shm`; its checkpoint remains resumable. The paired displacement
jobs were cancelled, and no final geometry result exists. External-method
comparison launches are local-only; see
`docs/FUTURE2000_LOCAL_COMPARISON_TASKS.md`.

## Sofa50 direct-raw controlled experiments

The current 28-view synthetic-current line uses direct raw current-graph
Laplacian output, Huber `delta=0.01`, seed 7, local jitter off and 20,000 global
optimizer steps. Available orchestration entry points are:

- `submit_sofa50_synthetic_current_28view_loss_ablation_3gpu.sh`: raw MSE arm
  followed by four-shard Huber/MSE evaluation and merge;
- `submit_sofa50_dynamic_residual_expert_from_scratch_4gpu.sh`: four-L40
  from-scratch dynamic residual expert and unified evaluation;
- `evaluate_sofa50_dynamic_gate_causal_ablation_4gpu.slurm` plus
  `merge_sofa50_dynamic_gate_causal_ablation.slurm`: no-retraining base,
  constant, shuffled and learned-gate interventions;
- `submit_sofa50_image_feature_ablation_2x2gpu.sh`: Gaussian-only and
  original-plus-HF arms, two L40 GPUs per arm;
- `submit_sofa50_hf1920_4gpu.sh`: native-1920 preparation, smoke, four-L40
  training, four-shard 960/1920 evaluation and report merge.

The native-1920 launcher preserves all 28 views but uses view chunks of four
and gradient checkpointing for activation memory. Its global batch is four,
whereas the completed 960 HF arm uses two; reports must retain this non-strict
ablation caveat. All long chains use `afterok` dependencies and refuse to
overwrite a completed report or shard.

## Sofa50 v2 recovery-aware A-E extension

The strong-smoothing v2 follow-up uses all-equation regularised sparse
integration and disables visibility, confidence, recovery Huber and Adam.

- `train_sofa50_recovery_aware_arm_a_2l40.slurm` and
  `train_sofa50_recovery_aware_arm_b_2l40.slurm` created A/B; both later resumed
  at epoch boundaries on eight Blackwell GPUs while preserving global batch 8.
- `train_sofa50_recovery_aware_lambda_extension_8blackwell.slurm` launches C/D
  with `lambda=1e-3/1e-4`, `beta=1e-2`, float64 PCG, tolerance `1e-4` and at
  most 2,048 iterations. The dtype/iteration change is the documented response
  to float32 PCG stagnation; the objective and lambda are unchanged.
- `train_sofa50_direct_vertex_arm_e_8blackwell.slurm` launches the direct
  residual baseline with exactly 826,115 parameters and no Laplacian/recovery
  path.
- Evaluation, merge and matched visualisation remain dependency-gated by the
  corresponding `evaluate_*`, `merge_*` and `render_*` Slurm entry points.

At the 2026-08-24 snapshot, Arm C job `17274` is at step 3,200/20,000 on eight
RTX PRO 6000 Blackwell GPUs with zero PCG failures and zero NaN/Inf. D `17275`,
E `17278` and downstream jobs remain dependency-queued. Do not report C/D/E as
completed or select a representation winner from this snapshot.
