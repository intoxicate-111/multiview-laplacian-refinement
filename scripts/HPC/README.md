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

The Future2000 path uses 2,000 distinct 3D-FUTURE source objects, five frozen
deterministic current-mesh perturbation variants per object, 28 calibrated
960-pixel views and an object-level `8000/1000/1000` mesh split. Variants share
their source object's RGB/camera observations but retain variant-specific
geometry, connectivity, query graphs and visibility.

Relevant entry points:

- `audit_future2000_gt_adaptive_2000mesh.slurm`: contract and split audit.
- `train_future2000_current_arm_b_mixed_loss_200k_4blackwell.slurm`: formal
  current-architecture Arm B with
  `L_raw-Laplacian-Huber + 10^-2 L_recovered-vertex`.
- `smoke_future2000_mixed_eval_blackwell.slurm`: frozen-contract evaluation
  smoke test.
- `evaluate_future2000_mixed_vs_old_external_8blackwell.slurm`: full-1000
  formal/archived/external paired evaluation.
- `finalize_future2000_mixed_vs_old_external.slurm`: audited report finalizer.
- `train_future2000_current_arm_e_200k_4blackwell.slurm`: from-scratch
  direct-vertex Arm-E specialist.
- `train_future2000_current_arm_e_200k_2blackwell_gb8.slurm`: two-Blackwell
  launch of the same Arm-E specialist with per-rank accumulation four and
  effective global batch eight. A later epoch-boundary resume may use four or
  eight Blackwell GPUs with accumulation two or one, respectively, while
  retaining global batch eight.
- `evaluate_future2000_frozen_be_validation_8blackwell.slurm`: eight-shard
  validation-only frozen Arm-B/Arm-E fusion sweep.
- `select_future2000_frozen_be_lambda.slurm`: CPU audit/merge that locks lambda
  by validation mean Chamfer and writes `lambda_lock.json`.
- `evaluate_future2000_frozen_be_test_8blackwell.slurm`: locked-lambda test
  evaluation; it refuses to run without the audited validation lock.
- `finalize_future2000_frozen_be_report.slurm`: merges Arm-E/Hybrid results
  with the frozen Arm-B, old-structure, NDS, nvdiffrec and ExMesh rows and
  writes the comprehensive report.

Jobs `15794` and `15795` are retained as infrastructure history: the first
exhausted file descriptors and the second exhausted `/dev/shm`. Archived job
`16607` later completed the old-structure checkpoint and reached Chamfer
`0.00522955` with 959/1000 improvements, but it is not the formal
current-architecture result.

Formal evaluation jobs `17805`, `17806` and `17807` completed. The
validation-selected epoch-195 Arm-B checkpoint reaches Chamfer `0.00476457`,
P2S p95 `0.01462829`, F-score `0.88103565` and 975/1000 improvements. It wins
882/1000 meshes and 185/200 object means against the archived predictor; the
object-bootstrap CI excludes zero. External paired wins are 804/998 versus
NDS, 829/999 versus nvdiffrec and 974/996 versus ExMesh, with invalid outputs
kept explicit.

On 2026-09-01, the never-started four-GPU Arm-E job `17800` was cancelled and
replaced by job `17883` on two RTX PRO 6000 Blackwell GPUs. The run was later
resumed at an epoch boundary as four-GPU job `17888`, using per-rank
accumulation two and preserving effective global batch eight. Job `17888`
completed all 200,000 steps on 2026-09-04 with exit `0:0`; validation selected
epoch 160 (checkpoint SHA-256
`5a6aaa32bec6edcdd2c30face02c4ae8bc139fef18d4d05b3394c987057cb50f`).

The frozen B+E comparison completed its validation stage as `18673` validation
sweep → `18677` lambda lock. The lock selected `lambda=0.1` at validation mean
CD `0.00295644415` without test access. Test shards 0–3 completed under `18678`;
after the remaining tasks stalled at the dynamically reduced array throttle,
only shards 4–7 were resubmitted as `18780` with `ArrayTaskThrottle=4`. Report
job `18679` now depends on `18780_*`. This recovery preserves the completed
shards, the one frozen test opening and the maximum four-GPU allocation. Do not
claim aggregate Future2000 Arm-E or B+E test results before the chain completes.
Chamfer equals the
bidirectional P2S mean in the current evaluator because both directions use
equal 3,000-sample sets; retain P2S p95 but do not present P2S mean as
independent evidence. See the
[formal report](../../reports/future2000_mixed_vs_old_external_20260831_v2/FINAL_REPORT.md)
and the [local reproduction guide](../../docs/FUTURE2000_LOCAL_COMPARISON_TASKS.md).

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

Arms C/D/E and their matched evaluation are complete. Weakening the recovery
anchor in C/D worsens geometry; direct-vertex E reaches Chamfer `0.00334039`,
and validation-selected frozen B+E reaches `0.00302983`. The later scalar
vertex blend reaches `0.00318814`, so it is a strong control but does not
explain the operator hybrid. Current consolidated results are in the
[recent Sofa50 report](../../reports/sofa50_multitopology_rawlap500_v2/recent_ablation_and_old_domain_comparison_v1/REPORT.md).
