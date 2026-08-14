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

The active 200,000-step launch overrides the development budget stored in the
JSON configuration. Job 15794 reached step 32,000 and then failed because a
persistent DataLoader worker exhausted the 51,200 file-descriptor limit; the
remaining ranks later hit the NCCL watchdog. Job 15795 resumes the preserved
checkpoint with:

```text
workers_per_rank=4
persistent_workers=false
multiprocessing_sharing_strategy=file_system
```

These settings preserve parallel image loading while recreating workers at
epoch boundaries. The downstream jobs must use `afterok` dependencies on the
replacement training job. External-method comparison launches are local-only;
see `docs/FUTURE2000_LOCAL_COMPARISON_TASKS.md`.
