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
