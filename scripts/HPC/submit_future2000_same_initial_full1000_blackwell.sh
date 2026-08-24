#!/bin/bash
set -euo pipefail

PROJECT=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
cd "${PROJECT}"

OURS=$(sbatch --parsable scripts/HPC/run_future2000_same_initial_full1000_ours_blackwell.slurm)
NDS=$(sbatch --parsable --dependency=afterok:${OURS} scripts/HPC/run_future2000_same_initial_full1000_external_blackwell.slurm nds)
NDS28=$(sbatch --parsable --dependency=afterok:${NDS} --job-name=f2k1000_nds28v scripts/HPC/run_future2000_same_initial_full1000_external_blackwell.slurm nds_28v_full)
NVDIFFREC=$(sbatch --parsable --dependency=afterok:${NDS28} scripts/HPC/run_future2000_same_initial_full1000_external_blackwell.slurm nvdiffrec)
DA3=$(sbatch --parsable --dependency=afterok:${NVDIFFREC} scripts/HPC/prepare_future2000_same_initial_full1000_da3_blackwell.slurm)
EXMESH=$(sbatch --parsable --dependency=afterok:${DA3} scripts/HPC/run_future2000_same_initial_full1000_external_blackwell.slurm exmesh)
FINALIZE=$(sbatch --parsable --dependency=afterany:${OURS}:${NDS}:${NDS28}:${NVDIFFREC}:${DA3}:${EXMESH} scripts/HPC/finalize_future2000_same_initial_full1000_blackwell.slurm)

printf 'ours=%s\nnds=%s\nnds_28v_full=%s\nnvdiffrec=%s\nda3=%s\nexmesh=%s\nfinalize=%s\n' \
    "${OURS}" "${NDS}" "${NDS28}" "${NVDIFFREC}" "${DA3}" "${EXMESH}" "${FINALIZE}"
