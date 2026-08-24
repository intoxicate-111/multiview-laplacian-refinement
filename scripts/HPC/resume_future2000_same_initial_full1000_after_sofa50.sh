#!/bin/bash
set -euo pipefail

AFTER_JOB=${1:?usage: resume_future2000_same_initial_full1000_after_sofa50.sh SOFA50_FINAL_JOB_ID}
PROJECT=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
cd "${PROJECT}"

# Ours and the original one-view-per-iteration NDS arm already have complete
# 1000/1000 results.  The evaluator's --retry-failed contract preserves the
# 144 completed nvdiffrec samples and retries only the remaining failures.
NVDIFFREC=$(sbatch --parsable \
    --dependency=afterok:${AFTER_JOB} \
    --job-name=f2k1000_nvdiffrec_resume \
    scripts/HPC/run_future2000_same_initial_full1000_external_blackwell.slurm \
    nvdiffrec)

# Validate the full-28-view NDS optimizer-step contract before launching all
# 1000 samples.  The full array cannot start if this smoke test fails.
NDS28_SMOKE=$(sbatch --parsable \
    --dependency=afterok:${AFTER_JOB} \
    scripts/HPC/smoke_future2000_nds_28v_full_blackwell.slurm)
NDS28=$(sbatch --parsable \
    --dependency=afterok:${NDS28_SMOKE} \
    --job-name=f2k1000_nds28v \
    scripts/HPC/run_future2000_same_initial_full1000_external_blackwell.slurm \
    nds_28v_full)

# ExMesh requires DA3 priors.  Both stages remain on gpu-04 Blackwell.
DA3=$(sbatch --parsable \
    --dependency=afterok:${AFTER_JOB} \
    scripts/HPC/prepare_future2000_same_initial_full1000_da3_blackwell.slurm)
EXMESH=$(sbatch --parsable \
    --dependency=afterok:${DA3} \
    scripts/HPC/run_future2000_same_initial_full1000_external_blackwell.slurm \
    exmesh)

# Generate a report even if an external method exposes a genuine failure; the
# merge then records incomplete coverage instead of silently losing the run.
FINALIZE=$(sbatch --parsable \
    --dependency=afterany:${NVDIFFREC}:${NDS28}:${DA3}:${EXMESH} \
    scripts/HPC/finalize_future2000_same_initial_full1000_blackwell.slurm)

printf 'after_sofa50=%s\nnvdiffrec_resume=%s\nnds_28v_smoke=%s\nnds_28v_full=%s\nda3=%s\nexmesh=%s\nfinalize=%s\n' \
    "${AFTER_JOB}" "${NVDIFFREC}" "${NDS28_SMOKE}" "${NDS28}" \
    "${DA3}" "${EXMESH}" "${FINALIZE}"
