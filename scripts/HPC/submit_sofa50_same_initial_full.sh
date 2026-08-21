#!/bin/bash
set -euo pipefail

PROJECT=/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement
ROOT=${PROJECT}/runs/benchmarks/sofa50_same_initial_20260820
PYTHON=/networkhome/WMGDS/zhou_c/miniconda3/envs/test/bin/python

cd "${PROJECT}"
PYTHONPATH=src "${PYTHON}" scripts/audit_sofa50_same_initial_sanity.py \
    --manifest "${ROOT}/benchmark_manifest.json" \
    --coordinate-audit "${ROOT}/coordinate_audit/coordinate_audit.json" \
    --sanity-root "${ROOT}/sanity" \
    --output "${ROOT}/SANITY_GATE.json"

OURS_JOB=$(sbatch --parsable --array=0-24%2 scripts/HPC/run_sofa50_same_initial_ours_full.slurm)
OURS_JOB=${OURS_JOB%%;*}
NDS_JOB=$(sbatch --parsable --array=0-24%2 scripts/HPC/run_sofa50_same_initial_external_full.slurm nds)
NDS_JOB=${NDS_JOB%%;*}
NVDIFFREC_JOB=$(sbatch --parsable --array=0-24%2 scripts/HPC/run_sofa50_same_initial_external_full.slurm nvdiffrec)
NVDIFFREC_JOB=${NVDIFFREC_JOB%%;*}
DA3_JOB=$(sbatch --parsable --array=0-4%2 scripts/HPC/prepare_sofa50_same_initial_da3_full.slurm)
DA3_JOB=${DA3_JOB%%;*}
EXMESH_JOB=$(sbatch --parsable --dependency=afterok:${DA3_JOB} --array=0-24%2 scripts/HPC/run_sofa50_same_initial_external_full.slurm exmesh)
EXMESH_JOB=${EXMESH_JOB%%;*}
FINALIZE_JOB=$(sbatch --parsable --dependency=afterany:${OURS_JOB}:${NDS_JOB}:${NVDIFFREC_JOB}:${DA3_JOB}:${EXMESH_JOB} scripts/HPC/finalize_sofa50_same_initial_benchmark.slurm)
FINALIZE_JOB=${FINALIZE_JOB%%;*}

mkdir -p "${ROOT}"
"${PYTHON}" -c 'import json,sys; p=sys.argv[1]; d={"ours":sys.argv[2],"nds":sys.argv[3],"nvdiffrec":sys.argv[4],"da3":sys.argv[5],"exmesh":sys.argv[6],"finalize":sys.argv[7]}; open(p,"w").write(json.dumps(d,indent=2)+"\n")' \
    "${ROOT}/FULL_SUBMISSION.json" "${OURS_JOB}" "${NDS_JOB}" "${NVDIFFREC_JOB}" "${DA3_JOB}" "${EXMESH_JOB}" "${FINALIZE_JOB}"
printf 'ours=%s\nnds=%s\nnvdiffrec=%s\nda3=%s\nexmesh=%s\nfinalize=%s\n' "${OURS_JOB}" "${NDS_JOB}" "${NVDIFFREC_JOB}" "${DA3_JOB}" "${EXMESH_JOB}" "${FINALIZE_JOB}"
