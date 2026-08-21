#!/bin/bash
set -euo pipefail

PROJECT="/networkhome/WMGDS/zhou_c/multiview-laplacian-refinement"
OUTPUT="${PROJECT}/runs/exmesh_baselines/exmesh_official"

test -f "${OUTPUT}/environment.json"
test -d "/networkhome/WMGDS/zhou_c/external_baselines/ExMesh/workdir/DTU/scan24/mono_priors/da3"

JOB_ID="$(sbatch --parsable "${PROJECT}/scripts/HPC/run_exmesh_official_scene.slurm")"
echo "submitted official ExMesh 15-scene reproduction: ${JOB_ID}"
