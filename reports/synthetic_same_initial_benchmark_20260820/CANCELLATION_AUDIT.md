# Obsolete ExMesh/DTU Slurm cancellation audit

Audit date: 2026-08-20 (Europe/London)

The active queue was inspected with `squeue -u zhou_c`; every listed job was then inspected with `scontrol show job -dd` before cancellation. Only the job whose command, working directory, and log paths tied it to the obsolete ExMesh/DTU scan24 protocol was cancelled.

| Job ID | Job name | State before | Classification evidence | Result |
|---:|---|---|---|---|
| 16491 | `exmesh_nvdr_s24` | RUNNING | Command: `scripts/HPC/run_exmesh_nvdiffrec_sanity.slurm`; workdir: this learned-Laplacian repository; logs: `slurm_logs/exmesh_nvdr_s24_16491.*`; DTU scan24/old ExMesh baseline sanity run | `CANCELLED by 1922630442`; ended 2026-08-20 16:47:32 BST |

## Explicitly retained unrelated jobs

| Job IDs | Names | Reason retained |
|---|---|---|
| 16494, 16495 | `uav_l40_night`, `uav_l40_day` | Independent `video_sr_hpc/slurm_uav.sh` jobs outside this project. |
| 16496, 16497 | `star_l40_night`, `star_l40_day` | Independent `video_sr_hpc/slurm_star.sh` jobs outside this project; pending on a failed unrelated dependency. |

Post-cancellation `squeue` contains only the four unrelated video-super-resolution jobs above. No active/pending/requeued/array job belonging to the obsolete ExMesh/DTU protocol remains.

No previous result, log, mesh, snapshot, or report was deleted.
