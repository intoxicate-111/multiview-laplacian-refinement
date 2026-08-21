# DTU scan24 intended prepared/current-mesh provenance audit

Audit date: 2026-08-20 (Europe/London)

Scope: provenance only. No learned-model inference, recovery, DTU evaluation, mesh generation, or HPC job submission was performed for this audit.

## Decision

| Question | Finding |
|---|---|
| Does a distinct learned-Laplacian-prepared DTU scan24 current mesh already exist in the searched local, HPC, snapshot, run, backup, or synchronization trees? | **No evidence of one.** |
| Was such a mesh generated but omitted from synchronization? | **No.** No generation command/job/output lineage exists to precede a missed sync. |
| Was it ever generated? | **No.** The audited timeline shows that the only scan24 current/initial mesh generation was the official PGSR stage. |
| Confidence | **High**, based on command and file lineage, not on the absence of a metadata label. |

The phrase “already-prepared meshes ... intended for ExMesh/DTU evaluation” first appears in the correction request saved at 2026-08-20 16:17:57. The original ExMesh suite request, saved at 2026-08-16 01:45:26, instead explicitly required the same ExMesh-generated initial mesh and said to use geometry produced by the ExMesh pipeline as the current/coarse input. The implementation and all archived job commands followed that original contract.

## Proven scan24 file lineage

The successful scan24 reproduction ran the following existing commands:

```bash
cd /networkhome/WMGDS/zhou_c/external_baselines/ExMesh/PGSR
/networkhome/WMGDS/zhou_c/miniconda3/envs/exmesh_official/bin/python train.py \
  -s ../workdir/DTU/scan24 \
  -m ../outputs/coarse_meshes/dtu_scan24/test \
  --quiet -r2 --ncc_scale 0.5
/networkhome/WMGDS/zhou_c/miniconda3/envs/exmesh_official/bin/python render.py \
  -m ../outputs/coarse_meshes/dtu_scan24/test \
  --quiet --num_cluster 1 --voxel_size 0.01 --max_depth 5.0 --use_depth_filter
cp ../outputs/coarse_meshes/dtu_scan24/test/mesh/tsdf_fusion_post.ply \
  ../workdir/DTU/scan24/mesh.ply
cp ../outputs/coarse_meshes/dtu_scan24/test/mesh/tsdf_fusion_post.ply \
  /networkhome/WMGDS/zhou_c/multiview-laplacian-refinement/runs/exmesh_baselines/exmesh_initial/scan24/meshes/final_mesh.ply
```

The three successful-run files are byte-identical:

| Role | Timestamp (BST) | Bytes | SHA-256 |
|---|---:|---:|---|
| PGSR `tsdf_fusion_post.ply` | 2026-08-16 03:01:07 | 1,529,861 | `642675e1122a4b0ba6369d219f742cc34ed0c21ee1645fa0c6e9437daae6c66a` |
| ExMesh scene `workdir/DTU/scan24/mesh.ply` | 2026-08-16 03:14:53 | 1,529,861 | same |
| archived `exmesh_initial/.../final_mesh.ply` | 2026-08-16 03:01:08 | 1,529,861 | same |

The old learned-HF run's own `status.json` records its absolute input as the scene `mesh.ply`; its graph was constructed from that mesh's faces and its query positions were that mesh's vertices. Therefore the old `ours = 0.616526 mm` row is conclusively an ExMesh-initial/PGSR zero-shot result, not a separate prepared-current result.

The misleading cancelled-job candidate is also closed by lineage:

| File | Bytes | SHA-256 |
|---|---:|---|
| `_cancelled/15910/current_meshes/scan24_mesh.ply` | 1,517,886 | `74c95e51eb018e3d1ad99549ca41e7a070b83eb651df3a14691cacd348a74bb1` |
| `_cancelled/15910/exmesh_initial/scan24/meshes/final_mesh.ply` | 1,517,886 | same |

Its archived `command.sh` explicitly copies that cancelled run's PGSR `tsdf_fusion_post.ply` to both paths. The different hash reflects a different PGSR run, not a different preparation method.

## Timeline and synchronization evidence

1. The 2026-08-16 original protocol required official PGSR/ExMesh initialization for methods accepting an initial mesh.
2. The first synchronized ExMesh runner and suite files, archived in `.codex_exmesh_sync` at 2026-08-16 02:23, already encode `scene/mesh.ply` as the fixed graph for `ours`; their older versions contain no removed intermediate preparation stage.
3. Slurm history contains ExMesh setup/materialization/reproduction jobs on 2026-08-16 and external-baseline/HF jobs on 2026-08-20, but no DTU learned-Laplacian mesh-preparation job.
4. `.codex_sync_stage`, `.codex_backups`, `.remote_backups`, shell history, project run trees, `data_prepare`, `sofa_mesh`, `48mesh_res`, project `mesh/` and `meshes/`, and recent mesh outputs contain no command/output pair for a distinct scan24 prepared mesh.
5. The local comparison snapshot contains copies of the official initial, official ExMesh output, external baselines, and the PGSR-input HF output. Its hashes agree with the HPC lineage; it contains no omitted prepared-current source.
6. The prepared-current Slurm runner was added only after the 2026-08-20 correction. It requires an externally supplied `PREPARED_MESH` and rejects geometry identical to the official initial; it is a consumer, not a generator, and has not been run.

These facts rule out “generated but not synchronized”: there is neither an earlier generator invocation nor an HPC-only output from such an invocation.

## Existing preparation pipelines and why none generates this mesh

There is **no existing DTU-specific command that generates a non-PGSR prepared/current mesh satisfying the corrected contract**.

The nearest existing project pipelines have different semantics:

| Pipeline | What it actually does | Why it cannot be used as the missing DTU generator |
|---|---|---|
| `data_prepare/scripts/prepare_sofa50_synthetic_current.py` | Creates five current meshes by smooth normal perturbation of Sofa50 **GT vertices/topology** and uses GT as `P_proxy`. | Dataset-specific and GT-derived; applying this to DTU evaluation GT would leak test ground truth. |
| `data_prepare/scripts/prepare_sofa50_synthetic_current_28view.py` | Attaches 28 Sofa50 observations to already-created fixed current variants. | Does not generate geometry. |
| `scripts/prepare_sofa50_synthetic_current_28view_1920.py` | Rerenders observations at native 1920 while preserving `vertices`, `faces`, targets, and visibility tensors. | Does not generate or alter the current mesh. |
| `data_prepare/scripts/prepare_sofa50_query_resolution_ablation.py` and `scripts/prepare_future2000_synthetic_current_28view.py` | Builds `GT-sub1`, `GT-sub2`, and area-adaptive graphs on the same piecewise-linear **GT surface**. | GT-derived and changes topology; it violates both no-test-GT and exact-connectivity requirements for the corrected inference. |
| `scripts/prepare_single_object_sample.py` | Packages a supplied coarse mesh, observations, and GT into a training sample. | Requires the coarse mesh to exist first; it is not a reconstruction/current-mesh generator. |
| `scripts/run_exmesh_hf_zero_shot.py` | Loads a supplied mesh, constructs its graph, normals/visibility, predicts, and recovers. | Inference consumer only; it does not create the input mesh. |

## Exact actionable conclusion

Under the corrected constraints, there is no honest existing command to run. The only exact existing scan24 mesh-generation command is the PGSR command shown above, and using its output directly is precisely the substitution now prohibited.

To create a valid new primary `ours` input, the protocol first needs an explicitly selected **non-GT, image-derived upstream coarse reconstruction** in the ExMesh scan24 frame. A DTU adapter would then preserve that reconstruction's vertices/faces and compute graph, normals, visibility, and recovery inputs from it. No such adapter/generator invocation exists in the audited history. Choosing NDS, nvdiffrec, Neuralangelo, or ExMesh output after the fact would define a new upstream method and must be declared as such; it would not recover a previously prepared mesh.

Therefore:

```text
provenance_status = never_generated
generated_but_unsynchronized = false
safe_existing_non_pgsr_generation_command = none
inference_run = not_started
```
