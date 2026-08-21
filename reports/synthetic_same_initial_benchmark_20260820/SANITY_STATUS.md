# Sofa50 same-initial benchmark sanity status

Updated: 2026-08-20 17:25 BST. This is an execution-status record, not the final benchmark report.

## Gate state

- Full 25-sample submission: **blocked by design** until all representative Group A runs pass `scripts/audit_sofa50_same_initial_sanity.py`.
- Coordinate/projection audit: passed.
- Common representative sample: `5ac05fe8-b550-4786-8e8c-c5a43c30d112__v01`.
- Common initial OBJ SHA-256: `7ec770f0c688b79de833ff24bb1c79cd8a6f2187be9801af2e222209ffbf13b6`.
- Common input counts: 8,476 vertices / 16,827 faces; 28 native-1920 RGB views.

| Method/stage | Job | State | Resource | Gate result |
|---|---:|---|---|---|
| ours sanity | 16567 | completed | 1 Blackwell GPU | input/artifact checks passed; final canonical run remains L40 |
| NDS sanity | 16560 | pending (resources) | 1 L40, short/1h | pending |
| nvdiffrec sanity | 16562 | pending (priority) | 1 L40, medium | pending |
| DA3 RGB prior for ExMesh | 16563 | completed in 86 s | 1 L40, short/1h | 28/28 priors; RGB/camera-only metadata passed |
| ExMesh sanity | 16564 | pending (priority) | 1 L40, medium | DA3 dependency fulfilled; pending |
| fail-closed coordinator | 16573 | pending (`afterany:16560:16562:16564`) | CPU only | will submit full arrays only after a passing gate |

The DA3 job was changed in place from `medium/4h` to `short/1h`; its job ID and the ExMesh dependency were preserved. It ran on `gpu-02` from 17:21:16 to 17:22:42 and produced 28 compressed priors under the pinned DA3 commit. NDS was subsequently moved in place to `short/1h` because its official single-sample budget is 2,000 iterations; ExMesh 10k and nvdiffrec 5k retain their 6-hour medium allocations. At this point `gpu-02` remained occupied by two unrelated video jobs and two p7m tasks, with further p7m array tasks queued.

The current machine-readable gate is false for exactly three reasons: ExMesh, NDS, and nvdiffrec status files do not exist yet. Every coordinate, common-SHA, ours identity, final-mesh, and ours recovery-artifact check already passes.

## Ours representative result

The successful Blackwell sanity run used the frozen 20,000-step canonical HF1920 checkpoint and the exact common mesh/RGB/camera tuple.

| Metric | Value |
|---|---:|
| Raw EPE | 0.00245001958 |
| Raw RMS | 0.00709436020 |
| Recovery-weighted raw RMS | 0.00618183695 |
| Bottom-90 EPE | 0.00164565156 |
| Top-10 EPE | 0.00968553760 |
| Top-1 EPE | 0.05161941343 |
| Initial Chamfer | 0.00362823423 |
| Refined Chamfer | 0.00355046913 |
| Initial P2S | 0.00374036017 |
| Refined P2S | 0.00358907050 |
| Initial normal consistency | 0.94845313 |
| Refined normal consistency | 0.93753566 |
| Introduced flipped faces | 250 |
| Runtime | 6.246 s |
| Peak allocated GPU memory | 12,076 MiB |

The Blackwell result is a contract/execution sanity only. Its prediction and recovered vertices differ slightly from the earlier L40 canonical evaluation, so the formal 25-sample `ours` arm is explicitly pinned to L40; Blackwell timing is not used for runtime fairness.

## Completed execution infrastructure

- The external runner now records the original common-OBJ path/SHA, image directory, camera/GT container, method config, final mesh, coordinate transform, and both source/exported identity audits per sample.
- ExMesh iterations are read from the pinned benchmark config rather than duplicated as a literal.
- A fail-closed sanity gate checks identical original SHA, numerical exported V/F identity, exact RGB path list, 28-view count, no forbidden GT input, method-specific initialization bypass, final mesh, and ours recovery artifacts.
- Full jobs are prewritten as 25 single-sample L40 array tasks, but the submission script cannot proceed unless `SANITY_GATE.json` says `full_benchmark_submission_allowed: true`.
- Aggregation retains all failures, topology changes, runtime/memory, per-sample rows, Group A/Group B separation, fixed-camera panels, and surface-error PLYs.
