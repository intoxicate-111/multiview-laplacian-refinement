# Sofa50 controlled same-initial-mesh benchmark

Primary claim scope: `same prepared synthetic mesh + same 28-view RGB/cameras -> different refinement methods`.

Contract audit: **true**. Completed methods are evaluated with the same deterministic 3,000-point surface protocol (seed 7).

## Group A aggregate

| Method | Complete | Mean initial CD | Mean final CD | CD improvement | Mean P2S | Normal | Improved | Worsened | Vertices | Faces | Runtime/sample | Peak GPU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| initial | 25/25 | 0.017070468 | 0.017070468 | 0 | 0.017070468 | 0.95519095 | 0 | 0 | 7060 | 13969 | 0 | 0 |
| ours | 25/25 | 0.017070468 | 0.0113478 | 33.52379 | 0.0113478 | 0.94451441 | 25 | 0 | 7060 | 13969 | 7.2063994 | 12053.083 |
| exmesh | 25/25 | 0.017070468 | 0.020170615 | -18.160877 | 0.020170615 | 0.84533666 | 8 | 17 | 49329.84 | 98557.56 | 762.40041 | 4776 |
| nds | 25/25 | 0.017070468 | 0.011204992 | 34.36037 | 0.011204992 | 0.87380513 | 22 | 3 | 7060 | 13969 | 227.30963 | 22285 |
| nvdiffrec | 25/25 | 0.017070468 | 0.01365466 | 20.010048 | 0.01365466 | 0.84812228 | 18 | 7 | 7060 | 13969 | 824.98201 | 4295 |

CD improvement is `(mean initial CD - mean final CD) / mean initial CD * 100%`; per-sample mean and median values are retained in `summary.json`.

## Input and method contract

- Dataset: `/networkhome/WMGDS/zhou_c/sofa_mesh/sofa50_synthetic_current_28view_native1920_v1/manifest.json` (25 canonical test samples).
- Common input: the exact existing prepared current OBJ, the same 28 native-1920 RGB images, and the same prepared cameras.
- GT is consumed only by the common evaluator.
- Group A: initial, ours, ExMesh, NDS, nvdiffrec.
- Group B: Neuralangelo (neural SDF/marching-cubes reconstruction) and MAtCha (point-map/Gaussian/chart reconstruction); neither officially refines an arbitrary supplied triangular mesh.
- ExMesh PGSR and NDS visual-hull initialization are bypassed. nvdiffrec uses its official fixed-topology DLMesh path with the supplied base mesh.

## Topology and metric limitations

Introduced flipped faces are reported only when output V/F and face ordering preserve the common connectivity. For a topology-changing output such as ExMesh this metric is unavailable rather than inferred from unrelated faces. The configured NDS and nvdiffrec adapters preserve the supplied connectivity. Runtime and memory are implementation/hardware measurements, not algorithm-independent complexity estimates. Ours peak memory is PyTorch peak allocated memory; external-method peak memory is process-tree GPU usage sampled through nvidia-smi, so the two columns are useful operational measurements but not byte-identical profiler definitions.

## Failures

No failed Group A runs.

The obsolete `ours_exmesh_initial_zero_shot = 0.616526 mm` result is excluded from every table in this benchmark and remains provenance-only.
