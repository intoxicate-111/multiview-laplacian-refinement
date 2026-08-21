# Chamfer evaluation incident: Sofa50 same-initial benchmark

[English](CHAMFER_EVALUATION_INCIDENT_2026-08-21.md) | [简体中文](CHAMFER_EVALUATION_INCIDENT_2026-08-21.zh-CN.md)

Status: resolved on 2026-08-21. The corrected 25-sample report has
`contract_audit: true`.

## Summary

The first Sofa50 same-initial comparison incorrectly placed method-native
Chamfer values in one cross-method table. Those values came from different
evaluation paths and were therefore not comparable. The error was exposed by
the common initial mesh: the learned-method path reported a mean initial
Chamfer of `0.003913228`, while the external-method path reported
`0.017070468`, even though the per-sample input mesh identity audit passed.

This was an evaluation/aggregation error, not evidence that one method received
a better initial mesh. No affected preliminary table is a valid scientific
result.

## Scope and impact

The incident affects the preliminary Sofa50 comparison among `ours`, NDS,
nvdiffrec and ExMesh. It does not change:

- the mesh outputs produced by any method;
- the 25 canonical test sample IDs;
- the supplied current/coarse mesh, RGB observations or cameras;
- the official DTU/ExMesh evaluator and its separately reported millimetre
  metrics;
- the Sofa50 learned-Laplacian prediction metrics.

It does invalidate any cross-method ranking made by mixing `native_*` Chamfer,
P2S or normal values before the unified re-evaluation.

## Detection and root cause

Every method was supposed to start from the same prepared current mesh. The
input audit checked the exact file lineage and per-sample mesh identity. A
shared initial mesh must have one shared score under one deterministic
evaluator, so the large initial-score discrepancy was a hard protocol
violation.

The aggregator had trusted each adapter's native metric fields. Those native
fields are useful provenance, but their sampling, GT loading and metric
semantics are not guaranteed to be identical. The aggregation layer did not
previously enforce a single evaluator over the archived meshes.

## Correction

`scripts/aggregate_sofa50_same_initial_benchmark.py` now loads the common
initial mesh and every completed output mesh, then recomputes all primary
geometry metrics through exactly one implementation:

```text
mlr.learned_laplacian.evaluation.evaluate_mesh_geometry
surface_samples = 3000
seed = 7
fscore_threshold = 0.01
```

The primary Chamfer is the mean of the two sampled-surface-to-triangle-surface
directions. With equal sample counts, the reported bidirectional P2S mean is
numerically the same quantity. Method-native measurements remain in the
per-sample files only as `native_*` provenance fields and are never used for
the primary ranking.

The corrected contract requires:

- all 25 expected samples to complete for every Group-A method;
- exact per-sample equality of the common initial metric across methods;
- `unified_metric_audit: true` on every completed row;
- a recorded `metric_protocol` on every completed row;
- explicit nulls where topology-dependent measurements are not comparable.

Any violation makes `contract_audit` false.

## Corrected result

All four methods completed `25/25`; the shared mean initial Chamfer is
`0.017070468`.

| Method | Mean final Chamfer ↓ | Aggregate improvement | Improved samples | Normal consistency ↑ |
|---|---:|---:|---:|---:|
| Ours | 0.011347800 | 33.52% | 25/25 | **0.944514** |
| NDS | **0.011204992** | **34.36%** | 22/25 | 0.873805 |
| nvdiffrec | 0.013654660 | 20.01% | 18/25 | 0.848122 |
| ExMesh | 0.020170615 | -18.16% | 8/25 | 0.845337 |

NDS has a `1.26%` lower mean Chamfer than ours in this run. Ours is the only
method that improves every sample and has substantially higher normal
consistency. ExMesh changes topology and worsens the aggregate Chamfer under
this supplied-initial synthetic protocol.

These values are specific to the Sofa50 same-initial protocol. They must not be
mixed with the official DTU ExMesh `overall` metric, the obsolete
`ours_exmesh_initial_zero_shot = 0.616526 mm` exploratory result, or earlier
learned-recovery native Chamfer tables.

## Preventive rules

1. Cross-method primary tables are generated only from archived meshes through
   one evaluator invocation.
2. The common initial score is treated as an evaluator checksum, not merely a
   descriptive field.
3. Native metrics are retained for debugging and provenance only.
4. Metric implementation, sample count, seed and threshold are serialized in
   `summary.json` and every per-sample row.
5. Official-benchmark metrics and project-internal synthetic metrics stay in
   separate tables with explicit units and semantics.

## Artifacts

- [Corrected final report](../reports/synthetic_same_initial_benchmark_20260820/full_report/FINAL_REPORT.md)
- [Summary JSON](../reports/synthetic_same_initial_benchmark_20260820/full_report/summary.json)
- [Per-sample CSV](../reports/synthetic_same_initial_benchmark_20260820/full_report/per_sample.csv)
- [Method/input contract audit](../reports/synthetic_same_initial_benchmark_20260820/METHOD_INPUT_CONTRACT_AUDIT.md)
