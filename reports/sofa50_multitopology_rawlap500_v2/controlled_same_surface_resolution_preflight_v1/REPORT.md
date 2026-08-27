# Sofa50 controlled same-surface resolution and frozen Hybrid gain

Contract audit: **true**. Classification: **SINGLE_SHAPE_INCONCLUSIVE_PREFLIGHT**.

Each object uses four nested meshes obtained only by midpoint edge splits. The clean PL surface and perturbed initial PL surface are therefore exactly unchanged; RGB/cameras, frozen B/E checkpoints, visibility definition, and `lambda=3e-2` are fixed. No training or HPC queue job was used.

Primary gain is `E CD - Hybrid CD`; positive values favor the differential branch.

![Controlled resolution curve](controlled_resolution_gain.png)

## Per-level results

| Shape | Level | N | Faces | Median h | Initial CD | E CD | Hybrid CD | E-H gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 038b62bb | 0 | 5716 | 11242 | 0.03347203 | 0.004310649 | 0.003678825 | 0.004100866 | -0.000422041 |
| 038b62bb | 1 | 11432 | 22646 | 0.03340223 | 0.004233787 | 0.003857238 | 0.003832701 | 0.000024537 |
| 038b62bb | 2 | 22864 | 45487 | 0.02933573 | 0.004307056 | 0.003882770 | 0.003962606 | -0.000079836 |
| 038b62bb | 3 | 40012 | 79734 | 0.02501068 | 0.004308925 | 0.004075149 | 0.004049338 | 0.000025811 |

## Within-shape trend

| Shape | Spearman(h, gain) | Coarse gain | Fine gain | Fine-coarse | Monotonic |
|---|---:|---:|---:|---:|---:|
| 038b62bb | -0.8000 | -0.000422041 | 0.000025811 | 0.000447853 | False |

Macro slope of gain versus `log(h)`: `-0.0007894524`. A shape-bootstrap interval is undefined for this one-shape preflight, and the four gains are nonmonotonic; this artifact validates the protocol only and is not evidence for a resolution effect.

## Contract and numerical audit

- Objects: `1`; meshes: `4`.
- Maximum relative clean-surface area range within shape: `2.093e-16`.
- Maximum relative initial-surface area range within shape: `2.364e-16`.
- Maximum initial-CD relative range within shape: `1.792e-02` (remaining variation is evaluator sampling over identical surfaces).
- Maximum level-0 recomputed/prepared visibility disagreement: `6.248e-06`.
- Maximum base CPU/archive prediction relative RMS: B `0.5142%`, E `1.4304%`.
- Recovery: frozen B raw field + frozen E direct anchor, float64 PCG, `lambda=0.03`, `tol=1e-4`, maximum 2048 iterations.
- Metric protocol: `mlr.learned_laplacian.evaluation.evaluate_mesh_geometry;area_weighted_triangle_surface_sampling;bidirectional_sampled_surface_to_exact_triangle_surface;surface_samples=3000;seed=7;fscore_threshold=0.01;alignment=shared_prepared_coordinate_frame_no_ICP`.

The experiment controls discretization within each shape, but it still tests frozen predictors trained on the original mixed-resolution distribution. It identifies an inference-time discretization response, not a universal convergence theorem.
