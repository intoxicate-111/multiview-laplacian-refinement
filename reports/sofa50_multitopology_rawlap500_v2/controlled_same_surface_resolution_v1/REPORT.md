# Sofa50 controlled same-surface resolution and frozen Hybrid gain

Contract audit: **true**. Classification: **NO_RELIABLE_CONTROLLED_FINER_DISCRETIZATION_EFFECT**.

Each object uses four nested meshes obtained only by midpoint edge splits. The clean PL surface and perturbed initial PL surface are therefore exactly unchanged; RGB/cameras, frozen B/E checkpoints, visibility definition, and `lambda=3e-2` are fixed. No training or HPC queue job was used.

Primary gain is `E CD - Hybrid CD`; positive values favor the differential branch. Resolution is represented by the strictly decreasing characteristic spacing `h=sqrt(clean surface area / N)`; mean and median unique-edge lengths remain in the CSV.

![Controlled resolution curve](controlled_resolution_gain.png)

## Per-level results

| Shape | Level | N | Faces | h=sqrt(A/N) | Initial CD | E CD | Hybrid CD | E-H gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 038b62bb | 0 | 5716 | 11242 | 0.05449960 | 0.004310649 | 0.003678825 | 0.004100866 | -0.000422041 |
| 038b62bb | 1 | 11432 | 22646 | 0.03853703 | 0.004233787 | 0.003857238 | 0.003832701 | 0.000024537 |
| 038b62bb | 2 | 22864 | 45487 | 0.02724980 | 0.004307056 | 0.003882770 | 0.003962606 | -0.000079836 |
| 038b62bb | 3 | 40012 | 79734 | 0.02059891 | 0.004308925 | 0.004075149 | 0.004049338 | 0.000025811 |
| 43bd0910 | 0 | 7578 | 14906 | 0.04343172 | 0.002766356 | 0.002564608 | 0.002584995 | -0.000020387 |
| 43bd0910 | 1 | 15156 | 30005 | 0.03071087 | 0.002790152 | 0.002683050 | 0.002737604 | -0.000054554 |
| 43bd0910 | 2 | 30312 | 60255 | 0.02171586 | 0.002786206 | 0.002595847 | 0.002616136 | -0.000020289 |
| 43bd0910 | 3 | 53046 | 105668 | 0.01641565 | 0.002790949 | 0.002621593 | 0.002634827 | -0.000013235 |
| 5ac05fe8 | 0 | 8476 | 16827 | 0.04184417 | 0.003166058 | 0.002428613 | 0.002315646 | 0.000112967 |
| 5ac05fe8 | 1 | 16952 | 33773 | 0.02958829 | 0.003110163 | 0.002639608 | 0.002601358 | 0.000038250 |
| 5ac05fe8 | 2 | 33904 | 67674 | 0.02092208 | 0.003136096 | 0.002755892 | 0.002716325 | 0.000039567 |
| 5ac05fe8 | 3 | 59332 | 118523 | 0.01581561 | 0.003121144 | 0.002804127 | 0.002866416 | -0.000062289 |
| 5c226f2b | 0 | 7127 | 14202 | 0.04274697 | 0.003496023 | 0.003452969 | 0.002971357 | 0.000481612 |
| 5c226f2b | 1 | 14254 | 28456 | 0.03022668 | 0.003598470 | 0.003332662 | 0.003525995 | -0.000193333 |
| 5c226f2b | 2 | 28508 | 56964 | 0.02137349 | 0.003601894 | 0.003525185 | 0.003581182 | -0.000055997 |
| 5c226f2b | 3 | 49889 | 99726 | 0.01615684 | 0.003613393 | 0.003510662 | 0.003534352 | -0.000023690 |
| 653efc24 | 0 | 6403 | 12668 | 0.05676479 | 0.008235795 | 0.003880181 | 0.004696209 | -0.000816029 |
| 653efc24 | 1 | 12806 | 25434 | 0.04013877 | 0.008262993 | 0.007587973 | 0.007069477 | 0.000518496 |
| 653efc24 | 2 | 25612 | 50970 | 0.02838239 | 0.008256907 | 0.007746170 | 0.007933885 | -0.000187716 |
| 653efc24 | 3 | 44821 | 89276 | 0.02145507 | 0.008313207 | 0.007470877 | 0.007601451 | -0.000130574 |

## Within-shape trend

| Shape | Spearman(h, gain) | Coarse gain | Fine gain | Fine-coarse | Monotonic |
|---|---:|---:|---:|---:|---:|
| 038b62bb | -0.8000 | -0.000422041 | 0.000025811 | 0.000447853 | False |
| 43bd0910 | -0.8000 | -0.000020387 | -0.000013235 | 0.000007153 | False |
| 5ac05fe8 | 0.8000 | 0.000112967 | -0.000062289 | -0.000175256 | False |
| 5c226f2b | 0.2000 | 0.000481612 | -0.000023690 | -0.000505302 | False |
| 653efc24 | -0.4000 | -0.000816029 | -0.000130574 | 0.000685455 | False |

Macro slope of gain versus `log(h)`: `-4.797259e-05` (shape-bootstrap 95% CI `[-0.0003309621, 0.000235017]`). A negative slope means finer discretization is associated with larger Hybrid gain.

Mean finest-minus-coarsest gain: `9.198045e-05` (shape-bootstrap 95% CI `[-0.00027079, 0.00045475]`). Mean within-shape Spearman(h, gain): `-0.2000` (`[-0.7200, 0.3600]`).

No shape is monotonic. Per-level macro gains from coarse to fine are: `r0=-0.000132776`, `r1=0.0000666793`, `r2=-0.0000608540`, and `r3=-0.0000407953`; none has a bootstrap interval that establishes a positive gain.

## Decision

The controlled experiment does not support the claim that finer discretization systematically improves differential recovery for these frozen predictors. The slope, endpoint change, and mean within-shape rank association all have intervals spanning zero, and the response is strongly nonmonotonic. Resolution changes can alter B/E behavior, but there is no stable direction of effect here.

## Contract and numerical audit

- Objects: `5`; meshes: `20`.
- Maximum relative clean-surface area range within shape: `3.444e-16`.
- Maximum relative initial-surface area range within shape: `2.728e-16`.
- Maximum initial-CD relative range within shape: `3.281e-02` (remaining variation is evaluator sampling over identical surfaces).
- Maximum level-0 recomputed/prepared visibility disagreement: `2.004e-05`.
- Maximum base CPU/archive prediction relative RMS: B `0.5930%`, E `2.1278%`.
- Recovery: frozen B raw field + frozen E direct anchor, float64 PCG, `lambda=0.03`, `tol=1e-4`, maximum 2048 iterations.
- Metric protocol: `mlr.learned_laplacian.evaluation.evaluate_mesh_geometry;area_weighted_triangle_surface_sampling;bidirectional_sampled_surface_to_exact_triangle_surface;surface_samples=3000;seed=7;fscore_threshold=0.01;alignment=shared_prepared_coordinate_frame_no_ICP`.

The experiment controls discretization within each shape, but it still tests frozen predictors trained on the original mixed-resolution distribution. It identifies an inference-time discretization response, not a universal convergence theorem.
