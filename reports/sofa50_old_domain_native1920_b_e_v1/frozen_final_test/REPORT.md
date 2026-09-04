# Old-domain native-1920 frozen B+E final test

Contract audit: **true**.

Arm-E and Frozen B+E were opened on the test split once, after both specialist checkpoints and the validation-selected fusion lambda were locked. Arm-B test metrics had previously been opened in the authorized Arm-B-only comparison, so this is sealed for E/Hybrid rather than a claim that no method had ever touched the test set.

Locked fusion: `lambda_old=0.01`; B SHA `c250afad2b63828a4b8ae3d692dca22757e6477a50d13e882461035a7a35522a`; E SHA `14f935c42eb31e675d2fd064d3f15bbf53bf3d0c4c00af7663be6d6ad592f034`.

## Unified same-input comparison

| Method | CD | CD gain | P2S p95 | F-score | Normal | Improved/worsened |
|---|---:|---:|---:|---:|---:|---:|
| Initial mesh | 0.0170704685 | +0.00% | 0.0724794854 | 0.577250432 | 0.955190949 | 0/0 |
| NDS | 0.0112049924 | +34.36% | 0.0398475607 | 0.652827299 | 0.873805125 | 22/3 |
| nvdiffrec | 0.0136546593 | +20.01% | 0.0457457720 | 0.558673128 | 0.848122276 | 18/7 |
| ExMesh | 0.0201706152 | -18.16% | 0.0696287606 | 0.478513280 | 0.845337056 | 8/17 |
| Old-domain Arm B | 0.0085377693 | +49.99% | 0.0271284035 | 0.716572715 | 0.948334515 | 25/0 |
| Old-domain Arm E | 0.0080658043 | +52.75% | 0.0274944580 | 0.750907436 | 0.954472757 | 25/0 |
| Old-domain Frozen B+E | 0.0067045978 | +60.72% | 0.0208419391 | 0.793502547 | 0.949512478 | 25/0 |

## Paired Frozen B+E comparisons

Differences are Frozen B+E minus comparator. Negative CD/P2S and positive F-score/normal favor Frozen. CIs are reported both over 25 meshes and over the five object clusters (five variants per object).

| Comparator | CD difference [mesh 95% CI] | Object-cluster 95% CI | CD W/L/T | P2S-p95 difference | F-score difference | Normal difference |
|---|---:|---:|---:|---:|---:|---:|
| Old-domain Arm B | -0.0018331715 [-0.0024621883, -0.0012539769] | [-0.0029988108, -0.0008916676] | 22/3/0 | -0.0062864644 | 0.076929831 | 0.001177963 |
| Old-domain Arm E | -0.0013612065 [-0.0018119447, -0.0009232948] | [-0.0021255390, -0.0005675016] | 23/2/0 | -0.0066525189 | 0.042595111 | -0.004960278 |
| NDS | -0.0045003946 [-0.0052805585, -0.0037927501] | [-0.0058853458, -0.0033796185] | 25/0/0 | -0.0190056217 | 0.140675248 | 0.075707353 |
| nvdiffrec | -0.0069500616 [-0.0078533558, -0.0060381559] | [-0.0085337014, -0.0054727902] | 25/0/0 | -0.0249038330 | 0.234829419 | 0.101390203 |
| ExMesh | -0.0134660174 [-0.0171267291, -0.0101594013] | [-0.0219418620, -0.0066405162] | 25/0/0 | -0.0487868215 | 0.314989266 | 0.104175422 |

## Geometry trade-offs

| Method | Vertex RMS | Introduced flips / rate | New degeneracies |
|---|---:|---:|---:|
| Old-domain Arm B | 0.0108807326 | 8131 / 2.328% | 0 |
| Old-domain Arm E | 0.0086640004 | 7565 / 2.166% | 0 |
| Old-domain Frozen B+E | 0.0104264906 | 9435 / 2.702% | 0 |

## Compute time

Our timing excludes mesh export and the common evaluator. Frozen model-forward time is the sum of independent B and E forward calls; total time is model forward plus the float64 sparse fusion solve. External totals are historical pipeline measurements and are not hardware-normalized.

| Method | Model forward s/mesh | Sparse solve s/mesh | Total compute s/mesh |
|---|---:|---:|---:|
| Old-domain Arm B | 2.252271 | 0.041308 | 2.293578 |
| Old-domain Arm E | 2.320147 | 0.000000 | 2.320147 |
| Old-domain Frozen B+E | 4.572418 | 0.037529 | 4.609947 |
| NDS | n/a | n/a | 227.309600 |
| nvdiffrec | n/a | n/a | 824.982000 |
| ExMesh | n/a | n/a | 762.400400 |

## Audit

- Samples: `25` exact common native-1920 inputs.
- Archived NDS/nvdiffrec/ExMesh reproduction: `true`.
- Solver: float64 PCG, tolerance `1e-8`; all converged: `true`; maximum residual `9.977e-09`.
- Metric protocol: `mlr.learned_laplacian.evaluation.evaluate_mesh_geometry;area_weighted_triangle_surface_sampling;bidirectional_sampled_surface_to_exact_triangle_surface;surface_samples=3000;seed=7;fscore_threshold=0.01;alignment=shared_prepared_coordinate_frame_no_ICP`.
- Test was not used to choose either checkpoint or lambda, and no test lambda sweep was run.
