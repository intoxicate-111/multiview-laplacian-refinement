# Sofa50 v2 pure vertex-error Arm-B versus original recovery-aware Arm-B

Contract audit: **true**. This is a paired, single-pass Arm-B comparison on the exact same 50 validation and 50 test meshes. Recursive R1--R5 evaluation is excluded from this formal comparison.

Both models use the same 826,115-parameter Arm-B predictor, 28x960 RGB inputs, current-query/current-graph raw Laplacian representation, Uniform random-walk operator, and `lambda=1e-2` recovery. Only the training objective changes:

```text
Original Arm B: L = L_raw-Laplacian-Huber + 1e-2 * mean_i ||V_recovered_i - V_clean_i||_2^2
New Arm B:      L = mean_i ||V_recovered_i - V_clean_i||_2^2
```

The new validation-selected checkpoint is epoch `312` (optimizer step `15600`), SHA-256 `3f29d66302f30a487e3aac9c7c09a5875328602cbcc715f3780aa24ba5b6367a`. The original checkpoint SHA-256 is `a483e2212f568e771873594cf1e37d13d62cbd2e1e72244baded7dd15573970c`.

## Aggregate results

| Split | Model | Initial CD | Refined CD | P2S p95 | F-score | Normal | Raw EPE (vertex-wtd.) | Vertex RMS | Improved/worsened |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | Original Arm B | 0.00381765892 | 0.00320962349 | 0.00999485296 | 0.950224425 | 0.96877577 | 0.0020152807 | 0.00767549059 | 46/4 |
| validation | Pure vertex-error Arm B | 0.00381765892 | 0.00345892192 | 0.0107392229 | 0.936345803 | 0.965681501 | 0.0076741744 | 0.00622738242 | 24/26 |
| test | Original Arm B | 0.00438635163 | 0.00358497023 | 0.0105580821 | 0.935012989 | 0.959365744 | 0.00263985669 | 0.0115531855 | 36/14 |
| test | Pure vertex-error Arm B | 0.00438635163 | 0.00397816927 | 0.0117444276 | 0.91756792 | 0.959623804 | 0.00857208259 | 0.0105424394 | 27/23 |

## Paired objective comparison

Differences are pure vertex-error Arm B minus original Arm B. Negative CD/P2S/raw-EPE/vertex-RMS values and positive F-score/normal values favor the new objective. Aggregate raw EPE is vertex-weighted to reproduce the original report; paired differences and confidence intervals treat meshes as sampling units.

| Split | Metric | Mean difference [95% CI] | New W/L/T |
|---|---|---:|---:|
| validation | Refined CD | 0.00024929843 [1.24214217e-05, 0.000465836171] | 15/35/0 |
| validation | P2S p95 | 0.000744369992 [-0.00062423976, 0.00196702667] | 12/38/0 |
| validation | F-score | -0.013878622 [-0.0249428521, -0.00179037792] | 12/38/0 |
| validation | Normal | -0.0030942685 [-0.00452219323, -0.00169348518] | 15/35/0 |
| validation | Raw EPE | 0.00561309645 [0.00548097855, 0.00574025929] | 0/50/0 |
| validation | Vertex RMS | -0.00144810816 [-0.00176312232, -0.00111605549] | 47/3/0 |
| test | Refined CD | 0.000393199046 [0.000164230947, 0.000595043315] | 10/40/0 |
| test | P2S p95 | 0.00118634551 [0.000445349511, 0.00189025193] | 14/36/0 |
| test | F-score | -0.0174450691 [-0.0286728654, -0.00552241081] | 12/38/0 |
| test | Normal | 0.000258060561 [-0.000786339872, 0.00130130835] | 26/24/0 |
| test | Raw EPE | 0.00587576449 [0.00569190362, 0.00605583645] | 0/50/0 |
| test | Vertex RMS | -0.00101074614 [-0.00166600908, -0.000321752053] | 39/11/0 |

## Decision

Classification: **PURE_VERTEX_ERROR_WORSE**.

The pure recovered-vertex objective worsens test Chamfer relative to the original mixed objective.
Test CD is `0.00397816927` for pure vertex-error training versus `0.00358497023` for the original Arm B; the paired mean difference is `0.000393199046` with 95% CI `[0.000164230947, 0.000595043315]` and W/L/T `10/40/0`.

The new objective does optimize its direct target: test same-index vertex RMS falls from `0.0115531855` to `0.0105424394`. However, test raw EPE rises from `0.00263985669` to `0.00857208259`, P2S p95 and F-score both worsen, and only `27/50` meshes improve over their initial geometry versus `36/50` for the original Arm B. The evidence therefore favors retaining the raw-Laplacian auxiliary term for the formal Arm-B method.

This isolates the loss objective within the completed matched-v2 Arm-B setup. It does not make a claim about recursive refinement, B+E fusion, old native-1920 inputs, or Future2000.

## Numerical and contract audit

- Samples: `100` exact paired meshes (`50` validation, `50` test).
- Recovery: Uniform random-walk Laplacian, `lambda=1e-2`, float64 LSMR; all `200` model/split solves across the two compared shards converged.
- Maximum paired initial-Chamfer discrepancy: `0.000e+00`.
- Local CUDA inference used execution-only image-view chunking of `4` views; model parameters, 28-view inputs, predictions, and recovery equations are unchanged.
- Checkpoint selection used validation only. Test rows were evaluated after selection and were not used to choose a checkpoint.
