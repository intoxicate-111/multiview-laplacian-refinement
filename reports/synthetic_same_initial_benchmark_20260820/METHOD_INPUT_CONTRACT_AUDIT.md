# Method input-contract audit

Audit date: 2026-08-20. Repositories were inspected at the pinned commits below.

| Method | Commit | Group | Actual geometry entry | Decision |
|---|---|---|---|---|
| initial | project data | A | Exact existing `coarse.obj` validated against prepared `vertices`/`faces` | Common baseline. |
| ours | frozen canonical HF1920 checkpoint | A | Prepared vertices are queries; prepared faces are graph/recovery connectivity | Same initialization; recovery preserves connectivity. |
| ExMesh | `09950d283fc5372a09079e30c88d998f1c40b2d0` | A | Official refinement discovers `<scene>/mesh.ply`; adapter writes the exact prepared mesh there and bypasses PGSR | Legitimate same-initial refinement; official topology adaptation may change connectivity. |
| NDS | `760e4549f59adaed9adf1bd705599786a00ba6b8` | A | `reconstruct.py --initial_mesh <path>` loads the supplied mesh instead of `vh16/vh32/vh64/sphere16` | Legitimate same-initial refinement; default visual hull is bypassed. |
| nvdiffrec | `abf3a34b1eb6e782abffefc2462c7e9bcd89f9bb` | A | `train.py` loads `base_mesh`, constructs official `DLMesh`, and optimizes it | Legitimate fixed-topology same-initial refinement. Exact per-view camera adapter is required because stock DatasetNERF assumes centered single-FOV intrinsics. |
| Neuralangelo | `94390b64683c067c620d9e075224ccfe582647d0` | B | Optimizes a neural SDF from images; output is extracted by marching cubes | Image-based reconstruction reference only; no arbitrary triangular initial-mesh refinement path. |
| MAtCha | `b119fd96e484fc81eb40623c1ea92ad3dbd3c21e` | B | MASt3R/point-map and 2D-Gaussian/chart pipeline; mesh is subsequently extracted | Image-based reconstruction reference only; no supplied triangular mesh optimizer. |

Group A adapters have an aborting preflight that round-trips the exported mesh and verifies vertex count, face count, face ordering, and maximum vertex error against the prepared tensors before launching the external method.

The common initial file SHA-256 remains the canonical provenance identifier. Method-specific OBJ/PLY serialization hashes are also recorded, but are not expected to be byte-identical across formats; their decoded geometry must pass the exact connectivity and `1e-6` vertex-tolerance gate.

For methods that change topology, the project's existing face-correspondence definition of “introduced flipped faces” is not mathematically comparable and will be reported as unavailable rather than replaced with a different metric. For topology-preserving methods it is computed with the canonical face-normal sign-change definition.
