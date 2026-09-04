# Sofa50 paired A+E/B+E result meshes at lambda=0.01

This directory contains the exact 50-test-sample frozen fusion outputs used by the matched comparison.
Each sample directory contains `A_plus_E_lambda1e2.obj` and `B_plus_E_lambda1e2.obj` with the original input connectivity.
No evaluator, model inference, training, or HPC job was run during export; vertices were reconstructed from the archived A/B/E predictions with the same float64 PCG fusion solve.

Git tracks this README and `MANIFEST.json` as the reproducibility index. The 100
OBJ payloads are intentionally excluded from Git because the complete local
bundle is approximately 252 MiB; regenerate them with
`scripts/export_sofa50_direct_lap_positional_fusion_lambda1e2_meshes.py`.

- OBJ files: `100`.
- Maximum VRMS reproduction error: `0.000e+00`.
- Maximum OBJ vertex round-trip absolute error: `5.000e-09`.
- Exact relative paths, SHA-256 hashes, sample IDs, topology counts, and solver audits are in `MANIFEST.json`.
