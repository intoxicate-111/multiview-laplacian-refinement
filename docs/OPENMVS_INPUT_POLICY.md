# OpenMVS input policy

Status: active from 2026-08-22.

The Sofa50 OpenMVS meshes are low-quality external reconstructions with severe missing detail and reconstruction artefacts. They are substantially outside the controlled synthetic-current input distribution and must not be treated as a desired mesh target.

## Required interpretation

OpenMVS meshes may be retained only as:

- out-of-distribution, low-quality-input stress tests;
- robustness and failure-analysis cases;
- an external reconstruction baseline when its own initial quality is reported.

They must not be used as:

- training targets, pseudo-GT, supervision proxies or target-topology templates;
- checkpoint-selection or hyperparameter-selection endpoints;
- a quality ceiling for the learned refinement method;
- primary evidence that one learned model, loss or architecture is better;
- the target distribution for future dataset scaling or method design.

Any retained OpenMVS table must be labelled `diagnostic only / non-decisional` and must report the initial mesh quality beside the refined result. Historical OpenMVS numbers remain valid records of the executed stress tests, but conclusions and future decisions must be based on controlled GT-derived current meshes, same-initial benchmarks, or external inputs that pass an explicit mesh-quality gate.

The projected-GT oracle experiment remains useful for decomposing representation, recovery and prediction failure on this particular poor input. It does not convert OpenMVS into a target, establish a general method ceiling, or justify optimising the pipeline toward OpenMVS-specific artefacts.
