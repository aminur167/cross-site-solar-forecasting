# Research Scope and Claim Audit

The prior version of this file contained unsupported novelty, literature-comparison, hardware resilience, training-time, and statistical-rigor claims. Those claims are withdrawn. This two-site rerun does not establish priority over prior work.

## Supported observations

- The bundled data contains two roughly 50 kW-class sites in the same southwestern Romanian region.
- Retrospective source-only transfer achieved 6.93% daytime nMAE on the Site B holdout; one-day pooled adaptation achieved 6.97%, so it did not beat zero-shot.
- Synthetic 30% feature-wise MCAR yielded mean 7.46% nMAE across five perturbation seeds; this is not measured field-sensor robustness.
- Target-calibrated CQR achieved 94.51% empirical daytime coverage at nominal 90%; no guarantee is asserted under temporal dependence/site shift.
- The weather features are historical estimates unavailable at an operational forecast issue time.

## Unsupported claims removed

No claim of a first/unprecedented framework, cross-climate or global generalization, real hardware fault tolerance, all-season stability, specific operational benefits, optimized hyperparameters, deployment-time forecast skill, or verified journal novelty is supported. A systematic literature search and author review are still required.
