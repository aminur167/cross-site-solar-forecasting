# Research result reconciliation

The original result files are preserved in Git history at the inspected base commit (`b9bd805d39065ccb313250d2739e33a7295d93fa`). This rerun regenerated processed matrices and experiment JSON from bundled raw CSVs after target-time alignment, common cutoff, target-day masking, missingness flags, and five-seed synthetic perturbation corrections. Differences therefore reflect protocol and data-funnel changes, not a controlled one-variable comparison.

| Metric | Historical committed value | Rerun value | Change | Assessment |
|---|---:|---:|---:|---|
| Site A GB nMAE | 5.93% | 6.11% | +0.18 pp | Historical claim superseded |
| Site B zero-shot nMAE | 6.02% | 6.93% | +0.91 pp | Superseded; aligned target day and common cutoff |
| Target-only 1d nMAE | 28.17% | 28.06% | -0.11 pp | Similar, not identical |
| Pooled adaptation 1d nMAE | 6.05% | 6.97% | +0.92 pp | Superseded; adaptation not better than zero-shot |
| Raw CQR daytime PICP | 87.83% | 88.20% | +0.37 pp | Superseded |
| Source CQR daytime PICP | 87.83% | 90.27% | +2.44 pp | Superseded; source daytime calibration applied |
| Target CQR daytime PICP | 89.26% | 94.51% | +5.25 pp | Superseded; target daytime 14-calendar-day calibration |
| Sensor test baseline | 8.50% nMAE | 6.93% nMAE | -1.57 pp | Prior used CQR midpoint; rerun uses same LightGBM point model as transfer |
| 30% independent MCAR | 8.85% / 85.84% PICP | 7.46% / 92.78% | -1.39 pp / +6.94 pp | Five-seed synthetic feature corruption |
| GHI outage | 11.30% / 71.58% | 10.84% / 86.02% | -0.46 pp / +14.44 pp | Point and interval models are separate; synthetic input outage only |
| LightGBM daytime nMAE (DL comparison) | 6.02% | 6.93% | +0.91 pp | Superseded; contiguous sequence windows and common cutoff |
| LSTM daytime nMAE | 6.73% | 7.37% | +0.64 pp | Single fixed seed |
| TCN daytime nMAE | 6.43% | 7.69% | +1.26 pp | Single fixed seed |

Final values are in `experiments/results/*.json`; tables and figures were regenerated from those files. The reported historical values are transcribed from the prior committed artifacts and are superseded, not independently revalidated. The unrounded result payloads are authoritative.
