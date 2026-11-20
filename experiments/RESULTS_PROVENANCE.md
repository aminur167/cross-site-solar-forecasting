# Results Provenance & Traceability Matrix

This document maps every empirical claim, table entry, and figure in the manuscript to its exact underlying JSON file, generating script, and data slice.

---

## 1. Table Provenance

| Manuscript Table | LaTeX Source File | Serialized Source JSON | Generating Script | Key Metric / Verification Column |
| :--- | :--- | :--- | :--- | :--- |
| **Table 1: Baseline Benchmark (Site A)** | `tables/table1_baseline_benchmark.tex` | `experiments/results/baseline_benchmark_results.json` | `experiments/run_baseline_benchmark.py` | `daytime` metrics: Gradient Boosting nMAE 5.93%, nRMSE 10.08%, $R^2$ 0.9089, Skill +0.4068 |
| **Table 2: Cross-Site Adaptation Frontier** | `tables/table2_cross_site_adaptation.tex` | `experiments/results/cross_site_adaptation_results.json` | `experiments/run_cross_site_experiments.py` | Zero-shot nMAE 6.02%; Target-only 1-day 28.17% vs Adapted 1-day 6.05% (-78.5% error reduction) |
| **Table 3: CQR & Sensor Robustness** | `tables/table3_uncertainty_robustness.tex` | `experiments/results/uncertainty_robustness_results.json` | `experiments/run_uncertainty_and_robustness.py` | Target CQR PICP 89.26%, MPIW 0.2882; 50% MCAR nMAE 9.56%; Total GHI Outage nMAE 11.30% |
| **Table 4: Global SHAP Feature Importance** | `tables/table4_feature_importance.tex` | `experiments/results/shap_importance_results.json` | `src/evaluation/explainability.py` | Rank 1: `lag_0` (54.45%), Rank 2: `hour_sin` (10.81%), Rank 3: `ghi_diff_1h` (7.77%) |
| **Table 5: Deep Learning vs GBDT** | `tables/table5_deep_learning_benchmark.tex` | `experiments/results/deep_learning_and_statistics.json` | `experiments/run_deep_learning_benchmark.py` | LightGBM nMAE 6.02% [5.41, 6.68] vs LSTM 6.73% (DM=-6.17, $p<0.0001$), TCN 6.43% (DM=-3.51, $p=0.0005$) |

---

## 2. Figure Provenance

| Manuscript Figure | Figure File (PNG & PDF) | Underlying Data Source | Generating Script | Diagnostic Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Figure 1: Benchmark Error & Skill** | `figures/fig1_baseline_benchmark.png` | `experiments/results/baseline_benchmark_results.json` | `src/visualization/plot_benchmark.py` | Demonstrates within-site superiority of tree-based ensembles over persistence & ridge |
| **Figure 2: Adaptation Frontier Curve** | `figures/fig2_adaptation_frontier.png` | `experiments/results/cross_site_adaptation_results.json` | `src/visualization/plot_cross_site_adaptation.py` | Visualizes the learning curve frontier comparing Target-Only from scratch vs Transfer Adaptation |
| **Figure 3: CQR Calibration & Robustness** | `figures/fig3_uncertainty_robustness.png` | `experiments/results/uncertainty_robustness_results.json` | `src/visualization/plot_uncertainty_robustness.py` | (a) Empirical PICP vs MPIW trade-off; (b) Dual-axis error and coverage degradation curves |
| **Figure 4: SHAP Feature Attribution** | `figures/fig4_shap_feature_importance.png` | `experiments/results/shap_importance_results.json` | `src/evaluation/explainability.py` | Horizontal bar chart of top physical drivers of cross-site power predictions |
| **Figure 5: Parity Diagnostic & Residuals** | `figures/fig5_parity_plots.png` | Target Site B Holdout Predictions | `src/visualization/plot_parity.py` | 2D hexbin scatter plot with 1:1 identity line, $\pm 10\%$ error envelope, and marginal residual histogram |
| **Figure 6: Reliability & Sharpness Curves** | `figures/fig6_reliability_diagram.png` | Conformal Quantile Calibration Sweep | `src/visualization/plot_reliability_diagram.py` | (a) Nominal vs Empirical coverage calibration; (b) Mean prediction interval width (sharpness) curve |

---

## 3. Data Funnel Verification

```
Raw Telemetry Ingestion:
  ├── Site A: 13,031 continuous hourly timestamps (2025-01-01 to 2026-06-30)
  │    ├── Zero generation at night: 6,537 hours (physically verified, elevation <= 0°)
  │    └── Daytime active generation: 6,494 hours (elevation > 0°)
  └── Site B: 9,504 continuous hourly timestamps (2025-05-01 to 2026-05-31)
       ├── Zero generation at night: 4,714 hours (physically verified, elevation <= 0°)
       └── Daytime active generation: 4,790 hours (elevation > 0°)

Causal Feature Construction (Lag order = 24):
  ├── Site A Feature Matrix: 12,713 continuous hourly rows (6,452 daytime hours)
  └── Site B Feature Matrix: 9,291 continuous hourly rows (4,755 daytime hours)

Target Evaluation Partition (Site B 20% Holdout):
  ├── Total Holdout Hours: 1,859 continuous hours
  └── Daytime Active Holdout Hours: 1,052 hours (evaluation test horizon)
```
