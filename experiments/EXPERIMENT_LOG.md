# Scientific Experiment Execution Log

## Protocol Metadata
- **Project Title:** Data-Efficient and Uncertainty-Calibrated Cross-Site Solar PV Power Forecasting Under Temporal and Sensor-Availability Shifts
- **Execution Date:** 2026-11-20
- **Random Seed:** 42 (fixed across all stochastic components, train splits, bootstrap samplers, and perturbation generators)
- **Environment:** Windows 11, Python 3.12, PyTorch 2.5+, LightGBM 4.5+, Scikit-Learn 1.6+, Pandas 2.2+, NumPy 2.1+
- **Execution Mode:** Strictly causal, forward-fill only ($L_{\max}=2$), zero future interpolation, explicit daytime masking ($\theta_{\text{elev}} > 0^\circ$).

---

## Experiment Run Registry

### 1. Data Quality Audit & Ingestion
- **Script:** `src/data/data_quality_auditor.py`
- **Output:** `results/data_quality_report.json`
- **Verified Record Counts:**
  - Site A (Source): 13,031 raw continuous hourly records (2025-01-01 00:00 to 2026-06-30 23:00 UTC). Valid daytime active records: 6,494.
  - Site B (Target): 9,504 raw continuous hourly records (2025-05-01 00:00 to 2026-05-31 23:00 UTC). Valid daytime active records: 4,790.
  - Negative Generation / Infeasible Solar Elevation: 0 records.

### 2. Experiment 1: Within-Site Classical & Baseline Benchmark (Site A)
- **Script:** `experiments/run_baseline_benchmark.py`
- **Dataset:** `data/processed/site_a_features.csv` (12,713 valid rows after causal lag generation).
- **Split Protocol:** Chronological 70% Train (8,899 hrs), 15% Validation (1,907 hrs), 15% Holdout Test (1,907 hrs, 1,159 daytime active hours).
- **Test Period:** 2026-04-12 09:00:00 to 2026-06-30 19:00:00 UTC.
- **Models Evaluated:** Persistence, Seasonal Naive (24h), Ridge Regression, Random Forest, Gradient Boosting.
- **Key Metric Findings (Daytime Primary):**
  - Gradient Boosting: nMAE = 5.93%, nRMSE = 10.08%, $R^2 = 0.9089$, Skill Score vs Persistence = +0.4068.
  - All-Hours Comparison: All-hours nMAE = 3.65% (demonstrating why nighttime separation is vital to prevent artificial deflation).
- **Serialized Output:** `experiments/results/baseline_benchmark_results.json`

### 3. Experiment 2 & 3: Cross-Site Transfer and Few-Shot Adaptation Frontier
- **Script:** `experiments/run_cross_site_experiments.py`
- **Source Domain:** Site A (10,806 training samples, 85% chronological slice).
- **Target Domain:** Site B Adaptation Pool (7,432 samples), Holdout Test Set (1,859 total hours, 1,052 daytime active hours).
- **Test Period:** 2026-03-12 21:00:00 to 2026-05-31 19:00:00 UTC.
- **Budgets Tested:** $k \in \{0, 1, 3, 7, 14\}$ consecutive calendar days (strictly right-open $[t_0, t_0 + k\text{ days})$).
- **Key Findings (Daytime Active Hours):**
  - Persistence Reference: nMAE = 13.38%, nRMSE = 17.18%, $R^2 = 0.7184$.
  - Target Oracle (Full Target Data): nMAE = 6.62%, nRMSE = 10.68%, $R^2 = 0.8912$.
  - Zero-Shot Transfer ($k=0$): nMAE = 6.02%, nRMSE = 9.93%, $R^2 = 0.9059$, Skill Score = +0.4219.
  - Target-Only (1 day): nMAE = 28.17% ($R^2 = -0.1325$, catastrophic overfitting).
  - Target-Only (3 days): nMAE = 15.21% (inferior to simple persistence).
  - Target-Only (14 days): nMAE = 9.40%.
  - Adapted Transfer ($k=1$ to $14$ days): Stabilizes error between 6.05% and 6.10% nMAE, achieving an immediate 78.5% error reduction over target-only learning at $k=1$ day.
- **Serialized Output:** `experiments/results/cross_site_adaptation_results.json`

### 4. Experiment 4: Uncertainty Quantification & Conformalized Quantile Regression (CQR)
- **Script:** `experiments/run_uncertainty_and_robustness.py`
- **Target Horizon:** Site B Holdout ($N=1,052$ daytime active hours). Nominal coverage $1 - \alpha = 90\%$.
- **Regimes Evaluated:**
  - Raw Quantile Regression (Uncalibrated pinball loss): Day PICP = 87.83%, MPIW = 0.2852, Winkler = 0.3927 (All-hours PICP = 92.58%).
  - Source-Calibrated CQR: Day PICP = 87.83%, MPIW = 0.2852, Winkler = 0.3927.
  - Target-Adapted CQR (14-day local calibration): Day PICP = 89.26%, MPIW = 0.2882, Winkler = 0.3921.
- **Theoretical Finding:** Quantile regression under distribution shift suffers slight empirical undercoverage ($87.83\% < 90\%$). Local 14-day conformal calibration restores empirical coverage to $89.26\%$, closely tracking the nominal $90\%$ target.

### 5. Experiment 5: Sensor-Availability Robustness Stress-Testing
- **Script:** `experiments/run_uncertainty_and_robustness.py`
- **Scenarios Evaluated (Daytime Primary):**
  - Complete Telemetry: nMAE = 8.50%, PICP = 89.26%, MPIW = 0.2882.
  - 10% MCAR: nMAE = 8.65%, PICP = 88.31%.
  - 30% MCAR: nMAE = 8.85%, PICP = 85.84%.
  - 50% MCAR: nMAE = 9.56%, PICP = 82.60%.
  - Contiguous 3h / 6h Block Blackouts: nMAE = 8.52% / 8.50%, PICP = 89.16%.
  - Total Pyranometer Failure (Zero GHI): nMAE jumps to 11.30%, PICP drops to 71.58% (revealing primary operational sensitivity).
  - Pyranometer Drift (+15% soiling bias): nMAE = 8.44%, PICP = 89.26%.
- **Serialized Output:** `experiments/results/uncertainty_robustness_results.json`

### 6. Experiment 6: Deep Learning Sequence Models vs. GBDT & Statistical Testing
- **Script:** `experiments/run_deep_learning_benchmark.py`
- **Horizon:** Site B Aligned Holdout ($N=1,052$ daytime active hours). Lookback: $L=6$ hours.
- **Architectures:** LightGBM (GBDT), 2-layer LSTM (hidden dim 64), Dilated Causal TCN (channels 32/64, kernel 3).
- **Statistical Tests:** Diebold-Mariano test with Harvey-Leybourne-Newbold correction and Newey-West HAC spectral density estimator; Stationary Block Bootstrap (block size = 12).
- **Key Findings:**
  - LightGBM (GBDT): Day nMAE = 6.02%, 95% CI: [5.41, 6.68]%, Day nRMSE = 9.93%.
  - PyTorch LSTM: Day nMAE = 6.73%, 95% CI: [6.16, 7.38]%, Day nRMSE = 10.51%, DM Stat = -6.1747, $p < 0.0001$.
  - PyTorch TCN: Day nMAE = 6.43%, 95% CI: [5.82, 7.11]%, Day nRMSE = 10.42%, DM Stat = -3.5131, $p = 0.0005$.
- **Serialized Output:** `experiments/results/deep_learning_and_statistics.json`

### 7. Global SHAP Feature Attribution
- **Script:** `src/evaluation/explainability.py`
- **Model:** LightGBM trained on Source Site A, evaluated on Target Site B holdout.
- **Top Physical Drivers:** `lag_0` (54.45%), `hour_sin` (10.81%), `ghi_diff_1h` (7.77%), `lag_23` (5.85%), `solar_azimuth_deg` (4.49%), `solar_elevation_deg` (3.14%).
- **Serialized Output:** `experiments/results/shap_importance_results.json`
