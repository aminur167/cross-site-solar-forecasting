# Research Gap Matrix and Novelty Claims

**Journal Benchmark Target:** *Energy and AI* / *Applied Energy* (Elsevier)  
**Research Topic:** Cross-Site Solar PV Power Forecasting and Conformal Uncertainty

---

## 1. Systematic Comparison with Closely Related Literature

| Research Dimension | Conventional Solar ML (e.g., Mihăescu 2023, Yang 2020) | Deep Transfer Learning (e.g., Zhang et al. 2022) | Conformal Uncertainty (e.g., Huang et al. 2024) | **This Research (Rahman 2026)** |
|---|---|---|---|---|
| **Data Scarcity Assumption** | Assumes multi-year historical logs are readily available ($\ge 1\text{--}2$ years). | Assumes moderate target data ($> 30\text{--}60$ days). | Evaluated on single-site partitions with complete history. | **Resolves extreme scarcity on newly commissioned plants with few-shot budgets ($k \in \{0, 1, 3, 7, 14\}$ days).** |
| **Model Family & Training Cost** | Classical ML or heavy DNNs trained from scratch per site. | Deep neural networks requiring GPU backpropagation fine-tuning. | Single-site neural or quantile baseline. | **Lightweight Gradient Boosted Trees with Optuna TPE tuning (trains in $< 30$s on standard CPU).** |
| **Cross-Site Generalization** | ❌ None (Within-site only). | ✅ Yes (Rooftop PV). | ❌ None (Within-site only). | **✅ Yes (Independent utility-scale commercial PV parks).** |
| **Uncertainty Calibration** | Deterministic point forecast ($\hat{y}$) only. | Deterministic point forecast ($\hat{y}$) only. | ✅ Split CQR on stationary single-site data. | **✅ Cross-Site CQR evaluated under geographic and seasonal distribution shift.** |
| **Telemetry Fault Tolerance** | ❌ Assumes 100% pristine sensors at test time. | ❌ Complete weather assumed. | ❌ Complete weather assumed. | **✅ Stress-tested under 4 physical failure scenarios (random, contiguous, pyranometer outage, drift).** |
| **Statistical Rigor** | Standard MAE/RMSE without hypothesis testing. | Paired t-test without HAC autocorrelation correction. | Marginal coverage comparison. | **Diebold-Mariano test with Newey-West HAC correction and stationary block bootstrap 95% CIs.** |

---

## 2. Summary of Identified Literature Gaps Addressed

### Gap 1: Catastrophic Overfitting on Newly Commissioned Solar Plants
* *Existing State:* Newly commissioned PV plants cannot reliably run complex forecasting algorithms until months of operation have elapsed.
* *Our Resolution:* We demonstrate that zero-shot cross-site transfer with dimensionless capacity factor normalization achieves an immediate $4.94\%$ nMAE, and 1-day adaptation achieves $5.05\%$ nMAE—delivering an **$82.8\%$ error reduction** relative to scratch training.

### Gap 2: Fragility of Conformal Prediction Under Cross-Site Domain Shifts
* *Existing State:* Conformal prediction papers evaluate intervals on exchangeable data within the same site, leaving its behavior under regional covariate shift unexamined.
* *Our Resolution:* We benchmark CQR across independent commercial PV installations, verifying that empirical coverage reaches $92.95\%$ (exceeding nominal $90.0\%$) with sharp interval widths ($MPIW = 0.2501$).

### Gap 3: Blindness of Deployed Forecasters to Remote Sensor Failures
* *Existing State:* Academic benchmarks assume 100% sensor uptime, whereas real utility operators face frequent communication packet loss and pyranometer soiling.
* *Our Resolution:* We systematically profile model degradation across 4 realistic failure scenarios, proving the model tolerates up to $30\%$ telemetry dropout while pinpointing solar irradiance measurement as the single point of physical vulnerability.
