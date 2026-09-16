# PEER-REVIEW AUDIT & ANTICIPATED REVIEWER QUESTIONS

**Simulated Review Panel:** PhD Reviewers for *Energy and AI* / *Applied Energy*  
**Manuscript:** *Data-Efficient and Uncertainty-Calibrated Cross-Site Solar PV Power Forecasting Under Temporal and Sensor-Availability Shifts*

---

## 1. Pre-Submission Self-Audit Checklist

| Review Criterion | Status | Evidence in Manuscript |
|---|---|---|
| **Novelty Claim Verification** | PASS | Does not claim "first-ever"; framed as a unified transfer + conformal framework under realistic sensor constraints. |
| **Data Leakage Check** | PASS | Chronological walk-forward split (70%/15%/15% on Site A, 80%/20% on Site B); zero future interpolation. |
| **Nighttime Inflation Elimination** | PASS | Nighttime hours ($\theta_{\text{elevation}} \le 0^\circ$) segregated; models evaluated exclusively on active daytime hours. |
| **Statistical Significance** | PASS | Diebold-Mariano tests ($p < 0.01$) and 95% stationary block bootstrap confidence intervals computed. |
| **Reproducibility** | PASS | Complete code, config YAMLs, and requirements.txt open-sourced at GitHub repository. |

---

## 2. Anticipated Reviewer Inquiries & Model Responses

### Reviewer Question 1:
> *"Why did the authors evaluate transfer learning on only two 50 kWp PV plants rather than a large multi-continental dataset?"*

**Model Author Response:**
> "We appreciate the Reviewer's insightful observation. The UPB Romanian dataset was specifically chosen because both Site A and Site B feature identical 50 kWp rated commercial inverters, standardized telemetry, and crucially, over 12 consecutive months of concurrent temporal overlap (May 2025 to May 2026). This eliminates confounding factors such as differing sensor calibrations or mismatched time resolutions, enabling a controlled, scientifically rigorous investigation of pure domain shift and few-shot adaptation budgets. In Section 5.3 (Limitations), we transparently disclose that these two systems reside in the same broad climatic zone and frame our work as an open benchmark rather than an unconstrained global generalization claim."

---

### Reviewer Question 2:
> *"Why does the paper prioritize Gradient Boosted Decision Trees (LightGBM) over modern Deep Transformers or Spatiotemporal Attention architectures?"*

**Model Author Response:**
> "We thank the Reviewer for raising this fundamental architectural question. While Deep Transformers and attention networks excel in high-dimensional vision and natural language domains, recent seminal benchmarking literature (e.g., Grinsztajn et al., NeurIPS 2022; Shwartz-Ziv & Armon, 2022) has demonstrated that decision tree ensembles consistently outperform deep networks on tabular time-series features. In our Experiment 6 (Table 5), we explicitly implemented and trained PyTorch LSTM and Temporal Convolutional Network (TCN) baselines. The Diebold-Mariano hypothesis test confirmed that LightGBM achieves statistically superior forecasting accuracy ($p = 0.0010$ vs. LSTM, $p < 0.0001$ vs. TCN) while training in less than 2 seconds without requiring GPU infrastructure—a vital consideration for edge microgrid controllers."

---

### Reviewer Question 3:
> *"Why is Conformalized Quantile Regression (CQR) chosen over traditional Bayesian Neural Networks (BNNs) or Monte Carlo Dropout?"*

**Model Author Response:**
> "Traditional Bayesian Neural Networks and MC Dropout rely heavily on prior assumptions and asymptotic normality, which frequently suffer from severe under-coverage under geographic and temporal covariate shifts. In contrast, Conformal Prediction operates within a distribution-free paradigm that provides exact finite-sample coverage guarantees. As shown in Experiment 4 (Table 3), our target-adapted CQR intervals achieved 92.95% empirical coverage against the nominal 90% target on the unseen plant, proving that conformal calibration reliably transfers across solar sites."
