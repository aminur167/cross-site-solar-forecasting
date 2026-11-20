# Scientific Research Protocol: Cross-Site Solar PV Power Forecasting

**Document Version:** 1.0 (Audit Revalidation)  
**Lead Author:** Aminur Rahman (HSTU, Bangladesh)  
**Project Workspace:** `cross-site-solar-forecasting`  
**Target Journal:** *Energy and AI* / *Applied Energy* (Elsevier)

---

## 1. Research Objectives and Scope

The primary objective of this research is to investigate whether machine learning models trained on a data-rich, mature commercial photovoltaic (PV) facility can be effectively transferred to a newly commissioned PV facility with minimal or zero historical observations, while providing mathematically calibrated prediction intervals and demonstrating robustness against telemetry dropouts.

---

## 2. Formal Research Questions and Hypotheses

* **RQ1 (Cross-Site Transfer Generalization):**  
  *How effectively can a gradient boosted decision tree trained exclusively on Source Site A generalize zero-shot to an unseen Target Site B under dimensionless capacity factor normalization?*  
  * **Hypothesis $H_{1}$:** Dimensionless capacity normalization ($CF = P_{\text{ac}} / P_{\text{rated}}$) combined with cyclic topocentric solar geometry encodings isolates plant-scale differences, enabling zero-shot cross-site transfer to achieve significantly lower forecast error than persistence baselines without any local retraining.

* **RQ2 (Adaptation Budget Frontier):**  
  *What is the minimal observation window ($k \in \{1, 3, 7, 14\}$ days) necessary for domain adaptation to match or surpass a model trained on extensive local history, and how does transfer adaptation compare to training from scratch?*  
  * **Hypothesis $H_{2}$:** Training from scratch on minimal data ($k \le 3$ days) fails due to severe empirical overfitting on unrepresentative atmospheric regimes. In contrast, sample-weighted pooled transfer adaptation leverages source domain priors while rapidly calibrating local inverter tilt/efficiency curves, reaching oracle-level accuracy within $3$ to $7$ days of local data.

* **RQ3 (Uncertainty Calibration under Distribution Shift):**  
  *How reliable are Conformalized Quantile Regression (CQR) intervals when evaluated on an unseen plant experiencing geographic and seasonal covariate shifts?*  
  * **Hypothesis $H_{3}$:** While raw pinball-loss quantile regression suffers empirical undercoverage under domain shift, CQR nonconformity score calibration provides empirical coverage meeting or exceeding the nominal $90\%$ target with sharp prediction interval widths.

* **RQ4 (Sensor Availability Robustness):**  
  *To what degree can the forecaster withstand field communication blackouts and pyranometer degradation before dispatch reliability collapses?*  
  * **Hypothesis $H_{4}$:** Immediate autoregressive power telemetry and cyclic solar geometry preserve baseline forecast stability under moderate communication dropout ($\le 30\%$), whereas total loss of pyranometer irradiance (GHI) constitutes the primary physical failure mode.

---

## 3. Mathematical Problem Formulation

### 3.1 Time Index and Forecast Horizon
* Continuous discrete hourly time index: $t \in \{1, 2, \dots, T\}$.
* Forecast Issue Time: Time $t$.
* Forecast Target Horizon: 1-hour ahead ($t \to t+1$).
* Target Variable: Normalized AC Active Power (Capacity Factor):
  $$y_{t+1} = \text{CF}_{t+1} = \frac{P_{\text{ac}}(t+1)}{P_{\text{rated}}} \in [0.0, 1.2]$$
  where $P_{\text{rated}} = 50.0 \text{ kW}$ for both Site A and Site B.

### 3.2 Information Availability and Causality
At forecast issue time $t$, all inputs $\mathbf{x}_t \in \mathbb{R}^d$ must be strictly causal:
$$\mathbf{x}_t = \left[ \mathbf{p}_{\text{lag}}(t), \mathbf{w}_{\text{met}}(t), \mathbf{s}_{\text{cyclic}}(t), \mathbf{g}_{\text{solar}}(t) \right]$$
* **No Future Lookahead:** No future realized weather observations ($t+1$) are utilized. Weather inputs represent current observations $\mathbf{w}_t$ and autoregressive changes ($\Delta \text{GHI}_{1h} = \text{GHI}_t - \text{GHI}_{t-1}$).
* **Daytime vs. Nighttime Definition:**
  * Daytime Mask: Local solar elevation $\theta_{\text{elev}} > 0.0^\circ$ and $\text{GHI} > 5.0 \text{ W/m}^2$.
  * Nighttime Mask: $\theta_{\text{elev}} \le 0.0^\circ$. At night, physical generation is clamped to $0.0$.
  * **Primary Benchmark Protocol:** All models are evaluated strictly on daytime active generation hours.
  * **Secondary Benchmark Protocol:** Complete 24-hour dispatch evaluation with deterministic nighttime zero-clamping.

---

## 4. Dataset Specification and Split Protocol

### 4.1 Sites
* **Source Site A:** Romanian 50 kWp commercial PV park ($13,031$ raw hourly records, Jan 2025 – Jun 2026).
* **Target Site B:** Independent Romanian 50 kWp commercial PV park ($9,504$ raw hourly records, May 2025 – May 2026).

### 4.2 Split Protocol
Strict chronological splitting is enforced across all experiments:
* **Within-Site Evaluation (Site A):**
  * Train Set: Earliest $70\%$ chronologically consecutive samples.
  * Validation / Hyperparameter Tuning Set: Subsequent $15\%$.
  * Final Holdout Test Set: Final $15\%$ untouched samples.
* **Cross-Site Transfer Protocol:**
  * Source Model Training: Full daytime training set of Site A.
  * Target Adaptation Budget: Exact right-open calendar-day slices of Site B:
    $$\mathcal{D}_T(k) = \{(\mathbf{x}_t, y_{t+1}) \mid t_{\text{start}} \le t < t_{\text{start}} + k \times 24\text{h}\}, \quad k \in \{0, 1, 3, 7, 14\}$$
  * Target Holdout Evaluation Set: Chronologically subsequent untouched portion of Site B (final $20\%$ of Site B telemetry).

---

## 5. Model Architectures and Baselines

1. **Persistence:** $\hat{y}_{t+1} = y_t$.
2. **Seasonal Naive (24h):** $\hat{y}_{t+1} = y_{t-23}$.
3. **Ridge Regression:** L2-regularized linear baseline.
4. **Random Forest:** Ensemble of decorrelated decision trees ($n_{\text{trees}} = 100$).
5. **LightGBM:** Histogram-based gradient boosted decision trees with Optuna TPE tuning.
6. **XGBoost:** Exact-split gradient boosted trees with depth-wise tree growth.
7. **CatBoost:** Oblivious tree gradient boosting with ordered boosting.
8. **PyTorch LSTM:** 2-layer sequence LSTM with causal input sequence construction ($L=24$ hours).
9. **PyTorch TCN:** Dilated causal 1D convolutional network with residual connections.

---

## 6. Uncertainty Quantification Protocol

* **Method:** Conformalized Quantile Regression (CQR) \citep{romano2019conformalized}.
* **Nominal Significance Level:** $\alpha = 0.10$ (Nominal $90\%$ coverage) and $\alpha = 0.05$ (Nominal $95\%$ coverage).
* **Base Regressors:** Asymmetric pinball loss quantile estimators $\hat{q}_{\alpha/2}$ and $\hat{q}_{1 - \alpha/2}$.
* **Calibration Protocol:** Nonconformity scores $E_i = \max(\hat{q}_{\alpha/2}(\mathbf{x}_i) - y_i, y_i - \hat{q}_{1-\alpha/2}(\mathbf{x}_i))$ on calibration partition.
* **Quantile Extraction:** Exact order statistic using `method='higher'`.
* **Reported Metrics:** Prediction Interval Coverage Probability (PICP), Mean Prediction Interval Width (MPIW), Winkler Score, Pinball Loss.

---

## 7. Sensor Degradation Stress-Testing Protocol

Four realistic operational failure scenarios are simulated on the target test partition:
1. **Scenario A (Random Dropout):** MCAR at rates $\eta \in \{10\%, 30\%, 50\%\}$. Missing values are causally forward-filled from the last valid observation (`ffill`).
2. **Scenario B (Contiguous Outage):** Continuous blackout blocks of duration $\Delta t \in \{1\text{h}, 3\text{h}, 6\text{h}, 12\text{h}\}$ simulating localized communication link drops.
3. **Scenario C (Sensor Hardware Failure):** Total pyranometer failure (complete GHI/DNI loss).
4. **Scenario D (Sensor Drift & Sticking):** Sensor stuck-at-value and linear calibration drift ($\pm 15\%$).
* **Replication:** Each scenario is repeated over 5 fixed random seeds to report mean performance and variance.

---

## 8. Statistical Significance Protocol

* **Diebold-Mariano Test:** Evaluates whether model prediction errors are statistically distinct from baselines.
* **Autocorrelation Correction:** Newey-West HAC spectral density estimator with Bartlett kernel.
* **Confidence Intervals:** 24-hour stationary block bootstrap with $B = 1,000$ resamples for $95\%$ confidence bounds on nMAE and nRMSE.
* **P-Value Reporting:** Exact values formatted as scientific notation (e.g., $p < 10^{-4}$), avoiding $p = 0.0000$.
