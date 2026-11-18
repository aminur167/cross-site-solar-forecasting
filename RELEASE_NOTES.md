# Release Notes - v1.0.0 Research Benchmark & Manuscript Submission Package

**Release Tag:** `v1.0.0`  
**Date:** November 2026  
**Lead Author:** Aminur Rahman (HSTU, Bangladesh)  
**Target Journal:** *Energy and AI* (Elsevier, Q1, IF ~12.5)  

---

## 🌟 Highlights & Summary
This official v1.0.0 release packages the complete, fully reproducible research benchmark on **Data-Efficient and Uncertainty-Aware Cross-Site Solar PV Power Forecasting Under Temporal and Sensor-Availability Shifts**.

### 1. Key Methodological Contributions
- **Zero-Shot & Few-Shot Domain Adaptation:** Resolves data scarcity at newly commissioned PV sites, reducing forecast error by up to **82.8%** with just 1 day of target telemetry ($\text{nMAE} = 5.05\%$).
- **Finite-Sample Uncertainty Quantification:** Conformalized Quantile Regression (CQR) guarantees distribution-free coverage ($\text{PICP} = 92.95\%$ vs. 90.0% nominal) on unseen target installations.
- **Sensor-Availability Degradation Stress-Testing:** Demonstrates empirical robustness under up to 30% random telemetry dropout and quantifies physical failure modes under total pyranometer outage.
- **Diebold-Mariano Statistical Superiority:** Confirms Gradient Boosted Decision Trees statistically outperform deep sequence models (PyTorch LSTM & TCN) on solar feature tabular representations ($p < 0.001$).
- **Interpretability & Parity Diagnostics:** Game-theoretic SHAP tree attribution and Figure 5 parity diagnostics with marginal residual analysis.

### 2. Included Publication Assets
- `main.tex`: Formatted using official Elsevier `elsarticle` LaTeX template.
- `references.bib`: 15+ peer-reviewed citations from top-tier journals (*Energy and AI*, *Applied Energy*, *Solar Energy*, *IEEE Trans. Sust. Energy*).
- `tables/`: 5 publication-grade LaTeX tables (Tables 1 to 5).
- `figures/`: 5 publication-grade figures in PNG and vector formats.
- `paper_submission_package.zip`: 1-click Overleaf import archive containing all manuscript and asset files.
- `app.py`: High-contrast cyber-renewable Streamlit research dashboard running on localhost.
- `run_full_pipeline.py`: Master orchestrator with CLI flags running the entire experimental workflow in ~58 seconds.

---
**Repository:** https://github.com/aminur167/cross-site-solar-forecasting
