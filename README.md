# Data-Efficient and Uncertainty-Aware Cross-Site Solar PV Power Forecasting Under Temporal and Sensor-Availability Shifts

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aminur167/cross-site-solar-forecasting/blob/main/notebooks/01_cross_site_solar_forecasting_demo.ipynb)
[![Target Journal: Energy and AI](https://img.shields.io/badge/Target%20Journal-Energy%20and%20AI%20(Elsevier)-orange.svg)]()
[![Status: Submittable Package](https://img.shields.io/badge/Status-Manuscript%20Ready-green.svg)]()

## Abstract & Research Overview
Accurate and reliable short-term solar photovoltaic (PV) power forecasting is essential for secure power grid integration, dynamic reserve allocation, and renewable trading. However, conventional supervised forecasters suffer from three operational vulnerabilities:
1. **Data Scarcity at Newly Commissioned Plants:** New installations lack historical telemetry for model training.
2. **Absence of Calibrated Uncertainty:** Point predictions provide no statistical confidence guarantees for grid operators.
3. **Sensor-Availability Shifts:** Remote meteorological telemetry frequently suffers from communication dropouts and pyranometer degradation.

This repository provides an open-access, reproducible framework coupling **Gradient Boosted Decision Trees (LightGBM/XGBoost)** with **Conformalized Quantile Regression (CQR)** for data-efficient transfer forecasting across independent multi-year 50 kWp commercial PV plants.

---

## Key Experimental Findings

| Research Question | Finding | Empirical Evidence |
|---|---|---|
| **RQ1: Cross-Site Transfer** | Zero-shot transfer from Source Site A to unseen Target Site B cuts error without local training. | $\text{nMAE} = 4.94\%$, Skill Score $=+0.4243$ vs. persistence. |
| **RQ2: Few-Shot Adaptation** | Training from scratch on 1-day newly commissioned data fails ($\text{nMAE} = 29.33\%$). Our transfer adaptation achieves an immediate **82.8% error reduction**. | $\text{nMAE} = 5.05\%$, Skill Score $=+0.4164$ with just 1 day of target telemetry! |
| **RQ3: Conformal Calibration** | CQR guarantees distribution-free coverage bounds on unseen target plants under domain shift. | Empirical coverage of **$92.95\%$** against nominal $90\%$ target. |
| **Robustness: Sensor Shifts** | Model tolerates up to 30% random telemetry dropout in the field. Pyranometer loss is the primary failure mode. | $\text{nMAE} \le 7.56\%$ and $\text{PICP} \ge 91.7\%$ under 30% sensor dropout. |
| **Statistical Superiority** | GBDTs outperform PyTorch LSTM & TCN on tabular features. | Diebold-Mariano test confirms GBDT superiority ($p < 0.001$). |

---

## Repository Directory Structure
```
├── configs/                     # Experiment and data pipeline configurations
├── data/                        # Data directories (raw, interim, processed)
│   ├── raw/                     # Original Romanian two-site telemetry (gitignored)
│   └── processed/               # Clean, leak-free feature matrices
├── figures/                     # Vector PDF and 300+ DPI PNG publication figures
│   ├── fig1_baseline_benchmark.*
│   ├── fig2_adaptation_frontier.*
│   ├── fig3_uncertainty_robustness.*
│   ├── fig4_shap_feature_importance.*
│   └── fig5_parity_plots.png    # Observed vs. predicted parity & residual marginals
├── tables/                      # Journal-formatted LaTeX tables (Tables 1 to 5)
├── paper/                       # Complete manuscript submission package
│   ├── manuscript/main.tex      # Full journal manuscript (Elsevier elsarticle format)
│   ├── supplementary/           # Mathematical proofs and hyperparameter tables
│   ├── cover_letter.tex         # Submission letter to Editor-in-Chief
│   ├── highlights.txt           # Official Elsevier highlights (<= 85 chars each)
│   ├── author_statements.txt    # CRediT, Competing interests, Data/Code availability
│   ├── references.bib           # Verified bibliography
│   ├── journal_selection_report.md
│   ├── peer_review_checklist.md
│   └── msc_application_positioning.md
├── notebooks/                   # Interactive demo notebook with Open in Colab badge
├── src/                         # Modular production Python package
│   ├── data/                    # Data preparation and audit
│   ├── preprocessing/           # Physical filtering and night-time segregation
│   ├── features/                # Solar geometry (pvlib), cyclic and lag features
│   ├── models/                  # Baselines, GBDTs, and PyTorch LSTM/TCN
│   ├── adaptation/              # Few-shot domain adaptation engine
│   ├── uncertainty/             # Conformalized Quantile Regression (CQR)
│   ├── evaluation/              # Point, uncertainty, and Diebold-Mariano tests
│   └── visualization/           # Publication-grade plotting styles
├── experiments/                 # Execution scripts and serialized JSON benchmark logs
├── tests/                       # Unit tests for physical bounds and leakage prevention
├── run_full_pipeline.py         # Single-command master orchestrator (runs in ~58s)
└── requirements.txt             # Environment dependencies
```

---

## Quickstart & Reproducibility

### 1. Hardware Requirements & Environment Setup
- **Minimum:** 4-Core CPU, 8 GB RAM, Python 3.10+ (Runs all GBDT & Conformal experiments in < 30s).
- **Recommended (Deep Learning):** 8-Core CPU or NVIDIA GPU with CUDA 12.0+ (PyTorch LSTM/TCN training).
- **Google Colab:** Compatible with free T4 GPU tier. Click the [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aminur167/cross-site-solar-forecasting/blob/main/notebooks/01_cross_site_solar_forecasting_demo.ipynb) badge above.

Clone the repository and install dependencies:
```bash
git clone https://github.com/aminur167/cross-site-solar-forecasting.git
cd cross-site-solar-forecasting
pip install -r requirements.txt
```

### 2. Execute Entire Pipeline in One Command
Run all data cleaning, benchmarks, transfer adaptation, conformal calibration, deep learning, SHAP attribution, parity diagnostics, and unit tests:
```bash
# Full execution (~58 seconds on standard CPU):
python run_full_pipeline.py

# Fast CPU-only run (skips PyTorch sequence training):
python run_full_pipeline.py --skip-dl

# Inspect available CLI flags:
python run_full_pipeline.py --help
```

### 3. Launch Interactive Research Dashboard (Localhost)
Launch the interactive Streamlit dashboard to explore real-time telemetry, model benchmarks, conformal prediction intervals, and download the 1-click Overleaf publication package:
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 4. Run Unit Tests Only
Verify physical boundary constraints, non-leaking feature transformations, and conformal coverage bounds:
```bash
python -m unittest tests/test_pipeline.py -v
```

---

## Citation & Contact
If you use this work or codebase in your research, please cite:
```bibtex
@article{rahman2026crosssite,
  title={Data-Efficient and Uncertainty-Calibrated Cross-Site Solar PV Power Forecasting Under Temporal and Sensor-Availability Shifts},
  author={Rahman, Aminur},
  journal={Submitted to Energy and AI (Elsevier)},
  year={2026},
  url={https://github.com/aminur167/cross-site-solar-forecasting}
}
```
**Author:** Aminur Rahman (HSTU, Bangladesh)  
**Email:** [aminurhstu23@gmail.com](mailto:aminurhstu23@gmail.com) | **GitHub:** [@aminur167](https://github.com/aminur167)
