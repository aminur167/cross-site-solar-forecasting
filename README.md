# Data-Efficient and Uncertainty-Aware Cross-Site Solar PV Power Forecasting Under Temporal and Sensor-Availability Shifts

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aminur167/cross-site-solar-forecasting/blob/main/notebooks/01_cross_site_solar_forecasting_demo.ipynb)
[![Status: Research Draft](https://img.shields.io/badge/Status-Research%20Draft-orange.svg)]()

## Project overview
This repository contains a retrospective comparison of transfer and target-only models for two PV sites in southwestern Romania. The weather fields come from Open-Meteo historical estimates and were not available when the target hour began. The results therefore describe an offline experiment, not an operational forecast test.

---

## Key Experimental Findings

| Research Question | Finding | Empirical Evidence |
|---|---|---|
| **RQ1: Cross-Site Transfer** | Retrospective zero-shot transfer from Site A to Site B. | $\text{nMAE} = 6.93\%$ vs. persistence $14.14\%$ on target-aligned daytime holdout. |
| **RQ2: Few-Shot Adaptation** | One-day pooled adaptation scored slightly worse than zero-shot. | Adapted $6.97\%$ vs. scratch $28.06\%$ nMAE (75.1% reduction); zero-shot $6.93\%$. |
| **RQ3: Conformal Calibration** | Empirical split CQR coverage under temporal/domain shift; no guarantee is claimed. | 90% nominal: 88.20% raw, 90.27% source-calibrated, 94.51% target-calibrated daytime PICP. |
| **Robustness: Synthetic Feature Corruption** | Retrospective weather-feature perturbation; not field sensor fault evidence. | 30% feature-wise MCAR: mean nMAE 7.46%, mean PICP 92.78% (five seeds). |
| **Deep Learning Comparison** | A single-seed comparison; results are exploratory. | LightGBM 6.93%, LSTM 7.37%, TCN 7.69% daytime nMAE. |

---

## Repository Directory Structure
```
├── configs/                     # Experiment and data pipeline configurations
├── data/                        # Data directories (raw, interim, processed)
│   ├── raw/                     # Original Romanian two-site telemetry (gitignored)
│   └── processed/               # Clean, leak-free feature matrices
├── figures/                     # Regenerated figures for review
│   ├── fig1_baseline_benchmark.*
│   ├── fig2_adaptation_frontier.*
│   ├── fig3_uncertainty_robustness.*
│   ├── fig4_shap_feature_importance.*
│   └── fig5_parity_plots.png    # Observed vs. predicted parity & residual marginals
│   └── fig6_reliability_diagram.png
├── tables/                      # Regenerated LaTeX result tables
├── paper/                       # Manuscript draft and supporting notes
│   ├── manuscript/main.tex      # Full journal manuscript (Elsevier elsarticle format)
│   ├── supplementary/           # Supplementary methods
│   ├── cover_letter.tex         # Unapproved cover-letter draft
│   ├── highlights.txt           # Draft research highlights
│   ├── author_statements.txt    # Author statements requiring confirmation
│   ├── references.bib           # Citation-audited bibliography
│   ├── journal_selection_report.md
│   ├── peer_review_checklist.md
│   └── msc_application_positioning.md
├── notebooks/                   # Interactive demo notebook
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
├── run_full_pipeline.py         # Single-command master orchestrator (run time varies by environment)
└── requirements.txt             # Environment dependencies
```

---

## Quickstart & Reproducibility

### 1. Hardware Requirements & Environment Setup
- **Minimum:** 4-Core CPU, 8 GB RAM, Python 3.10+ (the installed package versions are not fully pinned).
- **Recommended (Deep Learning):** 8-Core CPU or NVIDIA GPU with CUDA 12.0+ (PyTorch LSTM/TCN training).
- **Google Colab:** Compatible with free T4 GPU tier. Click the [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aminur167/cross-site-solar-forecasting/blob/main/notebooks/01_cross_site_solar_forecasting_demo.ipynb) badge above.

Clone the repository and install dependencies:
```bash
git clone https://github.com/aminur167/cross-site-solar-forecasting.git
cd cross-site-solar-forecasting
pip install -r requirements.txt
```

### 2. Execute Entire Pipeline in One Command
Run preprocessing, model comparisons, interval evaluation, attribution, figures, tables and unit tests:
```bash
# Full pipeline:
python run_full_pipeline.py

# Fast CPU-only run (skips PyTorch sequence training):
python run_full_pipeline.py --skip-dl

# Inspect available CLI flags:
python run_full_pipeline.py --help
```

### 3. Launch Interactive Research Dashboard (Localhost)
Launch the Streamlit dashboard to explore bundled historical telemetry and current experiment results:
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 4. Run Unit Tests Only
Run the current unit tests for feature alignment, temporal gaps, data quality handling, and interval behavior:
```bash
python -m unittest tests/test_pipeline.py -v
```

---

## Dataset citation
The dataset citation, version and license are recorded in `data/raw/romanian_pv/CITATION.cff`. This code repository contains an unpublished research draft.
