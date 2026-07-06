# Data-Efficient and Uncertainty-Aware Cross-Site Solar PV Power Forecasting Under Temporal and Sensor-Availability Shifts

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research: In Progress](https://img.shields.io/badge/Status-Research%20Phase%200-orange.svg)]()

## Abstract & Research Overview
This research develops a data-efficient, uncertainty-calibrated transfer learning framework for solar photovoltaic (PV) power forecasting across geographically and climatically distinct sites. Using Conformalized Quantile Regression (CQR) and few-shot domain adaptation, we examine cross-site generalization limits, minimal target adaptation data budgets, and model robustness under severe sensor-availability degradation.

## Directory Structure
```
├── configs/               # Experiment and data pipeline YAML configurations
├── data/                  # Data directories (raw, interim, processed) - gitignored
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/             # Numbered exploratory and prototyping Jupyter notebooks
├── src/                   # Production modular Python packages
│   ├── data/              # Fetching and loading raw feeds
│   ├── preprocessing/     # Physical filtering, imputation, and timezone alignment
│   ├── features/          # Sun-position geometry (pvlib), cyclic and lag features
│   ├── models/            # Baselines, tree ensembles (XGB/LGBM), and deep networks
│   ├── adaptation/        # Few-shot fine-tuning and domain adaptation modules
│   ├── uncertainty/       # Quantile regression and conformal calibration (MAPIE)
│   ├── evaluation/        # Point and probabilistic metrics (nMAE, nRMSE, PICP, MPIW)
│   └── visualization/     # Publication-grade plotting styles
├── experiments/           # Run logs, hyperparameter trials, and serialized models
├── figures/               # High-resolution vector figures (PDF/SVG/PNG 300+ DPI)
├── tables/                # LaTeX and CSV evaluation matrices
├── paper/                 # Journal manuscript, bibliography (.bib), and supplementary files
└── tests/                 # Unit tests for data leakage prevention and physical validity
```

## Reproducibility & Setup
Please refer to the Phase 0 report and `requirements.txt` for full setup instructions.
