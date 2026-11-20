"""
Figure 6: Conformal Reliability Diagram and Sharpness Curve
===========================================================
Generates publication-quality calibration diagnostic:
- Subplot 1: Empirical vs. Nominal Coverage across multiple confidence levels
- Subplot 2: Prediction Interval Width (Sharpness) trade-off curve
"""

import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0


import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.uncertainty.conformal_predictor import ConformalizedQuantileRegressor

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0


def generate_reliability_diagram(output_path: str = "figures/fig6_reliability_diagram.png", dpi: int = 300):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    path_a = "data/processed/site_a_features.csv"
    path_b = "data/processed/site_b_features.csv"
    if not (os.path.exists(path_a) and os.path.exists(path_b)):
        raise FileNotFoundError("Processed datasets missing. Run prepare_datasets.py first.")

    df_a = pd.read_csv(path_a, index_col=0, parse_dates=True)
    df_b = pd.read_csv(path_b, index_col=0, parse_dates=True)

    feature_cols = [
        'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos',
        'lag_0', 'lag_1', 'lag_2', 'lag_23',
        'rolling_mean_3h', 'rolling_std_3h',
        'ghi_lag_0', 'ghi_lag_1', 'ghi_diff_1h',
        'temp_lag_0', 'cloud_lag_0',
        'humidity_pct', 'wind_speed_kmh',
        'solar_elevation_deg', 'solar_azimuth_deg'
    ]
    feature_cols = [c for c in feature_cols if c in df_a.columns and c in df_b.columns]
    target_col = 'target_1h'

    n_a = len(df_a)
    X_train_a = df_a[feature_cols].iloc[:int(n_a * 0.70)]
    y_train_a = df_a[target_col].iloc[:int(n_a * 0.70)]

    n_b = len(df_b)
    target_pool_b = df_b.iloc[:int(n_b * 0.80)]
    target_cal_14d = target_pool_b.iloc[:14 * 24] # 14-day calibration budget
    target_test_b = df_b.iloc[int(n_b * 0.80):]

    daytime_mask = target_test_b['is_daytime'].values == True
    X_test_b = target_test_b[feature_cols]
    y_test_b = target_test_b[target_col].values

    nominal_levels = np.array([0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95])
    empirical_cqr = []
    uncalibrated = []
    interval_widths = []

    for nom in nominal_levels:
        alpha = float(1.0 - nom)
        cqr = ConformalizedQuantileRegressor(alpha=alpha, n_estimators=60)
        cqr.fit(X_train_a, y_train_a)

        # Uncalibrated quantile intervals
        l_raw, u_raw = cqr.predict_intervals(X_test_b, apply_conformal=False)
        cov_raw = np.mean((y_test_b[daytime_mask] >= l_raw[daytime_mask]) & (y_test_b[daytime_mask] <= u_raw[daytime_mask]))
        uncalibrated.append(cov_raw)

        # Calibrated CQR intervals (using target 14-day calibration)
        cqr.calibrate(target_cal_14d[feature_cols], target_cal_14d[target_col])
        l_cqr, u_cqr = cqr.predict_intervals(X_test_b, apply_conformal=True)
        cov_cqr = np.mean((y_test_b[daytime_mask] >= l_cqr[daytime_mask]) & (y_test_b[daytime_mask] <= u_cqr[daytime_mask]))
        empirical_cqr.append(cov_cqr)
        interval_widths.append(np.mean(u_cqr[daytime_mask] - l_cqr[daytime_mask]))

    empirical_cqr = np.array(empirical_cqr)
    uncalibrated = np.array(uncalibrated)
    interval_widths = np.array(interval_widths)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=dpi)

    # 1. Reliability Plot
    ax1.plot([0.45, 1.0], [0.45, 1.0], color="#94A3B8", linestyle="--", linewidth=1.5, label="Ideal Calibration ($y = x$)")
    ax1.plot(nominal_levels, empirical_cqr, marker="o", color="#0284C7", linewidth=2.0, markersize=6, label="Proposed CQR (Calibrated)")
    ax1.plot(nominal_levels, uncalibrated, marker="s", color="#EF4444", linestyle=":", linewidth=1.5, markersize=5, label="Raw Quantile Reg (Shifted)")

    ax1.fill_between(nominal_levels, nominal_levels, empirical_cqr, color="#0284C7", alpha=0.15, label="Coverage Bound")
    ax1.set_xlim([0.45, 1.01])
    ax1.set_ylim([0.40, 1.02])
    ax1.set_xlabel("Nominal Coverage Level $(1 - \\alpha)$", fontweight="bold")
    ax1.set_ylabel("Empirical Daytime Coverage (PICP)", fontweight="bold")
    ax1.set_title("A. Conformal Reliability Calibration", fontsize=11, fontweight="bold", pad=8)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    # 2. Sharpness Curve
    ax2.plot(nominal_levels * 100, interval_widths, marker="^", color="#D97706", linewidth=2.0, markersize=6)
    ax2.axvline(90.0, color="#0284C7", linestyle="--", alpha=0.7, label=r"Operational Target ($90\%$)")
    ax2.set_xlabel("Nominal Confidence Level (%)", fontweight="bold")
    ax2.set_ylabel("Mean Interval Width (MPIW, CF)", fontweight="bold")
    ax2.set_title("B. Interval Sharpness Frontier", fontsize=11, fontweight="bold", pad=8)
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Reliability diagram dynamically generated and saved to {output_path}")


if __name__ == "__main__":
    generate_reliability_diagram()
