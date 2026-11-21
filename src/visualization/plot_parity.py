"""
Figure 5: Publication-Grade Parity and Residual Plot
===================================================
Generates high-contrast scientific parity diagram (Actual vs Predicted)
with 1:1 identity line, residual histogram inset, and goodness-of-fit metrics.
Formatted for Elsevier Energy and AI double-column publication.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import r2_score, mean_absolute_error

# Publication styling
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'


def generate_parity_plot(
    y_true: np.ndarray = None,
    y_pred: np.ndarray = None,
    output_path: str = "figures/fig5_parity_plots.png",
    dpi: int = 300,
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # If data not provided directly, compute from REAL target holdout test set!
    if y_true is None or y_pred is None:
        path_a = "data/processed/site_a_features.csv"
        path_b = "data/processed/site_b_features.csv"
        if os.path.exists(path_a) and os.path.exists(path_b):
            import sys
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
            from src.models.classical_ml import SolarLightGBM
            
            df_a = pd.read_csv(path_a, index_col=0, parse_dates=True)
            df_b = pd.read_csv(path_b, index_col=0, parse_dates=True)
            df_a['target_timestamp'] = pd.to_datetime(df_a['target_timestamp'])
            df_b['target_timestamp'] = pd.to_datetime(df_b['target_timestamp'])
            
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
            feature_cols += [c for c in df_a.columns if c.endswith('_was_missing') and c in df_b.columns]
            target_col = 'target_1h'
            
            # Train LightGBM on Source Site A
            model = SolarLightGBM(n_estimators=150, learning_rate=0.05, max_depth=6)
            test_start = df_b.index[int(len(df_b) * 0.80)]
            source = df_a.loc[df_a['target_timestamp'] < test_start]
            source = source.iloc[:int(len(source) * 0.85)]
            model.fit(source[feature_cols], source[target_col])
            
            # Evaluate on Target Site B holdout test set (daytime active hours)
            n_b = len(df_b)
            target_test_b = df_b.iloc[int(n_b * 0.80):]
            daytime_mask = target_test_b['target_is_daytime'].values.astype(bool)
            
            y_true = target_test_b[target_col].values[daytime_mask]
            y_pred = model.predict(target_test_b[feature_cols])[daytime_mask]
        else:
            raise FileNotFoundError("Processed datasets not found. Run prepare_datasets.py first.")

    # Compute goodness-of-fit metrics
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    nmae_pct = mae * 100.0
    residuals = y_pred - y_true

    fig = plt.figure(figsize=(7.5, 6.5))
    gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Main parity scatter plot
    ax_main = fig.add_subplot(gs[0:3, 0:3])

    # 2D Hexbin density to avoid overplotting
    hb = ax_main.hexbin(
        y_true,
        y_pred,
        gridsize=45,
        cmap="Blues",
        mincnt=1,
        linewidths=0.2,
        edgecolors="none",
    )
    cb = fig.colorbar(hb, ax=ax_main, orientation="vertical", pad=0.03, shrink=0.85)
    cb.set_label("Sample Density", fontsize=9)

    # 1:1 Identity Line
    lims = [0.0, 1.15]
    ax_main.plot(lims, lims, color="#D9534F", linestyle="--", linewidth=1.8, label="1:1 Perfect Forecast", zorder=4)

    # +/- 10% Error Boundary Band
    ax_main.plot(lims, [x * 1.1 for x in lims], color="#F0AD4E", linestyle=":", linewidth=1.2, label=r"$\pm 10\%$ Tolerance Band")
    ax_main.plot(lims, [x * 0.9 for x in lims], color="#F0AD4E", linestyle=":", linewidth=1.2)

    ax_main.set_xlim(lims)
    ax_main.set_ylim(lims)
    ax_main.set_xlabel("Observed next-hour Capacity Factor", fontweight="bold")
    ax_main.set_ylabel(r"Zero-shot LightGBM Prediction", fontweight="bold")
    ax_main.set_title("Parity Diagnostic: Observed vs Predicted Solar Generation", fontsize=11, fontweight="bold", pad=8)
    ax_main.grid(True)
    ax_main.legend(loc="upper left", framealpha=0.9, fontsize=8.5)

    # Metric text box
    metric_text = (
        f"$R^2 = {r2:.4f}$\n"
        f"nMAE = {nmae_pct:.2f}%\n"
        f"Residual Mean = {np.mean(residuals):+.4f}\n"
        f"Residual Std = {np.std(residuals):.4f}"
    )
    ax_main.text(
        0.95,
        0.08,
        metric_text,
        transform=ax_main.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#CBD5E1", alpha=0.9),
    )

    # Marginal Top Histogram (True)
    ax_top = fig.add_subplot(gs[3, 0:3], sharex=ax_main)
    ax_top.hist(y_true, bins=40, color="#0284C7", alpha=0.7, edgecolor="white")
    ax_top.set_ylabel("Count", fontsize=8)
    ax_top.set_xlabel("True Generation Distribution", fontsize=8.5)
    ax_top.grid(True, alpha=0.2)

    # Marginal Right Histogram (Residuals)
    ax_right = fig.add_subplot(gs[0:3, 3])
    ax_right.hist(residuals, bins=35, orientation="horizontal", color="#0D9488", alpha=0.7, edgecolor="white")
    ax_right.axhline(0, color="#D9534F", linestyle="--", linewidth=1.2)
    ax_right.set_xlabel("Count", fontsize=8)
    ax_right.set_ylabel(r"Residual $(\hat{y}_t - y_t)$", fontsize=8.5)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Parity plot saved successfully to {output_path}")


if __name__ == "__main__":
    generate_parity_plot()
