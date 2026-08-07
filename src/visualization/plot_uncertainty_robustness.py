"""
Publication-Quality Visualization for Experiment 4 & 5:
Uncertainty Quantification (CQR) and Sensor-Availability Robustness Curves.
Generates Figure 3 (PNG 300 DPI and PDF vector format).
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

def generate_uncertainty_robustness_figure():
    res_path = "experiments/results/uncertainty_robustness_results.json"
    with open(res_path, "r") as f:
        data = json.load(f)

    # Style configuration
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'grid.alpha': 0.3,
        'axes.grid': True
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

    # 1. Uncertainty Calibration Comparison (PICP vs MPIW)
    cal_data = data["Uncertainty_Calibration_Regimes"]
    regimes = list(cal_data.keys())
    regime_labels = ["Raw Quantile\n(No CP)", "Source CQR\n(Site A Cal)", "Target CQR\n(14d Cal)"]
    picp_vals = [cal_data[r]["PICP_pct"] for r in regimes]
    mpiw_vals = [cal_data[r]["MPIW"] * 100.0 for r in regimes] # express as % capacity

    x = np.arange(len(regimes))
    width = 0.35

    rects1 = ax1.bar(x - width/2, picp_vals, width, label='PICP Coverage (%)', color='#2b5c8f', edgecolor='black', linewidth=0.8)
    rects2 = ax1.bar(x + width/2, mpiw_vals, width, label='MPIW Width (% Cap)', color='#7570b3', edgecolor='black', linewidth=0.8)

    ax1.axhline(90.0, color='red', linestyle='--', linewidth=1.5, label='Nominal 90% Target')
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('(a) Conformal Calibration Across Regimes (Site B Test)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(regime_labels)
    ax1.set_ylim(0, 105)
    ax1.legend(frameon=True, loc='lower right')

    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(f'{h:.1f}%', (rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(f'{h:.1f}%', (rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

    # 2. Sensor-Availability Degradation Curves
    stress_data = data["Sensor_Availability_Stress_Test"]
    scenarios = list(stress_data.keys())
    scenario_labels = ['Complete\n(0% Miss)', '10% Drop', '30% Drop', '50% Drop', 'Total GHI\nOutage']
    nmae_vals = [stress_data[s]["point_metrics"]["nMAE_pct"] for s in scenarios]
    picp_stress = [stress_data[s]["uncertainty_metrics"]["PICP_pct"] for s in scenarios]

    x2 = np.arange(len(scenarios))
    ax2.plot(x2, nmae_vals, 'o-', color='#d95f02', linewidth=2.4, markersize=8, label='Forecasting Error (nMAE %)')
    ax2.set_ylabel('Forecasting Error (nMAE %)', color='#d95f02')
    ax2.tick_params(axis='y', labelcolor='#d95f02')
    ax2.set_ylim(6.0, 11.0)

    # Twin axis for PICP coverage
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x2, picp_stress, 's--', color='#1b9e77', linewidth=2.4, markersize=8, label='Empirical Coverage (PICP %)')
    ax2_twin.axhline(90.0, color='red', linestyle=':', linewidth=1.2, label='Nominal 90% Bound')
    ax2_twin.set_ylabel('Prediction Interval Coverage (PICP %)', color='#1b9e77')
    ax2_twin.tick_params(axis='y', labelcolor='#1b9e77')
    ax2_twin.set_ylim(80.0, 96.0)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(scenario_labels)
    ax2.set_title('(b) Resilience Under Sensor Failures & Telemetry Dropout')

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True)

    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    png_path = "figures/fig3_uncertainty_robustness.png"
    pdf_path = "figures/fig3_uncertainty_robustness.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"Figure 3 saved:\n -> {png_path}\n -> {pdf_path}")

if __name__ == "__main__":
    generate_uncertainty_robustness_figure()
