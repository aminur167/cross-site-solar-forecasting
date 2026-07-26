"""
Publication-Quality Visualization Script
Generates high-resolution vector (PDF) and 300 DPI raster (PNG) charts for Experiment 1 baselines.
Style configured according to IEEE/Elsevier guidelines.
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generate_benchmark_figure():
    results_path = "experiments/results/baseline_benchmark_results.json"
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Missing results file: {results_path}")

    with open(results_path, "r") as f:
        results = json.load(f)

    # Styling for Academic Publications (Elsevier / IEEE standard)
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'grid.alpha': 0.3,
        'axes.grid': True
    })

    models = list(results.keys())
    # Display-friendly names
    display_names = [m.replace("_", " ") for m in models]
    nmae = [results[m]["nMAE_pct"] for m in models]
    nrmse = [results[m]["nRMSE_pct"] for m in models]
    skill_scores = [results[m].get("Skill_Score_vs_Persist", 0.0) for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # 1. Error Comparison (nMAE and nRMSE)
    x = np.arange(len(models))
    width = 0.35

    color_nmae = '#2b5c8f'
    color_nrmse = '#d95f02'

    rects1 = ax1.bar(x - width/2, nmae, width, label='nMAE (%)', color=color_nmae, edgecolor='black', linewidth=0.8)
    rects2 = ax1.bar(x + width/2, nrmse, width, label='nRMSE (%)', color=color_nrmse, edgecolor='black', linewidth=0.8)

    ax1.set_ylabel('Normalized Error (%)')
    ax1.set_title('(a) Forecasting Errors on Unseen Test Horizon')
    ax1.set_xticks(x)
    ax1.set_xticklabels(display_names, rotation=25, ha='right')
    ax1.legend(frameon=True)
    ax1.set_ylim(0, max(nrmse) * 1.2)

    # Add value labels
    for rect in rects1:
        height = rect.get_height()
        ax1.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    for rect in rects2:
        height = rect.get_height()
        ax1.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    # 2. Skill Score Relative to Persistence
    colors_skill = ['#7570b3' if s >= 0 else '#e7298a' for s in skill_scores]
    bars = ax2.bar(x, skill_scores, width=0.5, color=colors_skill, edgecolor='black', linewidth=0.8)

    ax2.axhline(0, color='black', linewidth=1.0, linestyle='--')
    ax2.set_ylabel('Skill Score (vs. Persistence Baseline)')
    ax2.set_title('(b) Relative Skill Improvement')
    ax2.set_xticks(x)
    ax2.set_xticklabels(display_names, rotation=25, ha='right')
    ax2.set_ylim(-0.4, 0.55)

    for bar in bars:
        h = bar.get_height()
        offset = 3 if h >= 0 else -12
        ax2.annotate(f'{h:+.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, offset), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    png_path = "figures/fig1_baseline_benchmark.png"
    pdf_path = "figures/fig1_baseline_benchmark.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close()

    print(f"Publication-grade figures generated:\n -> {png_path}\n -> {pdf_path}")

if __name__ == "__main__":
    generate_benchmark_figure()
