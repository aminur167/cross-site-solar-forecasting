"""
Publication-Quality Visualization for Experiment 2 & 3:
Cross-Site Zero-Shot Transfer and Few-Shot Target Adaptation Frontier.
Generates Figure 2 (PNG 300 DPI and PDF vector format).
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

def generate_adaptation_figure():
    res_path = "experiments/results/cross_site_adaptation_results.json"
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

    budgets = [1, 3, 7, 14]
    target_only_nmae = []
    adapted_nmae = []

    for k in budgets:
        entry = data["Zero_Shot_and_Adaptation"][f"k_{k}_days"]
        target_only_nmae.append(entry["target_only_metrics"]["nMAE_pct"])
        adapted_nmae.append(entry["transfer_adapted_metrics"]["nMAE_pct"])

    zero_shot_nmae = data["Zero_Shot_and_Adaptation"]["k_0_days"]["transfer_adapted_metrics"]["nMAE_pct"]
    oracle_nmae = data["Target_Oracle_Full"]["nMAE_pct"]
    persist_nmae = data["Persistence_Baseline"]["nMAE_pct"]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    # Plot curves
    ax.plot(budgets, target_only_nmae, 'o--', color='#d95f02', linewidth=2.2, markersize=8, label='Target-Only (Train from Scratch)')
    ax.plot(budgets, adapted_nmae, 's-', color='#2b5c8f', linewidth=2.5, markersize=8, label='Transfer-Adapted (Source + Target Budget)')

    # Reference benchmark lines
    ax.axhline(zero_shot_nmae, color='#7570b3', linestyle=':', linewidth=1.8, label=f'Zero-Shot Transfer (0 days, {zero_shot_nmae:.2f}%)')
    ax.axhline(oracle_nmae, color='#1b9e77', linestyle='-.', linewidth=1.8, label=f'Target Oracle (Full ~315 days, {oracle_nmae:.2f}%)')
    ax.axhline(persist_nmae, color='gray', linestyle='--', linewidth=1.2, label=f'Persistence Baseline ({persist_nmae:.2f}%)')

    # Value annotations on curves
    for x, y in zip(budgets, target_only_nmae):
        ax.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=9, color='#d95f02', fontweight='bold')
    for x, y in zip(budgets, adapted_nmae):
        ax.annotate(f'{y:.2f}%', (x, y), textcoords="offset points", xytext=(0, -14), ha='center', fontsize=9, color='#2b5c8f', fontweight='bold')

    ax.set_xlabel('Target-Site Commissioning Budget $k$ (Calendar Days)')
    ax.set_ylabel('Normalized Mean Absolute Error (nMAE %)')
    ax.set_title('Figure 2: Few-Shot Adaptation Frontier & Cross-Site Generalization\n(Source: Site A $\\to$ Target: Site B 50 kWp Inverter)')
    ax.set_xticks(budgets)
    ax.set_xticklabels(['1 Day\n(~14h)', '3 Days\n(~42h)', '7 Days\n(1 Week)', '14 Days\n(2 Weeks)'])
    ax.set_ylim(2, 33)
    ax.legend(frameon=True, loc='upper right')

    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    png_path = "figures/fig2_adaptation_frontier.png"
    pdf_path = "figures/fig2_adaptation_frontier.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"Figure 2 saved:\n -> {png_path}\n -> {pdf_path}")

if __name__ == "__main__":
    generate_adaptation_figure()
