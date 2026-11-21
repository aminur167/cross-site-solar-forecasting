"""
Explainability and Physics Attribution Module using SHAP (Shapley Additive Explanations)
Analyzes feature importance, non-linear interactions, and physical drivers of solar generation forecasts.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.classical_ml import SolarLightGBM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def run_shap_analysis():
    path_a = "data/processed/site_a_features.csv"
    path_b = "data/processed/site_b_features.csv"

    logging.info("Loading feature matrices for SHAP attribution...")
    df_a = pd.read_csv(path_a, index_col=0, parse_dates=True)
    df_b = pd.read_csv(path_b, index_col=0, parse_dates=True)
    df_a['target_timestamp'] = pd.to_datetime(df_a['target_timestamp'])

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

    # Train model on Source Site A
    model = SolarLightGBM(n_estimators=150, learning_rate=0.05, max_depth=6)
    # Use source labels available before the target holdout begins.
    n_b = len(df_b)
    target_test_b = df_b.iloc[int(n_b * 0.80):]
    source = df_a.loc[df_a['target_timestamp'] < target_test_b.index.min()]
    source = source.iloc[:int(len(source) * 0.85)]
    model.fit(source[feature_cols], source[target_col])
    X_test = target_test_b[feature_cols]

    logging.info(f"Computing TreeExplainer SHAP values on {len(X_test)} target test samples...")
    explainer = shap.TreeExplainer(model.model)
    shap_values = explainer.shap_values(X_test)

    # Calculate Mean Absolute SHAP per feature
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Mean_Abs_SHAP": mean_abs_shap
    }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

    # Relative importance percentage
    total_shap = importance_df["Mean_Abs_SHAP"].sum()
    importance_df["Relative_Importance_pct"] = (importance_df["Mean_Abs_SHAP"] / total_shap) * 100.0

    print("\n" + "="*70)
    print("GLOBAL SHAP FEATURE IMPORTANCE & ATTRIBUTION (TARGET SITE B)")
    print("="*70)
    print(f"{'Rank':<5} | {'Feature Name':<22} | {'Mean |SHAP|':<14} | {'Relative (%)':<12}")
    print("-" * 70)
    for idx, row in importance_df.iterrows():
        print(f"{idx+1:<5} | {row['Feature']:<22} | {row['Mean_Abs_SHAP']:<14.5f} | {row['Relative_Importance_pct']:<12.2f}%")
    print("="*70 + "\n")

    # Save to JSON
    out_dir = "experiments/results"
    os.makedirs(out_dir, exist_ok=True)
    importance_df.to_json(os.path.join(out_dir, "shap_importance_results.json"), orient="records", indent=4)

    # Generate Figure 4: Horizontal Bar Chart of Feature Importance
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'grid.alpha': 0.3,
        'axes.grid': True
    })

    top_n = 12
    plot_df = importance_df.head(top_n).iloc[::-1] # Ascending for horizontal bar

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    colors = plt.cm.viridis(np.linspace(0.25, 0.85, top_n))

    bars = ax.barh(plot_df["Feature"], plot_df["Relative_Importance_pct"], color=colors, edgecolor='black', linewidth=0.8)

    ax.set_xlabel('Relative Predictive Contribution (%)')
    ax.set_title('Figure 4: Global SHAP Feature Attribution Across Target Solar Site')
    ax.set_xlim(0, max(plot_df["Relative_Importance_pct"]) * 1.18)

    for bar in bars:
        w = bar.get_width()
        ax.annotate(f'{w:.1f}%', (w, bar.get_y() + bar.get_height()/2),
                    xytext=(5, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    png_path = "figures/fig4_shap_feature_importance.png"
    pdf_path = "figures/fig4_shap_feature_importance.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.close()

    logging.info(f"SHAP figures generated:\n -> {png_path}\n -> {pdf_path}")

    # Generate LaTeX Table 4
    tex_path = "tables/table4_feature_importance.tex"
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Global SHAP feature importance rankings and relative predictive contributions for cross-site solar PV forecasting.}\n")
        f.write("\\label{tab:feature_importance}\n")
        f.write("\\begin{tabular}{clcc}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Rank} & \\textbf{Feature Description} & \\textbf{Mean $|\\text{SHAP}|$ Value} & \\textbf{Relative Contribution (\\%)} \\\\\n")
        f.write("\\hline\n")
        for idx, row in importance_df.head(10).iterrows():
            f.write(f"{idx+1} & \\texttt{{{row['Feature']}}} & {row['Mean_Abs_SHAP']:.4f} & {row['Relative_Importance_pct']:.2f}\\% \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    logging.info(f"LaTeX Table 4 generated -> {tex_path}")

if __name__ == "__main__":
    run_shap_analysis()
