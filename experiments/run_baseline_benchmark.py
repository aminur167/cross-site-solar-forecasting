"""
Experiment 1: Baseline and Classical ML Benchmark on Site A (Within-Site)
Evaluates Persistence, Seasonal Naive, Ridge, Random Forest, and Gradient Boosting.
Protocol: Chronological Walk-Forward Train/Val/Test Split (70% / 15% / 15%).
"""

import os
import sys
import json

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import logging

from src.models.baselines import PersistenceModel, SeasonalNaiveModel, RidgeRegressionBaseline
from src.models.classical_ml import SolarRandomForest, SolarGradientBoosting
from src.evaluation.metrics import calculate_point_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def run_benchmark():
    print("INSIDE run_benchmark()", flush=True)
    data_path = "data/processed/site_a_features.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Feature dataset not found at {data_path}. Run prepare_datasets.py first.")

    logging.info(f"Loading Site A features from {data_path}...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # Candidate feature columns (strictly available at time t)
    feature_cols = [
        'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos',
        'lag_0', 'lag_1', 'lag_2', 'lag_23',
        'rolling_mean_3h', 'rolling_std_3h',
        'ghi_lag_0', 'ghi_lag_1', 'ghi_diff_1h',
        'temp_lag_0', 'cloud_lag_0',
        'humidity_pct', 'wind_speed_kmh',
        'solar_elevation_deg', 'solar_azimuth_deg'
    ]
    # Filter only available columns
    feature_cols = [c for c in feature_cols if c in df.columns]
    feature_cols += [c for c in df.columns if c.endswith('_was_missing')]
    target_col = 'target_1h'

    logging.info(f"Features used ({len(feature_cols)}): {feature_cols}")

    # Chronological Split (No Shuffling!)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    logging.info(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    logging.info(f"Test period: {test_df.index.min()} to {test_df.index.max()}")

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val, y_val = val_df[feature_cols], val_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    # Models dictionary
    models = {
        "Persistence": PersistenceModel(target_lag_col='lag_0'),
        "Seasonal_Naive_24h": SeasonalNaiveModel(seasonal_lag_col='lag_23'),
        "Ridge_Regression": RidgeRegressionBaseline(alpha=10.0),
        "Random_Forest": SolarRandomForest(n_estimators=100, max_depth=12),
        "Gradient_Boosting": SolarGradientBoosting(max_iter=150, learning_rate=0.05, max_depth=8)
    }

    results = {}
    y_test_arr = y_test.values
    is_daytime = test_df['target_is_daytime'].values.astype(bool)
    n_daytime = int(np.sum(is_daytime))
    logging.info(f"Test Set Total Hours: {len(test_df)}, Daytime Active Hours (solar_elevation > 0): {n_daytime}")

    # Persistence baseline for Skill Score
    persist_preds = models["Persistence"].predict(X_test)

    print("\n" + "="*95)
    print("EXPERIMENT 1 BENCHMARK RESULTS (SITE A - WITHIN-SITE TEST SET: DAYTIME PRIMARY)")
    print(f"Total Test Hours: {len(test_df)} | Daytime Active Hours: {n_daytime}")
    print("="*95)
    print(f"{'Model':<22} | {'Day nMAE(%)':<12} | {'Day nRMSE(%)':<12} | {'Day R2':<8} | {'All nMAE(%)':<12} | {'Day Skill':<10}")
    print("-" * 95)

    for name, model in models.items():
        if hasattr(model, "fit"):
            model.fit(X_train, y_train)

        preds = model.predict(X_test)
        
        # Primary: Daytime-only evaluation (physically meaningful)
        metrics_day = calculate_point_metrics(
            y_true=y_test_arr[is_daytime],
            y_pred=preds[is_daytime],
            y_persistence=persist_preds[is_daytime],
            capacity=1.0
        )
        
        # Secondary: All-hours evaluation (continuous time-series)
        metrics_all = calculate_point_metrics(
            y_true=y_test_arr,
            y_pred=preds,
            y_persistence=persist_preds,
            capacity=1.0
        )

        model_result = dict(metrics_day) # Top-level reflects primary daytime evaluation
        model_result["daytime"] = metrics_day
        model_result["all_hours"] = metrics_all
        results[name] = model_result

        print(f"{name:<22} | {metrics_day['nMAE_pct']:<12.2f} | {metrics_day['nRMSE_pct']:<12.2f} | {metrics_day['R2']:<8.4f} | {metrics_all['nMAE_pct']:<12.2f} | {metrics_day.get('Skill_Score_vs_Persist', 0.0):<10.4f}")

    print("="*95 + "\n")

    print("="*85 + "\n")

    # Save results to experiments/results/
    out_file = "experiments/results/baseline_benchmark_results.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    logging.info(f"Results serialized to -> {out_file}")

if __name__ == "__main__":
    print("ENTRYPOINT: Starting run_baseline_benchmark.py", flush=True)
    run_benchmark()
