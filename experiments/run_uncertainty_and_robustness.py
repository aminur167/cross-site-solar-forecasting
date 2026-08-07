"""
Experiment 4 & 5: Conformalized Quantile Regression (CQR) and Sensor-Availability Robustness
Evaluates RQ3 (Uncertainty Calibration under Domain Shift) and Robustness to Sensor Failures.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.uncertainty.conformal_predictor import ConformalizedQuantileRegressor
from src.uncertainty.sensor_perturbation import SensorPerturbationEngine
from src.evaluation.metrics import calculate_point_metrics, calculate_uncertainty_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def run_uncertainty_and_robustness():
    path_a = "data/processed/site_a_features.csv"
    path_b = "data/processed/site_b_features.csv"

    logging.info("Loading feature matrices...")
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

    # Split Site A (Source) into Train (70%) and Calibration (30%)
    n_a = len(df_a)
    train_a = df_a.iloc[:int(n_a * 0.70)]
    cal_a = df_a.iloc[int(n_a * 0.70):]

    # Site B (Target) holdout test set (final 20%)
    n_b = len(df_b)
    target_pool_b = df_b.iloc[:int(n_b * 0.80)]
    target_test_b = df_b.iloc[int(n_b * 0.80):]

    # Target 14-day adaptation calibration subset
    target_cal_14d = target_pool_b.iloc[:14 * 24] # ~14 days

    X_train_a, y_train_a = train_a[feature_cols], train_a[target_col]
    X_cal_a, y_cal_a = cal_a[feature_cols], cal_a[target_col]
    X_test_b, y_test_b = target_test_b[feature_cols], target_test_b[target_col].values

    logging.info(f"Training CQR base models on Source Site A ({len(X_train_a)} samples)...")
    cqr = ConformalizedQuantileRegressor(alpha=0.10)
    cqr.fit(X_train_a, y_train_a)

    results = {
        "Uncertainty_Calibration_Regimes": {},
        "Sensor_Availability_Stress_Test": {}
    }

    # -------------------------------------------------------------
    # 1. EVALUATE UNCERTAINTY QUANTIFICATION (RQ3)
    # -------------------------------------------------------------
    print("\n" + "="*95)
    print("EXPERIMENT 4: UNCERTAINTY QUANTIFICATION & CONFORMAL CALIBRATION (NOMINAL 90% INTERVAL)")
    print(f"Target Test Set: Site B Holdout ({len(X_test_b)} daytime hours)")
    print("="*95)
    print(f"{'Calibration Regime':<32} | {'PICP (%)':<10} | {'MPIW':<10} | {'Winkler Score':<14} | {'Coverage OK?':<12}")
    print("-" * 95)

    # Regime 1: Raw Uncalibrated Quantile Regression (Baseline)
    lower_raw, upper_raw = cqr.predict_intervals(X_test_b, apply_conformal=False)
    metrics_raw = calculate_uncertainty_metrics(y_true=y_test_b, lower_bound=lower_raw, upper_bound=upper_raw, alpha=0.10)
    print(f"{'Raw Quantile Regression (No CP)':<32} | {metrics_raw['PICP_pct']:<10.2f} | {metrics_raw['MPIW']:<10.4f} | {metrics_raw['Winkler_Score']:<14.4f} | {str(metrics_raw['Coverage_Guaranteed']):<12}")
    results["Uncertainty_Calibration_Regimes"]["Raw_Quantile_Regression"] = metrics_raw

    # Regime 2: Source-Calibrated CQR (Calibrated on Site A, tested on Site B)
    cqr.calibrate(X_cal_a, y_cal_a)
    lower_source_cqr, upper_source_cqr = cqr.predict_intervals(X_test_b, apply_conformal=True)
    metrics_source_cqr = calculate_uncertainty_metrics(y_true=y_test_b, lower_bound=lower_source_cqr, upper_bound=upper_source_cqr, alpha=0.10)
    print(f"{'Source-Calibrated CQR':<32} | {metrics_source_cqr['PICP_pct']:<10.2f} | {metrics_source_cqr['MPIW']:<10.4f} | {metrics_source_cqr['Winkler_Score']:<14.4f} | {str(metrics_source_cqr['Coverage_Guaranteed']):<12}")
    results["Uncertainty_Calibration_Regimes"]["Source_Calibrated_CQR"] = metrics_source_cqr

    # Regime 3: Target-Adapted CQR (Calibrated locally on Target 14-day budget)
    cqr_target = ConformalizedQuantileRegressor(alpha=0.10)
    cqr_target.fit(X_train_a, y_train_a)
    cqr_target.calibrate(target_cal_14d[feature_cols], target_cal_14d[target_col])
    lower_target_cqr, upper_target_cqr = cqr_target.predict_intervals(X_test_b, apply_conformal=True)
    metrics_target_cqr = calculate_uncertainty_metrics(y_true=y_test_b, lower_bound=lower_target_cqr, upper_bound=upper_target_cqr, alpha=0.10)
    print(f"{'Target-Adapted CQR (14-day local)':<32} | {metrics_target_cqr['PICP_pct']:<10.2f} | {metrics_target_cqr['MPIW']:<10.4f} | {metrics_target_cqr['Winkler_Score']:<14.4f} | {str(metrics_target_cqr['Coverage_Guaranteed']):<12}")
    results["Uncertainty_Calibration_Regimes"]["Target_Adapted_CQR"] = metrics_target_cqr
    print("="*95 + "\n")

    # -------------------------------------------------------------
    # 2. EVALUATE SENSOR AVAILABILITY SHIFTS & STRESS-TESTS
    # -------------------------------------------------------------
    print("\n" + "="*95)
    print("EXPERIMENT 5: SENSOR-AVAILABILITY DEGRADATION & STRESS-TESTING")
    print(f"Target Test Set: Site B Holdout ({len(X_test_b)} daytime hours)")
    print("="*95)
    print(f"{'Degradation Scenario':<32} | {'nMAE (%)':<10} | {'nRMSE (%)':<10} | {'PICP (%)':<10} | {'MPIW':<10}")
    print("-" * 95)

    perturb_engine = SensorPerturbationEngine(random_seed=42)
    scenarios = [
        ("Complete Telemetry (100% Sensors)", 0.0, None),
        ("Moderate Dropout (10% Missing)", 0.10, None),
        ("Severe Dropout (30% Missing)", 0.30, None),
        ("Extreme Dropout (50% Missing)", 0.50, None),
        ("Total Solar Irradiance Outage (No GHI)", -1.0, "ghi")
    ]

    for name, rate, ch_keyword in scenarios:
        if ch_keyword is not None:
            X_pert = perturb_engine.simulate_channel_outage(X_test_b, channel_keyword=ch_keyword)
        else:
            X_pert = perturb_engine.inject_random_missingness(X_test_b, missing_rate=rate)

        # Point forecast (median of lower/upper or point predictor)
        low, high = cqr_target.predict_intervals(X_pert, apply_conformal=True)
        point_preds = (low + high) / 2.0

        p_metrics = calculate_point_metrics(y_true=y_test_b, y_pred=point_preds, capacity=1.0)
        u_metrics = calculate_uncertainty_metrics(y_true=y_test_b, lower_bound=low, upper_bound=high, alpha=0.10)

        print(f"{name:<32} | {p_metrics['nMAE_pct']:<10.2f} | {p_metrics['nRMSE_pct']:<10.2f} | {u_metrics['PICP_pct']:<10.2f} | {u_metrics['MPIW']:<10.4f}")
        results["Sensor_Availability_Stress_Test"][name] = {
            "point_metrics": p_metrics,
            "uncertainty_metrics": u_metrics
        }

    print("="*95 + "\n")

    out_file = "experiments/results/uncertainty_robustness_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    logging.info(f"Uncertainty & Robustness results saved to -> {out_file}")

if __name__ == "__main__":
    run_uncertainty_and_robustness()
