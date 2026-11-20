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
    X_test_b = target_test_b[feature_cols]
    y_test_b = target_test_b[target_col].values
    is_daytime_b = target_test_b['is_daytime'].values == True
    n_daytime_b = int(np.sum(is_daytime_b))

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
    print(f"Target Test Set: Site B Holdout ({len(X_test_b)} total hours, {n_daytime_b} daytime active hours)")
    print("="*95)
    print(f"{'Calibration Regime':<32} | {'Day PICP(%)':<12} | {'Day MPIW':<10} | {'Winkler':<10} | {'All PICP(%)':<12}")
    print("-" * 95)

    def evaluate_regime(name, lower_all, upper_all):
        day_u = calculate_uncertainty_metrics(y_true=y_test_b[is_daytime_b], lower_bound=lower_all[is_daytime_b], upper_bound=upper_all[is_daytime_b], alpha=0.10)
        all_u = calculate_uncertainty_metrics(y_true=y_test_b, lower_bound=lower_all, upper_bound=upper_all, alpha=0.10)
        reg_dict = dict(day_u)
        reg_dict["daytime"] = day_u
        reg_dict["all_hours"] = all_u
        results["Uncertainty_Calibration_Regimes"][name] = reg_dict
        print(f"{name:<32} | {day_u['PICP_pct']:<12.2f} | {day_u['MPIW']:<10.4f} | {day_u['Winkler_Score']:<10.4f} | {all_u['PICP_pct']:<12.2f}")
        return reg_dict

    # Regime 1: Raw Uncalibrated Quantile Regression (Baseline)
    lower_raw, upper_raw = cqr.predict_intervals(X_test_b, apply_conformal=False)
    evaluate_regime("Raw_Quantile_Regression", lower_raw, upper_raw)

    # Regime 2: Source-Calibrated CQR (Calibrated on Site A, tested on Site B)
    cqr.calibrate(X_cal_a, y_cal_a)
    lower_source_cqr, upper_source_cqr = cqr.predict_intervals(X_test_b, apply_conformal=True)
    evaluate_regime("Source_Calibrated_CQR", lower_source_cqr, upper_source_cqr)

    # Regime 3: Target-Adapted CQR (Calibrated locally on Target 14-day budget)
    cqr_target = ConformalizedQuantileRegressor(alpha=0.10)
    cqr_target.fit(X_train_a, y_train_a)
    cqr_target.calibrate(target_cal_14d[feature_cols], target_cal_14d[target_col])
    lower_target_cqr, upper_target_cqr = cqr_target.predict_intervals(X_test_b, apply_conformal=True)
    evaluate_regime("Target_Adapted_CQR", lower_target_cqr, upper_target_cqr)
    print("="*95 + "\n")

    # -------------------------------------------------------------
    # 2. EVALUATE SENSOR AVAILABILITY SHIFTS & STRESS-TESTS
    # -------------------------------------------------------------
    print("\n" + "="*95)
    print("EXPERIMENT 5: SENSOR-AVAILABILITY DEGRADATION & STRESS-TESTING (DAYTIME PRIMARY)")
    print(f"Target Test Set: Site B Holdout ({len(X_test_b)} total hours, {n_daytime_b} daytime active hours)")
    print("="*95)
    print(f"{'Degradation Scenario':<36} | {'Day nMAE(%)':<12} | {'Day nRMSE(%)':<12} | {'Day PICP(%)':<12} | {'Day MPIW':<10}")
    print("-" * 95)

    perturb_engine = SensorPerturbationEngine(random_seed=42)
    scenarios = [
        ("Complete Telemetry (100% Sensors)", lambda x: x.copy()),
        ("Moderate Dropout (10% MCAR)", lambda x: perturb_engine.simulate_random_missingness(x, missing_rate=0.10)),
        ("Severe Dropout (30% MCAR)", lambda x: perturb_engine.simulate_random_missingness(x, missing_rate=0.30)),
        ("Extreme Dropout (50% MCAR)", lambda x: perturb_engine.simulate_random_missingness(x, missing_rate=0.50)),
        ("Contiguous Outage (3h Block)", lambda x: perturb_engine.simulate_contiguous_dropout(x, block_hours=3, n_outages=8)),
        ("Contiguous Outage (6h Block)", lambda x: perturb_engine.simulate_contiguous_dropout(x, block_hours=6, n_outages=5)),
        ("Total Pyranometer Outage (No GHI)", lambda x: perturb_engine.simulate_channel_outage(x, channel_keyword="ghi")),
        ("Pyranometer Drift (+15% Soiling Bias)", lambda x: perturb_engine.simulate_sensor_drift_and_sticking(x, drift_magnitude_pct=0.15))
    ]

    for name, perturb_func in scenarios:
        X_pert = perturb_func(X_test_b)

        # Point forecast (median of lower/upper)
        low, high = cqr_target.predict_intervals(X_pert, apply_conformal=True)
        point_preds = (low + high) / 2.0

        p_day = calculate_point_metrics(y_true=y_test_b[is_daytime_b], y_pred=point_preds[is_daytime_b], capacity=1.0)
        p_all = calculate_point_metrics(y_true=y_test_b, y_pred=point_preds, capacity=1.0)
        p_metrics = dict(p_day)
        p_metrics["daytime"] = p_day
        p_metrics["all_hours"] = p_all

        u_day = calculate_uncertainty_metrics(y_true=y_test_b[is_daytime_b], lower_bound=low[is_daytime_b], upper_bound=high[is_daytime_b], alpha=0.10)
        u_all = calculate_uncertainty_metrics(y_true=y_test_b, lower_bound=low, upper_bound=high, alpha=0.10)
        u_metrics = dict(u_day)
        u_metrics["daytime"] = u_day
        u_metrics["all_hours"] = u_all

        print(f"{name:<36} | {p_day['nMAE_pct']:<12.2f} | {p_day['nRMSE_pct']:<12.2f} | {u_day['PICP_pct']:<12.2f} | {u_day['MPIW']:<10.4f}")
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
