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
from src.models.classical_ml import SolarLightGBM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def run_uncertainty_and_robustness():
    path_a = "data/processed/site_a_features.csv"
    path_b = "data/processed/site_b_features.csv"

    logging.info("Loading feature matrices...")
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

    # Site B (Target) holdout test set (final 20%)
    n_b = len(df_b)
    target_pool_b = df_b.iloc[:int(n_b * 0.80)]
    target_test_b = df_b.iloc[int(n_b * 0.80):]

    # Calibration/training must precede target test. Historical Open-Meteo
    # weather is retrospective, so this experiment is not an operational
    # forecast evaluation.
    test_start = target_test_b.index.min()
    source = df_a.loc[df_a['target_timestamp'] < test_start]
    split_a = int(len(source) * 0.70)
    train_a = source.iloc[:split_a]
    cal_a = source.iloc[split_a:]

    # Target 14-day adaptation calibration subset
    cal_start = target_pool_b.index.min()
    target_cal_14d = target_pool_b.loc[(target_pool_b.index >= cal_start) & (target_pool_b.index < cal_start + pd.Timedelta(days=14))]
    transfer_source = source.iloc[:int(len(source) * 0.85)]
    point_model = SolarLightGBM(n_estimators=150, learning_rate=0.05, max_depth=6)
    point_model.fit(transfer_source[feature_cols], transfer_source[target_col])

    X_train_a, y_train_a = train_a[feature_cols], train_a[target_col]
    cal_a_day = cal_a.loc[cal_a['target_is_daytime'].astype(bool)]
    X_cal_a, y_cal_a = cal_a_day[feature_cols], cal_a_day[target_col]
    X_test_b = target_test_b[feature_cols]
    y_test_b = target_test_b[target_col].values
    is_daytime_b = target_test_b['target_is_daytime'].values.astype(bool)
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
    target_cal_day = target_cal_14d.loc[target_cal_14d['target_is_daytime'].astype(bool)]
    cqr_target.calibrate(target_cal_day[feature_cols], target_cal_day[target_col])
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
        ("Synchronized Packet Loss (30%)", lambda x: perturb_engine.simulate_synchronized_packet_loss(x, missing_rate=0.30)),
        ("Severe Dropout (30% MCAR)", lambda x: perturb_engine.simulate_random_missingness(x, missing_rate=0.30)),
        ("Extreme Dropout (50% MCAR)", lambda x: perturb_engine.simulate_random_missingness(x, missing_rate=0.50)),
        ("Contiguous Outage (3h Block)", lambda x: perturb_engine.simulate_contiguous_dropout(x, block_hours=3, n_outages=8)),
        ("Contiguous Outage (6h Block)", lambda x: perturb_engine.simulate_contiguous_dropout(x, block_hours=6, n_outages=5)),
        ("Total Pyranometer Outage (No GHI)", lambda x: perturb_engine.simulate_channel_outage(x, channel_keyword="ghi")),
        ("Pyranometer Drift (+15% Soiling Bias)", lambda x: perturb_engine.simulate_sensor_drift_and_sticking(x, drift_magnitude_pct=0.15))
        ,("GHI Stuck-at-Value (24h)", lambda x: perturb_engine.simulate_stuck_at_value(x, duration_hours=24))
    ]

    for name, perturb_func in scenarios:
        p_runs, u_runs = [], []
        for seed in range(42, 47):
            perturb_engine.set_seed(seed)
            X_pert = perturb_func(X_test_b)
            low, high = cqr_target.predict_intervals(X_pert, apply_conformal=True)
            point_preds = point_model.predict(X_pert)
            p_day = calculate_point_metrics(y_true=y_test_b[is_daytime_b], y_pred=point_preds[is_daytime_b], capacity=1.0)
            p_all = calculate_point_metrics(y_true=y_test_b, y_pred=point_preds, capacity=1.0)
            u_day = calculate_uncertainty_metrics(y_true=y_test_b[is_daytime_b], lower_bound=low[is_daytime_b], upper_bound=high[is_daytime_b], alpha=0.10)
            u_all = calculate_uncertainty_metrics(y_true=y_test_b, lower_bound=low, upper_bound=high, alpha=0.10)
            p_runs.append({"daytime": p_day, "all_hours": p_all})
            u_runs.append({"daytime": u_day, "all_hours": u_all})

        def summarize(runs):
            summary = {}
            for scope in ("daytime", "all_hours"):
                keys = [k for k, v in runs[0][scope].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
                summary[scope] = {k: round(float(np.mean([r[scope][k] for r in runs])), 4) for k in keys}
                summary[scope]["std"] = {k: round(float(np.std([r[scope][k] for r in runs], ddof=1)), 4) for k in keys}
                summary[scope]["seeds"] = [42, 43, 44, 45, 46]
            return summary

        p_metrics = summarize(p_runs)
        u_metrics = summarize(u_runs)
        p_day, u_day = p_metrics["daytime"], u_metrics["daytime"]
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
