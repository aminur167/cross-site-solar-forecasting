"""
Experiment 2 & 3: Cross-Site Zero-Shot Transfer and Few-Shot Target Adaptation
Evaluates RQ1 (Generalization Penalty) and RQ2 (Minimum Adaptation Budget Frontier).
Source Domain: Site A | Target Domain: Site B
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.classical_ml import SolarLightGBM, SolarRandomForest, SolarXGBoost
from src.models.baselines import PersistenceModel
from src.adaptation.few_shot_adapter import FewShotDomainAdapter
from src.evaluation.metrics import calculate_point_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def run_cross_site_adaptation_experiment():
    path_a = "data/processed/site_a_features.csv"
    path_b = "data/processed/site_b_features.csv"

    logging.info("Loading feature matrices for Site A (Source) and Site B (Target)...")
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

    # 1. Source Split (Site A: 85% train/val, used as Source knowledge base)
    n_a = len(df_a)
    source_train_df = df_a.iloc[:int(n_a * 0.85)]
    X_source, y_source = source_train_df[feature_cols], source_train_df[target_col]

    # 2. Target Split (Site B: Fixed holdout 20% test set at the end)
    n_b = len(df_b)
    target_split_idx = int(n_b * 0.80)
    target_pool_df = df_b.iloc[:target_split_idx] # Adaptation pool
    target_test_df = df_b.iloc[target_split_idx:] # Fixed holdout evaluation set

    X_test_b = target_test_df[feature_cols]
    y_test_b = target_test_df[target_col].values
    is_daytime_b = target_test_df['is_daytime'].values == True
    n_daytime_b = int(np.sum(is_daytime_b))

    logging.info(f"Source Training Samples: {len(X_source)}")
    logging.info(f"Target Adaptation Pool: {len(target_pool_df)}, Target Test Samples: {len(X_test_b)} (Daytime: {n_daytime_b})")
    logging.info(f"Target Test Horizon: {target_test_df.index.min()} to {target_test_df.index.max()}")

    # Baseline persistence on Target Test set
    persist_model = PersistenceModel(target_lag_col='lag_0')
    persist_preds = persist_model.predict(X_test_b)
    
    persist_metrics_day = calculate_point_metrics(y_true=y_test_b[is_daytime_b], y_pred=persist_preds[is_daytime_b], capacity=1.0)
    persist_metrics_all = calculate_point_metrics(y_true=y_test_b, y_pred=persist_preds, capacity=1.0)
    persist_metrics = dict(persist_metrics_day)
    persist_metrics["daytime"] = persist_metrics_day
    persist_metrics["all_hours"] = persist_metrics_all

    # Oracle Benchmark: Model trained on FULL target adaptation pool
    logging.info("Training Target-Site Oracle (Full Target Data)...")
    oracle_model = SolarLightGBM(n_estimators=150, learning_rate=0.05, max_depth=6)
    oracle_model.fit(target_pool_df[feature_cols], target_pool_df[target_col])
    oracle_preds = oracle_model.predict(X_test_b)
    
    oracle_metrics_day = calculate_point_metrics(y_true=y_test_b[is_daytime_b], y_pred=oracle_preds[is_daytime_b], y_persistence=persist_preds[is_daytime_b], capacity=1.0)
    oracle_metrics_all = calculate_point_metrics(y_true=y_test_b, y_pred=oracle_preds, y_persistence=persist_preds, capacity=1.0)
    oracle_metrics = dict(oracle_metrics_day)
    oracle_metrics["daytime"] = oracle_metrics_day
    oracle_metrics["all_hours"] = oracle_metrics_all

    # 3. Adaptation Budgets to Evaluate: k in [0, 1, 3, 7, 14] days
    budgets = [0, 1, 3, 7, 14]
    results = {
        "Persistence_Baseline": persist_metrics,
        "Target_Oracle_Full": oracle_metrics,
        "Zero_Shot_and_Adaptation": {}
    }

    print("\n" + "="*95)
    print("EXPERIMENT 2 & 3: CROSS-SITE GENERALIZATION & FEW-SHOT ADAPTATION FRONTIER")
    print(f"Target Test Set: Site B Holdout ({len(target_test_df)} total hours, {n_daytime_b} daytime active hours)")
    print("="*95)
    print(f"{'Regime':<28} | {'Target Days':<11} | {'Day nMAE(%)':<12} | {'Day nRMSE(%)':<12} | {'Day R2':<8} | {'Skill Score':<11}")
    print("-" * 95)
    print(f"{'Persistence (Target Ref)':<28} | {'-':<11} | {persist_metrics['nMAE_pct']:<12.2f} | {persist_metrics['nRMSE_pct']:<12.2f} | {persist_metrics['R2']:<8.4f} | {0.0:<11.4f}")
    print(f"{'Oracle (Full Target)':<28} | {'Full (~315d)':<11} | {oracle_metrics['nMAE_pct']:<12.2f} | {oracle_metrics['nRMSE_pct']:<12.2f} | {oracle_metrics['R2']:<8.4f} | {oracle_metrics['Skill_Score_vs_Persist']:<11.4f}")
    print("-" * 95)

    adapter = FewShotDomainAdapter(target_weight=2.5)

    for k in budgets:
        # Extract k calendar days from Target Pool
        target_budget_df = FewShotDomainAdapter.extract_budget_subset(target_pool_df, budget_days=k)
        X_target_k = target_budget_df[feature_cols] if len(target_budget_df) > 0 else pd.DataFrame(columns=feature_cols)
        y_target_k = target_budget_df[target_col] if len(target_budget_df) > 0 else pd.Series(dtype=float)

        # A. Target-Only (Train from scratch using only k target days, if k > 0)
        target_only_metrics = None
        if k > 0 and len(X_target_k) >= 10:
            target_only_model = SolarLightGBM(n_estimators=max(30, k*15), learning_rate=0.08, max_depth=4)
            target_only_model.fit(X_target_k, y_target_k)
            to_preds = target_only_model.predict(X_test_b)
            to_day = calculate_point_metrics(y_true=y_test_b[is_daytime_b], y_pred=to_preds[is_daytime_b], y_persistence=persist_preds[is_daytime_b], capacity=1.0)
            to_all = calculate_point_metrics(y_true=y_test_b, y_pred=to_preds, y_persistence=persist_preds, capacity=1.0)
            target_only_metrics = dict(to_day)
            target_only_metrics["daytime"] = to_day
            target_only_metrics["all_hours"] = to_all
            print(f"{'Target-Only (No Transfer)':<28} | {f'{k} days':<11} | {to_day['nMAE_pct']:<12.2f} | {to_day['nRMSE_pct']:<12.2f} | {to_day['R2']:<8.4f} | {to_day['Skill_Score_vs_Persist']:<11.4f}")

        # B. Transfer Learning (Source + Target k days adaptation)
        X_adapted, y_adapted, sample_weights = adapter.prepare_adaptation_dataset(
            X_source, y_source, X_target_k, y_target_k
        )
        adapted_model = SolarLightGBM(n_estimators=150, learning_rate=0.05, max_depth=6)
        adapted_model.model.fit(X_adapted, y_adapted, sample_weight=sample_weights)
        tl_preds = adapted_model.predict(X_test_b)
        
        tl_day = calculate_point_metrics(y_true=y_test_b[is_daytime_b], y_pred=tl_preds[is_daytime_b], y_persistence=persist_preds[is_daytime_b], capacity=1.0)
        tl_all = calculate_point_metrics(y_true=y_test_b, y_pred=tl_preds, y_persistence=persist_preds, capacity=1.0)
        tl_metrics = dict(tl_day)
        tl_metrics["daytime"] = tl_day
        tl_metrics["all_hours"] = tl_all

        label = "Zero-Shot Transfer" if k == 0 else f"Adapted ({k} days)"
        print(f"{label:<28} | {f'{k} days':<11} | {tl_day['nMAE_pct']:<12.2f} | {tl_day['nRMSE_pct']:<12.2f} | {tl_day['R2']:<8.4f} | {tl_day['Skill_Score_vs_Persist']:<11.4f}")

        results["Zero_Shot_and_Adaptation"][f"k_{k}_days"] = {
            "budget_days": k,
            "samples_used": len(target_budget_df),
            "transfer_adapted_metrics": tl_metrics,
            "target_only_metrics": target_only_metrics
        }

    print("="*95 + "\n")

    # Serialize results
    out_file = "experiments/results/cross_site_adaptation_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    logging.info(f"Cross-site adaptation results saved to -> {out_file}")

if __name__ == "__main__":
    run_cross_site_adaptation_experiment()
