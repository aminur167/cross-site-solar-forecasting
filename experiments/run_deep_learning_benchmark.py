"""
Experiment 6: Deep Learning Sequence Models (LSTM & TCN) vs. GBDT
Includes Diebold-Mariano Hypothesis Testing and Stationary Block Bootstrap Confidence Intervals.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import torch
import random
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.classical_ml import SolarLightGBM
from src.models.deep_networks import TimeSeriesSequenceDataset, SolarLSTM, SolarTCN
from src.evaluation.metrics import calculate_point_metrics
from src.evaluation.statistical_tests import diebold_mariano_test, block_bootstrap_confidence_interval

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def train_torch_model(model: nn.Module, train_loader: DataLoader, epochs: int = 12, lr: float = 0.003):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return model

def predict_torch_model(model: nn.Module, test_loader: DataLoader) -> np.ndarray:
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            preds = model(X_batch)
            all_preds.extend(preds.numpy())
    return np.array(all_preds)

def run_deep_learning_experiments():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    path_a = "data/processed/site_a_features.csv"
    path_b = "data/processed/site_b_features.csv"

    logging.info("Loading feature matrices for Deep Learning benchmark...")
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

    seq_len = 6 # 6 hours lookback window
    n_a = len(df_a)
    n_b = len(df_b)
    target_test_b = df_b.iloc[int(n_b * 0.80):]
    test_start = target_test_b.index.min()
    source = df_a.loc[df_a['target_timestamp'] < test_start]
    train_a = source.iloc[:int(len(source) * 0.85)]

    # Normalize tabular features for neural networks
    X_train_raw = train_a[feature_cols].values
    y_train_raw = train_a[target_col].values

    X_test_raw = target_test_b[feature_cols].values
    y_test_raw = target_test_b[target_col].values

    mean_vec = np.nanmean(X_train_raw, axis=0)
    std_vec = np.nanstd(X_train_raw, axis=0) + 1e-6

    X_train_norm = (X_train_raw - mean_vec) / std_vec
    X_test_norm = (X_test_raw - mean_vec) / std_vec

    # Datasets
    train_dataset = TimeSeriesSequenceDataset(X_train_norm, y_train_raw, seq_len=seq_len, timestamps=train_a.index)
    test_dataset = TimeSeriesSequenceDataset(X_test_norm, y_test_raw, seq_len=seq_len, timestamps=target_test_b.index)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    aligned_ends = test_dataset.end_indices
    y_test_aligned = y_test_raw[aligned_ends]
    is_daytime_aligned = target_test_b['target_is_daytime'].iloc[aligned_ends].values.astype(bool)
    n_day_aligned = int(np.sum(is_daytime_aligned))

    # 1. LightGBM Baseline on the same aligned test sequence
    lgb_model = SolarLightGBM(n_estimators=150, learning_rate=0.05, max_depth=6)
    lgb_model.fit(train_a[feature_cols], train_a[target_col])
    lgb_preds = lgb_model.predict(target_test_b[feature_cols].iloc[aligned_ends])

    # 2. PyTorch LSTM
    logging.info("Training PyTorch LSTM...")
    lstm_model = SolarLSTM(input_dim=len(feature_cols), hidden_dim=64, num_layers=2)
    train_torch_model(lstm_model, train_loader, epochs=12, lr=0.003)
    lstm_preds = predict_torch_model(lstm_model, test_loader)

    # 3. PyTorch TCN
    logging.info("Training PyTorch TCN...")
    tcn_model = SolarTCN(input_dim=len(feature_cols), num_channels=[32, 64], kernel_size=3)
    train_torch_model(tcn_model, train_loader, epochs=12, lr=0.003)
    tcn_preds = predict_torch_model(tcn_model, test_loader)

    models_preds = {
        "LightGBM (GBDT)": lgb_preds,
        "PyTorch LSTM": lstm_preds,
        "PyTorch TCN": tcn_preds
    }

    results = {}
    print("\n" + "="*95)
    print("EXPERIMENT 6: DEEP LEARNING (LSTM & TCN) VS. GBDT WITH STATISTICAL SIGNIFICANCE")
    print(f"Target Test Set: Site B Aligned Horizon ({len(y_test_aligned)} total, {n_day_aligned} daytime active)")
    print("="*95)
    print(f"{'Model':<18} | {'Day nMAE(%)':<12} | {'95% CI (Day nMAE)':<20} | {'Day nRMSE(%)':<12} | {'DM Stat':<10} | {'p-value':<10}")
    print("-" * 95)

    ref_preds = lgb_preds
    contiguous_positions = np.split(np.arange(len(aligned_ends)), np.where(np.diff(aligned_ends) != 1)[0] + 1)
    dm_positions = max(contiguous_positions, key=len)

    for name, preds in models_preds.items():
        # Daytime primary
        m_day = calculate_point_metrics(y_true=y_test_aligned[is_daytime_aligned], y_pred=preds[is_daytime_aligned], capacity=1.0)
        ci_day = block_bootstrap_confidence_interval(y_true=y_test_aligned[is_daytime_aligned], y_pred=preds[is_daytime_aligned], block_size=12)

        # All-hours secondary
        m_all = calculate_point_metrics(y_true=y_test_aligned, y_pred=preds, capacity=1.0)
        ci_all = block_bootstrap_confidence_interval(y_true=y_test_aligned, y_pred=preds, block_size=24)

        if name == "LightGBM (GBDT)":
            dm_stat, p_val = 0.0, 1.0
            p_val_str = "1.0000"
            dm_res = {"dm_stat": 0.0, "raw_p_value": 1.0, "p_value_str": "1.0000"}
        else:
            if len(dm_positions) < 10:
                dm_res = {"status": "not_computed", "reason": "No contiguous hourly test segment of at least 10 samples."}
                dm_stat, p_val, p_val_str = float("nan"), float("nan"), "N/A"
            else:
                dm_res = diebold_mariano_test(y_true=y_test_aligned[dm_positions], y_pred1=ref_preds[dm_positions], y_pred2=preds[dm_positions], h=1)
                dm_res["evaluation_scope"] = "longest_contiguous_all_hours_test_segment"
                dm_stat = dm_res["dm_stat"]
                p_val = dm_res["raw_p_value"]
                p_val_str = dm_res["p_value_str"]

        res_entry = dict(m_day)
        res_entry["metrics"] = m_day
        res_entry["daytime"] = m_day
        res_entry["all_hours"] = m_all
        res_entry["confidence_intervals"] = ci_day
        res_entry["confidence_intervals_all"] = ci_all
        res_entry["diebold_mariano_vs_gbdt"] = dm_res
        results[name] = res_entry

        ci_str = f"[{ci_day['nMAE_CI_pct'][0]:.2f}, {ci_day['nMAE_CI_pct'][1]:.2f}]"
        print(f"{name:<18} | {m_day['nMAE_pct']:<12.2f} | {ci_str:<20} | {m_day['nRMSE_pct']:<12.2f} | {dm_stat:<10.4f} | {p_val_str:<10}")

    print("="*95 + "\n")

    out_file = "experiments/results/deep_learning_and_statistics.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    logging.info(f"Deep learning & statistical results saved to -> {out_file}")

if __name__ == "__main__":
    run_deep_learning_experiments()
