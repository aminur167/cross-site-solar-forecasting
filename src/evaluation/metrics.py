"""
Scientific Evaluation Metrics Module for Solar PV Power Forecasting
Calculates point prediction metrics (MAE, RMSE, nMAE, nRMSE, R2, Skill Score)
and probabilistic/conformal uncertainty metrics (Pinball Loss, PICP, MPIW, Winkler Score).
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

def calculate_point_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_persistence: np.ndarray = None, capacity: float = 1.0) -> Dict[str, float]:
    """
    Computes standard academic solar PV point forecast metrics.
    y_true, y_pred are given in Capacity Factor units [0, 1] (or kW).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Errors
    errors = y_true - y_pred
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    # Normalized metrics (relative to capacity, expressed in %)
    nmae = (mae / capacity) * 100.0
    nrmse = (rmse / capacity) * 100.0

    # R2 Score
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    metrics = {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "nMAE_pct": round(float(nmae), 2),
        "nRMSE_pct": round(float(nrmse), 2),
        "R2": round(float(r2), 4)
    }

    # Skill Score relative to persistence: SS = 1 - (RMSE_model / RMSE_persistence)
    if y_persistence is not None:
        y_persist = np.asarray(y_persistence, dtype=float)
        rmse_persist = np.sqrt(np.mean((y_true - y_persist) ** 2))
        skill_score = 1.0 - (rmse / rmse_persist) if rmse_persist > 0 else 0.0
        metrics["Skill_Score_vs_Persist"] = round(float(skill_score), 4)

    return metrics

def calculate_uncertainty_metrics(y_true: np.ndarray, lower_bound: np.ndarray, upper_bound: np.ndarray, alpha: float = 0.10) -> Dict[str, float]:
    """
    Computes rigorous uncertainty and conformal prediction metrics.
    Nominal coverage is (1 - alpha), e.g. 90% for alpha=0.10.
    """
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower_bound, dtype=float)
    upper = np.asarray(upper_bound, dtype=float)

    # 1. Prediction Interval Coverage Probability (PICP)
    covered = (y_true >= lower) & (y_true <= upper)
    picp = np.mean(covered) * 100.0

    # 2. Mean Prediction Interval Width (MPIW)
    widths = upper - lower
    mpiw = np.mean(widths)

    # 3. Winkler Score (Penalizes interval width + boundary violations)
    penalty_factor = 2.0 / alpha
    winkler_penalties = widths.copy()
    under_mask = (y_true < lower)
    over_mask = (y_true > upper)

    winkler_penalties[under_mask] += penalty_factor * (lower[under_mask] - y_true[under_mask])
    winkler_penalties[over_mask] += penalty_factor * (y_true[over_mask] - upper[over_mask])
    winkler_score = np.mean(winkler_penalties)

    return {
        "PICP_pct": round(float(picp), 2),
        "Target_Coverage_pct": round((1.0 - alpha) * 100.0, 1),
        "MPIW": round(float(mpiw), 4),
        "Winkler_Score": round(float(winkler_score), 4),
        "Coverage_Guaranteed": bool(picp >= (1.0 - alpha) * 100.0)
    }

def pinball_loss(y_true: np.ndarray, y_pred_q: np.ndarray, quantile: float) -> float:
    """
    Computes pinball (quantile) loss for quantile regression evaluation.
    """
    error = y_true - y_pred_q
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))
