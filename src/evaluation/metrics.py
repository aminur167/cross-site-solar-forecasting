"""
Scientific Evaluation Metrics Module for Solar PV Power Forecasting
===================================================================
Calculates deterministic point forecast metrics and probabilistic uncertainty metrics:
- Point Metrics: MAE, RMSE, nMAE (%), nRMSE (%), R2, Skill Score vs. Persistence.
- Uncertainty Metrics: PICP (%), MPIW, Winkler Score, Pinball Loss.
"""

from typing import Any, Dict, Tuple, Optional
import numpy as np
import pandas as pd


def calculate_point_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_persistence: Optional[np.ndarray] = None,
    capacity: float = 1.0
) -> Dict[str, float]:
    """
    Computes standard point forecast metrics normalized relative to rated capacity.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    errors = y_true - y_pred
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    nmae = (mae / capacity) * 100.0
    nrmse = (rmse / capacity) * 100.0

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

    if y_persistence is not None:
        y_persist = np.asarray(y_persistence, dtype=float)
        rmse_persist = np.sqrt(np.mean((y_true - y_persist) ** 2))
        skill_score = 1.0 - (rmse / rmse_persist) if rmse_persist > 0 else 0.0
        metrics["Skill_Score_vs_Persist"] = round(float(skill_score), 4)

    return metrics


def calculate_uncertainty_metrics(
    y_true: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    alpha: float = 0.10
) -> Dict[str, Any]:
    """
    Computes prediction interval metrics at significance level alpha (nominal coverage = 1 - alpha).
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

    # 4. Pinball Loss at alpha/2 and 1 - alpha/2
    loss_low = pinball_loss(y_true, lower, alpha / 2.0)
    loss_high = pinball_loss(y_true, upper, 1.0 - alpha / 2.0)
    mean_pinball = (loss_low + loss_high) / 2.0

    target_cov = (1.0 - alpha) * 100.0
    return {
        "PICP_pct": round(float(picp), 2),
        "Target_Coverage_pct": round(target_cov, 1),
        "MPIW": round(float(mpiw), 4),
        "Winkler_Score": round(float(winkler_score), 4),
        "Mean_Pinball_Loss": round(float(mean_pinball), 4),
        # SCIENTIFIC HONESTY: An empirical metric meeting a nominal target is NOT a mathematical guarantee
        "Empirical_Coverage_Meets_Nominal_Target": bool(picp >= target_cov)
    }


def pinball_loss(y_true: np.ndarray, y_pred_q: np.ndarray, quantile: float) -> float:
    """Computes asymmetric pinball loss for quantile regression."""
    error = y_true - y_pred_q
    return float(np.mean(np.maximum(quantile * error, (quantile - 1.0) * error)))


if __name__ == "__main__":
    y = np.array([0.5, 0.8, 0.2])
    low = np.array([0.4, 0.7, 0.1])
    high = np.array([0.6, 0.9, 0.3])
    u = calculate_uncertainty_metrics(y, low, high, alpha=0.10)
    print("[INFO] calculate_uncertainty_metrics test:", u)
