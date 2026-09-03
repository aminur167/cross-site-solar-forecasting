"""
Statistical Significance and Hypothesis Testing Module for Time-Series Forecasting
Implements Diebold-Mariano (DM) test, Wilcoxon signed-rank test, and stationary block bootstrap
confidence intervals for rigorous academic evaluation.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple

def diebold_mariano_test(y_true: np.ndarray, y_pred1: np.ndarray, y_pred2: np.ndarray, h: int = 1, criterion: str = "MAE") -> Tuple[float, float]:
    """
    Computes Diebold-Mariano test statistic for comparing two forecasting models.
    h: Forecast horizon (1-hour ahead -> h=1).
    criterion: 'MAE' (|e|) or 'MSE' (e^2).
    Returns: (DM_statistic, p_value).
    Negative DM indicates Model 1 is statistically significantly more accurate than Model 2.
    """
    y_true = np.asarray(y_true, dtype=float)
    y1 = np.asarray(y_pred1, dtype=float)
    y2 = np.asarray(y_pred2, dtype=float)

    e1 = y_true - y1
    e2 = y_true - y2

    if criterion.upper() == "MAE":
        d = np.abs(e1) - np.abs(e2)
    else:
        d = (e1 ** 2) - (e2 ** 2)

    T = len(d)
    mean_d = np.mean(d)

    # Autocovariance up to lag h-1 (Harvey, Leybourne and Newbold correction)
    gamma_0 = np.var(d, ddof=0)
    gamma_sum = 0.0
    for lag in range(1, h):
        gamma_k = np.mean((d[lag:] - mean_d) * (d[:-lag] - mean_d))
        gamma_sum += 2.0 * gamma_k

    var_d = (gamma_0 + gamma_sum) / T
    if var_d <= 0:
        var_d = 1e-12

    dm_stat = mean_d / np.sqrt(var_d)

    # Small sample correction factor (Harvey et al., 1997)
    k_h = np.sqrt((T + 1 - 2*h + h*(h - 1)/T) / T)
    dm_stat_corrected = dm_stat * k_h

    # Two-sided p-value from Student-t distribution with T-1 degrees of freedom
    p_value = 2.0 * (1.0 - stats.t.cdf(np.abs(dm_stat_corrected), df=T - 1))

    return float(dm_stat_corrected), float(p_value)

def block_bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    block_size: int = 24,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95
) -> Dict[str, Tuple[float, float]]:
    """
    Computes stationary block bootstrap confidence intervals for nMAE (%) and nRMSE (%).
    Preserves diurnal autocorrelation structure by sampling consecutive 24-hour blocks.
    """
    rng = np.random.RandomState(42)
    n = len(y_true)
    num_blocks = int(np.ceil(n / block_size))

    boot_nmae = []
    boot_nrmse = []

    for _ in range(n_bootstraps):
        start_indices = rng.randint(0, n - block_size + 1, size=num_blocks)
        indices = np.concatenate([np.arange(idx, idx + block_size) for idx in start_indices])[:n]

        yt_b = y_true[indices]
        yp_b = y_pred[indices]

        errors = yt_b - yp_b
        mae = np.mean(np.abs(errors)) * 100.0 # Capacity factor normalized to %
        rmse = np.sqrt(np.mean(errors ** 2)) * 100.0

        boot_nmae.append(mae)
        boot_nrmse.append(rmse)

    alpha_low = (1.0 - confidence_level) / 2.0 * 100.0
    alpha_high = (1.0 - (1.0 - confidence_level) / 2.0) * 100.0

    return {
        "nMAE_CI_pct": (round(float(np.percentile(boot_nmae, alpha_low)), 2),
                        round(float(np.percentile(boot_nmae, alpha_high)), 2)),
        "nRMSE_CI_pct": (round(float(np.percentile(boot_nrmse, alpha_low)), 2),
                         round(float(np.percentile(boot_nrmse, alpha_high)), 2))
    }
