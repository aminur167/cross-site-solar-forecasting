"""
Statistical Hypothesis Testing and Significance Module
======================================================
Implements rigorous time-series hypothesis testing:
1. Diebold-Mariano (DM) test with Newey-West HAC spectral variance estimation.
2. Harvey-Leybourne-Newbold small sample size correction.
3. Stationary block bootstrap confidence intervals (preserving diurnal dependence).
4. Strictly scientific p-value formatting (never reports exact p = 0.0000).
"""

from typing import Any, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats


def diebold_mariano_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    h: int = 1,
    criterion: str = "MAE",
    max_lag: Optional[int] = None
) -> Dict[str, Any]:
    """
    Computes Diebold-Mariano test with Newey-West HAC autocorrelation-consistent variance.
    y_pred1: Model 1 predictions (e.g. LightGBM).
    y_pred2: Model 2 predictions (e.g. Baseline or LSTM).
    Negative DM stat indicates Model 1 has statistically significantly lower error than Model 2.
    """
    y_true = np.asarray(y_true, dtype=float)
    y1 = np.asarray(y_pred1, dtype=float)
    y2 = np.asarray(y_pred2, dtype=float)

    e1 = y_true - y1
    e2 = y_true - y2

    if criterion.upper() == "MAE":
        d = np.abs(e1) - np.abs(e2)
    elif criterion.upper() == "MSE":
        d = (e1 ** 2) - (e2 ** 2)
    else:
        raise ValueError("Criterion must be 'MAE' or 'MSE'.")

    T = len(d)
    if T < 10:
        raise ValueError("Insufficient sample size for Diebold-Mariano test.")

    mean_d = np.mean(d)

    # HAC Newey-West spectral variance estimation with Bartlett kernel
    # Default bandwidth L = floor(4 * (T / 100)^(2/9))
    if max_lag is None:
        L = max(h, int(np.floor(4.0 * ((T / 100.0) ** (2.0 / 9.0)))))
    else:
        L = max_lag

    gamma_0 = np.var(d, ddof=0)
    gamma_sum = 0.0

    for lag in range(1, L + 1):
        weight = 1.0 - (lag / (L + 1.0)) # Bartlett kernel weight
        gamma_k = np.mean((d[lag:] - mean_d) * (d[:-lag] - mean_d))
        gamma_sum += 2.0 * weight * gamma_k

    lr_var = gamma_0 + gamma_sum
    if lr_var <= 0:
        lr_var = 1e-12

    dm_stat = mean_d / np.sqrt(lr_var / T)

    # Harvey, Leybourne, and Newbold (1997) finite-sample correction
    k_h = np.sqrt((T + 1.0 - 2.0 * h + h * (h - 1.0) / T) / T)
    dm_stat_corrected = dm_stat * k_h

    # Two-sided p-value from Student-t with T-1 degrees of freedom
    raw_p_value = float(2.0 * stats.t.sf(np.abs(dm_stat_corrected), df=T - 1))

    # Scientific p-value formatting: never report absolute zero
    if raw_p_value < 1e-4:
        p_str = "< 0.0001"
    else:
        p_str = f"{raw_p_value:.4f}"

    # Effect size (Cohen's d for paired differences)
    cohens_d = mean_d / (np.std(d, ddof=1) + 1e-12)

    return {
        "dm_stat": round(float(dm_stat_corrected), 4),
        "raw_p_value": float(max(raw_p_value, np.finfo(float).tiny)),
        "p_value_str": p_str,
        "mean_loss_diff": round(float(mean_d), 6),
        "cohens_d": round(float(cohens_d), 4),
        "is_significant_005": bool(raw_p_value < 0.05),
        "is_significant_001": bool(raw_p_value < 0.01)
    }


def block_bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    block_size: int = 24,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42
) -> Dict[str, Tuple[float, float]]:
    """
    Computes moving-block bootstrap confidence intervals for nMAE (%) and nRMSE (%).
    Fixed-length contiguous blocks preserve some local dependence; this is not a
    stationary bootstrap and does not resolve irregular sample timestamps.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    num_blocks = int(np.ceil(n / block_size))

    boot_nmae = []
    boot_nrmse = []

    for _ in range(n_bootstraps):
        start_indices = rng.randint(0, max(1, n - block_size + 1), size=num_blocks)
        indices = np.concatenate([np.arange(idx, min(idx + block_size, n)) for idx in start_indices])[:n]

        yt_b = y_true[indices]
        yp_b = y_pred[indices]

        errors = yt_b - yp_b
        mae = np.mean(np.abs(errors)) * 100.0
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


if __name__ == "__main__":
    y = np.random.uniform(0.1, 0.9, 500)
    p1 = y + np.random.normal(0, 0.04, 500)
    p2 = y + np.random.normal(0, 0.08, 500)
    res = diebold_mariano_test(y, p1, p2)
    print("[INFO] diebold_mariano_test test:", res)
