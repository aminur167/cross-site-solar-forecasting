"""
Conformalized Quantile Regression (CQR) and Split Conformal Prediction Module
Provides distribution-free prediction intervals with rigorous finite-sample coverage guarantees.
Mathematical basis: Romano et al. (2019) & Angelopoulos and Bates (2023).
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Tuple, Dict, Any, Optional

class ConformalizedQuantileRegressor:
    """
    Implements Conformalized Quantile Regression (CQR) for solar power interval forecasting.
    Guarantees empirical coverage >= (1 - alpha) on exchangeable data.
    """
    def __init__(self, alpha: float = 0.10, n_estimators: int = 150, learning_rate: float = 0.05, max_depth: int = 6):
        self.alpha = alpha
        self.alpha_low = alpha / 2.0
        self.alpha_high = 1.0 - (alpha / 2.0)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

        # Base quantile models
        self.model_low = lgb.LGBMRegressor(
            objective='quantile',
            alpha=self.alpha_low,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=42,
            n_jobs=1,
            verbose=-1
        )
        self.model_high = lgb.LGBMRegressor(
            objective='quantile',
            alpha=self.alpha_high,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=42,
            n_jobs=1,
            verbose=-1
        )
        self.conformal_adjustment_ = 0.0
        self.is_calibrated_ = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Fits lower and upper pinball loss quantile estimators."""
        self.model_low.fit(X_train, y_train)
        self.model_high.fit(X_train, y_train)
        return self

    def calibrate(self, X_cal: pd.DataFrame, y_cal: pd.Series):
        """
        Calibrates the quantile predictions on an independent calibration set.
        Calculates non-conformity scores and conformal quantile adjustment.
        """
        y_cal_arr = np.asarray(y_cal, dtype=float)
        q_low = self.model_low.predict(X_cal)
        q_high = self.model_high.predict(X_cal)

        # Non-conformity score E_i = max(q_low - y, y - q_high)
        scores = np.maximum(q_low - y_cal_arr, y_cal_arr - q_high)

        n = len(scores)
        # Finite sample conformal correction factor: ceil((n + 1) * (1 - alpha)) / n
        quantile_level = np.clip(np.ceil((n + 1) * (1.0 - self.alpha)) / n, 0.0, 1.0)
        self.conformal_adjustment_ = float(np.quantile(scores, quantile_level))
        self.is_calibrated_ = True
        return self

    def predict_intervals(self, X_test: pd.DataFrame, apply_conformal: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates lower and upper prediction intervals.
        apply_conformal=True applies the finite-sample CQR calibration.
        """
        q_low = self.model_low.predict(X_test)
        q_high = self.model_high.predict(X_test)

        # Enforce quantile monotonicity (q_high >= q_low)
        q_high = np.maximum(q_high, q_low)

        if apply_conformal and self.is_calibrated_:
            lower_bound = q_low - self.conformal_adjustment_
            upper_bound = q_high + self.conformal_adjustment_
        else:
            lower_bound = q_low
            upper_bound = q_high

        # Physical capacity factor clipping [0.0, 1.2]
        lower_bound = np.clip(lower_bound, 0.0, 1.2)
        upper_bound = np.clip(upper_bound, 0.0, 1.2)
        # Ensure lower <= upper after clipping
        upper_bound = np.maximum(upper_bound, lower_bound)

        return lower_bound, upper_bound
