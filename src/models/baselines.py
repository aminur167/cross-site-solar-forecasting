"""
Baseline Forecasting Models for Solar Photovoltaic Generation
Implements Persistence, Seasonal Naive (24h), and Regularized Ridge Linear Baselines.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from typing import Dict, Any

class PersistenceModel:
    """
    Standard Persistence Baseline:
    Predicts that generation at t+1 is identical to generation at time t.
    """
    def __init__(self, target_lag_col: str = "lag_0"):
        self.target_lag_col = target_lag_col

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.target_lag_col].values

class SeasonalNaiveModel:
    """
    Seasonal Naive Baseline (24-hour Diurnal Cycle):
    Predicts that generation at t+1 is identical to the power output at the exact same hour yesterday.
    """
    def __init__(self, seasonal_lag_col: str = "lag_23"):
        self.seasonal_lag_col = seasonal_lag_col

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.seasonal_lag_col].values

class RidgeRegressionBaseline:
    """
    L2-Regularized Linear Baseline.
    """
    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha, random_state=42)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X)
        return np.clip(preds, 0.0, 1.2) # Bound predictions physically
