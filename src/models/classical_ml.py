"""
Classical Machine Learning Models for Solar PV Power Forecasting
Implements Random Forest, LightGBM, and XGBoost with physical constraints.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
from typing import Dict, Any, Optional

class SolarRandomForest:
    """
    Random Forest Regressor with physical bounding.
    """
    def __init__(self, n_estimators: int = 100, max_depth: int = 12, random_state: int = 42):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=1,
            random_state=random_state
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X)
        return np.clip(preds, 0.0, 1.2)

class SolarLightGBM:
    """
    LightGBM Regressor optimized for tabular time-series forecasting.
    """
    def __init__(self, n_estimators: int = 150, learning_rate: float = 0.05, max_depth: int = 6, max_iter: Optional[int] = None, random_state: int = 42):
        actual_n_estimators = max_iter if max_iter is not None else n_estimators
        self.model = lgb.LGBMRegressor(
            n_estimators=actual_n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=31,
            random_state=random_state,
            n_jobs=1,
            verbose=-1
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X)
        return np.clip(preds, 0.0, 1.2)

# Backward-compatible alias for gradient boosting baseline
SolarGradientBoosting = SolarLightGBM

class SolarXGBoost:
    """
    XGBoost Regressor with squared error objective.
    """
    def __init__(self, n_estimators: int = 150, learning_rate: float = 0.05, max_depth: int = 6, random_state: int = 42):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=1,
            verbosity=0
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X)
        return np.clip(preds, 0.0, 1.2)
