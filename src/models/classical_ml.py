"""
Classical Machine Learning Models for Solar PV Power Forecasting
Implements Random Forest and Gradient Boosted Decision Trees (HistGradientBoosting / LightGBM).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
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

class SolarGradientBoosting:
    """
    Histogram-based Gradient Boosted Trees (similar to LightGBM algorithm).
    Highly optimized for tabular time-series features.
    """
    def __init__(self, max_iter: int = 150, learning_rate: float = 0.05, max_depth: int = 8, random_state: int = 42):
        self.model = HistGradientBoostingRegressor(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X)
        return np.clip(preds, 0.0, 1.2)
