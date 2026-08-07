"""
Sensor-Availability Perturbation and Stress-Testing Module
Simulates realistic hardware and communication failures:
1. Random telemetry dropout (10%, 30%, 50% missingness).
2. Complete catastrophic sensor outage (Loss of pyranometer / GHI or temperature).
"""

import numpy as np
import pandas as pd
from typing import List, Optional

class SensorPerturbationEngine:
    """
    Stress-tests forecasting models under sensor degradation and availability shifts.
    """
    def __init__(self, random_seed: int = 42):
        self.rng = np.random.RandomState(random_seed)

    def inject_random_missingness(
        self,
        X: pd.DataFrame,
        missing_rate: float,
        target_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Simulates intermittent telemetry loss by masking values with forward-filled previous readings.
        """
        if missing_rate <= 0.0:
            return X.copy()

        X_pert = X.copy()
        cols = target_columns if target_columns is not None else [
            c for c in X.columns if any(k in c for k in ['ghi', 'temp', 'cloud', 'humidity', 'wind'])
        ]

        for col in cols:
            if col in X_pert.columns:
                mask = self.rng.rand(len(X_pert)) < missing_rate
                # In real operational plants, a dead sensor sticks to its last known value or 0
                X_pert.loc[mask, col] = np.nan
                X_pert[col] = X_pert[col].ffill().bfill().fillna(0.0)

        return X_pert

    def simulate_channel_outage(
        self,
        X: pd.DataFrame,
        channel_keyword: str = "ghi"
    ) -> pd.DataFrame:
        """
        Simulates total failure of a sensor channel (e.g., broken pyranometer).
        Replaces all features derived from that sensor with 0.0.
        """
        X_pert = X.copy()
        matched_cols = [c for c in X_pert.columns if channel_keyword in c]
        for col in matched_cols:
            X_pert[col] = 0.0
        return X_pert
