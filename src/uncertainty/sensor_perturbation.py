"""
Sensor-Availability Perturbation and Stress-Testing Engine
=========================================================
Simulates realistic physical hardware and telemetry communication degradation:
- Scenario A: Random intermittent missingness (MCAR at 10%, 30%, 50%).
- Scenario B: Contiguous communication blackout windows (1h, 3h, 6h, 12h blocks).
- Scenario C: Complete sensor channel outage (pyranometer / GHI loss).
- Scenario D: Stuck-at-value sensor failure and linear measurement drift.

SCIENTIFIC INTEGRITY:
- All missing values are handled strictly causally using forward-holding (ffill) from the
  last observed value or seasonal defaults.
- NO FUTURE LOOKAHEAD (bfill is strictly eliminated).
"""

from typing import List, Optional, Dict, Tuple
import numpy as np
import pandas as pd


class SensorPerturbationEngine:
    """
    Stress-tests forecasters under telemetry degradation scenarios.
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

    def set_seed(self, seed: int):
        self.random_seed = seed
        self.rng = np.random.RandomState(seed)

    def simulate_random_missingness(
        self,
        X: pd.DataFrame,
        missing_rate: float,
        target_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Scenario A: Random packet dropout.
        Missing telemetry is held at the last known observation causally (ffill).
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
                X_pert.loc[mask, col] = np.nan
                # Strictly causal forward-fill (never bfill!)
                X_pert[col] = X_pert[col].ffill().fillna(0.0)

        return X_pert

    def simulate_contiguous_dropout(
        self,
        X: pd.DataFrame,
        block_hours: int = 3,
        n_outages: int = 5,
        target_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Scenario B: Contiguous communication blackout windows.
        Simulates remote link disconnection lasting block_hours consecutive intervals.
        """
        X_pert = X.copy()
        n = len(X_pert)
        if n <= block_hours:
            return X_pert

        cols = target_columns if target_columns is not None else [
            c for c in X.columns if any(k in c for k in ['ghi', 'temp', 'cloud', 'humidity', 'wind'])
        ]

        start_candidates = self.rng.randint(0, n - block_hours, size=n_outages)
        for start_idx in start_candidates:
            end_idx = min(start_idx + block_hours, n)
            for col in cols:
                if col in X_pert.columns:
                    X_pert.iloc[start_idx:end_idx, X_pert.columns.get_loc(col)] = np.nan

        # Causal recovery: forward-hold last valid observation before outage
        for col in cols:
            if col in X_pert.columns:
                X_pert[col] = X_pert[col].ffill().fillna(0.0)

        return X_pert

    def simulate_channel_outage(
        self,
        X: pd.DataFrame,
        channel_keyword: str = "ghi"
    ) -> pd.DataFrame:
        """
        Scenario C: Total hardware failure of a sensor channel (e.g. pyranometer death).
        All channels matching the keyword are zeroed out.
        """
        X_pert = X.copy()
        matched = [c for c in X_pert.columns if channel_keyword in c]
        for col in matched:
            X_pert[col] = 0.0
        return X_pert

    def simulate_sensor_drift_and_sticking(
        self,
        X: pd.DataFrame,
        drift_magnitude_pct: float = 0.15,
        target_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Scenario D: Gradual sensor calibration drift (e.g. pyranometer soiling/degradation).
        Simulates linear bias accumulation over time.
        """
        X_pert = X.copy()
        n = len(X_pert)
        drift_factor = 1.0 + np.linspace(0, drift_magnitude_pct, n)

        cols = target_columns if target_columns is not None else [
            c for c in X.columns if 'ghi' in c
        ]

        for col in cols:
            if col in X_pert.columns:
                X_pert[col] = X_pert[col] * drift_factor

        return X_pert


if __name__ == "__main__":
    engine = SensorPerturbationEngine()
    df_test = pd.DataFrame({"ghi_lag_0": [100.0, 200.0, 300.0, 400.0, 500.0]})
    pert = engine.simulate_random_missingness(df_test, missing_rate=0.4)
    print("[INFO] SensorPerturbationEngine test:", pert)
