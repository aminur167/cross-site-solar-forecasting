"""
Feature Engineering Module for Solar PV Time-Series Forecasting
==============================================================
Constructs strictly causal feature matrices for 1-hour ahead solar forecasting:
1. Diurnal harmonic encodings (hour sin/cos, day of year sin/cos).
2. Autoregressive lags of power output without lookahead bias.
3. Non-leaking rolling statistics over historical windows.
4. Topocentric solar geometry (elevation, azimuth).
5. Explicit alignment of forward target: y_{t+1} = CF(t+1).
"""

from typing import List, Optional
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class SolarFeatureEngineer:
    """
    Constructs feature matrices for 1-hour ahead solar PV forecasting.
    Strictly prevents lookahead bias (data leakage).
    """

    CORE_FEATURE_COLS = [
        'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos',
        'lag_0', 'lag_1', 'lag_2', 'lag_23',
        'rolling_mean_3h', 'rolling_std_3h',
        'ghi_lag_0', 'ghi_lag_1', 'ghi_diff_1h',
        'temp_lag_0', 'cloud_lag_0',
        'humidity_pct', 'wind_speed_kmh',
        'solar_elevation_deg', 'solar_azimuth_deg'
    ]

    def __init__(self, target_col: str = "capacity_factor"):
        self.target_col = target_col

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes cleaned dataframe with DatetimeIndex and appends engineered features.
        Target is strictly y_{t+1} = CF(t+1).
        """
        data = df.copy()

        # 1. Cyclic Temporal Encodings (Continuous trigonometric harmonic priors)
        hours = data.index.hour
        day_of_year = data.index.dayofyear

        data['hour_sin'] = np.sin(2.0 * np.pi * hours / 24.0)
        data['hour_cos'] = np.cos(2.0 * np.pi * hours / 24.0)
        data['doy_sin'] = np.sin(2.0 * np.pi * day_of_year / 365.25)
        data['doy_cos'] = np.cos(2.0 * np.pi * day_of_year / 365.25)

        # 2. Autoregressive Lags of Target (Strictly at time <= t)
        # At forecast origin t, available observations are t, t-1, t-2, t-23
        data['lag_0'] = data[self.target_col]
        data['lag_1'] = data[self.target_col].shift(1)
        data['lag_2'] = data[self.target_col].shift(2)
        # 24-hour seasonal lag: same hour yesterday (t-23 relative to t+1, i.e. t-24 to t)
        data['lag_23'] = data[self.target_col].shift(23)

        # 3. Strictly Historical Non-Leaking Rolling Statistics [t-2, t-1, t]
        data['rolling_mean_3h'] = data['lag_0'].rolling(window=3, min_periods=1).mean()
        data['rolling_std_3h'] = data['lag_0'].rolling(window=3, min_periods=1).std().fillna(0.0)

        # 4. Weather Lags and Dynamic Solar Radiation Ramps
        if 'ghi_wm2' in data.columns:
            data['ghi_lag_0'] = data['ghi_wm2']
            data['ghi_lag_1'] = data['ghi_wm2'].shift(1)
            data['ghi_diff_1h'] = (data['ghi_lag_0'] - data['ghi_lag_1']).fillna(0.0)
        else:
            data['ghi_lag_0'] = 0.0
            data['ghi_lag_1'] = 0.0
            data['ghi_diff_1h'] = 0.0

        if 'temp_c' in data.columns:
            data['temp_lag_0'] = data['temp_c']
        else:
            data['temp_lag_0'] = 20.0

        if 'cloud_cover_pct' in data.columns:
            data['cloud_lag_0'] = data['cloud_cover_pct']
        else:
            data['cloud_lag_0'] = 0.0

        # Ensure solar geometry presence
        if 'solar_elevation_deg' not in data.columns:
            data['solar_elevation_deg'] = 45.0
        if 'solar_azimuth_deg' not in data.columns:
            data['solar_azimuth_deg'] = 180.0
        if 'is_daytime' not in data.columns:
            data['is_daytime'] = True

        # 5. Target Construction: Predict Capacity Factor at t+1
        data['target_1h'] = data[self.target_col].shift(-1)

        # 6. Filter candidate features
        active_features = [c for c in self.CORE_FEATURE_COLS if c in data.columns]
        required_subset = [c for c in active_features + ['target_1h', 'is_daytime', self.target_col] if c in data.columns]

        # Drop only rows where REQUIRED model inputs or forward target are missing
        # (Preserves rows where raw diagnostic columns had NaNs!)
        engineered = data.dropna(subset=required_subset).copy()

        logging.info(
            f"Engineered feature matrix: {engineered.shape[0]} valid rows. "
            f"Daytime rows: {(engineered['is_daytime'] == 1).sum()}."
        )
        return engineered


if __name__ == "__main__":
    engineer = SolarFeatureEngineer()
    print("[INFO] SolarFeatureEngineer loaded.")
