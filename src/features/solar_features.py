"""
Feature Engineering Module for Solar PV Time-Series Forecasting
Generates diurnal cyclic encodings, non-leaking autoregressive lags, rolling statistics,
and solar geometry features.
"""

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class SolarFeatureEngineer:
    """
    Constructs feature matrices for 1-hour ahead solar PV forecasting.
    Strictly prevents lookahead bias (data leakage).
    """
    def __init__(self, target_col: str = "capacity_factor"):
        self.target_col = target_col

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes cleaned dataframe with DatetimeIndex and appends engineered features.
        Target is y = CF(t+1).
        """
        data = df.copy()

        # 1. Cyclic Temporal Features
        hours = data.index.hour
        day_of_year = data.index.dayofyear

        data['hour_sin'] = np.sin(2 * np.pi * hours / 24.0)
        data['hour_cos'] = np.cos(2 * np.pi * hours / 24.0)
        data['doy_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
        data['doy_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)

        # 2. Autoregressive Lags of Target (P_t, P_{t-1}, P_{t-2}, P_{t-23})
        # Note: At time t, we predict t+1. So available power observations are t, t-1, t-2, ...
        data['lag_0'] = data[self.target_col]           # CF at time t
        data['lag_1'] = data[self.target_col].shift(1)  # CF at time t-1
        data['lag_2'] = data[self.target_col].shift(2)  # CF at time t-2
        data['lag_23'] = data[self.target_col].shift(23) # CF at same hour yesterday (t-23 relative to t+1, i.e., t-24 to t)

        # 3. Non-leaking Rolling Statistics over past 3 hours [t-2, t-1, t]
        data['rolling_mean_3h'] = data['lag_0'].rolling(window=3).mean()
        data['rolling_std_3h'] = data['lag_0'].rolling(window=3).std().fillna(0.0)

        # 4. Weather Lags & Trends
        if 'ghi_wm2' in data.columns:
            data['ghi_lag_0'] = data['ghi_wm2']
            data['ghi_lag_1'] = data['ghi_wm2'].shift(1)
            data['ghi_diff_1h'] = data['ghi_lag_0'] - data['ghi_lag_1']

        if 'temp_c' in data.columns:
            data['temp_lag_0'] = data['temp_c']

        if 'cloud_cover_pct' in data.columns:
            data['cloud_lag_0'] = data['cloud_cover_pct']

        # 5. Define Forward Target Label: y_{t+1} = CF(t+1)
        data['target_1h'] = data[self.target_col].shift(-1)

        # 6. Drop unneeded or high-leakage intermediate columns & NaN rows caused by shifting
        engineered = data.dropna().copy()
        logging.info(f"Engineered feature matrix: {engineered.shape[0]} rows, {engineered.shape[1]} columns.")
        return engineered
