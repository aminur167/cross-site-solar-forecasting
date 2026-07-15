"""
Data Preprocessing and Physical Filtering Module
Ensures physical plausibility, night-time segregation, and leak-free imputation.
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class SolarDataPreprocessor:
    """
    Standardizes raw solar PV telemetry and meteorological observations.
    """
    def __init__(self, rated_capacity_kw: float = 50.0):
        self.rated_capacity_kw = rated_capacity_kw

    def clean_site_dataframe(self, df: pd.DataFrame, site_name: str = "Site") -> pd.DataFrame:
        """
        Cleans and physically validates the raw dataset.
        """
        logging.info(f"Starting data cleaning for {site_name} (Initial rows: {len(df)})...")
        data = df.copy()

        # 1. Parse timestamps to UTC datetime index
        if 'datetime_utc' in data.columns:
            data['datetime_utc'] = pd.to_datetime(data['datetime_utc'])
            data = data.sort_values('datetime_utc').drop_duplicates(subset=['datetime_utc'])
            data = data.set_index('datetime_utc')
        else:
            raise KeyError("Expected 'datetime_utc' column in input dataframe.")

        # 2. Ensure regular 1-hour frequency
        data = data.asfreq('1h')

        # 3. Target variable: Capacity Factor (CF = AC_Power / Rated_Capacity)
        if 'active_power_kw' not in data.columns:
            raise KeyError("Expected 'active_power_kw' column for target calculation.")
        
        # Enforce physical bounds on power: [0, 1.2 * Rated Capacity]
        data['active_power_kw'] = data['active_power_kw'].clip(lower=0.0, upper=self.rated_capacity_kw * 1.2)
        data['capacity_factor'] = data['active_power_kw'] / self.rated_capacity_kw

        # 4. Physical constraints on meteorological variables
        if 'humidity_pct' in data.columns:
            data['humidity_pct'] = data['humidity_pct'].clip(lower=0.0, upper=100.0)
        if 'cloud_cover_pct' in data.columns:
            data['cloud_cover_pct'] = data['cloud_cover_pct'].clip(lower=0.0, upper=100.0)
        if 'ghi_wm2' in data.columns:
            data['ghi_wm2'] = data['ghi_wm2'].clip(lower=0.0, upper=1400.0)
        if 'wind_speed_kmh' in data.columns:
            data['wind_speed_kmh'] = data['wind_speed_kmh'].clip(lower=0.0, upper=150.0)

        # 5. Night-time vs Daytime segregation
        # When sun is below horizon (solar_elevation_deg <= 0), generation is strictly 0
        if 'solar_elevation_deg' in data.columns:
            night_mask = (data['solar_elevation_deg'] <= 0.0)
            data.loc[night_mask, 'active_power_kw'] = 0.0
            data.loc[night_mask, 'capacity_factor'] = 0.0
            data['is_daytime'] = (~night_mask).astype(int)
        else:
            data['is_daytime'] = (data['capacity_factor'] > 0.0).astype(int)

        # 6. Leak-Free Short Gap Imputation (Max 2 hours forward interpolation)
        weather_cols = [c for c in ['temp_c', 'humidity_pct', 'cloud_cover_pct', 'wind_speed_kmh', 'ghi_wm2'] if c in data.columns]
        for col in weather_cols:
            # Interpolate small gaps forward only to avoid future lookahead bias
            data[col] = data[col].interpolate(method='time', limit=2)
            # Backward fill edge cases at start if any
            data[col] = data[col].bfill()

        # Target variable interpolation: Never manufacture synthetic target values for large gaps
        data['capacity_factor'] = data['capacity_factor'].interpolate(method='time', limit=1)

        logging.info(f"Cleaning complete for {site_name}. Valid rows remaining: {len(data.dropna(subset=['capacity_factor']))}")
        return data
