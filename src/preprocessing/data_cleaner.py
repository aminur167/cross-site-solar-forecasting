"""
Data Preprocessing and Physical Validation Module
=================================================
Performs strictly causal, leak-free physical quality control on multi-site
solar PV and meteorological telemetry:
1. ISO 8601 UTC timestamp monotonicity and regular 1-hour frequency alignment.
2. Capacity factor normalization relative to verified rated AC capacity.
3. Explicit astronomical solar elevation daytime vs. nighttime segregation.
4. Strictly past-only forward-fill (ffill) short-gap imputation (NO future lookahead).
5. Target variable physical boundary validation without synthetic target generation.
"""

from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class SolarDataPreprocessor:
    """
    Physically validates and cleans raw solar PV plant telemetry.
    Strictly prevents lookahead bias (data leakage).
    """

    def __init__(self, rated_capacity_kw: float = 50.0, solar_elevation_threshold_deg: float = 5.0):
        self.rated_capacity_kw = rated_capacity_kw
        self.solar_elevation_threshold_deg = solar_elevation_threshold_deg

    def clean_site_dataframe(self, df: pd.DataFrame, site_name: str = "Site") -> pd.DataFrame:
        """
        Cleans and physically validates the raw dataset:
        - Establishes regular 1-hour continuous time series.
        - Enforces physical limits based on physical inverter limits.
        - Segregates daytime generation from astronomical nighttime.
        - Imputes small weather gaps causally (ffill only, max 2 steps).
        """
        logging.info(f"Starting causal data cleaning for {site_name} (Raw rows: {len(df)})...")
        data = df.copy()

        # 1. Parse timestamps to UTC DatetimeIndex
        if 'datetime_utc' not in data.columns:
            raise KeyError(f"Expected 'datetime_utc' column in {site_name} dataframe.")

        data['datetime_utc'] = pd.to_datetime(data['datetime_utc'], errors='coerce')
        # Check for unparseable timestamps
        if data['datetime_utc'].isna().any():
            logging.warning(f"Dropping {data['datetime_utc'].isna().sum()} unparseable timestamp rows.")
            data = data.dropna(subset=['datetime_utc'])

        # Sort and deduplicate
        data = data.sort_values('datetime_utc').drop_duplicates(subset=['datetime_utc'])
        data = data.set_index('datetime_utc')

        # 2. Reindex to strictly regular 1-hour frequency
        data = data.asfreq('1h')

        # 3. Target variable: Capacity Factor
        if 'active_power_kw' not in data.columns:
            raise KeyError("Expected 'active_power_kw' column for target calculation.")

        # Preserve observations and expose quality flags separately. Analytical
        # target is bounded to the observed 1.15x operating envelope; source
        # readings remain available unchanged in active_power_kw_raw.
        data['active_power_kw_raw'] = data['active_power_kw']
        data['negative_power_flag'] = (data['active_power_kw_raw'] < 0).fillna(False).astype(int)
        data['over_nameplate_flag'] = (data['active_power_kw_raw'] > self.rated_capacity_kw).fillna(False).astype(int)
        data['active_power_kw_analysis'] = data['active_power_kw_raw'].clip(lower=0.0, upper=self.rated_capacity_kw * 1.15)
        data['capacity_factor'] = data['active_power_kw_analysis'] / self.rated_capacity_kw

        # 4. Physical constraints on meteorological sensors
        if 'humidity_pct' in data.columns:
            data['humidity_pct'] = data['humidity_pct'].clip(lower=0.0, upper=100.0)
        if 'cloud_cover_pct' in data.columns:
            data['cloud_cover_pct'] = data['cloud_cover_pct'].clip(lower=0.0, upper=100.0)
        if 'ghi_wm2' in data.columns:
            data['ghi_wm2'] = data['ghi_wm2'].clip(lower=0.0, upper=1450.0)
        if 'dni_wm2' in data.columns:
            data['dni_wm2'] = data['dni_wm2'].clip(lower=0.0, upper=1200.0)
        if 'dhi_wm2' in data.columns:
            data['dhi_wm2'] = data['dhi_wm2'].clip(lower=0.0, upper=800.0)
        if 'wind_speed_kmh' in data.columns:
            data['wind_speed_kmh'] = data['wind_speed_kmh'].clip(lower=0.0, upper=150.0)

        # 5. Astronomical Daytime vs Nighttime Segregation
        # Dataset documentation uses >5 degrees; preserve twilight power values
        # and flag possible nighttime/twilight generation rather than zeroing it.
        if 'solar_elevation_deg' in data.columns:
            daytime_mask = (data['solar_elevation_deg'] > self.solar_elevation_threshold_deg)
            data['nighttime_generation_anomaly'] = ((~daytime_mask) & (data['active_power_kw_raw'] > 0.1)).fillna(False).astype(int)
            data['is_daytime'] = daytime_mask.astype(int)
        else:
            # Fallback if solar elevation missing
            daytime_mask = (data['capacity_factor'] > 0.001) | (data.get('ghi_wm2', 0) > 5.0)
            data['is_daytime'] = daytime_mask.astype(int)

        # 6. Strictly Causal Forward-Only Gap Imputation (Max 2 hours)
        # SCIENTIFIC INTEGRITY: NEVER USE bfill() OR interpolate(method='time')!
        # Both methods leak future observations backwards into the past.
        weather_cols = [c for c in ['temp_c', 'humidity_pct', 'cloud_cover_pct', 'wind_speed_kmh', 'ghi_wm2', 'dni_wm2', 'dhi_wm2'] if c in data.columns]
        for col in weather_cols:
            # Record missingness indicator before forward-fill
            data[f'{col}_was_missing'] = data[col].isna().astype(int)
            # Past-only forward fill for small telemetry dropouts (up to 2 hours)
            data[col] = data[col].ffill(limit=2)

        # Target variable: DO NOT interpolate missing target values!
        # If true generation was unrecorded, training on interpolated targets manufactures false labels.
        valid_targets = data['capacity_factor'].notna().sum()
        valid_daytime = (data['is_daytime'] == 1) & data['capacity_factor'].notna()
        logging.info(
            f"Cleaning complete for {site_name}: {len(data)} continuous hours. "
            f"Valid targets: {valid_targets}, Valid daytime hours: {valid_daytime.sum()}."
        )
        return data


if __name__ == "__main__":
    preprocessor = SolarDataPreprocessor(rated_capacity_kw=50.0)
    print("[INFO] SolarDataPreprocessor loaded successfully.")
