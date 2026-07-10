"""
Comprehensive Dataset Audit for UPB Romanian Two-Site PV Dataset
Inspects Site A and Site B data quality, date range, missing values, and physical constraints.
"""

import pandas as pd
import numpy as np

def audit_site(csv_path: str, site_name: str):
    print(f"\n{'='*30} AUDIT: {site_name} {'='*30}")
    df = pd.read_csv(csv_path)
    
    # Timestamp parsing
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    start_date = df['datetime_utc'].min()
    end_date = df['datetime_utc'].max()
    total_hours = len(df)
    
    print(f"Total Records: {total_hours}")
    print(f"Date Range: {start_date} to {end_date} ({(end_date - start_date).days} days)")
    
    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / total_hours) * 100
    missing_report = pd.DataFrame({'Missing_Count': missing, 'Missing_Pct': missing_pct})
    key_features = ['active_power_kw', 'temp_c', 'humidity_pct', 'cloud_cover_pct', 
                    'wind_speed_kmh', 'ghi_wm2', 'capacity_factor', 'is_generating']
    print("\n--- Key Variables Missing Statistics ---")
    print(missing_report.loc[key_features])
    
    # Power generation stats
    p_max = df['active_power_kw'].max()
    p_min = df['active_power_kw'].min()
    p_mean = df['active_power_kw'].mean()
    daytime_df = df[df['solar_elevation_deg'] > 0]
    p_day_mean = daytime_df['active_power_kw'].mean()
    
    print("\n--- Power & Physical Validation ---")
    print(f"Max Active Power: {p_max:.2f} kW (Rated: 50.0 kW)")
    print(f"Min Active Power: {p_min:.2f} kW (Negative indicates sensor error / consumption)")
    print(f"Overall Mean Power: {p_mean:.2f} kW")
    print(f"Daytime (Sun Elevation > 0) Mean Power: {p_day_mean:.2f} kW")
    print(f"Total Daytime Hours: {len(daytime_df)} ({len(daytime_df)/total_hours*100:.1f}%)")
    
    return df

if __name__ == "__main__":
    audit_site("data/raw/romanian_pv/site_a/site_a_hourly_weather.csv", "Site A (50 kWp)")
    audit_site("data/raw/romanian_pv/site_b/site_b_hourly_weather.csv", "Site B (50 kWp)")
