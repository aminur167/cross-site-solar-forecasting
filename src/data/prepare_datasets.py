"""
Data Preparation Pipeline
Executes end-to-end cleaning and feature engineering for Site A and Site B.
Saves reproducible processed datasets into data/processed/.
"""

import os
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import logging

from src.preprocessing.data_cleaner import SolarDataPreprocessor
from src.features.solar_features import SolarFeatureEngineer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def process_and_save():
    raw_dir = "data/raw/romanian_pv"
    proc_dir = "data/processed"
    os.makedirs(proc_dir, exist_ok=True)

    cleaner = SolarDataPreprocessor(rated_capacity_kw=50.0)
    engineer = SolarFeatureEngineer(target_col="capacity_factor")

    sites = [
        {"name": "Site_A", "raw_csv": os.path.join(raw_dir, "site_a/site_a_hourly_weather.csv")},
        {"name": "Site_B", "raw_csv": os.path.join(raw_dir, "site_b/site_b_hourly_weather.csv")}
    ]

    for site in sites:
        logging.info(f"\nProcessing {site['name']}...")
        raw_df = pd.read_csv(site['raw_csv'])
        cleaned_df = cleaner.clean_site_dataframe(raw_df, site_name=site['name'])
        featured_df = engineer.build_features(cleaned_df)

        csv_path = os.path.join(proc_dir, f"{site['name'].lower()}_features.csv")
        featured_df.to_csv(csv_path)
        logging.info(f"Saved processed features for {site['name']} to -> {csv_path}")

        # Summary of features
        logging.info(f"{site['name']} summary: {len(featured_df)} samples, target mean CF: {featured_df['target_1h'].mean():.4f}")

if __name__ == "__main__":
    process_and_save()
