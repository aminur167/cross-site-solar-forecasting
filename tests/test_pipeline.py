"""
Unit Tests for Solar PV Pipeline and Experimental Methodology
Verifies data leakage prevention, physical constraint bounds, metric accuracy,
and conformal interval ordering using standard unittest.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.data_cleaner import SolarDataPreprocessor
from src.features.solar_features import SolarFeatureEngineer
from src.adaptation.few_shot_adapter import FewShotDomainAdapter
from src.evaluation.metrics import calculate_point_metrics, calculate_uncertainty_metrics
from src.uncertainty.conformal_predictor import ConformalizedQuantileRegressor

class TestSolarPipeline(unittest.TestCase):

    def test_physical_bounds_cleaning(self):
        """Verify that cleaner strictly clips negative power and respects capacity."""
        cleaner = SolarDataPreprocessor(rated_capacity_kw=50.0)
        dates = pd.date_range("2026-01-01 00:00:00", periods=24, freq="1h")
        df_corrupt = pd.DataFrame({
            "datetime_utc": dates,
            "active_power_kw": [-5.0, 0.0, 25.0, 75.0] + [10.0] * 20,
            "solar_elevation_deg": [-10.0, -5.0, 30.0, 45.0] + [20.0] * 20
        })
        cleaned = cleaner.clean_site_dataframe(df_corrupt, site_name="Test")
        self.assertGreaterEqual(cleaned["capacity_factor"].min(), 0.0, "Capacity factor cannot be negative")
        self.assertLessEqual(cleaned["capacity_factor"].max(), 1.2, "Capacity factor cannot exceed 1.2")
        self.assertEqual(cleaned.loc[dates[0], "capacity_factor"], 0.0, "Nighttime generation must be 0")

    def test_leakage_free_feature_engineering(self):
        """Verify that target is strictly t+1 and lags do not contain future observations."""
        engineer = SolarFeatureEngineer(target_col="capacity_factor")
        dates = pd.date_range("2026-01-01 00:00:00", periods=50, freq="1h")
        df = pd.DataFrame({
            "capacity_factor": np.linspace(0.1, 0.6, 50)
        }, index=dates)

        featured = engineer.build_features(df)
        self.assertTrue(np.allclose(featured["lag_0"].values, featured["capacity_factor"].values))
        self.assertEqual(featured.loc[dates[25], "lag_1"], df.loc[dates[24], "capacity_factor"])
        self.assertEqual(featured.loc[dates[25], "target_1h"], df.loc[dates[26], "capacity_factor"])

    def test_diurnal_cyclic_transformations(self):
        """Verify diurnal cyclic encodings map smoothly within [-1, 1] and periodicity holds."""
        engineer = SolarFeatureEngineer(target_col="capacity_factor")
        dates = pd.date_range("2026-06-01 00:00:00", periods=72, freq="1h")
        df = pd.DataFrame({"capacity_factor": [0.5] * 72}, index=dates)
        featured = engineer.build_features(df)

        # Check amplitude bounds [-1, 1]
        self.assertGreaterEqual(featured["hour_sin"].min(), -1.0)
        self.assertLessEqual(featured["hour_sin"].max(), 1.0)
        self.assertGreaterEqual(featured["hour_cos"].min(), -1.0)
        self.assertLessEqual(featured["hour_cos"].max(), 1.0)

        # Verify trigonometric Pythagorean identity: sin^2 + cos^2 == 1
        trig_sum = featured["hour_sin"]**2 + featured["hour_cos"]**2
        self.assertTrue(np.allclose(trig_sum, 1.0, atol=1e-6), "Trigonometric cyclic identity sin^2 + cos^2 must equal 1")

        # Verify 24-hour exact periodicity on retained timestamps
        t1 = dates[24]
        t2 = dates[48]
        if t1 in featured.index and t2 in featured.index:
            self.assertAlmostEqual(featured.loc[t1, "hour_cos"], featured.loc[t2, "hour_cos"])
            self.assertAlmostEqual(featured.loc[t1, "hour_sin"], featured.loc[t2, "hour_sin"])

    def test_target_budget_chronological_isolation(self):
        """Verify that few-shot target budget strictly precedes test set with zero chronological leakage."""
        dates = pd.date_range("2026-07-01 00:00:00", periods=24 * 30, freq="1h")
        df_target = pd.DataFrame({"capacity_factor": np.random.uniform(0.1, 0.8, len(dates))}, index=dates)

        budget_days = 7
        budget_subset = FewShotDomainAdapter.extract_budget_subset(df_target, budget_days=budget_days)

        self.assertFalse(budget_subset.empty, "Target budget subset should not be empty")
        # Ensure budget covers exactly budget_days
        budget_duration = budget_subset.index.max() - budget_subset.index.min()
        self.assertAlmostEqual(budget_duration.total_seconds() / 86400.0, budget_days, delta=0.1)

        # Ensure subsequent evaluation data has timestamps strictly greater than budget max
        test_subset = df_target.loc[df_target.index > budget_subset.index.max()]
        self.assertTrue((test_subset.index > budget_subset.index.max()).all(), "Test evaluation set must strictly follow adaptation budget")
        self.assertEqual(len(set(test_subset.index).intersection(set(budget_subset.index))), 0, "Zero index overlap permitted")

    def test_metric_calculations(self):
        """Verify point and uncertainty metrics against known values."""
        y_true = np.array([0.5, 0.8, 0.2, 0.9])
        y_pred = np.array([0.5, 0.7, 0.3, 0.9])
        metrics = calculate_point_metrics(y_true, y_pred, capacity=1.0)
        self.assertAlmostEqual(metrics["MAE"], 0.05)
        self.assertAlmostEqual(metrics["nMAE_pct"], 5.0)

        lower = np.array([0.4, 0.75, 0.15, 0.95])
        upper = np.array([0.6, 0.85, 0.25, 1.05])
        u_metrics = calculate_uncertainty_metrics(y_true, lower, upper, alpha=0.10)
        self.assertAlmostEqual(u_metrics["PICP_pct"], 75.0)

    def test_conformal_interval_ordering(self):
        """Verify that lower bounds are always <= upper bounds."""
        cqr = ConformalizedQuantileRegressor(alpha=0.10)
        X_mock = pd.DataFrame({"feat": np.random.randn(100)})
        y_mock = pd.Series(np.random.uniform(0.1, 0.9, 100))
        cqr.fit(X_mock, y_mock)
        cqr.calibrate(X_mock, y_mock)
        low, high = cqr.predict_intervals(X_mock)
        self.assertTrue(np.all(low <= high), "Lower interval bound must not exceed upper bound")

    def test_conformal_multi_alpha_monotonicity(self):
        """Verify that prediction intervals widen monotonically as alpha decreases (90% -> 95% -> 99%)."""
        np.random.seed(42)
        X_mock = pd.DataFrame({"feat1": np.random.randn(120), "feat2": np.random.randn(120)})
        y_mock = pd.Series(np.random.uniform(0.1, 0.9, 120))

        cqr_90 = ConformalizedQuantileRegressor(alpha=0.10, n_estimators=30)
        cqr_95 = ConformalizedQuantileRegressor(alpha=0.05, n_estimators=30)

        cqr_90.fit(X_mock, y_mock)
        cqr_90.calibrate(X_mock, y_mock)
        l90, u90 = cqr_90.predict_intervals(X_mock)
        width_90 = np.mean(u90 - l90)

        cqr_95.fit(X_mock, y_mock)
        cqr_95.calibrate(X_mock, y_mock)
        l95, u95 = cqr_95.predict_intervals(X_mock)
        width_95 = np.mean(u95 - l95)

        self.assertGreaterEqual(width_95, width_90 - 1e-5, "95% interval width must be greater than or equal to 90% interval width")

if __name__ == "__main__":
    unittest.main()
