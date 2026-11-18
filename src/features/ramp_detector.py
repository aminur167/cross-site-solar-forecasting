"""
Solar Ramp Event Detection and Metric Evaluation
===============================================
Detects high-impact solar ramp-rate events (sudden power spikes or cloud drops)
and evaluates forecaster ramp-capture sensitivity for dynamic grid reserve scheduling.
Reference: Standard IEC 61724-1 and North American Electric Reliability Corp (NERC).
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd


class SolarRampDetector:
    """
    Identifies ramp-rate events where |P(t) - P(t-1)| >= threshold * Capacity.
    Default threshold is 20% capacity change per hour (ramp rate >= 0.20 CF/h).
    """

    def __init__(self, ramp_threshold_cf: float = 0.20, rated_capacity_kw: float = 50.0):
        self.ramp_threshold_cf = ramp_threshold_cf
        self.rated_capacity_kw = rated_capacity_kw

    def detect_ramps(self, power_series: pd.Series) -> pd.DataFrame:
        """
        Flags ramp events:
        - ramp_up: delta >= threshold
        - ramp_down: delta <= -threshold
        """
        delta = power_series.diff().fillna(0.0)
        df_ramps = pd.DataFrame(index=power_series.index)
        df_ramps["power"] = power_series
        df_ramps["delta"] = delta
        df_ramps["is_ramp_up"] = delta >= self.ramp_threshold_cf
        df_ramps["is_ramp_down"] = delta <= -self.ramp_threshold_cf
        df_ramps["is_ramp_event"] = df_ramps["is_ramp_up"] | df_ramps["is_ramp_down"]
        return df_ramps

    def evaluate_ramp_forecasting(
        self,
        y_true: pd.Series,
        y_pred: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluates contingency metrics for ramp forecasting:
        - Ramp Capture Rate (True Positive Rate / Recall for ramps)
        - Ramp False Alarm Rate
        - Ramp F1 Score
        """
        ramps_true = self.detect_ramps(y_true)
        ramps_pred = self.detect_ramps(y_pred)

        true_pos = (ramps_true["is_ramp_event"] & ramps_pred["is_ramp_event"]).sum()
        false_neg = (ramps_true["is_ramp_event"] & ~ramps_pred["is_ramp_event"]).sum()
        false_pos = (~ramps_true["is_ramp_event"] & ramps_pred["is_ramp_event"]).sum()
        true_neg = (~ramps_true["is_ramp_event"] & ~ramps_pred["is_ramp_event"]).sum()

        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "total_observed_ramps": int(ramps_true["is_ramp_event"].sum()),
            "ramp_capture_rate_recall": round(float(recall), 4),
            "ramp_precision": round(float(precision), 4),
            "ramp_f1_score": round(float(f1), 4),
            "ramp_false_alarm_rate": round(float(false_pos / (false_pos + true_neg) if (false_pos + true_neg) > 0 else 0.0), 4)
        }


if __name__ == "__main__":
    dates = pd.date_range("2026-06-01", periods=100, freq="1h")
    cf_true = pd.Series(np.clip(np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.05, 100), 0, 1), index=dates)
    cf_pred = cf_true + np.random.normal(0, 0.04, 100)

    detector = SolarRampDetector(ramp_threshold_cf=0.15)
    metrics = detector.evaluate_ramp_forecasting(cf_true, cf_pred)
    print("[INFO] Ramp Event Forecast Evaluation:", metrics)
