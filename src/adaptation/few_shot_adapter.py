"""
Few-Shot Domain Adaptation Module for Cross-Site Solar Forecasting
=================================================================
Implements domain-weighted sample adaptation to transfer forecasting representations
from a data-rich Source PV plant to a newly commissioned Target PV plant.

Protocol:
- Extracts exact right-open calendar budgets [t_start, t_start + k days).
- Pairs Source telemetry with sample-weighted Target telemetry.
- Evaluates on chronologically subsequent holdout test partition.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class FewShotDomainAdapter:
    """
    Orchestrates data-efficient domain-weighted transfer learning from Source PV to Target PV.
    """

    def __init__(self, target_budget_days: int = 7, target_weight: float = 2.5):
        self.target_budget_days = target_budget_days
        self.target_weight = target_weight

    @staticmethod
    def extract_budget_subset(df: pd.DataFrame, budget_days: int) -> pd.DataFrame:
        """
        Extracts exactly budget_days consecutive calendar days from the beginning of df.
        STRICT RIGHT-OPEN INTERVAL: [start_time, start_time + k days).
        Guarantees exact calendar day bounds with ZERO boundary-hour leakage.
        """
        if budget_days <= 0 or len(df) == 0:
            return pd.DataFrame(columns=df.columns)

        start_time = df.index.min()
        end_time = start_time + pd.Timedelta(days=budget_days)

        # Right-open slice avoids including the 25th boundary hour on Day 1
        budget_subset = df.loc[(df.index >= start_time) & (df.index < end_time)].copy()
        return budget_subset

    def prepare_adaptation_dataset(
        self,
        X_source: pd.DataFrame,
        y_source: pd.Series,
        X_target_budget: pd.DataFrame,
        y_target_budget: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """
        Constructs combined adaptation training set with domain importance weighting.
        Source samples receive weight = 1.0; Target budget samples receive weight = target_weight.
        Target weighting prioritizes local inverter efficiency and orientation characteristics.
        """
        if len(X_target_budget) == 0:
            # Zero-shot transfer inference: Source data alone
            weights = np.ones(len(X_source), dtype=float)
            return X_source.copy(), y_source.copy(), weights

        X_combined = pd.concat([X_source, X_target_budget], axis=0)
        y_combined = pd.concat([y_source, y_target_budget], axis=0)

        source_weights = np.ones(len(X_source), dtype=float)
        target_weights = np.full(len(X_target_budget), fill_value=self.target_weight, dtype=float)
        sample_weights = np.concatenate([source_weights, target_weights])

        return X_combined, y_combined, sample_weights


if __name__ == "__main__":
    dates = pd.date_range("2026-01-01", periods=100, freq="1h")
    df = pd.DataFrame({"val": range(100)}, index=dates)
    sub = FewShotDomainAdapter.extract_budget_subset(df, budget_days=1)
    print(f"[TEST] 1-day budget extraction length: {len(sub)} (expected 24).")
    assert len(sub) == 24, f"Expected 24 hours, got {len(sub)}"
    print("[INFO] FewShotDomainAdapter verified successfully.")
