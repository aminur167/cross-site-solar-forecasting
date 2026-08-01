"""
Few-Shot Domain Adaptation Module for Cross-Site Solar Forecasting
Manages target-site data budgets (k in {0, 1, 3, 7, 14} days), sample weighting,
and transfer adaptation regimes.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class FewShotDomainAdapter:
    """
    Orchestrates data-efficient transfer learning from Source PV plant to Target PV plant.
    """
    def __init__(self, target_budget_days: int = 7, target_weight: float = 3.0):
        self.target_budget_days = target_budget_days
        self.target_weight = target_weight

    @staticmethod
    def extract_budget_subset(df: pd.DataFrame, budget_days: int) -> pd.DataFrame:
        """
        Extracts exactly budget_days consecutive calendar days from the beginning of df.
        """
        if budget_days <= 0:
            return pd.DataFrame(columns=df.columns)

        start_time = df.index.min()
        end_time = start_time + pd.Timedelta(days=budget_days)
        budget_subset = df.loc[start_time:end_time].copy()
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
        Target samples are given higher weights to prioritize local inverter/orientation characteristics.
        """
        if len(X_target_budget) == 0:
            # Zero-shot case
            weights = np.ones(len(X_source), dtype=float)
            return X_source.copy(), y_source.copy(), weights

        X_combined = pd.concat([X_source, X_target_budget], axis=0)
        y_combined = pd.concat([y_source, y_target_budget], axis=0)

        # Source weights = 1.0, Target weights = target_weight
        source_weights = np.ones(len(X_source), dtype=float)
        target_weights = np.full(len(X_target_budget), fill_value=self.target_weight, dtype=float)
        sample_weights = np.concatenate([source_weights, target_weights])

        return X_combined, y_combined, sample_weights
