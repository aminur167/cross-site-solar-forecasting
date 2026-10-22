"""
Optuna Automated Hyperparameter Optimization for Solar Forecasters
==================================================================
Implements time-series walk-forward hyperparameter optimization using Optuna
with Bayesian TPE (Tree-structured Parzen Estimator) search.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

# Suppress Optuna verbose logging by default
optuna.logging.set_verbosity(optuna.logging.WARNING)


class SolarHyperparameterTuner:
    """Walk-forward Bayesian hyperparameter optimizer for LightGBM solar forecasters."""

    def __init__(self, n_trials: int = 25, seed: int = 42):
        self.n_trials = n_trials
        self.seed = seed
        self.best_params: Dict[str, Any] = {}
        self.best_score: float = float("inf")

    def objective(
        self,
        trial: optuna.Trial,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> float:
        """Optuna objective function minimizing validation nMAE."""
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 350, step=20),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.1),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": self.seed,
            "verbose": -1,
            "n_jobs": 1,
        }

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        preds = np.clip(preds, 0.0, 1.2)

        # Calculate normalized MAE (percentage of capacity, capacity = 1.0)
        nmae = mean_absolute_error(y_val, preds) * 100.0
        return float(nmae)

    def tune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Dict[str, Any]:
        """Execute Bayesian optimization study."""
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)

        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
            n_trials=self.n_trials,
            show_progress_bar=False,
        )

        self.best_params = study.best_params
        self.best_params["random_state"] = self.seed
        self.best_params["verbose"] = -1
        self.best_params["n_jobs"] = 1
        self.best_score = study.best_value

        return self.best_params


if __name__ == "__main__":
    print("[INFO] SolarHyperparameterTuner loaded successfully.")
