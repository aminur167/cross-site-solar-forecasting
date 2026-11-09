"""
Seasonal and Monthly Performance Decomposition
=============================================
Computes monthly and seasonal breakdown of point forecast errors (nMAE, nRMSE)
and conformal uncertainty metrics (PICP, MPIW) to verify temporal robustness
across varied meteorological regimes (monsoon, winter fog, clear summer).
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class SeasonalEvaluator:
    """Evaluates forecaster across 12 calendar months and 4 meteorological seasons."""

    SEASON_MAP = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Pre-Monsoon/Spring", 4: "Pre-Monsoon/Spring", 5: "Pre-Monsoon/Spring",
        6: "Monsoon/Summer", 7: "Monsoon/Summer", 8: "Monsoon/Summer",
        9: "Post-Monsoon/Autumn", 10: "Post-Monsoon/Autumn", 11: "Post-Monsoon/Autumn"
    }

    MONTH_NAMES = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    }

    def evaluate_monthly(
        self,
        df_results: pd.DataFrame,
        y_true_col: str = "actual",
        y_pred_col: str = "predicted",
        lower_col: Optional[str] = "lower_bound",
        upper_col: Optional[str] = "upper_bound",
        capacity: float = 1.0,
    ) -> pd.DataFrame:
        """
        Calculates granular performance metrics segmented by calendar month.
        df_results must have a DatetimeIndex or 'datetime_utc' column.
        """
        df = df_results.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            if "datetime_utc" in df.columns:
                df.index = pd.to_datetime(df["datetime_utc"])
            else:
                raise ValueError("DataFrame must have a DatetimeIndex or 'datetime_utc' column.")

        df["month"] = df.index.month
        df["month_name"] = df["month"].map(self.MONTH_NAMES)
        df["season"] = df["month"].map(self.SEASON_MAP)

        records = []
        for month_num in sorted(df["month"].unique()):
            sub = df[df["month"] == month_num]
            if len(sub) == 0:
                continue

            yt = sub[y_true_col].values
            yp = sub[y_pred_col].values

            mae = mean_absolute_error(yt, yp)
            rmse = np.sqrt(mean_squared_error(yt, yp))
            nmae = (mae / capacity) * 100.0
            nrmse = (rmse / capacity) * 100.0
            r2 = r2_score(yt, yp) if len(np.unique(yt)) > 1 else 0.0

            rec = {
                "month": month_num,
                "month_name": self.MONTH_NAMES[month_num],
                "season": self.SEASON_MAP[month_num],
                "samples": len(sub),
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
                "nMAE_pct": round(nmae, 2),
                "nRMSE_pct": round(nrmse, 2),
                "R2": round(r2, 4),
            }

            if lower_col in sub.columns and upper_col in sub.columns:
                lb = sub[lower_col].values
                ub = sub[upper_col].values
                covered = ((yt >= lb) & (yt <= ub)).sum()
                picp = (covered / len(yt)) * 100.0
                mpiw = np.mean(ub - lb)
                rec["PICP_pct"] = round(picp, 2)
                rec["MPIW"] = round(mpiw, 4)

            records.append(rec)

        return pd.DataFrame(records)


if __name__ == "__main__":
    dates = pd.date_range("2026-01-01", periods=24 * 365, freq="1h")
    y_true = np.clip(np.sin(np.linspace(0, 50, len(dates))) + 0.3, 0.0, 1.0)
    y_pred = np.clip(y_true + np.random.normal(0, 0.05, len(dates)), 0.0, 1.0)
    df_sample = pd.DataFrame({"actual": y_true, "predicted": y_pred}, index=dates)

    evaluator = SeasonalEvaluator()
    monthly_df = evaluator.evaluate_monthly(df_sample)
    print("[INFO] Monthly evaluation preview:")
    print(monthly_df[["month_name", "season", "nMAE_pct", "R2"]].head())
