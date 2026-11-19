"""
Conditional Conformal Risk and Coverage Auditor
================================================
Evaluates localized and conditional coverage across meteorological sub-domains:
- Low-irradiance overcast conditions (GHI < 200 W/m^2)
- Moderate intermittent cloudiness (200 <= GHI < 600 W/m^2)
- High-irradiance clear sky (GHI >= 600 W/m^2)
Verifies that Conformalized Quantile Regression avoids localized conditional undercoverage.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd


class ConformalRiskAuditor:
    """Audits conditional coverage across irradiance tiers and diurnal segments."""

    IRRADIANCE_BINS = [
        ("Overcast / Low Light", 0.0, 200.0),
        ("Intermittent / Transition", 200.0, 600.0),
        ("Clear Sky / Peak Solar", 600.0, 1500.0),
    ]

    def audit_conditional_coverage(
        self,
        y_true: np.ndarray,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        ghi_values: np.ndarray,
        nominal_coverage: float = 0.90,
    ) -> pd.DataFrame:
        """
        Calculates stratified coverage and sharpness across irradiance regimes.
        """
        results = []

        # Overall marginal
        covered_all = ((y_true >= lower_bound) & (y_true <= upper_bound)).mean()
        mpiw_all = (upper_bound - lower_bound).mean()
        results.append({
            "Regime": "All Daytime (Marginal)",
            "Samples": len(y_true),
            "Nominal (%)": nominal_coverage * 100.0,
            "Empirical PICP (%)": round(covered_all * 100.0, 2),
            "MPIW (CF)": round(float(mpiw_all), 4),
            "Coverage Valid": covered_all >= nominal_coverage - 0.02
        })

        # Stratified regimes
        for label, low, high in self.IRRADIANCE_BINS:
            mask = (ghi_values >= low) & (ghi_values < high)
            if mask.sum() == 0:
                continue

            sub_true = y_true[mask]
            sub_low = lower_bound[mask]
            sub_high = upper_bound[mask]

            sub_covered = ((sub_true >= sub_low) & (sub_true <= sub_high)).mean()
            sub_mpiw = (sub_high - sub_low).mean()

            results.append({
                "Regime": label,
                "Samples": int(mask.sum()),
                "Nominal (%)": nominal_coverage * 100.0,
                "Empirical PICP (%)": round(sub_covered * 100.0, 2),
                "MPIW (CF)": round(float(sub_mpiw), 4),
                "Coverage Valid": sub_covered >= nominal_coverage - 0.05
            })

        return pd.DataFrame(results)


if __name__ == "__main__":
    n = 1000
    np.random.seed(42)
    ghi = np.random.uniform(50, 950, n)
    y = np.clip(ghi / 1000.0 + np.random.normal(0, 0.03, n), 0, 1)
    low = np.clip(y - 0.12, 0, 1)
    high = np.clip(y + 0.12, 0, 1)

    auditor = ConformalRiskAuditor()
    df_audit = auditor.audit_conditional_coverage(y, low, high, ghi)
    print("[INFO] Conditional Conformal Risk Audit:")
    print(df_audit.to_string(index=False))
