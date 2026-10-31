"""
Solar PV Telemetry Schema Validator
===================================
Enforces strict schema validation, physical boundary constraints, and temporal
monotonicity for multi-site solar PV and weather telemetry streams.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np


@dataclass
class TelemetryRecord:
    """Individual solar generation and meteorological observation record."""
    datetime_utc: str
    active_power_kw: float
    solar_elevation_deg: float
    ghi_wm2: Optional[float] = None
    dni_wm2: Optional[float] = None
    ambient_temp_c: Optional[float] = None
    relative_humidity_pct: Optional[float] = None
    wind_speed_ms: Optional[float] = None


class TelemetrySchemaValidator:
    """
    Validates solar telemetry DataFrames against physical laws:
    - GHI >= 0 and GHI <= 1400 W/m^2 (solar constant limit)
    - Active power >= 0
    - Solar elevation between -90 and +90 degrees
    - Strictly monotonic, non-duplicated timestamp index
    """

    REQUIRED_COLS = ["datetime_utc", "active_power_kw", "solar_elevation_deg"]

    def __init__(self, rated_capacity_kw: float = 50.0):
        self.rated_capacity_kw = rated_capacity_kw

    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Runs validation suite against incoming DataFrame.
        Returns (is_valid, list_of_violations).
        """
        violations = []

        # 1. Required column presence
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            violations.append(f"Missing required columns: {missing}")
            return False, violations

        # 2. Datetime index / column sanity
        if "datetime_utc" in df.columns:
            dt_series = pd.to_datetime(df["datetime_utc"], errors="coerce")
            if dt_series.isna().any():
                violations.append("Contains unparseable datetime entries.")
            if not dt_series.is_monotonic_increasing:
                violations.append("Datetime entries are not strictly monotonic increasing.")
            if dt_series.duplicated().any():
                violations.append("Contains duplicate datetime stamps.")

        # 3. Physical power bounds
        neg_power = (df["active_power_kw"] < -1e-3).sum()
        if neg_power > 0:
            violations.append(f"Found {neg_power} records with negative active power.")

        excess_power = (df["active_power_kw"] > self.rated_capacity_kw * 1.5).sum()
        if excess_power > 0:
            violations.append(f"Found {excess_power} records exceeding 150% rated capacity.")

        # 4. Solar elevation bounds
        elev = df["solar_elevation_deg"]
        if (elev < -90.0).any() or (elev > 90.0).any():
            violations.append("Solar elevation angles out of [-90, +90] degree bounds.")

        # 5. GHI bounds if present
        if "ghi_wm2" in df.columns:
            ghi = df["ghi_wm2"].dropna()
            if (ghi < 0.0).any():
                violations.append("Contains negative GHI solar irradiance values.")
            if (ghi > 1450.0).any():
                violations.append("Contains GHI exceeding extraterrestrial solar constant (1450 W/m^2).")

        is_valid = len(violations) == 0
        return is_valid, violations


if __name__ == "__main__":
    validator = TelemetrySchemaValidator()
    print("[INFO] TelemetrySchemaValidator loaded.")
