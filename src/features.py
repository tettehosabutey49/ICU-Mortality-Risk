"""
Domain-driven feature engineering for the ICU Mortality Risk dataset.

Each feature is grounded in clinical reasoning — if a feature cannot be
explained to a clinician, it should not be in the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Clinical constants                                                            #
# --------------------------------------------------------------------------- #

# APACHE II scores ≥ 25 are associated with a predicted mortality > 50 %
# (Knaus et al., 1985).  Used to create a binary severity flag.
APACHE_HIGH_RISK_THRESHOLD = 25

# Normal range for age-adjusted BMI categories (WHO classification)
BMI_UNDERWEIGHT = 18.5
BMI_OVERWEIGHT = 25.0
BMI_OBESE = 30.0

# Shock index > 1.0 indicates haemodynamic compromise (Rady et al., 1994)
SHOCK_INDEX_THRESHOLD = 1.0


def add_apache_risk_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flag: 1 if APACHE II score ≥ 25 (high predicted mortality).

    APACHE II is the standard ICU severity score.  Binarising it creates
    a stable interaction term that many tree models find easy to split on.
    """
    df = df.copy()
    if "apache_2_diagnosis" in df.columns:
        df["high_apache_risk"] = (df["apache_2_diagnosis"] >= APACHE_HIGH_RISK_THRESHOLD).astype(int)
    return df


def add_shock_index(df: pd.DataFrame) -> pd.DataFrame:
    """Shock index = heart_rate / systolic_bp.

    Elevated shock index predicts haemodynamic instability independent of
    either vital sign alone; it is a clinically validated composite marker.
    Capped at 5 to suppress division-by-near-zero artefacts.
    """
    df = df.copy()
    hr_col = next((c for c in df.columns if "heart_rate" in c and "apache" not in c), None)
    sbp_col = next((c for c in df.columns if "sys_bp" in c or "sysbp" in c.lower()), None)

    if hr_col and sbp_col:
        denom = df[sbp_col].replace(0, np.nan)
        df["shock_index"] = (df[hr_col] / denom).clip(upper=5.0)
        df["shock_index_high"] = (df["shock_index"] > SHOCK_INDEX_THRESHOLD).astype(int)
    return df


def add_bmi_category(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal BMI category (0=underweight, 1=normal, 2=overweight, 3=obese).

    Continuous BMI has a non-linear U-shaped relationship with ICU mortality;
    categorisation captures both extremes more explicitly.
    """
    df = df.copy()
    if "bmi" in df.columns:
        df["bmi_category"] = pd.cut(
            df["bmi"],
            bins=[-np.inf, BMI_UNDERWEIGHT, BMI_OVERWEIGHT, BMI_OBESE, np.inf],
            labels=[0, 1, 2, 3],
        ).astype(float)
    return df


def add_age_apache_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """Multiplicative interaction between age and APACHE II score.

    Older patients with high APACHE scores have disproportionately higher
    mortality than would be predicted by either variable independently.
    """
    df = df.copy()
    age_col = "age" if "age" in df.columns else None
    apache_col = "apache_2_diagnosis" if "apache_2_diagnosis" in df.columns else None

    if age_col and apache_col:
        df["age_x_apache"] = df[age_col] * df[apache_col]
    return df


def add_vital_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Add min-max range features for repeated vitals (e.g. HR, BP, SpO2).

    The *range* (max - min) over the first 24 h captures physiological
    variability, which is a known independent predictor of ICU mortality.
    """
    df = df.copy()
    vital_pairs = [
        ("d1_heartrate_min", "d1_heartrate_max", "hr_range_d1"),
        ("d1_sysbp_min", "d1_sysbp_max", "sbp_range_d1"),
        ("d1_spo2_min", "d1_spo2_max", "spo2_range_d1"),
        ("d1_temp_min", "d1_temp_max", "temp_range_d1"),
    ]
    for lo, hi, new_col in vital_pairs:
        if lo in df.columns and hi in df.columns:
            df[new_col] = df[hi] - df[lo]
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps in the correct order.

    Single entry-point used by notebooks to keep the feature pipeline
    reproducible and version-controlled in one place.
    """
    df = add_apache_risk_flag(df)
    df = add_shock_index(df)
    df = add_bmi_category(df)
    df = add_age_apache_interaction(df)
    df = add_vital_ranges(df)
    return df
