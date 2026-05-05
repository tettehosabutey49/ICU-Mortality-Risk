"""
Domain-driven feature engineering for the ICU Mortality Risk dataset.

Each transformation is grounded in clinical reasoning.  If a feature cannot be
explained to an intensivist, it does not belong in the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constants — named so every threshold is auditable                             #
# --------------------------------------------------------------------------- #

# Binary comorbidity flags: NULL in EHR means absence, not unknown.
# Standard clinical coding convention (ICD guidelines).
COMORBIDITY_COLS = [
    "aids", "cirrhosis", "diabetes_mellitus", "hepatic_failure",
    "immunosuppression", "leukemia", "lymphoma", "solid_tumor_with_metastasis",
]

# High-risk comorbidities: OR > 2.0 from EDA forest plot
HIGH_RISK_COMORBIDITIES = [
    "cirrhosis", "hepatic_failure", "solid_tumor_with_metastasis", "aids",
]

# Vitals vary by ICU care setting — grouped imputation preserves those norms
VITAL_COLS = [
    "d1_heartrate_max", "d1_heartrate_min",
    "d1_sysbp_max", "d1_sysbp_min",
    "d1_diasbp_max", "d1_diasbp_min",
    "d1_mbp_max", "d1_mbp_min",
    "d1_resprate_max", "d1_resprate_min",
    "d1_spo2_max", "d1_spo2_min",
    "d1_temp_max", "d1_temp_min",
    "h1_heartrate_max", "h1_heartrate_min",
    "h1_sysbp_max", "h1_sysbp_min",
    "h1_mbp_max", "h1_mbp_min",
    "h1_resprate_max", "h1_resprate_min",
    "h1_spo2_max", "h1_spo2_min",
]

# Lab reference ranges differ by diagnostic group — group imputation preserves that
LAB_COLS = [
    "d1_glucose_max", "d1_glucose_min",
    "d1_potassium_max", "d1_potassium_min",
    "d1_sodium_max", "d1_sodium_min",
    "d1_creatinine_max", "d1_creatinine_min",
    "d1_bun_max", "d1_bun_min",
    "d1_wbc_max", "d1_wbc_min",
    "d1_hematocrit_max", "d1_hematocrit_min",
    "d1_hemaglobin_max", "d1_hemaglobin_min",
    "d1_lactate_max", "d1_lactate_min",
    "d1_calcium_max", "d1_calcium_min",
]

# Pure row identifiers — zero predictive signal, only leakage risk
IDENTIFIER_COLS = ["encounter_id", "patient_id", "hospital_id", "icu_id"]

# Categoricals that receive target encoding (not one-hot — see docstring)
CATEGORICAL_COLS = [
    "ethnicity", "icu_type", "icu_admit_source",
    "apache_3j_bodysystem", "apache_2_bodysystem",
]

# Feature-selection thresholds
HIGH_MISSINGNESS_THRESHOLD = 0.40   # drop after domain imputation is done
CORRELATION_THRESHOLD = 0.95        # Pearson |r| above which one feature is redundant

# Clinical thresholds
SHOCK_INDEX_THRESHOLD = 1.0         # SI > 1.0 → haemodynamic instability (Rady 1994)
SPO2_HR_CAP = 5.0                   # physiological cap to suppress division artefacts
SHOCK_INDEX_CAP = 5.0

# Age-group boundaries (WHO / critical-care literature conventions)
AGE_BINS = [-np.inf, 45, 65, 80, np.inf]
AGE_LABELS = ["young_adult", "middle_aged", "older_adult", "elderly"]

# Smoothing for target encoder — prevents overfitting on rare categories
TARGET_ENCODING_SMOOTHING = 10.0


# --------------------------------------------------------------------------- #
# 1. Imputation                                                                 #
# --------------------------------------------------------------------------- #

def fill_comorbidities(df: pd.DataFrame) -> pd.DataFrame:
    """Fill binary comorbidity flags with 0 where missing.

    In electronic health records, binary comorbidity fields are left NULL
    when a condition is absent rather than explicitly set to 0.  The clinical
    convention — consistent with ICD coding guidelines — treats a missing flag
    as absence of the documented diagnosis.  Filling with 0 is therefore the
    medically appropriate choice, not a statistical imputation.
    """
    df = df.copy()
    present = [c for c in COMORBIDITY_COLS if c in df.columns]
    df[present] = df[present].fillna(0).astype(int)
    return df


def impute_vitals_by_icu_type(df: pd.DataFrame) -> pd.DataFrame:
    """Impute vital sign missings with the median for the patient's ICU type.

    Physiological baselines differ substantially across care settings.  A resting
    heart rate of 90 bpm is unremarkable in a medical ICU patient but noteworthy
    in a CTICU patient recovering from cardiac surgery.  Stratifying by icu_type
    preserves these distributional differences that a global median would collapse.
    Falls back to global median for any ICU type with too few observations.
    """
    df = df.copy()
    cols = [c for c in VITAL_COLS if c in df.columns]

    if "icu_type" in df.columns:
        group_med = df.groupby("icu_type")[cols].transform("median")
    else:
        group_med = pd.DataFrame(np.nan, index=df.index, columns=cols)

    global_med = df[cols].median()
    for col in cols:
        df[col] = df[col].fillna(group_med[col]).fillna(global_med[col])
    return df


def impute_labs_by_diagnosis(df: pd.DataFrame) -> pd.DataFrame:
    """Impute lab values with the median for the patient's diagnostic group.

    Reference ranges for lab values differ materially by diagnosis category.
    A creatinine of 2.5 mg/dL is expected for a patient with acute renal
    failure but is highly abnormal in an elective cardiac surgery patient.
    Grouping by apache_3j_bodysystem (primary diagnostic system) preserves
    those clinical norms.  Falls back to global median for any unseen group.
    """
    df = df.copy()
    cols = [c for c in LAB_COLS if c in df.columns]
    group_col = "apache_3j_bodysystem" if "apache_3j_bodysystem" in df.columns else None

    if group_col:
        group_med = df.groupby(group_col)[cols].transform("median")
    else:
        group_med = pd.DataFrame(np.nan, index=df.index, columns=cols)

    global_med = df[cols].median()
    for col in cols:
        df[col] = df[col].fillna(group_med[col]).fillna(global_med[col])
    return df


def impute_bmi(df: pd.DataFrame) -> pd.DataFrame:
    """Derive BMI from height/weight where possible, then median-impute residuals.

    Priority:
    1. Recorded BMI — use as-is.
    2. Derive from weight (kg) / height (cm→m)² when both are present and BMI
       is missing — reduces the effective missingness without statistical bias.
    3. Global median for any remaining gaps.

    Derived values outside 10–80 kg/m² are discarded as physiologically implausible.
    """
    df = df.copy()
    if "bmi" not in df.columns:
        return df

    if "weight" in df.columns and "height" in df.columns:
        height_m = df["height"] / 100
        derived = df["weight"] / height_m.pow(2)
        plausible = derived.between(10, 80)
        mask = df["bmi"].isna() & plausible
        df.loc[mask, "bmi"] = derived[mask]

    df["bmi"] = df["bmi"].fillna(df["bmi"].median())
    return df


# --------------------------------------------------------------------------- #
# 2. Derived clinical features                                                  #
# --------------------------------------------------------------------------- #

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all clinically motivated composite features in one pass.

    Every feature is constructed from first-principles physiology.  Rationale
    for each is documented inline.
    """
    df = df.copy()

    # Pulse pressure: SBP_max - DBP_max
    # PP > 60 mmHg → arterial stiffness; PP < 25 mmHg → low stroke volume / shock
    if "d1_sysbp_max" in df.columns and "d1_diasbp_max" in df.columns:
        df["pulse_pressure"] = df["d1_sysbp_max"] - df["d1_diasbp_max"]

    # Mean arterial pressure (derived): (SBP + 2×DBP) / 3
    # MAP < 65 mmHg is the vasopressor threshold in sepsis guidelines (Surviving Sepsis 2021)
    if "d1_sysbp_max" in df.columns and "d1_diasbp_max" in df.columns:
        df["map_derived"] = (df["d1_sysbp_max"] + 2 * df["d1_diasbp_max"]) / 3

    # Shock index: HR_max / SBP_min
    # SI > 1.0 predicts haemodynamic instability; validated ICU triage tool (Rady 1994)
    if "d1_heartrate_max" in df.columns and "d1_sysbp_min" in df.columns:
        denom = df["d1_sysbp_min"].replace(0, np.nan)
        df["shock_index"] = (df["d1_heartrate_max"] / denom).clip(upper=SHOCK_INDEX_CAP)
        df["shock_index_high"] = (df["shock_index"] > SHOCK_INDEX_THRESHOLD).astype(int)

    # SpO2/HR ratio: SpO2_min / HR_max
    # Proxy for oxygen delivery efficiency; lower ratio = more tachycardic per unit
    # of oxygen saturation, suggesting impaired O2 delivery.
    if "d1_spo2_min" in df.columns and "d1_heartrate_max" in df.columns:
        denom = df["d1_heartrate_max"].replace(0, np.nan)
        df["spo2_hr_ratio"] = (df["d1_spo2_min"] / denom).clip(lower=0, upper=SPO2_HR_CAP)

    # Temperature delta: temp_max - temp_min
    # Variability > 2°C suggests inflammatory instability, sepsis, or CNS dysfunction
    if "d1_temp_max" in df.columns and "d1_temp_min" in df.columns:
        df["temp_delta"] = df["d1_temp_max"] - df["d1_temp_min"]

    # Heart rate variability (range): HR_max - HR_min
    # Wide HR range reflects autonomic dysfunction — an independent ICU mortality predictor
    if "d1_heartrate_max" in df.columns and "d1_heartrate_min" in df.columns:
        df["hr_variability"] = df["d1_heartrate_max"] - df["d1_heartrate_min"]

    # Systolic BP variability: SBP_max - SBP_min
    # Large swings indicate haemodynamic instability and vasopressor weaning difficulty
    if "d1_sysbp_max" in df.columns and "d1_sysbp_min" in df.columns:
        df["bp_variability"] = df["d1_sysbp_max"] - df["d1_sysbp_min"]

    # Glucose variability: glucose_max - glucose_min
    # Glycaemic variability is an independent predictor of ICU mortality,
    # particularly in sepsis where stress hyperglycaemia is common (Egi 2006)
    if "d1_glucose_max" in df.columns and "d1_glucose_min" in df.columns:
        df["glucose_variability"] = df["d1_glucose_max"] - df["d1_glucose_min"]

    # Comorbidity burden: count of all active comorbidity flags (0–8)
    comorb_present = [c for c in COMORBIDITY_COLS if c in df.columns]
    if comorb_present:
        df["comorbidity_burden"] = df[comorb_present].fillna(0).sum(axis=1).astype(int)

    # High-risk comorbidity flag: any of the four highest-OR conditions from EDA
    hr_cols = [c for c in HIGH_RISK_COMORBIDITIES if c in df.columns]
    if hr_cols:
        df["high_risk_comorbidity"] = (df[hr_cols].fillna(0).max(axis=1) > 0).astype(int)

    # APACHE score delta: hospital_death_prob - icu_death_prob
    # Positive delta: patient expected to survive ICU but die later in hospital
    # (post-ICU complications, do-not-resuscitate decisions, ward deterioration)
    if "apache_4a_hospital_death_prob" in df.columns and "apache_4a_icu_death_prob" in df.columns:
        df["apache_score_delta"] = (
            df["apache_4a_hospital_death_prob"] - df["apache_4a_icu_death_prob"]
        )

    return df


# --------------------------------------------------------------------------- #
# 3. Categorical encoding                                                       #
# --------------------------------------------------------------------------- #

def add_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Add age_group column for the fairness audit (kept separate, not for modeling).

    Boundaries aligned with WHO and critical-care literature conventions.
    """
    df = df.copy()
    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False
        ).astype(str)
    return df


def encode_gender(df: pd.DataFrame) -> pd.DataFrame:
    """Binary-encode gender: M=1, F=0.

    Binary encoding is appropriate because gender is two-valued in this dataset.
    """
    df = df.copy()
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"M": 1, "F": 0}).fillna(0).astype(int)
    return df


def fit_target_encoder(
    train_df: pd.DataFrame,
    cols: list[str],
    target: str = "hospital_death",
    smoothing: float = TARGET_ENCODING_SMOOTHING,
) -> dict[str, dict]:
    """Fit a smoothed target encoder on the training set only.

    Why target encoding, not one-hot?
    - Some columns (apache_3j_bodysystem, icu_type) have 10–20+ levels.
    - OHE would add ~100 sparse columns that tree models handle inefficiently.
    - Target encoding collapses each category to a single float capturing its
      outcome-level information without dimensionality explosion.
    - Leakage risk is controlled by fitting on training rows only.

    Smoothing formula:
        encoded = (n_cat × mean_cat + smoothing × global_mean) / (n_cat + smoothing)
    where n_cat = count of rows in that category.  Small categories are pulled
    toward the global mean, preventing overfitting on rare levels.
    """
    global_mean = train_df[target].mean()
    maps: dict[str, dict] = {}
    for col in cols:
        if col not in train_df.columns:
            continue
        grp = train_df.groupby(col)[target].agg(["sum", "count"])
        smoothed = (grp["sum"] + smoothing * global_mean) / (grp["count"] + smoothing)
        maps[col] = {"mapping": smoothed.to_dict(), "global_mean": float(global_mean)}
    return maps


def apply_target_encoder(df: pd.DataFrame, encoder_maps: dict[str, dict]) -> pd.DataFrame:
    """Apply a fitted target encoder to any split.

    Unseen categories receive the global mean from the training set,
    preventing NaN bleed-through to model inputs.
    """
    df = df.copy()
    for col, enc in encoder_maps.items():
        if col in df.columns:
            df[col] = df[col].map(enc["mapping"]).fillna(enc["global_mean"])
    return df


# --------------------------------------------------------------------------- #
# 4. Feature selection                                                          #
# --------------------------------------------------------------------------- #

def drop_identifiers(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Remove patient / encounter identifier columns.

    Identifiers carry no predictive signal.  Retaining them risks a model
    memorising hospital_id → mortality rate associations that do not generalise.
    """
    to_drop = [c for c in IDENTIFIER_COLS if c in df.columns]
    return df.drop(columns=to_drop), to_drop


def drop_high_missingness(
    df: pd.DataFrame, threshold: float = HIGH_MISSINGNESS_THRESHOLD
) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns with missing rate > threshold after domain imputation.

    Using 40% here (stricter than the 60% pre-imputation pass in preprocessing.py)
    because after imputing vitals, labs, and comorbidities the remaining
    high-missing columns are specialist measurements (e.g. invasive haemodynamics)
    recorded only for a narrow patient subset.  Imputing them would introduce more
    noise than signal for the ~60-90% of patients who never had the measurement.
    """
    miss = df.isnull().mean()
    to_drop = miss[miss > threshold].index.tolist()
    return df.drop(columns=to_drop), to_drop


def drop_highly_correlated(
    df: pd.DataFrame,
    threshold: float = CORRELATION_THRESHOLD,
    target_col: str = "hospital_death",
) -> tuple[pd.DataFrame, list[tuple[str, str, float]]]:
    """Drop one feature from any pair with |Pearson r| > threshold.

    When two features are near-perfectly correlated they carry the same
    information.  Keeping both wastes model capacity, slows training, and can
    destabilise SHAP values by splitting importance between equivalent features.
    The feature with lower |correlation to target| is dropped, preserving the
    more predictive member of each pair.
    """
    num_df = df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors="ignore")
    corr = num_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    target_corr = (
        df[num_df.columns].corrwith(df[target_col]).abs()
        if target_col in df.columns
        else pd.Series(dtype=float)
    )

    to_drop: set[str] = set()
    pairs: list[tuple[str, str, float]] = []

    for col in upper.columns:
        if col in to_drop:
            continue
        partners = upper[col][upper[col] > threshold].index.tolist()
        for partner in partners:
            if partner in to_drop:
                continue
            r_val = float(upper.loc[partner, col]) if partner in upper.index else float(upper.loc[col, partner])
            keep = col if target_corr.get(col, 0) >= target_corr.get(partner, 0) else partner
            drop = partner if keep == col else col
            to_drop.add(drop)
            pairs.append((keep, drop, round(r_val, 4)))

    return df.drop(columns=list(to_drop)), pairs


# --------------------------------------------------------------------------- #
# 5. Master pipeline                                                            #
# --------------------------------------------------------------------------- #

def build_features(
    df: pd.DataFrame,
    target_col: str = "hospital_death",
    train_index: pd.Index | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run the full feature engineering pipeline end-to-end.

    Args:
        df: Raw DataFrame (output of pd.read_csv on patient_survival.csv).
        target_col: Name of the target column.
        train_index: Index of rows to use for fitting the target encoder.
                     Pass training-set indices to prevent leakage.
                     If None, the entire df is used (safe for EDA/demo only).

    Returns:
        (engineered_df, metadata) where metadata records all encoder maps,
        dropped column lists, and correlation pairs for reproducibility.
    """
    meta: dict = {}
    n_start = df.shape[1]

    # --- Imputation ---
    df = fill_comorbidities(df)
    df = impute_vitals_by_icu_type(df)
    df = impute_labs_by_diagnosis(df)
    df = impute_bmi(df)

    # --- Feature construction ---
    df = add_derived_features(df)
    df = add_age_groups(df)
    df = encode_gender(df)

    # --- Categorical encoding (fit on training rows only) ---
    train_rows = df.loc[train_index] if train_index is not None else df
    cat_present = [c for c in CATEGORICAL_COLS if c in df.columns]
    encoder_maps = fit_target_encoder(train_rows, cat_present, target=target_col)
    df = apply_target_encoder(df, encoder_maps)
    meta["encoder_maps"] = encoder_maps

    # --- Feature selection ---
    df, dropped_ids = drop_identifiers(df)
    meta["dropped_identifiers"] = dropped_ids

    df, dropped_miss = drop_high_missingness(df)
    meta["dropped_high_missing"] = dropped_miss

    df, dropped_pairs = drop_highly_correlated(df, target_col=target_col)
    meta["dropped_correlated_pairs"] = dropped_pairs

    meta["n_features_start"] = n_start
    meta["n_features_final"] = df.shape[1]
    return df, meta
