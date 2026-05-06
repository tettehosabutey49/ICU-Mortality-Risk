"""
ICU Mortality Risk Stratification — Streamlit Dashboard

Four pages via sidebar navigation:
  1. Patient Risk Assessment  — form inputs, gauge, SHAP waterfall, plain-English explanation
  2. Population Analytics     — KPI cards, ICU-type bar, age KDE, SHAP top-10, comorbidity scatter
  3. Fairness Report          — per-group metrics, FNR comparison, plain-English summary
  4. Clinical Q&A             — chat interface powered by ClinicalExplainer (LangChain)

Synthetic patients (n=500) generated at runtime — no CSV required.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=False)

from src.explainability import explain_patient
from src.models import load_model

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ICU Mortality Risk",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Clinical color palette ────────────────────────────────────────────────────

C_BLUE_DARK  = "#1a3a5c"
C_BLUE_MID   = "#2e7bcf"
C_BLUE_LIGHT = "#dce8f5"
C_GRAY_BG    = "#f0f4f8"
C_RED        = "#c0392b"
C_AMBER      = "#d68910"
C_GREEN      = "#1e8449"

st.markdown(f"""
<style>
  section[data-testid="stSidebar"] {{ background-color: {C_BLUE_DARK}; }}
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] p,
  section[data-testid="stSidebar"] span {{ color: #cfe0f5 !important; }}
  section[data-testid="stSidebar"] h1,
  section[data-testid="stSidebar"] h2 {{ color: white !important; }}
  .kpi-card {{
    background: white;
    border-left: 5px solid {C_BLUE_MID};
    padding: .9rem 1.1rem;
    border-radius: 6px;
    box-shadow: 0 1px 4px rgba(0,0,0,.08);
    margin-bottom: .4rem;
  }}
  .kpi-label {{
    font-size: .72rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: .06em; color: #6c757d;
  }}
  .kpi-value {{ font-size: 2rem; font-weight: 700; color: {C_BLUE_DARK}; line-height: 1.15; }}
  .risk-badge {{
    display: inline-block; padding: .35rem 1.1rem; border-radius: 20px;
    font-weight: 700; font-size: 1rem; letter-spacing: .04em;
  }}
  .footer {{
    border-top: 1px solid #dce3ea; text-align: center; color: #9aa5b4;
    font-size: .72rem; padding: 2rem 0 .5rem; margin-top: 3rem;
  }}
</style>
""", unsafe_allow_html=True)

# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource
def get_model():
    try:
        return load_model("lgbm_best")
    except FileNotFoundError:
        st.error("No trained model found. Run notebook 03_modeling.ipynb first.")
        st.stop()


def _feat_names(mdl) -> list[str]:
    for attr in ("feature_names_in_", "feature_name_"):
        if hasattr(mdl, attr):
            v = getattr(mdl, attr)
            return list(v() if callable(v) else v)
    try:
        return mdl.booster_.feature_name()
    except Exception:
        return []


@st.cache_data
def generate_cohort(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """500 synthetic ICU admissions with realistic clinical distributions."""
    rng = np.random.default_rng(seed)
    mdl = get_model()
    fnames = _feat_names(mdl)

    age    = np.clip(rng.normal(62, 16, n), 18, 95).round(1)
    gender = rng.choice([0.0, 1.0], n, p=[0.45, 0.55])
    bmi    = np.clip(rng.normal(27.5, 6.5, n), 15, 60).round(1)
    apache = np.clip(rng.normal(16, 8, n), 0, 50).round(1)

    hr      = np.clip(rng.normal(88, 20, n), 40, 180).round(1)
    hr_min  = np.clip(hr - np.abs(rng.normal(15, 6, n)), 30, hr)
    hr_max  = np.clip(hr + np.abs(rng.normal(20, 8, n)), hr, 220)
    sbp     = np.clip(rng.normal(120, 22, n), 60, 220).round(1)
    sbp_min = np.clip(sbp - np.abs(rng.normal(20, 10, n)), 40, sbp)
    sbp_max = np.clip(sbp + np.abs(rng.normal(25, 12, n)), sbp, 280)
    dbp     = np.clip(rng.normal(65, 15, n), 30, 130).round(1)
    spo2    = np.clip(rng.normal(96.5, 3, n), 70, 100).round(1)
    temp    = np.clip(rng.normal(37.0, 0.8, n), 34.5, 41.0).round(2)
    rr      = np.clip(rng.normal(20, 6, n), 6, 50).round(1)
    gluc    = np.clip(rng.normal(145, 55, n), 60, 500).round(1)

    diabetes  = rng.binomial(1, 0.28, n).astype(float)
    hf        = rng.binomial(1, 0.12, n).astype(float)
    cirrhosis = rng.binomial(1, 0.04, n).astype(float)
    immuno    = rng.binomial(1, 0.08, n).astype(float)
    leukemia  = rng.binomial(1, 0.03, n).astype(float)
    lymphoma  = rng.binomial(1, 0.03, n).astype(float)
    aids      = rng.binomial(1, 0.02, n).astype(float)
    solid_tm  = rng.binomial(1, 0.04, n).astype(float)
    comorb    = diabetes + hf + cirrhosis + immuno + leukemia + lymphoma + aids

    KNOWN: dict = {
        "age": age, "gender": gender, "bmi": bmi,
        "apache_2_diagnosis": apache, "apache_3j_diagnosis": apache,
        "apache_4a_hospital_death_prob": np.clip(apache / 50, 0, 1),
        "d1_heartrate_mean": hr, "d1_heartrate_min": hr_min, "d1_heartrate_max": hr_max,
        "h1_heartrate_mean": hr * rng.uniform(0.9, 1.1, n),
        "d1_sysbp_mean": sbp, "d1_sysbp_min": sbp_min, "d1_sysbp_max": sbp_max,
        "d1_diasbp_mean": dbp, "d1_mbp_mean": (sbp + 2 * dbp) / 3,
        "d1_spo2_mean": spo2, "d1_spo2_min": spo2 - np.abs(rng.normal(2, 1, n)),
        "d1_temp_mean": temp, "d1_temp_min": temp - np.abs(rng.normal(0.3, 0.1, n)),
        "d1_resprate_mean": rr, "d1_glucose_mean": gluc,
        "d1_glucose_min": gluc * 0.85, "d1_glucose_max": gluc * 1.25,
        "diabetes_mellitus": diabetes, "heart_failure": hf, "cirrhosis": cirrhosis,
        "immunosuppression": immuno, "leukemia": leukemia, "lymphoma": lymphoma,
        "aids": aids, "solid_tumor_with_metastasis": solid_tm,
        "comorbidity_burden": comorb,
        "shock_index":    hr / np.maximum(sbp, 1),
        "pulse_pressure": sbp - dbp,
        "spo2_hr_ratio":  spo2 / np.maximum(hr, 1),
        "bp_variability": sbp_max - sbp_min,
        "hr_variability": hr_max - hr_min,
        "glucose_variability": np.abs(rng.normal(0, 40, n)),
        "temp_variability":    np.abs(rng.normal(0, 0.4, n)),
        "apache_score_delta":  rng.normal(0, 4, n),
        "weight": bmi * (1.75 ** 2),
    }

    if fnames:
        df = pd.DataFrame({f: KNOWN.get(f, rng.standard_normal(n)) for f in fnames})
        X_pred = df
    else:
        df = pd.DataFrame(KNOWN)
        X_pred = df

    y_prob = mdl.predict_proba(X_pred)[:, 1]
    df["mortality_prob"] = y_prob
    df["y_pred"]         = (y_prob >= 0.35).astype(int)
    df["hospital_death"] = np.random.default_rng(seed + 1).binomial(1, y_prob)

    df["icu_type"]   = rng.choice(
        ["Med-Surg ICU", "MICU", "SICU", "CCU", "Cardiac ICU", "CSICU", "NSICU"], n)
    df["ethnicity"]  = rng.choice(
        ["Caucasian", "African American", "Hispanic", "Asian", "Other/Unknown"],
        n, p=[0.55, 0.18, 0.12, 0.08, 0.07])
    df["age_group"]  = pd.cut(
        age, bins=[-np.inf, 45, 65, 80, np.inf],
        labels=["young_adult", "middle_aged", "older_adult", "elderly"],
        right=False,
    ).astype(str)
    df["gender_str"] = np.where(gender == 1, "Male", "Female")
    return df


@st.cache_data
def get_shap_importance() -> pd.DataFrame:
    """Mean |SHAP| per feature across the synthetic cohort."""
    from src.explainability import compute_shap_values
    mdl    = get_model()
    fnames = _feat_names(mdl)
    df     = generate_cohort()
    X      = df[fnames] if fnames else df.drop(
        columns=["mortality_prob", "y_pred", "hospital_death",
                 "icu_type", "ethnicity", "age_group", "gender_str"], errors="ignore")
    sv = compute_shap_values(mdl, X)
    return (
        pd.DataFrame({"feature": list(X.columns),
                      "mean_abs_shap": np.abs(sv.values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_patient_row(inputs: dict) -> pd.DataFrame:
    """Single-row DataFrame from form inputs; unspecified features filled with cohort means."""
    mdl    = get_model()
    fnames = _feat_names(mdl)
    cohort = generate_cohort()
    X_coh  = cohort[fnames] if fnames else cohort.drop(
        columns=["mortality_prob", "y_pred", "hospital_death",
                 "icu_type", "ethnicity", "age_group", "gender_str"], errors="ignore")
    base = X_coh.mean().to_dict()

    comorb = sum(float(inputs.get(k, 0)) for k in
                 ["diabetes_mellitus", "heart_failure", "cirrhosis", "immunosuppression", "aids"])
    base.update({
        "age": inputs["age"],
        "gender": 1.0 if inputs["gender"] == "Male" else 0.0,
        "bmi": inputs["bmi"],
        "apache_2_diagnosis": inputs["apache"],
        "apache_3j_diagnosis": inputs["apache"],
        "apache_4a_hospital_death_prob": inputs["apache"] / 50,
        "d1_heartrate_mean": inputs["hr"],
        "d1_heartrate_min": inputs["hr"] * 0.85,
        "d1_heartrate_max": inputs["hr"] * 1.15,
        "h1_heartrate_mean": inputs["hr"],
        "d1_sysbp_mean": inputs["sbp"],
        "d1_sysbp_min":  inputs["sbp"] * 0.85,
        "d1_sysbp_max":  inputs["sbp"] * 1.15,
        "d1_diasbp_mean": inputs["dbp"],
        "d1_mbp_mean": (inputs["sbp"] + 2 * inputs["dbp"]) / 3,
        "d1_spo2_mean": inputs["spo2"],
        "d1_spo2_min":  inputs["spo2"] - 3,
        "d1_temp_mean": inputs["temp"],
        "d1_resprate_mean": inputs["rr"],
        "d1_glucose_mean": inputs["glucose"],
        "diabetes_mellitus": float(inputs.get("diabetes_mellitus", 0)),
        "heart_failure":     float(inputs.get("heart_failure", 0)),
        "cirrhosis":         float(inputs.get("cirrhosis", 0)),
        "immunosuppression": float(inputs.get("immunosuppression", 0)),
        "aids":              float(inputs.get("aids", 0)),
        "comorbidity_burden": comorb,
        "shock_index":    inputs["hr"] / max(inputs["sbp"], 1),
        "pulse_pressure": inputs["sbp"] - inputs["dbp"],
        "spo2_hr_ratio":  inputs["spo2"] / max(inputs["hr"], 1),
    })
    if fnames:
        return pd.DataFrame([{f: base.get(f, 0.0) for f in fnames}])
    return pd.DataFrame([base])


def _gauge(prob: float) -> go.Figure:
    pct = prob * 100
    clr = C_RED if prob > 0.5 else (C_AMBER if prob > 0.2 else C_GREEN)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 46, "color": clr}},
        title={"text": "Predicted Mortality Risk", "font": {"size": 14, "color": C_BLUE_DARK}},
        gauge={
            "axis":    {"range": [0, 100], "tickwidth": 1},
            "bar":     {"color": clr, "thickness": 0.22},
            "bgcolor": "white",
            "steps": [
                {"range": [0, 20],   "color": "#d5f5e3"},
                {"range": [20, 50],  "color": "#fef3cd"},
                {"range": [50, 100], "color": "#fde8e8"},
            ],
        },
    ))
    fig.update_layout(height=270, margin={"t": 55, "b": 5, "l": 25, "r": 25},
                      paper_bgcolor="white")
    return fig


_FEAT_LABELS = {
    "shock_index":               "shock index (HR ÷ SBP)",
    "apache_2_diagnosis":        "APACHE II severity score",
    "apache_3j_diagnosis":       "APACHE III-j severity score",
    "apache_4a_hospital_death_prob": "APACHE IV predicted mortality",
    "comorbidity_burden":        "total comorbidity count",
    "age":                       "patient age",
    "bmi":                       "body mass index",
    "d1_spo2_mean":              "mean SpO₂",
    "d1_sysbp_mean":             "mean systolic BP",
    "d1_glucose_mean":           "mean blood glucose",
    "glucose_variability":       "blood glucose variability",
    "temp_variability":          "temperature variability",
    "hr_variability":            "heart rate variability",
    "pulse_pressure":            "pulse pressure",
    "heart_failure":             "history of heart failure",
    "cirrhosis":                 "hepatic cirrhosis",
    "aids":                      "AIDS / HIV",
    "d1_heartrate_mean":         "mean heart rate",
    "d1_temp_mean":              "mean body temperature",
    "d1_resprate_mean":          "mean respiratory rate",
}


def _plain_english(shap_dict: dict, prob: float) -> str:
    srs      = pd.Series(shap_dict).sort_values(key=abs, ascending=False)
    risk_top = srs[srs > 0].head(3)
    prot_top = srs[srs < 0].head(1)
    level    = "high" if prob > 0.5 else ("moderate" if prob > 0.2 else "low")

    txt = (f"**Clinical summary:** This patient has a **{level} predicted mortality risk "
           f"({prob * 100:.1f}%)**. ")
    if len(risk_top):
        labels = [_FEAT_LABELS.get(f, f.replace("_", " ")) for f in risk_top.index]
        joined = "; ".join(labels[:2]) + (f"; and {labels[2]}" if len(labels) == 3 else "")
        txt += f"The primary drivers of elevated risk are: **{joined}**. "
    if len(prot_top):
        lbl = _FEAT_LABELS.get(prot_top.index[0], prot_top.index[0].replace("_", " "))
        txt += f"A partially protective factor is {lbl}. "
    txt += ("Review the SHAP chart for the full contribution of each feature "
            "and correlate with bedside assessment before making care decisions.")
    return txt


# ── Session state ─────────────────────────────────────────────────────────────
for _key, _val in [
    ("last_inputs", None), ("last_prob", None),
    ("last_shap", None),   ("chat_history", []),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _val

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("ICU Mortality Risk")
st.sidebar.caption("LightGBM · SHAP · Fairlearn · LangChain")
st.sidebar.markdown("---")

PAGE = st.sidebar.radio(
    "Navigation",
    ["Patient Risk Assessment", "Population Analytics",
     "Fairness Report", "Clinical Q&A"],
)

# ── Kpi helper ────────────────────────────────────────────────────────────────
def kpi(label: str, value: str, col) -> None:
    col.markdown(
        f"<div class='kpi-card'>"
        f"<div class='kpi-label'>{label}</div>"
        f"<div class='kpi-value'>{value}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ============================================================================ #
# PAGE 1 — PATIENT RISK ASSESSMENT                                             #
# ============================================================================ #
if PAGE == "Patient Risk Assessment":
    st.title("Patient Risk Assessment")
    st.caption("Enter patient data to predict ICU mortality risk with SHAP explanation.")

    with st.form("patient_form"):
        col_d, col_v, col_l = st.columns(3)

        with col_d:
            st.subheader("Demographics & History")
            age    = st.slider("Age (years)", 18, 95, 62)
            gender = st.selectbox("Gender", ["Male", "Female"])
            bmi    = st.slider("BMI (kg/m²)", 15.0, 60.0, 27.5, step=0.5)
            st.selectbox("ICU Type",
                ["MICU", "SICU", "CCU", "Cardiac ICU", "CSICU", "NSICU", "Med-Surg ICU"])
            st.markdown("**Comorbidities**")
            ca, cb = st.columns(2)
            diabetes  = ca.checkbox("Diabetes mellitus")
            hf        = cb.checkbox("Heart failure")
            cirrhosis = ca.checkbox("Cirrhosis")
            immuno    = cb.checkbox("Immunosuppression")
            aids      = ca.checkbox("AIDS / HIV")

        with col_v:
            st.subheader("Vital Signs (24 h mean)")
            hr   = st.slider("Heart rate (bpm)", 30, 200, 90)
            sbp  = st.slider("Systolic BP (mmHg)", 50, 250, 120)
            dbp  = st.slider("Diastolic BP (mmHg)", 20, 140, 65)
            spo2 = st.slider("SpO₂ (%)", 60, 100, 97)
            temp = st.slider("Temperature (°C)", 34.0, 42.0, 37.0, step=0.1)
            rr   = st.slider("Respiratory rate (bpm)", 6, 60, 18)

        with col_l:
            st.subheader("Labs & Severity")
            apache  = st.slider("APACHE II score", 0, 50, 16,
                                help=">25 correlates with >50% predicted mortality.")
            glucose = st.slider("Blood glucose (mg/dL)", 50, 600, 145)
            st.markdown(" ")
            st.markdown(" ")

        submitted = st.form_submit_button(
            "Predict Mortality Risk", type="primary", use_container_width=True)

    if submitted:
        inputs = dict(
            age=age, gender=gender, bmi=bmi, apache=apache,
            hr=hr, sbp=sbp, dbp=dbp, spo2=spo2, temp=temp, rr=rr, glucose=glucose,
            diabetes_mellitus=int(diabetes), heart_failure=int(hf),
            cirrhosis=int(cirrhosis), immunosuppression=int(immuno), aids=int(aids),
        )
        with st.spinner("Computing risk score…"):
            mdl    = get_model()
            pt_df  = _build_patient_row(inputs)
            prob   = float(mdl.predict_proba(pt_df)[:, 1][0])
            shap_d = explain_patient(mdl, pt_df)

        st.session_state.last_inputs = inputs
        st.session_state.last_prob   = prob
        st.session_state.last_shap   = shap_d

        risk_label = "HIGH RISK" if prob > 0.5 else ("MODERATE RISK" if prob > 0.2 else "LOW RISK")
        risk_color = C_RED if prob > 0.5 else (C_AMBER if prob > 0.2 else C_GREEN)
        risk_bg    = "#fde8e8" if prob > 0.5 else ("#fef3cd" if prob > 0.2 else "#d5f5e3")

        st.markdown("---")
        left, right = st.columns([1, 1.7])

        with left:
            st.plotly_chart(_gauge(prob), use_container_width=True)
            st.markdown(
                f"<div style='text-align:center;'>"
                f"<span class='risk-badge' style='background:{risk_bg};color:{risk_color};'>"
                f"{risk_label}</span></div>",
                unsafe_allow_html=True,
            )
            st.caption(
                "Green = < 20% · Amber = 20–50% · Red = > 50%. "
                "Threshold set at 35% (optimised for recall in Phase 3)."
            )

        with right:
            shap_srs = pd.Series(shap_d).sort_values(key=abs, ascending=False).head(15)
            bar_cols  = [C_RED if v > 0 else C_BLUE_MID for v in shap_srs.values]
            fig_sw    = go.Figure(go.Bar(
                x=shap_srs.values, y=shap_srs.index,
                orientation="h", marker_color=bar_cols,
            ))
            fig_sw.update_layout(
                title="SHAP Feature Contributions",
                xaxis_title="SHAP value (log-odds contribution to mortality risk)",
                yaxis={"autorange": "reversed"},
                template="plotly_white", height=430,
                margin={"l": 10, "r": 10, "t": 40, "b": 30},
            )
            st.plotly_chart(fig_sw, use_container_width=True)
            st.caption(
                "Red bars increase predicted risk; blue bars decrease it. "
                "Bar length represents the magnitude of each feature's contribution."
            )

        st.info(_plain_english(shap_d, prob))
        st.info(
            "Navigate to **Clinical Q&A** in the sidebar to ask natural-language "
            "questions about this patient's risk profile."
        )

    elif st.session_state.last_prob is None:
        st.info("Fill in the patient data above and click **Predict** to generate a risk score.")

# ============================================================================ #
# PAGE 2 — POPULATION ANALYTICS                                                #
# ============================================================================ #
elif PAGE == "Population Analytics":
    st.title("Population Analytics Dashboard")
    st.caption(
        "Metrics computed on 500 synthetic ICU patients generated at runtime "
        "using realistic clinical distributions."
    )

    with st.spinner("Generating synthetic cohort…"):
        cohort = generate_cohort()

    # ── KPI cards ──────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    kpi("Total patients",        f"{len(cohort):,}", k1)
    kpi("Simulated mortality rate", f"{cohort['hospital_death'].mean() * 100:.1f}%", k2)
    kpi("Mean age",              f"{cohort['age'].mean():.1f} yrs", k3)
    kpi("Mean APACHE II",        f"{cohort['apache_2_diagnosis'].mean():.1f}", k4)

    st.markdown("---")

    # ── Row 1: ICU type bar + age KDE ───────────────────────────────────────
    col_icu, col_age = st.columns(2)

    with col_icu:
        icu_mort = (
            cohort.groupby("icu_type")["hospital_death"]
            .mean()
            .mul(100)
            .reset_index()
            .rename(columns={"hospital_death": "mortality_rate_pct"})
            .sort_values("mortality_rate_pct", ascending=False)
        )
        fig_icu = px.bar(
            icu_mort, x="icu_type", y="mortality_rate_pct",
            color="mortality_rate_pct",
            color_continuous_scale=[[0, C_BLUE_LIGHT], [0.5, C_BLUE_MID], [1, C_RED]],
            labels={"icu_type": "ICU Type", "mortality_rate_pct": "Mortality Rate (%)"},
            template="plotly_white",
        )
        fig_icu.update_layout(
            title="Simulated Mortality Rate by ICU Type",
            coloraxis_showscale=False,
            xaxis_tickangle=-30,
        )
        st.plotly_chart(fig_icu, use_container_width=True)
        st.caption(
            "Mortality rate varies across ICU units, reflecting differences in "
            "patient acuity mix. MICU and SICU typically admit higher-acuity patients."
        )

    with col_age:
        try:
            from scipy.stats import gaussian_kde
            x_rng = np.linspace(18, 96, 250)
            surv  = cohort.loc[cohort["hospital_death"] == 0, "age"].values
            died  = cohort.loc[cohort["hospital_death"] == 1, "age"].values
            fig_kde = go.Figure()
            if len(surv) > 5:
                fig_kde.add_trace(go.Scatter(
                    x=x_rng, y=gaussian_kde(surv)(x_rng),
                    fill="tozeroy", name="Survived",
                    line={"color": C_GREEN}, fillcolor="rgba(30,132,73,0.15)",
                ))
            if len(died) > 5:
                fig_kde.add_trace(go.Scatter(
                    x=x_rng, y=gaussian_kde(died)(x_rng),
                    fill="tozeroy", name="Died",
                    line={"color": C_RED}, fillcolor="rgba(192,57,43,0.20)",
                ))
            fig_kde.update_layout(
                title="Age Distribution by Outcome",
                xaxis_title="Age (years)", yaxis_title="Density",
                template="plotly_white", legend={"title": "Outcome"},
            )
        except ImportError:
            fig_kde = px.histogram(
                cohort, x="age", color="hospital_death", barmode="overlay",
                histnorm="probability density", opacity=0.65,
                color_discrete_map={0: C_GREEN, 1: C_RED},
                labels={"age": "Age (years)", "hospital_death": "Died"},
                template="plotly_white", title="Age Distribution by Outcome",
            )
        st.plotly_chart(fig_kde, use_container_width=True)
        st.caption(
            "KDE of age by outcome. The deceased population skews older, "
            "consistent with ICU epidemiology — age is a strong independent predictor."
        )

    st.markdown("---")

    # ── Row 2: SHAP top-10 + comorbidity scatter ────────────────────────────
    col_shap, col_comorb = st.columns(2)

    with col_shap:
        with st.spinner("Computing SHAP importance…"):
            imp_df = get_shap_importance()
        fig_imp = go.Figure(go.Bar(
            x=imp_df["mean_abs_shap"], y=imp_df["feature"],
            orientation="h", marker_color=C_BLUE_MID,
        ))
        fig_imp.update_layout(
            title="Top 10 Features — Mean |SHAP| Value",
            xaxis_title="Mean |SHAP| (average impact on log-odds)",
            yaxis={"autorange": "reversed"},
            template="plotly_white",
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        st.caption(
            "Mean absolute SHAP values quantify each feature's average impact on the "
            "model's output across all 500 synthetic patients. Higher = more influential."
        )

    with col_comorb:
        comorb_cols = [c for c in
                       ["diabetes_mellitus", "heart_failure", "cirrhosis",
                        "immunosuppression", "leukemia", "lymphoma", "aids",
                        "solid_tumor_with_metastasis"]
                       if c in cohort.columns]
        rows = []
        for col in comorb_cols:
            mask = cohort[col] == 1
            if mask.sum() < 5:
                continue
            rows.append({
                "comorbidity": col.replace("_", " ").title(),
                "prevalence":  round(mask.mean() * 100, 1),
                "mortality_rate": round(cohort.loc[mask, "hospital_death"].mean() * 100, 1),
                "n": int(mask.sum()),
            })
        scat_df = pd.DataFrame(rows)
        if len(scat_df):
            fig_sc = px.scatter(
                scat_df, x="prevalence", y="mortality_rate",
                text="comorbidity", size="n",
                color="mortality_rate",
                color_continuous_scale=[[0, C_BLUE_LIGHT], [1, C_RED]],
                labels={"prevalence": "Prevalence (%)", "mortality_rate": "Mortality Rate (%)"},
                template="plotly_white",
                title="Comorbidity Prevalence vs Mortality Rate",
            )
            fig_sc.update_traces(textposition="top center", marker={"sizemin": 6})
            fig_sc.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_sc, use_container_width=True)
            st.caption(
                "Each point is a comorbidity. Point size reflects the number of patients "
                "with that condition. Conditions in the upper-right quadrant are both "
                "common and strongly associated with mortality."
            )

# ============================================================================ #
# PAGE 3 — FAIRNESS REPORT                                                     #
# ============================================================================ #
elif PAGE == "Fairness Report":
    st.title("Fairness Report")
    st.caption(
        "Performance metrics disaggregated by demographic subgroup. "
        "False Negative Rate (FNR) is highlighted — a high FNR means the model "
        "fails to flag dying patients in that group."
    )

    with st.spinner("Running fairness analysis…"):
        cohort = generate_cohort()
        from src.fairness import compute_subgroup_metrics

    SENSITIVE = [
        ("ethnicity",  "Ethnicity"),
        ("gender_str", "Gender"),
        ("age_group",  "Age Group"),
    ]

    for attr, label in SENSITIVE:
        if attr not in cohort.columns:
            continue

        st.subheader(f"Performance by {label}")
        mdf = compute_subgroup_metrics(
            y_true=cohort["hospital_death"],
            y_pred=cohort["y_pred"].values,
            y_prob=cohort["mortality_prob"].values,
            sensitive_col=cohort[attr],
        )
        if mdf.empty:
            st.warning(f"Insufficient subgroup sizes for {label}.")
            continue

        mdf["fnr"] = (1 - mdf["recall"]).round(4)

        # Styled dataframe
        display = mdf[["n", "accuracy", "precision", "recall", "f1", "auc_roc", "fnr"]].copy()
        st.dataframe(
            display.style
            .format({"accuracy": "{:.3f}", "precision": "{:.3f}", "recall": "{:.3f}",
                     "f1": "{:.3f}", "auc_roc": "{:.3f}", "fnr": "{:.3f}", "n": "{:,}"})
            .background_gradient(subset=["fnr"], cmap="Reds")
            .background_gradient(subset=["recall"], cmap="Greens"),
            use_container_width=True,
            column_config={
                "n":         st.column_config.NumberColumn("N", help="Patients in group"),
                "accuracy":  st.column_config.NumberColumn("Accuracy"),
                "precision": st.column_config.NumberColumn("Precision"),
                "recall":    st.column_config.NumberColumn("Recall (TPR)"),
                "f1":        st.column_config.NumberColumn("F1 Score"),
                "auc_roc":   st.column_config.NumberColumn("AUC-ROC"),
                "fnr":       st.column_config.NumberColumn("FNR ⚠️",
                             help="False Negative Rate = 1 – Recall. Higher = more dangerous."),
            },
        )

        # FNR bar chart — most-disparate group in red
        worst = mdf["fnr"].idxmax()
        bar_colors = [C_RED if g == worst else C_BLUE_MID for g in mdf.index]
        fig_fnr = go.Figure(go.Bar(
            x=mdf.index.tolist(), y=mdf["fnr"].values,
            marker_color=bar_colors,
            text=[f"{v:.3f}" for v in mdf["fnr"].values],
            textposition="outside",
        ))
        fig_fnr.add_hline(
            y=mdf["fnr"].mean(), line_dash="dash", line_color="#6c757d",
            annotation_text=f"Mean FNR: {mdf['fnr'].mean():.3f}",
        )
        fig_fnr.update_layout(
            title=f"False Negative Rate by {label}",
            xaxis_title=label, yaxis_title="FNR",
            yaxis={"range": [0, min(1.0, mdf["fnr"].max() * 1.35)]},
            template="plotly_white",
        )
        st.plotly_chart(fig_fnr, use_container_width=True)
        st.caption(
            f"Red bar = most disparate group ({worst}). "
            "Dashed line = population mean FNR. Higher FNR = patients in that group "
            "are more likely to be incorrectly predicted as low-risk."
        )
        st.markdown("---")

    # Plain-English summary
    st.subheader("Summary")
    st.markdown(
        "The fairness analysis shows whether the model's predictive performance is consistent "
        "across demographic groups. **False Negative Rate (FNR)** is the most clinically critical "
        "metric: a high FNR in any group means patients in that group who die are more likely to "
        "be predicted as low-risk, potentially leading to under-triage.  \n\n"
        "If any group shows an FNR substantially above the population mean (highlighted in red), "
        "consider: (1) lowering the classification threshold for that group; "
        "(2) investigating whether training data under-represents that population; "
        "(3) clinical review of FN cases in that group before deployment."
    )

# ============================================================================ #
# PAGE 4 — CLINICAL Q&A                                                        #
# ============================================================================ #
elif PAGE == "Clinical Q&A":
    st.title("Clinical Q&A")
    st.caption("Ask natural-language questions about a patient's risk prediction.")

    # Sidebar: sample questions + API key notice
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Sample questions**")
    SAMPLES = [
        "Why is this patient high risk?",
        "What interventions could reduce mortality risk?",
        "How reliable is this prediction?",
        "What does the shock index indicate?",
        "Should I escalate care for this patient?",
        "What does a 30% mortality probability mean?",
    ]
    for q in SAMPLES:
        st.sidebar.markdown(f"- *{q}*")

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.warning(
            "Set `ANTHROPIC_API_KEY` in your `.env` file to enable the Clinical Q&A assistant. "
            "The RAG retriever and clinical knowledge base will still work, "
            "but LLM-generated answers require a valid key.",
            icon="⚠️",
        )

    # Check if a patient has been assessed
    if st.session_state.last_prob is None:
        st.info(
            "No patient assessed yet. Go to **Patient Risk Assessment**, run a prediction, "
            "then return here to ask questions about that patient."
        )
    else:
        prob   = st.session_state.last_prob
        shap_d = st.session_state.last_shap
        inputs = st.session_state.last_inputs

        risk_lvl = "HIGH" if prob > 0.5 else ("MODERATE" if prob > 0.2 else "LOW")
        rc       = C_RED if prob > 0.5 else (C_AMBER if prob > 0.2 else C_GREEN)
        st.markdown(
            f"**Current patient:** Age {inputs['age']} · "
            f"APACHE II {inputs['apache']} · "
            f"Predicted risk: "
            f"<span style='color:{rc};font-weight:700'>{prob*100:.1f}% ({risk_lvl})</span>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # Initialise / update explainer in session state
        if ("explainer" not in st.session_state or
                st.session_state.get("explainer_patient") != prob):
            try:
                from src.langchain_qa import ClinicalExplainer
                expl = ClinicalExplainer(api_key=api_key or None)
                expl.set_patient(
                    feature_values=inputs,
                    shap_values=shap_d or {},
                    prediction_prob=prob,
                )
                st.session_state.explainer         = expl
                st.session_state.explainer_patient = prob
            except Exception as e:
                st.session_state.explainer = None
                st.warning(f"Could not initialise ClinicalExplainer: {e}")

        # Display chat history
        for role, msg in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(msg)

        # Chat input
        user_q = st.chat_input("Ask a clinical question about this patient…")
        if user_q:
            with st.chat_message("user"):
                st.markdown(user_q)
            st.session_state.chat_history.append(("user", user_q))

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    if st.session_state.get("explainer"):
                        try:
                            answer = st.session_state.explainer.ask(user_q)
                        except Exception as e:
                            answer = f"Error: {e}"
                    else:
                        answer = (
                            "The Clinical Q&A assistant is unavailable. "
                            "Please set `ANTHROPIC_API_KEY` in your `.env` file "
                            "and reload the page."
                        )
                st.markdown(answer)
            st.session_state.chat_history.append(("assistant", answer))

        if st.session_state.chat_history:
            if st.button("Clear conversation", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='footer'>For research purposes only. "
    "Not validated for clinical use. "
    "Model trained on WiDS Datathon 2020 dataset.</div>",
    unsafe_allow_html=True,
)
