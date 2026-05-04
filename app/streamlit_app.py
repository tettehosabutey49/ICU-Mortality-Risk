"""
Streamlit dashboard for the ICU Mortality Risk Stratification model.

Features:
  - Upload a patient CSV and get cohort-level risk scores
  - Individual patient deep-dive with SHAP waterfall explanation
  - LangChain Q&A panel for natural-language clinical queries
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Allow imports from the project root when running from the app/ directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.explainability import compute_shap_values, explain_patient
from src.models import load_model
from src.preprocessing import drop_high_missingness, impute

# --------------------------------------------------------------------------- #
# Page config                                                                   #
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="ICU Mortality Risk",
    page_icon="🏥",
    layout="wide",
)

# --------------------------------------------------------------------------- #
# Cached resources — loaded once per session                                    #
# --------------------------------------------------------------------------- #


@st.cache_resource
def get_model():
    """Load the best trained model from disk (cached to avoid reload on rerun)."""
    try:
        return load_model("lgbm_best")
    except FileNotFoundError:
        st.error("No trained model found.  Run notebook 03_modeling.ipynb first.")
        st.stop()


# --------------------------------------------------------------------------- #
# Sidebar                                                                       #
# --------------------------------------------------------------------------- #

st.sidebar.title("ICU Mortality Risk")
st.sidebar.markdown("**Upload a patient cohort CSV to get risk scores.**")

uploaded_file = st.sidebar.file_uploader("Patient cohort CSV", type="csv")

MORTALITY_THRESHOLD = st.sidebar.slider(
    "Decision threshold (predicted probability)",
    min_value=0.10,
    max_value=0.90,
    value=0.30,
    step=0.05,
    help=(
        "Patients above this threshold are flagged as high-risk.  "
        "Lower thresholds increase sensitivity (catch more deaths) at the cost of more false positives."
    ),
)

# --------------------------------------------------------------------------- #
# Main content                                                                  #
# --------------------------------------------------------------------------- #

st.title("ICU Mortality Risk Stratification")
st.caption("Gradient boosting + SHAP explainability | Trained on ~91 000 ICU admissions")

if uploaded_file is None:
    st.info("Upload a patient CSV in the sidebar to begin.")
    st.stop()

# ---- Load & preprocess ----
raw_df = pd.read_csv(uploaded_file)
target_col = "hospital_death"
feature_df = raw_df.drop(columns=[target_col], errors="ignore")

cleaned_df, _ = drop_high_missingness(feature_df)
imputed_df = impute(cleaned_df)

model = get_model()

# ---- Predict ----
try:
    probs = model.predict_proba(imputed_df)[:, 1]
except Exception as exc:
    st.error(f"Prediction failed: {exc}")
    st.stop()

result_df = raw_df.copy()
result_df["mortality_prob"] = probs
result_df["risk_flag"] = (probs >= MORTALITY_THRESHOLD).astype(int)

# --------------------------------------------------------------------------- #
# Tab layout                                                                    #
# --------------------------------------------------------------------------- #

tab_cohort, tab_patient, tab_qa = st.tabs(["Cohort Overview", "Patient Deep-Dive", "Clinical Q&A"])

# ---- Cohort Overview ----
with tab_cohort:
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    col_kpi1.metric("Total patients", f"{len(result_df):,}")
    col_kpi2.metric(
        "High-risk patients",
        f"{result_df['risk_flag'].sum():,}",
        delta=f"{result_df['risk_flag'].mean() * 100:.1f} %",
        delta_color="inverse",
    )
    col_kpi3.metric("Median risk score", f"{np.median(probs):.3f}")

    st.subheader("Predicted Mortality Probability Distribution")
    fig_hist = px.histogram(
        result_df,
        x="mortality_prob",
        nbins=50,
        color_discrete_sequence=["steelblue"],
        labels={"mortality_prob": "Predicted Mortality Probability"},
    )
    fig_hist.add_vline(
        x=MORTALITY_THRESHOLD,
        line_dash="dash",
        line_color="crimson",
        annotation_text=f"Threshold = {MORTALITY_THRESHOLD}",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Cohort Risk Table")
    display_cols = ["mortality_prob", "risk_flag"] + [
        c for c in ["age", "apache_2_diagnosis", "bmi", "ethnicity"] if c in result_df.columns
    ]
    st.dataframe(
        result_df[display_cols]
        .sort_values("mortality_prob", ascending=False)
        .reset_index(drop=True)
        .style.background_gradient(subset=["mortality_prob"], cmap="RdYlGn_r"),
        use_container_width=True,
        height=400,
    )

# ---- Patient Deep-Dive ----
with tab_patient:
    st.subheader("Individual Patient Explanation")
    patient_idx = st.selectbox(
        "Select patient (sorted by risk, highest first)",
        options=result_df.sort_values("mortality_prob", ascending=False).index.tolist(),
        format_func=lambda i: f"Patient {i}  |  Risk = {result_df.loc[i, 'mortality_prob']:.3f}",
    )

    patient_row = imputed_df.loc[[patient_idx]]
    shap_contributions = explain_patient(model, patient_row)

    shap_series = (
        pd.Series(shap_contributions)
        .sort_values(key=abs, ascending=False)
        .head(15)
    )

    fig_shap = px.bar(
        x=shap_series.values,
        y=shap_series.index,
        orientation="h",
        color=shap_series.values,
        color_continuous_scale=["#2166ac", "#f7f7f7", "#d6604d"],
        labels={"x": "SHAP value (log-odds contribution)", "y": "Feature"},
        title=f"Patient {patient_idx} — SHAP Feature Contributions",
    )
    fig_shap.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_shap, use_container_width=True)

    st.markdown(f"**Predicted mortality probability: `{result_df.loc[patient_idx, 'mortality_prob']:.3f}`**")

# ---- LangChain Q&A ----
with tab_qa:
    st.subheader("Clinical Q&A (LangChain + Claude)")
    st.caption(
        "Ask questions about patients or cohort patterns.  "
        "Answers are grounded in model predictions and SHAP explanations."
    )
    st.warning(
        "Q&A requires a valid `ANTHROPIC_API_KEY` in your `.env` file and "
        "a pre-built Chroma vector store (run notebook 06_langchain_interface.ipynb first).",
        icon="⚠️",
    )

    question = st.text_input(
        "Your question",
        placeholder="e.g. Which patients are most at risk and why?",
    )

    if question:
        try:
            from src.langchain_qa import ask, build_qa_chain, build_vector_store
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import Chroma

            CHROMA_DIR = Path("models/chroma_db")
            if not CHROMA_DIR.exists():
                st.error("Vector store not found.  Run notebook 06 first.")
                st.stop()

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            store = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)
            chain = build_qa_chain(store)

            with st.spinner("Querying..."):
                answer, sources = ask(chain, question)

            st.markdown("**Answer:**")
            st.write(answer)

            if sources:
                with st.expander("Source documents used"):
                    for doc in sources:
                        st.markdown(f"- {doc.page_content}")

        except Exception as exc:
            st.error(f"Q&A error: {exc}")
