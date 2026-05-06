# ICU Mortality Risk Stratification

Predicting in-hospital death for ICU patients using gradient boosting, SHAP explainability, demographic fairness auditing, and a natural-language clinical Q&A interface.

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-3.x-brightgreen)](https://lightgbm.readthedocs.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7-orange)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-0.44-red)](https://shap.readthedocs.io)
[![Fairlearn](https://img.shields.io/badge/Fairlearn-0.10-blue)](https://fairlearn.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-9cf)](https://python.langchain.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)

---

## Why This Project

Roughly 1 in 12 ICU admissions ends in death — but those deaths are not randomly distributed. Patients admitted through the emergency department die at substantially higher rates than elective surgical admissions, elderly patients face compounding physiological vulnerabilities that simple severity scores underweight, and algorithmic risk tools have been shown to exhibit racial bias in multiple healthcare contexts (Obermeyer et al., Science 2019). A model that achieves high average AUC can still systematically fail specific patient populations.

This project builds a mortality prediction pipeline on 91,071 real ICU admissions (WiDS Datathon 2020, 8.6% mortality rate) and treats fairness as a first-class deliverable, not an afterthought. The goal is not to replace clinical judgement but to make it transparent — using SHAP to explain every prediction at the individual patient level, and a LangChain RAG layer to let clinicians ask natural-language questions about why the model flagged a particular patient.

---

## Architecture

```
patient_survival.csv (91,071 rows · 186 features · 8.6% mortality)
        │
        ▼
01_eda.ipynb ─────────────── Distribution analysis · UMAP clustering
        │                    Mann-Whitney U tests · odds ratios · Bonferroni correction
        ▼
02_feature_engineering ───── 12 derived clinical features
        │                    shock_index · comorbidity_burden · apache_score_delta
        │                    Smoothed target encoding · recursive feature selection
        ▼
03_modeling.ipynb ─────────── 70/15/15 stratified split · SMOTE (train only)
        │                    LR baseline → RF → XGBoost → LightGBM
        │                    RandomizedSearchCV (AUC-PR) · threshold optimisation
        │                    Best: LightGBM  AUC-ROC 0.906  AUC-PR 0.578
        ▼
04_explainability.ipynb ───── SHAP TreeExplainer · global beeswarm · dependence plots
        │                    Waterfall case studies (TP · TN · FP · FN)
        │                    LangChain RAG (ChromaDB + Claude Sonnet)
        ▼
05_fairness_audit.ipynb ───── Demographic parity · equalized odds · intersectional FNR
        │                    Bootstrap significance · ThresholdOptimizer mitigation
        ▼
app/streamlit_app.py ──────── 4-page live demo (synthetic cohort, no PHI)
                             Patient risk · Population analytics · Fairness · Q&A chat
```

---

## Key Findings

**Best model:** LightGBM (tuned) — AUC-ROC **0.906**, AUC-PR **0.578**, Recall **0.74** at threshold 0.35

**Top 3 predictive features (SHAP):**
1. `apache_4a_hospital_death_prob` — APACHE IV predicted mortality probability; the single strongest predictor, confirming the score's clinical validity while also showing the model learns non-linear adjustments on top of it
2. `comorbidity_burden` — engineered count of active comorbidities (diabetes, heart failure, cirrhosis, AIDS, etc.); each additional comorbidity compounds predicted mortality risk non-linearly
3. `shock_index` — heart rate ÷ systolic BP; values above 0.9 correlate with haemodynamic instability and are captured by the model as a continuous risk signal

**Most significant fairness finding:**
Patients aged 80+ (elderly subgroup) showed a false negative rate **14 percentage points above** the population mean. The model systematically under-predicts mortality risk in the oldest patients, whose atypical physiological presentations diverge from patterns learned from the broader training population. Post-processing with `ThresholdOptimizer` reduces the equalized-odds gap at the cost of a modest drop in overall F1 — a tradeoff that requires clinical and ethics review.

**Notable EDA finding:**
Patients admitted from the operating room had **2.4× lower odds** of in-hospital death compared to emergency department admissions (OR = 0.42, p < 0.001 after Bonferroni correction). This confirms the well-established elective surgery paradox: OR patients are physiologically selected and optimised pre-operatively, while ED patients typically arrive in acute decompensation.

---

## Project Structure

```
ICU-Mortality-Risk/
├── app/
│   └── streamlit_app.py          # 4-page Streamlit demo
├── data/
│   ├── raw/                       # not tracked — add patient_survival.csv here
│   └── processed/
│       ├── features_engineered.csv
│       ├── feature_metadata.csv
│       └── split_indices.npz
├── models/                        # .pkl files not tracked
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_explainability.ipynb
│   └── 05_fairness_audit.ipynb
├── reports/
│   └── figures/
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── models.py
│   ├── explainability.py
│   ├── fairness.py
│   └── langchain_qa.py
├── tests/
│   └── test_preprocessing.py
├── requirements.txt
└── README.md
```

---

## Setup

### Local

```bash
git clone https://github.com/your-username/ICU-Mortality-Risk.git
cd ICU-Mortality-Risk

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Download patient_survival.csv from Kaggle WiDS Datathon 2020
# and place it in data/raw/patient_survival.csv

# Optional: enable LangChain Q&A
cp .env.example .env            # then add your ANTHROPIC_API_KEY

# Run notebooks in order (1 → 5), then launch the app
streamlit run app/streamlit_app.py
```

### Docker

```bash
# Build and run (no dataset required — app uses synthetic data)
docker build -t icu-risk .
docker run -p 8501:8501 -e ANTHROPIC_API_KEY=your_key icu-risk
# Open http://localhost:8501
```

---

## Notebook Guide

| Notebook | What it does |
|---|---|
| `01_eda.ipynb` | Distribution analysis, UMAP patient clustering, Mann-Whitney U tests for feature-target associations, odds ratios for comorbidities |
| `02_feature_engineering.ipynb` | Engineers 12 clinical features (shock index, comorbidity burden, etc.), applies smoothed target encoding, saves `features_engineered.csv` |
| `03_modeling.ipynb` | Trains 4 models with SMOTE and hyperparameter tuning; saves `best_model.pkl` and `lgbm_best.pkl`; tunes classification threshold |
| `04_explainability.ipynb` | Global SHAP beeswarm and bar plots, individual waterfall plots for TP/TN/FP/FN case studies, LangChain clinical Q&A demo |
| `05_fairness_audit.ipynb` | Audits performance gaps across ethnicity, gender, and age group; computes intersectional FNR; demonstrates ThresholdOptimizer |

---

## Live Demo

> Deploy to Streamlit Cloud and add link here.
>
> The demo generates 500 synthetic patients at runtime — no patient data is loaded or transmitted.

---

## Limitations

This model was trained on a single dataset from a specific set of ICU admissions. External validity is unknown.

- **Site specificity**: Admission criteria, protocols, and documentation practices vary across institutions. Performance at a different hospital may differ materially.
- **Temporal validity**: Clinical practice changes over time. A model trained on historical data may become miscalibrated as treatment protocols evolve.
- **Measurement bias**: The analysis cannot distinguish whether `d1_spo2_mean` reflects true patient physiology or pulse oximetry measurement error that disproportionately affects dark-skinned patients (Sjoding et al., NEJM 2020).
- **Not for clinical use**: This project has not undergone prospective validation, IRB review, or regulatory approval. It is a portfolio demonstration only.

---

## License

MIT
