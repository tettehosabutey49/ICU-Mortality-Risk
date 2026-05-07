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
patient_survival.csv (91,713 rows · 85 columns · 8.63% mortality)
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
        │                    Best: LightGBM  AUC-ROC 0.8956  AUC-PR 0.5722
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

**Best model:** LightGBM (tuned) — AUC-ROC **0.8956**, AUC-PR **0.5722**, F1 **0.5229** at threshold 0.35

**Top predictive features (SHAP global ranking):**
1. `apache_4a_hospital_death_prob` (SHAP rank #1) — APACHE IVa predicted mortality probability; the single strongest predictor. Crucially, XGBoost native gain ranks it **#16** — a major disagreement that proves native importance is misleading
2. `apache_4a_icu_death_prob` (SHAP rank #2) — ICU-specific death probability from APACHE IVa; carries complementary signal to the hospital-death probability
3. `apache_score_delta` (SHAP rank #4, mean |SHAP| = 0.2166) — engineered feature: hospital-death probability minus ICU-death probability; captures patients at risk of post-ICU deterioration, a signal absent from raw severity scores
4. `d1_resprate_max` (SHAP rank #6) — maximum respiratory rate in the first 24 hours; a direct marker of respiratory compromise and respiratory failure risk
5. `spo2_hr_ratio` (SHAP rank #7, mean |SHAP| = 0.1322) — engineered SpO₂-to-heart-rate ratio; strongest interaction partner with `apache_4a_hospital_death_prob`

**8 of 12 engineered features** ranked above median SHAP importance across all features. `shock_index` (mean |SHAP| = 0.1437) and `spo2_hr_ratio` both outperformed many raw clinical measurements.

**SHAP vs native importance disagreement**: `pre_icu_los_days` ranks **#1 by XGBoost gain** but **#12 by SHAP** — it appears in many shallow splits but has low marginal impact once other features are present. `apache_4a_hospital_death_prob` has the inverse problem: ignored by gain-based importance, dominant by SHAP.

**Most significant fairness findings:**
- **Ethnicity**: FNR ranges from 30.0% to 46.0% across ethnic groups — a **16.3 percentage-point gap**. Equalized-odds difference: **0.1603** (FLAGGED). The model systematically under-flags dying patients in the highest-FNR ethnic group.
- **Age**: Equalized-odds difference **0.1423** (FLAGGED); demographic parity gap **0.1831** (FLAGGED). Counterintuitively, the **youngest adults (18–45) show the highest FNR at 45.8%**, not the elderly — likely because atypical severity presentations in young critical patients are under-represented in training.
- **Gender**: No meaningful disparity — FNR gap of only **1.1%** (OK).

Post-processing with `ThresholdOptimizer` (equalized-odds constraint) reduces these gaps at the cost of a modest drop in overall F1 — a tradeoff that requires clinical and ethics review before deployment.

**Key EDA findings:**
- **Dataset**: 91,713 rows · 85 columns · 8.63% mortality (7,915 deaths); zero duplicate rows; 1 feature above the 60% missingness threshold
- **Age**: Patients who died were 7 years older at median (71 yr vs 64 yr for survivors); Mann-Whitney p = 1.49×10⁻²⁴⁷, rank-biserial r = 0.24 — small but unambiguous. Gender association is statistically significant but negligible (chi-square p = 0.034, Cramér's V = 0.007)
- **ICU type**: MICU carries the highest mortality at **12.1%**; CSICU the lowest at **5.5%**. Cardiac ICU: 10.3%; CCU-CTICU: 7.6%
- **Comorbidity ORs**: Solid tumor with metastasis (OR **2.47**, CI 2.20–2.78), leukemia (OR **2.43**, CI 1.99–2.97), and hepatic failure (OR **2.39**, CI 2.05–2.77) are the strongest mortality predictors. AIDS (OR 1.56) does not reach significance (CI includes 1.0)
- **Counterintuitive finding**: `diabetes_mellitus` is **protective** (OR **0.87**, CI 0.82–0.92) — likely because diabetic ICU patients receive more structured glucose monitoring, which also catches early deterioration
- **Admission pathway**: Patients admitted from the operating room had **2.4× lower mortality odds** than emergency department admissions — the elective surgery paradox

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
