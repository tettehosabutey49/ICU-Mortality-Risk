# ICU Mortality Risk Stratification

End-to-end data science project predicting in-hospital mortality for ICU patients using the
[WIDS Datathon 2020 / patient survival dataset](https://www.kaggle.com/datasets/mitishaagarwal/patient).

## Project Overview

| Item | Detail |
|---|---|
| Target | `hospital_death` (binary: 0 = survived, 1 = died) |
| Dataset | `patient_survival.csv` — ~91,000 rows, 180+ features |
| Class imbalance | ~8-9% positive rate |

## Pipeline

```
01_eda.ipynb            → exploratory data analysis & storytelling
02_feature_engineering  → domain-driven feature construction
03_modeling             → baseline → XGBoost / LightGBM + tuning
04_explainability       → SHAP global + local + UMAP embedding
05_fairness_audit       → demographic parity / equalized odds
06_langchain_interface  → clinical Q&A over predictions
streamlit_app.py        → interactive risk dashboard
```

## Quick Start

```bash
pip install -r requirements.txt
# Place patient_survival.csv in data/raw/
jupyter lab
```

## Stack

Python · pandas · scikit-learn · XGBoost · LightGBM · SHAP · UMAP ·
Fairlearn · LangChain · Streamlit
