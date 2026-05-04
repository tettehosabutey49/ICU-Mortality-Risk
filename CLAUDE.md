## Project: ICU Mortality Risk Stratification

**Goal**: End-to-end DS project predicting hospital mortality in ICU patients with EDA storytelling,
feature engineering, ML modeling, SHAP explainability, fairness audit, and LangChain clinical Q&A.

**Target variable**: hospital_death (binary: 0=survived, 1=died)
**Dataset**: patient_survival.csv — 180+ features, ~91,000 rows, real ICU data

**Stack**: Python, pandas, scikit-learn, XGBoost, LightGBM, SHAP, UMAP, Fairlearn, LangChain, Streamlit

**Code style**:
- All functions have docstrings explaining what they do and why
- No magic numbers — all thresholds defined as named constants with comments explaining the choice
- Every plot saved to reports/figures/ with descriptive filename
- src/ contains reusable logic; notebooks call src/ functions, they don't reimplement them
- Write code that looks like an experienced data scientist wrote it, not boilerplate AI output

**What NOT to do**:
- Do not use placeholder comments like "# add analysis here"
- Do not leave empty cells in notebooks
- Do not write generic variable names like df2, temp, x1
- Do not skip axis labels, titles, or annotations on any plot
