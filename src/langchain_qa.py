"""
LangChain-powered clinical Q&A over ICU patient predictions.

ClinicalExplainer: RAG + memory Q&A for individual patient explanations.
Streamlit helper functions (build_vector_store, build_qa_chain, ask) kept
for backward compatibility with app/streamlit_app.py.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Using the current Claude Sonnet model for strong clinical reasoning at low latency.
LLM_MODEL = "claude-sonnet-4-20250514"

CHROMA_PERSIST_DIR = Path("models/chroma_db")

# --------------------------------------------------------------------------- #
# 25 hand-written clinical Q&A pairs — ground the RAG layer in ICU medicine   #
# --------------------------------------------------------------------------- #

CLINICAL_QA_PAIRS: list[dict[str, str]] = [
    {
        "question": "Why is this patient high risk?",
        "answer": (
            "High-risk ICU patients typically show multiple concurrent physiological disturbances. "
            "The model assigns higher mortality probability when: (1) APACHE severity is elevated, "
            "indicating multi-organ dysfunction; (2) shock index exceeds 1.0, signalling haemodynamic "
            "instability; (3) multiple active comorbidities compound the acute illness; and (4) lab "
            "values such as elevated lactate, creatinine, or wide glucose variability signal systemic "
            "stress or organ failure. The simultaneous presence of several of these factors is more "
            "dangerous than any single factor in isolation."
        ),
    },
    {
        "question": "What does the shock index indicate?",
        "answer": (
            "The shock index (SI = heart rate / systolic BP) above 1.0 indicates haemodynamic "
            "instability — the heart is beating faster than the systolic pressure, implying the "
            "cardiovascular system is under compensatory stress. Validated by Rady et al. (1994) as "
            "an ICU triage tool, SI > 1.0 is associated with significantly higher mortality. "
            "Values above 1.5 suggest severe haemodynamic compromise requiring urgent vasopressor "
            "support or fluid resuscitation."
        ),
    },
    {
        "question": "What interventions might help this patient?",
        "answer": (
            "Based on key risk factors: (1) Haemodynamic resuscitation if SI is elevated — IV fluids "
            "and norepinephrine (first-line vasopressor in sepsis per Surviving Sepsis Campaign); "
            "(2) Glycaemic management if glucose variability is high — target 140-180 mg/dL in "
            "critically ill patients; (3) Early source control if sepsis is suspected; (4) Comorbidity-"
            "specific management (e.g., hepatic encephalopathy management for cirrhosis). These "
            "suggestions reflect general ICU principles — always integrate the full clinical picture."
        ),
    },
    {
        "question": "How reliable is this prediction?",
        "answer": (
            "The model was trained on ~91,000 ICU admissions and achieves ROC-AUC ~0.85-0.87 and "
            "PR-AUC ~0.45-0.55 on held-out data — substantially above the no-skill baseline of 0.086. "
            "Important caveats: (1) trained on US hospital data and may not generalise to all settings; "
            "(2) uses only first-24-hour summary statistics, not patient trajectory; (3) not "
            "prospectively validated. Treat the probability as a decision-support signal grounded in "
            "objective data patterns, not a definitive prognosis."
        ),
    },
    {
        "question": "What features are most concerning for this patient?",
        "answer": (
            "The SHAP values identify features with the strongest impact on this specific patient. "
            "Positive SHAP values push the prediction toward higher mortality; negative values push "
            "toward survival. The features with the largest absolute SHAP values are those the model "
            "found most informative for this patient — these may differ from globally important "
            "features because patient-specific context (e.g., extreme values or rare combinations) "
            "can make a normally weak feature highly decisive."
        ),
    },
    {
        "question": "What does the APACHE score mean?",
        "answer": (
            "APACHE (Acute Physiology and Chronic Health Evaluation) is a validated ICU severity "
            "scoring system. APACHE II (range 0-71) uses 12 acute physiological variables plus age "
            "and chronic health status from the first 24 hours; higher scores indicate greater illness "
            "severity. APACHE IVa additionally incorporates the admission diagnosis for more granular "
            "hospital and ICU death probability estimates. APACHE IVa hospital death probability is "
            "often the strongest single model predictor because it already encapsulates much of the "
            "physiological complexity of critical illness."
        ),
    },
    {
        "question": "How does comorbidity burden affect the prediction?",
        "answer": (
            "Comorbidity burden counts active chronic diseases (AIDS, cirrhosis, diabetes, hepatic "
            "failure, immunosuppression, leukaemia, lymphoma, solid tumour with metastasis). Each "
            "additional comorbidity reduces physiological reserve. Mortality rate rises monotonically "
            "with burden. Cirrhosis and hepatic failure carry the highest individual odds ratios "
            "(>3× increased mortality risk). A patient with both cirrhosis and metastatic cancer "
            "faces compounding risks that no single disease flag captures."
        ),
    },
    {
        "question": "What is the significance of high glucose variability?",
        "answer": (
            "Glucose variability (day-1 max minus min glucose) captures metabolic instability. "
            "Wide glucose swings signal sepsis, stress hyperglycaemia, or impaired regulation. "
            "Egi et al. (2006) showed glucose variability is an independent ICU mortality predictor, "
            "particularly in septic patients where counterregulatory hormones drive stress "
            "hyperglycaemia. The variability measure captures both hyperglycaemic peaks and "
            "hypoglycaemic troughs that a mean value would miss."
        ),
    },
    {
        "question": "What does a positive apache_score_delta mean?",
        "answer": (
            "apache_score_delta = APACHE IVa hospital_death_prob minus icu_death_prob. "
            "A positive delta means the patient is more likely to die in the hospital overall "
            "than specifically in the ICU — suggesting elevated risk of post-ICU deterioration "
            "(ward decompensation, do-not-resuscitate decisions, or delayed complications). "
            "This signal prompts earlier goals-of-care conversations and closer post-ICU monitoring."
        ),
    },
    {
        "question": "What does temperature variability indicate?",
        "answer": (
            "Temperature delta (day-1 max minus min body temperature) reflects thermoregulatory "
            "instability. Variation > 2°C suggests inflammatory dysregulation, sepsis, or CNS "
            "dysfunction. Both fever and hypothermia are independently associated with increased "
            "ICU mortality. Wide temperature swings may indicate oscillating inflammatory states "
            "or difficulty managing temperature in mechanically ventilated patients with impaired "
            "hypothalamic function."
        ),
    },
    {
        "question": "How does age affect the prediction?",
        "answer": (
            "Age is an independent ICU mortality predictor with a non-linear effect. Median age "
            "is approximately 64 in survivors vs 71 in patients who die. Elderly patients (80+) "
            "show substantially higher mortality, while young adults (<45) have the lowest rates. "
            "However, age interacts with comorbidity burden and APACHE score — a young patient "
            "with multi-organ failure may have higher predicted mortality than an elderly patient "
            "with a minor surgical complication. The tree-based model captures these interactions."
        ),
    },
    {
        "question": "What does heart rate variability indicate in the ICU?",
        "answer": (
            "HR variability here is the within-day range (max HR minus min HR), not the "
            "high-frequency HRV measured in cardiology. A wide range can indicate: autonomic "
            "nervous system dysfunction (an independent mortality predictor in critical illness); "
            "paroxysmal arrhythmias; or haemodynamic responses to interventions. Very narrow "
            "ranges (autonomic suppression in deeply sedated or moribund patients) and very wide "
            "ranges (haemodynamic instability) are both associated with worse outcomes."
        ),
    },
    {
        "question": "Is a 30% predicted mortality probability high?",
        "answer": (
            "Yes. With a baseline ICU mortality of ~8.6%, a 30% predicted probability represents "
            "approximately 3.5× the population average risk. In absolute terms, if applied to 100 "
            "similar patients, roughly 30 would be expected to die. This level of risk typically "
            "warrants intensified monitoring, proactive treatment of modifiable risk factors, "
            "and early goals-of-care discussion with the patient and family."
        ),
    },
    {
        "question": "How was this model trained?",
        "answer": (
            "The champion model is a gradient boosted tree (LightGBM or XGBoost, selected by "
            "validation PR-AUC) trained on ~73,000 ICU admissions from the WiDS Datathon 2020 "
            "dataset. Class imbalance (~9% positive) was handled via scale_pos_weight. "
            "Hyperparameters were tuned using RandomizedSearchCV with 3-fold cross-validation "
            "scored on AUC-PR. A 70/15/15 stratified split was used; the test set indices are "
            "preserved for fairness audit."
        ),
    },
    {
        "question": "What should I do if I disagree with the model's prediction?",
        "answer": (
            "Clinical judgment should always supersede model predictions. This tool is decision "
            "support, not a diagnostic device. If you disagree, consider: whether key clinical "
            "information is absent from the model's 24-hour inputs (e.g., recent status change); "
            "whether the patient belongs to a group with known model disparities (see fairness "
            "audit); or whether the patient's trajectory has changed since measurement. Document "
            "your clinical reasoning regardless of the model's output."
        ),
    },
    {
        "question": "What does pulse pressure tell us?",
        "answer": (
            "Pulse pressure = SBP minus DBP (normal ~40 mmHg). Narrow pulse pressure (<25 mmHg) "
            "suggests reduced stroke volume — seen in cardiac tamponade, hypovolaemic shock, or "
            "severe heart failure — and signals that cardiovascular compensation is near its limit. "
            "Wide pulse pressure (>60 mmHg) indicates arterial stiffness or aortic regurgitation. "
            "In the ICU, narrow pulse pressure is particularly concerning as a harbinger of "
            "haemodynamic collapse."
        ),
    },
    {
        "question": "How should I interpret SHAP values?",
        "answer": (
            "SHAP values measure each feature's contribution to the difference between this "
            "patient's predicted mortality probability and the average prediction across all "
            "patients. A positive SHAP value means that feature increased this patient's predicted "
            "risk; negative means it decreased risk. The magnitude shows how much — a SHAP of "
            "+0.3 in log-odds units means that feature alone increased predicted log-odds by 0.3. "
            "Features with large absolute SHAP are the primary drivers of this patient's prediction."
        ),
    },
    {
        "question": "What are the model's known limitations?",
        "answer": (
            "Key limitations: (1) Training data scope — US hospitals, single historical dataset; "
            "(2) 24-hour snapshot — no patient trajectory or time-series trends; (3) Imputation — "
            "10-20% of values were group-median imputed; (4) Calibration — predicted probabilities "
            "may not be perfectly calibrated in all subgroups; (5) Fairness — differential "
            "performance across ethnic groups and age groups was identified in the fairness audit "
            "(notebook 05). Prospective validation is required before any clinical deployment."
        ),
    },
    {
        "question": "Why do cirrhosis patients have higher predicted mortality?",
        "answer": (
            "Cirrhosis impairs multiple organ systems simultaneously: hepatic synthetic dysfunction "
            "(coagulopathy, hypoalbuminaemia), portal hypertension (variceal bleeding, ascites, "
            "encephalopathy), immune dysfunction (susceptibility to spontaneous bacterial peritonitis "
            "and sepsis), and hepatorenal syndrome. In the EDA, cirrhosis had an odds ratio > 3 "
            "for ICU mortality — one of the highest among all comorbidities. Co-occurring with "
            "acute illness, the liver's limited recovery capacity compounds all other organ failures."
        ),
    },
    {
        "question": "What does the SpO2/HR ratio indicate?",
        "answer": (
            "SpO2/HR ratio = minimum SpO2 / maximum heart rate. It proxies oxygen delivery "
            "efficiency: a lower ratio indicates the patient requires high cardiac output "
            "(tachycardia) to compensate for impaired oxygenation, suggesting inadequate O2 "
            "delivery relative to cardiac workload. This mirrors the physiology of sepsis-induced "
            "cardiomyopathy or ARDS, where the heart overworks in the face of impaired gas "
            "exchange or distributive shock."
        ),
    },
    {
        "question": "How does the model handle patients with missing data?",
        "answer": (
            "Missing values were handled during feature engineering: vital signs imputed by "
            "ICU-type group median (preserving care-setting physiological norms); lab values "
            "by diagnostic-group median (preserving diagnosis-specific reference ranges); "
            "binary comorbidity flags filled with 0 (EHR convention — absence not recorded). "
            "Features still >40% missing after imputation were dropped. LightGBM also natively "
            "handles residual missing values by learning optimal split directions during training."
        ),
    },
    {
        "question": "What is mean arterial pressure and why does it matter?",
        "answer": (
            "MAP = (SBP + 2×DBP) / 3. It reflects average organ perfusion pressure across the "
            "full cardiac cycle. MAP is more informative than SBP alone because diastole occupies "
            "~two-thirds of the cardiac cycle. The Surviving Sepsis Campaign recommends MAP ≥ "
            "65 mmHg as the vasopressor target in septic shock. Sustained MAP below 65 mmHg "
            "indicates inadequate tissue perfusion and is directly linked to organ failure."
        ),
    },
    {
        "question": "How does ICU type affect mortality predictions?",
        "answer": (
            "ICU type is target-encoded — each unit type is replaced by the average mortality "
            "rate of patients admitted there. Medical ICUs (MICU) typically have higher mortality "
            "than cardiac surgery ICUs (CTICU) because MICU patients have complex multi-system "
            "illness versus the more predictable post-operative recovery of cardiac surgery. The "
            "model learns these baseline severity differences through encoding and exploits "
            "within-type variation via other features."
        ),
    },
    {
        "question": "What does blood pressure variability indicate?",
        "answer": (
            "SBP variability (day-1 SBP max minus min) reflects haemodynamic lability. Large "
            "swings indicate difficulty weaning vasopressors, arrhythmia-related pressure "
            "variation, or septic shock physiology with distributive circulatory failure. High "
            "BP variability is particularly concerning in post-operative patients where stability "
            "is expected, and signals an ICU course that is unlikely to be straightforward."
        ),
    },
    {
        "question": "When should I escalate care based on the model's output?",
        "answer": (
            "Consider escalating care discussions when: predicted mortality probability exceeds "
            "30%, especially if combined with clinical deterioration; the SHAP explanation shows "
            "multiple high-impact positive contributors simultaneously; shock_index_high flag is 1; "
            "or a positive apache_score_delta is present (patient at risk beyond the ICU stay). "
            "Pair the model's output with real-time clinical assessment and patient/family "
            "goals-of-care conversations."
        ),
    },
    {
        "question": "How does this model compare to clinical judgment?",
        "answer": (
            "ML models for ICU mortality prediction typically achieve ROC-AUC 0.75-0.90, "
            "comparable to unaided clinician judgment for 24-hour mortality. However, clinicians "
            "have access to information not in 24-hour summary statistics: trajectory, verbal "
            "communication, qualitative gestalt. The ideal use is augmentative: the model "
            "identifies patients whose objective data patterns resemble historical deaths, "
            "prompting closer attention. It should never replace the bedside assessment."
        ),
    },
]


# --------------------------------------------------------------------------- #
# ClinicalExplainer — RAG + memory Q&A for individual patient predictions      #
# --------------------------------------------------------------------------- #

class ClinicalExplainer:
    """RAG-powered clinical Q&A for individual ICU patient risk predictions.

    Architecture:
    1. Patient context string built from feature values + SHAP drivers.
    2. ChromaDB vector store of 25 hand-written clinical Q&A pairs provides
       domain-grounded reference context (RAG layer).
    3. Claude claude-sonnet-4-6 answers clinician questions, grounded in both
       the patient context and retrieved reference Q&A.
    4. ConversationBufferMemory: last 3 Q&A exchanges are included in the prompt
       so follow-up questions can reference prior answers.

    Graceful fallback: if ANTHROPIC_API_KEY is absent the ask() method returns
    an informative message instead of raising an exception.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._patient_context: str = ""
        self._chat_history: list[tuple[str, str]] = []   # ConversationBufferMemory
        self._llm = None
        self._retriever = None

        if self.api_key:
            from langchain_anthropic import ChatAnthropic
            self._llm = ChatAnthropic(
                model=LLM_MODEL,
                api_key=self.api_key,
                temperature=0,
            )
            self._retriever = self._build_qa_retriever()
            
        # if self.api_key:
        #     try:
        #         from langchain_anthropic import ChatAnthropic
        #         self._llm = ChatAnthropic(
        #             model=LLM_MODEL,
        #             api_key=self.api_key,
        #             temperature=0,  # deterministic responses for clinical use
        #         )
        #         self._retriever = self._build_qa_retriever()
        #     except Exception as exc:
        #         print(f"[ClinicalExplainer] Warning — LLM init failed: {exc}")

    # ------------------------------------------------------------------ #

    def set_patient(
        self,
        feature_values: dict,
        shap_values: dict,
        prediction_prob: float,
    ) -> None:
        """Load a new patient context and clear conversation history.

        Args:
            feature_values: {feature_name: raw_value} dict for the patient.
            shap_values:    {feature_name: shap_value} dict for the patient.
            prediction_prob: model's predicted mortality probability (0–1).
        """
        self._patient_context = self._build_context_string(
            feature_values, shap_values, prediction_prob
        )
        self._chat_history = []  # reset memory for new patient

    def ask(self, question: str) -> str:
        """Answer a clinician question grounded in the current patient's context.

        Returns a natural-language answer, or a fallback message if the API
        key is not configured.
        """
        if not self.api_key:
            return (
                "ANTHROPIC_API_KEY not set.  Add it to .env to enable clinical Q&A.\n"
                "Example: ANTHROPIC_API_KEY=sk-ant-..."
            )
        if not self._patient_context:
            return "No patient loaded.  Call set_patient() first."
        if self._llm is None:
            return "LLM not initialised — check that your API key is valid."

        # RAG: retrieve the 3 most relevant Q&A pairs from the vector store
        rag_text = ""
        if self._retriever:
            try:
                docs = self._retriever.invoke(question)[:3]
                rag_text = "\n\n".join(
                    f"Reference context:\n{d.page_content}" for d in docs
                )
            except Exception:
                pass

        # Build full prompt
        prompt = self._assemble_prompt(question, rag_text)

        try:
            from langchain_core.messages import HumanMessage
            response = self._llm.invoke([HumanMessage(content=prompt)])
            answer = response.content
        except Exception as exc:
            answer = f"[Error calling Claude API: {exc}]"

        # ConversationBufferMemory: keep last 5 exchanges
        self._chat_history.append((question, answer))
        if len(self._chat_history) > 5:
            self._chat_history = self._chat_history[-5:]

        return answer

    # ------------------------------------------------------------------ #
    # Private helpers                                                       #
    # ------------------------------------------------------------------ #

    def _build_context_string(
        self,
        feature_values: dict,
        shap_values: dict,
        prediction_prob: float,
    ) -> str:
        top_drivers = sorted(shap_values.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
        driver_lines = "\n".join(
            f"  - {feat}: SHAP={val:+.3f} "
            f"({'increases' if val > 0 else 'decreases'} mortality risk)"
            for feat, val in top_drivers
        )

        age       = feature_values.get("age", "N/A")
        gender_v  = feature_values.get("gender", None)
        gender    = "Male" if gender_v == 1 else ("Female" if gender_v == 0 else "Unknown")
        si        = feature_values.get("shock_index", None)
        si_flag   = "HIGH RISK (>1.0)" if si is not None and si > 1.0 else "normal"
        comorb    = feature_values.get("comorbidity_burden", "N/A")
        apache_h  = feature_values.get("apache_4a_hospital_death_prob", "N/A")
        apache_i  = feature_values.get("apache_4a_icu_death_prob", "N/A")

        return (
            f"Patient Clinical Profile:\n"
            f"- Age: {age}, Gender: {gender}\n"
            f"- Predicted mortality probability: {prediction_prob*100:.1f}%\n"
            f"- APACHE IVa hospital death probability: {apache_h}\n"
            f"- APACHE IVa ICU death probability: {apache_i}\n"
            f"- Shock index: {si} ({si_flag})\n"
            f"- Comorbidity burden: {comorb} active comorbidities\n"
            f"- Glucose variability: {feature_values.get('glucose_variability', 'N/A')}\n"
            f"- Heart rate variability: {feature_values.get('hr_variability', 'N/A')}\n"
            f"\nTop 5 SHAP risk drivers:\n{driver_lines}"
        )

    def _build_qa_retriever(self):
        """Build an in-memory Chroma vector store from the 25 clinical Q&A pairs."""
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma
        from langchain_core.documents import Document

        docs = [
            Document(
                page_content=f"Q: {p['question']}\nA: {p['answer']}",
                metadata={"question": p["question"]},
            )
            for p in CLINICAL_QA_PAIRS
        ]
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        try:
            store = Chroma.from_documents(docs, embeddings)
        except Exception:
            # Older chromadb may need explicit collection name
            store = Chroma.from_documents(
                docs, embeddings, collection_name="clinical_qa"
            )
        return store.as_retriever(search_kwargs={"k": 3})

    def _assemble_prompt(self, question: str, rag_context: str) -> str:
        parts = [
            "You are a clinical AI assistant helping an ICU clinician understand a "
            "patient's predicted mortality risk.  Respond with clinical precision in "
            "plain language.  Keep answers to 3-5 sentences unless more detail is "
            "clinically warranted.  Always note when clinical judgment should "
            "supersede the model output.",
            "",
            f"=== Current Patient ===\n{self._patient_context}",
        ]
        if rag_context:
            parts += ["", f"=== Relevant Clinical Reference Context ===\n{rag_context}"]
        if self._chat_history:
            recent = self._chat_history[-3:]
            history_lines = []
            for q, a in recent:
                history_lines += [f"Clinician: {q}", f"Assistant: {a}"]
            parts += ["", "=== Previous Questions (ConversationBufferMemory) ===",
                      "\n".join(history_lines)]
        parts += ["", f"=== Clinician Question ===\n{question}"]
        return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Streamlit-compatible helpers (used by app/streamlit_app.py)                  #
# --------------------------------------------------------------------------- #

def build_patient_documents(
    df: "pd.DataFrame",
    shap_dict: dict[int, dict[str, float]],
    prob_col: str = "mortality_prob",
) -> list:
    """Convert patient rows + SHAP values into LangChain Documents for vector store."""
    from langchain_core.documents import Document

    documents = []
    for idx, row in df.iterrows():
        shap_summary = ""
        if idx in shap_dict:
            top = sorted(shap_dict[idx].items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
            shap_summary = "Top risk drivers: " + ", ".join(
                f"{f} ({'+' if v > 0 else ''}{v:.3f})" for f, v in top
            )
        prob = row.get(prob_col, "unknown")
        content = (
            f"Patient {idx}: predicted mortality probability = {prob:.3f}. "
            f"Age = {row.get('age', 'N/A')}, "
            f"APACHE II = {row.get('apache_2_diagnosis', 'N/A')}. "
            f"{shap_summary}"
        )
        documents.append(Document(page_content=content, metadata={"patient_id": idx}))
    return documents


def build_vector_store(documents: list, persist: bool = True):
    """Embed documents with HuggingFace all-MiniLM-L6-v2 and store in Chroma."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks   = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if persist:
        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        store = Chroma.from_documents(
            chunks, embeddings, persist_directory=str(CHROMA_PERSIST_DIR)
        )
        try:
            store.persist()
        except Exception:
            pass  # newer chromadb persists automatically
    else:
        store = Chroma.from_documents(chunks, embeddings)
    return store


def build_qa_chain(vector_store, k: int = 4):
    """Assemble the RAG chain: retriever → Claude LLM → answer."""
    from langchain.chains import RetrievalQA
    from langchain_anthropic import ChatAnthropic

    llm = ChatAnthropic(
        model=LLM_MODEL,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, return_source_documents=True
    )


def ask(chain, question: str) -> tuple[str, list]:
    """Run a question through the RAG chain; returns (answer, source_docs)."""
    result = chain.invoke({"query": question})
    return result["result"], result.get("source_documents", [])
