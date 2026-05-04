"""
LangChain-powered clinical Q&A over ICU patient predictions.

Builds a retrieval-augmented generation (RAG) pipeline that lets clinicians
ask natural-language questions about individual patients or cohort-level
patterns, grounded in the model's predictions and SHAP explanations.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()

# Using claude-sonnet-4-6 (current Sonnet) for strong clinical reasoning with
# low latency.  Switch to claude-opus-4-7 if deeper reasoning is required.
LLM_MODEL = "claude-sonnet-4-6"

# Chunk size chosen to fit roughly one patient record + SHAP summary per chunk,
# keeping retrieval granular enough for patient-level questions.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

CHROMA_PERSIST_DIR = Path("models/chroma_db")


def build_patient_documents(
    df: pd.DataFrame,
    shap_dict: dict[int, dict[str, float]],
    prob_col: str = "mortality_prob",
) -> list[Document]:
    """Convert each patient row + SHAP values into a LangChain Document.

    Each document captures: patient ID, key vitals, predicted mortality
    probability, and the top-5 SHAP drivers.  This structured prose is what
    the retriever embeds and searches over.
    """
    documents = []
    for idx, row in df.iterrows():
        shap_summary = ""
        if idx in shap_dict:
            top_features = sorted(shap_dict[idx].items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
            shap_summary = "Top risk drivers: " + ", ".join(
                f"{feat} ({'+' if val > 0 else ''}{val:.3f})" for feat, val in top_features
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


def build_vector_store(documents: list[Document], persist: bool = True) -> Chroma:
    """Embed documents with a lightweight HuggingFace model and store in Chroma.

    all-MiniLM-L6-v2 is chosen for its speed / quality trade-off on short
    clinical text fragments.  No API key required — runs locally.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if persist:
        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        store = Chroma.from_documents(
            chunks, embeddings, persist_directory=str(CHROMA_PERSIST_DIR)
        )
        store.persist()
    else:
        store = Chroma.from_documents(chunks, embeddings)

    return store


def build_qa_chain(vector_store: Chroma, k: int = 4) -> RetrievalQA:
    """Assemble the RAG chain: retriever → Claude LLM → answer.

    k=4 retrieved chunks balances context completeness against prompt length.
    The chain is configured with return_source_documents=True so the Streamlit
    UI can cite which patient records informed each answer.
    """
    llm = ChatAnthropic(
        model=LLM_MODEL,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0,  # deterministic responses for clinical use
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )
    return chain


def ask(chain: RetrievalQA, question: str) -> tuple[str, list[Document]]:
    """Run a natural-language question through the RAG chain.

    Returns the answer string and the source documents so callers can
    display provenance information alongside the response.
    """
    result = chain.invoke({"query": question})
    return result["result"], result.get("source_documents", [])
