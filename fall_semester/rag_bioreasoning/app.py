from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ephys_rag.chain import RAGEngine
from ephys_rag.config import DATA_DIR
from ephys_rag.contrasts import pathway_contrasts
from ephys_rag.llm import llm_available
from ephys_rag.tools import ToolRegistry

EXAMPLE_QUESTIONS = [
    "Is NRXN1 DEG-supported in IDH-mutant cycling Ephys_2?",
    "Are neurons present as a labeled identity?",
    "Can we trust MES_like Ephys_2 CellChat edges?",
    "Do T-cell ligand-receptor pairs look like immunological synapse activity?",
]


@st.cache_resource(show_spinner="Loading fall_semester/data and indexing…")
def get_engine() -> RAGEngine:
    return RAGEngine.from_disk()


def _contrast_frame(engine: RAGEngine, compartment: str) -> pd.DataFrame:
    rows = [
        {
            "pathway": item.pathway,
            "Ephys_1": item.ephys_1,
            "Ephys_2": item.ephys_2,
            "delta (E2-E1)": item.delta_e2_minus_e1,
            "mean prob E1": round(item.mean_prob_e1, 4),
            "mean prob E2": round(item.mean_prob_e2, 4),
        }
        for item in pathway_contrasts(engine.interactions)
        if item.compartment == compartment
    ]
    return pd.DataFrame(rows)


st.set_page_config(page_title="Ephys bioreasoning RAG", layout="wide")
st.title("Ephys bioreasoning (CellChat + DEGs + counts)")
st.caption(f"DATA_DIR = {DATA_DIR}. Tools first, then MedGemma + Gemini on the same traces.")

engine = get_engine()

with st.sidebar:
    st.metric("Significant CellChat pairs", len(engine.interactions))
    st.metric("RAG documents", len(engine.documents))
    st.write("LLM:", "configured" if llm_available() else "extractive")
    top_k = st.slider("Retrieved chunks", min_value=4, max_value=20, value=10)
    force_extractive = st.checkbox("Force extractive mode", value=False)
    dual_models = st.checkbox("Dual MedGemma + Gemini", value=True)
    provider = st.selectbox("Single provider (if not dual)", ["medgemma", "gemini"])

overview, tools_tab, chat = st.tabs(["Contrasts", "Tools", "Ask"])

with overview:
    compartment = st.radio("Compartment", ["tumor", "tcell", "myeloid"], horizontal=True)
    frame = _contrast_frame(engine, compartment)
    left, right = st.columns([1.2, 1])
    with left:
        st.dataframe(frame, use_container_width=True, hide_index=True)
    with right:
        st.bar_chart(frame.set_index("pathway")[["Ephys_1", "Ephys_2"]])
    st.markdown(
        "Positive **delta (E2-E1)** means more sender events from Ephys_2. "
        "Join `cellchat_group_counts.csv` before you trust a small group."
    )

with tools_tab:
    st.write("Challenge B-style lookups. Empty is allowed.")
    gene = st.text_input("Gene", value="NRXN1")
    celltype = st.text_input("Cell type contains", value="cycling_tumor")
    idh = st.selectbox("IDH filter", ["", "IDH_Mutant", "IDH_WT"])
    if st.button("Run lookups", type="primary") and gene.strip():
        registry = ToolRegistry(interactions=engine.interactions)
        traces = [
            registry.cellchat_lookup(gene=gene.strip(), celltype=celltype or None),
            registry.deg_lookup(gene=gene.strip(), celltype=celltype or None, idh=idh or None),
            registry.count_lookup(celltype=celltype or None),
        ]
        for trace in traces:
            st.markdown(f"**{trace.tool}** · `{trace.source_file}` · {trace.n_hits} hits")
            st.code(trace.as_text())

with chat:
    choice = st.selectbox("Example questions", ["(type your own)"] + EXAMPLE_QUESTIONS)
    question = st.text_area(
        "Question",
        value="" if choice == "(type your own)" else choice,
        height=90,
    )
    if st.button("Retrieve and answer", type="primary") and question.strip():
        result = engine.ask(
            question.strip(),
            top_k=top_k,
            use_llm=False if force_extractive else True,
            provider=None if dual_models else provider,
            dual=bool(dual_models and not force_extractive),
        )
        st.markdown("### Answer")
        st.write(result["answer"])
        st.markdown("### Tool traces")
        for trace in result["traces"]:
            with st.expander(f"{trace.tool} · {trace.n_hits} hits · {trace.source_file}"):
                st.code(trace.as_text())
        st.markdown("### Retrieved chunks")
        for hit in result["hits"]:
            with st.expander(f"{hit.score:.3f} · {hit.document.kind} · {hit.document.title}"):
                st.write(hit.document.text)
