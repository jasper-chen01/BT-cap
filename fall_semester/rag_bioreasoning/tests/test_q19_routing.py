from __future__ import annotations

import pytest

from ephys_rag.chain import RAGEngine
from ephys_rag.tools.registry import dispatch_tools, infer_query_slots


Q19 = (
    "Interpret NRXN1–NLGN1 and glutamate–GRIA in this dataset. "
    "Are neurons required for that interpretation?"
)


@pytest.fixture(scope="module")
def engine() -> RAGEngine:
    return RAGEngine.from_disk()


def test_q19_pathway_name_does_not_infer_a_myeloid_compartment():
    """Catches matching the TAM alias inside the word ``glutamate``."""
    slots = infer_query_slots(Q19)

    assert slots["pathway"] == "Glutamate"
    assert slots["celltype"] is None
    assert slots["compartment"] is None


@pytest.mark.parametrize("separator", ["_", "-", "–"])
def test_gene_pair_inference_accepts_common_pair_separators(separator: str):
    """Catches losing ligand–receptor routing when pair punctuation changes."""
    slots = infer_query_slots(f"Interpret NRXN1{separator}NLGN1 in this dataset.")

    assert slots["pair"] == ("NRXN1", "NLGN1")


def test_hyphenated_gene_symbol_is_not_misread_as_a_pair():
    """Catches the ASCII-hyphen pair syntax breaking HLA-DRA gene lookup."""
    slots = infer_query_slots("Is HLA-DRA scored immune or synaptic in TAMs?")

    assert slots["genes"][0] == "HLA-DRA"
    assert slots["pair"] is None


@pytest.mark.parametrize(
    ("question", "tool", "filters"),
    [
        (
            "What does CLEC bind on T cells in this table?",
            "cellchat_lookup",
            {"celltype": "T cell", "compartment": "tcell"},
        ),
        (
            "How many T cells are labeled Ephys_1 vs Ephys_2?",
            "count_lookup",
            {"celltype": "T cell"},
        ),
        (
            "Do T cells send NRXN or Glutamate in this table?",
            "cellchat_lookup",
            {"celltype": "T cell", "compartment": "tcell"},
        ),
        (
            "Which gene ranks highest for immune synapse in T cells?",
            "nominate_lookup",
            {"celltype": "T cell"},
        ),
    ],
)
def test_plural_t_cell_questions_keep_their_cell_filters(
    engine: RAGEngine, question: str, tool: str, filters: dict[str, str]
):
    """Catches Q15/Q28/Q36/Q41 silently widening plural T-cell queries."""
    slots = infer_query_slots(question)
    traces = dispatch_tools(question, engine.tools)

    assert slots["celltype"] == "T cell"
    assert slots["compartment"] == "tcell"
    assert any(
        trace.tool == tool
        and all(trace.filters.get(key) == value for key, value in filters.items())
        for trace in traces
    )


@pytest.mark.parametrize(
    "question",
    [
        "Does HLA-DRA bind CD4?",
        "Interpret HLA-DRA in this dataset.",
    ],
)
def test_known_hyphenated_gene_stays_a_gene_in_pair_wording(question: str):
    """Catches pair-oriented wording splitting the known HLA-DRA symbol."""
    slots = infer_query_slots(question)

    assert slots["genes"][0] == "HLA-DRA"
    assert slots["pair"] is None


def test_rejected_hyphenated_symbol_does_not_hide_a_later_valid_pair():
    """Catches stopping pair search after rejecting the first candidate."""
    slots = infer_query_slots(
        "Compare HLA-DRA with NRXN1_NLGN1 in this dataset."
    )

    assert slots["pair"] == ("NRXN1", "NLGN1")


def test_q19_engine_returns_both_programs_and_neuron_annotation_without_tam_filter(
    engine: RAGEngine,
):
    """Catches Q19 dropping NRXN evidence by constraining it to TAM/myeloid rows."""
    result = engine.ask(Q19, use_llm=False)
    traces = result["traces"]
    trace_text = "\n".join(trace.as_text(limit=100) for trace in traces)

    assert "NRXN" in trace_text
    assert "Glutamate" in trace_text
    assert any(
        trace.n_hits > 0 and trace.filters.get("pathway") == "Glutamate"
        for trace in traces
    )
    assert any(
        trace.n_hits > 0
        and (
            trace.filters.get("gene") == "NRXN1"
            or trace.filters.get("ligand") == "NRXN1"
        )
        for trace in traces
    )
    assert any(trace.tool == "annotation_lookup" for trace in traces)
    assert all(trace.filters.get("celltype") != "TAM" for trace in traces)
    assert all(trace.filters.get("compartment") != "myeloid" for trace in traces)
