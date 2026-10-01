from ephys_rag.chain import format_traces
from ephys_rag.schema import ToolResult


def test_tool_trace_preserves_provenance_and_empty_results():
    """Removing trace provenance or empty-result disclosure must fail."""
    trace = ToolResult(
        tool="deg_lookup",
        source_file="degs.csv",
        filters={"gene": "NRXN1"},
        n_hits=0,
        rows=[],
    )

    rendered = format_traces([trace])

    assert "tool=deg_lookup" in rendered
    assert "file=degs.csv" in rendered
    assert "gene=NRXN1" in rendered
    assert "empty result" in rendered


def test_extractive_answer_includes_question_and_context():
    """Dropping either the question or evidence from offline output must fail."""
    from ephys_rag.llm import extractive_answer

    answer = extractive_answer("What does NRXN bind?", "tool evidence")

    assert "What does NRXN bind?" in answer
    assert "tool evidence" in answer


def test_tool_trace_uses_ascii_omission_marker_for_windows_terminals():
    trace = ToolResult(
        tool="cellchat_lookup",
        source_file="interactions.csv",
        filters={},
        n_hits=2,
        rows=[{"pair": "A_B"}, {"pair": "C_D"}],
    )

    rendered = trace.as_text(limit=1)
    assert "... 1 more rows omitted" in rendered
    assert "…" not in rendered


def test_pathway_knowledge_reads_utf8_independently_of_windows_locale():
    """Unicode biological prose must not depend on the machine code page."""
    from ephys_rag.ingest import load_pathway_knowledge

    pathways = load_pathway_knowledge()

    assert "NRXN" in pathways
