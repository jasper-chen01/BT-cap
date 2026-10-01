from ephys_rag.week2_report import (
    build_week2_rows,
    concise_answer,
    render_week2_markdown,
)


def _record(provider: str, answer: str) -> dict:
    return {
        "question_id": "q9",
        "question": "Is PTN shared?",
        "provider": provider,
        "answer": answer,
        "tools_fired": ["cellchat_lookup", "deg_lookup"],
        "trace_found_terms": ["PTN", "PTPRZ1"],
        "trace_missing_terms": [],
    }


def test_build_week2_rows_pairs_models_and_reports_trace_coverage():
    medgemma = {"records": [_record("medgemma", "PTN is shared.\n\n* tool=cellchat_lookup")]}
    gemini = {"records": [_record("gemini", "PTN is a shared scaffold.")]}

    rows = build_week2_rows(
        medgemma,
        gemini,
        invention_audit={"q9": "No"},
    )

    assert rows == [
        {
            "Question": "Q9: Is PTN shared?",
            "Tool that fired": "cellchat_lookup, deg_lookup",
            "Required term in traces?": "Yes (2/2)",
            "MedGemma answer": "PTN is shared.",
            "Gemini answer": "PTN is a shared scaffold.",
            "Either model invented a pair/DEG?": "No",
        }
    ]


def test_render_week2_markdown_uses_assignment_columns_and_escapes_pipes():
    row = {
        "Question": "Q9: Is PTN shared?",
        "Tool that fired": "cellchat_lookup",
        "Required term in traces?": "Yes (2/2)",
        "MedGemma answer": "Shared | not private",
        "Gemini answer": "Shared scaffold",
        "Either model invented a pair/DEG?": "No",
    }

    report = render_week2_markdown([row])

    assert "| Question | Tool that fired | Required term in traces? | MedGemma answer | Gemini answer | Either model invented a pair/DEG? |" in report
    assert "Shared \\| not private" in report


def test_concise_answer_prefers_interpretation_over_observation_details():
    answer = (
        "### Observation\nPTN_PTPRZ1 appears in many rows.\n\n"
        "### Interpretation\nPTN is a shared scaffold across Ephys states.\n\n"
        "(Note: neurons are not a labeled identity.)"
    )

    assert concise_answer(answer) == "PTN is a shared scaffold across Ephys states."


def test_concise_answer_recognizes_numbered_and_inline_interpretation_labels():
    numbered = "### 1. Observation\nEvidence.\n### 2. Interpretation\nShared scaffold."
    inline = "Evidence list.\n* **Interpretation:** There are 2,662 versus 1,463 T cells."

    assert concise_answer(numbered) == "Shared scaffold."
    assert concise_answer(inline) == "There are 2,662 versus 1,463 T cells."
