from __future__ import annotations

import pytest

from ephys_rag.evaluation import EvaluationQuestion


def audit_api():
    try:
        from ephys_rag.week3_audit import aggregate_question, score_run
    except ModuleNotFoundError:
        pytest.fail("Week 3 audit API is not implemented")
    return aggregate_question, score_run


def question() -> EvaluationQuestion:
    return EvaluationQuestion(
        id="q21",
        question="Is NRXN1 supported?",
        required_terms=["NRXN1", "Ephys_2"],
        required_tools=["cellchat_lookup", "deg_lookup"],
    )


def grounded_record(answer: str) -> dict:
    return {
        "qid": "Q21",
        "run": 1,
        "answer": answer,
        "traces": [
            {
                "tool": "cellchat_lookup",
                "rows": [
                    {
                        "ligand": "NRXN1",
                        "receptor": "NLGN1",
                        "pair": "NRXN1_NLGN1",
                    }
                ],
            },
            {
                "tool": "deg_lookup",
                "rows": [
                    {
                        "gene": "NRXN1",
                        "direction": "Ephys_2_high",
                        "celltype_id": "cycling_tumor",
                    }
                ],
            },
        ],
    }


def test_score_run_accepts_trace_supported_pair_and_deg():
    """Supported interactions and DEG directions must not be called inventions."""
    _, score_run = audit_api()

    scored = score_run(
        grounded_record(
            "NRXN1 is Ephys_2-high and participates in NRXN1_NLGN1."
        ),
        question(),
    )

    assert scored["hit"] is True
    assert scored["trace_terms"] is True
    assert scored["invented_pairs"] == []
    assert scored["invented_degs"] == []
    assert scored["faithful"] is True


def test_score_run_flags_pair_and_deg_not_present_in_traces():
    """An invented pair or reversed DEG direction must make a run unfaithful."""
    _, score_run = audit_api()

    scored = score_run(
        grounded_record(
            "NRXN1 is Ephys_1-high and participates in NRXN1_GRIA2."
        ),
        question(),
    )

    assert scored["invented_pairs"] == ["NRXN1_GRIA2"]
    assert scored["invented_degs"] == ["NRXN1:Ephys_1_high"]
    assert scored["faithful"] is False


def test_score_run_accepts_prefixed_glutamate_pair_and_trace_field_names():
    """Pair suffixes and words such as CSV/GENE must not become false inventions."""
    _, score_run = audit_api()
    item = EvaluationQuestion(
        id="q35",
        question="Name a glutamate ligand.",
        required_terms=["SLC1A3"],
        required_tools=["pathway_filter"],
    )
    record = {
        "qid": "Q35",
        "run": 1,
        "answer": (
            "SLC1A3 is supported by Glutamate-Glu-SLC1A3_GLS_GRIA2. "
            "CSV row direction is Ephys_2_high."
        ),
        "traces": [
            {
                "tool": "pathway_filter",
                "rows": [
                    {
                        "pair": "Glutamate-Glu-SLC1A3_GLS_GRIA2",
                        "gene": "SLC1A3",
                        "direction": "Ephys_2_high",
                    }
                ],
            }
        ],
    }

    scored = score_run(record, item)

    assert scored["invented_pairs"] == []
    assert scored["invented_degs"] == []
    assert scored["faithful"] is True


def test_aggregate_question_reports_k_of_five_and_unstable_state():
    """Collapsing repeated runs must preserve the requested k/5 interpretation."""
    aggregate_question, _ = audit_api()
    scores = [
        {"faithful": value, "hit": value, "invented_pairs": [], "invented_degs": []}
        for value in [True, True, True, False, False]
    ]

    result = aggregate_question(question(), scores)

    assert result["medgemma_faithful"] == "3/5"
    assert result["medgemma_hit"] == "3/5"
    assert result["unstable"] is True


def test_build_reports_writes_required_table_appendix_and_focus_notes(tmp_path):
    """Replay must materialize every Week 3 deliverable, including Q36 notes."""
    try:
        from ephys_rag.week3_audit import build_week3_reports
    except ImportError:
        pytest.fail("Week 3 report builder is not implemented")
    questions = [
        EvaluationQuestion("q25", "EGFR direction?", ["EGFR"], required_tools=[]),
        EvaluationQuestion("q27", "Which group has six cells?", ["6"], required_tools=[]),
        EvaluationQuestion("q36", "Do T cells send NRXN?", ["T cell"], required_tools=[]),
    ]
    records = []
    for item in questions:
        for run in range(1, 6):
            records.append(
                {
                    "qid": item.id.upper(),
                    "run": run,
                    "answer": item.required_terms[0],
                    "traces": [{"tool": "lookup", "rows": [{"gene": item.required_terms[0]}]}],
                }
            )

    paths = build_week3_reports(
        questions,
        records,
        output_dir=tmp_path,
        gemini_answers={"Q25": "cached Gemini answer"},
    )

    assert paths["table_csv"].exists()
    assert paths["appendix_csv"].exists()
    assert "Q25" in paths["table_csv"].read_text(encoding="utf-8")
    assert "Q36" in paths["appendix_csv"].read_text(encoding="utf-8")
    notes = paths["instability_md"].read_text(encoding="utf-8")
    assert all(qid in notes for qid in ("Q25", "Q27", "Q36"))


def test_build_reports_strip_trailing_whitespace_from_multiline_answers(tmp_path):
    """Generated CSVs must remain clean even when cached model prose has spaces."""
    from ephys_rag.week3_audit import build_week3_reports

    item = EvaluationQuestion("q9", "Is PTN supported?", ["PTN"], required_tools=[])
    records = [
        {
            "qid": "Q9",
            "run": run,
            "answer": "PTN is supported.  \nSecond line. \t",
            "traces": [{"tool": "lookup", "rows": [{"gene": "PTN"}]}],
        }
        for run in range(1, 6)
    ]

    paths = build_week3_reports(
        [item],
        records,
        output_dir=tmp_path,
        gemini_answers={"Q9": "Gemini line.  \nSecond line. \t"},
    )

    for key in ("table_csv", "run_csv"):
        lines = paths[key].read_text(encoding="utf-8").splitlines()
        assert all(line == line.rstrip() for line in lines)


def test_build_reports_applies_explicit_human_faithfulness_overrides(tmp_path):
    """Human role/logic review must be distinct from exact required-term hits."""
    try:
        from ephys_rag.week3_audit import build_week3_reports
    except ImportError:
        pytest.fail("Week 3 report builder is not implemented")
    item = EvaluationQuestion(
        "q10",
        "Which receptor does MHC-II use?",
        ["CD4"],
        required_tools=["cellchat_lookup"],
    )
    records = [
        {
            "qid": "Q10",
            "run": run,
            "answer": "HLA-DRA is the receptor in HLA-DRA_CD4.",
            "traces": [
                {"tool": "cellchat_lookup", "rows": [{"pair": "HLA-DRA_CD4"}]}
            ],
        }
        for run in range(1, 6)
    ]

    paths = build_week3_reports(
        [item],
        records,
        output_dir=tmp_path,
        audit_overrides={
            "Q10": {
                "faithful_runs": [],
                "note": "All runs reverse ligand and receptor roles.",
            }
        },
    )

    table = paths["table_csv"].read_text(encoding="utf-8")
    assert "0/5" in table
    assert "5/5" in table
    assert "human-reviewed" in table
    assert "reverse ligand and receptor roles" in table
    run_rows = paths["run_csv"].read_text(encoding="utf-8")
    assert "automatic_faithful,audited_faithful" in run_rows
