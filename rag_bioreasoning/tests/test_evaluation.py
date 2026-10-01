import json
from types import SimpleNamespace

from ephys_rag.evaluation import (
    EvaluationQuestion,
    load_question_manifest,
    run_evaluation,
    score_required_terms,
    write_evaluation_reports,
)


class FakeEvaluationEngine:
    """Provides the stable RAGEngine.ask result boundary without cloud calls."""

    def ask(self, question, *, provider, **kwargs):
        if question == "fails":
            raise RuntimeError(
                "Bearer abc123 credential C:\\private\\service-account.json"
            )
        trace = SimpleNamespace(
            tool="cellchat_lookup",
            source_file="all_significant_interactions.csv",
            filters={"gene": "NRXN1"},
            n_hits=1,
            rows=[{"pair": "NRXN1_NLGN1"}],
            note="",
        )
        hit = SimpleNamespace(
            document=SimpleNamespace(title="NRXN1_NLGN1 interaction")
        )
        return {
            "provider": provider,
            "model": "test-model",
            "answer": "NRXN1 binds NLGN1.",
            "elapsed_ms": 10.5,
            "traces": [trace],
            "hits": [hit],
        }


def test_required_terms_are_case_insensitive_and_preserve_manifest_spelling():
    """Capitalization differences must not create false misses."""
    found, missing = score_required_terms(
        "NRXN1 binds NLGN1 and LRRTM2.",
        ["nrxn1", "NLGN1", "LRRTM2", "DAG1"],
    )

    assert found == ["nrxn1", "NLGN1", "LRRTM2"]
    assert missing == ["DAG1"]


def test_partial_failure_keeps_successful_records_and_redacts_secrets():
    """One provider error must not erase completed evaluations or leak credentials."""
    questions = [
        EvaluationQuestion("q1", "works", ["NLGN1"]),
        EvaluationQuestion("q2", "fails", ["answer"]),
    ]

    summary = run_evaluation(
        FakeEvaluationEngine(),
        questions,
        ["medgemma"],
    )

    assert len(summary.records) == 2
    assert summary.records[0].error is None
    assert summary.records[0].found_terms == ["NLGN1"]
    assert summary.records[1].error is not None
    assert "abc123" not in summary.records[1].error
    assert "service-account.json" not in summary.records[1].error
    assert summary.success is False


def test_reports_write_complete_json_and_markdown(tmp_path):
    """Losing machine-readable or presentation-readable output must fail."""
    questions = [EvaluationQuestion("q1", "works", ["NLGN1"])]
    summary = run_evaluation(FakeEvaluationEngine(), questions, ["medgemma"])

    json_path, markdown_path = write_evaluation_reports(summary, tmp_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert payload["success"] is True
    assert payload["records"][0]["traces"][0]["tool"] == "cellchat_lookup"
    assert payload["records"][0]["tools_fired"] == ["cellchat_lookup"]
    assert payload["records"][0]["trace_found_terms"] == ["NLGN1"]
    assert payload["records"][0]["trace_missing_terms"] == []
    assert payload["records"][0]["retrieved_titles"] == [
        "NRXN1_NLGN1 interaction"
    ]
    assert "Medgemma" in markdown
    assert "NRXN1 binds NLGN1" in markdown


def test_evaluation_reports_missing_required_tools():
    question = EvaluationQuestion(
        "q1",
        "works",
        ["NLGN1"],
        required_tools=["deg_lookup"],
    )

    summary = run_evaluation(FakeEvaluationEngine(), [question], ["medgemma"])

    assert summary.records[0].tools_fired == ["cellchat_lookup"]
    assert summary.records[0].missing_required_tools == ["deg_lookup"]


def test_evaluation_forwards_compact_context_controls():
    """A requested compact profile must reach every engine call unchanged."""

    class RecordingEngine(FakeEvaluationEngine):
        def __init__(self):
            self.calls = []

        def ask(self, question, *, provider, **kwargs):
            self.calls.append({"provider": provider, **kwargs})
            return super().ask(question, provider=provider, **kwargs)

    engine = RecordingEngine()
    questions = [EvaluationQuestion("q1", "works", ["NLGN1"])]

    run_evaluation(
        engine,
        questions,
        ["medgemma"],
        top_k=0,
        trace_limit=2,
        include_trace_catalog=True,
    )

    assert engine.calls == [
        {
            "provider": "medgemma",
            "top_k": 0,
            "trace_limit": 2,
            "include_trace_catalog": True,
        }
    ]


def test_committed_manifest_contains_every_unanswered_week_two_question():
    """Dropping any Q9-Q38 question must fail the weekly evaluation contract."""
    questions = load_question_manifest()

    assert len(questions) == 30
    assert {question.id for question in questions} == {
        f"q{number}" for number in range(9, 39)
    }
    by_id = {question.id: question for question in questions}
    assert by_id["q21"].required_tools == ["deg_lookup"]
    assert by_id["q27"].required_tools == ["count_lookup"]
    assert by_id["q30"].required_tools == ["cellchat_lookup", "deg_lookup"]
