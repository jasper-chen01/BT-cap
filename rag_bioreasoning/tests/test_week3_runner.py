from __future__ import annotations

import pytest

from ephys_rag.evaluation import EvaluationQuestion


def runner_api():
    try:
        from ephys_rag.week3_cache import cache_path, write_run_atomic
        from ephys_rag.week3_runner import load_cached_runs, run_repeated_evaluation
    except ModuleNotFoundError:
        pytest.fail("Week 3 runner API is not implemented")
    return cache_path, write_run_atomic, load_cached_runs, run_repeated_evaluation


class CountingEngine:
    def __init__(self) -> None:
        self.calls = 0

    def ask(self, question, *, provider, **kwargs):
        self.calls += 1
        return {
            "question": question,
            "answer": f"answer {self.calls}",
            "provider": provider,
            "model": "trace-rules-v1",
            "elapsed_ms": 1.0,
            "context": "Tool: count_lookup\n- group=T cell / Ephys_1, n_cells=2662",
            "traces": [
                {
                    "tool": "count_lookup",
                    "source_file": "counts.csv",
                    "filters": {},
                    "n_hits": 1,
                    "rows": [{"group": "T cell / Ephys_1", "n_cells": 2662}],
                    "note": "",
                }
            ],
            "hits": [],
        }


def q28() -> EvaluationQuestion:
    return EvaluationQuestion(
        id="q28",
        question="How many T cells are Ephys_1?",
        required_terms=["T cell", "Ephys_1"],
        required_tools=["count_lookup"],
    )


def existing_record() -> dict:
    return {
        "qid": "Q28",
        "provider": "mock",
        "model": "trace-rules-v1",
        "run": 1,
        "question": q28().question,
        "context": "cached context",
        "answer": "cached answer",
        "ts": "2026-10-01T12:00:00+00:00",
        "traces": [],
        "retrieved_titles": [],
    }


def test_runner_calls_model_only_for_missing_cache_files(tmp_path):
    """Resume must never pay for or overwrite a valid completed run."""
    cache_path, write_run_atomic, _, run_repeated_evaluation = runner_api()
    write_run_atomic(cache_path(tmp_path, "q28", "mock", 1), existing_record())
    engine = CountingEngine()

    records = run_repeated_evaluation(
        engine,
        [q28()],
        provider="mock",
        cache_label="mock",
        runs_dir=tmp_path,
        repeats=3,
    )

    assert engine.calls == 2
    assert [record["run"] for record in records] == [1, 2, 3]
    assert records[0]["answer"] == "cached answer"


def test_replay_reads_cache_without_accepting_an_engine(tmp_path):
    """Replay must be structurally unable to invoke a live provider."""
    cache_path, write_run_atomic, load_cached_runs, _ = runner_api()
    write_run_atomic(cache_path(tmp_path, "q28", "mock", 1), existing_record())

    records = load_cached_runs(
        [q28()], cache_label="mock", runs_dir=tmp_path, repeats=1
    )

    assert len(records) == 1
    assert records[0]["answer"] == "cached answer"
