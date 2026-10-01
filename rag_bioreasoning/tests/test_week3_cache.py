from __future__ import annotations

import json

import pytest


def cache_api():
    try:
        from ephys_rag.week3_cache import (
            InvalidRunCache,
            cache_path,
            load_run,
            write_run_atomic,
        )
    except ModuleNotFoundError:
        pytest.fail("Week 3 cache API is not implemented")
    return InvalidRunCache, cache_path, load_run, write_run_atomic


def sample_record() -> dict:
    return {
        "qid": "Q21",
        "provider": "ollama",
        "model": "medgemma:4b-it-q4_K_M",
        "run": 3,
        "question": "Is NRXN1 supported?",
        "context": "gene=NRXN1, direction=Ephys_2_high",
        "answer": "Yes. NRXN1 is Ephys_2-high.",
        "ts": "2026-10-01T12:00:00+00:00",
        "traces": [],
        "retrieved_titles": [],
    }


def test_cache_round_trip_uses_assignment_filename_and_complete_json(tmp_path):
    """A mismatched filename or partial record would make replay irreproducible."""
    _, cache_path, load_run, write_run_atomic = cache_api()
    path = cache_path(tmp_path, "q21", "medgemma", 3)

    write_run_atomic(path, sample_record())

    assert path.name == "Q21_medgemma_run3.json"
    assert load_run(path) == sample_record()
    assert list(tmp_path.glob("*.tmp")) == []


def test_corrupt_or_incomplete_cache_is_rejected(tmp_path):
    """Scoring a truncated response as a real run would corrupt k/5 counts."""
    InvalidRunCache, _, load_run, _ = cache_api()
    corrupt = tmp_path / "Q21_medgemma_run3.json"
    corrupt.write_text(json.dumps({"qid": "Q21"}), encoding="utf-8")

    with pytest.raises(InvalidRunCache, match="missing"):
        load_run(corrupt)

