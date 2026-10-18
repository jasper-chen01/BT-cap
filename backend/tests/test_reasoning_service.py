"""Cache and grounding contracts; providers are replaced to avoid network calls."""
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from backend.services import reasoning_service as service
from ephys_rag.chain import RAGEngine
from ephys_rag.retrieve import RetrievalHit
from ephys_rag.schema import RAGDocument, ToolResult


@pytest.fixture
def grounded():
    trace = ToolResult("deg_lookup", "degs.csv", {"gene": "NRXN1"}, 1,
                       [{"gene": "NRXN1", "n_cells": 53}], "observed")
    hit = RetrievalHit(RAGDocument("one", "NRXN1 evidence", "NRXN1 row", "deg",
                                   {"source_file": "degs.csv"}), 0.75)
    return {"answer": "extractive NRXN1", "context": "current NRXN1 evidence",
            "traces": [trace], "hits": [hit]}


def test_singletons_share_nomination_store(monkeypatch):
    service.get_rag_engine.cache_clear()
    service.get_nomination_store.cache_clear()
    engine = Mock(spec=RAGEngine)
    engine.tools = Mock()
    store = object()
    monkeypatch.setattr(service.RAGEngine, "from_disk", lambda: engine)
    monkeypatch.setattr(service.NominationStore, "from_config", lambda: store)
    try:
        assert service.get_rag_engine() is service.get_rag_engine() is engine
        assert engine.tools.nomination_store is service.get_nomination_store() is store
    finally:
        service.get_rag_engine.cache_clear()
        service.get_nomination_store.cache_clear()


def test_none_preserves_cli_answer_and_exact_json_trace(tmp_path, grounded, monkeypatch):
    provider = Mock(side_effect=AssertionError("provider forbidden"))
    monkeypatch.setattr(service, "generate_answer", provider)
    engine = Mock()
    engine.ask.return_value = grounded
    reasoning = service.ReasoningService(engine=engine, cache=service.PortalAnswerCache(tmp_path))
    first = reasoning.ask("NRXN1?", "none")
    assert first == reasoning.ask("NRXN1?", "none")
    assert first == {"answer": "extractive NRXN1", "model": "none", "cached": False,
                     "traces": [{"tool": "deg_lookup", "source_file": "degs.csv",
                                 "n_hits": 1, "filters": {"gene": "NRXN1"},
                                 "rows": [{"gene": "NRXN1", "n_cells": 53}], "note": "observed"}],
                     "hits": [{"title": "NRXN1 evidence", "score": 0.75,
                               "kind": "deg", "source_file": "degs.csv"}]}
    json.dumps(first, allow_nan=False)
    engine.ask.assert_called_with("NRXN1?", use_llm=False)
    assert list(tmp_path.iterdir()) == []


def test_cache_key_normalizes_question_and_separates_provenance(tmp_path):
    cache = service.PortalAnswerCache(tmp_path)
    key = cache.make_key("  NRXN1  evidence?\n", "medgemma", "context")
    assert key == cache.make_key("nrxn1 evidence?", "medgemma", "context")
    assert len(key) == 64
    assert key != cache.make_key("nrxn1 evidence?", "gemini", "context")
    assert key != cache.make_key("nrxn1 evidence?", "medgemma", "changed")
    cache.schema_version = 2
    assert key != cache.make_key("nrxn1 evidence?", "medgemma", "context")


def test_cache_hit_avoids_provider_and_context_change_misses(tmp_path, grounded, monkeypatch):
    provider = Mock(side_effect=["generated", "updated"])
    monkeypatch.setattr(service, "generate_answer", provider)
    engine = Mock()
    engine.ask.return_value = grounded
    reasoning = service.ReasoningService(engine, service.PortalAnswerCache(tmp_path))
    assert reasoning.ask("NRXN1?", "medgemma")["cached"] is False
    hit = reasoning.ask("NRXN1?", "medgemma")
    assert hit["answer"] == "generated" and hit["cached"] is True
    assert provider.call_count == 1
    provider.assert_called_with("NRXN1?", "current NRXN1 evidence", provider="medgemma")
    grounded["context"] = "changed evidence"
    assert reasoning.ask("NRXN1?", "medgemma")["answer"] == "updated"
    assert provider.call_count == 2


@pytest.mark.parametrize("model", ["none", "medgemma", "gemini"])
def test_empty_rows_and_hits_never_generate(tmp_path, model, monkeypatch):
    monkeypatch.setattr(service, "generate_answer", Mock(side_effect=AssertionError("no evidence")))
    engine = Mock()
    engine.ask.return_value = {"answer": "empty CLI", "context": "empty result diagnostic",
                               "traces": [ToolResult("deg_lookup", "degs.csv", {}, 0, [])], "hits": []}
    result = service.ReasoningService(engine, service.PortalAnswerCache(tmp_path)).ask("missing?", model)
    assert result["answer"] == "not in the tables"
    assert result["cached"] is False
    assert result["traces"][0]["n_hits"] == 0


def test_cache_atomic_write_and_interrupted_files_are_not_hits(tmp_path, monkeypatch):
    cache = service.PortalAnswerCache(tmp_path)
    key = cache.make_key("q", "gemini", "ctx")
    (tmp_path / f"{key}.json.tmp").write_text('{"answer":"partial"}')
    assert cache.get(key) is None
    real_replace = service.os.replace

    def check_replace(source, destination):
        assert Path(source).parent == tmp_path
        assert Path(destination) == tmp_path / f"{key}.json"
        assert cache.get(key) is None
        assert json.loads(Path(source).read_text())["answer"] == "complete"
        real_replace(source, destination)

    monkeypatch.setattr(service.os, "replace", check_replace)
    cache.set(key, "complete")
    assert cache.get(key)["answer"] == "complete"
    (tmp_path / f"{key}.json").write_text("{broken")
    assert cache.get(key) is None


def test_failed_provider_does_not_poison_cache(tmp_path, grounded, monkeypatch):
    monkeypatch.setattr(service, "generate_answer", Mock(side_effect=RuntimeError("offline")))
    engine = Mock()
    engine.ask.return_value = grounded
    with pytest.raises(RuntimeError, match="offline"):
        service.ReasoningService(engine, service.PortalAnswerCache(tmp_path)).ask("q", "gemini")
    assert list(tmp_path.iterdir()) == []


def test_invalid_model_rejected_before_engine(tmp_path):
    engine = Mock(side_effect=AssertionError("must validate first"))
    with pytest.raises(ValueError, match="model"):
        service.ReasoningService(engine, service.PortalAnswerCache(tmp_path)).ask("q", "mg")
    engine.ask.assert_not_called()


def test_import_week2_uses_current_context_and_keeps_source(tmp_path, grounded, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    saved = {"question_id": 9, "question": "NRXN1?", "medgemma": "saved answer",
             "medgemma_error": None, "context": "old evidence"}
    (source / "Q9.json").write_text(json.dumps(saved))
    (source / "Q10.json").write_text(json.dumps({**saved, "question_id": 10, "medgemma_error": "offline"}))
    (source / "Q8.json").write_text(json.dumps({**saved, "question_id": 8}))
    engine = Mock()
    engine.ask.return_value = grounded
    cache = service.PortalAnswerCache(tmp_path / "cache")
    assert service.import_week2_cache(source, cache, engine) == 1
    assert service.import_week2_cache(source, cache, engine) == 0
    record = cache.get(cache.make_key("NRXN1?", "medgemma", "current NRXN1 evidence"))
    assert record["provenance"]["source_file"] == str(source / "Q9.json")
    assert record["provenance"]["question_id"] == 9
    assert cache.get(cache.make_key("NRXN1?", "medgemma", "old evidence")) is None
    monkeypatch.setattr(service, "generate_answer", Mock(side_effect=AssertionError("import must hit")))
    result = service.ReasoningService(engine, cache).ask("NRXN1?", "medgemma")
    assert result["cached"] is True and result["answer"] == "saved answer"
