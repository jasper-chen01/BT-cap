"""HTTP contracts exercise real score filtering and grounded answer serialization."""
import importlib
import sys
from types import ModuleType

import pandas as pd
import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from backend.services import reasoning_service as service
from ephys_rag.nomination import NominationStore
from ephys_rag.schema import ToolResult


@pytest.fixture
def store():
    rows = pd.DataFrame([
        dict(gene="ALPHA", celltype_id="cycling_tumor", IDH_status="IDH_Mutant",
             p_tumor_synaptic=.8, p_immune_synapse=.2, novel=0, housekeeping=0,
             rule_pass=1, ctx_min_n=80, in_tumor_synaptic=1, in_immune_synapse=0,
             in_published_33=1, why_tumor_synaptic="feature=1 (+.4)",
             folds_tumor_synaptic="seed0:fold2"),
        dict(gene="BETA", celltype_id="cycling_tumor", IDH_status="IDH_Mutant",
             p_tumor_synaptic=.7, p_immune_synapse=.3, novel=1, housekeeping=0,
             rule_pass=1, ctx_min_n=60, in_tumor_synaptic=0, in_immune_synapse=1),
        dict(gene="BETA", celltype_id="T cell", IDH_status="IDH_WT",
             p_tumor_synaptic=None, p_immune_synapse=.9, novel=1, housekeeping=0,
             rule_pass=0, ctx_min_n=90, in_tumor_synaptic=0, in_immune_synapse=1),
        dict(gene="RPL5", celltype_id="cycling_tumor", IDH_status="IDH_Mutant",
             p_tumor_synaptic=.95, p_immune_synapse=.4, novel=1, housekeeping=1,
             rule_pass=0, ctx_min_n=90, in_tumor_synaptic=0, in_immune_synapse=0),
    ])
    ranks = pd.DataFrame([
        {"gene": gene,
         **{f"p_{model}__{arm}": score
            for model in ("tumor_synaptic", "immune_synapse")
            for arm in ("tables", "tables_scetm", "scetm_only")},
         **{f"gene_rank_{model}__{arm}": rank
            for model in ("tumor_synaptic", "immune_synapse")
            for arm in ("tables", "tables_scetm", "scetm_only")}}
        for gene, score, rank in [("ALPHA", .85, 2), ("BETA", .75, 3), ("RPL5", .96, 1)]
    ])
    return NominationStore({arm: rows.copy() for arm in ("tables", "tables_scetm", "scetm_only")},
                           ranks, pd.DataFrame({"gene": ["ALPHA", "BETA", "RPL5"],
                                                "emb_00": [1., 1., 0.], "emb_01": [0., 0., 1.]}))


class EvidenceRegistry:
    def cellchat_lookup(self, *, gene):
        return ToolResult("cellchat_lookup", "cellchat.csv", {"gene": gene}, 1,
                          [{"ligand": gene, "receptor": "PARTNER", "prob": .4}])

    def deg_lookup(self, *, gene, pooled=False):
        return ToolResult("deg_lookup", "pooled.csv" if pooled else "idh.csv", {"gene": gene},
                          1, [{"gene": gene, "avg_log2FC": 1.2, "IDH_status": "IDH_WT"}])


@pytest.fixture
def client(store, tmp_path):
    from backend.api import reasoning

    class Engine:
        tools = EvidenceRegistry()

        def ask(self, question, *, use_llm):
            assert use_llm is False
            return {"answer": "observed evidence", "context": "context", "hits": [],
                    "traces": [self.tools.deg_lookup(gene="ALPHA")]}

    app = FastAPI()
    app.include_router(reasoning.router, prefix="/api")
    app.dependency_overrides[reasoning.get_store] = lambda: store
    app.dependency_overrides[reasoning.get_engine] = lambda: Engine()
    app.dependency_overrides[reasoning.get_reasoning_service] = lambda: service.ReasoningService(
        Engine(), service.PortalAnswerCache(tmp_path))
    with TestClient(app) as test_client:
        yield test_client


def test_ask_returns_grounded_trace_contract_and_default_model(client):
    response = client.post("/api/reasoning/ask", json={"question": "ALPHA evidence?"})
    assert response.status_code == 200
    assert response.json() == {"answer": "observed evidence", "model": "none", "cached": False,
                               "hits": [], "traces": [{"tool": "deg_lookup", "source_file": "idh.csv",
                               "filters": {"gene": "ALPHA"}, "n_hits": 1,
                               "rows": [{"gene": "ALPHA", "avg_log2FC": 1.2, "IDH_status": "IDH_WT"}],
                               "note": ""}]}


@pytest.mark.parametrize("model", ["medgemma", "gemini"])
def test_ask_accepts_configured_models(client, monkeypatch, model):
    monkeypatch.setattr(service, "generate_answer", lambda question, context, provider: f"{provider} answer")
    response = client.post("/api/reasoning/ask", json={"question": "ALPHA?", "model": model})
    assert response.status_code == 200
    assert response.json()["answer"] == f"{model} answer"
    assert response.json()["model"] == model


@pytest.mark.parametrize("payload", [{"question": "ALPHA?", "model": "other"},
                                     {"question": ""}, {"question": "   "}, {}, {"question": 42}])
def test_ask_rejects_invalid_requests(client, payload):
    assert client.post("/api/reasoning/ask", json=payload).status_code == 422


def test_gene_detail_preserves_missing_rows_and_evidence(client):
    response = client.get("/api/genes/beta")
    assert response.status_code == 200
    result = response.json()
    assert result["gene"] == "BETA"
    assert len(result["rows"]) == 9
    assert not any(row["model"] == "tumor_synaptic" and row["celltype_id"] == "T cell"
                   for row in result["rows"])
    assert next(row for row in result["rows"] if row["celltype_id"] == "T cell")["p_tumor_synaptic"] is None
    assert result["badges"]["in_go_0001772"] is True
    assert result["cellchat"]["rows"][0]["ligand"] == "BETA"
    assert result["deg"]["source_file"] == "idh.csv"
    assert result["pooled_deg"]["source_file"] == "pooled.csv"
    alpha = client.get("/api/genes/alpha").json()["rows"][0]
    assert alpha["why"] == "feature=1 (+.4)"
    assert alpha["scoring_folds"] == "seed0:fold2"


@pytest.mark.parametrize("params, expected", [
    ({}, ["ALPHA", "BETA"]),
    ({"exclude_housekeeping": False}, ["RPL5", "ALPHA", "BETA"]),
    ({"novel_only": True}, ["BETA"]),
    ({"exclude_housekeeping": False, "rule_pass": False}, ["RPL5"]),
    ({"model": "immune_synapse", "novel_only": True, "rule_pass": True, "min_n": 70}, []),
    ({"model": "immune_synapse", "novel_only": True, "rule_pass": True, "min_n": 50}, ["BETA"]),
    ({"min_n": 100}, []), ({"limit": 1}, ["ALPHA"]),
    ({"limit": 0}, []), ({"arm": "scetm_only"}, ["ALPHA", "BETA"]),
])
def test_ranking_filters_and_static_route_precedence(client, params, expected):
    response = client.get("/api/genes/ranked", params=params)
    assert response.status_code == 200
    result = response.json()
    assert [row["gene"] for row in result["rows"]] == expected
    if result["rows"]:
        row = result["rows"][0]
        assert list(row) == result["columns"]
        assert row["score"] in [.96, .85, .75]
        assert "context_score" in row
    assert result["columns"][:6] == ["gene", "model", "arm", "celltype_id", "IDH_status", "score"]


@pytest.mark.parametrize("params", [{"model": "none"}, {"arm": "bad"}, {"min_n": -1},
                                    {"limit": -1}, {"rule_pass": "maybe"}, {"min_n": "1.5"}])
def test_ranking_rejects_invalid_filters(client, params):
    assert client.get("/api/genes/ranked", params=params).status_code == 422


def test_ranking_csv_columns_stay_stable_when_filters_empty_the_result(client):
    populated = client.get("/api/genes/ranked").json()
    empty = client.get("/api/genes/ranked", params={"min_n": 1000}).json()
    assert empty["rows"] == []
    assert empty["columns"] == populated["columns"]


def test_neighbors_return_similarity_and_model_labels(client):
    result = client.get("/api/genes/alpha/neighbors", params={"model": "immune_synapse", "k": 1})
    assert result.status_code == 200
    assert result.json() == {"gene": "ALPHA", "model": "immune_synapse", "rows": [
        {"gene": "BETA", "similarity": 1., "in_label_set": True}]}
    assert len(client.get("/api/genes/ALPHA/neighbors", params={"k": 100}).json()["rows"]) == 2


@pytest.mark.parametrize("params", [{"k": 0}, {"k": 101}, {"k": "1.5"}, {"model": "gemini"}])
def test_neighbors_validate_model_and_bounds(client, params):
    assert client.get("/api/genes/ALPHA/neighbors", params=params).status_code == 422


@pytest.mark.parametrize("path", ["/api/genes/MISSING", "/api/genes/MISSING/neighbors"])
def test_unknown_gene_returns_404(client, path):
    assert client.get(path).status_code == 404


@pytest.mark.parametrize("path,method", [("/api/genes/ALPHA", "get"), ("/api/genes/ranked", "get"),
                                         ("/api/genes/ALPHA/neighbors", "get"), ("/api/reasoning/ask", "post")])
@pytest.mark.parametrize("partial", [False, True])
def test_absent_or_partial_runtime_returns_preparation_503(tmp_path, monkeypatch, store, path, method, partial):
    from backend.api import reasoning
    from ephys_rag import config

    missing = tmp_path / "missing.csv"
    if partial:
        present = tmp_path / "present.csv"
        pd.DataFrame({"gene": ["ALPHA"]}).to_csv(present, index=False)
        monkeypatch.setattr(config, "CELLCHAT_COUNTS_CSV", present)
        # Even a successfully cached score store must not mask absent evidence.
        monkeypatch.setattr(reasoning, "get_nomination_store", lambda: store)
    else:
        monkeypatch.setattr(config, "ARM_SCORE_FILES", {"tables": missing})
    monkeypatch.setattr(config, "CELLCHAT_CSV", missing)
    service.get_nomination_store.cache_clear()
    service.get_rag_engine.cache_clear()
    app = FastAPI()
    app.include_router(reasoning.router, prefix="/api")
    with TestClient(app) as test_client:
        kwargs = {"json": {"question": "ALPHA?"}} if method == "post" else {}
        response = getattr(test_client, method)(path, **kwargs)
    assert response.status_code == 503
    assert "prepare_reasoning_data.py" in response.json()["detail"]


def test_unavailable_model_returns_503(client, monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("Gemini is not configured")
    monkeypatch.setattr(service, "generate_answer", unavailable)
    response = client.post("/api/reasoning/ask", json={"question": "ALPHA?", "model": "gemini"})
    assert response.status_code == 503
    assert "Gemini" in response.json()["detail"]


def test_main_registers_reasoning_and_genes_under_api(monkeypatch, client, tmp_path):
    # Legacy annotation/cloud dependencies are outside this isolated API contract.
    import backend.api
    from backend.config import settings
    for name in ("annotation", "health", "chat", "auth", "visualization", "embeddings", "preps"):
        module = ModuleType(f"backend.api.{name}")
        module.router = APIRouter()
        monkeypatch.setitem(sys.modules, module.__name__, module)
        monkeypatch.setattr(backend.api, name, module, raising=False)
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.delitem(sys.modules, "backend.main", raising=False)
    main = importlib.import_module("backend.main")
    main.app.dependency_overrides.update(client.app.dependency_overrides)
    try:
        with TestClient(main.app) as registered:
            assert registered.get("/api/genes/ranked").status_code == 200
            assert registered.get("/api/genes/ALPHA").status_code == 200
            assert registered.get("/api/genes/ALPHA/neighbors").status_code == 200
            assert registered.post("/api/reasoning/ask", json={"question": "ALPHA?"}).status_code == 200
    finally:
        main.app.dependency_overrides.clear()
        sys.modules.pop("backend.main", None)
