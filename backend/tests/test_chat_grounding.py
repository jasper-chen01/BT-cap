"""Chat must preserve table evidence and never send biology to bare Gemini."""
import asyncio
import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.models.schemas import ChatSession
from backend.services import reasoning_service
from ephys_rag.schema import ToolResult


@pytest.fixture
def chat(monkeypatch, tmp_path):
    # The legacy annotation runtime needs scanpy/faiss, unrelated to routing.
    annotation = ModuleType("backend.services.annotation_service")
    annotation.AnnotationService = lambda: SimpleNamespace(
        get_status=lambda: dict(reference_loaded=True, index_loaded=True, index_size=12))
    monkeypatch.setitem(sys.modules, annotation.__name__, annotation)
    monkeypatch.setitem(sys.modules, "openpyxl", ModuleType("openpyxl"))
    sys.modules.pop("backend.services.chat_agent", None)
    sys.modules.pop("backend.api.chat", None)
    module = importlib.import_module("backend.services.chat_agent")
    monkeypatch.setattr(module.ChatAgent, "_init_gemini", lambda self: None)
    agent = module.ChatAgent()
    calls = []

    class Engine:
        def ask(self, question, *, use_llm):
            assert use_llm is False
            calls.append(question)
            rows = [] if "invented" in question else [{"gene": "NRXN1", "avg_log2FC": 1.2}]
            return dict(answer="observed table evidence", context="evidence context", hits=[],
                        traces=[ToolResult("deg_lookup", "idh.csv", {}, len(rows), rows)])

    service = reasoning_service.ReasoningService(
        Engine(), reasoning_service.PortalAnswerCache(tmp_path))
    # Replace only acquisition, keeping real grounding and serialization.
    monkeypatch.setattr(reasoning_service, "get_rag_engine", lambda: service.engine)
    agent.gemini_enabled = True

    async def bare_gemini(session, message):
        pytest.fail(f"Bare Gemini received biological question: {message}")

    monkeypatch.setattr(agent, "_gemini_response", bare_gemini)
    yield module, agent, calls
    sys.modules.pop("backend.services.chat_agent", None)
    sys.modules.pop("backend.api.chat", None)


def run(agent, text, **kwargs):
    session = kwargs.pop("session", ChatSession(session_id="grounding", messages=[]))
    return asyncio.run(agent.process_message(session, text, **kwargs))


@pytest.mark.parametrize("question", [
    "What evidence supports NRXN1?", "Show ligand–receptor pairs",
    "Which pathways have the most interactions?", "Compare Ephys states",
    "Compare IDH-mutant and IDH-WT", "Which cell types communicate?",
    "Show microglia evidence", "What about invented pancreatic biology?",
    "Does the invented pathway occur?", "Is invented gene qxyz represented?",
    "What is IDH status?", "NRXN1 status", "Ephys status",
    "Help me find NRXN1 evidence", "Hi, what supports NRXN1?",
])
@pytest.mark.parametrize("with_context", [False, True])
def test_biological_chat_uses_grounded_service(chat, question, with_context):
    _, agent, calls = chat
    session = ChatSession(session_id="grounding", messages=[],
                          analysis_context={"total_cells": 10} if with_context else None)
    response = run(agent, question, session=session)
    assert calls == [question]
    assert response.message.content == ("not in the tables" if "invented" in question
                                        else "observed table evidence")
    assert response.message.traces == [{"tool": "deg_lookup", "source_file": "idh.csv",
        "filters": {}, "n_hits": 0 if "invented" in question else 1,
        "rows": [] if "invented" in question else [{"gene": "NRXN1", "avg_log2FC": 1.2}],
        "note": ""}]


@pytest.mark.parametrize("text,content,action", [
    ("hello", "Hello!", None), ("help", "How to use", None),
    ("help me use the portal", "How to use", None),
    ("Check system status", "System Status: Ready", None),
    ("Check portal status", "System Status: Ready", None),
    ("Check job status", "System Status: Ready", None),
    ("help me read portal messages", "How to use", None),
    ("Check system status messages", "System Status: Ready", None),
    ("Help me understand portal messages", "How to use", None),
    ("annotate", "No file uploaded", "upload_file"),
])
def test_existing_portal_commands_keep_precedence(chat, text, content, action):
    _, agent, calls = chat
    response = run(agent, text)
    assert content in response.message.content
    assert response.requires_action == action
    assert calls == []


def test_upload_and_annotation_keep_precedence(chat, monkeypatch):
    _, agent, calls = chat
    session = ChatSession(session_id="upload", messages=[],
                          uploaded_files=[{"filename": "cells.h5ad", "path": "cells.h5ad"}])
    response = run(agent, "NRXN1 evidence", session=session, file_path="cells.h5ad")
    assert response.message.file_uploaded == "cells.h5ad"
    assert response.requires_action == "annotate"
    from backend.models.schemas import ChatMessage, ChatResponse

    async def annotate(session, file_path):
        assert file_path == "cells.h5ad"
        return ChatResponse(session_id=session.session_id,
                            message=ChatMessage(role="assistant", content="Annotated cells"))

    monkeypatch.setattr(agent, "annotate_file", annotate)
    assert run(agent, "annotate", session=session).message.content == "Annotated cells"
    assert calls == []


@pytest.mark.parametrize("gemini_enabled", [True, False])
def test_umap_summary_flow_is_preserved(chat, monkeypatch, gemini_enabled):
    _, agent, calls = chat
    agent.gemini_enabled = gemini_enabled
    session = ChatSession(session_id="umap", messages=[], analysis_context={"total_cells": 42})
    from backend.models.schemas import ChatMessage, ChatResponse

    async def summarize(session, message):
        return ChatResponse(session_id=session.session_id,
                            message=ChatMessage(role="assistant", content="UMAP summary"))

    monkeypatch.setattr(agent, "_gemini_response", summarize)
    response = run(agent, "What are the total cells?", session=session)
    assert response.message.content == ("UMAP summary" if gemini_enabled else "total_cells = 42")
    assert calls == []


def test_chat_api_keeps_traces_in_response_and_session(chat, monkeypatch):
    _, agent, _ = chat
    api = importlib.import_module("backend.api.chat")
    monkeypatch.setattr(api, "ChatAgent", lambda: agent)
    app = FastAPI()
    app.include_router(api.router, prefix="/api")
    with TestClient(app) as client:
        session_id = client.post("/api/chat/session").json()["session_id"]
        response = client.post(f"/api/chat/{session_id}/message", data={"message": "NRXN1 evidence?"})
        assert response.status_code == 200
        message = response.json()["message"]
        assert message["content"] == "observed table evidence"
        assert message["traces"][0]["source_file"] == "idh.csv"
        saved = client.get(f"/api/chat/session/{session_id}").json()
        assert saved["messages"][-1] == message


@pytest.mark.parametrize("error", [FileNotFoundError("missing runtime"),
                                    RuntimeError("configured model unavailable")])
def test_chat_api_reports_unavailable_grounding_without_gemini(chat, monkeypatch, error):
    _, agent, _ = chat

    def unavailable():
        raise error

    monkeypatch.setattr(reasoning_service, "get_rag_engine", unavailable)
    api = importlib.import_module("backend.api.chat")
    monkeypatch.setattr(api, "ChatAgent", lambda: agent)
    app = FastAPI()
    app.include_router(api.router, prefix="/api")
    with TestClient(app) as client:
        session_id = client.post("/api/chat/session").json()["session_id"]
        response = client.post(f"/api/chat/{session_id}/message", data={"message": "NRXN1 evidence?"})
        assert response.status_code == 503
        assert "Reasoning runtime unavailable" in response.json()["detail"]


def test_grounded_chat_works_with_general_model_disabled(chat):
    _, agent, calls = chat
    agent.gemini_enabled = False
    assert run(agent, "NRXN1 evidence?").message.content == "observed table evidence"
    assert calls == ["NRXN1 evidence?"]


def test_umap_acronym_is_not_a_gene(chat, monkeypatch):
    _, agent, calls = chat
    from backend.models.schemas import ChatMessage, ChatResponse

    async def summarize(session, message):
        return ChatResponse(session_id=session.session_id,
                            message=ChatMessage(role="assistant", content="UMAP summary"))

    monkeypatch.setattr(agent, "_gemini_response", summarize)
    assert run(agent, "Summarize my UMAP").message.content == "UMAP summary"
    assert calls == []


@pytest.fixture(scope="module")
def table_engine():
    # Use the real intent dispatcher, annotation tables, and target validation.
    return reasoning_service.RAGEngine.from_disk()


@pytest.mark.parametrize("question,unsupported", [
    ("Are neurons present?", False), ("How do neurons communicate?", False),
    ("Hi, are neurons present?", False), ("What about B cells?", True),
    ("Hello, what about B cells?", True), ("What about invented cells?", True),
])
@pytest.mark.parametrize("with_context", [False, True])
def test_cell_identity_chat_uses_real_table_reasoning(
        chat, monkeypatch, table_engine, question, unsupported, with_context):
    _, agent, _ = chat
    monkeypatch.setattr(reasoning_service, "get_rag_engine", lambda: table_engine)
    session = ChatSession(session_id="cells", messages=[],
                          analysis_context={"total_cells": 42} if with_context else None)
    response = run(agent, question, session=session)
    assert response.message.traces
    assert any(trace["tool"] == "annotation_lookup" for trace in response.message.traces)
    if unsupported:
        assert response.message.content == "not in the tables"
    else:
        assert any(trace["filters"].get("ask_neurons") for trace in response.message.traces)
        assert "neuron" in response.message.content.lower()
