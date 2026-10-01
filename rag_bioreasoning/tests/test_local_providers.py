from __future__ import annotations

from types import SimpleNamespace

import pytest

from ephys_rag.providers.factory import ProviderSettings, build_provider, provider_status


def local_provider_types():
    try:
        from ephys_rag.providers.mock import MockTraceProvider
        from ephys_rag.providers.ollama import OllamaProvider
    except ModuleNotFoundError:
        pytest.fail("local Ollama and mock providers are not implemented")
    return OllamaProvider, MockTraceProvider


class RecordingCompletions:
    def __init__(self) -> None:
        self.kwargs: dict = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        message = SimpleNamespace(content="NRXN1 binds NLGN1 in the supplied trace.")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class RecordingClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=RecordingCompletions())


def local_settings(**overrides) -> ProviderSettings:
    values = {
        "provider": "ollama",
        "ollama_base_url": "http://localhost:11434/v1",
        "ollama_model": "medgemma:4b-it-q4_K_M",
        "ollama_max_output_tokens": 321,
        "ollama_temperature": 0.2,
    }
    values.update(overrides)
    return ProviderSettings(**values)


def test_ollama_provider_sends_grounded_openai_chat_request():
    """Changing the local request to omit the evidence or pinned settings is a bug."""
    client = RecordingClient()
    OllamaProvider, _ = local_provider_types()
    provider = OllamaProvider(local_settings(), client=client)

    result = provider.generate("What does NRXN bind?", "pair=NRXN1_NLGN1")

    request = client.chat.completions.kwargs
    assert request["model"] == "medgemma:4b-it-q4_K_M"
    assert request["max_tokens"] == 321
    assert request["temperature"] == 0.2
    assert "pair=NRXN1_NLGN1" in request["messages"][1]["content"]
    assert result.text == "NRXN1 binds NLGN1 in the supplied trace."
    assert result.provider == "ollama"


def test_factory_builds_local_and_mock_providers_without_cloud_configuration():
    """Requiring Vertex settings for local providers would break the free workflow."""
    client = RecordingClient()
    OllamaProvider, MockTraceProvider = local_provider_types()

    local = build_provider("ollama", settings=local_settings(), ollama_client=client)
    mock = build_provider("mock", settings=local_settings(provider="mock"))
    status = provider_status(local_settings())

    assert isinstance(local, OllamaProvider)
    assert isinstance(mock, MockTraceProvider)
    assert status["ollama"]["configured"] is True


def test_mock_provider_returns_trace_facts_without_model_call():
    """Dropping trace rows from the mock answer would hide pipeline evidence."""
    context = """=== tool traces ===
[1] Tool: deg_lookup
Source: deg.csv
Rows:
- gene=NRXN1, direction=Ephys_2_high, celltype_id=cycling_tumor
"""

    _, MockTraceProvider = local_provider_types()
    result = MockTraceProvider().generate("Is NRXN1 supported?", context)

    assert result.provider == "mock"
    assert "deg_lookup" in result.text
    assert "NRXN1" in result.text
    assert "Ephys_2_high" in result.text
