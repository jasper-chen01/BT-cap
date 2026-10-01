import json
from types import SimpleNamespace

import pytest

from ephys_rag.providers.base import ProviderRequestError
from ephys_rag.providers.factory import ProviderSettings
from ephys_rag.providers.medgemma import MedGemmaProvider, parse_medgemma_prediction


class FakeEndpoint:
    """Imitates the documented Vertex Endpoint.predict boundary only."""

    def __init__(self, predictions):
        self.predictions = predictions
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(predictions=self.predictions)


class FakeRawResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeRawEndpoint:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def raw_predict(self, **kwargs):
        self.calls.append(kwargs)
        return FakeRawResponse(self.payload)


def medgemma_settings(**overrides):
    values = {
        "provider": "medgemma",
        "project": "elec-594-bt-cap",
        "location": "us-central1",
        "medgemma_endpoint_id": "123",
        "medgemma_use_dedicated_endpoint": True,
        "medgemma_max_output_tokens": 900,
    }
    values.update(overrides)
    return ProviderSettings(**values)


def test_medgemma_builds_documented_chat_completions_request():
    """Changing the request format, generation limits, or endpoint mode must fail."""
    endpoint = FakeEndpoint(
        {"choices": [{"message": {"content": "NRXN binds NLGN1."}}]}
    )
    provider = MedGemmaProvider(medgemma_settings(), endpoint=endpoint)

    result = provider.generate("What does NRXN bind?", "NRXN1_NLGN1 evidence")

    call = endpoint.calls[0]
    instance = call["instances"][0]
    assert instance["@requestFormat"] == "chatCompletions"
    assert instance["temperature"] == 0
    assert instance["max_tokens"] == 900
    assert call["use_dedicated_endpoint"] is True
    assert instance["messages"][0]["role"] == "system"
    assert "UNTRUSTED RETRIEVED EVIDENCE" in instance["messages"][1]["content"]
    assert result.text == "NRXN binds NLGN1."
    assert result.provider == "medgemma"
    assert result.model == "medgemma-instruction-tuned"


def test_medgemma_raw_mode_sends_openai_chat_payload_without_instances_wrapper():
    """A custom vLLM route must receive the OpenAI body expected by rawPredict."""
    endpoint = FakeRawEndpoint(
        {"choices": [{"message": {"content": "Grounded raw answer."}}]}
    )
    provider = MedGemmaProvider(
        medgemma_settings(
            medgemma_api_mode="raw",
            medgemma_model_name="google/medgemma-4b-it",
            medgemma_use_dedicated_endpoint=False,
        ),
        endpoint=endpoint,
    )

    result = provider.generate("What does NRXN bind?", "NRXN1_NLGN1 evidence")

    call = endpoint.calls[0]
    payload = json.loads(call["body"])
    assert payload["model"] == "google/medgemma-4b-it"
    assert payload["messages"][0]["role"] == "system"
    assert "UNTRUSTED RETRIEVED EVIDENCE" in payload["messages"][1]["content"]
    assert "instances" not in payload
    assert call["headers"] == {"Content-Type": "application/json"}
    assert call["use_dedicated_endpoint"] is False
    assert result.text == "Grounded raw answer."
    assert result.metadata["api_mode"] == "raw"


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": [{"message": {"content": "answer"}}]},
        [{"choices": [{"message": {"content": "answer"}}]}],
    ],
)
def test_medgemma_parser_accepts_mapping_and_one_element_sequence(payload):
    """SDK serialization differences must not discard a valid answer."""
    assert parse_medgemma_prediction(payload) == "answer"


def test_medgemma_parser_removes_thinking_trace():
    """Internal reasoning tokens must never be shown as the final answer."""
    payload = {
        "choices": [
            {
                "message": {
                    "content": "<unused94>thought\nprivate reasoning<unused95>Grounded answer"
                }
            }
        ]
    }

    assert parse_medgemma_prediction(payload) == "Grounded answer"


@pytest.mark.parametrize(
    "payload",
    [
        {"predictions": []},
        [],
        {"choices": []},
        {"choices": [{"message": {"content": ""}}]},
    ],
)
def test_medgemma_parser_rejects_malformed_or_empty_payload(payload):
    """Unexpected responses must fail visibly rather than look like abstentions."""
    with pytest.raises(ProviderRequestError, match="response shape|empty"):
        parse_medgemma_prediction(payload)


def test_medgemma_request_failure_is_reported_without_fallback():
    """A denied MedGemma request must remain a MedGemma failure."""

    class BrokenEndpoint:
        def predict(self, **kwargs):
            raise PermissionError("denied")

    provider = MedGemmaProvider(medgemma_settings(), endpoint=BrokenEndpoint())

    with pytest.raises(ProviderRequestError, match="MedGemma") as exc_info:
        provider.generate("question", "evidence")

    assert isinstance(exc_info.value.__cause__, PermissionError)
