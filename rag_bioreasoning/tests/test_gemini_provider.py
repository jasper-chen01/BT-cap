from types import SimpleNamespace

import pytest

from ephys_rag.providers.base import ProviderRequestError
from ephys_rag.providers.factory import ProviderSettings
from ephys_rag.providers.gemini import GeminiProvider


class FakeModels:
    """Imitates the external Gen AI models boundary."""

    def __init__(self, text="Grounded answer", error=None):
        self.text = text
        self.error = error
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(text=self.text)


def gemini_settings(**overrides):
    values = {
        "provider": "gemini",
        "project": "elec-594-bt-cap",
        "location": "us-central1",
        "gemini_model": "gemini-3.5-flash",
        "gemini_max_output_tokens": 900,
    }
    values.update(overrides)
    return ProviderSettings(**values)


def test_gemini_uses_vertex_model_and_grounded_context():
    """Changing the selected model or losing the evidence boundary must fail."""
    models = FakeModels()
    provider = GeminiProvider(
        gemini_settings(),
        client=SimpleNamespace(models=models),
    )

    result = provider.generate("What does NRXN bind?", "NRXN1_NLGN1 evidence")

    call = models.calls[0]
    assert call["model"] == "gemini-3.5-flash"
    assert "UNTRUSTED RETRIEVED EVIDENCE" in call["contents"]
    assert "temperature" not in call["config"]
    assert call["config"]["thinking_config"] == {"thinking_level": "MINIMAL"}
    assert call["config"]["max_output_tokens"] == 900
    assert "source file" in call["config"]["system_instruction"]
    assert result.text == "Grounded answer"
    assert result.provider == "gemini"
    assert result.model == "gemini-3.5-flash"


def test_gemini_rejects_empty_text():
    """An empty SDK response must be reported instead of accepted as an answer."""
    provider = GeminiProvider(
        gemini_settings(),
        client=SimpleNamespace(models=FakeModels(text="  ")),
    )

    with pytest.raises(ProviderRequestError, match="empty"):
        provider.generate("question", "evidence")


def test_gemini_request_failure_preserves_cause():
    """Authentication or transport failures must remain visible and attributable."""
    provider = GeminiProvider(
        gemini_settings(),
        client=SimpleNamespace(models=FakeModels(error=PermissionError("denied"))),
    )

    with pytest.raises(ProviderRequestError, match="Gemini") as exc_info:
        provider.generate("question", "evidence")

    assert isinstance(exc_info.value.__cause__, PermissionError)
