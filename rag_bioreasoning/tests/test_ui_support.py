from pathlib import Path

from ephys_rag.ui import PROVIDER_CHOICES, format_provider_status, format_result_caption


ROOT = Path(__file__).resolve().parents[1]


def test_provider_controls_have_all_supported_choices_and_safe_status():
    status = {
        "selected": "medgemma",
        "medgemma": {"configured": True},
        "gemini": {"configured": False},
        "none": {"configured": True},
    }

    assert PROVIDER_CHOICES == ("none", "medgemma", "gemini", "auto")
    rendered = format_provider_status(status)
    assert "MedGemma: configured" in rendered
    assert "Gemini: not configured" in rendered
    assert "selected: medgemma" in rendered
    assert "endpoint" not in rendered.lower()


def test_result_caption_identifies_actual_provider_model_and_latency():
    caption = format_result_caption(
        {"provider": "medgemma", "model": "medgemma-it", "elapsed_ms": 12.345}
    )

    assert caption == "Provider: medgemma · Model: medgemma-it · 12.3 ms"


def test_env_example_lists_runtime_names_without_credentials():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")

    assert "GOOGLE_CLOUD_PROJECT=elec-594-bt-cap" in text
    assert "MEDGEMMA_ENDPOINT_ID=" in text
    assert "GOOGLE_APPLICATION_CREDENTIALS=" in text
    assert "GEMINI_MODEL=gemini-3.5-flash" in text
    assert "Bearer " not in text
    assert "PRIVATE KEY" not in text
