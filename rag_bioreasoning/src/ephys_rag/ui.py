from __future__ import annotations

from typing import Any, Mapping


PROVIDER_CHOICES = ("none", "medgemma", "gemini", "auto")


def format_provider_status(status: Mapping[str, Any]) -> str:
    medgemma_ready = bool(status.get("medgemma", {}).get("configured"))
    gemini_ready = bool(status.get("gemini", {}).get("configured"))
    selected = str(status.get("selected", "none"))
    return (
        f"MedGemma: {'configured' if medgemma_ready else 'not configured'} · "
        f"Gemini: {'configured' if gemini_ready else 'not configured'} · "
        f"selected: {selected}"
    )


def format_result_caption(result: Mapping[str, Any]) -> str:
    provider = str(result.get("provider", "unknown"))
    model = str(result.get("model", "unknown"))
    elapsed_ms = float(result.get("elapsed_ms", 0.0))
    return f"Provider: {provider} · Model: {model} · {elapsed_ms:.1f} ms"
