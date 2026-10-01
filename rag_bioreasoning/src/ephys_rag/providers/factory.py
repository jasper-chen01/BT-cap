from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from ephys_rag.providers.base import (
    ExtractiveProvider,
    LLMProvider,
    ProviderConfigurationError,
)

SUPPORTED_PROVIDERS = ("auto", "none", "medgemma", "gemini", "ollama", "mock")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ProviderConfigurationError(
        f"{name} must be one of true/false, yes/no, on/off, or 1/0."
    )


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ProviderConfigurationError(f"{name} must be an integer.") from exc
    if parsed <= 0:
        raise ProviderConfigurationError(f"{name} must be greater than zero.")
    return parsed


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ProviderConfigurationError(f"{name} must be a number.") from exc


@dataclass(frozen=True)
class ProviderSettings:
    provider: str = "auto"
    project: str = ""
    location: str = "us-central1"
    medgemma_endpoint_id: str = ""
    medgemma_endpoint_location: str = ""
    medgemma_use_dedicated_endpoint: bool = True
    medgemma_model_name: str = "medgemma-instruction-tuned"
    medgemma_max_output_tokens: int = 1200
    medgemma_api_mode: str = "predict"
    gemini_location: str = "global"
    gemini_model: str = "gemini-3.5-flash"
    gemini_max_output_tokens: int = 1200
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "medgemma:4b-it-q4_K_M"
    ollama_max_output_tokens: int = 600
    ollama_temperature: float = 0.2

    @classmethod
    def from_env(cls) -> "ProviderSettings":
        location = (
            os.getenv("GOOGLE_CLOUD_LOCATION", "").strip()
            or os.getenv("VERTEX_LOCATION", "").strip()
            or "us-central1"
        )
        project = (
            os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
            or os.getenv("VERTEX_PROJECT_ID", "").strip()
        )
        return cls(
            provider=os.getenv("LLM_PROVIDER", "auto").strip().lower() or "auto",
            project=project,
            location=location,
            medgemma_endpoint_id=os.getenv("MEDGEMMA_ENDPOINT_ID", "").strip(),
            medgemma_endpoint_location=os.getenv(
                "MEDGEMMA_ENDPOINT_LOCATION", ""
            ).strip(),
            medgemma_use_dedicated_endpoint=_env_bool(
                "MEDGEMMA_USE_DEDICATED_ENDPOINT", True
            ),
            medgemma_model_name=os.getenv(
                "MEDGEMMA_MODEL_NAME", "medgemma-instruction-tuned"
            ).strip()
            or "medgemma-instruction-tuned",
            medgemma_max_output_tokens=_env_int(
                "MEDGEMMA_MAX_OUTPUT_TOKENS", 1200
            ),
            medgemma_api_mode=_medgemma_api_mode(),
            gemini_location=os.getenv("GEMINI_LOCATION", "global").strip()
            or "global",
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()
            or "gemini-3.5-flash",
            gemini_max_output_tokens=_env_int("GEMINI_MAX_OUTPUT_TOKENS", 1200),
            ollama_base_url=os.getenv(
                "OLLAMA_BASE_URL", "http://localhost:11434/v1"
            ).strip()
            or "http://localhost:11434/v1",
            ollama_model=os.getenv(
                "OLLAMA_MODEL", "medgemma:4b-it-q4_K_M"
            ).strip()
            or "medgemma:4b-it-q4_K_M",
            ollama_max_output_tokens=_env_int("OLLAMA_MAX_OUTPUT_TOKENS", 600),
            ollama_temperature=_env_float("OLLAMA_TEMPERATURE", 0.2),
        )

    @property
    def endpoint_location(self) -> str:
        return self.medgemma_endpoint_location or self.location


def _medgemma_api_mode() -> str:
    value = os.getenv("MEDGEMMA_API_MODE", "predict").strip().lower() or "predict"
    if value not in {"predict", "raw"}:
        raise ProviderConfigurationError(
            "MEDGEMMA_API_MODE must be either 'predict' or 'raw'."
        )
    return value


def _medgemma_missing(settings: ProviderSettings) -> list[str]:
    missing: list[str] = []
    if not settings.project:
        missing.append("GOOGLE_CLOUD_PROJECT")
    if not settings.endpoint_location:
        missing.append("MEDGEMMA_ENDPOINT_LOCATION")
    if not settings.medgemma_endpoint_id:
        missing.append("MEDGEMMA_ENDPOINT_ID")
    return missing


def _gemini_missing(settings: ProviderSettings) -> list[str]:
    missing: list[str] = []
    if not settings.project:
        missing.append("GOOGLE_CLOUD_PROJECT")
    if not settings.gemini_location:
        missing.append("GEMINI_LOCATION")
    return missing


def _selected_name(settings: ProviderSettings) -> str:
    requested = settings.provider.lower()
    if requested not in SUPPORTED_PROVIDERS:
        raise ProviderConfigurationError(
            f"Unsupported provider '{settings.provider}'. Choose from: "
            + ", ".join(SUPPORTED_PROVIDERS)
            + "."
        )
    if requested != "auto":
        return requested
    if not _medgemma_missing(settings):
        return "medgemma"
    if not _gemini_missing(settings):
        return "gemini"
    return "none"


def provider_status(settings: ProviderSettings | None = None) -> dict[str, Any]:
    settings = settings or ProviderSettings.from_env()
    selected = _selected_name(settings)
    return {
        "selected": selected,
        "medgemma": {"configured": not _medgemma_missing(settings)},
        "gemini": {"configured": not _gemini_missing(settings)},
        "ollama": {
            "configured": bool(settings.ollama_base_url and settings.ollama_model)
        },
        "mock": {"configured": True},
        "none": {"configured": True},
    }


def _require_configured(name: str, missing: list[str]) -> None:
    if missing:
        raise ProviderConfigurationError(
            f"Provider '{name}' is not configured. Set: {', '.join(missing)}."
        )


def build_provider(
    name: str | None = None,
    *,
    settings: ProviderSettings | None = None,
    medgemma_endpoint: Any | None = None,
    gemini_client: Any | None = None,
    ollama_client: Any | None = None,
) -> LLMProvider:
    settings = settings or ProviderSettings.from_env()
    requested = (name or settings.provider).strip().lower()
    if requested not in SUPPORTED_PROVIDERS:
        raise ProviderConfigurationError(
            f"Unsupported provider '{requested}'. Choose from: "
            + ", ".join(SUPPORTED_PROVIDERS)
            + "."
        )
    selected = _selected_name(
        ProviderSettings(**{**settings.__dict__, "provider": requested})
    )
    if selected == "none":
        return ExtractiveProvider()
    if selected == "medgemma":
        _require_configured(selected, _medgemma_missing(settings))
        from ephys_rag.providers.medgemma import MedGemmaProvider

        return MedGemmaProvider(settings, endpoint=medgemma_endpoint)
    if selected == "ollama":
        from ephys_rag.providers.ollama import OllamaProvider

        return OllamaProvider(settings, client=ollama_client)
    if selected == "mock":
        from ephys_rag.providers.mock import MockTraceProvider

        return MockTraceProvider()
    _require_configured(selected, _gemini_missing(settings))
    from ephys_rag.providers.gemini import GeminiProvider

    return GeminiProvider(settings, client=gemini_client)
