from __future__ import annotations

import time
from typing import Any

from ephys_rag.llm import SYSTEM_PROMPT, build_grounded_prompt
from ephys_rag.providers.base import (
    GenerationResult,
    ProviderConfigurationError,
    ProviderRequestError,
)
from ephys_rag.providers.factory import ProviderSettings


class GeminiProvider:
    """Grounded comparison generation through Gemini on Vertex AI."""

    name = "gemini"

    def __init__(
        self,
        settings: ProviderSettings,
        *,
        client: Any | None = None,
    ) -> None:
        self.settings = settings
        self.model = settings.gemini_model
        self._client = client

    def _create_client(self) -> Any:
        try:
            from google import genai
        except ImportError as exc:
            raise ProviderConfigurationError(
                "Gemini on Vertex requires the cloud extra. Install with: "
                "pip install -e '.[cloud]'"
            ) from exc
        try:
            return genai.Client(
                vertexai=True,
                project=self.settings.project,
                location=self.settings.gemini_location,
            )
        except Exception as exc:
            raise ProviderConfigurationError(
                "Could not initialize Gemini on Vertex. Check Application Default "
                "Credentials, project access, and region."
            ) from exc

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def generate(self, question: str, context: str) -> GenerationResult:
        started = time.perf_counter()
        try:
            response = self.client.models.generate_content(
                model=self.settings.gemini_model,
                contents=build_grounded_prompt(question, context),
                config={
                    "system_instruction": SYSTEM_PROMPT,
                    "thinking_config": {"thinking_level": "MINIMAL"},
                    "max_output_tokens": self.settings.gemini_max_output_tokens,
                },
            )
            answer = (getattr(response, "text", None) or "").strip()
        except (ProviderConfigurationError, ProviderRequestError):
            raise
        except Exception as exc:
            raise ProviderRequestError(
                "Gemini request failed. Check Vertex authentication, model access, "
                "region, quota, and network connectivity."
            ) from exc
        if not answer:
            raise ProviderRequestError("Gemini returned an empty answer.")
        elapsed_ms = (time.perf_counter() - started) * 1000
        return GenerationResult(
            text=answer,
            provider=self.name,
            model=self.model,
            elapsed_ms=round(elapsed_ms, 3),
        )
