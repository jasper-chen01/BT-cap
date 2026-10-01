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


class OllamaProvider:
    """Grounded MedGemma generation through Ollama's OpenAI-compatible API."""

    name = "ollama"

    def __init__(self, settings: ProviderSettings, *, client: Any | None = None) -> None:
        self.settings = settings
        self.model = settings.ollama_model
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ProviderConfigurationError(
                    "The Ollama provider requires the local extra. Install with: "
                    "pip install -e '.[local]'"
                ) from exc
            self._client = OpenAI(
                base_url=self.settings.ollama_base_url,
                api_key="ollama",
            )
        return self._client

    def generate(self, question: str, context: str) -> GenerationResult:
        started = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_grounded_prompt(question, context),
                    },
                ],
                max_tokens=self.settings.ollama_max_output_tokens,
                temperature=self.settings.ollama_temperature,
            )
            answer = response.choices[0].message.content
        except (ProviderConfigurationError, ProviderRequestError):
            raise
        except Exception as exc:
            raise ProviderRequestError(
                "Local MedGemma request failed. Confirm Ollama is running and the "
                f"model '{self.model}' is installed."
            ) from exc
        if not isinstance(answer, str) or not answer.strip():
            raise ProviderRequestError("Local MedGemma returned an empty answer.")
        elapsed_ms = (time.perf_counter() - started) * 1000
        return GenerationResult(
            text=answer.strip(),
            provider=self.name,
            model=self.model,
            elapsed_ms=round(elapsed_ms, 3),
            metadata={"base_url": self.settings.ollama_base_url},
        )
