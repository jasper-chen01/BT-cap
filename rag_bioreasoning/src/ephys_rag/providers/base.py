from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class ProviderConfigurationError(RuntimeError):
    """Raised when a requested provider cannot be configured safely."""


class ProviderRequestError(RuntimeError):
    """Raised when a configured provider cannot return a usable answer."""


@dataclass(frozen=True)
class GenerationResult:
    text: str
    provider: str
    model: str
    elapsed_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMProvider(Protocol):
    name: str
    model: str

    def generate(self, question: str, context: str) -> GenerationResult:
        """Generate one answer from a question and bounded evidence context."""


class ExtractiveProvider:
    """Offline provider that exposes evidence without generative inference."""

    name = "none"
    model = "extractive"

    def generate(self, question: str, context: str) -> GenerationResult:
        text = (
            "LLM generation is off. Tool traces and retrieved rows for "
            f"the question: {question}\n\n{context}"
        )
        return GenerationResult(
            text=text,
            provider=self.name,
            model=self.model,
        )

