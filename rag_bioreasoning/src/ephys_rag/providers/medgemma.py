from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from typing import Any

from ephys_rag.llm import SYSTEM_PROMPT, build_grounded_prompt
from ephys_rag.providers.base import (
    GenerationResult,
    ProviderConfigurationError,
    ProviderRequestError,
)
from ephys_rag.providers.factory import ProviderSettings


def _unwrap_prediction(predictions: Any) -> Mapping[str, Any]:
    if isinstance(predictions, Mapping):
        return predictions
    if (
        isinstance(predictions, Sequence)
        and not isinstance(predictions, (str, bytes))
        and len(predictions) == 1
        and isinstance(predictions[0], Mapping)
    ):
        return predictions[0]
    raise ProviderRequestError(
        "MedGemma returned an unsupported response shape; expected a choices mapping."
    )


def _strip_thinking_trace(text: str) -> str:
    """Keep only the final answer when a thinking-capable model emits trace tokens."""
    if "<unused95>" in text:
        text = text.split("<unused95>", 1)[1]
    return text.strip()


def parse_medgemma_prediction(predictions: Any) -> str:
    """Parse the documented Model Garden chat-completions response."""
    payload = _unwrap_prediction(predictions)
    try:
        choices = payload["choices"]
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
            raise TypeError("choices is not a sequence")
        content = choices[0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderRequestError(
            "MedGemma returned an unsupported response shape; "
            "expected choices[0].message.content."
        ) from exc
    if not isinstance(content, str):
        raise ProviderRequestError(
            "MedGemma returned an unsupported response shape; content was not text."
        )
    answer = _strip_thinking_trace(content)
    if not answer:
        raise ProviderRequestError("MedGemma returned an empty answer.")
    return answer


class MedGemmaProvider:
    """Grounded text generation through a deployed Vertex MedGemma endpoint."""

    name = "medgemma"

    def __init__(
        self,
        settings: ProviderSettings,
        *,
        endpoint: Any | None = None,
    ) -> None:
        self.settings = settings
        self.model = settings.medgemma_model_name
        self._endpoint = endpoint

    def _create_endpoint(self) -> Any:
        try:
            from google.cloud import aiplatform
        except ImportError as exc:
            raise ProviderConfigurationError(
                "MedGemma requires the cloud extra. Install with: "
                "pip install -e '.[cloud]'"
            ) from exc
        try:
            aiplatform.init(
                project=self.settings.project,
                location=self.settings.endpoint_location,
                api_transport="rest",
            )
            return aiplatform.Endpoint(
                endpoint_name=self.settings.medgemma_endpoint_id,
                location=self.settings.endpoint_location,
            )
        except Exception as exc:
            raise ProviderConfigurationError(
                "Could not initialize the MedGemma Vertex endpoint. Check "
                "Application Default Credentials, project access, endpoint ID, and region."
            ) from exc

    @property
    def endpoint(self) -> Any:
        if self._endpoint is None:
            self._endpoint = self._create_endpoint()
        return self._endpoint

    def generate(self, question: str, context: str) -> GenerationResult:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_grounded_prompt(question, context),
                },
            ],
            "max_tokens": self.settings.medgemma_max_output_tokens,
            "temperature": 0,
        }
        started = time.perf_counter()
        try:
            if self.settings.medgemma_api_mode == "raw":
                response = self.endpoint.raw_predict(
                    body=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    use_dedicated_endpoint=self.settings.medgemma_use_dedicated_endpoint,
                )
                response.raise_for_status()
                predictions = response.json()
            else:
                instance = {"@requestFormat": "chatCompletions", **payload}
                instance.pop("model", None)
                response = self.endpoint.predict(
                    instances=[instance],
                    use_dedicated_endpoint=self.settings.medgemma_use_dedicated_endpoint,
                )
                predictions = response.predictions
            answer = parse_medgemma_prediction(predictions)
        except (ProviderConfigurationError, ProviderRequestError):
            raise
        except Exception as exc:
            raise ProviderRequestError(
                "MedGemma request failed. Check Vertex authentication, endpoint "
                "permissions, deployment health, region, and quota."
            ) from exc
        elapsed_ms = (time.perf_counter() - started) * 1000
        return GenerationResult(
            text=answer,
            provider=self.name,
            model=self.model,
            elapsed_ms=round(elapsed_ms, 3),
            metadata={
                "dedicated_endpoint": self.settings.medgemma_use_dedicated_endpoint,
                "api_mode": self.settings.medgemma_api_mode,
            },
        )
