from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

from dotenv import load_dotenv
from pathlib import Path

# Load repo-root .env (capstone) then local, without overriding existing env.
_ROOT = Path(__file__).resolve().parents[4]
load_dotenv(_ROOT / ".env")
load_dotenv()

SYSTEM_PROMPT = """You are a computational-biology assistant. Answer only from the
tool traces and retrieved table rows. Cite tool name + file + pair/gene + n_cells.
Separate observation from interpretation. If neurons are not a labeled identity, say so.
Never invent a ligand-receptor pair or a DEG that is not in the traces.
Empty tool results are allowed — say the lookup was empty.
Keep the answer concise (under 200 words). Do not show hidden chain-of-thought.
"""


def llm_available() -> bool:
    return medgemma_available() or gemini_available()


def medgemma_available() -> bool:
    return bool(os.getenv("MEDGEMMA_ENDPOINT_ID") and os.getenv("VERTEX_PROJECT_ID"))


def gemini_available() -> bool:
    return bool(
        (os.getenv("VERTEX_PROJECT_ID") and os.getenv("GEMINI_MODEL"))
        or os.getenv("GEMINI_API_KEY")
    )


def _strip_medgemma_thought(text: str) -> str:
    """MedGemma-IT often emits <unused94>thought ... before the real answer."""
    raw = (text or "").strip()
    if not raw:
        return ""
    # Drop explicit think blocks.
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    # If a thought marker is present, keep only text after the last thought header
    # or after a clear Answer: label.
    if re.search(r"<unused\d+>\s*thought|^\s*thought\b", cleaned, flags=re.IGNORECASE | re.MULTILINE):
        answer_match = re.search(
            r"(?:^|\n)\s*(?:Final answer|Answer|Observation)\s*:\s*(.*)\Z",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if answer_match:
            cleaned = answer_match.group(1).strip()
        else:
            # Remove the thought preamble; keep trailing paragraphs if any.
            cleaned = re.sub(
                r"^.*?(?:<unused\d+>\s*)?thought\b",
                "",
                cleaned,
                count=1,
                flags=re.DOTALL | re.IGNORECASE,
            )
            # Heuristic: model often ends mid-thought when truncated; mark that.
            if "Identify the core" in cleaned or "**Scan the tool" in cleaned:
                cleaned = (
                    "[MedGemma returned chain-of-thought only / truncated. "
                    "Re-run with higher MEDGEMMA_MAX_TOKENS or use Gemini for this item.]\n"
                    + cleaned[-500:]
                )
    return cleaned.strip() or raw


def _invention_flags(text: str, context: str) -> list[str]:
    """Heuristic: uppercase gene-like tokens in the answer absent from traces."""
    if not text:
        return []
    ctx = context.upper()
    stop = {
        "EPHYS",
        "DEG",
        "DEGS",
        "IDH",
        "WT",
        "MUTANT",
        "CELLCHAT",
        "MHC",
        "TAM",
        "OPC",
        "MES",
        "LLM",
        "RAG",
        "CSV",
        "HTTP",
        "JSON",
    }
    suspects = []
    for token in set(re.findall(r"\b([A-Z][A-Z0-9]{1,}(?:-[A-Z0-9]+)?)\b", text)):
        if token in stop or token.startswith("EPHYS") or token.startswith("IDH"):
            continue
        if len(token) < 3:
            continue
        if token not in ctx:
            suspects.append(token)
    return suspects[:12]


@dataclass
class LLMCallResult:
    provider: str
    text: str
    raw: Any = None
    error: str | None = None


class MedGemmaClient:
    """Vertex dedicated-endpoint MedGemma via OpenAI-style chatCompletions rawPredict."""

    def __init__(self) -> None:
        self.project = os.getenv("VERTEX_PROJECT_ID", "").strip()
        self.location = os.getenv("VERTEX_LOCATION", "us-central1").strip()
        self.endpoint_id = os.getenv("MEDGEMMA_ENDPOINT_ID", "").strip()
        self.dedicated_dns = os.getenv("MEDGEMMA_DEDICATED_DNS", "").strip()
        self.max_tokens = int(os.getenv("MEDGEMMA_MAX_TOKENS", "1024"))
        self.temperature = float(os.getenv("MEDGEMMA_TEMPERATURE", "0.2"))
        self._creds = None

    def _credentials(self):
        if self._creds is None:
            from google.oauth2 import service_account
            from google.auth.transport.requests import Request

            path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
            if not path:
                raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS is not set")
            creds = service_account.Credentials.from_service_account_file(
                path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            creds.refresh(Request())
            self._creds = creds
        elif not self._creds.valid:
            from google.auth.transport.requests import Request

            self._creds.refresh(Request())
        return self._creds

    def _url(self) -> str:
        if not self.dedicated_dns:
            # Shared domain is rejected for dedicated endpoints; require DNS.
            raise RuntimeError(
                "MEDGEMMA_DEDICATED_DNS is required for dedicated MedGemma endpoints"
            )
        return (
            f"https://{self.dedicated_dns}/v1/projects/{self.project}/locations/"
            f"{self.location}/endpoints/{self.endpoint_id}:rawPredict"
        )

    def chat(self, question: str, context: str) -> LLMCallResult:
        payload = {
            "@requestFormat": "chatCompletions",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Tool traces and retrieved rows:\n{context}\n\n"
                        "Write the final answer only. Start with 'Observation:' "
                        "then 'Interpretation:'. No chain-of-thought."
                    ),
                },
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        creds = self._credentials()
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._url(),
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {creds.token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            return LLMCallResult(provider="medgemma", text="", raw=detail, error=f"HTTP {exc.code}: {detail[:500]}")
        except Exception as exc:  # noqa: BLE001
            return LLMCallResult(provider="medgemma", text="", error=str(exc))

        predictions = body.get("predictions", body)
        text = ""
        if isinstance(predictions, dict):
            choices = predictions.get("choices") or []
            if choices:
                message = choices[0].get("message") or {}
                text = str(message.get("content") or "")
        elif isinstance(predictions, list) and predictions:
            first = predictions[0]
            if isinstance(first, dict):
                text = str(first.get("content") or first.get("generated_text") or first)
            else:
                text = str(first)
        text = _strip_medgemma_thought(text)
        if not text:
            return LLMCallResult(
                provider="medgemma",
                text="",
                raw=body,
                error="Empty MedGemma completion",
            )
        return LLMCallResult(provider="medgemma", text=text, raw=body)


class GeminiClient:
    """Vertex AI Gemini (preferred) or API-key Gemini."""

    def __init__(self) -> None:
        self.project = os.getenv("VERTEX_PROJECT_ID", "").strip()
        self.location = os.getenv("VERTEX_LOCATION", "us-central1").strip()
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
        self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self._model = None
        self._mode: Optional[str] = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
        if self.project:
            import vertexai
            from vertexai.generative_models import GenerativeModel
            from google.oauth2 import service_account

            credentials = None
            if cred_path:
                credentials = service_account.Credentials.from_service_account_file(cred_path)
            vertexai.init(project=self.project, location=self.location, credentials=credentials)
            self._model = GenerativeModel(self.model_name)
            self._mode = "vertex"
            return
        if self.api_key:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model_name)
            self._mode = "api_key"
            return
        raise RuntimeError("Gemini is not configured (VERTEX_PROJECT_ID or GEMINI_API_KEY)")

    def chat(self, question: str, context: str) -> LLMCallResult:
        try:
            self._ensure()
            prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"Question: {question}\n\n"
                f"Tool traces and retrieved rows:\n{context}\n\n"
                "Answer from the traces only."
            )
            response = self._model.generate_content(prompt)
            text = getattr(response, "text", None) or str(response)
            return LLMCallResult(provider=f"gemini:{self._mode}", text=text.strip(), raw=response)
        except Exception as exc:  # noqa: BLE001
            return LLMCallResult(provider="gemini", text="", error=str(exc))


def generate_answer(question: str, context: str, *, provider: str | None = None) -> str:
    """Generate with one provider. Default prefers MedGemma, then Gemini."""
    choice = (provider or os.getenv("LLM_PROVIDER", "medgemma")).lower()
    if choice in {"medgemma", "mg"}:
        if not medgemma_available():
            raise RuntimeError("MedGemma is not configured (MEDGEMMA_ENDPOINT_ID / VERTEX_PROJECT_ID)")
        result = MedGemmaClient().chat(question, context)
    elif choice in {"gemini", "flash"}:
        if not gemini_available():
            raise RuntimeError("Gemini is not configured")
        result = GeminiClient().chat(question, context)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
    if result.error:
        raise RuntimeError(f"{result.provider} failed: {result.error}")
    return result.text


def generate_answers_dual(question: str, context: str) -> dict[str, LLMCallResult]:
    """Same traces → MedGemma and Gemini (Week 2 requirement)."""
    out: dict[str, LLMCallResult] = {}
    if medgemma_available():
        out["medgemma"] = MedGemmaClient().chat(question, context)
    else:
        out["medgemma"] = LLMCallResult(provider="medgemma", text="", error="not configured")
    if gemini_available():
        out["gemini"] = GeminiClient().chat(question, context)
    else:
        out["gemini"] = LLMCallResult(provider="gemini", text="", error="not configured")
    return out


def extractive_answer(question: str, context: str) -> str:
    return (
        "LLM generation is off. Tool traces and retrieved rows for "
        f"the question: {question}\n\n{context}"
    )
