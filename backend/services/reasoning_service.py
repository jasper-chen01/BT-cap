"""Grounded portal answers with a context-versioned, atomic provider cache."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from backend.config import settings
from ephys_rag.chain import RAGEngine
from ephys_rag.llm import generate_answer
from ephys_rag.nomination import NominationStore
from ephys_rag.schema import ToolResult
from ephys_rag.tools.registry import infer_query_slots


@lru_cache(maxsize=1)
def get_nomination_store() -> NominationStore:
    return NominationStore.from_config()


@lru_cache(maxsize=1)
def get_rag_engine() -> RAGEngine:
    # Validate the complete nomination runtime before exposing an engine.
    store = get_nomination_store()
    engine = RAGEngine.from_disk()
    engine.tools.nomination_store = store
    return engine


def serialize_tool_result(result: ToolResult) -> dict:
    return {"tool": result.tool, "source_file": result.source_file,
            "n_hits": result.n_hits, "filters": result.filters,
            "rows": result.rows, "note": result.note}


class PortalAnswerCache:
    schema_version = 1

    def __init__(self, directory: Path | None = None):
        self.directory = Path(directory if directory is not None else settings.EPHYS_RAG_RUNS_DIR)

    def make_key(self, question: str, model: str, context: str) -> str:
        payload = {"schema_version": self.schema_version,
                   "question": " ".join(question.split()).casefold(),
                   "model": model, "context": context}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", key):
            raise ValueError("Invalid cache key")
        return self.directory / f"{key}.json"

    def get(self, key: str) -> dict | None:
        try:
            record = json.loads(self._path(key).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if (not isinstance(record, dict) or record.get("schema_version") != self.schema_version
                or record.get("key") != key or not isinstance(record.get("answer"), str)
                or not record["answer"].strip()):
            return None
        return record

    def set(self, key: str, answer: str, *, provenance: dict | None = None) -> None:
        destination = self._path(key)
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Cannot cache an empty answer")
        record = {"schema_version": self.schema_version, "key": key, "answer": answer}
        if provenance is not None:
            record["provenance"] = provenance
        encoded = json.dumps(record, ensure_ascii=False, allow_nan=False)
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.directory,
                                             prefix=f"{key}.", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def _question_targets_supported(question: str, result: dict) -> bool:
    """Keep explicit targets that the permissive dispatcher may broaden away."""
    slots = infer_query_slots(question)
    if slots["pair"]:
        ligand, receptor = slots["pair"]
        exact = [trace for trace in result["traces"]
                 if trace.filters.get("ligand") == ligand
                 and trace.filters.get("receptor") == receptor]
        # Ligand-only CellChat evidence cannot establish the requested pair.
        if not any(trace.rows for trace in exact):
            return False

    named_pathway = re.search(
        r"\b(?:([\w-]+)\s+pathway|pathway\s+(?:called|named)\s+([\w-]+))\b",
        question, re.IGNORECASE)
    if named_pathway:
        name = next(value for value in named_pathway.groups() if value).casefold()
        descriptive = {"signaling", "signalling", "exclusive", "dominant", "same", "other",
                       "the", "a", "an", "which", "what", "this", "that", "any", "each",
                       "tumor", "tumour", "myeloid", "t-cell", "immune", "neuronal"}
        if name not in descriptive and not name.endswith(("-sender", "-sent")):
            matching = [trace for trace in result["traces"]
                        if str(trace.filters.get("pathway", "")).casefold() == name]
            if not any(trace.rows for trace in matching):
                return False

    # Unknown cell identities otherwise become an unfiltered CellChat lookup.
    # Resolve simple explicit location phrases against structured cell fields,
    # never against generic document titles or other free text.
    location = re.search(
        r"\b(?:in|among|within)\s+(?:the\s+)?([\w-]+(?:\s+[\w-]+){0,3}?)(?:\s+cells?)?\s*[?.!]*$",
        question, re.IGNORECASE)
    if location and not slots["celltype"] and not slots["compartment"]:
        cell = location.group(1).casefold()
        if cell not in {"dataset", "data", "table", "tables", "export", "neurons",
                        "this dataset", "this data", "this table", "these tables", "this export"}:
            fields = {"celltype", "celltype_id", "source_cell", "target_cell", "cell", "group",
                      "source", "target", "source_label", "target_label"}
            rows = [row for trace in result["traces"] for row in trace.rows]
            rows.extend(hit.document.metadata for hit in result["hits"])
            normalize = lambda value: re.sub(r"[^a-z0-9]", "", str(value).casefold())
            if not any(normalize(cell) in normalize(value) for row in rows
                       for key, value in row.items() if key in fields):
                return False
    return True


def _has_evidence(result: dict, question: str = "") -> bool:
    if question and not _question_targets_supported(question, result):
        return False
    # Dispatched filters identify the requested biological target. Prefer the
    # most specific lookup over broad dataset summaries and lexical retrieval:
    # matching "tumor" is not evidence for an absent named gene or pair.
    for fields in (("gene", "ligand", "receptor", "pathway"),
                   ("celltype", "sender", "receiver", "idh", "ephys")):
        targeted = [trace for trace in result["traces"]
                    if any(trace.filters.get(field) not in (None, "", []) for field in fields)]
        if targeted:
            return any(trace.rows for trace in targeted)
    return bool(result["hits"] or any(trace.rows for trace in result["traces"]))


class ReasoningService:
    def __init__(self, engine: RAGEngine | None = None, cache: PortalAnswerCache | None = None):
        self.engine = engine
        self.cache = cache if cache is not None else PortalAnswerCache()

    def ask(self, question: str, model: str = "none") -> dict:
        if model not in {"none", "medgemma", "gemini"}:
            raise ValueError("model must be none, medgemma, or gemini")
        engine = self.engine if self.engine is not None else get_rag_engine()
        grounded = engine.ask(question, use_llm=False)
        response = {
            "answer": grounded["answer"], "model": model, "cached": False,
            "traces": [serialize_tool_result(trace) for trace in grounded["traces"]],
            "hits": [{"title": hit.document.title, "score": float(hit.score),
                      "kind": hit.document.kind,
                      "source_file": hit.document.metadata.get("source_file", "")}
                     for hit in grounded["hits"]],
        }
        if not _has_evidence(grounded, question):
            response["answer"] = "not in the tables"
        elif model != "none":
            context = grounded["context"]
            key = self.cache.make_key(question, model, context)
            saved = self.cache.get(key)
            if saved is not None:
                response.update(answer=saved["answer"], cached=True)
            else:
                answer = generate_answer(question, context, provider=model)
                self.cache.set(key, answer)
                response["answer"] = answer
        return response


def import_week2_cache(source_dir: Path, cache: PortalAnswerCache, engine: RAGEngine) -> int:
    """Convert successful saved Q9–Q32 MedGemma runs without provider calls.

    Keys use today's grounded context; provenance retains the original context
    digest and run path so this historical-answer conversion remains auditable.
    Existing entries are untouched, making repeated imports idempotent.
    """
    imported = 0
    for question_id in range(9, 33):
        source = Path(source_dir) / f"Q{question_id}.json"
        if not source.is_file():
            continue
        saved = json.loads(source.read_text(encoding="utf-8"))
        if saved.get("question_id") != question_id:
            raise ValueError(f"Question ID does not match {source}")
        question, answer = saved.get("question"), saved.get("medgemma")
        if (saved.get("medgemma_error") or not isinstance(question, str) or not question.strip()
                or not isinstance(answer, str) or not answer.strip()):
            continue
        grounded = engine.ask(question, use_llm=False)
        if not _has_evidence(grounded, question):
            continue
        key = cache.make_key(question, "medgemma", grounded["context"])
        if cache.get(key) is not None:
            continue
        old_context = saved.get("context", "")
        provenance = {"source_file": str(source), "question_id": question_id,
                      "provider": "medgemma", "imported_from": "week2_traces",
                      "original_context_sha256": hashlib.sha256(old_context.encode("utf-8")).hexdigest(),
                      "current_context_sha256": hashlib.sha256(grounded["context"].encode("utf-8")).hexdigest()}
        cache.set(key, answer, provenance=provenance)
        imported += 1
    return imported
