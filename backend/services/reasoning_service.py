"""Grounded portal answers with a context-versioned, atomic provider cache."""
from __future__ import annotations

from functools import lru_cache
from dataclasses import dataclass
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


def _target_name(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


@dataclass(frozen=True)
class TargetIntent:
    pair: tuple[str, str] | None
    pathways: tuple[str, ...]
    unsupported: tuple[str, ...]


def extract_target_intent(question: str, engine: RAGEngine | None = None) -> TargetIntent:
    """Resolve bounded named phrases against table vocabulary and cell aliases.

    Unknown targets stay explicit rather than disappearing into broad dispatch.
    Quantifiers and ranking modifiers do not turn broad pathway requests into
    requests for a pathway literally named 'top' or 'every'.
    """
    slots = infer_query_slots(question)
    interactions = getattr(engine, "interactions", [])
    if not isinstance(interactions, (list, tuple)):
        interactions = []
    pathways = {_target_name(row.pathway): row.pathway for row in interactions}
    if slots["pathway"]:
        pathways.setdefault(_target_name(slots["pathway"]), slots["pathway"])
    cells = {_target_name(value) for row in interactions
             for value in (row.source_cell, row.target_cell,
                           row.source_compartment, row.target_compartment)}
    # Aliases describe the compartments actually represented by the tools.
    aliases = {"tumor": {"tumor", "tumour", "glioma", "tumor cell", "tumor cells"},
               "tcell": {"t", "t cell", "t cells", "tcell", "tcells"},
               "myeloid": {"tam", "tams", "myeloid", "microglia", "macrophage", "macrophages"}}
    for compartment, names in aliases.items():
        if compartment in cells:
            cells.update(names)
    # Keep exact table identities usable with common singular/plural wording.
    cells.update(name.removesuffix("s") for name in tuple(cells))

    broad = {"top", "every", "all", "which", "what", "how", "any", "each", "some", "many",
             "the", "a", "an", "this", "that", "these", "those", "same", "other",
             "signaling", "signalling", "exclusive", "dominant", "highest", "lowest",
             "ranked", "major", "immune", "neuronal", "sender", "senders", "sent"}
    boundaries = {"for", "of", "in", "on", "among", "within", "from", "with", "to",
                  "is", "are", "does", "do", "has", "have", "the", "a", "an", "and",
                  "list", "show", "describe", "name", "identify", "compare"}
    words = re.findall(r"[\w-]+", question.casefold())

    def preceding(index: int) -> list[str]:
        phrase = words[max(0, index - 4):index]
        cuts = [i for i, word in enumerate(phrase) if word in boundaries]
        return phrase[cuts[-1] + 1:] if cuts else phrase

    requested, unsupported = set(), set()
    for index, word in enumerate(words):
        if word in {"pathway", "pathways"}:
            phrase = preceding(index)
            if words[index + 1:index + 2] in (["called"], ["named"]):
                phrase = []
                for token in words[index + 2:index + 6]:
                    if token in boundaries:
                        break
                    phrase.append(token)
            name = _target_name(" ".join(phrase))
            if not name:
                continue
            if name in pathways:
                requested.add(pathways[name])
                continue
            # 'EGF signaling pathway' retains EGF; 'nonexistent signaling'
            # retains the complete unknown name instead of dropping modifiers.
            remainder = " ".join(token for token in name.split() if token not in broad)
            if remainder in pathways:
                requested.add(pathways[remainder])
            elif remainder and remainder not in cells:
                unsupported.add(name)
        if re.fullmatch(r"[a-z-]+(?:cytes?|blasts?|phages?)", word):
            name = _target_name(word)
            if name not in cells and name.removesuffix("s") not in cells:
                unsupported.add(name)
        if word in {"cell", "cells"}:
            phrase = preceding(index)
            name = _target_name(" ".join(phrase))
            remainder = " ".join(token for token in name.split() if token not in broad)
            if (remainder and not any(token.isdigit() for token in phrase)
                    and remainder not in cells and f"{remainder} cell" not in cells):
                unsupported.add(f"{remainder} cells")
    return TargetIntent(slots["pair"], tuple(sorted(requested)), tuple(sorted(unsupported)))


def _question_targets_supported(question: str, result: dict,
                                engine: RAGEngine | None = None) -> bool:
    intent = extract_target_intent(question, engine)
    if intent.unsupported:
        return False
    if intent.pair:
        ligand, receptor = intent.pair
        exact = [trace for trace in result["traces"]
                 if trace.filters.get("ligand") == ligand
                 and trace.filters.get("receptor") == receptor]
        # Ligand-only CellChat evidence cannot establish the requested pair.
        if not any(trace.rows for trace in exact):
            return False

    for name in intent.pathways:
        matching = [trace for trace in result["traces"]
                    if _target_name(str(trace.filters.get("pathway", ""))) == _target_name(name)]
        if not any(trace.rows for trace in matching):
            return False
    return True


def _has_evidence(result: dict, question: str = "", engine: RAGEngine | None = None) -> bool:
    if question and not _question_targets_supported(question, result, engine):
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
        if not _has_evidence(grounded, question, engine):
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
        if not _has_evidence(grounded, question, engine):
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
