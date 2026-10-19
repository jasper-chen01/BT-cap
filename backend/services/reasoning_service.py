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


@lru_cache(maxsize=1)
def _router_cell_aliases() -> dict[str, str]:
    aliases = {}
    for alias in ("tumor", "tumour", "glioma", "T cell", "T cells", "T-cell", "T",
                  "TAM", "TAM/microglia", "microglia", "myeloid", "OPC-like", "OPC",
                  "MES-like", "cycling tumor"):
        parsed = infer_query_slots(alias)
        canonical = parsed["celltype"] or parsed["compartment"]
        if canonical:
            aliases[_target_name(alias)] = canonical
    aliases["t"] = aliases.get("t cell", "T cell")
    return aliases


@lru_cache(maxsize=2048)
def _gene_tokens(token: str) -> tuple[str, ...]:
    # These are context labels in the shared schema, never gene symbols.
    if re.fullmatch(r"IDH[-_](?:WT|MUTANT)", token):
        return ()
    return tuple(infer_query_slots(token)["genes"])


@dataclass(frozen=True)
class TargetIntent:
    pairs: tuple[tuple[str, str], ...]
    genes: tuple[str, ...]
    pathways: tuple[str, ...]
    cells: tuple[str, ...]
    unsupported: tuple[str, ...]


def extract_target_intent(question: str, engine: RAGEngine | None = None) -> TargetIntent:
    """Extract every explicit target, retaining unknown names for exact lookup."""
    slots = infer_query_slots(question)
    interactions = getattr(engine, "interactions", [])
    if not isinstance(interactions, (list, tuple)):
        interactions = []
    pathways = {_target_name(row.pathway): row.pathway for row in interactions}
    if slots["pathway"]:
        pathways.setdefault(_target_name(slots["pathway"]), slots["pathway"])
    cell_aliases = {_target_name(value): value for row in interactions
                    for value in (row.source_cell, row.target_cell)}
    if isinstance(engine, RAGEngine):
        cell_aliases.update({_target_name(row["celltype_id"]): row["celltype_id"]
                             for row in engine.tools.annotation_lookup().rows})
    # Resolve language aliases with the router's own normalization, including
    # OPC-like/MES-like rather than reproducing its biological mappings here.
    cell_aliases.update(_router_cell_aliases())
    for alias, canonical in tuple(cell_aliases.items()):
        if alias.endswith(" tumor"):
            cell_aliases.setdefault(alias.removesuffix(" tumor"), canonical)
        cell_aliases.setdefault(alias.removesuffix("s"), canonical)

    broad = {"top", "every", "all", "which", "what", "how", "any", "each", "some", "many",
             "the", "a", "an", "this", "that", "these", "those", "same", "other",
             "signaling", "signalling", "exclusive", "dominant", "highest", "lowest",
             "ranked", "major", "immune", "neuronal", "sender", "senders", "sent",
             "strongest", "weakest", "probability", "only", "labeled", "labelled"}
    boundaries = {"for", "of", "in", "on", "among", "within", "from", "with", "to",
                  "is", "are", "does", "do", "has", "have", "the", "a", "an", "and",
                  "list", "show", "describe", "name", "identify", "compare"}
    pathway_predicates = {"hit", "hits", "bind", "binds", "use", "uses", "occur", "occurs",
                          "target", "targets", "signal", "signals", "involve", "involves"}
    words = re.findall(r"[\w-]+", question.casefold())

    def preceding(index: int) -> list[str]:
        phrase = words[max(0, index - 4):index]
        cuts = [i for i, word in enumerate(phrase) if word in boundaries]
        return phrase[cuts[-1] + 1:] if cuts else phrase

    normalized = _target_name(question)
    requested = {canonical for name, canonical in pathways.items()
                 if re.search(rf"\b{re.escape(name)}\b", normalized)}
    requested_cells = {canonical for alias, canonical in cell_aliases.items()
                       if alias != "t" and re.search(rf"\b{re.escape(alias)}\b", normalized)}
    unsupported = set()
    # Enumerate all pairs and standalone genes; infer_query_slots itself keeps
    # only the first pair/four genes. Remove pair spans before standalone scan.
    pair_pattern = r"\b([A-Z][A-Z0-9-]+)(?:_|–|→|->)([A-Z][A-Z0-9-]+)\b"
    pairs = {match.groups() for match in re.finditer(pair_pattern, question)
             if match.group(1) not in {"IDH", "EPHYS"}}
    standalone = re.sub(pair_pattern, " ", question)
    # A table-backed pathway/family expression such as glutamate–GRIA is a
    # receptor-family target, not a standalone gene literally named GRIA.
    for match in re.finditer(r"\b([a-z][a-z-]*)[–-]([A-Z]+)\b", standalone):
        pathway, family = match.groups()
        canonical = pathways.get(_target_name(pathway))
        if canonical and any(row.pathway == canonical and row.receptor.startswith(family)
                             for row in interactions):
            standalone = standalone.replace(match.group(), pathway)
    known_genes = {
        member
        for row in interactions
        for value in (row.ligand, row.receptor)
        for member in str(value).upper().split("_")
        if member
    }
    if isinstance(engine, RAGEngine):
        registry = engine.tools
        known_genes.update(
            str(row.gene).upper()
            for rows in (registry.degs_idh, registry.degs_pooled)
            for row in rows
            if row.gene
        )
        store = registry.nomination_store
        if store is not None:
            for frame in [store.gene_ranks, *store.arm_scores.values()]:
                if not frame.empty and "gene" in frame:
                    known_genes.update(frame["gene"].dropna().astype(str).str.upper())
    gene_candidates = set(
        re.findall(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*\b", standalone)
    )
    # Gene symbols are case-insensitive at the API boundary. Resolve lowercase
    # names against the table-backed vocabulary without promoting ordinary
    # English words that happen to be valid symbols (for example WAS or SET).
    explicit_gene_stops = broad | boundaries | {
        "bad", "candidate", "candidates", "cat", "evidence", "expression",
        "kit", "not", "program", "programs", "rank", "ranks", "ranking",
        "rankings", "score", "scores", "set", "sets", "son", "was", "were",
    }
    left_gene_cue = re.compile(
        r"\b(?:about|candidate|for|gene|ligand|nominate|rank|receptor|score|supports?)\s*$",
        re.IGNORECASE,
    )
    right_gene_cue = re.compile(
        r"^\s*(?:binds?|candidate|evidence|gene|interaction|rank|score|support)\b",
        re.IGNORECASE,
    )
    word_pattern = r"\b[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*\b"
    for match in re.finditer(word_pattern, standalone):
        token = match.group()
        symbol = token.upper()
        if symbol not in known_genes or token.casefold() in explicit_gene_stops:
            continue
        has_symbol_shape = bool(re.search(r"[0-9-]", token))
        has_target_context = (
            left_gene_cue.search(standalone[:match.start()]) is not None
            or right_gene_cue.match(standalone[match.end():]) is not None
        )
        if has_symbol_shape or has_target_context:
            gene_candidates.add(symbol)
    # Retain explicitly named unknown symbols so fail-closed validation can
    # reject them instead of allowing a broad lexical hit to reach an LLM.
    gene_candidates.update(
        candidate
        for match in re.finditer(
            r"\bgenes?\s+(?:named\s+|called\s+)?"
            r"([A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*)\b",
            standalone,
            re.IGNORECASE,
        )
        for candidate in (match.group(1).upper(),)
        if candidate.casefold() not in explicit_gene_stops
    )
    genes = set()
    for token in gene_candidates:
        if _target_name(token) in pathways or _target_name(token) in broad:
            continue
        genes.update(_gene_tokens(token))
    for index, word in enumerate(words):
        if word in {"pathway", "pathways"}:
            phrase = preceding(index)
            following = words[index + 1:index + 2]
            if following and (following[0] in {"called", "named"}
                              or following[0] not in boundaries | broad | pathway_predicates):
                phrase = []
                start = index + (2 if following[0] in {"called", "named"} else 1)
                for token in words[start:start + 4]:
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
            remainder = " ".join(token for token in name.split()
                                 if token not in broad and not token.isdigit())
            if remainder in pathways:
                requested.add(pathways[remainder])
            elif remainder and remainder not in cell_aliases:
                # A phrase containing separately known pathway names is a
                # multi-target query (e.g. EGF and MK), not one invented name.
                known_words = {part for name in pathways for part in name.split()}
                if all(token in known_words or token == "and" for token in remainder.split()):
                    continue
                requested.add(name)
                unsupported.add(name)
        if re.fullmatch(r"[a-z-]+(?:cytes?|blasts?|phages?|glia)", word):
            name = _target_name(word)
            canonical = cell_aliases.get(name, cell_aliases.get(name.removesuffix("s")))
            requested_cells.add(canonical or name)
            if canonical is None:
                unsupported.add(name)
        if word in {"cell", "cells"}:
            phrase = preceding(index)
            name = _target_name(" ".join(phrase))
            remainder = " ".join(token for token in name.split() if token not in broad)
            # Pure counts are broad; numbers inside names ('type 2 endothelial')
            # remain part of the explicit cell identity.
            if remainder and not all(token.isdigit() for token in remainder.split()):
                canonical = cell_aliases.get(remainder)
                requested_cells.add(canonical or remainder)
                if canonical is None:
                    unsupported.add(f"{remainder} cells")
    return TargetIntent(tuple(sorted(pairs)), tuple(sorted(genes)), tuple(sorted(requested)),
                        tuple(sorted(requested_cells)), tuple(sorted(unsupported)))


def _normalize_routing_question(question: str, intent: TargetIntent) -> str:
    """Canonicalize explicit genes so the shared uppercase router targets them."""
    routed = question
    for gene in sorted(intent.genes, key=len, reverse=True):
        routed = re.sub(
            rf"(?<![A-Za-z0-9-]){re.escape(gene)}(?![A-Za-z0-9-])",
            gene,
            routed,
            flags=re.IGNORECASE,
        )
    return routed


def _is_negative_control_question(question: str) -> bool:
    lowered = question.casefold().replace("\u2019", "'")
    pattern = re.compile(
        r"\b(?:refut\w*|falsif\w*|disprov\w*|counter[-\s]?evidence|"
        r"counterexamples?|contradict\w*|negative controls?|rule\w*\s+out)\b"
    )
    negation = re.compile(
        r"(?:\bdo\s+not\b|\bdon't\b|\bdont\b|\bnot\b|\bnever\b|\bwithout\b|"
        r"\bavoid(?:ing)?\b|\brather\s+than\b|\binstead\s+of\b)"
    )
    question_pivot = re.compile(
        r"\b(?:what|which|how|why|when|where)\b"
        r"(?=[^;.!?]*\b(?:would|could|can|will|does|do|is|are)\b)"
    )
    # A keyword only expresses an affirmative challenge when it is not under a
    # negator in the same sentence/semicolon-bounded clause. Because commas can
    # enclose parentheticals, they cannot discard a governing "do not". An
    # explicit question pivot starts the requested action, preserving
    # "Without using an LLM, what would refute ..." as affirmative.
    for clause in re.split(r"[;.!?]+", lowered):
        for match in pattern.finditer(clause):
            local_prefix = clause[:match.start()]
            pivots = list(question_pivot.finditer(local_prefix))
            if pivots:
                local_prefix = local_prefix[pivots[-1].start():]
            if negation.search(local_prefix):
                continue
            return True
    return False


def _participant_scope_texts(row: dict) -> tuple[str, ...]:
    """Return participant labels separately so their attributes cannot mix."""
    return tuple(
        _target_name(str(row.get(field, "")))
        for field in (
            "source", "target", "celltype", "celltype_id",
            "source_cell", "target_cell", "cell",
        )
        if row.get(field)
    )


def _scope_matches_negative_evidence(
        trace: ToolResult, target: ToolResult, question: str, family: str) -> bool:
    """Require corroborating evidence to describe the empty lookup's scope/program."""
    if not trace.rows:
        return False
    target_compartment = _target_name(str(target.filters.get("compartment") or ""))
    evidence_compartment = _target_name(str(trace.filters.get("compartment") or ""))
    target_ephys = str(target.filters.get("ephys") or "")
    target_celltype = _target_name(str(target.filters.get("celltype") or ""))

    if not target_celltype:
        return False
    identity_pattern = re.compile(rf"\b{re.escape(target_celltype)}\b")
    matching_participants = [
        participant
        for row in trace.rows
        for participant in _participant_scope_texts(row)
        if identity_pattern.search(participant)
    ]
    if not matching_participants:
        return False

    if target_ephys:
        normalized_ephys = _target_name(target_ephys)
        ephys_pattern = re.compile(rf"\b{re.escape(normalized_ephys)}\b")
        if not any(ephys_pattern.search(participant)
                   for participant in matching_participants):
            return False

    if trace.tool == "pathway_contrast_lookup":
        if not target_compartment or evidence_compartment != target_compartment:
            return False
        return True

    if trace.tool not in {"pathway_filter", "exclusive_pathway_lookup"}:
        return False
    if not target_compartment or evidence_compartment != target_compartment:
        return False
    evidence = _target_name(json.dumps(
        {"filters": trace.filters, "rows": trace.rows}, ensure_ascii=False))
    family_relevant = bool(re.search(rf"\b{re.escape(_target_name(family))}[a-z0-9]*\b", evidence))
    program_relevant = (
        bool(re.search(r"\bsynap\w*\b", question, flags=re.I))
        and ("neuron tumor" in evidence or "immune synapse" in evidence)
    ) or (
        "immune" in question.casefold() and "immune synapse" in evidence
    )
    return family_relevant or program_relevant


def _uses_grounded_negative_family_evidence(
        question: str, result: dict, gene: str, found: list[dict]) -> bool:
    """Accept a real table-backed family as negative evidence in a scoped challenge.

    Family lookup is deliberately prefix-based in the table tools. It is useful
    when a refutation asks whether any GRIA-family member occurs in one
    cell/Ephys context, but it must not make an unsupported positive GRIA claim
    look like an exact gene hit.
    """
    if not _is_negative_control_question(question):
        return False
    members = {
        member
        for row in found
        for key in ("gene", "ligand", "receptor")
        for member in str(row.get(key, "")).upper().split("_")
        if member
    }
    if not any(member != gene and member.startswith(gene) for member in members):
        return False
    targeted_empty = [
        trace
        for trace in result["traces"]
        if (
            str(trace.filters.get("gene") or "").upper() == gene
            and not trace.rows
            and any(trace.filters.get(key) for key in (
                "celltype", "ephys", "compartment", "source_compartment", "target_compartment"
            ))
        )
    ]
    return bool(targeted_empty) and any(
        _scope_matches_negative_evidence(trace, target, question, gene)
        for target in targeted_empty
        for trace in result["traces"]
    )


def _question_targets_supported(question: str, result: dict,
                                engine: RAGEngine | None = None) -> bool:
    intent = extract_target_intent(question, engine)
    if isinstance(engine, RAGEngine):
        registry = engine.tools
        for ligand, receptor in intent.pairs:
            if not registry.support_join(ligand=ligand, receptor=receptor).rows:
                return False
        for gene in intent.genes:
            found = registry.cellchat_lookup(gene=gene).rows
            found += registry.deg_lookup(gene=gene).rows
            found += registry.deg_lookup(gene=gene, pooled=True).rows
            # Tool lookups intentionally support gene families/prefixes. A
            # standalone symbol must occur exactly, including complex members.
            exact = any(gene == member for row in found
                        for key in ("gene", "ligand", "receptor")
                        for member in str(row.get(key, "")).upper().split("_"))
            if not exact:
                exact = any(row.get("gene") == gene
                            for row in registry.nominate_lookup(gene=gene).rows)
            if not exact and not _uses_grounded_negative_family_evidence(
                    question, result, gene, found):
                return False
        for name in intent.pathways:
            if not any(_target_name(row["pathway"]) == _target_name(name)
                       for row in registry.pathway_filter(pathway=name).rows):
                return False
        for cell in intent.cells:
            if not (registry.cellchat_lookup(celltype=cell).rows
                    or registry.annotation_lookup(celltype=cell).rows):
                return False
    # Unknown names cannot become valid merely because a permissive tool
    # matched their substring (e.g. 'micro' in 'microglia').
    return not intent.unsupported


def _is_grounded_negative_family_control(
        question: str, result: dict, engine: RAGEngine | None = None) -> bool:
    if not isinstance(engine, RAGEngine):
        return False
    registry = engine.tools
    for gene in extract_target_intent(question, engine).genes:
        found = registry.cellchat_lookup(gene=gene).rows
        found += registry.deg_lookup(gene=gene).rows
        found += registry.deg_lookup(gene=gene, pooled=True).rows
        if _uses_grounded_negative_family_evidence(question, result, gene, found):
            return True
    return False


def _has_evidence(result: dict, question: str = "", engine: RAGEngine | None = None) -> bool:
    if question and not _question_targets_supported(question, result, engine):
        return False
    if isinstance(engine, RAGEngine):
        # The exact direct lookups above are authoritative. Requiring the
        # dispatcher's partial filters again would reject supported targets it
        # omitted or broad words it accidentally treated as gene symbols.
        return bool(result["hits"] or any(trace.rows for trace in result["traces"]))
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
        intent = extract_target_intent(question, engine)
        grounded = engine.ask(_normalize_routing_question(question, intent), use_llm=False)
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
        elif model != "none" and not _is_grounded_negative_family_control(
                question, grounded, engine):
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
