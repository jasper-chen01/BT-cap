"""Grounded reasoning and read-only held-out gene scores."""
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query

from backend.models.schemas import ReasoningAskRequest
from backend.services.reasoning_service import (
    ReasoningService, get_nomination_store, get_rag_engine, serialize_tool_result,
)
from ephys_rag import config
from ephys_rag.chain import RAGEngine
from ephys_rag.nomination import NominationStore

router = APIRouter()
GeneModel = Literal["tumor_synaptic", "immune_synapse"]
FeatureArm = Literal["tables", "tables_scetm", "scetm_only"]

# Keep a predictable leading column order even for empty ranked results.
RANKED_COLUMNS = (
    "gene", "model", "arm", "celltype_id", "IDH_status", "score", "gene_rank",
    "gene_score", "context_score", "rank_in_context", "why", "scoring_folds",
    "min_n", "passes_three_table", "in_label_set", "in_syngo", "in_go_0001772",
    "in_published_33", "housekeeping", "novel",
)


def _unavailable(error: Exception) -> HTTPException:
    return HTTPException(status_code=503, detail=(
        "Reasoning runtime unavailable; run scripts/prepare_reasoning_data.py "
        f"and check configured data/model settings. {error}"
    ))


def _validate_runtime() -> None:
    # Nomination and evidence are one prepared runtime. Reject partial drops
    # before a cached service can expose scores without the required evidence.
    paths = [config.CELLCHAT_CSV, config.CELLCHAT_COUNTS_CSV,
             config.ANNOTATION_COUNTS_CSV, config.ANNOTATION_CELLS_CSV,
             config.DEG_IDH_CSV, config.DEG_POOLED_CSV,
             config.DEG_IDH_SUMMARY_CSV, config.DEG_POOLED_SUMMARY_CSV,
             *config.ARM_SCORE_FILES.values(), config.ARMS_GENE_RANKS_CSV,
             config.SCETM_GENE_EMBEDDINGS_CSV]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))


def get_store() -> NominationStore:
    try:
        _validate_runtime()
        return get_nomination_store()
    except (OSError, ValueError, KeyError) as error:
        raise _unavailable(error) from error


def get_engine() -> RAGEngine:
    try:
        _validate_runtime()
        return get_rag_engine()
    except (OSError, ValueError, KeyError) as error:
        raise _unavailable(error) from error


def get_reasoning_service(engine: RAGEngine = Depends(get_engine)) -> ReasoningService:
    return ReasoningService(engine=engine)


@router.post("/reasoning/ask")
def ask_reasoning(payload: ReasoningAskRequest,
                  service: ReasoningService = Depends(get_reasoning_service)) -> dict:
    try:
        return service.ask(payload.question, payload.model)
    except (OSError, RuntimeError) as error:
        raise _unavailable(error) from error


# The static path must precede /genes/{symbol}.
@router.get("/genes/ranked")
def ranked_genes(model: GeneModel = "tumor_synaptic", arm: FeatureArm = "tables_scetm",
                 exclude_housekeeping: bool = True, novel_only: bool = False,
                 rule_pass: bool | None = None, min_n: int = Query(0, ge=0),
                 limit: int = Query(15, ge=0), store: NominationStore = Depends(get_store)) -> dict:
    rows = store.ranked_rows(model=model, arm=arm, exclude_housekeeping=exclude_housekeeping,
                             novel_only=novel_only, rule_pass=rule_pass, min_n=min_n, limit=limit)
    extra = sorted(set(store.arm_scores[arm].columns) - set(RANKED_COLUMNS))
    columns = [*RANKED_COLUMNS, *extra]
    return {"rows": [{key: row.get(key) for key in columns} for row in rows], "columns": columns}


@router.get("/genes/{symbol}")
def gene_detail(symbol: str, store: NominationStore = Depends(get_store),
                engine: RAGEngine = Depends(get_engine)) -> dict:
    gene = symbol.strip().upper()
    try:
        rows = store.gene_rows(gene)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=f"Unknown gene: {gene}") from error
    badge_names = ("in_syngo", "in_go_0001772", "in_published_33", "housekeeping",
                   "passes_three_table")
    return {"gene": gene, "rows": rows,
            "badges": {name: any(bool(row.get(name)) for row in rows) for name in badge_names},
            "cellchat": serialize_tool_result(engine.tools.cellchat_lookup(gene=gene)),
            "deg": serialize_tool_result(engine.tools.deg_lookup(gene=gene)),
            "pooled_deg": serialize_tool_result(engine.tools.deg_lookup(gene=gene, pooled=True))}


@router.get("/genes/{symbol}/neighbors")
def gene_neighbors(symbol: str, model: GeneModel = "tumor_synaptic", k: int = Query(20, ge=1, le=100),
                   store: NominationStore = Depends(get_store)) -> dict:
    gene = symbol.strip().upper()
    try:
        rows = store.neighbors(gene, model=model, k=k)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=f"Unknown embedding gene: {gene}") from error
    return {"gene": gene, "model": model, "rows": rows}
