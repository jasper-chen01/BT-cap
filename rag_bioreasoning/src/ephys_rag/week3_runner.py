from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ephys_rag.evaluation import EvaluationQuestion
from ephys_rag.week3_cache import InvalidRunCache, cache_path, load_run, write_run_atomic


def _serialize_trace(trace: Any) -> dict[str, Any]:
    if isinstance(trace, dict):
        return dict(trace)
    return {
        "tool": getattr(trace, "tool", "unknown"),
        "source_file": getattr(trace, "source_file", ""),
        "filters": getattr(trace, "filters", {}),
        "n_hits": getattr(trace, "n_hits", 0),
        "rows": getattr(trace, "rows", []),
        "note": getattr(trace, "note", ""),
    }


def _retrieved_title(hit: Any) -> str:
    if isinstance(hit, str):
        return hit
    document = getattr(hit, "document", None)
    return str(getattr(document, "title", ""))


def _record_from_result(
    question: EvaluationQuestion,
    result: dict[str, Any],
    run: int,
) -> dict[str, Any]:
    return {
        "qid": question.id.upper(),
        "provider": str(result.get("provider", "")),
        "model": str(result.get("model", "")),
        "run": int(run),
        "question": question.question,
        "context": str(result.get("context", "")),
        "answer": str(result.get("answer", "")),
        "ts": datetime.now(timezone.utc).isoformat(),
        "traces": [_serialize_trace(trace) for trace in result.get("traces", [])],
        "retrieved_titles": [
            title
            for title in (_retrieved_title(hit) for hit in result.get("hits", []))
            if title
        ],
        "elapsed_ms": float(result.get("elapsed_ms", 0.0)),
    }


def run_repeated_evaluation(
    engine: Any,
    questions: Iterable[EvaluationQuestion],
    *,
    provider: str,
    cache_label: str,
    runs_dir: str | Path,
    repeats: int = 5,
    top_k: int = 0,
    trace_limit: int = 2,
    include_trace_catalog: bool = True,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for question in questions:
        for run in range(1, repeats + 1):
            path = cache_path(runs_dir, question.id, cache_label, run)
            if path.exists():
                try:
                    records.append(load_run(path))
                    continue
                except InvalidRunCache:
                    pass
            result = engine.ask(
                question.question,
                provider=provider,
                top_k=top_k,
                trace_limit=trace_limit,
                include_trace_catalog=include_trace_catalog,
            )
            record = _record_from_result(question, result, run)
            write_run_atomic(path, record)
            records.append(record)
    return records


def load_cached_runs(
    questions: Iterable[EvaluationQuestion],
    *,
    cache_label: str,
    runs_dir: str | Path,
    repeats: int = 5,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for question in questions:
        for run in range(1, repeats + 1):
            records.append(
                load_run(cache_path(runs_dir, question.id, cache_label, run))
            )
    return records
