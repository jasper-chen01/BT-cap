from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PACKAGE_ROOT / "evaluation" / "questions.json"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "analysis_runs" / "medgemma_evaluation"


@dataclass(frozen=True)
class EvaluationQuestion:
    id: str
    question: str
    required_terms: list[str]
    notes: str = ""
    required_tools: list[str] = field(default_factory=list)


@dataclass
class EvaluationRecord:
    question_id: str
    question: str
    provider: str
    model: str
    answer: str
    elapsed_ms: float
    found_terms: list[str]
    missing_terms: list[str]
    traces: list[dict[str, Any]]
    retrieved_titles: list[str]
    tools_fired: list[str] = field(default_factory=list)
    missing_required_tools: list[str] = field(default_factory=list)
    trace_found_terms: list[str] = field(default_factory=list)
    trace_missing_terms: list[str] = field(default_factory=list)
    notes: str = ""
    error: str | None = None


@dataclass
class EvaluationSummary:
    records: list[EvaluationRecord]
    started_at: str
    success: bool
    providers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "success": self.success,
            "providers": self.providers,
            "records": [asdict(record) for record in self.records],
        }


def load_question_manifest(path: str | Path | None = None) -> list[EvaluationQuestion]:
    manifest_path = Path(path) if path is not None else DEFAULT_MANIFEST
    with manifest_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("Evaluation manifest must contain a JSON list.")
    questions: list[EvaluationQuestion] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Every evaluation manifest entry must be an object.")
        questions.append(
            EvaluationQuestion(
                id=str(item["id"]),
                question=str(item["question"]),
                required_terms=[str(term) for term in item.get("required_terms", [])],
                notes=str(item.get("notes", "")),
                required_tools=[str(tool) for tool in item.get("required_tools", [])],
            )
        )
    return questions


def score_required_terms(answer: str, required_terms: Iterable[str]) -> tuple[list[str], list[str]]:
    lowered = answer.casefold()
    found: list[str] = []
    missing: list[str] = []
    for term in required_terms:
        target = str(term)
        (found if target.casefold() in lowered else missing).append(target)
    return found, missing


_BEARER_RE = re.compile(r"(?i)\bBearer\s+[^\s,;]+")
_JSON_PATH_RE = re.compile(
    r"(?:[A-Za-z]:\\[^\r\n]*?\.json|/(?:[^\s/]+/)*[^\s]+\.json)",
    re.IGNORECASE,
)


def redact_error(message: str) -> str:
    redacted = _BEARER_RE.sub("Bearer [REDACTED]", str(message))
    redacted = _JSON_PATH_RE.sub("[REDACTED_CREDENTIAL_PATH]", redacted)
    return redacted


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


def run_evaluation(
    engine: Any,
    questions: Iterable[EvaluationQuestion],
    providers: Iterable[str],
    *,
    top_k: int = 10,
    trace_limit: int = 12,
    include_trace_catalog: bool = False,
) -> EvaluationSummary:
    provider_names = [str(provider) for provider in providers]
    records: list[EvaluationRecord] = []
    started_at = datetime.now(timezone.utc).isoformat()
    for provider in provider_names:
        for question in questions:
            try:
                result = engine.ask(
                    question.question,
                    provider=provider,
                    top_k=top_k,
                    trace_limit=trace_limit,
                    include_trace_catalog=include_trace_catalog,
                )
                answer = str(result["answer"])
                found, missing = score_required_terms(answer, question.required_terms)
                traces = [
                    _serialize_trace(trace)
                    for trace in result.get("traces", [])
                ]
                trace_found, trace_missing = score_required_terms(
                    json.dumps(traces, ensure_ascii=False),
                    question.required_terms,
                )
                tools_fired = list(dict.fromkeys(
                    str(trace.get("tool", "unknown")) for trace in traces
                ))
                records.append(
                    EvaluationRecord(
                        question_id=question.id,
                        question=question.question,
                        provider=str(result.get("provider", provider)),
                        model=str(result.get("model", "")),
                        answer=answer,
                        elapsed_ms=float(result.get("elapsed_ms", 0.0)),
                        found_terms=found,
                        missing_terms=missing,
                        traces=traces,
                        retrieved_titles=[
                            title
                            for title in (
                                _retrieved_title(hit) for hit in result.get("hits", [])
                            )
                            if title
                        ],
                        tools_fired=tools_fired,
                        missing_required_tools=[
                            tool
                            for tool in question.required_tools
                            if tool not in tools_fired
                        ],
                        trace_found_terms=trace_found,
                        trace_missing_terms=trace_missing,
                        notes=question.notes,
                    )
                )
            except Exception as exc:
                records.append(
                    EvaluationRecord(
                        question_id=question.id,
                        question=question.question,
                        provider=provider,
                        model="",
                        answer="",
                        elapsed_ms=0.0,
                        found_terms=[],
                        missing_terms=list(question.required_terms),
                        traces=[],
                        retrieved_titles=[],
                        tools_fired=[],
                        missing_required_tools=list(question.required_tools),
                        trace_found_terms=[],
                        trace_missing_terms=list(question.required_terms),
                        notes=question.notes,
                        error=redact_error(str(exc)),
                    )
                )
    return EvaluationSummary(
        records=records,
        started_at=started_at,
        success=all(record.error is None for record in records),
        providers=provider_names,
    )


def _markdown_report(summary: EvaluationSummary) -> str:
    lines = [
        "# Bio-Reasoning Model Evaluation",
        "",
        f"Started: {summary.started_at}",
        f"Overall request status: {'PASS' if summary.success else 'FAIL'}",
        "",
        "| Question | Provider | Model | Trace terms | Answer terms | Tools fired | Request |",
        "|---|---|---|---|---|---|---|",
    ]
    for record in summary.records:
        answer_term_status = f"{len(record.found_terms)}/{len(record.found_terms) + len(record.missing_terms)}"
        trace_term_status = f"{len(record.trace_found_terms)}/{len(record.trace_found_terms) + len(record.trace_missing_terms)}"
        request_status = "ERROR" if record.error else "OK"
        lines.append(
            f"| {record.question_id} | {record.provider.title()} | {record.model or '—'} | "
            f"{trace_term_status} | {answer_term_status} | "
            f"{', '.join(record.tools_fired) or 'none'} | {request_status} |"
        )
    for record in summary.records:
        lines.extend(
            [
                "",
                f"## {record.question_id} — {record.provider.title()}",
                "",
                f"**Question:** {record.question}",
                "",
                f"**Model:** {record.model or 'unavailable'}",
                "",
                f"**Elapsed:** {record.elapsed_ms:.3f} ms",
                "",
                f"**Required terms found:** {', '.join(record.found_terms) or 'none'}",
                "",
                f"**Required terms missing:** {', '.join(record.missing_terms) or 'none'}",
                "",
                f"**Trace terms found:** {', '.join(record.trace_found_terms) or 'none'}",
                "",
                f"**Trace terms missing:** {', '.join(record.trace_missing_terms) or 'none'}",
                "",
                f"**Tools fired:** {', '.join(record.tools_fired) or 'none'}",
                "",
                f"**Required tools missing:** {', '.join(record.missing_required_tools) or 'none'}",
                "",
            ]
        )
        if record.error:
            lines.append(f"**Error:** {record.error}")
        else:
            lines.extend(["**Answer:**", "", record.answer])
        lines.extend(
            [
                "",
                "**Tool traces:**",
                "",
                "```json",
                json.dumps(record.traces, indent=2, ensure_ascii=False),
                "```",
                "",
                "**Retrieved titles:**",
                "",
            ]
        )
        lines.extend(
            [f"- {title}" for title in record.retrieved_titles]
            or ["- none"]
        )
    return "\n".join(lines) + "\n"


def write_evaluation_reports(
    summary: EvaluationSummary,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = (
        summary.started_at.replace("-", "")
        .replace(":", "")
        .replace("+00:00", "Z")
        .replace(".", "")
    )
    json_path = directory / f"evaluation_{stamp}.json"
    markdown_path = directory / f"evaluation_{stamp}.md"
    json_path.write_text(
        json.dumps(summary.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_markdown_report(summary), encoding="utf-8")
    return json_path, markdown_path
