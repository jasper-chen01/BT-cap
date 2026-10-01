from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ephys_rag.evaluation import EvaluationQuestion


_PAIR_RE = re.compile(r"\b[A-Z][A-Z0-9-]*(?:_[A-Z][A-Z0-9-]*)+\b")
_DEG_RE = re.compile(
    r"\b([A-Z][A-Z0-9-]{1,})\b[^.\n]{0,35}?\b(Ephys_[12](?:[_-]high))\b",
    re.IGNORECASE,
)
_PAIR_IGNORE_PREFIXES = ("IDH_", "OPC_", "TAM_", "EPHYS_")
_DEG_FIELD_WORDS = {"BOTH", "CSV", "DIRECTION", "FILE", "GENE", "PAIR", "ROW", "TOOL"}


def _clean_multiline_text(value: Any) -> str:
    """Remove line-end padding without flattening readable model prose."""
    return "\n".join(line.rstrip() for line in str(value).splitlines()).strip()


def _contains_all(text: str, terms: Iterable[str]) -> bool:
    folded = text.casefold()
    return all(str(term).casefold() in folded for term in terms)


def _trace_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace in record.get("traces", []):
        if isinstance(trace, dict):
            rows.extend(row for row in trace.get("rows", []) if isinstance(row, dict))
    return rows


def _valid_pairs(rows: list[dict[str, Any]]) -> set[str]:
    pairs: set[str] = set()
    for row in rows:
        for key in ("pair", "interaction", "interaction_name"):
            value = row.get(key)
            if value:
                rendered = str(value).upper()
                pairs.add(rendered)
                # Neurotransmitter complexes carry descriptive prefixes, while
                # the conservative regex sees only the final gene-like suffix.
                # Treat each hyphen-delimited suffix containing an underscore
                # as an alias of the trace-supported full interaction name.
                pieces = rendered.split("-")
                for index in range(1, len(pieces)):
                    suffix = "-".join(pieces[index:])
                    if "_" in suffix:
                        pairs.add(suffix)
        ligand = row.get("ligand")
        receptor = row.get("receptor")
        if ligand and receptor:
            pairs.add(f"{ligand}_{receptor}".upper())
    return pairs


def _invented_pairs(answer: str, rows: list[dict[str, Any]]) -> list[str]:
    valid = _valid_pairs(rows)
    candidates = {
        token.upper()
        for token in _PAIR_RE.findall(answer)
        if not token.upper().startswith(_PAIR_IGNORE_PREFIXES)
    }
    return sorted(candidates - valid)


def _normalize_direction(value: str) -> str:
    return value.replace("-high", "_high").replace("-HIGH", "_HIGH")


def _invented_degs(answer: str, rows: list[dict[str, Any]]) -> list[str]:
    valid = {
        (str(row["gene"]).upper(), _normalize_direction(str(row["direction"])).casefold())
        for row in rows
        if row.get("gene") and row.get("direction")
    }
    invented: set[str] = set()
    for gene, direction in _DEG_RE.findall(answer):
        if gene.upper() in _DEG_FIELD_WORDS:
            continue
        normalized = _normalize_direction(direction)
        if (gene.upper(), normalized.casefold()) not in valid:
            invented.add(f"{gene.upper()}:{normalized}")
    return sorted(invented)


def score_run(
    record: dict[str, Any],
    question: EvaluationQuestion,
) -> dict[str, Any]:
    answer = str(record.get("answer", ""))
    traces = record.get("traces", [])
    trace_text = json.dumps(traces, ensure_ascii=False)
    rows = _trace_rows(record)
    hit = _contains_all(answer, question.required_terms)
    trace_terms = _contains_all(trace_text, question.required_terms)
    tools = {
        str(trace.get("tool", ""))
        for trace in traces
        if isinstance(trace, dict)
    }
    tools_ok = all(tool in tools for tool in question.required_tools)
    invented_pairs = _invented_pairs(answer, rows)
    invented_degs = _invented_degs(answer, rows)
    faithful = bool(
        answer.strip()
        and hit
        and trace_terms
        and tools_ok
        and not invented_pairs
        and not invented_degs
    )
    return {
        "qid": question.id.upper(),
        "run": record.get("run"),
        "hit": hit,
        "trace_terms": trace_terms,
        "tools_ok": tools_ok,
        "invented_pairs": invented_pairs,
        "invented_degs": invented_degs,
        "faithful": faithful,
        "answer": answer,
    }


def aggregate_question(
    question: EvaluationQuestion,
    scores: list[dict[str, Any]],
) -> dict[str, Any]:
    n = len(scores)
    faithful_n = sum(bool(score.get("faithful")) for score in scores)
    hit_n = sum(bool(score.get("hit")) for score in scores)
    inventions = [
        f"run {score.get('run', idx)}: "
        + "; ".join(
            filter(
                None,
                (
                    "pairs=" + ",".join(score.get("invented_pairs", []))
                    if score.get("invented_pairs")
                    else "",
                    "DEGs=" + ",".join(score.get("invented_degs", []))
                    if score.get("invented_degs")
                    else "",
                ),
            )
        )
        for idx, score in enumerate(scores, start=1)
        if score.get("invented_pairs") or score.get("invented_degs")
    ]
    return {
        "qid": question.id.upper(),
        "question": question.question,
        "medgemma_faithful": f"{faithful_n}/{n}",
        "medgemma_hit": f"{hit_n}/{n}",
        "unstable": (0 < faithful_n < n) or (0 < hit_n < n),
        "inventions": " | ".join(inventions) if inventions else "none",
    }


def load_gemini_answers(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    source = Path(path)
    if not source.exists():
        return {}
    payload = json.loads(source.read_text(encoding="utf-8"))
    records = payload.get("records", []) if isinstance(payload, dict) else []
    return {
        str(record.get("question_id", record.get("qid", ""))).upper(): str(
            record.get("answer", "")
        )
        for record in records
        if isinstance(record, dict)
    }


def load_audit_overrides(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    source = Path(path)
    if not source.exists():
        return {}
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Human audit override file must contain a JSON object")
    return {str(qid).upper(): value for qid, value in payload.items() if isinstance(value, dict)}


def _question_number(qid: str) -> int:
    match = re.search(r"\d+", qid)
    return int(match.group()) if match else -1


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._\n"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in frame.fillna("").astype(str).itertuples(index=False, name=None):
        cells = [value.replace("|", "\\|").replace("\n", "<br>") for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def build_week3_reports(
    questions: Iterable[EvaluationQuestion],
    records: list[dict[str, Any]],
    *,
    output_dir: str | Path,
    gemini_answers: dict[str, str] | None = None,
    audit_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    gemini_answers = gemini_answers or {}
    audit_overrides = {
        str(qid).upper(): value for qid, value in (audit_overrides or {}).items()
    }
    by_qid: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_qid.setdefault(str(record.get("qid", "")).upper(), []).append(record)

    rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    score_map: dict[str, list[dict[str, Any]]] = {}
    for question in questions:
        qid = question.id.upper()
        q_records = sorted(by_qid.get(qid, []), key=lambda item: int(item.get("run", 0)))
        scores = [score_run(record, question) for record in q_records]
        override = audit_overrides.get(qid)
        if override is not None:
            faithful_runs = {int(run) for run in override.get("faithful_runs", [])}
            for score in scores:
                score["automatic_faithful"] = score["faithful"]
                score["faithful"] = int(score.get("run", 0)) in faithful_runs
        else:
            for score in scores:
                score["automatic_faithful"] = score["faithful"]
        score_map[qid] = scores
        aggregate = aggregate_question(question, scores)
        tools = sorted(
            {
                str(trace.get("tool", ""))
                for record in q_records
                for trace in record.get("traces", [])
                if isinstance(trace, dict) and trace.get("tool")
            }
        )
        aggregate.update(
            {
                "tools_fired": ", ".join(tools) or "none",
                "required_terms_in_traces": (
                    "yes" if scores and all(score["trace_terms"] for score in scores) else "no"
                ),
                "gemini_cached_answer": _clean_multiline_text(
                    gemini_answers.get(qid, "not available")
                ),
                "audit_basis": "human-reviewed" if override is not None else "automatic",
                "audit_note": str(override.get("note", "")) if override is not None else "",
            }
        )
        rows.append(aggregate)
        for score in scores:
            run_rows.append(
                {
                    "qid": qid,
                    "run": score["run"],
                    "automatic_faithful": score["automatic_faithful"],
                    "audited_faithful": score["faithful"],
                    "hit": score["hit"],
                    "trace_terms": score["trace_terms"],
                    "invented_pairs": ", ".join(score["invented_pairs"]) or "none",
                    "invented_degs": ", ".join(score["invented_degs"]) or "none",
                    "answer": _clean_multiline_text(score["answer"]),
                }
            )

    ordered_columns = [
        "qid",
        "question",
        "tools_fired",
        "required_terms_in_traces",
        "medgemma_faithful",
        "medgemma_hit",
        "gemini_cached_answer",
        "inventions",
        "unstable",
        "audit_basis",
        "audit_note",
    ]
    frame = pd.DataFrame(rows, columns=ordered_columns)
    numbers = frame["qid"].map(_question_number) if not frame.empty else pd.Series(dtype=int)
    primary = frame[(numbers >= 9) & (numbers <= 32)].copy()
    appendix = frame[(numbers >= 33) & (numbers <= 38)].copy()
    table_csv = destination / "WEEK3_Q9_Q32_AUDIT.csv"
    table_md = destination / "WEEK3_Q9_Q32_AUDIT.md"
    appendix_csv = destination / "WEEK3_Q33_Q38_APPENDIX.csv"
    appendix_md = destination / "WEEK3_Q33_Q38_APPENDIX.md"
    run_csv = destination / "WEEK3_RUN_LEVEL_AUDIT.csv"
    instability_md = destination / "WEEK3_INSTABILITY_NOTES.md"
    primary.to_csv(table_csv, index=False)
    appendix.to_csv(appendix_csv, index=False)
    pd.DataFrame(run_rows).to_csv(run_csv, index=False)
    table_md.write_text(
        "# Week 3 Q9-Q32 local MedGemma audit\n\n" + _markdown_table(primary),
        encoding="utf-8",
    )
    appendix_md.write_text(
        "# Week 3 Q33-Q38 appendix\n\n" + _markdown_table(appendix),
        encoding="utf-8",
    )

    unstable_ids = set(frame.loc[frame["unstable"].astype(bool), "qid"])
    focus_ids = {"Q25", "Q27", "Q36"}
    note_ids = sorted(unstable_ids | focus_ids, key=_question_number)
    note_lines = [
        "# Week 3 stability notes",
        "",
        "A question is unstable when either the faithful or required-term hit rate is between 1/5 and 4/5.",
        "",
    ]
    row_by_qid = {str(row["qid"]): row for row in rows}
    for qid in note_ids:
        row = row_by_qid.get(qid)
        if row is None:
            note_lines.extend([f"## {qid}", "", "No cached runs were available.", ""])
            continue
        status = "unstable" if row["unstable"] else "stable"
        note_lines.extend(
            [
                f"## {qid}",
                "",
                f"Status: **{status}**; faithful {row['medgemma_faithful']}; hit {row['medgemma_hit']}.",
                "",
            ]
        )
        wrong = [score for score in score_map.get(qid, []) if not score["faithful"]]
        if not wrong:
            note_lines.extend(["All cached runs passed the automatic grounding audit.", ""])
        else:
            note_lines.append("Wrong or incomplete runs:")
            note_lines.append("")
            for score in wrong:
                answer = " ".join(str(score["answer"]).split())[:500]
                note_lines.append(f"- Run {score['run']}: {answer}")
            note_lines.append("")
    instability_md.write_text("\n".join(note_lines).rstrip() + "\n", encoding="utf-8")
    return {
        "table_csv": table_csv,
        "table_md": table_md,
        "appendix_csv": appendix_csv,
        "appendix_md": appendix_md,
        "run_csv": run_csv,
        "instability_md": instability_md,
    }
