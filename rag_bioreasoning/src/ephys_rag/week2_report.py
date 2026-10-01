from __future__ import annotations

import re
from typing import Any, Mapping


_TRACE_MARKER = re.compile(r"\n\s*(?:\*\s*)?(?:tool=|Tool:)", re.IGNORECASE)
_INTERPRETATION_HEADING = re.compile(
    r"(?im)^#{1,6}\s*(?:\d+\.\s*)?\**Interpretation\**\s*:?\s*$"
)
_INLINE_INTERPRETATION = re.compile(
    r"(?im)^\s*\*+\s*\**Interpretation\s*:?\**\s*:?\s*"
)
_SECTION_END = re.compile(r"(?im)^#{1,6}\s|^\s*\*?\s*\(?Note:")
WEEK2_COLUMNS = [
    "Question",
    "Tool that fired",
    "Required term in traces?",
    "MedGemma answer",
    "Gemini answer",
    "Either model invented a pair/DEG?",
]


def concise_answer(answer: str) -> str:
    """Keep the model's conclusion while removing echoed tool-trace blocks."""

    text = str(answer).strip()
    interpretation = _INTERPRETATION_HEADING.search(text)
    if not interpretation:
        interpretation = _INLINE_INTERPRETATION.search(text)
    if interpretation:
        text = text[interpretation.end() :].strip()
        section_end = _SECTION_END.search(text)
        if section_end:
            text = text[: section_end.start()]
    else:
        marker = _TRACE_MARKER.search(text)
        if marker:
            text = text[: marker.start()]
    return " ".join(text.split())


def _records_by_question(summary: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(record["question_id"]): record
        for record in summary.get("records", [])
    }


def build_week2_rows(
    medgemma_summary: Mapping[str, Any],
    gemini_summary: Mapping[str, Any],
    *,
    invention_audit: Mapping[str, str],
) -> list[dict[str, str]]:
    """Combine two provider runs into the table required by WEEK2.md."""

    medgemma = _records_by_question(medgemma_summary)
    gemini = _records_by_question(gemini_summary)
    rows: list[dict[str, str]] = []
    for question_id, med_record in medgemma.items():
        gemini_record = gemini[question_id]
        found = list(med_record.get("trace_found_terms", []))
        missing = list(med_record.get("trace_missing_terms", []))
        total = len(found) + len(missing)
        trace_status = f"{'Yes' if not missing else 'No'} ({len(found)}/{total})"
        tools = list(
            dict.fromkeys(
                [
                    *med_record.get("tools_fired", []),
                    *gemini_record.get("tools_fired", []),
                ]
            )
        )
        number = question_id.removeprefix("q").removeprefix("Q")
        rows.append(
            {
                "Question": f"Q{number}: {med_record['question']}",
                "Tool that fired": ", ".join(tools),
                "Required term in traces?": trace_status,
                "MedGemma answer": concise_answer(str(med_record.get("answer", ""))),
                "Gemini answer": concise_answer(str(gemini_record.get("answer", ""))),
                "Either model invented a pair/DEG?": invention_audit.get(
                    question_id, "Not reviewed"
                ),
            }
        )
    return rows


def _markdown_cell(value: Any) -> str:
    return " ".join(str(value).split()).replace("|", "\\|")


def render_week2_markdown(rows: list[Mapping[str, Any]]) -> str:
    """Render the exact comparison-table shape requested in WEEK2.md."""

    lines = [
        "# WEEK2 Q9–Q32: MedGemma and Gemini on the Same Traces",
        "",
        "| " + " | ".join(WEEK2_COLUMNS) + " |",
        "|" + "|".join("---" for _ in WEEK2_COLUMNS) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_markdown_cell(row.get(column, "")) for column in WEEK2_COLUMNS)
            + " |"
        )
    return "\n".join(lines) + "\n"
