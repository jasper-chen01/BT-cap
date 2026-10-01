from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping


REQUIRED_RUN_FIELDS = (
    "qid",
    "provider",
    "model",
    "run",
    "question",
    "context",
    "answer",
    "ts",
    "traces",
    "retrieved_titles",
)


class InvalidRunCache(ValueError):
    """Raised when a cached model response is incomplete or malformed."""


def _safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_-]+", "-", str(value).strip())
    if not label:
        raise ValueError("Cache labels must not be empty.")
    return label


def cache_path(
    runs_dir: str | Path,
    qid: str,
    cache_label: str,
    run: int,
) -> Path:
    if int(run) < 1:
        raise ValueError("Run numbers start at 1.")
    normalized_qid = str(qid).strip().upper()
    if not normalized_qid.startswith("Q"):
        normalized_qid = f"Q{normalized_qid}"
    return Path(runs_dir) / (
        f"{_safe_label(normalized_qid)}_{_safe_label(cache_label)}_run{int(run)}.json"
    )


def validate_run(record: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in REQUIRED_RUN_FIELDS if key not in record]
    if missing:
        raise InvalidRunCache("Cached run is missing fields: " + ", ".join(missing))
    if not isinstance(record["run"], int) or record["run"] < 1:
        raise InvalidRunCache("Cached run has an invalid run number.")
    for key in ("qid", "provider", "model", "question", "context", "answer", "ts"):
        if not isinstance(record[key], str):
            raise InvalidRunCache(f"Cached run field '{key}' must be text.")
    for key in ("traces", "retrieved_titles"):
        if not isinstance(record[key], list):
            raise InvalidRunCache(f"Cached run field '{key}' must be a list.")
    return dict(record)


def write_run_atomic(path: str | Path, record: Mapping[str, Any]) -> Path:
    destination = Path(path)
    payload = validate_run(record)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


def load_run(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InvalidRunCache(f"Cached run is unreadable: {source.name}") from exc
    if not isinstance(payload, dict):
        raise InvalidRunCache("Cached run must contain a JSON object.")
    return validate_run(payload)
