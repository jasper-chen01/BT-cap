"""Compare the real portal router with a separately loaded, in-process CLI engine.

No inference or cache import is performed. Missing historical cache entries fail
the check. The published-33 check compares tables_scetm context scores/ranks with
nominate_lookup (six decimal probability precision), and saved gene-level ranks
with arms_gene_ranks.csv; context and gene ranks are different quantities.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class QuestionCase:
    question_id: int
    question: str
    # Each tuple is an allowed complete tool combination (OR of ANDs).
    expected_tools: tuple[tuple[str, ...], ...]
    # Every group is required; alternatives within a group are interchangeable.
    required_terms: tuple[tuple[str, ...], ...]
    requirement: str = ""


# Descriptive phrases in questions.md need predicates, not literal substring
# searches. Literal gene/pathway lists are parsed from the source, not copied.
DESCRIPTIVE_TERMS = {
    "a dataset-overview or annotation chunk that lists cell types": (("cell-type overview",),),
    "overview or raw interaction stats": (("interaction statistics",),),
    "a row that shows the pval field": (("pval",),),
    "NRXN tumor contrast": (("NRXN",), ("tumor",), ("Ephys_1",), ("Ephys_2",)),
    "GRIA2 and both IDH groups": (("GRIA2",), ("both IDH groups",)),
    "T cell and a count": (("T cell",), ("cell count",)),
    "NRXN1 from CellChat and from deg_lookup": (("NRXN1",), ("CellChat and DEG",)),
    "T cell. An empty NRXN/Glutamate hit on T-cell senders is correct.": (("T cell",),),
    "T cell and AREG or EGF. Empty T-cell NRXN is allowed.": (("T cell",), ("AREG", "EGF")),
    "NRXN1, p_tumor_synaptic, a feature name, and scoring fold":
        (("NRXN1",), ("p_tumor_synaptic",), ("feature name",), ("scoring fold",)),
    "the rank-1 gene, p_immune_synapse, and T cell":
        (("rank-1 gene",), ("p_immune_synapse",), ("T cell",)),
    "a gene, in_syngo=False, and p_tumor_synaptic":
        (("a gene",), ("in_syngo=False",), ("p_tumor_synaptic",)),
    "GRIA2, tumor rank, top features, and tables, scETM, or mixed feature family":
        (("GRIA2",), ("tumor rank",), ("feature name",), ("feature family",)),
}


def _clean(text: str) -> str:
    return re.sub(r"[`*]", "", text).strip()


def _term_groups(text: str) -> tuple[tuple[str, ...], ...]:
    if text in DESCRIPTIVE_TERMS:
        return DESCRIPTIVE_TERMS[text]
    if ", or " in text and " and " not in text:
        return (tuple(part.strip() for part in re.split(r",\s*(?:or\s+)?|\s+or\s+", text)
                      if part.strip()),)
    text = re.sub(r",?\s+and\s+", ", ", text)
    return tuple(tuple(part.strip() for part in re.split(r",?\s+or\s+", group))
                 for group in text.split(",") if group.strip())


def load_question_cases(path: Path) -> list[QuestionCase]:
    text = Path(path).read_text(encoding="utf-8-sig")
    matches = list(re.finditer(r"(?m)^(\d+)\.\s+(.+)$", text))
    numbers = [int(match[1]) for match in matches]
    if sorted(numbers) != list(range(1, 45)):
        raise ValueError(f"Expected exactly Q1 through Q44 once each; found {numbers}")
    cases = []
    for index, match in enumerate(matches):
        body = text[match.end():matches[index + 1].start() if index + 1 < len(matches) else len(text)]
        required = re.findall(r"(?m)^\s+- Must retrieve:\s*(.+)$", body)
        if len(required) != 1:
            raise ValueError(f"Q{match[1]} must have one Must retrieve line")
        requirement = _clean(required[0])
        tools = re.findall(r"(?m)^\s+- Tool:\s*(.+)$", body)
        if tools:
            named = re.findall(r"`(\w+)`", tools[0])
            if "both lookups" in tools[0]:
                expected = (("support_join",), ("cellchat_lookup", "deg_lookup"))
            else:
                expected = tuple((name,) for name in named)
            if not expected:
                raise ValueError(f"Q{match[1]} has an unrecognized tool requirement")
        else:
            # Q1-Q20 and stretch questions have no explicit Tool line. Keep the
            # independently specified minimal intent where it is unambiguous.
            implicit = {1: "exclusive_pathway_lookup", 3: "annotation_lookup",
                        6: "dataset_stats_lookup", 16: "cellchat_lookup",
                        20: "pathway_contrast_lookup"}
            expected = ((implicit[int(match[1])],),) if int(match[1]) in implicit else ()
        cases.append(QuestionCase(int(match[1]), _clean(match[2]), expected,
                                  _term_groups(requirement), requirement))
    return sorted(cases, key=lambda case: case.question_id)


def _normalize(text: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).casefold()))


def _contains(blob: str, term: str) -> bool:
    # Keep number boundaries (6 must not match a probability or a larger count).
    words = _normalize(term).split()
    pattern = r"\s+".join(re.escape(word) for word in words)
    return bool(re.search(r"(?<![a-z0-9])" + pattern + r"(?![a-z0-9])", blob))


def _evidence(result: dict) -> tuple[str, list[dict]]:
    traces = result.get("traces", [])
    # Query filters and the echoed question are not retrieved evidence.
    parts = [json.dumps(trace.get("rows", []), ensure_ascii=False) for trace in traces]
    parts += [trace.get("note", "") for trace in traces]
    parts += [hit.get("title", "") for hit in result.get("hits", [])]
    if "retrieved_context" in result:
        parts.append(result["retrieved_context"])
    else:
        # The API's extractive answer contains full retrieval blocks. Its
        # leading question and tool-filter text must not satisfy requirements.
        start = re.search(r"(?m)^\[1\] \([^\n,]+, score=[0-9.]+\) ", result.get("answer", ""))
        if start:
            parts.append(result["answer"][start.start():])
    return _normalize("\n".join(parts)), traces


def _term_found(term: str, blob: str, traces: list[dict]) -> bool:
    rows = [row for trace in traces for row in trace.get("rows", [])]
    nominated = [row for trace in traces if trace["tool"] == "nominate_lookup"
                 for row in trace.get("rows", [])]
    if term == "cell-type overview":
        return any(trace["tool"] in {"annotation_lookup", "dataset_stats_lookup"}
                   and (trace.get("rows") or trace.get("note")) for trace in traces)
    if term == "interaction statistics":
        return any(trace["tool"] == "dataset_stats_lookup" and trace.get("rows") for trace in traces)
    if term == "both IDH groups":
        return _contains(blob, "IDH_Mutant") and _contains(blob, "IDH_WT")
    if term == "cell count":
        return any(isinstance(row.get("n_cells"), (int, float)) for row in rows)
    if term == "6":
        return any(row.get("n_cells") == 6 and _contains(_normalize(json.dumps(row)), "MES")
                   for row in rows)
    if term == "CellChat and DEG":
        cellchat = any("NRXN1" in str(row.get(field, "")).upper().split("_")
                       for trace in traces if trace["tool"] == "cellchat_lookup"
                       for row in trace.get("rows", []) for field in ("ligand", "receptor"))
        deg = any(row.get("gene", "").upper() == "NRXN1"
                  for trace in traces if trace["tool"] == "deg_lookup"
                  for row in trace.get("rows", []))
        joined = any(row.get("pair", "").upper().startswith("NRXN1_")
                     and row.get("ligand_deg") not in (None, "missing")
                     for trace in traces if trace["tool"] == "support_join"
                     for row in trace.get("rows", []))
        return (cellchat and deg) or joined
    if term == "feature name":
        return any(row.get("top3_tumor_features") for row in nominated)
    if term == "scoring fold":
        return any(row.get("tumor_scoring_folds") for row in nominated)
    if term == "rank-1 gene":
        return any(row.get("gene") and row.get("immune_rank_in_context") == 1 for row in nominated)
    if term == "a gene":
        return any(row.get("gene") for row in nominated)
    if term == "in_syngo=False":
        return any(row.get("in_syngo") is False for row in nominated)
    if term == "tumor rank":
        return any(row.get("tumor_rank_in_context") is not None for row in nominated)
    if term == "feature family":
        return any(row.get("tumor_feature_family") in {"tables", "scETM", "mixed"} for row in nominated)
    if term == "TAM/microglia":
        return _contains(blob, "microglia") and bool(re.search(r"\btam[12]?\b", blob))
    return _contains(blob, term)


def compare_api_and_cli(question: QuestionCase, api_result: dict, cli_result: dict) -> dict:
    api_blob, api_traces = _evidence(api_result)
    cli_blob, cli_traces = _evidence(cli_result)
    api_tools = sorted({trace["tool"] for trace in api_traces})
    cli_tools = sorted({trace["tool"] for trace in cli_traces})
    api_terms = {" or ".join(group): any(_term_found(term, api_blob, api_traces) for term in group)
                 for group in question.required_terms}
    cli_terms = {" or ".join(group): any(_term_found(term, cli_blob, cli_traces) for term in group)
                 for group in question.required_terms}
    if question.question_id == 36:
        # This source requirement explicitly accepts an empty targeted lookup.
        # Only here can a target filter establish the negative-control evidence:
        # both named pathways must have executed for the requested T-cell group.
        def valid_negative(traces):
            targets = set()
            for trace in traces:
                filters = trace.get("filters", {})
                if (trace["tool"] == "cellchat_lookup"
                        and _normalize(filters.get("celltype", "")) == "t cell"
                        and trace.get("n_hits") == 0 and not trace.get("rows")):
                    targets.add(filters.get("gene") or filters.get("pathway"))
            return {"NRXN", "Glutamate"} <= targets
        api_terms["T cell"] = api_terms["T cell"] or valid_negative(api_traces)
        cli_terms["T cell"] = cli_terms["T cell"] or valid_negative(cli_traces)

    def expected(actual):
        return not question.expected_tools or any(set(option) <= set(actual) for option in question.expected_tools)

    result = {"question_id": question.question_id, "question": question.question,
              "requirement": question.requirement, "expected_tools": question.expected_tools,
              "api_tools": api_tools, "cli_tools": cli_tools,
              "tools_match": api_tools == cli_tools,
              "expected_tools_ok": expected(api_tools) and expected(cli_tools),
              "api_terms": api_terms, "cli_terms": cli_terms,
              "api_required_ok": all(api_terms.values()), "cli_required_ok": all(cli_terms.values()),
              "required_terms_match": api_terms == cli_terms}
    result["matched"] = all(result[key] for key in (
        "tools_match", "expected_tools_ok", "api_required_ok", "cli_required_ok", "required_terms_match"))
    return result


def compare_published_rows(api_rows: list[dict], tool_rows: list[dict], gene_ranks: dict) -> dict:
    errors = []
    expected = {}
    for row in tool_rows:
        for model, prefix in (("tumor_synaptic", "tumor"), ("immune_synapse", "immune")):
            if row.get(f"p_{model}") is not None:
                key = (row["gene"], row["celltype_id"], row["IDH_status"], model)
                expected[key] = (row[f"p_{model}"], row.get(f"{prefix}_rank_in_context"))
    if not expected:
        errors.append("Scored published gene has no nominate_lookup contexts")
    actual = {}
    for row in api_rows:
        if row.get("arm") == "tables_scetm":
            key = (row["gene"], row["celltype_id"], row["IDH_status"], row["model"])
            if key in actual:
                errors.append(f"duplicate API context {key}")
            actual[key] = row
    if set(actual) != set(expected):
        errors.append(f"context keys differ: missing={sorted(set(expected)-set(actual))}; extra={sorted(set(actual)-set(expected))}")
    for key in sorted(set(expected) & set(actual)):
        score, rank = expected[key]
        row, model = actual[key], key[-1]
        # The tool deliberately rounds probabilities to six decimal places.
        if row.get("score") is None or round(row["score"], 6) != score:
            errors.append(f"{key}: context probability differs")
        raw_probability = row.get(f"p_{model}")
        if raw_probability is None or round(raw_probability, 6) != score:
            errors.append(f"{key}: raw probability differs")
        if row.get("rank_in_context") != rank:
            errors.append(f"{key}: context rank differs")
        prefix = "tumor" if model == "tumor_synaptic" else "immune"
        if row.get(f"{prefix}_rank_in_context") != rank:
            errors.append(f"{key}: raw context rank differs")
        if row.get("gene_rank") != gene_ranks.get(f"gene_rank_{model}__tables_scetm"):
            errors.append(f"{key}: saved gene rank differs")
        if row.get("gene_score") != gene_ranks.get(f"p_{model}__tables_scetm"):
            errors.append(f"{key}: saved gene probability differs")
    return {"matched": not errors, "contexts_checked": len(expected), "errors": errors}


def check_published_gene(gene: str, api_status: int, api_rows: list[dict],
                         tool_rows: list[dict], saved_ranks: dict | None) -> dict:
    if saved_ranks is None:
        equal = api_status == 404 and not api_rows and not tool_rows
        return {"gene": gene, "matched": equal, "contexts_checked": 0,
                "status": "unscored_absence_equal" if equal else "unscored_absence_mismatch",
                "errors": [] if equal else ["Unscored published gene must be absent in both API and nominate_lookup"]}
    if api_status != 200:
        return {"gene": gene, "matched": False, "errors": [f"API status {api_status} for scored gene"]}
    return {"gene": gene, "status": "scored", **compare_published_rows(api_rows, tool_rows, saved_ranks)}


def _snapshot(directory: Path) -> dict:
    if not directory.exists():
        return {}
    return {str(path.relative_to(directory)): {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
    } for path in sorted(directory.rglob("*")) if path.is_file()}


def check_cached_responses(cases: list[QuestionCase], ask, cache_dir: Path) -> dict:
    before = _snapshot(cache_dir)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(kwargs.get("provider", "unknown"))
        raise RuntimeError("Regression forbids provider inference; prepare historical cache first")

    rows = []
    with patch("backend.services.reasoning_service.generate_answer", side_effect=forbidden), \
            patch("ephys_rag.chain.generate_answer", side_effect=forbidden), \
            patch("ephys_rag.chain.generate_answers_dual", side_effect=forbidden):
        for case in cases:
            try:
                response = ask(case.question, "medgemma")
                rows.append({"question_id": case.question_id, "cached": response.get("cached") is True,
                             "answer_sha256": hashlib.sha256(response.get("answer", "").encode()).hexdigest()})
            except Exception as error:
                rows.append({"question_id": case.question_id, "cached": False,
                             "error": f"{type(error).__name__}: {error}"})
    after = _snapshot(cache_dir)
    return {"passed": bool(rows) and all(row["cached"] for row in rows) and not calls and before == after,
            "hits": sum(row["cached"] for row in rows), "total": len(rows),
            "provider_calls": len(calls), "cache_unchanged": before == after,
            "before": before, "after": after, "questions": rows}


@contextmanager
def api_client(base_url: str | None):
    if base_url:
        import httpx
        with httpx.Client(base_url=base_url.rstrip("/"), timeout=120) as client:
            yield client
    else:
        # Mount the actual production reasoning router without importing legacy
        # annotation/Scanpy services. No dependency overrides or fake responses.
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from backend.api.reasoning import router
        app = FastAPI()
        app.include_router(router, prefix="/api")
        with TestClient(app) as client:
            yield client


def _response(response):
    response.raise_for_status()
    return response.json()


def run_regression(questions: Path, output_dir: Path, base_url: str | None = None) -> dict:
    from ephys_rag import cli, config
    from ephys_rag.retrieve import format_context
    from backend.config import settings

    cases = load_question_cases(questions)
    # Separate construction and collection are essential: never reuse the API
    # engine, response, or API serializer to manufacture the CLI result.
    cli_engine = cli.RAGEngine.from_disk()
    comparisons = []
    with api_client(base_url) as client:
        def ask(question, model):
            return _response(client.post("/api/reasoning/ask", json={"question": question, "model": model}))

        for case in cases:
            args = cli.build_parser().parse_args(["ask", case.question, "--no-llm"])
            cli_raw = cli_engine.ask(args.question, top_k=args.top_k, use_llm=args.llm,
                                     provider=args.provider, dual=args.dual)
            cli_result = {"answer": cli_raw["answer"], "retrieved_context": format_context(cli_raw["hits"]),
                          "traces": [{"tool": trace.tool, "rows": trace.rows, "note": trace.note,
                                      "filters": trace.filters, "n_hits": trace.n_hits}
                                     for trace in cli_raw["traces"]]}
            try:
                api_result = ask(case.question, "none")
                comparison = compare_api_and_cli(case, api_result, cli_result)
                comparison["extractive_answer_equal"] = api_result["answer"] == cli_raw["answer"]
            except Exception as error:
                comparison = {"question_id": case.question_id, "question": case.question,
                              "matched": False, "error": f"{type(error).__name__}: {error}"}
            comparisons.append(comparison)
            print(f"Q{case.question_id}: {'PASS' if comparison['matched'] else 'FAIL'}", flush=True)

        # Use saved published labels independently of the API. Include missing
        # genes explicitly; never silently restrict checks to returned API rows.
        import pandas as pd
        ranks = pd.read_csv(config.ARMS_GENE_RANKS_CSV)
        import yaml
        canonical = yaml.safe_load((config.KNOWLEDGE_DIR / "literature_positives.yaml").read_text(encoding="utf-8"))
        published = sorted({entry["gene"] for entry in canonical["positives"]})
        published_checks = []
        for gene in published:
            try:
                response = client.get(f"/api/genes/{gene}")
                api_rows = response.json()["rows"] if response.status_code == 200 else []
                tool_rows = cli_engine.tools.nominate_lookup(gene=gene, limit=10000).rows
                # None is intentional for absent model scores/ranks.
                saved = ranks[ranks["gene"] == gene]
                gene_ranks = (saved.iloc[0].astype(object).where(saved.iloc[0].notna(), None).to_dict()
                              if not saved.empty else None)
                checked = check_published_gene(gene, response.status_code, api_rows, tool_rows, gene_ranks)
            except Exception as error:
                checked = {"matched": False, "errors": [f"{type(error).__name__}: {error}"]}
            published_checks.append({"gene": gene, **checked})
        published_result = {"passed": len(published_checks) == 33 and all(row["matched"] for row in published_checks),
                            "genes_checked": len(published_checks), "expected_genes": 33,
                            "scored_genes": sum(row.get("status") == "scored" for row in published_checks),
                            "unscored_genes": [row["gene"] for row in published_checks if row.get("status") == "unscored_absence_equal"],
                            "precision": "nominate_lookup context probabilities rounded to 6 decimal places; ranks exact",
                            "arm": "tables_scetm", "genes": published_checks}
        # A remote server cannot be instrumented from this process. Do not claim
        # its cached flag proves absence of provider calls or file mutations.
        if base_url:
            cache_result = {"passed": False, "reason": "Remote cache/provider instrumentation unavailable; rerun in-process for acceptance"}
        else:
            cache_result = check_cached_responses(cases[8:32], ask, settings.EPHYS_RAG_RUNS_DIR)

    matched = sum(row["matched"] for row in comparisons)
    report = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
              "transport": base_url or "FastAPI TestClient /api production reasoning router",
              "questions_file": str(questions), "questions_sha256": hashlib.sha256(questions.read_bytes()).hexdigest(),
              "data_dir": str(config.DATA_DIR), "cli_collection": "separately constructed cli.RAGEngine; CLI ask parser, --no-llm",
              "matched": matched, "total": len(cases), "questions": comparisons,
              "published_33": published_result, "medgemma_cache": cache_result,
              "passed": matched == 44 and published_result["passed"] and cache_result["passed"]}
    write_reports(report, output_dir)
    return report


def write_reports(report: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "portal_regression.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    columns = ["question_id", "question", "matched", "api_tools", "cli_tools", "tools_match",
               "expected_tools_ok", "api_required_ok", "cli_required_ok", "required_terms_match",
               "api_terms", "cli_terms", "extractive_answer_equal", "error"]
    with (output_dir / "portal_regression.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in report["questions"]:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items() if key in columns})
    cache, published = report["medgemma_cache"], report["published_33"]
    different_answers = [f"Q{row['question_id']}" for row in report["questions"]
                         if row.get("extractive_answer_equal") is False]
    answer_diagnostic = (f"Extractive answer text equality: "
                         f"{sum(row.get('extractive_answer_equal') is True for row in report['questions'])}"
                         f"/{len(report['questions'])}. ")
    if different_answers:
        answer_diagnostic += (f"Extractive answer text differs for {', '.join(different_answers)}. "
                              "These differences require separate product review; the 44-question acceptance comparison measures tool and required-term parity, not literal answer equality.")
    lines = ["# Week 6 portal regression", "", f"Generated: {report['generated_at']}", "",
             f"Result: **{'PASS' if report['passed'] else 'FAIL'}**. Q1–Q44: **{report['matched']}/{report['total']}** API/CLI tool-and-term matches.", "",
             "Each match requires equal tool sets, an allowed expected tool combination, and every required evidence group on both sides. Tool order is ignored; alternatives use OR and groups use AND. Echoed questions and lookup filters do not count as evidence, except Q36's explicitly allowed empty T-cell NRXN and Glutamate lookups, which must both execute against the requested cell group.", "",
             f"Transport: {report['transport']}. CLI results come from a separately constructed engine using the CLI ask parser with `--no-llm`. No API result is reused as CLI evidence.", "",
             answer_diagnostic.strip(), "",
             f"Published-33: {'PASS' if published['passed'] else 'FAIL'}; {published['genes_checked']}/33 canonical literature genes checked against `nominate_lookup` in tables_scetm, for both program models and every scored context. {published['scored_genes']} genes have scores; unscored genes ({', '.join(published['unscored_genes']) or 'none'}) must return HTTP 404 and an empty tool result. Context probabilities use the tool's six-decimal precision; context ranks are exact. Gene-level probabilities and ranks are checked separately against saved `arms_gene_ranks.csv`. Absent model scores remain absent.", "",
             f"MedGemma Q9–Q32: {'PASS' if cache['passed'] else 'FAIL'}; hits={cache.get('hits', 'unverified')}/24; provider calls={cache.get('provider_calls', 'unverified')}; cache unchanged={cache.get('cache_unchanged', 'unverified')}.", "",
             "The in-process cache check blocks provider entry points and snapshots every cache file's SHA-256, byte size, and modification time before and after. It neither imports nor primes the runtime cache. Historical answer provenance remains in the existing cache entries; these results establish cache reuse, not a new inference or biological validation.", "",
             "| Question | Match | API tools | Missing API requirements |", "|---|---|---|---|"]
    for row in report["questions"]:
        missing = [term for term, found in row.get("api_terms", {}).items() if not found]
        lines.append(f"| Q{row['question_id']} | {'PASS' if row['matched'] else 'FAIL'} | {', '.join(row.get('api_tools', []))} | {', '.join(missing) or row.get('error', '')} |")
    if not report["passed"]:
        lines += ["", "Failures remain visible in the JSON and CSV. This run does not establish the Week 6 acceptance target."]
    (output_dir / "WEEK6_PORTAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=ROOT / "fall_semester/questions.md")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "fall_semester/turnin/week6_portal")
    parser.add_argument("--base-url")
    args = parser.parse_args()
    report = run_regression(args.questions, args.output_dir, args.base_url)
    print(f"{report['matched']}/{report['total']} matches; published-33={report['published_33']['passed']}; MedGemma cache={report['medgemma_cache']['passed']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    # Match portal runtime defaults before importing the RAG package. Explicit
    # caller configuration always wins; no process-wide changes on module import.
    os.environ.setdefault("EPHYS_RAG_DATA_DIR", str(ROOT / "data/reasoning"))
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "fall_semester/rag_bioreasoning/src"))
    raise SystemExit(main())
