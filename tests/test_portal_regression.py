"""Catch incomplete question parsing and false-positive portal evidence checks."""
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.portal_regression import (
    QuestionCase, check_cached_responses, compare_api_and_cli,
    compare_published_rows, load_question_cases, check_published_gene, write_reports,
)

ROOT = Path(__file__).resolve().parents[1]


def test_question_file_preserves_all_44_questions_and_requirements():
    cases = load_question_cases(ROOT / "fall_semester/questions.md")
    assert [case.question_id for case in cases] == list(range(1, 45))
    assert cases[0].expected_tools == (("exclusive_pathway_lookup",),)
    assert cases[0].required_terms == (("Glutamate",), ("MK",), ("CNTN",), ("NOTCH",))
    assert cases[20].expected_tools == (("deg_lookup",),)
    assert cases[20].required_terms == (("NRXN1",), ("cycling",), ("IDH",))
    assert cases[38].expected_tools == (("nominate_lookup",),)
    assert cases[38].required_terms == (("NRXN1",), ("p_tumor_synaptic",),
                                       ("feature name",), ("scoring fold",))
    assert "**" not in cases[0].question


@pytest.mark.parametrize("replacement", ["45.", "43.", ""])
def test_parser_rejects_extra_duplicate_or_missing_question(tmp_path, replacement):
    source = (ROOT / "fall_semester/questions.md").read_text(encoding="utf-8")
    path = tmp_path / "questions.md"
    path.write_text(source.replace("44.", replacement), encoding="utf-8")
    with pytest.raises(ValueError, match="Q1.*Q44"):
        load_question_cases(path)


def test_comparison_ignores_tool_and_row_order_but_requires_real_terms():
    case = QuestionCase(21, "Is NRXN1 supported?", (("deg_lookup",),), (("NRXN1",), ("cycling",)))
    api = {"traces": [{"tool": "deg_lookup", "rows": [{"gene": "NRXN1", "cell": "cycling"}]},
                      {"tool": "cellchat_lookup", "rows": []}]}
    cli = {"traces": list(reversed(api["traces"]))}
    assert compare_api_and_cli(case, api, cli)["matched"]
    cli = deepcopy(cli)
    cli["traces"][1]["rows"][0]["gene"] = "OTHER"
    assert not compare_api_and_cli(case, api, cli)["matched"]


def test_missing_requirements_on_both_sides_are_not_a_match():
    case = QuestionCase(1, "question", (("cellchat_lookup",),), (("NRXN1",),))
    empty = {"answer": "question NRXN1", "traces": [
        {"tool": "cellchat_lookup", "filters": {"gene": "NRXN1"}, "rows": []}]}
    result = compare_api_and_cli(case, empty, empty)
    assert result["tools_match"]
    assert not result["matched"]
    assert not result["api_required_ok"]


def test_alternatives_are_or_but_separate_groups_are_and():
    case = QuestionCase(4, "question", (), (("GALECTIN", "Glutamate"), ("T cell",)))
    result = {"traces": [{"tool": "pathway_filter", "rows": [
        {"pathway": "Glutamate", "cell": "T_cell"}]}]}
    assert compare_api_and_cli(case, result, result)["matched"]
    result["traces"][0]["rows"][0]["cell"] = "tumor"
    assert not compare_api_and_cli(case, result, result)["matched"]


def test_oxford_comma_or_list_does_not_require_every_receptor():
    case = load_question_cases(ROOT / "fall_semester/questions.md")[6]
    assert case.required_terms == (("GRIA2", "GRIA3", "GRIA4", "GRIK2"),)
    result = {"traces": [{"tool": "cellchat_lookup", "rows": [{"receptor": "GRIK2"}]}]}
    assert compare_api_and_cli(case, result, result)["matched"]
    result["traces"][0]["rows"] = []
    assert not compare_api_and_cli(case, result, result)["matched"]


def test_q36_allows_explicit_empty_t_cell_queries_but_other_questions_do_not():
    cases = load_question_cases(ROOT / "fall_semester/questions.md")
    result = {"traces": [
        {"tool": "cellchat_lookup", "filters": {"celltype": "T cell", "pathway": "Glutamate"},
         "n_hits": 0, "rows": []},
        {"tool": "cellchat_lookup", "filters": {"celltype": "T cell", "gene": "NRXN"},
         "n_hits": 0, "rows": []},
    ]}
    assert compare_api_and_cli(cases[35], result, result)["matched"]
    assert not compare_api_and_cli(cases[27], result, result)["matched"]
    wrong = deepcopy(result)
    wrong["traces"][0]["filters"]["celltype"] = "tumor"
    assert not compare_api_and_cli(cases[35], wrong, wrong)["matched"]


def test_tam_microglia_requirement_accepts_canonical_table_identity():
    case = QuestionCase(40, "question", (), (("TAM/microglia",),))
    result = {"traces": [{"tool": "nominate_lookup", "rows": [
        {"celltype_id": "TAM1/TAM2/microglia"}]}]}
    assert compare_api_and_cli(case, result, result)["matched"]
    result["traces"][0]["rows"][0]["celltype_id"] = "T cell"
    assert not compare_api_and_cli(case, result, result)["matched"]


def test_q19_cannot_pass_from_question_or_empty_nrxn_filter():
    case = load_question_cases(ROOT / "fall_semester/questions.md")[18]
    result = {"answer": case.question, "traces": [
        {"tool": "cellchat_lookup", "filters": {"gene": "NRXN1"}, "rows": []},
        {"tool": "pathway_filter", "rows": [{"pathway": "Glutamate"}]},
    ]}
    assert not compare_api_and_cli(case, result, result)["matched"]


def test_q27_requires_six_cells_not_a_decimal_probability_containing_six():
    case = load_question_cases(ROOT / "fall_semester/questions.md")[26]
    result = {"traces": [{"tool": "count_lookup", "rows": [
        {"cell": "MES", "n_cells": 60, "prob": .6}]}]}
    assert not compare_api_and_cli(case, result, result)["matched"]
    result["traces"][0]["rows"][0]["n_cells"] = 6
    assert compare_api_and_cli(case, result, result)["matched"]


def test_q30_requires_the_named_gene_in_both_tables_not_unrelated_deg_rows():
    case = load_question_cases(ROOT / "fall_semester/questions.md")[29]
    result = {"traces": [
        {"tool": "cellchat_lookup", "rows": [{"ligand": "NRXN1"}]},
        {"tool": "deg_lookup", "rows": [{"gene": "OTHER"}]},
    ]}
    assert not compare_api_and_cli(case, result, result)["matched"]
    result["traces"][1]["rows"][0]["gene"] = "NRXN1"
    assert compare_api_and_cli(case, result, result)["matched"]


def test_q39_requires_tumor_features_and_fold_not_only_an_immune_explanation():
    case = load_question_cases(ROOT / "fall_semester/questions.md")[38]
    row = {"gene": "NRXN1", "p_tumor_synaptic": .8, "top3_tumor_features": [],
           "tumor_scoring_folds": None, "top3_immune_features": ["emb_03"],
           "immune_scoring_folds": "seed0:fold1"}
    result = {"traces": [{"tool": "nominate_lookup", "rows": [row]}]}
    assert not compare_api_and_cli(case, result, result)["matched"]
    row.update(top3_tumor_features=["emb_07"], tumor_scoring_folds="seed0:fold2")
    assert compare_api_and_cli(case, result, result)["matched"]


def test_published_probability_and_context_rank_comparison_catches_missing_and_changed_rows():
    tool = [{"gene": "NRXN1", "celltype_id": "cycling_tumor", "IDH_status": "IDH_Mutant",
             "p_tumor_synaptic": .823457, "p_immune_synapse": None,
             "tumor_rank_in_context": 3, "immune_rank_in_context": None}]
    api = [{**tool[0], "model": "tumor_synaptic", "arm": "tables_scetm", "score": .82345678,
            "rank_in_context": 3, "gene_rank": 7, "gene_score": .9}]
    ranks = {"p_tumor_synaptic__tables_scetm": .9,
             "gene_rank_tumor_synaptic__tables_scetm": 7}
    assert compare_published_rows(api, tool, ranks)["matched"]
    for field, value in [("score", .7), ("p_tumor_synaptic", .7),
                         ("gene_rank", 8), ("rank_in_context", 4), ("tumor_rank_in_context", 4)]:
        changed = [{**api[0], field: value}]
        assert not compare_published_rows(changed, tool, ranks)["matched"]
    assert not compare_published_rows([], tool, ranks)["matched"]
    assert not compare_published_rows([], [], ranks)["matched"]


def test_unscored_published_gene_requires_absence_from_both_api_and_tool():
    result = check_published_gene("AXL", 404, [], [], None)
    assert result["matched"]
    assert result["status"] == "unscored_absence_equal"
    assert not check_published_gene("AXL", 200, [{"score": 0}], [], None)["matched"]
    assert not check_published_gene("AXL", 404, [], [{"gene": "AXL"}], None)["matched"]
    assert not check_published_gene("AXL", 503, [], [], None)["matched"]


def test_markdown_report_exposes_answer_differences_even_when_tool_term_contract_passes(tmp_path):
    report = {"generated_at": "2026-10-18", "passed": True, "matched": 1, "total": 1,
              "transport": "TestClient", "questions": [
                  {"question_id": 7, "matched": True, "api_tools": ["cellchat_lookup"],
                   "api_terms": {"GRIA2": True}, "extractive_answer_equal": False}],
              "published_33": {"passed": True, "genes_checked": 33, "scored_genes": 32,
                               "unscored_genes": ["AXL"]},
              "medgemma_cache": {"passed": True, "hits": 24, "provider_calls": 0,
                                  "cache_unchanged": True}}
    write_reports(report, tmp_path)
    text = (tmp_path / "WEEK6_PORTAL_REPORT.md").read_text(encoding="utf-8")
    assert "Extractive answer text differs for Q7" in text


def test_cache_check_detects_writes_even_when_cached_flag_is_true(tmp_path):
    path = tmp_path / "entry.json"
    path.write_text("original", encoding="utf-8")
    case = QuestionCase(9, "question", (), ())

    def ask(question, model):
        path.write_text("changed", encoding="utf-8")
        return {"cached": True, "answer": "saved"}

    result = check_cached_responses([case], ask, tmp_path)
    assert not result["passed"]
    assert not result["cache_unchanged"]


def test_cache_check_forbids_provider_and_detects_a_false_cached_flag(tmp_path):
    from backend.services import reasoning_service

    case = QuestionCase(9, "question", (), ())

    def ask(question, model):
        return {"answer": reasoning_service.generate_answer(question, "context", provider=model),
                "cached": True}

    result = check_cached_responses([case], ask, tmp_path)
    assert not result["passed"]
    assert result["provider_calls"] == 1
    result = check_cached_responses([case], lambda q, m: {"cached": False}, tmp_path)
    assert not result["passed"]


def test_cached_q9_to_q32_use_saved_answers_without_provider_or_cache_mutation(tmp_path):
    from backend.services.reasoning_service import (
        PortalAnswerCache, ReasoningService, get_rag_engine, import_week2_cache,
    )
    cache = PortalAnswerCache(tmp_path)
    engine = get_rag_engine()
    assert import_week2_cache(ROOT / "fall_semester/turnin/week2_traces", cache, engine) == 24
    service = ReasoningService(engine, cache)
    cases = load_question_cases(ROOT / "fall_semester/questions.md")[8:32]
    result = check_cached_responses(cases, service.ask, tmp_path)
    assert result["passed"]
    assert result["hits"] == 24
    assert result["provider_calls"] == 0
    assert result["cache_unchanged"]
