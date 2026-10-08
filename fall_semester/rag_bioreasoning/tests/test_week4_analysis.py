from __future__ import annotations

from types import SimpleNamespace

import pandas as pd


def _metrics() -> dict:
    return {
        "sets": {
            "tumor_synaptic": {
                "set_genes_in_universe": 120,
                "n_positive_rows": 240,
                "prevalence_rows": 0.10,
                "row_level": {
                    "lightgbm_oof": {
                        "auroc": 0.75,
                        "auprc": 0.40,
                        "precision_at_25": 0.60,
                    },
                    "cc_max_prob": {
                        "auroc": 0.60,
                        "auprc": 0.20,
                        "precision_at_25": 0.76,
                    },
                    "rule_pass": {
                        "auroc": 0.55,
                        "auprc": 0.25,
                        "precision_at_25": 0.20,
                    },
                },
            }
        }
    }


def _scores() -> pd.DataFrame:
    rows = [
        ("A", "cycling_tumor", "IDH_Mutant", 0.90, 0.70, 0, 0, 0, 0, 1),
        ("A", "cycling_tumor", "IDH_WT", 0.80, 0.65, 0, 0, 0, 0, 1),
        ("B", "cycling_tumor", "IDH_Mutant", 0.99, 0.99, 1, 0, 0, 0, 1),
        ("C", "T cell", "IDH_Mutant", 0.98, 0.98, 0, 1, 0, 0, 1),
        ("D", "T cell", "IDH_WT", 0.97, 0.97, 0, 0, 1, 0, 1),
        ("E", "cycling_tumor", "IDH_WT", 0.96, 0.96, 0, 0, 0, 1, 1),
        ("F", "cycling_tumor", "IDH_WT", 0.95, 0.95, 0, 0, 0, 0, 0),
        ("G", "TAM1/TAM2/microglia", "IDH_WT", 0.85, 0.90, 0, 0, 0, 0, 1),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "gene",
            "celltype_id",
            "IDH_status",
            "p_tumor_synaptic",
            "p_immune_synapse",
            "in_tumor_synaptic",
            "in_immune_synapse",
            "in_published_33",
            "housekeeping",
            "rule_pass",
        ],
    )
    frame["deg_log2fc"] = 1.5
    frame["deg_neglog10_padj"] = 10.0
    frame["cc_edges"] = 3
    frame["cc_max_prob"] = 0.2
    frame["ctx_min_n"] = 100
    frame["why_tumor_synaptic"] = "deg_log2fc"
    frame["why_immune_synapse"] = "cc_edges"
    return frame


def test_baseline_comparison_reports_if_lightgbm_beats_both_baselines():
    """A wrong scorer mapping or win rule must change the reported decision."""
    from ephys_rag.week4_analysis import build_baseline_comparison

    comparison = build_baseline_comparison(_metrics())

    auroc = comparison.loc[comparison["metric"] == "auroc"].iloc[0]
    assert auroc["lightgbm"] == 0.75
    assert auroc["cellchat_probability"] == 0.60
    assert auroc["three_table_rule"] == 0.55
    assert auroc["delta_vs_best_baseline"] == 0.15
    assert bool(auroc["lightgbm_beats_both"])

    precision = comparison.loc[comparison["metric"] == "precision_at_25"].iloc[0]
    assert precision["best_method"] == "cellchat_probability"
    assert not bool(precision["lightgbm_beats_both"])


def test_novel_nominees_exclude_public_published_and_housekeeping_genes():
    """No public-set, published-33, or housekeeping gene may enter the nominee list."""
    from ephys_rag.week4_analysis import build_novel_nominees

    strict = build_novel_nominees(_scores(), top_per_program=10, require_three_table=True)
    tumor = strict.loc[strict["program"] == "tumor_synaptic"]

    assert tumor["gene"].tolist() == ["A", "G"]
    assert tumor["nominee_rank"].tolist() == [1, 2]
    assert tumor["gene"].is_unique
    assert (tumor[["in_tumor_synaptic", "in_immune_synapse", "in_published_33"]] == 0).all().all()
    assert (tumor["passes_three_table_rule"] == 1).all()

    relaxed = build_novel_nominees(_scores(), top_per_program=10, require_three_table=False)
    relaxed_tumor = relaxed.loc[relaxed["program"] == "tumor_synaptic"]
    assert relaxed_tumor["gene"].tolist() == ["F", "A", "G"]


def test_week4_writer_creates_focused_csvs_and_summary(tmp_path):
    """The Week 4 rerun must leave directly shareable task 6 and task 7 files."""
    from ephys_rag.week4_analysis import write_week4_focus_outputs

    result = SimpleNamespace(metrics=_metrics(), scores=_scores())
    paths = write_week4_focus_outputs(result, tmp_path, top_per_program=10)

    assert paths["baseline_csv"].exists()
    assert paths["nominees_csv"].exists()
    assert paths["score_only_csv"].exists()
    assert paths["summary_md"].exists()
    assert "Does LightGBM beat the baselines?" in paths["summary_md"].read_text(encoding="utf-8")
    assert "Novel nominees" in paths["summary_md"].read_text(encoding="utf-8")
