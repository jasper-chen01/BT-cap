from __future__ import annotations

import pandas as pd

from ephys_rag.tools.registry import ToolRegistry, dispatch_tools, infer_query_slots


def _scores() -> pd.DataFrame:
    rows = [
        {
            "gene": "NRXN1",
            "celltype_id": "cycling_tumor",
            "IDH_status": "IDH_Mutant",
            "p_tumor_synaptic": 0.82,
            "p_immune_synapse": 0.08,
            "why_tumor_synaptic": "knn_frac_synaptic=0.4 (+1.2); emb_15=2.1 (+0.8); deg_log2fc=1.7 (+0.3)",
            "why_immune_synapse": "emb_03=0.2 (+0.4); cc_edges=2 (+0.1)",
            "folds_tumor_synaptic": "seed0:fold1;seed1:fold0",
            "folds_immune_synapse": "seed0:fold0;seed1:fold1",
            "cc_edges": 2,
            "ctx_min_n": 136,
            "rule_pass": 1,
            "in_tumor_synaptic": 1,
            "in_immune_synapse": 0,
        },
        {
            "gene": "NCL",
            "celltype_id": "cycling_tumor",
            "IDH_status": "IDH_Mutant",
            "p_tumor_synaptic": 0.72,
            "p_immune_synapse": 0.12,
            "why_tumor_synaptic": "deg_log2fc=0.8 (+0.7); cc_edges=18 (+0.4); emb_05=1.1 (+0.2)",
            "why_immune_synapse": "emb_03=0.1 (+0.2)",
            "folds_tumor_synaptic": "seed0:fold0;seed1:fold1",
            "folds_immune_synapse": "seed0:fold1;seed1:fold0",
            "cc_edges": 18,
            "ctx_min_n": 136,
            "rule_pass": 1,
            "in_tumor_synaptic": 0,
            "in_immune_synapse": 0,
        },
        {
            "gene": "CD28",
            "celltype_id": "T cell",
            "IDH_status": "IDH_WT",
            "p_tumor_synaptic": float("nan"),
            "p_immune_synapse": 0.91,
            "why_tumor_synaptic": "",
            "why_immune_synapse": "emb_03=2.2 (+1.3); knn_frac_immune=0.3 (+0.7); cc_edges=1 (+0.1)",
            "folds_tumor_synaptic": float("nan"),
            "folds_immune_synapse": "seed0:fold1;seed1:fold0",
            "cc_edges": 1,
            "ctx_min_n": 392,
            "rule_pass": 1,
            "in_tumor_synaptic": 0,
            "in_immune_synapse": 1,
        },
        {
            "gene": "LCP2",
            "celltype_id": "T cell",
            "IDH_status": "IDH_WT",
            "p_tumor_synaptic": float("nan"),
            "p_immune_synapse": 0.75,
            "why_tumor_synaptic": "",
            "why_immune_synapse": "emb_10=1.2 (+0.9); deg_log2fc=0.6 (+0.2)",
            "folds_tumor_synaptic": "",
            "folds_immune_synapse": "seed0:fold0;seed1:fold1",
            "cc_edges": 0,
            "ctx_min_n": 392,
            "rule_pass": 0,
            "in_tumor_synaptic": 0,
            "in_immune_synapse": 1,
        },
    ]
    return pd.DataFrame(rows)


def _registry() -> ToolRegistry:
    return ToolRegistry(
        interactions=[],
        degs_idh=[],
        degs_pooled=[],
        nominee_scores=_scores(),
    )


def test_nominate_lookup_returns_context_scores_evidence_and_fold_provenance():
    """Catches omitted filters, explanations, evidence fields, or scoring-fold provenance."""
    result = _registry().nominate_lookup(
        gene="NRXN1", celltype="cycling", idh="IDH-mutant", order_by="tumor"
    )

    assert result.tool == "nominate_lookup"
    assert result.n_hits == 1
    row = result.rows[0]
    assert row["gene"] == "NRXN1"
    assert row["celltype_id"] == "cycling_tumor"
    assert row["IDH_status"] == "IDH_Mutant"
    assert row["p_tumor_synaptic"] == 0.82
    assert row["p_immune_synapse"] == 0.08
    assert row["top3_tumor_features"] == ["knn_frac_synaptic", "emb_15", "deg_log2fc"]
    assert row["tumor_feature_family"] == "mixed"
    assert row["in_cellchat"] is True
    assert row["min_n"] == 136
    assert row["passes_three_table"] is True
    assert row["tumor_scoring_folds"] == "seed0:fold1;seed1:fold0"
    assert row["tumor_rank_in_context"] == 1


def test_nominate_lookup_can_return_top_candidate_and_exclude_syngo():
    """Catches a gene-required implementation that cannot answer Q41/Q42."""
    top_immune = _registry().nominate_lookup(celltype="T cell", order_by="immune", limit=1)
    assert top_immune.rows[0]["gene"] == "CD28"
    assert top_immune.rows[0]["immune_rank_in_context"] == 1
    assert top_immune.rows[0]["tumor_scoring_folds"] is None

    not_syngo = _registry().nominate_lookup(order_by="tumor", exclude_syngo=True, limit=1)
    assert not_syngo.rows[0]["gene"] == "NCL"
    assert not_syngo.rows[0]["in_syngo"] is False


def test_router_uses_nominate_lookup_for_rank_and_three_table_questions():
    """Catches missing score/rank routing and the Q44 three-table wording gap."""
    ranked = dispatch_tools("Which gene ranks highest for immune synapse in T cells?", _registry())
    assert any(trace.tool == "nominate_lookup" for trace in ranked)
    nominate = next(trace for trace in ranked if trace.tool == "nominate_lookup")
    assert nominate.rows[0]["gene"] == "CD28"

    checked = dispatch_tools(
        "Does NCL pass the CellChat + DEG + cell-count checks?", _registry()
    )
    assert any(trace.tool == "nominate_lookup" for trace in checked)


def test_opc_like_wording_selects_the_specific_opc_like_tumor_identity():
    """Catches mixing the OPC-like group with the OPC/GABA/AC mixed identity in Q43."""
    slots = infer_query_slots(
        "Where does GRIA2 rank in OPC-like tumor, and does scETM drive it?"
    )
    assert slots["celltype"] == "OPC_GABA_like_tumor"
