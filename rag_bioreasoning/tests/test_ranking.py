from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def ranking_api():
    try:
        from ephys_rag.ranking import (
            FEATURE_COLUMNS,
            build_candidate_features_from_frames,
            candidate_genes,
            evaluate_ranking,
            gene_group_splits,
            load_seed_labels,
            parse_complex_genes,
            write_ranking_outputs,
        )
    except ModuleNotFoundError:
        pytest.fail("Glioma ranking module is not implemented")
    return (
        FEATURE_COLUMNS,
        build_candidate_features_from_frames,
        candidate_genes,
        evaluate_ranking,
        gene_group_splits,
        load_seed_labels,
        parse_complex_genes,
        write_ranking_outputs,
    )


def tiny_frames():
    interactions = pd.DataFrame(
        [
            {
                "source": "cycling_tumor / Ephys_2",
                "target": "T cell / Ephys_1",
                "ligand": "Glu-SLC1A1_GLS",
                "receptor": "CD74_CXCR4",
                "interaction_name": "Glutamate_test",
                "pathway": "Glutamate",
                "prob": 0.2,
                "pval": 0.0,
            }
        ]
    )
    degs = pd.DataFrame(
        [
            {
                "IDH_status": "IDH_Mutant",
                "compartment": "tumor",
                "celltype_id": "cycling_tumor",
                "gene": "SLC1A1",
                "direction": "Ephys_2_high",
                "n_Ephys_1": 270,
                "n_Ephys_2": 136,
                "pct_Ephys_2": 33.5,
                "p_val": 1e-8,
                "avg_log2FC": 2.0,
                "pct.1": 0.4,
                "pct.2": 0.1,
                "p_val_adj": 1e-5,
            },
            {
                "IDH_status": "IDH_WT",
                "compartment": "T_cell",
                "celltype_id": "T cell",
                "gene": "CD74",
                "direction": "Ephys_1_high",
                "n_Ephys_1": 100,
                "n_Ephys_2": 80,
                "pct_Ephys_2": 44.4,
                "p_val": 1e-6,
                "avg_log2FC": -1.5,
                "pct.1": 0.2,
                "pct.2": 0.5,
                "p_val_adj": 0.01,
            },
        ]
    )
    counts = pd.DataFrame(
        [
            {"celltype_id": "cycling_tumor", "ephys_cluster_id": "Ephys_1", "n_cells": 270},
            {"celltype_id": "cycling_tumor", "ephys_cluster_id": "Ephys_2", "n_cells": 136},
            {"celltype_id": "T cell", "ephys_cluster_id": "Ephys_1", "n_cells": 100},
            {"celltype_id": "T cell", "ephys_cluster_id": "Ephys_2", "n_cells": 80},
        ]
    )
    return interactions, degs, counts


def test_complex_parser_produces_biological_gene_symbols():
    """Leaving CellChat complexes unsplit would omit candidate genes."""
    *_, parse_complex_genes, _ = ranking_api()

    assert parse_complex_genes("Glu-SLC1A1_GLS") == ("SLC1A1", "GLS")
    assert parse_complex_genes("CD74_CXCR4") == ("CD74", "CXCR4")
    assert parse_complex_genes("TGFbR1_R2") == ("TGFBR1", "TGFBR2")


def test_candidate_matrix_crosses_genes_contexts_and_has_39_numeric_features():
    """Dropping a context or changing the fixed feature schema breaks comparability."""
    (
        feature_columns,
        build_from_frames,
        candidate_genes,
        *_,
    ) = ranking_api()
    interactions, degs, counts = tiny_frames()

    genes = candidate_genes(interactions)
    frame = build_from_frames(interactions, degs, counts)

    assert genes == ["CD74", "CXCR4", "GLS", "SLC1A1"]
    assert len(feature_columns) == 39
    assert frame.shape[0] == 8
    assert set(feature_columns).issubset(frame.columns)
    assert all(pd.api.types.is_numeric_dtype(frame[column]) for column in feature_columns)
    row = frame[
        (frame["gene"] == "SLC1A1")
        & (frame["IDH_status"] == "IDH_Mutant")
        & (frame["celltype_id"] == "cycling_tumor")
    ].iloc[0]
    assert row["sender_edge_count"] == 1
    assert row["ephys2_edge_count"] == 1
    assert row["deg_present"] == 1
    assert row["n_ephys1"] == 270


def test_seed_labels_are_33_compartment_aware_cellchat_genes():
    """The prototype label set must stay explicit rather than hidden in code."""
    _, _, _, _, _, load_seed_labels, _, _ = ranking_api()
    labels_path = Path(__file__).parents[1] / "evaluation" / "glioma_seed_labels.csv"
    labels = load_seed_labels(labels_path)

    assert len(labels) == 33
    assert set(labels["compartment"]) == {"tumor", "TAM_microglia", "T_cell"}
    assert labels.duplicated(["gene", "compartment"]).sum() == 0


def test_grouped_splits_never_train_on_a_validation_gene():
    """Gene overlap between train and validation would inflate reported metrics."""
    _, _, _, _, gene_group_splits, _, _, _ = ranking_api()
    frame = pd.DataFrame(
        {
            "gene": [gene for gene in "ABCDEFGH" for _ in range(2)],
            "label": [0, 1] * 8,
        }
    )

    splits = list(gene_group_splits(frame, n_splits=5))

    assert len(splits) == 5
    for train_idx, valid_idx in splits:
        assert set(frame.iloc[train_idx]["gene"]).isdisjoint(
            set(frame.iloc[valid_idx]["gene"])
        )


def test_ranking_evaluation_returns_grouped_oof_predictions_and_baselines(tmp_path):
    """The scored output must be out-of-fold and include both stated baselines."""
    (
        feature_columns,
        _,
        _,
        evaluate_ranking,
        _,
        _,
        _,
        write_ranking_outputs,
    ) = ranking_api()
    rows = []
    for gene_index in range(12):
        for context_index in range(2):
            row = {
                "gene": f"G{gene_index:02d}",
                "IDH_status": f"IDH_{context_index}",
                "compartment": "tumor",
                "celltype_id": "cycling_tumor",
                "label": int(gene_index in {0, 3, 6, 9}),
            }
            row.update(
                {
                    column: float((gene_index + context_index + offset) % 7)
                    for offset, column in enumerate(feature_columns)
                }
            )
            row["max_prob"] = gene_index / 12
            row["edge_count"] = float(gene_index % 2)
            row["deg_present"] = float(gene_index % 3 == 0)
            row["low_count_flag"] = float(gene_index % 5 == 0)
            rows.append(row)
    frame = pd.DataFrame(rows)

    result = evaluate_ranking(frame, seeds=(11, 23), n_splits=3, top_k=5)

    assert set(result.metrics["scorer"]) == {
        "LightGBM",
        "CellChat probability alone",
        "3-table rule alone",
    }
    assert len(result.predictions) == len(frame)
    assert result.predictions["oof_score"].between(0, 1).all()
    assert result.predictions["rank"].notna().all()
    assert result.predictions["gene"].nunique() == 12
    paths = write_ranking_outputs(result, frame, tmp_path)
    assert {"metrics", "ranked_candidates", "features", "model_card"}.issubset(paths)
    assert all(path.exists() for path in paths.values())
