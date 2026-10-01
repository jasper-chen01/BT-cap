"""Leakage-aware gene ranking features for the Week 3 prototype.

The ranked unit is a CellChat ligand/receptor gene in one biological context
(IDH status x transcriptomic cell type).  Features are derived only from the
three evidence tables already used by the RAG tools.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import math
import re
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score


FEATURE_COLUMNS = [
    "is_ligand",
    "is_receptor",
    "edge_count",
    "sender_edge_count",
    "receiver_edge_count",
    "self_edge_count",
    "ephys1_edge_count",
    "ephys2_edge_count",
    "ephys_edge_delta",
    "mean_prob",
    "max_prob",
    "sum_prob",
    "sender_mean_prob",
    "receiver_mean_prob",
    "ephys1_mean_prob",
    "ephys2_mean_prob",
    "ephys_prob_delta",
    "pathway_count",
    "partner_gene_count",
    "source_celltype_count",
    "target_celltype_count",
    "same_ephys_edge_count",
    "cross_ephys_edge_count",
    "deg_present",
    "avg_log2FC",
    "abs_log2FC",
    "ephys2_high",
    "ephys1_high",
    "neg_log10_padj",
    "pct_ephys2",
    "pct_1",
    "pct_2",
    "pct_delta",
    "idh_direction_agreement",
    "other_idh_abs_log2fc",
    "n_ephys1",
    "n_ephys2",
    "min_group_size",
    "low_count_flag",
]


_SPECIAL_COMPLEXES = {
    "TGFBR1_R2": ("TGFBR1", "TGFBR2"),
}


def parse_complex_genes(value: object) -> tuple[str, ...]:
    """Return biological gene symbols from a CellChat ligand/receptor label."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ()
    label = str(value).strip()
    if not label:
        return ()
    upper = label.upper()
    if upper in _SPECIAL_COMPLEXES:
        return _SPECIAL_COMPLEXES[upper]
    # CellChat uses descriptive prefixes for neurotransmitter-producing
    # complexes, e.g. Glu-SLC1A1_GLS and PGE2-PTGES2/3.
    if "-" in label:
        label = label.split("-", 1)[1]
    genes: list[str] = []
    for token in re.split(r"[_/]", label):
        token = token.strip().upper()
        if not token:
            continue
        if token.isdigit() and genes:
            prefix = re.sub(r"\d+$", "", genes[-1])
            token = f"{prefix}{token}"
        genes.append(token)
    return tuple(dict.fromkeys(genes))


def candidate_genes(interactions: pd.DataFrame) -> list[str]:
    """Return the sorted union of genes represented in CellChat pairs."""
    genes: set[str] = set()
    for column in ("ligand", "receptor"):
        for value in interactions[column].dropna():
            genes.update(parse_complex_genes(value))
    return sorted(genes)


def load_seed_labels(path: str | Path) -> pd.DataFrame:
    """Load and validate the explicit, compartment-aware prototype labels."""
    labels = pd.read_csv(path, dtype=str)
    required = {"gene", "compartment"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Seed label file is missing columns: {sorted(missing)}")
    labels = labels.copy()
    labels["gene"] = labels["gene"].str.upper().str.strip()
    labels["compartment"] = labels["compartment"].str.strip()
    if labels.duplicated(["gene", "compartment"]).any():
        raise ValueError("Seed labels must be unique by gene and compartment")
    return labels


def _identity_parts(series: pd.Series) -> pd.DataFrame:
    parts = series.fillna("").str.rsplit(" / ", n=1, expand=True)
    if parts.shape[1] == 1:
        parts[1] = ""
    parts.columns = ["celltype", "ephys"]
    return parts


def _safe_mean(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else 0.0


def _first_numeric(row: pd.Series | None, name: str, default: float = 0.0) -> float:
    if row is None or name not in row or pd.isna(row[name]):
        return default
    return float(row[name])


def _cellchat_features(interactions: pd.DataFrame, gene: str, celltype: str) -> dict[str, float]:
    work = interactions.copy()
    source = _identity_parts(work["source"])
    target = _identity_parts(work["target"])
    work["_source_celltype"] = source["celltype"]
    work["_source_ephys"] = source["ephys"]
    work["_target_celltype"] = target["celltype"]
    work["_target_ephys"] = target["ephys"]
    work["_ligand_genes"] = work["ligand"].map(parse_complex_genes)
    work["_receptor_genes"] = work["receptor"].map(parse_complex_genes)

    ligand_mask = work["_ligand_genes"].map(lambda values: gene in values)
    receptor_mask = work["_receptor_genes"].map(lambda values: gene in values)
    sender_mask = ligand_mask & work["_source_celltype"].eq(celltype)
    receiver_mask = receptor_mask & work["_target_celltype"].eq(celltype)
    relevant = work[sender_mask | receiver_mask]
    sender = work[sender_mask]
    receiver = work[receiver_mask]
    ephys1 = relevant[
        ((sender_mask.loc[relevant.index]) & relevant["_source_ephys"].eq("Ephys_1"))
        | ((receiver_mask.loc[relevant.index]) & relevant["_target_ephys"].eq("Ephys_1"))
    ]
    ephys2 = relevant[
        ((sender_mask.loc[relevant.index]) & relevant["_source_ephys"].eq("Ephys_2"))
        | ((receiver_mask.loc[relevant.index]) & relevant["_target_ephys"].eq("Ephys_2"))
    ]

    partner_genes: set[str] = set()
    for _, edge in relevant.iterrows():
        if gene in edge["_ligand_genes"]:
            partner_genes.update(edge["_receptor_genes"])
        if gene in edge["_receptor_genes"]:
            partner_genes.update(edge["_ligand_genes"])
    partner_genes.discard(gene)

    same_ephys = relevant["_source_ephys"].eq(relevant["_target_ephys"])
    known_ephys = relevant["_source_ephys"].str.startswith("Ephys_") & relevant[
        "_target_ephys"
    ].str.startswith("Ephys_")
    return {
        "is_ligand": float(ligand_mask.any()),
        "is_receptor": float(receptor_mask.any()),
        "edge_count": float(len(relevant)),
        "sender_edge_count": float(len(sender)),
        "receiver_edge_count": float(len(receiver)),
        "self_edge_count": float(
            ((relevant["_source_celltype"] == celltype) & (relevant["_target_celltype"] == celltype)).sum()
        ),
        "ephys1_edge_count": float(len(ephys1)),
        "ephys2_edge_count": float(len(ephys2)),
        "ephys_edge_delta": float(len(ephys2) - len(ephys1)),
        "mean_prob": _safe_mean(relevant["prob"]),
        "max_prob": float(relevant["prob"].max()) if len(relevant) else 0.0,
        "sum_prob": float(relevant["prob"].sum()) if len(relevant) else 0.0,
        "sender_mean_prob": _safe_mean(sender["prob"]),
        "receiver_mean_prob": _safe_mean(receiver["prob"]),
        "ephys1_mean_prob": _safe_mean(ephys1["prob"]),
        "ephys2_mean_prob": _safe_mean(ephys2["prob"]),
        "ephys_prob_delta": _safe_mean(ephys2["prob"]) - _safe_mean(ephys1["prob"]),
        "pathway_count": float(relevant["pathway"].nunique()),
        "partner_gene_count": float(len(partner_genes)),
        "source_celltype_count": float(relevant["_source_celltype"].nunique()),
        "target_celltype_count": float(relevant["_target_celltype"].nunique()),
        "same_ephys_edge_count": float((same_ephys & known_ephys).sum()),
        "cross_ephys_edge_count": float(((~same_ephys) & known_ephys).sum()),
    }


def _deg_features(
    degs: pd.DataFrame, gene: str, idh_status: str, celltype: str
) -> dict[str, float]:
    selected = degs[
        degs["gene"].astype(str).str.upper().eq(gene)
        & degs["IDH_status"].eq(idh_status)
        & degs["celltype_id"].eq(celltype)
    ]
    row = selected.iloc[0] if len(selected) else None
    other = degs[
        degs["gene"].astype(str).str.upper().eq(gene)
        & ~degs["IDH_status"].eq(idh_status)
        & degs["celltype_id"].eq(celltype)
    ]
    other_row = other.iloc[0] if len(other) else None
    direction = str(row["direction"]) if row is not None else ""
    other_direction = str(other_row["direction"]) if other_row is not None else ""
    padj = max(_first_numeric(row, "p_val_adj", 1.0), np.finfo(float).tiny)
    pct1 = _first_numeric(row, "pct.1")
    pct2 = _first_numeric(row, "pct.2")
    return {
        "deg_present": float(row is not None),
        "avg_log2FC": _first_numeric(row, "avg_log2FC"),
        "abs_log2FC": abs(_first_numeric(row, "avg_log2FC")),
        "ephys2_high": float(direction == "Ephys_2_high"),
        "ephys1_high": float(direction == "Ephys_1_high"),
        "neg_log10_padj": float(-math.log10(padj)) if row is not None else 0.0,
        "pct_ephys2": _first_numeric(row, "pct_Ephys_2"),
        "pct_1": pct1,
        "pct_2": pct2,
        "pct_delta": pct2 - pct1,
        "idh_direction_agreement": float(bool(direction) and direction == other_direction),
        "other_idh_abs_log2fc": abs(_first_numeric(other_row, "avg_log2FC")),
    }


def _count_features(counts: pd.DataFrame, celltype: str) -> dict[str, float]:
    selected = counts[counts["celltype_id"].eq(celltype)]
    by_ephys = selected.groupby("ephys_cluster_id")["n_cells"].sum()
    n1 = float(by_ephys.get("Ephys_1", 0.0))
    n2 = float(by_ephys.get("Ephys_2", 0.0))
    minimum = min(n1, n2)
    return {
        "n_ephys1": n1,
        "n_ephys2": n2,
        "min_group_size": minimum,
        "low_count_flag": float(minimum < 50),
    }


def build_candidate_features_from_frames(
    interactions: pd.DataFrame,
    degs: pd.DataFrame,
    counts: pd.DataFrame,
    labels: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one row per candidate gene and observed DEG context."""
    contexts = (
        degs[["IDH_status", "compartment", "celltype_id"]]
        .drop_duplicates()
        .sort_values(["IDH_status", "celltype_id"])
    )
    rows: list[dict[str, object]] = []
    for gene in candidate_genes(interactions):
        for context in contexts.itertuples(index=False):
            record: dict[str, object] = {
                "gene": gene,
                "IDH_status": context.IDH_status,
                "compartment": context.compartment,
                "celltype_id": context.celltype_id,
            }
            record.update(_cellchat_features(interactions, gene, context.celltype_id))
            record.update(_deg_features(degs, gene, context.IDH_status, context.celltype_id))
            record.update(_count_features(counts, context.celltype_id))
            rows.append(record)
    frame = pd.DataFrame(rows)
    for column in FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    if labels is not None:
        label_pairs = set(zip(labels["gene"].str.upper(), labels["compartment"]))
        frame["label"] = [
            int((gene, compartment) in label_pairs)
            for gene, compartment in zip(frame["gene"], frame["compartment"])
        ]
    return frame


def build_candidate_features(data_dir: str | Path, labels_path: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    interactions = pd.read_csv(
        data_dir / "cellchat_ephys_plus_celltype" / "all_significant_interactions.csv"
    )
    degs = pd.read_csv(
        data_dir
        / "within_cluster_ephys_DEGs_by_IDH"
        / "combined_Ephys2_vs_Ephys1_significant_DEGs_by_IDH_and_cluster.csv"
    )
    counts = pd.read_csv(
        data_dir
        / "glioma_compartment_ephys_clustering"
        / "glioma_tumor_tcell_tam_ephys_counts.csv"
    )
    labels = load_seed_labels(labels_path)
    return build_candidate_features_from_frames(interactions, degs, counts, labels)


def gene_group_splits(
    frame: pd.DataFrame, n_splits: int = 5
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield folds that hold out entire genes to avoid context leakage."""
    splitter = GroupKFold(n_splits=n_splits)
    yield from splitter.split(frame, frame.get("label"), groups=frame["gene"])


@dataclass(frozen=True)
class RankingEvaluation:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame
    seeds: tuple[int, ...]
    n_splits: int
    top_k: int


def _precision_at_k(y_true: pd.Series | np.ndarray, scores: np.ndarray, k: int) -> float:
    order = np.argsort(-np.asarray(scores), kind="stable")[: min(k, len(scores))]
    return float(np.asarray(y_true)[order].mean()) if len(order) else 0.0


def _metric_row(name: str, y_true: pd.Series, scores: np.ndarray, top_k: int) -> dict[str, object]:
    return {
        "scorer": name,
        "AUROC": float(roc_auc_score(y_true, scores)),
        "AUPRC": float(average_precision_score(y_true, scores)),
        f"Top-{top_k} precision": _precision_at_k(y_true, scores, top_k),
    }


def evaluate_ranking(
    frame: pd.DataFrame,
    seeds: tuple[int, ...] = (11, 23, 37, 53, 71),
    n_splits: int = 5,
    top_k: int = 25,
) -> RankingEvaluation:
    """Evaluate LightGBM with gene-grouped out-of-fold predictions.

    Every candidate is scored only by models for which that candidate's gene
    was absent from training. Predictions are averaged over the requested
    random seeds.
    """
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:  # pragma: no cover - exercised by installation workflow
        raise RuntimeError(
            "LightGBM is required; install the project with the 'ranking' extra"
        ) from exc

    required = set(FEATURE_COLUMNS) | {"gene", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Candidate frame is missing columns: {sorted(missing)}")
    X = frame[FEATURE_COLUMNS].astype(float)
    y = frame["label"].astype(int)
    predictions_by_seed: list[np.ndarray] = []
    importances: list[np.ndarray] = []
    splits = list(gene_group_splits(frame, n_splits=n_splits))
    for seed in seeds:
        seed_predictions = np.full(len(frame), np.nan, dtype=float)
        for train_idx, valid_idx in splits:
            model = LGBMClassifier(
                objective="binary",
                n_estimators=150,
                learning_rate=0.04,
                num_leaves=15,
                min_child_samples=10,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=seed,
                n_jobs=1,
                verbosity=-1,
            )
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            seed_predictions[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]
            importances.append(model.feature_importances_.astype(float))
        if np.isnan(seed_predictions).any():
            raise RuntimeError("Out-of-fold predictions are incomplete")
        predictions_by_seed.append(seed_predictions)

    oof_score = np.mean(np.vstack(predictions_by_seed), axis=0)
    cellchat_score = frame["max_prob"].astype(float).to_numpy()
    three_table_score = (
        (frame["edge_count"].astype(float) > 0).astype(float)
        + (frame["deg_present"].astype(float) > 0).astype(float)
        + (frame["low_count_flag"].astype(float) == 0).astype(float)
    ).to_numpy() / 3.0
    metrics = pd.DataFrame(
        [
            _metric_row("LightGBM", y, oof_score, top_k),
            _metric_row("CellChat probability alone", y, cellchat_score, top_k),
            _metric_row("3-table rule alone", y, three_table_score, top_k),
        ]
    )

    identity_columns = [
        column
        for column in ["gene", "IDH_status", "compartment", "celltype_id", "label"]
        if column in frame.columns
    ]
    predictions = frame[identity_columns].copy()
    predictions["oof_score"] = oof_score
    predictions["cellchat_probability_score"] = cellchat_score
    predictions["three_table_rule_score"] = three_table_score
    predictions["rank"] = predictions["oof_score"].rank(method="first", ascending=False).astype(int)
    predictions = predictions.sort_values("rank").reset_index(drop=True)
    importance_values = np.mean(np.vstack(importances), axis=0)
    feature_importance = pd.DataFrame(
        {"feature": FEATURE_COLUMNS, "mean_split_importance": importance_values}
    ).sort_values("mean_split_importance", ascending=False, ignore_index=True)
    return RankingEvaluation(
        metrics=metrics,
        predictions=predictions,
        feature_importance=feature_importance,
        seeds=tuple(seeds),
        n_splits=n_splits,
        top_k=top_k,
    )


def write_ranking_outputs(
    result: RankingEvaluation,
    feature_frame: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write reproducible model inputs, rankings, metrics, and limitations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": output_dir / "ranking_metrics.csv",
        "ranked_candidates": output_dir / "ranked_gene_contexts.csv",
        "features": output_dir / "candidate_feature_matrix.csv",
        "feature_importance": output_dir / "feature_importance.csv",
        "model_card": output_dir / "MODEL_CARD.md",
    }
    result.metrics.to_csv(paths["metrics"], index=False)
    result.predictions.to_csv(paths["ranked_candidates"], index=False)
    feature_frame.to_csv(paths["features"], index=False)
    result.feature_importance.to_csv(paths["feature_importance"], index=False)
    metric_table = result.metrics.to_markdown(index=False, floatfmt=".3f")
    paths["model_card"].write_text(
        "# Week 3 gene-ranking prototype\n\n"
        "## Ranked unit\n\n"
        "One CellChat ligand/receptor gene in one IDH status × transcriptomic "
        "cell-type context.\n\n"
        "## Evidence and labels\n\n"
        "The 39 features come only from the CellChat, within-cluster DEG, and "
        "cell-count tables used by the RAG tools. Positive labels are an internal "
        "33-gene, compartment-aware seed list derived from the project hypotheses; "
        "they are **not** an independently curated published gold standard. Metrics "
        "therefore measure recovery of those seeds and should not be interpreted as "
        "clinical or external biological validation.\n\n"
        "## Leakage control\n\n"
        f"GroupKFold with {result.n_splits} folds holds out entire genes. The "
        f"out-of-fold score averages {len(result.seeds)} LightGBM fits per candidate "
        f"using seeds {list(result.seeds)}. No model scores a gene it trained on.\n\n"
        "## Comparison\n\n"
        f"{metric_table}\n\n"
        "The baselines are maximum CellChat probability and a three-table rule "
        "combining any CellChat edge, DEG presence, and adequate group size.\n",
        encoding="utf-8",
    )
    return paths
