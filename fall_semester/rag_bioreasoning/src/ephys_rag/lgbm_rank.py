"""LightGBM PU ranker: nominate gene x cell type x IDH contexts.

Features come from the same tables the RAG tools read (CellChat edges, IDH-stratified
DEGs, group n_cells) plus optional LLM-run features (repeated MedGemma / Ollama runs).
Labels are external literature positives (knowledge/literature_positives.yaml); every
other row is unlabeled. Rows are scored out-of-fold with folds grouped by gene, so a
gene's own label never informs its score.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import pandas as pd
import yaml

from ephys_rag.config import (
    CELLCHAT_COUNTS_CSV,
    CELLCHAT_CSV,
    DEG_IDH_CSV,
    DEG_IDH_SUMMARY_CSV,
    DEG_POOLED_CSV,
    KNOWLEDGE_DIR,
    LOW_N_CELLS,
)

LITERATURE_YAML = KNOWLEDGE_DIR / "literature_positives.yaml"
IDH_GROUPS = ("IDH_Mutant", "IDH_WT")
KEY = ["gene", "celltype_id", "IDH_status"]
META_COLUMNS = {
    "gene",
    "celltype_id",
    "IDH_status",
    "compartment",
    "label",
    "lit_tier",
    "lit_compartments",
    "lit_hypothesis",
    "lit_reference",
}
DEFAULT_LLM_KEYS = ("answer", "response", "text", "ollama", "medgemma")

_TOKEN_FIX = {"TGFbR1": "TGFBR1", "R2": "TGFBR2"}
_METABOLITE_PREFIXES = ("Glu-", "PGE2-")
_CELLCHAT_COMPARTMENT = {"T cell": "T_cell", "TAM1/TAM2/microglia": "TAM_microglia"}
_COMPARTMENT_WORDS = {
    "tumor": re.compile(r"tumou?r|glioma|OPC|AC[_-]like|cycling", re.I),
    "T_cell": re.compile(r"\bT[ -]?cells?\b"),
    "TAM_microglia": re.compile(r"\bTAMs?\b|microglia|myeloid|macrophage", re.I),
}


# --------------------------------------------------------------------------- tables


def split_cellchat_genes(field_value: str) -> list[str]:
    """`Glu-SLC1A3_GLS` -> [SLC1A3, GLS]; `TGFbR1_R2` -> [TGFBR1, TGFBR2]."""
    genes = []
    for token in str(field_value).split("_"):
        for prefix in _METABOLITE_PREFIXES:
            if token.startswith(prefix):
                token = token[len(prefix):]
        token = _TOKEN_FIX.get(token, token)
        if token:
            genes.append(token)
    return genes


def _split_group(group: str) -> tuple[str, str]:
    celltype, ephys = group.rsplit(" / ", 1)
    return celltype.strip(), ephys.strip()


def _cellchat_compartment(celltype: str) -> str:
    if celltype in _CELLCHAT_COMPARTMENT:
        return _CELLCHAT_COMPARTMENT[celltype]
    if "/" in celltype:
        return "mixed"
    return "tumor"


def cellchat_long(path: Path = CELLCHAT_CSV) -> pd.DataFrame:
    """One row per (gene, role, edge): the gene's own cell type, Ephys and partner."""
    frame = pd.read_csv(path)
    rows = []
    for rec in frame.to_dict(orient="records"):
        src_ct, src_ephys = _split_group(rec["source"])
        tgt_ct, tgt_ephys = _split_group(rec["target"])
        sides = (
            ("ligand", rec["ligand"], src_ct, src_ephys, tgt_ct),
            ("receptor", rec["receptor"], tgt_ct, tgt_ephys, src_ct),
        )
        for role, field_value, celltype, ephys, partner in sides:
            for gene in split_cellchat_genes(field_value):
                rows.append(
                    {
                        "gene": gene,
                        "role": role,
                        "celltype_id": celltype,
                        "ephys": ephys,
                        "compartment_cc": _cellchat_compartment(celltype),
                        "partner_celltype": partner,
                        "partner_compartment": _cellchat_compartment(partner),
                        "pathway": rec["pathway"],
                        "prob": float(rec["prob"]),
                        "pval": float(rec["pval"]),
                    }
                )
    return pd.DataFrame(rows)


def _cellchat_context_features(long: pd.DataFrame) -> pd.DataFrame:
    grouped = long.groupby(["gene", "celltype_id"])
    feats = pd.DataFrame(
        {
            "cc_lig_edges": grouped["role"].apply(lambda s: int((s == "ligand").sum())),
            "cc_rec_edges": grouped["role"].apply(lambda s: int((s == "receptor").sum())),
            "cc_edges_e2": grouped["ephys"].apply(lambda s: int((s == "Ephys_2").sum())),
            "cc_edges_e1": grouped["ephys"].apply(lambda s: int((s == "Ephys_1").sum())),
            "cc_max_prob": grouped["prob"].max(),
            "cc_mean_prob": grouped["prob"].mean(),
            "cc_max_pval": grouped["pval"].max(),
            "cc_n_pathways": grouped["pathway"].nunique(),
            "cc_n_partners": grouped["partner_celltype"].nunique(),
        }
    )
    cross = long[long["partner_compartment"] != long["compartment_cc"]]
    feats["cc_cross_compartment_edges"] = cross.groupby(["gene", "celltype_id"]).size()
    feats = feats.fillna({"cc_cross_compartment_edges": 0})
    feats["cc_edges"] = feats["cc_lig_edges"] + feats["cc_rec_edges"]
    total = feats["cc_edges_e2"] + feats["cc_edges_e1"]
    feats["cc_ephys_bias"] = (feats["cc_edges_e2"] - feats["cc_edges_e1"]) / total.where(total > 0)
    return feats.reset_index()


def _cellchat_gene_features(long: pd.DataFrame) -> pd.DataFrame:
    grouped = long.groupby("gene")
    return pd.DataFrame(
        {
            "cc_edges_total": grouped.size(),
            "cc_is_ligand": grouped["role"].apply(lambda s: int((s == "ligand").any())),
            "cc_is_receptor": grouped["role"].apply(lambda s: int((s == "receptor").any())),
            "cc_n_pathways_total": grouped["pathway"].nunique(),
            "cc_n_celltypes_total": grouped["celltype_id"].nunique(),
        }
    ).reset_index()


def _cellchat_group_n(path: Path = CELLCHAT_COUNTS_CSV) -> pd.DataFrame:
    frame = pd.read_csv(path)
    parts = frame["ephys_plus_celltype"].map(_split_group)
    frame["celltype_id"] = parts.map(lambda p: p[0])
    frame["ephys"] = parts.map(lambda p: p[1])
    wide = frame.pivot_table(index="celltype_id", columns="ephys", values="n_cells", aggfunc="sum")
    wide = wide.rename(columns={"Ephys_1": "cc_n_cells_e1", "Ephys_2": "cc_n_cells_e2"})
    return wide.reset_index()[["celltype_id", "cc_n_cells_e1", "cc_n_cells_e2"]]


def _deg_features(deg_idh: pd.DataFrame, deg_pooled: pd.DataFrame) -> pd.DataFrame:
    deg = deg_idh.drop_duplicates(KEY).copy()
    deg["deg_is_sig"] = 1
    deg["deg_log2fc"] = deg["avg_log2FC"]
    deg["deg_abs_log2fc"] = deg["avg_log2FC"].abs()
    deg["deg_neglog10_padj"] = -np.log10(deg["p_val_adj"].clip(lower=1e-300))
    deg["deg_pct_e2"] = deg["pct.1"]
    deg["deg_pct_e1"] = deg["pct.2"]
    deg["deg_pct_diff"] = deg["pct.1"] - deg["pct.2"]
    deg["deg_dir"] = np.sign(deg["avg_log2FC"]).astype(int)
    cols = KEY + [
        "deg_is_sig",
        "deg_log2fc",
        "deg_abs_log2fc",
        "deg_neglog10_padj",
        "deg_pct_e2",
        "deg_pct_e1",
        "deg_pct_diff",
        "deg_dir",
    ]
    deg = deg[cols]

    other = deg[KEY + ["deg_dir"]].copy()
    other["IDH_status"] = other["IDH_status"].map({"IDH_Mutant": "IDH_WT", "IDH_WT": "IDH_Mutant"})
    other = other.rename(columns={"deg_dir": "_other_dir"})
    deg = deg.merge(other, on=KEY, how="left")
    deg["deg_other_idh_sig"] = deg["_other_dir"].notna().astype(int)
    deg["deg_other_idh_same_dir"] = (deg["_other_dir"] == deg["deg_dir"]).astype(int)
    deg = deg.drop(columns="_other_dir")

    pooled = deg_pooled.drop_duplicates(["gene", "celltype_id"])[["gene", "celltype_id", "avg_log2FC"]]
    pooled = pooled.rename(columns={"avg_log2FC": "deg_pooled_log2fc"})
    pooled["deg_pooled_sig"] = 1
    return deg.merge(pooled, on=["gene", "celltype_id"], how="left")


def _context_n(summary: pd.DataFrame) -> pd.DataFrame:
    ctx = summary[["IDH_status", "celltype_id", "n_Ephys_1", "n_Ephys_2"]].rename(
        columns={"n_Ephys_1": "ctx_n_e1", "n_Ephys_2": "ctx_n_e2"}
    )
    ctx["ctx_min_n"] = ctx[["ctx_n_e1", "ctx_n_e2"]].min(axis=1)
    ctx["ctx_low_n"] = (ctx["ctx_min_n"] < LOW_N_CELLS).astype(int)
    return ctx


# --------------------------------------------------------------------------- labels


def load_literature(path: Path = LITERATURE_YAML) -> pd.DataFrame:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    rows = []
    for entry in payload.get("positives", []):
        rows.append(
            {
                "gene": entry["gene"],
                "lit_tier": int(entry.get("tier", 2)),
                "lit_compartments": ",".join(entry.get("compartments", [])),
                "lit_hypothesis": entry.get("hypothesis", ""),
                "lit_reference": entry.get("reference", ""),
            }
        )
    return pd.DataFrame(rows)


def attach_labels(frame: pd.DataFrame, literature: pd.DataFrame, tiers: Sequence[int]) -> pd.DataFrame:
    out = frame.merge(literature, on="gene", how="left")
    in_compartment = [
        isinstance(comps, str) and comp in comps.split(",")
        for comp, comps in zip(out["compartment"], out["lit_compartments"])
    ]
    out["label"] = (pd.Series(in_compartment, index=out.index) & out["lit_tier"].isin(tiers)).astype(int)
    return out


def row_status(frame: pd.DataFrame, tiers: Sequence[int]) -> pd.Series:
    def status(rec) -> str:
        if rec["label"] == 1:
            return f"literature tier {int(rec['lit_tier'])} (label)"
        if pd.notna(rec["lit_tier"]):
            comps = str(rec["lit_compartments"]).split(",")
            if rec["compartment"] not in comps:
                return "literature gene, other compartment"
            return f"literature tier {int(rec['lit_tier'])} (unlabeled)"
        return "novel"

    return frame.apply(status, axis=1)


# --------------------------------------------------------------------------- LLM hook


@dataclass
class LLMAnswer:
    run_id: str
    question_id: str
    question: str
    text: str


def iter_llm_answers(runs_dir: Path, keys: Sequence[str] = DEFAULT_LLM_KEYS) -> Iterator[LLMAnswer]:
    """Read repeated-run answers from JSON / JSONL files anywhere under `runs_dir`.

    Each record needs a question (`question`), an id (`question_id`, else file stem) and at
    least one text field in `keys`. The run id is `run_id` if present, else the file's parent
    folder; each text key is its own run (e.g. run3:ollama).
    """
    root = Path(runs_dir)
    for path in sorted(root.rglob("*")):
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            records = payload if isinstance(payload, list) else [payload]
        elif path.suffix == ".jsonl":
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            continue
        folder = path.parent.relative_to(root).as_posix()
        for rec in records:
            if not isinstance(rec, dict):
                continue
            for key in keys:
                text = rec.get(key)
                if not isinstance(text, str) or not text.strip():
                    continue
                run = str(rec.get("run_id") or folder)
                yield LLMAnswer(
                    run_id=f"{run}:{key}",
                    question_id=str(rec.get("question_id", path.stem)),
                    question=str(rec.get("question", "")),
                    text=text,
                )


def _gene_pattern(genes: Iterable[str]) -> re.Pattern:
    alternation = "|".join(re.escape(g) for g in sorted(set(genes), key=len, reverse=True))
    return re.compile(rf"(?<![A-Za-z0-9-])({alternation})(?![A-Za-z0-9-])")


def llm_run_features(
    runs_dir: Path, genes: Iterable[str], keys: Sequence[str] = DEFAULT_LLM_KEYS
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Gene-level and gene x compartment mention consistency across runs."""
    pattern = _gene_pattern(genes)
    mentions: dict[tuple[str, str], int] = {}
    unprompted: set[tuple[str, str]] = set()
    ctx_mentions: set[tuple[str, str, str]] = set()
    runs: set[str] = set()
    for answer in iter_llm_answers(runs_dir, keys):
        runs.add(answer.run_id)
        prompted = set(pattern.findall(answer.question))
        for gene in pattern.findall(answer.text):
            mentions[(gene, answer.run_id)] = mentions.get((gene, answer.run_id), 0) + 1
            if gene not in prompted:
                unprompted.add((gene, answer.run_id))
        for sentence in re.split(r"(?<=[.!?\n])\s+", answer.text):
            found = set(pattern.findall(sentence))
            if not found:
                continue
            for comp, word in _COMPARTMENT_WORDS.items():
                if word.search(sentence):
                    for gene in found:
                        ctx_mentions.add((gene, comp, answer.run_id))
    n_runs = len(runs)
    if n_runs == 0:
        return pd.DataFrame(columns=["gene"]), pd.DataFrame(columns=["gene", "compartment"]), 0

    gene_rows: dict[str, dict] = {}
    for (gene, run), count in mentions.items():
        row = gene_rows.setdefault(gene, {"gene": gene, "_runs": set(), "_unprompted": set(), "_total": 0})
        row["_runs"].add(run)
        row["_total"] += count
        if (gene, run) in unprompted:
            row["_unprompted"].add(run)
    gene_frame = pd.DataFrame(
        [
            {
                "gene": gene,
                "llm_mention_rate": len(row["_runs"]) / n_runs,
                "llm_unprompted_mention_rate": len(row["_unprompted"]) / n_runs,
                "llm_mentions_per_run": row["_total"] / n_runs,
            }
            for gene, row in gene_rows.items()
        ]
    )
    ctx_counts: dict[tuple[str, str], int] = {}
    for gene, comp, _run in ctx_mentions:
        ctx_counts[(gene, comp)] = ctx_counts.get((gene, comp), 0) + 1
    ctx_frame = pd.DataFrame(
        [
            {"gene": gene, "compartment": comp, "llm_ctx_mention_rate": count / n_runs}
            for (gene, comp), count in ctx_counts.items()
        ]
    )
    return gene_frame, ctx_frame, n_runs


# --------------------------------------------------------------------------- features


@dataclass
class FeatureTable:
    frame: pd.DataFrame
    feature_columns: list[str]
    n_llm_runs: int = 0
    notes: list[str] = field(default_factory=list)


def build_feature_table(
    universe: str = "cellchat",
    llm_runs: Path | None = None,
    llm_keys: Sequence[str] = DEFAULT_LLM_KEYS,
    compartment_features: bool = False,
) -> FeatureTable:
    """Rows = gene x DEG cell type x IDH.

    universe="cellchat": every CellChat ligand/receptor gene crossed with all 12 contexts,
    so not being a DEG in a context is itself informative.
    universe="deg": every significant DEG row (all genes), CellChat features where present.
    """
    long = cellchat_long()
    deg_idh = pd.read_csv(DEG_IDH_CSV)
    deg_pooled = pd.read_csv(DEG_POOLED_CSV)
    contexts = deg_idh[["celltype_id", "IDH_status", "compartment"]].drop_duplicates()

    if universe == "cellchat":
        genes = pd.DataFrame({"gene": sorted(long["gene"].unique())})
        base = genes.merge(contexts, how="cross")
    elif universe == "deg":
        base = deg_idh[KEY + ["compartment"]].drop_duplicates(KEY)
    else:
        raise ValueError(f"unknown universe {universe!r}")

    frame = base.merge(_deg_features(deg_idh, deg_pooled), on=KEY, how="left")
    zero_fill = [
        "deg_is_sig",
        "deg_log2fc",
        "deg_abs_log2fc",
        "deg_neglog10_padj",
        "deg_pct_diff",
        "deg_dir",
        "deg_other_idh_sig",
        "deg_other_idh_same_dir",
        "deg_pooled_sig",
    ]
    frame[zero_fill] = frame[zero_fill].fillna(0)
    frame["deg_n_celltypes_sig"] = frame.groupby(["gene", "IDH_status"])["deg_is_sig"].transform("sum")

    frame = frame.merge(_cellchat_context_features(long), on=["gene", "celltype_id"], how="left")
    cc_zero = [c for c in frame.columns if c.startswith("cc_") and c not in ("cc_ephys_bias", "cc_max_pval")]
    frame[cc_zero] = frame[cc_zero].fillna(0)
    frame = frame.merge(_cellchat_gene_features(long), on="gene", how="left")
    cc_gene_cols = ["cc_edges_total", "cc_is_ligand", "cc_is_receptor", "cc_n_pathways_total", "cc_n_celltypes_total"]
    frame[cc_gene_cols] = frame[cc_gene_cols].fillna(0)
    frame = frame.merge(_cellchat_group_n(), on="celltype_id", how="left")
    frame = frame.merge(_context_n(pd.read_csv(DEG_IDH_SUMMARY_CSV)), on=["IDH_status", "celltype_id"], how="left")

    frame["deg_cc_concordance"] = frame["deg_dir"] * frame["cc_ephys_bias"].fillna(0)
    frame["idh_mutant"] = (frame["IDH_status"] == "IDH_Mutant").astype(int)
    if compartment_features:
        for comp in ("tumor", "T_cell", "TAM_microglia"):
            frame[f"comp_{comp}"] = (frame["compartment"] == comp).astype(int)
    frame["rule_pass"] = (
        (frame["cc_edges"] > 0) & (frame["deg_is_sig"] == 1) & (frame["ctx_min_n"] >= LOW_N_CELLS)
    ).astype(int)

    notes = []
    n_runs = 0
    if llm_runs is not None:
        gene_llm, ctx_llm, n_runs = llm_run_features(Path(llm_runs), frame["gene"].unique(), llm_keys)
        if n_runs:
            frame = frame.merge(gene_llm, on="gene", how="left")
            frame = frame.merge(ctx_llm, on=["gene", "compartment"], how="left")
            llm_cols = [c for c in frame.columns if c.startswith("llm_")]
            frame[llm_cols] = frame[llm_cols].fillna(0)
            frame["llm_n_runs"] = n_runs
        else:
            notes.append(f"No LLM answers found under {llm_runs} for keys {list(llm_keys)}.")

    feature_columns = [
        c
        for c in frame.columns
        if c not in META_COLUMNS and c != "llm_n_runs" and pd.api.types.is_numeric_dtype(frame[c])
    ]
    return FeatureTable(frame=frame, feature_columns=feature_columns, n_llm_runs=n_runs, notes=notes)


# --------------------------------------------------------------------------- model


def _lgb_params(seed: int, pos_weight: float) -> dict:
    # Shallow, heavily regularized trees: with ~15-35 positive genes, deeper settings
    # memorize gene identity and score worse out-of-fold than single-column baselines.
    return {
        "objective": "binary",
        "n_estimators": 150,
        "learning_rate": 0.05,
        "num_leaves": 4,
        "max_depth": 2,
        "min_child_samples": 30,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.6,
        "reg_lambda": 5.0,
        "scale_pos_weight": pos_weight,
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }


def _pos_weight(y: np.ndarray) -> float:
    return float((y == 0).sum() / max((y == 1).sum(), 1))


def oof_scores(
    X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, seeds: Sequence[int], n_splits: int
) -> tuple[np.ndarray, np.ndarray]:
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedGroupKFold

    scores = np.zeros((len(seeds), len(y)))
    for i, seed in enumerate(seeds):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, test_idx in cv.split(X, y, groups):
            model = lgb.LGBMClassifier(**_lgb_params(seed, _pos_weight(y[train_idx])))
            model.fit(X.iloc[train_idx], y[train_idx])
            scores[i, test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
    return scores.mean(axis=0), scores.std(axis=0)


def fit_full(X: pd.DataFrame, y: np.ndarray, seeds: Sequence[int]) -> tuple[pd.DataFrame, np.ndarray]:
    """Seed-averaged gain importance and SHAP contributions (LightGBM pred_contrib)."""
    import lightgbm as lgb

    gains = np.zeros(X.shape[1])
    splits = np.zeros(X.shape[1])
    contrib = np.zeros((X.shape[0], X.shape[1] + 1))
    for seed in seeds:
        model = lgb.LGBMClassifier(**_lgb_params(seed, _pos_weight(y)))
        model.fit(X, y)
        booster = model.booster_
        gains += booster.feature_importance(importance_type="gain")
        splits += booster.feature_importance(importance_type="split")
        contrib += model.predict(X, pred_contrib=True)
    n = len(seeds)
    contrib /= n
    importance = pd.DataFrame(
        {
            "feature": X.columns,
            "gain": gains / n,
            "splits": splits / n,
            "mean_abs_shap": np.abs(contrib[:, :-1]).mean(axis=0),
        }
    ).sort_values("gain", ascending=False)
    return importance, contrib[:, :-1]


def _top_reasons(X: pd.DataFrame, contrib: np.ndarray, k: int = 3) -> list[str]:
    reasons = []
    cols = X.columns.to_numpy()
    for i in range(len(X)):
        order = np.argsort(-contrib[i])[:k]
        parts = []
        for j in order:
            if contrib[i, j] <= 0:
                continue
            value = X.iat[i, j]
            shown = f"{value:.3g}" if isinstance(value, (float, np.floating)) else str(value)
            parts.append(f"{cols[j]}={shown} (+{contrib[i, j]:.2f})")
        reasons.append("; ".join(parts))
    return reasons


def _ranking_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    score = np.nan_to_num(np.asarray(score, dtype=float), nan=0.0)
    order = np.argsort(-score)
    out = {
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
    }
    for k in (10, 25, 50):
        if k <= len(y):
            out[f"precision_at_{k}"] = float(y[order[:k]].mean())
    return out


def evaluate(frame: pd.DataFrame, score_col: str = "score") -> dict:
    y = frame["label"].to_numpy()
    baselines = {
        "lightgbm_oof": frame[score_col],
        "rule_pass": frame["rule_pass"] + 1e-3 * frame["deg_abs_log2fc"],
        "deg_abs_log2fc": frame["deg_abs_log2fc"],
        "cc_edges": frame["cc_edges"],
        "cc_max_prob": frame["cc_max_prob"],
    }
    rows = {name: _ranking_metrics(y, s.to_numpy()) for name, s in baselines.items()}

    by_gene = frame.groupby("gene").agg(label=("label", "max"), **{
        name: (col, "max") for name, col in [("lightgbm_oof", score_col), ("deg_abs_log2fc", "deg_abs_log2fc"), ("cc_edges", "cc_edges")]
    })
    gene_rows = {
        name: _ranking_metrics(by_gene["label"].to_numpy(), by_gene[name].to_numpy())
        for name in ("lightgbm_oof", "deg_abs_log2fc", "cc_edges")
    }
    return {
        "n_rows": int(len(frame)),
        "n_positive_rows": int(y.sum()),
        "n_genes": int(frame["gene"].nunique()),
        "n_positive_genes": int(by_gene["label"].sum()),
        "prevalence_rows": float(y.mean()),
        "row_level": rows,
        "gene_level": gene_rows,
    }


@dataclass
class RankResult:
    nominees: pd.DataFrame
    by_gene: pd.DataFrame
    importance: pd.DataFrame
    metrics: dict
    table: FeatureTable


def rank(
    universe: str = "cellchat",
    tiers: Sequence[int] = (1, 2),
    llm_runs: Path | None = None,
    llm_keys: Sequence[str] = DEFAULT_LLM_KEYS,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    n_splits: int = 5,
) -> RankResult:
    table = build_feature_table(universe=universe, llm_runs=llm_runs, llm_keys=llm_keys)
    frame = attach_labels(table.frame, load_literature(), tiers)
    if frame["label"].sum() < n_splits:
        raise ValueError("Fewer literature-positive rows than CV folds; widen --tiers or the universe.")

    X = frame[table.feature_columns].astype(float)
    y = frame["label"].to_numpy()
    groups = frame["gene"].to_numpy()

    frame["score"], frame["score_sd"] = oof_scores(X, y, groups, seeds, n_splits)
    importance, contrib = fit_full(X, y, seeds)
    frame["top_reasons"] = _top_reasons(X, contrib)
    frame["status"] = row_status(frame, tiers)

    metrics = evaluate(frame)
    metrics.update(
        {
            "universe": universe,
            "label_tiers": list(tiers),
            "seeds": list(seeds),
            "cv": f"StratifiedGroupKFold(n_splits={n_splits}) grouped by gene",
            "n_features": len(table.feature_columns),
            "features": table.feature_columns,
            "llm_runs_dir": str(llm_runs) if llm_runs else None,
            "n_llm_runs": table.n_llm_runs,
            "notes": table.notes,
        }
    )

    nominees = frame.sort_values("score", ascending=False).reset_index(drop=True)
    nominees.insert(0, "rank", np.arange(1, len(nominees) + 1))
    by_gene = (
        nominees.drop_duplicates("gene")
        .reset_index(drop=True)
        .assign(gene_rank=lambda d: np.arange(1, len(d) + 1))
    )
    return RankResult(nominees=nominees, by_gene=by_gene, importance=importance, metrics=metrics, table=table)


# --------------------------------------------------------------------------- report

DISPLAY_COLUMNS = [
    "rank",
    "gene",
    "celltype_id",
    "IDH_status",
    "score",
    "status",
    "rule_pass",
    "deg_log2fc",
    "deg_neglog10_padj",
    "cc_edges",
    "cc_ephys_bias",
    "ctx_min_n",
    "top_reasons",
]


def _md_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    def fmt(value) -> str:
        if isinstance(value, (float, np.floating)):
            return "" if np.isnan(value) else f"{value:.3g}"
        return str(value).replace("|", "/")

    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for rec in frame[list(columns)].itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in rec) + " |")
    return "\n".join(lines)


def write_outputs(result: RankResult, out_dir: Path, top: int = 30) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "features": out_dir / "lgbm_features.csv",
        "nominees": out_dir / "lgbm_nominees.csv",
        "by_gene": out_dir / "lgbm_nominees_by_gene.csv",
        "importance": out_dir / "lgbm_feature_importance.csv",
        "metrics": out_dir / "lgbm_metrics.json",
        "report": out_dir / "LGBM_NOMINEES.md",
    }
    result.table.frame.to_csv(paths["features"], index=False)
    result.nominees.to_csv(paths["nominees"], index=False)
    result.by_gene.to_csv(paths["by_gene"], index=False)
    result.importance.to_csv(paths["importance"], index=False)
    paths["metrics"].write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")

    m = result.metrics
    metric_rows = pd.DataFrame(
        [{"scorer": name, **vals} for name, vals in m["row_level"].items()]
    )
    gene_rows = pd.DataFrame([{"scorer": name, **vals} for name, vals in m["gene_level"].items()])
    novel = result.nominees[(result.nominees["status"] == "novel") & (result.nominees["rule_pass"] == 1)]
    gene_cols = ["gene_rank", "gene", "celltype_id", "IDH_status", "score", "status", "rule_pass", "top_reasons"]

    report = [
        "# LightGBM gene nominees (gene x cell type x IDH)",
        "",
        f"- Universe: `{m['universe']}`. {m['n_rows']} rows, {m['n_genes']} genes, {m['n_features']} features.",
        f"- Labels: literature tiers {m['label_tiers']} from `knowledge/literature_positives.yaml`: "
        f"{m['n_positive_rows']} positive rows ({m['n_positive_genes']} genes). Everything else is unlabeled, not negative.",
        f"- Scores are out-of-fold: {m['cv']}, averaged over seeds {m['seeds']}.",
        f"- LLM run features: {'%d runs from `%s`' % (m['n_llm_runs'], m['llm_runs_dir']) if m['n_llm_runs'] else 'none (table-only model)'}.",
        "",
        "## Does the model beat single-table baselines?",
        "",
        f"Row-level prevalence = {m['prevalence_rows']:.3f} (AUPRC of a random ranking).",
        "",
        _md_table(metric_rows, list(metric_rows.columns)),
        "",
        "Gene level (best context per gene):",
        "",
        _md_table(gene_rows, list(gene_rows.columns)),
        "",
        f"## Top {top} contexts",
        "",
        _md_table(result.nominees.head(top), DISPLAY_COLUMNS),
        "",
        "## Top novel nominees that pass the 3-table rule",
        "",
        "Not in the literature list, but CellChat edge + DEG in this cell type/IDH + adequate n_cells.",
        "",
        _md_table(novel.head(20), DISPLAY_COLUMNS),
        "",
        "## Top genes (best context per gene)",
        "",
        _md_table(result.by_gene.head(top), gene_cols),
        "",
        "## Feature importance (seed-averaged gain)",
        "",
        _md_table(result.importance.head(15), ["feature", "gain", "splits", "mean_abs_shap"]),
        "",
        "## Caveats",
        "",
        "- The positive list is a small, hand-curated draft. Labeled genes are biased toward well-studied biology.",
        "- Tier 2 genes are pathway-level picks that overlap the H1-H3 hypotheses. `--tiers 1` is the strict check;"
        " with tier 1 alone (16 genes) the model does not beat single-column baselines.",
        "- CellChat is not IDH-stratified, so `cc_*` features repeat across IDH groups; only DEG and n features differ by IDH.",
        "- `cc_n_cells_*` and `ctx_n_*` are constant per cell type, so they act as a cell-type proxy; most labels are"
        " tumor-side, which lifts tumor contexts (e.g. CD74 in AC_like_tumor ranks high despite being a myeloid label).",
        "- Gene-level AUROC is on par with ranking genes by CellChat edge count; the model's added value is mainly"
        " picking the right cell type x IDH context for a gene.",
        "- High scores mean \"looks like known glioma communication genes in these tables\", not validation.",
        "- `top_reasons` are SHAP contributions from the full-data model; scores themselves are out-of-fold.",
    ]
    for note in m.get("notes", []):
        report.append(f"- {note}")
    paths["report"].write_text("\n".join(report) + "\n", encoding="utf-8")
    return list(paths.values())
