"""Two LightGBM PU models labeled by knowledge-tool gene sets instead of the 33 published genes.

p_tumor_synaptic  - SynGO genes as positives, trained and scored on tumor cell types
p_immune_synapse  - GO:0001772 genes as positives, trained and scored on all compartments

The 33 published glioma genes are not used for training. They are flagged
(`in_published_33`) and used as an external check of where each model ranks them.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ephys_rag.gene_sets import GENE_SETS, SOURCES_JSON, load_gene_set
from ephys_rag.lgbm_rank import (
    DEFAULT_LLM_KEYS,
    KEY,
    _md_table,
    _ranking_metrics,
    _top_reasons,
    build_feature_table,
    evaluate,
    fit_full,
    load_literature,
    oof_scores,
)

SET_NAMES = tuple(GENE_SETS)
SUMMARY_FEATURES = ["rule_pass", "deg_log2fc", "deg_neglog10_padj", "cc_edges", "ctx_min_n"]
# Mitochondrial, ribosomal, histone and translation-machinery genes: broadly expressed, so they
# score high on "looks like a synaptic gene" by expression alone. Flagged, kept in the CSV,
# left out of the novel-nominee tables.
HOUSEKEEPING = re.compile(
    r"^(MT-|MTRNR|RP[LS]\d|RP[LS]P|RPLP|MRP[LS]|H1F|H2AF|H2BF|H3F|HIST|EEF1|EEF2|EIF[1-5]|NACA$|BTF3$|TPT1$|PTMA$|FTL$|FTH1$|UBA52$|NDUF|COX\d|ATP5|UQCR)"
)
PUBLISHED_BASELINES = {"deg_abs_log2fc": "max", "deg_pct_e2": "max", "cc_edges": "max", "deg_n_celltypes_sig": "max"}


@dataclass
class GeneSetResult:
    scores: pd.DataFrame
    by_gene: pd.DataFrame
    published: pd.DataFrame
    importance: dict[str, pd.DataFrame]
    metrics: dict


def _published_check(sub: pd.DataFrame, score_col: str, set_col: str, published: set[str]) -> tuple[dict, pd.DataFrame]:
    """Gene-level ranking of the published genes under a model that never trained on them as a list."""
    best = (
        sub.sort_values(score_col, ascending=False)
        .drop_duplicates("gene")
        .reset_index(drop=True)
    )
    best["gene_rank"] = np.arange(1, len(best) + 1)
    best["gene_percentile"] = 1 - (best["gene_rank"] - 1) / len(best)
    best["published"] = best["gene"].isin(published).astype(int)

    out: dict = {
        "n_genes_scored": int(len(best)),
        "n_published_scored": int(best["published"].sum()),
        "published_missing_from_universe": sorted(published - set(best["gene"])),
    }
    if best["published"].sum() and best["published"].sum() < len(best):
        out["all_genes"] = _ranking_metrics(best["published"].to_numpy(), best[score_col].to_numpy())
        out["median_percentile_published"] = float(best.loc[best["published"] == 1, "gene_percentile"].median())
        per_gene = sub.groupby("gene").agg(**{col: (col, how) for col, how in PUBLISHED_BASELINES.items()})
        labels = per_gene.index.isin(published).astype(int)
        out["baselines_all_genes"] = {
            col: _ranking_metrics(labels, per_gene[col].to_numpy()) for col in PUBLISHED_BASELINES
        }
    outside = best[best[set_col] == 0]
    if outside["published"].sum() and outside["published"].sum() < len(outside):
        out["genes_outside_label_set"] = _ranking_metrics(
            outside["published"].to_numpy(), outside[score_col].to_numpy()
        )
        out["n_published_outside_label_set"] = int(outside["published"].sum())
    return out, best[best["published"] == 1]


def rank_gene_sets(
    universe: str = "deg",
    sets: Sequence[str] = SET_NAMES,
    immune_human_only: bool = False,
    llm_runs: Path | None = None,
    llm_keys: Sequence[str] = DEFAULT_LLM_KEYS,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    n_splits: int = 5,
) -> GeneSetResult:
    table = build_feature_table(
        universe=universe, llm_runs=llm_runs, llm_keys=llm_keys, compartment_features=True
    )
    frame = table.frame.copy()
    features = table.feature_columns

    literature = load_literature()
    published = set(literature["gene"])
    tier = dict(zip(literature["gene"], literature["lit_tier"]))
    frame["in_published_33"] = frame["gene"].isin(published).astype(int)
    frame["published_tier"] = frame["gene"].map(tier)

    metrics: dict = {
        "universe": universe,
        "n_rows": int(len(frame)),
        "n_genes": int(frame["gene"].nunique()),
        "n_features": len(features),
        "features": features,
        "seeds": list(seeds),
        "cv": f"StratifiedGroupKFold(n_splits={n_splits}) grouped by gene",
        "immune_human_only": immune_human_only,
        "n_llm_runs": table.n_llm_runs,
        "sources": json.loads(SOURCES_JSON.read_text(encoding="utf-8")) if SOURCES_JSON.exists() else {},
        "sets": {},
    }
    importance: dict[str, pd.DataFrame] = {}
    published_tables = []

    for name in sets:
        spec = GENE_SETS[name]
        genes = load_gene_set(name, human_only=immune_human_only and name == "immune_synapse")
        set_col, score_col = f"in_{name}", f"p_{name}"
        frame[set_col] = frame["gene"].isin(genes).astype(int)

        mask = frame["compartment"].isin(spec.label_compartments)
        sub = frame[mask]
        X = sub[features].astype(float)
        y = sub[set_col].to_numpy()
        if y.sum() < n_splits:
            raise ValueError(f"{name}: only {y.sum()} positive rows in {spec.label_compartments}.")

        score, sd = oof_scores(X, y, sub["gene"].to_numpy(), seeds, n_splits)
        frame.loc[mask, score_col] = score
        frame.loc[mask, f"{score_col}_sd"] = sd
        imp, contrib = fit_full(X, y, seeds)
        importance[name] = imp
        frame.loc[mask, f"why_{name}"] = _top_reasons(X, contrib)

        scored = frame[mask].assign(label=y, score=score)
        set_metrics = evaluate(scored)
        check, pub_rows = _published_check(frame[mask], score_col, set_col, published)
        pub_rows = pub_rows.assign(model=name)
        published_tables.append(
            pub_rows[
                ["model", "gene", "published_tier", set_col, "celltype_id", "IDH_status", score_col, "gene_rank", "gene_percentile"]
            ].rename(columns={set_col: "in_label_set", score_col: "best_score"})
        )
        set_metrics.update(
            {
                "description": spec.description,
                "label_compartments": list(spec.label_compartments),
                "set_size": len(genes),
                "set_genes_in_universe": int(frame.loc[mask, "gene"][frame.loc[mask, set_col] == 1].nunique()),
                "published_in_set": sorted(published & genes),
                "published_check": check,
            }
        )
        metrics["sets"][name] = set_metrics

    in_cols = [f"in_{n}" for n in sets]
    frame["housekeeping"] = frame["gene"].str.match(HOUSEKEEPING).astype(int)
    frame["novel"] = (
        (frame[in_cols].sum(axis=1) == 0) & (frame["in_published_33"] == 0) & (frame["housekeeping"] == 0)
    ).astype(int)

    by_gene_parts = []
    for name in sets:
        score_col = f"p_{name}"
        best = (
            frame.dropna(subset=[score_col])
            .sort_values(score_col, ascending=False)
            .drop_duplicates("gene")[["gene", score_col, "celltype_id", "IDH_status"]]
            .rename(columns={"celltype_id": f"best_ct_{name}", "IDH_status": f"best_idh_{name}"})
        )
        best[f"gene_rank_{name}"] = np.arange(1, len(best) + 1)
        by_gene_parts.append(best.set_index("gene"))
    flags = frame.groupby("gene")[in_cols + ["in_published_33", "novel"]].max()
    by_gene = pd.concat(by_gene_parts + [flags], axis=1).reset_index()
    by_gene["published_tier"] = by_gene["gene"].map(tier)

    published_frame = pd.concat(published_tables, ignore_index=True) if published_tables else pd.DataFrame()
    return GeneSetResult(
        scores=frame, by_gene=by_gene, published=published_frame, importance=importance, metrics=metrics
    )


def _metric_table(block: dict) -> pd.DataFrame:
    return pd.DataFrame([{"scorer": k, **v} for k, v in block.items()])


def write_gene_set_outputs(result: GeneSetResult, out_dir: Path, top: int = 25) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    m = result.metrics
    sets = list(m["sets"])
    paths = [out_dir / "genesets_scores.csv", out_dir / "genesets_by_gene.csv", out_dir / "genesets_published33.csv"]

    keep = KEY + ["compartment", "in_published_33", "published_tier", "novel", "housekeeping"]
    for name in sets:
        keep += [f"in_{name}", f"p_{name}", f"p_{name}_sd"]
    keep += SUMMARY_FEATURES + [f"why_{name}" for name in sets]
    result.scores[keep].to_csv(paths[0], index=False)
    result.by_gene.to_csv(paths[1], index=False)
    result.published.to_csv(paths[2], index=False)
    for name, imp in result.importance.items():
        path = out_dir / f"genesets_feature_importance_{name}.csv"
        imp.to_csv(path, index=False)
        paths.append(path)
    metrics_path = out_dir / "genesets_metrics.json"
    metrics_path.write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
    paths.append(metrics_path)

    lines = [
        "# LightGBM with knowledge-tool labels: tumor synaptic vs immune synapse",
        "",
        f"- Universe: `{m['universe']}`: {m['n_rows']} gene x cell type x IDH rows, {m['n_genes']} genes, {m['n_features']} features.",
        f"- Scores are out-of-fold ({m['cv']}, seeds {m['seeds']}): no gene is scored by a model that trained on it.",
        "- The 33 published glioma genes are **not** labels here. `in_published_33` flags them; the check below shows where each model ranks them.",
        f"- LLM run features: {m['n_llm_runs'] or 'none'}.",
        "",
    ]
    src = m.get("sources", {})
    if src:
        lines += ["## Label sources", ""]
        for name in sets:
            s = src.get(name, {})
            lines.append(
                f"- **{name}**: {s.get('source')} {s.get('go_id', '')}"
                + (f" release {s['release']}" if s.get("release") else "")
                + ": "
                f"{s.get('n_annotations')} annotations -> {s.get('n_genes')} genes"
                + (
                    f" ({s.get('n_genes_human_annotated')} with a human annotation, {s.get('n_genes_experimental')} with experimental evidence)"
                    if "n_genes_human_annotated" in s
                    else ""
                )
                + f". Fetched {src.get('fetched')}."
            )
        lines.append("")

    for name in sets:
        s = m["sets"][name]
        check = s["published_check"]
        lines += [
            f"## {name}",
            "",
            f"{s['description']}. Compartments: {', '.join(s['label_compartments'])}.",
            f"{s['set_genes_in_universe']} of {s['set_size']} set genes are in the universe; "
            f"{s['n_positive_rows']} positive rows of {s['n_rows']} (prevalence {s['prevalence_rows']:.3f}).",
            "",
            "**Model vs single-column baselines (row level)**",
            "",
            _md_table(_metric_table(s["row_level"]), list(_metric_table(s["row_level"]).columns)),
            "",
            "**Gene level (best context per gene)**",
            "",
            _md_table(_metric_table(s["gene_level"]), list(_metric_table(s["gene_level"]).columns)),
            "",
            "**Published-33 check**",
            "",
            f"- Published genes already in this label set: {', '.join(s['published_in_set']) or 'none'}.",
            f"- Published genes scored: {check['n_published_scored']} of 33"
            + (f"; not in universe: {', '.join(check['published_missing_from_universe'])}" if check["published_missing_from_universe"] else "")
            + ".",
        ]
        if "all_genes" in check:
            base = ", ".join(
                f"{col} {vals['auroc']:.3f}" for col, vals in check.get("baselines_all_genes", {}).items()
            )
            lines.append(
                f"- Median gene-rank percentile of published genes: {check['median_percentile_published']:.2f} "
                f"(1.00 = top). AUROC published vs all other genes: {check['all_genes']['auroc']:.3f} "
                f"(single-column baselines, best context per gene: {base})."
            )
        if "genes_outside_label_set" in check:
            lines.append(
                f"- Held-out test, genes **not** in the label set only ({check['n_published_outside_label_set']} published): "
                f"AUROC {check['genes_outside_label_set']['auroc']:.3f}."
            )
        pub = result.published[result.published["model"] == name].sort_values("gene_rank")
        lines += [
            "",
            _md_table(pub, ["gene", "published_tier", "in_label_set", "celltype_id", "IDH_status", "best_score", "gene_rank", "gene_percentile"]),
            "",
        ]

        score_col = f"p_{name}"
        rows = result.scores.dropna(subset=[score_col]).sort_values(score_col, ascending=False)
        novel = rows[rows["novel"] == 1].drop_duplicates("gene")
        cols = ["gene", "celltype_id", "IDH_status", score_col] + SUMMARY_FEATURES + [f"why_{name}"]
        lines += [
            f"**Top {top} novel genes** (in neither gene set nor the published 33, housekeeping genes removed; best context per gene)",
            "",
            _md_table(novel.head(top), cols),
            "",
            "**Novel genes that also pass the 3-table rule** (CellChat edge + DEG + n_cells)",
            "",
            _md_table(novel[novel["rule_pass"] == 1].head(15), cols),
            "",
            "**Feature importance (gain, top 10)**",
            "",
            _md_table(result.importance[name].head(10), ["feature", "gain", "splits", "mean_abs_shap"]),
            "",
        ]

    lines += [
        "## Caveats",
        "",
        "- GO:0001772 includes all taxa (matching the team's 5,614-annotation download); most are IEA (electronic) "
        "annotations mapped by upper-casing symbols. `--immune-human-only` restricts to genes with a human annotation.",
        "- SynGO annotations are mostly from rodent brain synapses; membership says a gene is synaptic in neurons, not in glioma.",
        "- Gene-set membership is a weak label: unlabeled genes are unknown, not negative.",
        "- CellChat features are zero for most genes in the DEG universe (CellChat covers ~114 genes); DEG and n_cells carry most signal.",
        "- SynGO genes are enriched for broadly expressed genes, so the tumor-synaptic model partly learns \"highly expressed\"."
        " Mitochondrial / ribosomal / histone genes are flagged `housekeeping` and left out of the novel tables.",
        "- Immune genes (HLA class II, ITGB2, C3, CCL2) show up as strong Ephys_1-high DEGs inside *tumor* clusters, which"
        " suggests myeloid contamination / doublets in Ephys_1 tumor clusters. Immune-synapse nominees in tumor cell"
        " types should be checked for this before follow-up.",
        "- High score = \"expression pattern resembles known synaptic / immune-synapse genes\", not validation.",
    ]
    report = out_dir / "GENESETS_REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    paths.append(report)
    return paths
