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
from ephys_rag.scetm_features import DEFAULT_EMBEDDINGS, KNN_FEATURE, NeighbourIndex, load_embeddings

SET_NAMES = tuple(GENE_SETS)
# tables: week-4 DEG / CellChat / n_cells features. tables_scetm: + Level 1 (rho) + Level 2 (knn
# label fraction). scetm_only: Level 1 + Level 2 with no fold-change or table features at all.
ARMS = ("tables", "tables_scetm", "scetm_only")
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
    arm: str = "tables",
    embeddings: Path | None = None,
    knn_k: int = 20,
) -> GeneSetResult:
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    table = build_feature_table(
        universe=universe, llm_runs=llm_runs, llm_keys=llm_keys, compartment_features=True
    )
    frame = table.frame.copy()
    features = list(table.feature_columns)
    emb = None
    emb_info: dict = {}
    if arm != "tables":
        emb_path = Path(embeddings or DEFAULT_EMBEDDINGS)
        emb = load_embeddings(emb_path)
        frame = frame.merge(emb, left_on="gene", right_index=True, how="left")
        universe_genes = frame["gene"].unique()
        emb_info = {
            "path": str(emb_path),
            "dim": emb.shape[1],
            "n_genes_embedded": int(emb.shape[0]),
            "universe_genes_embedded": int(np.isin(universe_genes, emb.index).sum()),
            "universe_genes": int(len(universe_genes)),
            "knn_k": knn_k,
        }
        features = (features if arm == "tables_scetm" else []) + list(emb.columns)

    literature = load_literature()
    published = set(literature["gene"])
    tier = dict(zip(literature["gene"], literature["lit_tier"]))
    frame["in_published_33"] = frame["gene"].isin(published).astype(int)
    frame["published_tier"] = frame["gene"].map(tier)

    metrics: dict = {
        "arm": arm,
        "universe": universe,
        "n_rows": int(len(frame)),
        "n_genes": int(frame["gene"].nunique()),
        "n_features": len(features) + (1 if emb is not None else 0),
        "features": features + (["knn_frac_<set> (per fold)"] if emb is not None else []),
        "embeddings": emb_info,
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

        row_genes = sub["gene"].to_numpy()
        fold_features = None
        if emb is not None:
            knn_col = KNN_FEATURE.get(name, f"knn_frac_{name}")
            neighbours = NeighbourIndex(emb, row_genes, k=knn_k)

            def fold_features(train_idx, neighbours=neighbours, genes=genes, knn_col=knn_col, index=X.index):
                values = neighbours.feature(row_genes, genes, set(row_genes[train_idx]))
                return pd.DataFrame({knn_col: values}, index=index)

        score, sd = oof_scores(X, y, row_genes, seeds, n_splits, fold_features=fold_features)
        frame.loc[mask, score_col] = score
        frame.loc[mask, f"{score_col}_sd"] = sd
        if emb is not None:
            # Explanations come from a model fit on every gene; its knn feature uses all labelled
            # genes as neighbours (self still excluded). Scores above never see this column.
            full_knn = neighbours.feature(row_genes, genes, set(row_genes))
            X = X.assign(**{knn_col: full_knn})
            frame.loc[mask, knn_col] = full_knn
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
    keep += SUMMARY_FEATURES + [c for c in KNN_FEATURE.values() if c in result.scores]
    keep += [f"why_{name}" for name in sets]
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
        f"- Feature arm: `{m.get('arm', 'tables')}`.",
    ]
    e = m.get("embeddings") or {}
    if e:
        lines.append(
            f"- scETM embeddings: `{Path(e['path']).name}` ({e['dim']}-d, {e['n_genes_embedded']} genes); "
            f"{e['universe_genes_embedded']} of {e['universe_genes']} universe genes embedded (others get NaN). "
            f"Level 2 = fraction of {e['knn_k']} nearest neighbours in the label set, rebuilt per fold from training genes."
        )
    lines.append("")
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


# --------------------------------------------------------------------------- arm comparison


def arm_checks(result: GeneSetResult, top: int = 100) -> dict:
    """Gene-level checks per model: best context per gene, out-of-fold scores."""
    out: dict = {}
    sets = list(result.metrics["sets"])
    for name in sets:
        col = f"p_{name}"
        genes = result.by_gene.dropna(subset=[col]).sort_values(col, ascending=False).reset_index(drop=True)
        genes["percentile"] = 1 - genes.index / len(genes)
        head = genes.head(top)
        housekeeping = head["gene"].str.match(HOUSEKEEPING)
        s = result.metrics["sets"][name]
        check = s["published_check"]
        labelled = genes[f"in_{name}"] == 1
        others = {o: int(head[f"in_{o}"].sum()) for o in sets if o != name}
        clean = genes[~genes["gene"].str.match(HOUSEKEEPING)].head(top)
        out[name] = {
            "auroc_rows": s["row_level"]["lightgbm_oof"]["auroc"],
            "auprc_rows": s["row_level"]["lightgbm_oof"]["auprc"],
            "auroc_genes": s["gene_level"]["lightgbm_oof"]["auroc"],
            "auprc_genes": s["gene_level"]["lightgbm_oof"]["auprc"],
            "n_genes_scored": int(len(genes)),
            f"housekeeping_top{top}": int(housekeeping.sum()),
            f"housekeeping_in_label_set_top{top}": int(head.loc[housekeeping, f"in_{name}"].sum()),
            f"housekeeping_genes_top{top}": head.loc[housekeeping, "gene"].tolist(),
            f"label_set_top{top}": int(head[f"in_{name}"].sum()),
            f"label_set_top{top}_excl_housekeeping": int(clean[f"in_{name}"].sum()),
            "top_genes_excl_housekeeping": clean["gene"].head(25).tolist(),
            "label_set_median_percentile": float(genes.loc[labelled, "percentile"].median()),
            **{f"{o}_genes_top{top}": n for o, n in others.items()},
            "published_median_percentile": check.get("median_percentile_published"),
            "published_auroc": check.get("all_genes", {}).get("auroc"),
            "top_genes": genes["gene"].head(25).tolist(),
        }
    return out


def compare_arms(
    arms: Sequence[str] = ARMS,
    embeddings: Path | None = None,
    knn_k: int = 20,
    top: int = 100,
    **kwargs,
) -> tuple[dict[str, GeneSetResult], dict]:
    results = {arm: rank_gene_sets(arm=arm, embeddings=embeddings, knn_k=knn_k, **kwargs) for arm in arms}
    checks = {arm: arm_checks(res, top=top) for arm, res in results.items()}
    return results, checks


def _rank_table(results: dict[str, GeneSetResult]) -> pd.DataFrame:
    """One row per gene: gene-level rank under each arm and model."""
    parts = []
    for arm, res in results.items():
        cols = {c: f"{c}__{arm}" for c in res.by_gene.columns if c.startswith(("p_", "gene_rank_"))}
        parts.append(res.by_gene.set_index("gene")[list(cols)].rename(columns=cols))
    first = next(iter(results.values())).by_gene.set_index("gene")
    flags = first[[c for c in first.columns if c.startswith("in_")] + ["published_tier"]]
    table = pd.concat([flags] + parts, axis=1).reset_index()
    table["housekeeping"] = table["gene"].str.match(HOUSEKEEPING).astype(int)
    return table


def write_arm_outputs(
    results: dict[str, GeneSetResult], checks: dict, out_dir: Path, top: int = 100, top_novel: int = 25
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for arm, res in results.items():
        paths += write_gene_set_outputs(res, out_dir / arm, top=top_novel)
    ranks = _rank_table(results)
    paths.append(out_dir / "arms_gene_ranks.csv")
    ranks.to_csv(paths[-1], index=False)
    paths.append(out_dir / "arms_checks.json")
    paths[-1].write_text(json.dumps(checks, indent=2, default=str), encoding="utf-8")

    arms = list(results)
    sets = list(next(iter(results.values())).metrics["sets"])
    first = results[arms[0]].metrics
    e = next((r.metrics["embeddings"] for r in results.values() if r.metrics.get("embeddings")), {})
    lines = [
        "# Week 5: do scETM gene embeddings help the gene-set rankers?",
        "",
        f"- Universe: `{first['universe']}`: {first['n_rows']} gene x cell type x IDH rows, {first['n_genes']} genes.",
        f"- Scores are out-of-fold ({first['cv']}, seeds {first['seeds']}); checks use each gene's best context.",
        "- Arms: `tables` = week-4 DEG / CellChat / n_cells features; `tables_scetm` = tables + Level 1 (`emb_00..`, "
        "the gene's rho vector) + Level 2 (`knn_frac_*`); `scetm_only` = Level 1 + Level 2 with no fold-change or table features.",
    ]
    if e:
        lines.append(
            f"- Embeddings: `{Path(e['path']).name}` ({e['dim']}-d); {e['universe_genes_embedded']} of {e['universe_genes']} "
            f"universe genes embedded. Level 2 uses k={e['knn_k']} cosine neighbours, rebuilt inside every CV fold from "
            "training-fold genes only, and a gene is never its own neighbour."
        )
    lines.append("")

    for name in sets:
        rows = []
        for arm in arms:
            c = checks[arm][name]
            other = {k: v for k, v in c.items() if k.endswith(f"_genes_top{top}") and not k.startswith("housekeeping")}
            rows.append(
                {
                    "arm": arm,
                    "AUROC rows": c["auroc_rows"],
                    "AUPRC rows": c["auprc_rows"],
                    "AUROC genes": c["auroc_genes"],
                    "AUPRC genes": c["auprc_genes"],
                    f"housekeeping in top {top}": c[f"housekeeping_top{top}"],
                    "of which in label set": c[f"housekeeping_in_label_set_top{top}"],
                    f"label-set genes in top {top}": c[f"label_set_top{top}"],
                    f"label-set genes in top {top} (housekeeping removed)": c[f"label_set_top{top}_excl_housekeeping"],
                    "label-set median pct": c["label_set_median_percentile"],
                    **{f"{k.replace(f'_genes_top{top}', '')} genes in top {top}": v for k, v in other.items()},
                    "published-33 median pct": c["published_median_percentile"],
                    "published-33 AUROC": c["published_auroc"],
                }
            )
        table = pd.DataFrame(rows)
        lines += [f"## {name}", "", _md_table(table, list(table.columns)), ""]
        for key, title in [("top_genes", "Top 15 genes per arm"), ("top_genes_excl_housekeeping", "Top 15 genes per arm, housekeeping removed")]:
            top_table = pd.DataFrame({arm: checks[arm][name][key][:15] for arm in arms})
            top_table.insert(0, "rank", range(1, len(top_table) + 1))
            lines += [f"**{title}** (best context per gene)", "", _md_table(top_table, list(top_table.columns)), ""]
        lines += [f"**Feature importance share (gain)**", ""]
        for arm in arms:
            imp = results[arm].importance[name]
            share = imp.set_index("feature")["gain"] / max(imp["gain"].sum(), 1e-12)
            emb_share = share[share.index.str.startswith("emb_")].sum()
            knn_share = share[share.index.str.startswith("knn_")].sum()
            top3 = ", ".join(f"{f} {v:.2f}" for f, v in share.head(3).items())
            lines.append(
                f"- `{arm}`: Level 1 (emb_*) {emb_share:.2f}, Level 2 (knn) {knn_share:.2f}, "
                f"tables {max(1 - emb_share - knn_share, 0.0):.2f}. Top: {top3}."
            )
        lines.append("")
        for arm in arms:
            hk = checks[arm][name][f"housekeeping_genes_top{top}"]
            if hk:
                lines.append(f"- `{arm}` housekeeping genes in top {top}: {', '.join(hk)}")
        lines.append("")

    lines += [
        "## Notes",
        "",
        "- Housekeeping = the week-4 regex (MT-, RPL/RPS, histones, NDUF, COX, ATP5, EEF/EIF, ...). SynGO itself annotates "
        "many ribosomal / mitochondrial genes as synaptic, so housekeeping genes in the tumor-synaptic top 100 can be true label hits.",
        "- Level 1 and Level 2 are gene-wide: in `scetm_only` every cell type x IDH row of a gene gets the same score.",
        "- `knn_frac_*` in the per-arm `genesets_scores.csv` is the all-genes version used only for SHAP explanations; "
        "the scores themselves use the per-fold version.",
        "- Sanity check during development: with random 32-d vectors in place of rho, `scetm_only` scored AUROC 0.49 (tumor) "
        "and 0.43 (immune), i.e. no label leakage through the kNN feature.",
    ]
    report = out_dir / "SCETM_ARMS_REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    paths.append(report)
    return paths
