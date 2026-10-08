"""Focused Week 4 outputs for baseline comparison and novel-gene review.

This module deliberately works from the already-computed out-of-fold scores.  It
does not refit either LightGBM model, which keeps the comparison and nominee
selection as a transparent reporting step over the same evaluation run.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd


PROGRAM_LABELS = {
    "tumor_synaptic": "Tumor synaptic (SynGO)",
    "immune_synapse": "Immune synapse (GO:0001772)",
}
SCORERS = {
    "lightgbm": "lightgbm_oof",
    "cellchat_probability": "cc_max_prob",
    "three_table_rule": "rule_pass",
}
METRICS = ("auroc", "auprc", "precision_at_25")


def _md_table(frame: pd.DataFrame, columns: list[str]) -> str:
    """Render the small report tables without importing the data-loading pipeline."""
    def fmt(value) -> str:
        if isinstance(value, float):
            return "" if pd.isna(value) else f"{value:.3g}"
        return str(value).replace("|", "/")

    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for record in frame[columns].itertuples(index=False):
        lines.append("| " + " | ".join(fmt(value) for value in record) + " |")
    return "\n".join(lines)


def build_baseline_comparison(metrics: Mapping) -> pd.DataFrame:
    """Return a slide-ready comparison of LightGBM with the two requested baselines."""
    rows: list[dict] = []
    for program, block in metrics.get("sets", {}).items():
        row_level = block.get("row_level", {})
        for metric in METRICS:
            values = {
                output_name: float(row_level[source_name][metric])
                for output_name, source_name in SCORERS.items()
            }
            best_method = max(values, key=values.get)
            best_baseline = max(values["cellchat_probability"], values["three_table_rule"])
            rows.append(
                {
                    "program": program,
                    "program_label": PROGRAM_LABELS.get(program, program),
                    "evaluation_level": "gene_context_row",
                    "metric": metric,
                    "labeled_genes": int(
                        block.get("set_genes_in_universe", block.get("n_positive_genes", 0))
                    ),
                    "positive_rows": int(block.get("n_positive_rows", 0)),
                    "prevalence": float(block.get("prevalence_rows", float("nan"))),
                    **values,
                    "best_method": best_method,
                    "best_baseline": best_baseline,
                    "delta_vs_best_baseline": round(values["lightgbm"] - best_baseline, 12),
                    "lightgbm_beats_both": values["lightgbm"] > best_baseline,
                }
            )
    return pd.DataFrame(rows)


def build_novel_nominees(
    scores: pd.DataFrame,
    top_per_program: int = 25,
    require_three_table: bool = True,
) -> pd.DataFrame:
    """Select the highest-scoring genes outside both public sets and the published 33.

    A gene is reported once per program, at its best-scoring cell type x IDH
    context.  The strict table additionally requires the existing three-table
    evidence rule (CellChat + DEG + sufficient cell count).
    """
    required_flags = ("in_tumor_synaptic", "in_immune_synapse", "in_published_33")
    missing = [column for column in required_flags if column not in scores]
    if missing:
        raise ValueError(f"Missing nominee exclusion columns: {', '.join(missing)}")

    eligible = scores.copy()
    for column in required_flags:
        eligible = eligible[eligible[column].fillna(0).astype(int) == 0]
    if "housekeeping" in eligible:
        eligible = eligible[eligible["housekeeping"].fillna(0).astype(int) == 0]
    if require_three_table:
        eligible = eligible[eligible["rule_pass"].fillna(0).astype(int) == 1]

    parts: list[pd.DataFrame] = []
    for program in PROGRAM_LABELS:
        score_column = f"p_{program}"
        if score_column not in eligible:
            continue
        ranked = (
            eligible.dropna(subset=[score_column])
            .sort_values([score_column, "gene"], ascending=[False, True])
            .drop_duplicates("gene")
            .head(top_per_program)
            .copy()
        )
        ranked.insert(0, "program", program)
        ranked.insert(1, "program_label", PROGRAM_LABELS[program])
        ranked.insert(2, "nominee_rank", range(1, len(ranked) + 1))
        ranked.insert(4, "model_score", ranked[score_column])
        ranked["passes_three_table_rule"] = ranked["rule_pass"].fillna(0).astype(int)
        ranked["selection_mode"] = (
            "high_score_and_three_table_support" if require_three_table else "high_score_only"
        )
        reason_column = f"why_{program}"
        ranked["top_model_reasons"] = ranked.get(reason_column, "")
        parts.append(ranked)

    if not parts:
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)
    preferred = [
        "program",
        "program_label",
        "nominee_rank",
        "gene",
        "model_score",
        "celltype_id",
        "IDH_status",
        "passes_three_table_rule",
        "deg_log2fc",
        "deg_neglog10_padj",
        "cc_edges",
        "cc_max_prob",
        "ctx_min_n",
        "top_model_reasons",
        "in_tumor_synaptic",
        "in_immune_synapse",
        "in_published_33",
        "housekeeping",
        "selection_mode",
    ]
    return combined[[column for column in preferred if column in combined.columns]]


def _baseline_conclusion(comparison: pd.DataFrame, program: str) -> str:
    subset = comparison[comparison["program"] == program]
    if subset.empty:
        return "No results were available."
    wins = subset.loc[subset["lightgbm_beats_both"], "metric"].tolist()
    losses = subset.loc[~subset["lightgbm_beats_both"], "metric"].tolist()
    parts = []
    if wins:
        parts.append("LightGBM beats both baselines for " + ", ".join(wins))
    if losses:
        parts.append("it does not beat both for " + ", ".join(losses))
    return "; ".join(parts) + "."


def write_week4_focus_outputs(result, out_dir: Path, top_per_program: int = 25) -> dict[str, Path]:
    """Write the focused Task 6/7 CSVs plus a concise audited Markdown summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison = build_baseline_comparison(result.metrics)
    nominees = build_novel_nominees(result.scores, top_per_program, require_three_table=True)
    score_only = build_novel_nominees(result.scores, top_per_program, require_three_table=False)

    paths = {
        "baseline_csv": out_dir / "baseline_comparison.csv",
        "nominees_csv": out_dir / "novel_nominees.csv",
        "score_only_csv": out_dir / "novel_nominees_score_only.csv",
        "summary_md": out_dir / "WEEK4_BASELINES_AND_NOMINEES.md",
    }
    comparison.to_csv(paths["baseline_csv"], index=False)
    nominees.to_csv(paths["nominees_csv"], index=False)
    score_only.to_csv(paths["score_only_csv"], index=False)

    display_comparison = comparison.copy()
    for column in ("lightgbm", "cellchat_probability", "three_table_rule", "delta_vs_best_baseline"):
        if column in display_comparison:
            display_comparison[column] = display_comparison[column].round(3)
    nominee_columns = [
        "program_label",
        "nominee_rank",
        "gene",
        "model_score",
        "celltype_id",
        "IDH_status",
        "deg_log2fc",
        "cc_edges",
        "ctx_min_n",
    ]
    lines = [
        "# Week 4: baseline comparison and novel nominees",
        "",
        "## Run facts",
        "",
        f"- Universe: `{result.metrics.get('universe', 'unknown')}`; "
        f"{result.metrics.get('n_rows', 'unknown')} gene-context rows, "
        f"{result.metrics.get('n_genes', 'unknown')} genes, and "
        f"{result.metrics.get('n_features', 'unknown')} numeric features.",
        f"- Cross-validation: {result.metrics.get('cv', 'not recorded')}; "
        f"seeds {result.metrics.get('seeds', 'not recorded')}.",
        "- SynGO and GO:0001772 provide the training labels. The published 33 are excluded from "
        "training-label construction and from the novel-nominee list.",
        "",
        "## Does LightGBM beat the baselines?",
        "",
        "The comparison uses the same out-of-fold gene-context scores for LightGBM, "
        "CellChat maximum probability, and the three-table rule.",
        "",
        _md_table(display_comparison, list(display_comparison.columns)),
        "",
    ]
    for program in PROGRAM_LABELS:
        lines.append(f"- **{PROGRAM_LABELS[program]}:** {_baseline_conclusion(comparison, program)}")
    lines += [
        "",
        "## Novel nominees",
        "",
        "Strict nominees are the top model-scored genes that are absent from both public label sets "
        "(SynGO and GO:0001772), absent from the published 33, not flagged as housekeeping, and "
        "supported by the CellChat + DEG + cell-count rule.",
        "",
        _md_table(nominees, [c for c in nominee_columns if c in nominees]),
        "",
        "`novel_nominees_score_only.csv` is the sensitivity list: it applies the same exclusions but "
        "does not require the three-table rule.",
        "",
        "## Interpretation guardrails",
        "",
        "- AUROC measures broad separation; AUPRC and precision@25 are more informative for the short candidate list.",
        "- Public gene-set membership is a weak label: genes outside the sets are unknown, not confirmed negatives.",
        "- A high model score nominates a gene for validation; it does not establish glioma function or causality.",
        "- Immune-score nominees whose best context is a tumor cluster need a contamination/doublet check before follow-up.",
    ]
    paths["summary_md"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths
