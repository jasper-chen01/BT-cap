# LightGBM gene nominees (gene x cell type x IDH)

- Universe: `cellchat`. 1368 rows, 114 genes, 43 features.
- Labels: literature tiers [1, 2] from `knowledge/literature_positives.yaml`: 200 positive rows (33 genes). Everything else is unlabeled, not negative.
- Scores are out-of-fold: StratifiedGroupKFold(n_splits=5) grouped by gene, averaged over seeds [0, 1, 2, 3, 4].
- LLM run features: 2 runs from `..\turnin\week2_traces`.

## Does the model beat single-table baselines?

Row-level prevalence = 0.146 (AUPRC of a random ranking).

| scorer | auroc | auprc | precision_at_10 | precision_at_25 | precision_at_50 |
|---|---|---|---|---|---|
| lightgbm_oof | 0.675 | 0.386 | 1 | 0.76 | 0.64 |
| rule_pass | 0.631 | 0.226 | 0 | 0.2 | 0.34 |
| deg_abs_log2fc | 0.612 | 0.188 | 0.2 | 0.12 | 0.1 |
| cc_edges | 0.656 | 0.238 | 0.4 | 0.24 | 0.32 |
| cc_max_prob | 0.649 | 0.243 | 0.4 | 0.56 | 0.32 |

Gene level (best context per gene):

| scorer | auroc | auprc | precision_at_10 | precision_at_25 | precision_at_50 |
|---|---|---|---|---|---|
| lightgbm_oof | 0.68 | 0.57 | 0.7 | 0.64 | 0.44 |
| deg_abs_log2fc | 0.691 | 0.457 | 0.5 | 0.44 | 0.46 |
| cc_edges | 0.69 | 0.562 | 0.7 | 0.52 | 0.44 |

## Top 5 contexts

| rank | gene | celltype_id | IDH_status | score | status | rule_pass | deg_log2fc | deg_neglog10_padj | cc_edges | cc_ephys_bias | ctx_min_n | top_reasons |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | GRIA4 | cycling_tumor | IDH_Mutant | 0.953 | literature tier 2 (label) | 1 | 3.82 | 38.1 | 3 | 1 | 136 | deg_pooled_log2fc=1.86 (+0.76); cc_n_partners=2 (+0.50); cc_edges_total=15 (+0.42) |
| 2 | PTPRZ1 | cycling_tumor | IDH_Mutant | 0.935 | literature tier 1 (label) | 1 | 4 | 54.1 | 17 | 0.176 | 136 | llm_ctx_mention_rate=1 (+0.87); deg_pooled_log2fc=2.05 (+0.75); cc_edges_total=77 (+0.67) |
| 3 | NRXN1 | cycling_tumor | IDH_Mutant | 0.932 | literature tier 2 (label) | 1 | 3.71 | 44.2 | 10 | 1 | 136 | cc_edges_total=50 (+0.87); llm_ctx_mention_rate=1 (+0.66); deg_pooled_log2fc=2.04 (+0.62) |
| 4 | GRIA4 | AC_like_tumor | IDH_Mutant | 0.93 | literature tier 2 (label) | 1 | 1.73 | 22.5 | 3 | 1 | 283 | deg_pooled_log2fc=1.49 (+0.69); cc_n_partners=2 (+0.51); cc_edges_total=15 (+0.41) |
| 5 | PTPRZ1 | cycling_tumor | IDH_WT | 0.929 | literature tier 1 (label) | 1 | 2.07 | 201 | 17 | 0.176 | 663 | llm_ctx_mention_rate=1 (+0.90); deg_pooled_log2fc=2.05 (+0.75); cc_edges_total=77 (+0.67) |

## Top novel nominees that pass the 3-table rule

Not in the literature list, but CellChat edge + DEG in this cell type/IDH + adequate n_cells.

| rank | gene | celltype_id | IDH_status | score | status | rule_pass | deg_log2fc | deg_neglog10_padj | cc_edges | cc_ephys_bias | ctx_min_n | top_reasons |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | GRIK2 | cycling_tumor | IDH_WT | 0.885 | novel | 1 | 4.53 | 300 | 3 | 1 | 663 | deg_pooled_log2fc=4.63 (+0.54); cc_n_partners=2 (+0.51); llm_ctx_mention_rate=0.5 (+0.27) |
| 17 | GRIK2 | cycling_tumor | IDH_Mutant | 0.883 | novel | 1 | 4.4 | 18.6 | 3 | 1 | 136 | deg_pooled_log2fc=4.63 (+0.54); cc_n_partners=2 (+0.51); llm_ctx_mention_rate=0.5 (+0.27) |
| 18 | GRIK2 | OPC_GABA_like_tumor | IDH_WT | 0.875 | novel | 1 | 2.36 | 255 | 3 | 1 | 1935 | cc_n_partners=2 (+0.52); deg_pooled_log2fc=1.95 (+0.52); llm_ctx_mention_rate=0.5 (+0.27) |
| 19 | GRIK2 | OPC_GABA_like_tumor | IDH_Mutant | 0.86 | novel | 1 | 0.708 | 46.9 | 3 | 1 | 1658 | deg_pooled_log2fc=1.95 (+0.52); cc_n_partners=2 (+0.52); llm_ctx_mention_rate=0.5 (+0.27) |
| 22 | NCL | cycling_tumor | IDH_Mutant | 0.851 | novel | 1 | 0.515 | 2.83 | 16 | 0.125 | 136 | cc_edges_total=70 (+0.58); llm_mentions_per_run=1 (+0.46); cc_n_cells_e1=6.41e+03 (+0.23) |
| 23 | NCL | cycling_tumor | IDH_WT | 0.851 | novel | 1 | 0.951 | 81.6 | 16 | 0.125 | 663 | cc_edges_total=70 (+0.58); llm_mentions_per_run=1 (+0.46); cc_n_cells_e1=6.41e+03 (+0.23) |
| 26 | BSG | cycling_tumor | IDH_WT | 0.842 | novel | 1 | 1.63 | 201 | 13 | 1 | 663 | cc_edges_total=56 (+0.66); deg_pooled_log2fc=1.47 (+0.61); cc_n_cells_e1=6.41e+03 (+0.23) |
| 34 | BSG | OPC_GABA_AC_like_tumor | IDH_WT | 0.823 | novel | 1 | 1.32 | 14.5 | 11 | 1 | 171 | cc_edges_total=56 (+0.67); cc_n_cells_e1=6.2e+03 (+0.22); llm_ctx_mention_rate=0.5 (+0.16) |
| 40 | BSG | OPC_GABA_AC_like_tumor | IDH_Mutant | 0.803 | novel | 1 | 0.569 | 1.36 | 11 | 1 | 352 | cc_edges_total=56 (+0.65); cc_n_cells_e1=6.2e+03 (+0.21); llm_ctx_mention_rate=0.5 (+0.15) |
| 41 | NCL | AC_like_tumor | IDH_Mutant | 0.803 | novel | 1 | 0.753 | 13.2 | 10 | 0.8 | 283 | cc_edges_total=70 (+0.60); llm_mentions_per_run=1 (+0.46); cc_n_cells_e1=7.25e+04 (+0.23) |
| 42 | NCL | OPC_GABA_like_tumor | IDH_WT | 0.803 | novel | 1 | 0.878 | 169 | 18 | 0 | 1935 | cc_edges_total=70 (+0.60); llm_mentions_per_run=1 (+0.46); cc_n_cells_e1=6.74e+03 (+0.23) |
| 43 | NCL | OPC_GABA_like_tumor | IDH_Mutant | 0.802 | novel | 1 | 0.585 | 75.5 | 18 | 0 | 1658 | cc_edges_total=70 (+0.60); llm_mentions_per_run=1 (+0.46); cc_n_cells_e1=6.74e+03 (+0.23) |
| 45 | GLS | TAM1/TAM2/microglia | IDH_Mutant | 0.8 | novel | 1 | 1.11 | 59.5 | 13 | 1 | 2106 | llm_ctx_mention_rate=1 (+0.83); llm_mention_rate=1 (+0.24); cc_n_cells_e1=1.21e+04 (+0.23) |
| 46 | GLS | TAM1/TAM2/microglia | IDH_WT | 0.797 | novel | 1 | 0.889 | 71.8 | 13 | 1 | 2745 | llm_ctx_mention_rate=1 (+0.83); llm_mention_rate=1 (+0.25); cc_n_cells_e1=1.21e+04 (+0.23) |
| 49 | BSG | OPC_GABA_like_tumor | IDH_WT | 0.796 | novel | 1 | 1.01 | 185 | 26 | 0 | 1935 | cc_edges_total=56 (+0.65); cc_n_cells_e1=6.74e+03 (+0.23); llm_ctx_mention_rate=0.5 (+0.16) |
| 55 | DAG1 | cycling_tumor | IDH_Mutant | 0.772 | novel | 1 | 2.37 | 7.61 | 6 | 1 | 136 | llm_mentions_per_run=1.5 (+0.47); llm_ctx_mention_rate=0.5 (+0.26); llm_mention_rate=0.5 (+0.23) |
| 62 | DAG1 | cycling_tumor | IDH_WT | 0.753 | novel | 1 | 1.16 | 128 | 6 | 1 | 663 | llm_mentions_per_run=1.5 (+0.46); llm_ctx_mention_rate=0.5 (+0.25); llm_mention_rate=0.5 (+0.22) |
| 75 | LRRTM2 | cycling_tumor | IDH_Mutant | 0.725 | novel | 1 | 4.01 | 5.28 | 6 | 1 | 136 | llm_ctx_mention_rate=0.5 (+0.26); cc_n_cells_e1=6.41e+03 (+0.21); llm_mention_rate=0.5 (+0.20) |
| 76 | MPZL1 | cycling_tumor | IDH_WT | 0.724 | novel | 1 | 2.13 | 192 | 4 | 1 | 663 | deg_pooled_log2fc=1.86 (+0.59); cc_n_partners=2 (+0.53); deg_n_celltypes_sig=4 (+0.28) |
| 83 | APP | cycling_tumor | IDH_Mutant | 0.714 | novel | 1 | 3.05 | 30.2 | 13 | 1 | 136 | cc_edges_total=78 (+0.90); cc_n_cells_e1=6.41e+03 (+0.22); deg_pct_diff=0.588 (+0.13) |

## Top genes (best context per gene)

| gene_rank | gene | celltype_id | IDH_status | score | status | rule_pass | top_reasons |
|---|---|---|---|---|---|---|---|
| 1 | GRIA4 | cycling_tumor | IDH_Mutant | 0.953 | literature tier 2 (label) | 1 | deg_pooled_log2fc=1.86 (+0.76); cc_n_partners=2 (+0.50); cc_edges_total=15 (+0.42) |
| 2 | PTPRZ1 | cycling_tumor | IDH_Mutant | 0.935 | literature tier 1 (label) | 1 | llm_ctx_mention_rate=1 (+0.87); deg_pooled_log2fc=2.05 (+0.75); cc_edges_total=77 (+0.67) |
| 3 | NRXN1 | cycling_tumor | IDH_Mutant | 0.932 | literature tier 2 (label) | 1 | cc_edges_total=50 (+0.87); llm_ctx_mention_rate=1 (+0.66); deg_pooled_log2fc=2.04 (+0.62) |
| 4 | NLGN1 | cycling_tumor | IDH_WT | 0.918 | literature tier 2 (label) | 1 | llm_ctx_mention_rate=1 (+0.93); deg_pooled_log2fc=2.04 (+0.68); cc_edges_total=18 (+0.47) |
| 5 | GRIK2 | cycling_tumor | IDH_WT | 0.885 | novel | 1 | deg_pooled_log2fc=4.63 (+0.54); cc_n_partners=2 (+0.51); llm_ctx_mention_rate=0.5 (+0.27) |

## Feature importance (seed-averaged gain)

| feature | gain | splits | mean_abs_shap |
|---|---|---|---|
| cc_edges_total | 2.72e+03 | 91.4 | 0.392 |
| cc_n_partners | 1.34e+03 | 21.8 | 0.122 |
| cc_n_cells_e1 | 1.01e+03 | 24.4 | 0.346 |
| llm_mentions_per_run | 936 | 32.6 | 0.123 |
| deg_pooled_log2fc | 932 | 36.6 | 0.129 |
| deg_n_celltypes_sig | 882 | 42.8 | 0.153 |
| cc_n_celltypes_total | 740 | 34.4 | 0.112 |
| llm_ctx_mention_rate | 739 | 27.2 | 0.128 |
| llm_mention_rate | 602 | 15.2 | 0.171 |
| cc_edges_e2 | 446 | 7.8 | 0.0598 |
| deg_pct_diff | 315 | 10.6 | 0.0554 |
| llm_unprompted_mention_rate | 192 | 6.4 | 0.0288 |
| cc_mean_prob | 143 | 9 | 0.0223 |
| cc_n_cells_e2 | 128 | 6 | 0.0397 |
| deg_log2fc | 124 | 4.2 | 0.0263 |

## Caveats

- The positive list is a small, hand-curated draft. Labeled genes are biased toward well-studied biology.
- Tier 2 genes are pathway-level picks that overlap the H1-H3 hypotheses. `--tiers 1` is the strict check; with tier 1 alone (16 genes) the model does not beat single-column baselines.
- CellChat is not IDH-stratified, so `cc_*` features repeat across IDH groups; only DEG and n features differ by IDH.
- High scores mean "looks like known glioma communication genes in these tables", not validation.
- `top_reasons` are SHAP contributions from the full-data model; scores themselves are out-of-fold.
