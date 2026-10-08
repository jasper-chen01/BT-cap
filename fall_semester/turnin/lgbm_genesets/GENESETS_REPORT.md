# LightGBM with knowledge-tool labels: tumor synaptic vs immune synapse

- Universe: `deg`: 27346 gene x cell type x IDH rows, 9580 genes, 42 features.
- Scores are out-of-fold (StratifiedGroupKFold(n_splits=5) grouped by gene, seeds [0, 1, 2, 3, 4]): no gene is scored by a model that trained on it.
- The 33 published glioma genes are **not** labels here. `in_published_33` flags them; the check below shows where each model ranks them.
- LLM run features: none.

## Label sources

- **tumor_synaptic**: SynGO  release 20250226: 5076 annotations -> 1789 genes. Fetched 2026-10-08.
- **immune_synapse**: QuickGO GO:0001772: 5614 annotations -> 244 genes (54 with a human annotation, 53 with experimental evidence). Fetched 2026-10-08.

## tumor_synaptic

SynGO synaptic genes, scored in tumor cell types (tumor synaptic program). Compartments: tumor.
1112 of 1789 set genes are in the universe; 3249 positive rows of 21843 (prevalence 0.149).

**Model vs single-column baselines (row level)**

| scorer | auroc | auprc | precision_at_10 | precision_at_25 | precision_at_50 |
|---|---|---|---|---|---|
| lightgbm_oof | 0.675 | 0.332 | 0.5 | 0.56 | 0.66 |
| rule_pass | 0.516 | 0.174 | 0.2 | 0.2 | 0.38 |
| deg_abs_log2fc | 0.507 | 0.154 | 0 | 0.04 | 0.02 |
| cc_edges | 0.514 | 0.164 | 0.6 | 0.64 | 0.54 |
| cc_max_prob | 0.514 | 0.165 | 0.9 | 0.76 | 0.64 |

**Gene level (best context per gene)**

| scorer | auroc | auprc | precision_at_10 | precision_at_25 | precision_at_50 |
|---|---|---|---|---|---|
| lightgbm_oof | 0.668 | 0.289 | 0.7 | 0.76 | 0.84 |
| deg_abs_log2fc | 0.539 | 0.146 | 0 | 0.04 | 0.02 |
| cc_edges | 0.512 | 0.14 | 0.6 | 0.48 | 0.54 |

**Published-33 check**

- Published genes already in this label set: APOE, CD44, CNTN1, EGFR, GRIA2, GRIA3, GRIA4, NCAM1, NLGN1, NLGN3, NOTCH1, NRXN1, PTN, PTPRZ1.
- Published genes scored: 30 of 33; not in universe: AXL, ENTPD1, KLRB1.
- Median gene-rank percentile of published genes: 0.98 (1.00 = top). AUROC published vs all other genes: 0.810 (single-column baselines, best context per gene: deg_abs_log2fc 0.906, deg_pct_e2 0.574, cc_edges 0.831, deg_n_celltypes_sig 0.758).
- Held-out test, genes **not** in the label set only (16 published): AUROC 0.698.

| gene | published_tier | in_label_set | celltype_id | IDH_status | best_score | gene_rank | gene_percentile |
|---|---|---|---|---|---|---|---|
| PTN | 1 | 1 | OPC_GABA_like_tumor | IDH_WT | 0.955 | 1 | 1 |
| PTPRZ1 | 1 | 1 | OPC_GABA_like_tumor | IDH_WT | 0.946 | 2 | 1 |
| NLGN1 | 2 | 1 | cycling_tumor | IDH_Mutant | 0.937 | 14 | 0.999 |
| GRIA2 | 1 | 1 | AC_like_tumor | IDH_Mutant | 0.935 | 22 | 0.998 |
| GRIA4 | 2 | 1 | cycling_tumor | IDH_Mutant | 0.922 | 78 | 0.991 |
| NCAM1 | 2 | 1 | OPC_GABA_like_tumor | IDH_Mutant | 0.906 | 105 | 0.988 |
| PDGFRA | 1 | 0 | cycling_tumor | IDH_Mutant | 0.897 | 116 | 0.987 |
| DLL3 | 1 | 0 | cycling_tumor | IDH_Mutant | 0.895 | 117 | 0.987 |
| CNTN1 | 2 | 1 | cycling_tumor | IDH_Mutant | 0.883 | 126 | 0.986 |
| CD99 | 2 | 0 | cycling_tumor | IDH_WT | 0.883 | 128 | 0.985 |
| NRXN1 | 2 | 1 | cycling_tumor | IDH_Mutant | 0.883 | 131 | 0.985 |
| NOTCH1 | 2 | 1 | cycling_tumor | IDH_WT | 0.878 | 134 | 0.985 |
| NLGN3 | 1 | 1 | cycling_tumor | IDH_Mutant | 0.878 | 135 | 0.985 |
| GRIA3 | 2 | 1 | cycling_tumor | IDH_Mutant | 0.875 | 136 | 0.985 |
| EGFR | 1 | 1 | cycling_tumor | IDH_Mutant | 0.865 | 142 | 0.984 |
| MIF | 1 | 0 | OPC_GABA_like_tumor | IDH_Mutant | 0.839 | 168 | 0.981 |
| SPP1 | 1 | 0 | AC_like_tumor | IDH_WT | 0.825 | 176 | 0.98 |
| HLA-DRA | 2 | 0 | AC_like_tumor | IDH_Mutant | 0.824 | 177 | 0.98 |
| TYROBP | 2 | 0 | AC_like_tumor | IDH_Mutant | 0.767 | 251 | 0.971 |
| APOE | 2 | 1 | AC_like_tumor | IDH_Mutant | 0.742 | 293 | 0.966 |
| CD74 | 1 | 0 | AC_like_tumor | IDH_WT | 0.723 | 316 | 0.964 |
| CXCR4 | 1 | 0 | cycling_tumor | IDH_WT | 0.503 | 2687 | 0.692 |
| LGALS9 | 2 | 0 | cycling_tumor | IDH_Mutant | 0.487 | 3025 | 0.653 |
| CD44 | 1 | 1 | OPC_GABA_like_tumor | IDH_WT | 0.44 | 4188 | 0.52 |
| CLEC2D | 1 | 0 | cycling_tumor | IDH_WT | 0.439 | 4227 | 0.515 |
| CD4 | 2 | 0 | cycling_tumor | IDH_Mutant | 0.43 | 4532 | 0.48 |
| HAVCR2 | 1 | 0 | cycling_tumor | IDH_Mutant | 0.403 | 5446 | 0.375 |
| TREM2 | 2 | 0 | cycling_tumor | IDH_WT | 0.366 | 6583 | 0.245 |
| LILRB4 | 2 | 0 | cycling_tumor | IDH_Mutant | 0.324 | 8057 | 0.0756 |
| TGFB1 | 2 | 0 | cycling_tumor | IDH_Mutant | 0.31 | 8365 | 0.0403 |

**Top 25 novel genes** (in neither gene set nor the published 33, housekeeping genes removed; best context per gene)

| gene | celltype_id | IDH_status | p_tumor_synaptic | rule_pass | deg_log2fc | deg_neglog10_padj | cc_edges | ctx_min_n | why_tumor_synaptic |
|---|---|---|---|---|---|---|---|---|---|
| NCL | OPC_GABA_like_tumor | IDH_Mutant | 0.935 | 1 | 0.585 | 75.5 | 18 | 1658 | deg_pct_e1=0.706 (+0.59); deg_pct_e2=0.834 (+0.39); cc_edges_total=70 (+0.30) |
| ZBTB20 | OPC_GABA_like_tumor | IDH_Mutant | 0.929 | 0 | 0.307 | 38.7 | 0 | 1658 | deg_pct_e1=0.898 (+1.42); deg_pct_e2=0.965 (+1.00); deg_n_celltypes_sig=5 (+0.12) |
| C1orf61 | OPC_GABA_like_tumor | IDH_Mutant | 0.913 | 0 | -1.58 | 300 | 0 | 1658 | deg_pct_e1=0.924 (+1.31); deg_pct_e2=0.945 (+0.99); deg_n_celltypes_sig=3 (+0.12) |
| KLRC2 | OPC_GABA_like_tumor | IDH_Mutant | 0.909 | 1 | 1.87 | 170 | 6 | 1658 | deg_pooled_log2fc=3.44 (+0.42); cc_edges=6 (+0.37); cc_edges_total=6 (+0.35) |
| CHCHD2 | OPC_GABA_like_tumor | IDH_WT | 0.908 | 0 | -0.402 | 29.1 | 0 | 1935 | deg_pct_e1=0.883 (+1.04); deg_pct_e2=0.959 (+0.96); deg_n_celltypes_sig=3 (+0.11) |
| DBI | cycling_tumor | IDH_WT | 0.907 | 0 | 0.35 | 12.4 | 0 | 663 | deg_pct_e1=0.935 (+1.42); deg_pct_e2=0.995 (+0.97); deg_pct_diff=0.06 (+0.07) |
| TMSB4X | OPC_GABA_like_tumor | IDH_WT | 0.901 | 0 | -1.67 | 300 | 0 | 1935 | deg_pct_e1=0.993 (+1.18); deg_pct_e2=0.978 (+0.99); deg_n_celltypes_sig=4 (+0.13) |
| SRI | OPC_GABA_like_tumor | IDH_Mutant | 0.894 | 0 | 1.13 | 300 | 0 | 1658 | deg_pct_e1=0.88 (+1.14); deg_pct_e2=0.97 (+0.98); deg_n_celltypes_sig=4 (+0.13) |
| BSG | OPC_GABA_AC_like_tumor | IDH_WT | 0.89 | 1 | 1.32 | 14.5 | 11 | 171 | cc_edges_total=56 (+0.47); cc_edges=11 (+0.27); cc_edges_e2=11 (+0.27) |
| TMSB10 | cycling_tumor | IDH_WT | 0.89 | 0 | -0.984 | 30.5 | 0 | 663 | deg_pct_e2=0.986 (+0.92); deg_pct_e1=0.853 (+0.82); deg_n_celltypes_sig=4 (+0.13) |
| HMGB1 | cycling_tumor | IDH_Mutant | 0.887 | 0 | 0.645 | 8.68 | 0 | 136 | deg_pct_e1=0.941 (+1.39); deg_pct_e2=1 (+0.95); deg_pct_diff=0.059 (+0.07) |
| HINT1 | cycling_tumor | IDH_WT | 0.886 | 0 | -0.566 | 12.6 | 0 | 663 | deg_pct_e2=0.98 (+0.92); deg_pct_e1=0.826 (+0.82); deg_n_celltypes_sig=3 (+0.11) |
| MYL6 | cycling_tumor | IDH_WT | 0.884 | 0 | -0.492 | 6.09 | 0 | 663 | deg_pct_e2=0.965 (+0.94); deg_pct_e1=0.835 (+0.83); deg_n_celltypes_sig=3 (+0.11) |
| PFDN5 | cycling_tumor | IDH_WT | 0.883 | 0 | -0.674 | 23 | 0 | 663 | deg_pct_e1=0.807 (+0.80); deg_pct_e2=0.94 (+0.78); deg_n_celltypes_sig=3 (+0.11) |
| ITM2B | OPC_GABA_like_tumor | IDH_Mutant | 0.883 | 0 | -1.12 | 300 | 0 | 1658 | deg_pct_e1=0.945 (+1.27); deg_pct_e2=0.905 (+0.54); deg_n_celltypes_sig=4 (+0.13) |
| ITGAV | OPC_GABA_like_tumor | IDH_Mutant | 0.879 | 1 | -0.936 | 54.2 | 14 | 1658 | cc_edges_total=39 (+0.62); cc_edges_e2=13 (+0.29); cc_edges=14 (+0.27) |
| B2M | cycling_tumor | IDH_WT | 0.874 | 0 | -1.7 | 128 | 0 | 663 | deg_pct_e1=0.946 (+1.05); deg_pct_e2=0.982 (+0.95); deg_n_celltypes_sig=4 (+0.14) |
| TSC22D1 | OPC_GABA_like_tumor | IDH_Mutant | 0.87 | 0 | 0.293 | 43.5 | 0 | 1658 | deg_pct_e1=0.88 (+1.03); deg_pct_e2=0.942 (+0.94); deg_pct_diff=0.062 (+0.07) |
| COL9A1 | cycling_tumor | IDH_WT | 0.87 | 1 | 4.56 | 300 | 5 | 663 | cc_edges_total=5 (+0.44); deg_pooled_log2fc=4.42 (+0.38); deg_log2fc=4.56 (+0.37) |
| JAM3 | cycling_tumor | IDH_WT | 0.87 | 1 | 2.63 | 205 | 3 | 663 | deg_pooled_log2fc=2.4 (+0.46); cc_edges=3 (+0.38); deg_log2fc=2.63 (+0.32) |
| CD63 | cycling_tumor | IDH_WT | 0.864 | 0 | -0.553 | 12.8 | 0 | 663 | deg_pct_e2=0.952 (+0.89); deg_pct_e1=0.806 (+0.78); deg_n_celltypes_sig=4 (+0.13) |
| HSP90AB1 | OPC_GABA_like_tumor | IDH_WT | 0.86 | 0 | 0.317 | 54.9 | 0 | 1935 | deg_pct_e1=0.874 (+1.05); deg_pct_e2=0.982 (+0.95); deg_n_celltypes_sig=3 (+0.11) |
| SLC1A3 | cycling_tumor | IDH_WT | 0.86 | 1 | -0.341 | 7.48 | 13 | 663 | cc_edges_total=26 (+0.74); cc_edges=13 (+0.29); cc_edges_e2=13 (+0.28) |
| GPM6B | OPC_GABA_like_tumor | IDH_Mutant | 0.86 | 0 | -1.55 | 300 | 0 | 1658 | deg_pct_e1=0.904 (+1.31); deg_pct_e2=0.908 (+0.53); deg_n_celltypes_sig=4 (+0.13) |
| UBC | OPC_GABA_like_tumor | IDH_Mutant | 0.86 | 0 | -0.597 | 72.4 | 0 | 1658 | deg_pct_e1=0.907 (+1.41); deg_pct_e2=0.886 (+0.54); deg_pct_diff=-0.021 (+0.08) |

**Novel genes that also pass the 3-table rule** (CellChat edge + DEG + n_cells)

| gene | celltype_id | IDH_status | p_tumor_synaptic | rule_pass | deg_log2fc | deg_neglog10_padj | cc_edges | ctx_min_n | why_tumor_synaptic |
|---|---|---|---|---|---|---|---|---|---|
| NCL | OPC_GABA_like_tumor | IDH_Mutant | 0.935 | 1 | 0.585 | 75.5 | 18 | 1658 | deg_pct_e1=0.706 (+0.59); deg_pct_e2=0.834 (+0.39); cc_edges_total=70 (+0.30) |
| KLRC2 | OPC_GABA_like_tumor | IDH_Mutant | 0.909 | 1 | 1.87 | 170 | 6 | 1658 | deg_pooled_log2fc=3.44 (+0.42); cc_edges=6 (+0.37); cc_edges_total=6 (+0.35) |
| BSG | OPC_GABA_AC_like_tumor | IDH_WT | 0.89 | 1 | 1.32 | 14.5 | 11 | 171 | cc_edges_total=56 (+0.47); cc_edges=11 (+0.27); cc_edges_e2=11 (+0.27) |
| ITGAV | OPC_GABA_like_tumor | IDH_Mutant | 0.879 | 1 | -0.936 | 54.2 | 14 | 1658 | cc_edges_total=39 (+0.62); cc_edges_e2=13 (+0.29); cc_edges=14 (+0.27) |
| COL9A1 | cycling_tumor | IDH_WT | 0.87 | 1 | 4.56 | 300 | 5 | 663 | cc_edges_total=5 (+0.44); deg_pooled_log2fc=4.42 (+0.38); deg_log2fc=4.56 (+0.37) |
| JAM3 | cycling_tumor | IDH_WT | 0.87 | 1 | 2.63 | 205 | 3 | 663 | deg_pooled_log2fc=2.4 (+0.46); cc_edges=3 (+0.38); deg_log2fc=2.63 (+0.32) |
| SLC1A3 | cycling_tumor | IDH_WT | 0.86 | 1 | -0.341 | 7.48 | 13 | 663 | cc_edges_total=26 (+0.74); cc_edges=13 (+0.29); cc_edges_e2=13 (+0.28) |
| MPZL1 | cycling_tumor | IDH_WT | 0.853 | 1 | 2.13 | 192 | 4 | 663 | cc_edges=4 (+0.38); cc_edges_total=8 (+0.36); deg_pooled_log2fc=1.86 (+0.34) |
| ROBO1 | cycling_tumor | IDH_Mutant | 0.848 | 1 | 3.24 | 18.5 | 1 | 136 | deg_pooled_log2fc=2.65 (+0.41); deg_log2fc=3.24 (+0.40); cc_edges_e2=1 (+0.16) |
| GPR37L1 | OPC_GABA_like_tumor | IDH_Mutant | 0.845 | 1 | 0.829 | 11.3 | 9 | 1658 | cc_edges_total=9 (+0.45); cc_edges_e2=9 (+0.29); cc_edges=9 (+0.28) |
| SDC3 | cycling_tumor | IDH_WT | 0.843 | 1 | 0.618 | 45.4 | 11 | 663 | cc_edges_total=11 (+0.78); cc_edges=11 (+0.29); cc_edges_e2=11 (+0.29) |
| DLL1 | OPC_GABA_like_tumor | IDH_WT | 0.813 | 1 | 3.43 | 286 | 2 | 1935 | deg_pooled_log2fc=2.83 (+0.46); deg_log2fc=3.43 (+0.37); cc_edges_total=4 (+0.15) |
| PTGES3 | cycling_tumor | IDH_WT | 0.812 | 1 | 0.601 | 51.2 | 2 | 663 | cc_edges_total=13 (+0.51); deg_pct_e2=0.925 (+0.41); deg_pct_e1=0.502 (+0.28) |
| PILRB | cycling_tumor | IDH_WT | 0.804 | 1 | 1.81 | 125 | 7 | 663 | cc_edges_total=7 (+0.45); cc_edges=7 (+0.37); cc_edges_e2=7 (+0.32) |
| JAM2 | cycling_tumor | IDH_Mutant | 0.798 | 1 | 0.796 | 7.6 | 6 | 136 | cc_edges_total=6 (+0.44); cc_edges=6 (+0.36); cc_edges_e2=6 (+0.30) |

**Feature importance (gain, top 10)**

| feature | gain | splits | mean_abs_shap |
|---|---|---|---|
| deg_pct_e1 | 1.1e+04 | 72.8 | 0.162 |
| deg_pct_e2 | 9.85e+03 | 56.6 | 0.15 |
| deg_pooled_log2fc | 5.04e+03 | 71 | 0.0825 |
| deg_log2fc | 2.99e+03 | 42.6 | 0.0581 |
| deg_n_celltypes_sig | 2.61e+03 | 17.2 | 0.0823 |
| cc_edges_total | 1.65e+03 | 27.4 | 0.0136 |
| deg_neglog10_padj | 1.08e+03 | 29.4 | 0.0423 |
| deg_pct_diff | 963 | 14.8 | 0.0198 |
| deg_abs_log2fc | 956 | 18.4 | 0.016 |
| ctx_n_e1 | 839 | 14.8 | 0.0332 |

## immune_synapse

GO:0001772 immunological synapse genes, scored on both sides of the synapse. Compartments: T_cell, TAM_microglia, tumor.
45 of 244 set genes are in the universe; 135 positive rows of 27346 (prevalence 0.005).

**Model vs single-column baselines (row level)**

| scorer | auroc | auprc | precision_at_10 | precision_at_25 | precision_at_50 |
|---|---|---|---|---|---|
| lightgbm_oof | 0.666 | 0.0151 | 0 | 0.04 | 0.08 |
| rule_pass | 0.555 | 0.0297 | 0.2 | 0.16 | 0.1 |
| deg_abs_log2fc | 0.54 | 0.00966 | 0.1 | 0.04 | 0.02 |
| cc_edges | 0.55 | 0.00914 | 0 | 0 | 0 |
| cc_max_prob | 0.55 | 0.00911 | 0 | 0 | 0 |

**Gene level (best context per gene)**

| scorer | auroc | auprc | precision_at_10 | precision_at_25 | precision_at_50 |
|---|---|---|---|---|---|
| lightgbm_oof | 0.676 | 0.0179 | 0.1 | 0.08 | 0.04 |
| deg_abs_log2fc | 0.533 | 0.0105 | 0.1 | 0.04 | 0.02 |
| cc_edges | 0.528 | 0.0062 | 0 | 0 | 0 |

**Published-33 check**

- Published genes already in this label set: HAVCR2, HLA-DRA.
- Published genes scored: 32 of 33; not in universe: AXL.
- Median gene-rank percentile of published genes: 0.91 (1.00 = top). AUROC published vs all other genes: 0.773 (single-column baselines, best context per gene: deg_abs_log2fc 0.901, deg_pct_e2 0.808, cc_edges 0.982, deg_n_celltypes_sig 0.737).
- Held-out test, genes **not** in the label set only (30 published): AUROC 0.767.

| gene | published_tier | in_label_set | celltype_id | IDH_status | best_score | gene_rank | gene_percentile |
|---|---|---|---|---|---|---|---|
| SPP1 | 1 | 0 | cycling_tumor | IDH_Mutant | 0.95 | 2 | 1 |
| APOE | 2 | 0 | OPC_GABA_like_tumor | IDH_WT | 0.938 | 3 | 1 |
| CD74 | 1 | 0 | cycling_tumor | IDH_WT | 0.925 | 4 | 1 |
| HLA-DRA | 2 | 1 | OPC_GABA_AC_like_tumor | IDH_WT | 0.892 | 10 | 0.999 |
| EGFR | 1 | 0 | OPC_GABA_AC_like_tumor | IDH_WT | 0.884 | 12 | 0.999 |
| CLEC2D | 1 | 0 | T cell | IDH_Mutant | 0.873 | 14 | 0.999 |
| CXCR4 | 1 | 0 | cycling_tumor | IDH_Mutant | 0.862 | 20 | 0.998 |
| TYROBP | 2 | 0 | cycling_tumor | IDH_WT | 0.847 | 31 | 0.997 |
| TREM2 | 2 | 0 | cycling_tumor | IDH_Mutant | 0.827 | 64 | 0.993 |
| CD44 | 1 | 0 | cycling_tumor | IDH_WT | 0.813 | 109 | 0.989 |
| LGALS9 | 2 | 0 | cycling_tumor | IDH_Mutant | 0.796 | 176 | 0.982 |
| TGFB1 | 2 | 0 | cycling_tumor | IDH_Mutant | 0.752 | 309 | 0.968 |
| MIF | 1 | 0 | TAM1/TAM2/microglia | IDH_WT | 0.727 | 376 | 0.961 |
| CD4 | 2 | 0 | cycling_tumor | IDH_Mutant | 0.707 | 449 | 0.953 |
| PTPRZ1 | 1 | 0 | T cell | IDH_WT | 0.701 | 475 | 0.951 |
| CD99 | 2 | 0 | AC_like_tumor | IDH_Mutant | 0.66 | 717 | 0.925 |
| LILRB4 | 2 | 0 | cycling_tumor | IDH_Mutant | 0.622 | 1039 | 0.892 |
| KLRB1 | 1 | 0 | T cell | IDH_Mutant | 0.603 | 1248 | 0.87 |
| DLL3 | 1 | 0 | AC_like_tumor | IDH_Mutant | 0.598 | 1317 | 0.863 |
| GRIA3 | 2 | 0 | AC_like_tumor | IDH_Mutant | 0.525 | 2383 | 0.751 |
| HAVCR2 | 1 | 1 | cycling_tumor | IDH_Mutant | 0.523 | 2422 | 0.747 |
| GRIA2 | 1 | 0 | OPC_GABA_AC_like_tumor | IDH_Mutant | 0.493 | 3004 | 0.687 |
| PTN | 1 | 0 | T cell | IDH_WT | 0.483 | 3231 | 0.663 |
| NCAM1 | 2 | 0 | cycling_tumor | IDH_WT | 0.482 | 3256 | 0.66 |
| GRIA4 | 2 | 0 | OPC_GABA_like_tumor | IDH_WT | 0.426 | 4523 | 0.528 |
| PDGFRA | 1 | 0 | AC_like_tumor | IDH_Mutant | 0.419 | 4685 | 0.511 |
| NLGN3 | 1 | 0 | OPC_GABA_like_tumor | IDH_Mutant | 0.385 | 5461 | 0.43 |
| ENTPD1 | 1 | 0 | TAM1/TAM2/microglia | IDH_Mutant | 0.369 | 5833 | 0.391 |
| NRXN1 | 2 | 0 | OPC_GABA_AC_like_tumor | IDH_Mutant | 0.339 | 6411 | 0.331 |
| CNTN1 | 2 | 0 | OPC_GABA_like_tumor | IDH_Mutant | 0.327 | 6628 | 0.308 |
| NLGN1 | 2 | 0 | AC_like_tumor | IDH_Mutant | 0.302 | 7039 | 0.265 |
| NOTCH1 | 2 | 0 | OPC_GABA_like_tumor | IDH_WT | 0.14 | 8547 | 0.108 |

**Top 25 novel genes** (in neither gene set nor the published 33, housekeeping genes removed; best context per gene)

| gene | celltype_id | IDH_status | p_immune_synapse | rule_pass | deg_log2fc | deg_neglog10_padj | cc_edges | ctx_min_n | why_immune_synapse |
|---|---|---|---|---|---|---|---|---|---|
| HLA-DPB1 | cycling_tumor | IDH_Mutant | 0.96 | 1 | -4.35 | 14.8 | 2 | 136 | deg_pooled_log2fc=-4.74 (+0.93); deg_pct_diff=-0.445 (+0.78); cc_n_celltypes_total=5 (+0.52) |
| HLA-DPA1 | cycling_tumor | IDH_WT | 0.922 | 1 | -6.15 | 105 | 2 | 663 | deg_pooled_log2fc=-6.1 (+0.93); deg_pct_diff=-0.457 (+0.77); cc_n_celltypes_total=5 (+0.52) |
| HLA-DMB | cycling_tumor | IDH_Mutant | 0.904 | 0 | -5.17 | 2.71 | 0 | 136 | deg_pooled_log2fc=-6.13 (+0.93); deg_pct_diff=-0.204 (+0.64); deg_log2fc=-5.17 (+0.44) |
| HLA-E | cycling_tumor | IDH_Mutant | 0.904 | 1 | -5.3 | 18.7 | 1 | 136 | deg_pooled_log2fc=-3.14 (+0.84); deg_pct_diff=-0.512 (+0.71); deg_log2fc=-5.3 (+0.44) |
| HLA-DMA | cycling_tumor | IDH_Mutant | 0.902 | 1 | -5.39 | 11.3 | 2 | 136 | deg_pooled_log2fc=-5.24 (+0.93); deg_pct_diff=-0.39 (+0.72); deg_log2fc=-5.39 (+0.45) |
| CLEC2B | cycling_tumor | IDH_Mutant | 0.874 | 0 | -7.54 | 3.03 | 0 | 136 | deg_pooled_log2fc=-7.75 (+0.91); deg_pct_diff=-0.196 (+0.63); deg_log2fc=-7.54 (+0.44) |
| SLC4A4 | OPC_GABA_AC_like_tumor | IDH_WT | 0.872 | 0 | -8.24 | 4.31 | 0 | 171 | deg_pooled_log2fc=-4.25 (+0.90); deg_pct_diff=-0.177 (+0.61); deg_log2fc=-8.24 (+0.44) |
| ITGB2 | cycling_tumor | IDH_Mutant | 0.871 | 0 | -5.38 | 11.7 | 0 | 136 | deg_pooled_log2fc=-5.02 (+0.93); deg_pct_diff=-0.4 (+0.73); deg_log2fc=-5.38 (+0.45) |
| CSGALNACT1 | TAM1/TAM2/microglia | IDH_Mutant | 0.866 | 0 | 1.59 | 82.2 | 0 | 2106 | deg_pooled_log2fc=2.48 (+1.33); deg_pct_e1=0.151 (+0.52); deg_neglog10_padj=82.2 (+0.36) |
| LRCH1 | TAM1/TAM2/microglia | IDH_Mutant | 0.866 | 0 | 2.03 | 87 | 0 | 2106 | deg_pooled_log2fc=2.5 (+1.32); deg_pct_e1=0.104 (+0.53); comp_tumor=0 (+0.36) |
| AK4 | OPC_GABA_AC_like_tumor | IDH_WT | 0.863 | 0 | -3.23 | 1.74 | 0 | 171 | deg_pooled_log2fc=-3.83 (+0.93); deg_pct_diff=-0.143 (+0.61); deg_log2fc=-3.23 (+0.43) |
| HLA-DQB1 | cycling_tumor | IDH_Mutant | 0.861 | 0 | -6.39 | 8.16 | 0 | 136 | deg_pooled_log2fc=-6.68 (+0.94); deg_pct_diff=-0.327 (+0.73); deg_log2fc=-6.39 (+0.45) |
| IGFBP2 | T cell | IDH_WT | 0.857 | 0 | 1.96 | 31.7 | 0 | 1118 | deg_pooled_log2fc=2.47 (+1.29); cc_n_cells_e1=2.66e+03 (+0.72); deg_pct_e1=0.106 (+0.56) |
| CXCR6 | T cell | IDH_Mutant | 0.855 | 0 | -1.87 | 1.65 | 0 | 334 | cc_n_cells_e1=2.66e+03 (+0.67); deg_pct_diff=-0.1 (+0.55); deg_pct_e1=0.142 (+0.45) |
| DIP2B | TAM1/TAM2/microglia | IDH_Mutant | 0.851 | 0 | 1.61 | 85.1 | 0 | 2106 | deg_pooled_log2fc=2.45 (+1.33); deg_pct_e1=0.147 (+0.51); comp_tumor=0 (+0.34) |
| HLA-DRB5 | cycling_tumor | IDH_Mutant | 0.851 | 0 | -5.47 | 6.95 | 0 | 136 | deg_pooled_log2fc=-5.03 (+0.94); deg_pct_diff=-0.304 (+0.73); deg_log2fc=-5.47 (+0.45) |
| CCL2 | cycling_tumor | IDH_WT | 0.85 | 0 | -5.69 | 17.8 | 0 | 663 | deg_pooled_log2fc=-6 (+0.91); deg_pct_diff=-0.139 (+0.64); deg_pct_e1=0.156 (+0.46) |
| C3 | cycling_tumor | IDH_Mutant | 0.848 | 0 | -5.79 | 16.7 | 0 | 136 | deg_pooled_log2fc=-5.03 (+0.93); deg_pct_diff=-0.482 (+0.73); deg_log2fc=-5.79 (+0.45) |
| OS9 | OPC_GABA_AC_like_tumor | IDH_WT | 0.848 | 0 | -2.15 | 3.39 | 0 | 171 | deg_pooled_log2fc=-3.24 (+0.82); deg_log2fc=-2.15 (+0.39); deg_pct_e1=0.358 (+0.33) |
| ICOS | T cell | IDH_Mutant | 0.847 | 0 | -3.87 | 10.9 | 0 | 334 | deg_pooled_log2fc=-2.35 (+0.72); cc_n_cells_e1=2.66e+03 (+0.65); deg_pct_diff=-0.193 (+0.57) |
| AQP4 | OPC_GABA_AC_like_tumor | IDH_WT | 0.846 | 0 | -5.23 | 4.88 | 0 | 171 | deg_pooled_log2fc=-2.96 (+0.81); deg_pct_diff=-0.191 (+0.60); deg_log2fc=-5.23 (+0.43) |
| HLA-DQA1 | cycling_tumor | IDH_WT | 0.844 | 0 | -4.75 | 15.8 | 0 | 663 | deg_pooled_log2fc=-4.59 (+0.91); deg_pct_diff=-0.126 (+0.65); deg_pct_e1=0.137 (+0.47) |
| FCGRT | T cell | IDH_WT | 0.843 | 0 | 0.981 | 3.31 | 0 | 1118 | cc_n_cells_e1=2.66e+03 (+0.74); deg_pct_diff=0.067 (+0.53); deg_pct_e1=0.082 (+0.48) |
| CCL4L2 | cycling_tumor | IDH_WT | 0.843 | 0 | -5.62 | 7.92 | 0 | 663 | deg_pooled_log2fc=-4.63 (+0.92); deg_pct_diff=-0.091 (+0.65); deg_pct_e1=0.126 (+0.47) |
| APOC1 | OPC_GABA_AC_like_tumor | IDH_WT | 0.843 | 0 | -4.4 | 10.7 | 0 | 171 | deg_pooled_log2fc=-2.96 (+0.81); deg_pct_diff=-0.289 (+0.66); deg_log2fc=-4.4 (+0.43) |

**Novel genes that also pass the 3-table rule** (CellChat edge + DEG + n_cells)

| gene | celltype_id | IDH_status | p_immune_synapse | rule_pass | deg_log2fc | deg_neglog10_padj | cc_edges | ctx_min_n | why_immune_synapse |
|---|---|---|---|---|---|---|---|---|---|
| HLA-DPB1 | cycling_tumor | IDH_Mutant | 0.96 | 1 | -4.35 | 14.8 | 2 | 136 | deg_pooled_log2fc=-4.74 (+0.93); deg_pct_diff=-0.445 (+0.78); cc_n_celltypes_total=5 (+0.52) |
| HLA-DPA1 | cycling_tumor | IDH_WT | 0.922 | 1 | -6.15 | 105 | 2 | 663 | deg_pooled_log2fc=-6.1 (+0.93); deg_pct_diff=-0.457 (+0.77); cc_n_celltypes_total=5 (+0.52) |
| HLA-E | cycling_tumor | IDH_Mutant | 0.904 | 1 | -5.3 | 18.7 | 1 | 136 | deg_pooled_log2fc=-3.14 (+0.84); deg_pct_diff=-0.512 (+0.71); deg_log2fc=-5.3 (+0.44) |
| HLA-DMA | cycling_tumor | IDH_Mutant | 0.902 | 1 | -5.39 | 11.3 | 2 | 136 | deg_pooled_log2fc=-5.24 (+0.93); deg_pct_diff=-0.39 (+0.72); deg_log2fc=-5.39 (+0.45) |
| PSAP | cycling_tumor | IDH_Mutant | 0.836 | 1 | -2.44 | 2.31 | 2 | 136 | cc_n_celltypes_total=5 (+0.52); deg_log2fc=-2.44 (+0.42); cc_ephys_bias=0 (+0.34) |
| C3AR1 | TAM1/TAM2/microglia | IDH_WT | 0.812 | 1 | 0.819 | 94 | 4 | 2745 | cc_ephys_bias=0 (+0.44); comp_tumor=0 (+0.33); deg_neglog10_padj=94 (+0.27) |
| CD69 | T cell | IDH_WT | 0.775 | 1 | -0.902 | 16.4 | 4 | 1118 | cc_n_cells_e1=2.66e+03 (+0.51); cc_ephys_bias=0 (+0.33); deg_log2fc=-0.902 (+0.30) |
| TNFRSF1B | TAM1/TAM2/microglia | IDH_Mutant | 0.75 | 1 | 0.456 | 14.1 | 2 | 2106 | cc_edges_total=2 (+0.47); deg_log2fc=0.456 (+0.40); cc_ephys_bias=0 (+0.40) |
| P4HB | TAM1/TAM2/microglia | IDH_WT | 0.705 | 1 | -0.563 | 2.32 | 1 | 2745 | deg_pct_diff=-0.032 (+0.49); cc_ephys_bias=-1 (+0.36); deg_log2fc=-0.563 (+0.26) |
| TNFRSF1A | TAM1/TAM2/microglia | IDH_WT | 0.675 | 1 | 0.256 | 9.38 | 1 | 2745 | comp_tumor=0 (+0.34); deg_pct_e1=0.192 (+0.33); deg_log2fc=0.256 (+0.22) |
| BSG | OPC_GABA_AC_like_tumor | IDH_Mutant | 0.655 | 1 | 0.569 | 1.36 | 11 | 352 | deg_log2fc=0.569 (+0.56); deg_pct_e1=0.111 (+0.38); cc_n_cells_e1=6.2e+03 (+0.30) |
| ITGAX | TAM1/TAM2/microglia | IDH_WT | 0.655 | 1 | 0.416 | 16 | 4 | 2745 | deg_log2fc=0.416 (+0.39); comp_tumor=0 (+0.31); deg_pct_e1=0.228 (+0.26) |
| SLIT1 | OPC_GABA_like_tumor | IDH_Mutant | 0.648 | 1 | 1.49 | 75.2 | 2 | 1658 | deg_pooled_log2fc=2.79 (+1.02); cc_edges_total=2 (+0.37); deg_pct_e1=0.071 (+0.32) |
| LAIR1 | TAM1/TAM2/microglia | IDH_Mutant | 0.648 | 1 | -0.275 | 3.68 | 4 | 2106 | deg_log2fc=-0.275 (+0.42); deg_pct_diff=-0.059 (+0.36); cc_ephys_bias=0 (+0.35) |
| ITGB8 | OPC_GABA_like_tumor | IDH_Mutant | 0.553 | 1 | -0.521 | 19.2 | 2 | 1658 | cc_ephys_bias=0 (+0.36); deg_log2fc=-0.521 (+0.30); deg_pct_e1=0.47 (+0.28) |

**Feature importance (gain, top 10)**

| feature | gain | splits | mean_abs_shap |
|---|---|---|---|
| deg_pooled_log2fc | 9.2e+04 | 80.2 | 0.247 |
| deg_pct_e1 | 9.07e+04 | 57.4 | 0.45 |
| deg_log2fc | 7.72e+04 | 32 | 0.302 |
| deg_pct_diff | 6.33e+04 | 42.4 | 0.25 |
| deg_pct_e2 | 3.19e+04 | 35.2 | 0.243 |
| deg_abs_log2fc | 2.54e+04 | 34.8 | 0.0909 |
| deg_neglog10_padj | 1.8e+04 | 42.8 | 0.109 |
| cc_n_cells_e1 | 1.32e+04 | 19.6 | 0.0584 |
| comp_tumor | 9.85e+03 | 12.6 | 0.114 |
| ctx_n_e2 | 8.25e+03 | 10 | 0.0645 |

## Caveats

- GO:0001772 includes all taxa (matching the team's 5,614-annotation download); most are IEA (electronic) annotations mapped by upper-casing symbols. `--immune-human-only` restricts to genes with a human annotation.
- SynGO annotations are mostly from rodent brain synapses; membership says a gene is synaptic in neurons, not in glioma.
- Gene-set membership is a weak label: unlabeled genes are unknown, not negative.
- CellChat features are zero for most genes in the DEG universe (CellChat covers ~114 genes); DEG and n_cells carry most signal.
- SynGO genes are enriched for broadly expressed genes, so the tumor-synaptic model partly learns "highly expressed". Mitochondrial / ribosomal / histone genes are flagged `housekeeping` and left out of the novel tables.
- Immune genes (HLA class II, ITGB2, C3, CCL2) show up as strong Ephys_1-high DEGs inside *tumor* clusters, which suggests myeloid contamination / doublets in Ephys_1 tumor clusters. Immune-synapse nominees in tumor cell types should be checked for this before follow-up.
- High score = "expression pattern resembles known synaptic / immune-synapse genes", not validation.
