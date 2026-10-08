# LightGBM with knowledge-tool labels: tumor synaptic vs immune synapse

- Universe: `deg`: 27346 gene x cell type x IDH rows, 9580 genes, 42 features.
- Scores are out-of-fold (StratifiedGroupKFold(n_splits=5) grouped by gene, seeds [0, 1, 2, 3, 4]): no gene is scored by a model that trained on it.
- The 33 published glioma genes are **not** labels here. `in_published_33` flags them; the check below shows where each model ranks them.
- LLM run features: none.

## Label sources

- **immune_synapse**: QuickGO GO:0001772: 5614 annotations -> 244 genes (54 with a human annotation, 53 with experimental evidence). Fetched 2026-10-08.

## immune_synapse

GO:0001772 immunological synapse genes, scored on both sides of the synapse. Compartments: T_cell, TAM_microglia, tumor.
40 of 54 set genes are in the universe; 122 positive rows of 27346 (prevalence 0.004).

**Model vs single-column baselines (row level)**

| scorer | auroc | auprc | precision_at_10 | precision_at_25 | precision_at_50 |
|---|---|---|---|---|---|
| lightgbm_oof | 0.7 | 0.0138 | 0 | 0.04 | 0.04 |
| rule_pass | 0.58 | 0.0321 | 0.2 | 0.16 | 0.1 |
| deg_abs_log2fc | 0.564 | 0.00996 | 0.1 | 0.04 | 0.02 |
| cc_edges | 0.556 | 0.00917 | 0 | 0 | 0 |
| cc_max_prob | 0.556 | 0.00914 | 0 | 0 | 0 |

**Gene level (best context per gene)**

| scorer | auroc | auprc | precision_at_10 | precision_at_25 | precision_at_50 |
|---|---|---|---|---|---|
| lightgbm_oof | 0.72 | 0.0175 | 0.1 | 0.04 | 0.02 |
| deg_abs_log2fc | 0.569 | 0.0111 | 0.1 | 0.04 | 0.02 |
| cc_edges | 0.532 | 0.00591 | 0 | 0 | 0 |

**Published-33 check**

- Published genes already in this label set: HAVCR2, HLA-DRA.
- Published genes scored: 32 of 33; not in universe: AXL.
- Median gene-rank percentile of published genes: 0.94 (1.00 = top). AUROC published vs all other genes: 0.822 (single-column baselines, best context per gene: deg_abs_log2fc 0.901, deg_pct_e2 0.808, cc_edges 0.982, deg_n_celltypes_sig 0.737).
- Held-out test, genes **not** in the label set only (30 published): AUROC 0.823.

| gene | published_tier | in_label_set | celltype_id | IDH_status | best_score | gene_rank | gene_percentile |
|---|---|---|---|---|---|---|---|
| SPP1 | 1 | 0 | OPC_GABA_AC_like_tumor | IDH_WT | 0.956 | 3 | 1 |
| CD74 | 1 | 0 | cycling_tumor | IDH_Mutant | 0.946 | 4 | 1 |
| TYROBP | 2 | 0 | cycling_tumor | IDH_WT | 0.91 | 9 | 0.999 |
| EGFR | 1 | 0 | OPC_GABA_AC_like_tumor | IDH_WT | 0.907 | 10 | 0.999 |
| TREM2 | 2 | 0 | cycling_tumor | IDH_Mutant | 0.894 | 15 | 0.999 |
| MIF | 1 | 0 | TAM1/TAM2/microglia | IDH_WT | 0.89 | 17 | 0.998 |
| CLEC2D | 1 | 0 | T cell | IDH_WT | 0.888 | 18 | 0.998 |
| PTN | 1 | 0 | T cell | IDH_WT | 0.879 | 25 | 0.997 |
| CXCR4 | 1 | 0 | cycling_tumor | IDH_WT | 0.868 | 41 | 0.996 |
| CD44 | 1 | 0 | cycling_tumor | IDH_WT | 0.843 | 93 | 0.99 |
| CD99 | 2 | 0 | cycling_tumor | IDH_Mutant | 0.79 | 249 | 0.974 |
| APOE | 2 | 0 | T cell | IDH_WT | 0.774 | 300 | 0.969 |
| TGFB1 | 2 | 0 | cycling_tumor | IDH_Mutant | 0.752 | 376 | 0.961 |
| PTPRZ1 | 1 | 0 | T cell | IDH_WT | 0.731 | 440 | 0.954 |
| KLRB1 | 1 | 0 | T cell | IDH_Mutant | 0.728 | 454 | 0.953 |
| LGALS9 | 2 | 0 | TAM1/TAM2/microglia | IDH_WT | 0.706 | 534 | 0.944 |
| HLA-DRA | 2 | 1 | OPC_GABA_AC_like_tumor | IDH_WT | 0.693 | 604 | 0.937 |
| CD4 | 2 | 0 | cycling_tumor | IDH_Mutant | 0.674 | 708 | 0.926 |
| LILRB4 | 2 | 0 | cycling_tumor | IDH_Mutant | 0.622 | 1118 | 0.883 |
| NCAM1 | 2 | 0 | OPC_GABA_AC_like_tumor | IDH_Mutant | 0.613 | 1203 | 0.875 |
| GRIA3 | 2 | 0 | AC_like_tumor | IDH_Mutant | 0.56 | 1769 | 0.815 |
| DLL3 | 1 | 0 | AC_like_tumor | IDH_Mutant | 0.543 | 2001 | 0.791 |
| ENTPD1 | 1 | 0 | TAM1/TAM2/microglia | IDH_WT | 0.532 | 2154 | 0.775 |
| PDGFRA | 1 | 0 | OPC_GABA_AC_like_tumor | IDH_Mutant | 0.499 | 2665 | 0.722 |
| HAVCR2 | 1 | 1 | cycling_tumor | IDH_Mutant | 0.482 | 2925 | 0.695 |
| GRIA2 | 1 | 0 | OPC_GABA_AC_like_tumor | IDH_Mutant | 0.477 | 3001 | 0.687 |
| NRXN1 | 2 | 0 | OPC_GABA_like_tumor | IDH_WT | 0.405 | 4455 | 0.535 |
| GRIA4 | 2 | 0 | OPC_GABA_like_tumor | IDH_WT | 0.381 | 4930 | 0.485 |
| CNTN1 | 2 | 0 | OPC_GABA_like_tumor | IDH_Mutant | 0.353 | 5398 | 0.437 |
| NLGN3 | 1 | 0 | OPC_GABA_like_tumor | IDH_WT | 0.298 | 6213 | 0.352 |
| NLGN1 | 2 | 0 | AC_like_tumor | IDH_Mutant | 0.277 | 6490 | 0.323 |
| NOTCH1 | 2 | 0 | OPC_GABA_like_tumor | IDH_WT | 0.259 | 6686 | 0.302 |

**Top 25 novel genes** (in neither gene set nor the published 33, housekeeping genes removed; best context per gene)

| gene | celltype_id | IDH_status | p_immune_synapse | rule_pass | deg_log2fc | deg_neglog10_padj | cc_edges | ctx_min_n | why_immune_synapse |
|---|---|---|---|---|---|---|---|---|---|
| HLA-DPB1 | cycling_tumor | IDH_WT | 0.969 | 1 | -4.83 | 84 | 2 | 663 | deg_pooled_log2fc=-4.74 (+0.93); deg_pct_diff=-0.375 (+0.91); deg_log2fc=-4.83 (+0.66) |
| HLA-DPA1 | cycling_tumor | IDH_WT | 0.958 | 1 | -6.15 | 105 | 2 | 663 | deg_pooled_log2fc=-6.1 (+0.94); deg_pct_diff=-0.457 (+0.87); deg_log2fc=-6.15 (+0.65) |
| IGFBP2 | T cell | IDH_WT | 0.924 | 0 | 1.96 | 31.7 | 0 | 1118 | deg_pooled_log2fc=2.47 (+1.28); cc_n_cells_e1=2.66e+03 (+0.76); deg_pct_e1=0.106 (+0.70) |
| CDC42EP3 | OPC_GABA_AC_like_tumor | IDH_WT | 0.918 | 0 | -4.52 | 1.71 | 0 | 171 | deg_pooled_log2fc=-4.81 (+0.89); deg_pct_e1=0.137 (+0.74); deg_pct_diff=-0.131 (+0.59) |
| CCL2 | TAM1/TAM2/microglia | IDH_Mutant | 0.911 | 0 | 1.91 | 96.8 | 0 | 2106 | deg_pooled_log2fc=2.47 (+1.36); deg_pct_e1=0.172 (+0.71); deg_pct_e2=0.405 (+0.41) |
| PTGER4 | T cell | IDH_WT | 0.906 | 0 | -1.02 | 2.71 | 0 | 1118 | deg_pct_e1=0.142 (+0.70); cc_n_cells_e1=2.66e+03 (+0.67); deg_pct_diff=-0.066 (+0.52) |
| LRCH1 | TAM1/TAM2/microglia | IDH_Mutant | 0.901 | 0 | 2.03 | 87 | 0 | 2106 | deg_pooled_log2fc=2.5 (+1.34); deg_pct_e1=0.104 (+0.71); deg_neglog10_padj=87 (+0.43) |
| HLA-DQA1 | cycling_tumor | IDH_WT | 0.898 | 0 | -4.75 | 15.8 | 0 | 663 | deg_pooled_log2fc=-4.59 (+0.91); deg_pct_e1=0.137 (+0.78); deg_pct_diff=-0.126 (+0.69) |
| HLA-DMA | cycling_tumor | IDH_Mutant | 0.895 | 1 | -5.39 | 11.3 | 2 | 136 | deg_pooled_log2fc=-5.24 (+0.93); deg_pct_diff=-0.39 (+0.81); deg_pct_e1=0.441 (+0.67) |
| GAB2 | TAM1/TAM2/microglia | IDH_Mutant | 0.893 | 0 | 2.13 | 104 | 0 | 2106 | deg_pooled_log2fc=2.62 (+1.30); deg_pct_e1=0.116 (+0.71); deg_pct_e2=0.346 (+0.41) |
| TMIGD3 | cycling_tumor | IDH_Mutant | 0.887 | 0 | -5.25 | 5.59 | 0 | 136 | deg_pooled_log2fc=-5.31 (+0.92); deg_pct_diff=-0.267 (+0.80); deg_pct_e1=0.274 (+0.62) |
| CD69 | T cell | IDH_WT | 0.887 | 1 | -0.902 | 16.4 | 4 | 1118 | cc_n_cells_e1=2.66e+03 (+0.62); deg_pct_e1=0.542 (+0.58); deg_pct_diff=-0.153 (+0.30) |
| SFMBT2 | TAM1/TAM2/microglia | IDH_WT | 0.886 | 0 | 1.03 | 82.8 | 0 | 2745 | deg_pooled_log2fc=2.64 (+1.28); deg_pct_e1=0.158 (+0.80); deg_neglog10_padj=82.8 (+0.45) |
| SERPINB1 | cycling_tumor | IDH_Mutant | 0.881 | 0 | -5.37 | 1.72 | 0 | 136 | deg_pooled_log2fc=-5.27 (+0.91); deg_pct_e1=0.2 (+0.75); deg_pct_diff=-0.178 (+0.63) |
| SIT1 | T cell | IDH_WT | 0.881 | 0 | -0.969 | 1.63 | 0 | 1118 | deg_pct_e1=0.128 (+0.72); deg_pct_diff=-0.057 (+0.64); cc_n_cells_e1=2.66e+03 (+0.64) |
| HLA-DMB | cycling_tumor | IDH_Mutant | 0.88 | 0 | -5.17 | 2.71 | 0 | 136 | deg_pooled_log2fc=-6.13 (+0.94); deg_pct_diff=-0.204 (+0.64); deg_pct_e1=0.226 (+0.61) |
| MARCH3 | TAM1/TAM2/microglia | IDH_Mutant | 0.878 | 0 | 2.26 | 89.7 | 0 | 2106 | deg_pooled_log2fc=2.7 (+1.31); deg_pct_e1=0.107 (+0.71); deg_neglog10_padj=89.7 (+0.43) |
| CDC14A | T cell | IDH_WT | 0.877 | 0 | -1.1 | 3.22 | 0 | 1118 | deg_pct_e1=0.112 (+0.69); cc_n_cells_e1=2.66e+03 (+0.64); deg_pct_diff=-0.061 (+0.52) |
| MYBPC1 | OPC_GABA_AC_like_tumor | IDH_WT | 0.876 | 0 | -7.75 | 2.03 | 0 | 171 | deg_pooled_log2fc=-3.91 (+0.90); deg_pct_e1=0.133 (+0.74); deg_pct_diff=-0.133 (+0.59) |
| HLA-E | cycling_tumor | IDH_WT | 0.875 | 1 | -2.99 | 11.2 | 1 | 663 | deg_pooled_log2fc=-3.14 (+0.83); deg_pct_e1=0.276 (+0.62); deg_pct_diff=-0.112 (+0.44) |
| ITGB2 | cycling_tumor | IDH_WT | 0.874 | 0 | -5.68 | 13.8 | 0 | 663 | deg_pooled_log2fc=-5.02 (+0.91); deg_pct_e1=0.126 (+0.76); deg_pct_diff=-0.115 (+0.70) |
| TRIP6 | OPC_GABA_AC_like_tumor | IDH_WT | 0.873 | 0 | -3.6 | 2.76 | 0 | 171 | deg_pooled_log2fc=-3.11 (+0.82); deg_pct_e1=0.183 (+0.69); deg_log2fc=-3.6 (+0.58) |
| HLA-DRB5 | cycling_tumor | IDH_Mutant | 0.873 | 0 | -5.47 | 6.95 | 0 | 136 | deg_pooled_log2fc=-5.03 (+0.93); deg_pct_diff=-0.304 (+0.81); deg_pct_e1=0.341 (+0.64) |
| PDE3B | TAM1/TAM2/microglia | IDH_Mutant | 0.872 | 0 | 1.42 | 86.8 | 0 | 2106 | deg_pooled_log2fc=2.48 (+1.36); deg_pct_e1=0.21 (+0.65); deg_neglog10_padj=86.8 (+0.44) |
| ARHGAP25 | TAM1/TAM2/microglia | IDH_Mutant | 0.872 | 0 | 1.77 | 72 | 0 | 2106 | deg_pooled_log2fc=2.48 (+1.35); deg_pct_e1=0.128 (+0.75); deg_pct_e2=0.317 (+0.41) |

**Novel genes that also pass the 3-table rule** (CellChat edge + DEG + n_cells)

| gene | celltype_id | IDH_status | p_immune_synapse | rule_pass | deg_log2fc | deg_neglog10_padj | cc_edges | ctx_min_n | why_immune_synapse |
|---|---|---|---|---|---|---|---|---|---|
| HLA-DPB1 | cycling_tumor | IDH_WT | 0.969 | 1 | -4.83 | 84 | 2 | 663 | deg_pooled_log2fc=-4.74 (+0.93); deg_pct_diff=-0.375 (+0.91); deg_log2fc=-4.83 (+0.66) |
| HLA-DPA1 | cycling_tumor | IDH_WT | 0.958 | 1 | -6.15 | 105 | 2 | 663 | deg_pooled_log2fc=-6.1 (+0.94); deg_pct_diff=-0.457 (+0.87); deg_log2fc=-6.15 (+0.65) |
| HLA-DMA | cycling_tumor | IDH_Mutant | 0.895 | 1 | -5.39 | 11.3 | 2 | 136 | deg_pooled_log2fc=-5.24 (+0.93); deg_pct_diff=-0.39 (+0.81); deg_pct_e1=0.441 (+0.67) |
| CD69 | T cell | IDH_WT | 0.887 | 1 | -0.902 | 16.4 | 4 | 1118 | cc_n_cells_e1=2.66e+03 (+0.62); deg_pct_e1=0.542 (+0.58); deg_pct_diff=-0.153 (+0.30) |
| HLA-E | cycling_tumor | IDH_WT | 0.875 | 1 | -2.99 | 11.2 | 1 | 663 | deg_pooled_log2fc=-3.14 (+0.83); deg_pct_e1=0.276 (+0.62); deg_pct_diff=-0.112 (+0.44) |
| ANXA1 | cycling_tumor | IDH_WT | 0.859 | 1 | -3.93 | 50.6 | 2 | 663 | deg_pooled_log2fc=-4.11 (+0.94); deg_pct_diff=-0.297 (+0.81); deg_pct_e1=0.353 (+0.67) |
| PSAP | cycling_tumor | IDH_Mutant | 0.821 | 1 | -2.44 | 2.31 | 2 | 136 | deg_pct_e1=0.519 (+0.53); cc_n_celltypes_total=5 (+0.43); deg_log2fc=-2.44 (+0.38) |
| PPIA | TAM1/TAM2/microglia | IDH_WT | 0.777 | 1 | -0.367 | 32.7 | 7 | 2745 | deg_pct_diff=-0.007 (+0.58); cc_n_celltypes_total=7 (+0.45); deg_pct_e1=0.842 (+0.43) |
| C3AR1 | TAM1/TAM2/microglia | IDH_WT | 0.777 | 1 | 0.819 | 94 | 4 | 2745 | deg_pct_e1=0.363 (+0.52); deg_neglog10_padj=94 (+0.41); comp_tumor=0 (+0.39) |
| LAIR1 | TAM1/TAM2/microglia | IDH_Mutant | 0.754 | 1 | -0.275 | 3.68 | 4 | 2106 | deg_pct_e1=0.385 (+0.57); deg_pct_diff=-0.059 (+0.41); deg_log2fc=-0.275 (+0.41) |
| GRN | TAM1/TAM2/microglia | IDH_Mutant | 0.748 | 1 | -0.264 | 7.98 | 2 | 2106 | deg_pct_e1=0.643 (+0.51); deg_pct_diff=-0.057 (+0.45); deg_log2fc=-0.264 (+0.41) |
| P4HB | TAM1/TAM2/microglia | IDH_WT | 0.705 | 1 | -0.563 | 2.32 | 1 | 2745 | deg_pct_e1=0.317 (+0.59); deg_pct_diff=-0.032 (+0.53); comp_tumor=0 (+0.28) |
| BSG | OPC_GABA_AC_like_tumor | IDH_Mutant | 0.669 | 1 | 0.569 | 1.36 | 11 | 352 | deg_pct_e1=0.111 (+0.58); deg_pooled_log2fc=nan (+0.38); deg_log2fc=0.569 (+0.35) |
| APP | OPC_GABA_like_tumor | IDH_Mutant | 0.657 | 1 | -0.317 | 7.99 | 26 | 1658 | deg_pct_diff=-0.035 (+0.62); deg_log2fc=-0.317 (+0.39); deg_pct_e1=0.7 (+0.36) |
| TNF | TAM1/TAM2/microglia | IDH_Mutant | 0.639 | 1 | 1.31 | 80.5 | 3 | 2106 | deg_pct_e1=0.299 (+0.51); deg_neglog10_padj=80.5 (+0.45); comp_tumor=0 (+0.43) |

**Feature importance (gain, top 10)**

| feature | gain | splits | mean_abs_shap |
|---|---|---|---|
| deg_pct_e1 | 1.31e+05 | 59.4 | 0.787 |
| deg_log2fc | 1.19e+05 | 34.4 | 0.246 |
| deg_pooled_log2fc | 9.64e+04 | 71.4 | 0.362 |
| deg_pct_diff | 7.94e+04 | 46.6 | 0.274 |
| deg_pct_e2 | 5.51e+04 | 37 | 0.382 |
| deg_neglog10_padj | 2.45e+04 | 48 | 0.158 |
| deg_abs_log2fc | 2.39e+04 | 32.2 | 0.0911 |
| cc_n_cells_e1 | 1.57e+04 | 18.4 | 0.0641 |
| comp_tumor | 1.31e+04 | 13.4 | 0.131 |
| ctx_n_e2 | 8.16e+03 | 9 | 0.0698 |

## Caveats

- GO:0001772 includes all taxa (matching the team's 5,614-annotation download); most are IEA (electronic) annotations mapped by upper-casing symbols. `--immune-human-only` restricts to genes with a human annotation.
- SynGO annotations are mostly from rodent brain synapses; membership says a gene is synaptic in neurons, not in glioma.
- Gene-set membership is a weak label: unlabeled genes are unknown, not negative.
- CellChat features are zero for most genes in the DEG universe (CellChat covers ~114 genes); DEG and n_cells carry most signal.
- SynGO genes are enriched for broadly expressed genes, so the tumor-synaptic model partly learns "highly expressed". Mitochondrial / ribosomal / histone genes are flagged `housekeeping` and left out of the novel tables.
- Immune genes (HLA class II, ITGB2, C3, CCL2) show up as strong Ephys_1-high DEGs inside *tumor* clusters, which suggests myeloid contamination / doublets in Ephys_1 tumor clusters. Immune-synapse nominees in tumor cell types should be checked for this before follow-up.
- High score = "expression pattern resembles known synaptic / immune-synapse genes", not validation.
