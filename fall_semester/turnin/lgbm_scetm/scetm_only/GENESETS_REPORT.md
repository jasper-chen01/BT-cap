# LightGBM with knowledge-tool labels: tumor synaptic vs immune synapse

- Universe: `deg`: 27346 gene x cell type x IDH rows, 9580 genes, 33 features.
- Scores are out-of-fold (StratifiedGroupKFold(n_splits=5) grouped by gene, seeds [0, 1, 2, 3, 4]): no gene is scored by a model that trained on it.
- The 33 published glioma genes are **not** labels here. `in_published_33` flags them; the check below shows where each model ranks them.
- LLM run features: none.
- Feature arm: `scetm_only`.
- scETM embeddings: `scetm_gene_embeddings_d32.csv` (32-d, 20126 genes); 9578 of 9580 universe genes embedded (others get NaN). Level 2 = fraction of 20 nearest neighbours in the label set, rebuilt per fold from training genes.

## Label sources

- **tumor_synaptic**: SynGO  release 20250226: 5076 annotations -> 1789 genes. Fetched 2026-10-08.
- **immune_synapse**: QuickGO GO:0001772: 5614 annotations -> 244 genes (54 with a human annotation, 53 with experimental evidence). Fetched 2026-10-08.

## tumor_synaptic

SynGO synaptic genes, scored in tumor cell types (tumor synaptic program). Compartments: tumor.
1112 of 1789 set genes are in the universe; 3249 positive rows of 21843 (prevalence 0.149).

**Model vs single-column baselines (row level)**

| scorer | auroc | auprc | precision_at_10 | precision_at_25 | precision_at_50 |
|---|---|---|---|---|---|
| lightgbm_oof | 0.738 | 0.433 | 0.8 | 0.92 | 0.96 |
| rule_pass | 0.516 | 0.174 | 0.2 | 0.2 | 0.38 |
| deg_abs_log2fc | 0.507 | 0.154 | 0 | 0.04 | 0.02 |
| cc_edges | 0.514 | 0.164 | 0.6 | 0.64 | 0.54 |
| cc_max_prob | 0.514 | 0.165 | 0.9 | 0.76 | 0.64 |

**Gene level (best context per gene)**

| scorer | auroc | auprc | precision_at_10 | precision_at_25 | precision_at_50 |
|---|---|---|---|---|---|
| lightgbm_oof | 0.721 | 0.363 | 0.9 | 0.88 | 0.88 |
| deg_abs_log2fc | 0.539 | 0.146 | 0 | 0.04 | 0.02 |
| cc_edges | 0.512 | 0.14 | 0.6 | 0.48 | 0.54 |

**Published-33 check**

- Published genes already in this label set: APOE, CD44, CNTN1, EGFR, GRIA2, GRIA3, GRIA4, NCAM1, NLGN1, NLGN3, NOTCH1, NRXN1, PTN, PTPRZ1.
- Published genes scored: 30 of 33; not in universe: AXL, ENTPD1, KLRB1.
- Median gene-rank percentile of published genes: 0.72 (1.00 = top). AUROC published vs all other genes: 0.679 (single-column baselines, best context per gene: deg_abs_log2fc 0.906, deg_pct_e2 0.574, cc_edges 0.831, deg_n_celltypes_sig 0.758).
- Held-out test, genes **not** in the label set only (16 published): AUROC 0.551.

| gene | published_tier | in_label_set | celltype_id | IDH_status | best_score | gene_rank | gene_percentile |
|---|---|---|---|---|---|---|---|
| NCAM1 | 2 | 1 | AC_like_tumor | IDH_WT | 0.897 | 89 | 0.99 |
| PTPRZ1 | 1 | 1 | OPC_GABA_like_tumor | IDH_WT | 0.888 | 98 | 0.989 |
| NLGN1 | 2 | 1 | OPC_GABA_like_tumor | IDH_WT | 0.87 | 113 | 0.987 |
| NRXN1 | 2 | 1 | OPC_GABA_like_tumor | IDH_WT | 0.86 | 118 | 0.987 |
| NLGN3 | 1 | 1 | cycling_tumor | IDH_Mutant | 0.833 | 136 | 0.985 |
| GRIA2 | 1 | 1 | AC_like_tumor | IDH_Mutant | 0.826 | 140 | 0.984 |
| EGFR | 1 | 1 | OPC_GABA_AC_like_tumor | IDH_WT | 0.823 | 143 | 0.984 |
| GRIA4 | 2 | 1 | OPC_GABA_like_tumor | IDH_WT | 0.821 | 147 | 0.983 |
| CNTN1 | 2 | 1 | OPC_GABA_like_tumor | IDH_Mutant | 0.781 | 191 | 0.978 |
| PDGFRA | 1 | 0 | AC_like_tumor | IDH_WT | 0.714 | 355 | 0.959 |
| DLL3 | 1 | 0 | OPC_GABA_like_tumor | IDH_Mutant | 0.647 | 664 | 0.924 |
| MIF | 1 | 0 | OPC_GABA_AC_like_tumor | IDH_WT | 0.609 | 910 | 0.896 |
| GRIA3 | 2 | 1 | cycling_tumor | IDH_WT | 0.519 | 1765 | 0.798 |
| NOTCH1 | 2 | 1 | OPC_GABA_like_tumor | IDH_WT | 0.503 | 2001 | 0.771 |
| HLA-DRA | 2 | 0 | cycling_tumor | IDH_WT | 0.481 | 2351 | 0.73 |
| CLEC2D | 1 | 0 | cycling_tumor | IDH_WT | 0.475 | 2456 | 0.718 |
| CD74 | 1 | 0 | cycling_tumor | IDH_Mutant | 0.459 | 2736 | 0.686 |
| TYROBP | 2 | 0 | AC_like_tumor | IDH_WT | 0.45 | 2904 | 0.667 |
| PTN | 1 | 1 | cycling_tumor | IDH_Mutant | 0.447 | 2948 | 0.662 |
| TREM2 | 2 | 0 | cycling_tumor | IDH_Mutant | 0.439 | 3097 | 0.645 |
| CD44 | 1 | 1 | cycling_tumor | IDH_WT | 0.413 | 3679 | 0.578 |
| HAVCR2 | 1 | 0 | cycling_tumor | IDH_Mutant | 0.388 | 4296 | 0.507 |
| APOE | 2 | 1 | OPC_GABA_like_tumor | IDH_Mutant | 0.329 | 5835 | 0.331 |
| SPP1 | 1 | 0 | OPC_GABA_like_tumor | IDH_WT | 0.326 | 5894 | 0.324 |
| TGFB1 | 2 | 0 | cycling_tumor | IDH_Mutant | 0.321 | 6045 | 0.306 |
| LGALS9 | 2 | 0 | cycling_tumor | IDH_Mutant | 0.32 | 6055 | 0.305 |
| LILRB4 | 2 | 0 | cycling_tumor | IDH_Mutant | 0.319 | 6081 | 0.302 |
| CD4 | 2 | 0 | cycling_tumor | IDH_Mutant | 0.275 | 7060 | 0.19 |
| CD99 | 2 | 0 | cycling_tumor | IDH_WT | 0.257 | 7393 | 0.152 |
| CXCR4 | 1 | 0 | cycling_tumor | IDH_WT | 0.185 | 8376 | 0.039 |

**Top 25 novel genes** (in neither gene set nor the published 33, housekeeping genes removed; best context per gene)

| gene | celltype_id | IDH_status | p_tumor_synaptic | rule_pass | deg_log2fc | deg_neglog10_padj | cc_edges | ctx_min_n | why_tumor_synaptic |
|---|---|---|---|---|---|---|---|---|---|
| PFDN5 | cycling_tumor | IDH_WT | 0.957 | 0 | -0.674 | 23 | 0 | 663 | knn_frac_synaptic=0.95 (+2.51); emb_07=1.17 (+0.25); emb_15=0.321 (+0.19) |
| NOP53 | cycling_tumor | IDH_WT | 0.915 | 0 | -0.278 | 33.1 | 0 | 663 | knn_frac_synaptic=0.9 (+2.51); emb_07=0.793 (+0.16); emb_15=0.145 (+0.10) |
| NSA2 | OPC_GABA_like_tumor | IDH_Mutant | 0.896 | 0 | 0.453 | 30.7 | 0 | 1658 | knn_frac_synaptic=0.95 (+2.29); emb_15=0.42 (+0.16); emb_09=-0.633 (+0.09) |
| C6orf48 | AC_like_tumor | IDH_Mutant | 0.895 | 0 | 0.408 | 4.78 | 0 | 283 | knn_frac_synaptic=0.8 (+2.06); emb_15=0.743 (+0.14); emb_09=-1.11 (+0.09) |
| NPM1 | OPC_GABA_like_tumor | IDH_Mutant | 0.895 | 0 | 0.584 | 95 | 0 | 1658 | knn_frac_synaptic=0.7 (+2.05); emb_04=-1.18 (+0.13); emb_09=-0.644 (+0.10) |
| GPR37L1 | OPC_GABA_like_tumor | IDH_Mutant | 0.895 | 1 | 0.829 | 11.3 | 9 | 1658 | knn_frac_synaptic=0.65 (+1.82); emb_07=0.837 (+0.27); emb_15=1.81 (+0.25) |
| SCG3 | cycling_tumor | IDH_WT | 0.89 | 0 | 0.834 | 73.1 | 0 | 663 | knn_frac_synaptic=0.6 (+1.82); emb_07=1.34 (+0.27); emb_15=1.96 (+0.27) |
| CHL1 | OPC_GABA_like_tumor | IDH_Mutant | 0.886 | 0 | -1.09 | 21.7 | 0 | 1658 | knn_frac_synaptic=0.6 (+1.81); emb_07=2.07 (+0.28); emb_15=2.55 (+0.28) |
| ADGRA3 | OPC_GABA_like_tumor | IDH_Mutant | 0.873 | 0 | -0.663 | 6.8 | 0 | 1658 | knn_frac_synaptic=0.55 (+1.97); emb_15=0.342 (+0.21); emb_07=0.617 (+0.17) |
| SLC25A6 | cycling_tumor | IDH_WT | 0.871 | 0 | 0.403 | 30.7 | 0 | 663 | knn_frac_synaptic=0.85 (+2.42); emb_15=0.778 (+0.12); emb_09=-0.892 (+0.09) |
| FUT9 | OPC_GABA_like_tumor | IDH_WT | 0.869 | 0 | 2.23 | 99.1 | 0 | 1935 | knn_frac_synaptic=0.45 (+1.08); emb_15=0.172 (+0.17); emb_05=-2.22 (+0.14) |
| DPP10 | OPC_GABA_like_tumor | IDH_Mutant | 0.869 | 0 | -1.17 | 8.01 | 0 | 1658 | knn_frac_synaptic=0.6 (+1.84); emb_07=1.06 (+0.22); emb_15=1.08 (+0.21) |
| SEC11A | OPC_GABA_like_tumor | IDH_WT | 0.866 | 0 | -0.381 | 9.56 | 0 | 1935 | knn_frac_synaptic=0.7 (+2.04); emb_09=-1.3 (+0.11); emb_04=-0.424 (+0.10) |
| LRP1B | OPC_GABA_like_tumor | IDH_WT | 0.86 | 0 | 1.04 | 103 | 0 | 1935 | knn_frac_synaptic=0.6 (+1.80); emb_15=1.27 (+0.26); emb_07=1.66 (+0.26) |
| PIANP | cycling_tumor | IDH_Mutant | 0.859 | 0 | 4.95 | 8.33 | 0 | 136 | knn_frac_synaptic=0.5 (+1.12); emb_07=1.31 (+0.27); emb_15=0.817 (+0.23) |
| LHFPL3 | AC_like_tumor | IDH_Mutant | 0.857 | 0 | 1.32 | 21 | 0 | 283 | knn_frac_synaptic=0.4 (+0.86); emb_15=1.86 (+0.30); emb_07=0.575 (+0.17) |
| ZFAS1 | OPC_GABA_AC_like_tumor | IDH_WT | 0.857 | 0 | -1.15 | 4.41 | 0 | 171 | knn_frac_synaptic=0.7 (+1.77); emb_07=1.06 (+0.26); emb_15=0.93 (+0.18) |
| GLDC | cycling_tumor | IDH_Mutant | 0.853 | 0 | 3.22 | 4.72 | 0 | 136 | knn_frac_synaptic=0.45 (+1.09); emb_15=0.476 (+0.19); emb_04=-0.978 (+0.12) |
| TOMM7 | AC_like_tumor | IDH_WT | 0.852 | 0 | -0.713 | 3.64 | 0 | 108 | knn_frac_synaptic=0.65 (+2.03); emb_07=0.521 (+0.17); emb_15=1.16 (+0.17) |
| PCDHB14 | cycling_tumor | IDH_WT | 0.847 | 0 | 1.29 | 32 | 0 | 663 | knn_frac_synaptic=0.45 (+1.09); emb_17=3.84 (+0.59); emb_15=1.35 (+0.30) |
| ZNF462 | cycling_tumor | IDH_WT | 0.837 | 0 | 1.99 | 245 | 0 | 663 | knn_frac_synaptic=0.45 (+1.14); emb_17=3.53 (+0.54); emb_07=0.939 (+0.28) |
| NAP1L1 | cycling_tumor | IDH_Mutant | 0.823 | 0 | 0.452 | 3.28 | 0 | 136 | knn_frac_synaptic=0.55 (+1.96); emb_07=1.5 (+0.19); emb_04=-1.16 (+0.13) |
| SNRPD2 | AC_like_tumor | IDH_Mutant | 0.822 | 0 | 0.282 | 3.01 | 0 | 283 | knn_frac_synaptic=0.65 (+2.08); emb_09=-0.244 (+0.10); emb_20=0.863 (+0.08) |
| PYGO1 | OPC_GABA_like_tumor | IDH_WT | 0.821 | 0 | 0.352 | 15.9 | 0 | 1935 | knn_frac_synaptic=0.55 (+1.76); emb_15=1.69 (+0.28); emb_07=0.572 (+0.16) |
| XKR4 | OPC_GABA_like_tumor | IDH_Mutant | 0.82 | 0 | 2.12 | 45.4 | 0 | 1658 | knn_frac_synaptic=0.4 (+0.94); emb_07=1.39 (+0.26); emb_15=0.422 (+0.21) |

**Novel genes that also pass the 3-table rule** (CellChat edge + DEG + n_cells)

| gene | celltype_id | IDH_status | p_tumor_synaptic | rule_pass | deg_log2fc | deg_neglog10_padj | cc_edges | ctx_min_n | why_tumor_synaptic |
|---|---|---|---|---|---|---|---|---|---|
| GPR37L1 | OPC_GABA_like_tumor | IDH_Mutant | 0.895 | 1 | 0.829 | 11.3 | 9 | 1658 | knn_frac_synaptic=0.65 (+1.82); emb_07=0.837 (+0.27); emb_15=1.81 (+0.25) |
| PCDHGA10 | OPC_GABA_like_tumor | IDH_WT | 0.756 | 1 | 0.841 | 72.6 | 2 | 1935 | knn_frac_synaptic=0.5 (+1.15); emb_15=1.61 (+0.27); emb_17=1.32 (+0.08) |
| ITGB8 | cycling_tumor | IDH_Mutant | 0.755 | 1 | 5.55 | 22.5 | 1 | 136 | emb_15=1.55 (+0.34); emb_07=1.32 (+0.27); emb_17=3.23 (+0.18) |
| ROBO1 | cycling_tumor | IDH_WT | 0.722 | 1 | 2.51 | 176 | 1 | 663 | knn_frac_synaptic=0.4 (+0.94); emb_07=0.949 (+0.24); emb_15=-0.0558 (+0.10) |
| PSAP | cycling_tumor | IDH_Mutant | 0.715 | 1 | -2.44 | 2.31 | 2 | 136 | emb_15=1.59 (+0.33); emb_07=1.23 (+0.23); emb_05=-1.38 (+0.12) |
| NCL | OPC_GABA_like_tumor | IDH_WT | 0.663 | 1 | 0.878 | 169 | 18 | 1935 | knn_frac_synaptic=0.25 (+0.39); emb_07=0.621 (+0.13); emb_04=-0.979 (+0.10) |
| JAM3 | cycling_tumor | IDH_WT | 0.652 | 1 | 2.63 | 205 | 3 | 663 | knn_frac_synaptic=0.4 (+0.98); emb_07=1.27 (+0.28); emb_15=1.14 (+0.23) |
| PTGES3 | cycling_tumor | IDH_WT | 0.639 | 1 | 0.601 | 51.2 | 2 | 663 | knn_frac_synaptic=0.3 (+0.47); emb_20=0.808 (+0.08); emb_04=-0.785 (+0.07) |
| ITGAV | cycling_tumor | IDH_Mutant | 0.615 | 1 | 0.539 | 1.86 | 13 | 136 | emb_15=2.58 (+0.35); emb_07=0.801 (+0.16); emb_05=-1.18 (+0.10) |
| SLC1A3 | cycling_tumor | IDH_WT | 0.545 | 1 | -0.341 | 7.48 | 13 | 663 | knn_frac_synaptic=0.25 (+0.40); emb_15=2.18 (+0.33); emb_07=0.519 (+0.14) |
| BSG | OPC_GABA_AC_like_tumor | IDH_Mutant | 0.52 | 1 | 0.569 | 1.36 | 11 | 352 | emb_15=1.15 (+0.21); emb_17=0.664 (+0.07); emb_20=1.43 (+0.07) |
| PDGFA | cycling_tumor | IDH_WT | 0.513 | 1 | 1.03 | 76.4 | 3 | 663 | emb_15=1.77 (+0.33); emb_17=1.77 (+0.09); emb_04=-0.748 (+0.06) |
| DLL1 | OPC_GABA_like_tumor | IDH_Mutant | 0.444 | 1 | 2.04 | 65.9 | 2 | 1658 | emb_04=-1.37 (+0.12); emb_15=-0.0244 (+0.10); emb_17=1.73 (+0.10) |
| MPZL1 | cycling_tumor | IDH_WT | 0.444 | 1 | 2.13 | 192 | 4 | 663 | emb_04=-2.09 (+0.26); emb_09=-0.754 (+0.08); emb_20=0.53 (+0.08) |
| PTGES2 | cycling_tumor | IDH_WT | 0.415 | 1 | 0.917 | 108 | 1 | 663 | emb_15=0.678 (+0.21); emb_09=-0.8 (+0.09); emb_20=0.909 (+0.07) |

**Feature importance (gain, top 10)**

| feature | gain | splits | mean_abs_shap |
|---|---|---|---|
| knn_frac_synaptic | 3.87e+04 | 84 | 0.326 |
| emb_15 | 8.85e+03 | 37.4 | 0.208 |
| emb_07 | 5.73e+03 | 28.2 | 0.12 |
| emb_17 | 4.61e+03 | 37.4 | 0.0963 |
| emb_04 | 3.4e+03 | 28.8 | 0.109 |
| emb_09 | 3.26e+03 | 23.2 | 0.1 |
| emb_21 | 2.31e+03 | 22.2 | 0.0692 |
| emb_20 | 2.11e+03 | 15.6 | 0.0647 |
| emb_19 | 1.56e+03 | 14.8 | 0.0302 |
| emb_26 | 1.48e+03 | 15.6 | 0.0446 |

## immune_synapse

GO:0001772 immunological synapse genes, scored on both sides of the synapse. Compartments: T_cell, TAM_microglia, tumor.
45 of 244 set genes are in the universe; 135 positive rows of 27346 (prevalence 0.005).

**Model vs single-column baselines (row level)**

| scorer | auroc | auprc | precision_at_10 | precision_at_25 | precision_at_50 |
|---|---|---|---|---|---|
| lightgbm_oof | 0.706 | 0.0202 | 0 | 0 | 0 |
| rule_pass | 0.555 | 0.0297 | 0.2 | 0.16 | 0.1 |
| deg_abs_log2fc | 0.54 | 0.00966 | 0.1 | 0.04 | 0.02 |
| cc_edges | 0.55 | 0.00914 | 0 | 0 | 0 |
| cc_max_prob | 0.55 | 0.00911 | 0 | 0 | 0 |

**Gene level (best context per gene)**

| scorer | auroc | auprc | precision_at_10 | precision_at_25 | precision_at_50 |
|---|---|---|---|---|---|
| lightgbm_oof | 0.732 | 0.019 | 0 | 0 | 0.04 |
| deg_abs_log2fc | 0.533 | 0.0105 | 0.1 | 0.04 | 0.02 |
| cc_edges | 0.528 | 0.0062 | 0 | 0 | 0 |

**Published-33 check**

- Published genes already in this label set: HAVCR2, HLA-DRA.
- Published genes scored: 32 of 33; not in universe: AXL.
- Median gene-rank percentile of published genes: 0.49 (1.00 = top). AUROC published vs all other genes: 0.557 (single-column baselines, best context per gene: deg_abs_log2fc 0.901, deg_pct_e2 0.808, cc_edges 0.982, deg_n_celltypes_sig 0.737).
- Held-out test, genes **not** in the label set only (30 published): AUROC 0.530.

| gene | published_tier | in_label_set | celltype_id | IDH_status | best_score | gene_rank | gene_percentile |
|---|---|---|---|---|---|---|---|
| TREM2 | 2 | 0 | TAM1/TAM2/microglia | IDH_WT | 0.822 | 23 | 0.998 |
| HAVCR2 | 1 | 1 | TAM1/TAM2/microglia | IDH_Mutant | 0.731 | 52 | 0.995 |
| CD74 | 1 | 0 | OPC_GABA_AC_like_tumor | IDH_Mutant | 0.724 | 59 | 0.994 |
| TYROBP | 2 | 0 | AC_like_tumor | IDH_WT | 0.577 | 174 | 0.982 |
| CD4 | 2 | 0 | cycling_tumor | IDH_Mutant | 0.47 | 321 | 0.967 |
| HLA-DRA | 2 | 1 | OPC_GABA_like_tumor | IDH_Mutant | 0.41 | 442 | 0.954 |
| SPP1 | 1 | 0 | OPC_GABA_like_tumor | IDH_WT | 0.379 | 511 | 0.947 |
| LILRB4 | 2 | 0 | TAM1/TAM2/microglia | IDH_WT | 0.341 | 631 | 0.934 |
| TGFB1 | 2 | 0 | cycling_tumor | IDH_Mutant | 0.34 | 641 | 0.933 |
| KLRB1 | 1 | 0 | T cell | IDH_Mutant | 0.312 | 747 | 0.922 |
| LGALS9 | 2 | 0 | cycling_tumor | IDH_Mutant | 0.267 | 982 | 0.898 |
| CXCR4 | 1 | 0 | T cell | IDH_WT | 0.204 | 1492 | 0.844 |
| CLEC2D | 1 | 0 | T cell | IDH_Mutant | 0.192 | 1623 | 0.831 |
| APOE | 2 | 0 | OPC_GABA_like_tumor | IDH_WT | 0.143 | 2259 | 0.764 |
| ENTPD1 | 1 | 0 | TAM1/TAM2/microglia | IDH_Mutant | 0.13 | 2491 | 0.74 |
| GRIA2 | 1 | 0 | OPC_GABA_like_tumor | IDH_WT | 0.0663 | 4134 | 0.569 |
| NLGN3 | 1 | 0 | OPC_GABA_like_tumor | IDH_WT | 0.0342 | 5634 | 0.412 |
| PTN | 1 | 0 | T cell | IDH_WT | 0.0301 | 5916 | 0.383 |
| NLGN1 | 2 | 0 | OPC_GABA_like_tumor | IDH_WT | 0.0283 | 6046 | 0.369 |
| CNTN1 | 2 | 0 | OPC_GABA_like_tumor | IDH_WT | 0.0277 | 6096 | 0.364 |
| CD44 | 1 | 0 | OPC_GABA_like_tumor | IDH_WT | 0.0216 | 6585 | 0.313 |
| NCAM1 | 2 | 0 | AC_like_tumor | IDH_WT | 0.0194 | 6764 | 0.294 |
| PTPRZ1 | 1 | 0 | AC_like_tumor | IDH_Mutant | 0.0192 | 6782 | 0.292 |
| PDGFRA | 1 | 0 | OPC_GABA_like_tumor | IDH_WT | 0.0182 | 6881 | 0.282 |
| DLL3 | 1 | 0 | OPC_GABA_like_tumor | IDH_WT | 0.0152 | 7172 | 0.251 |
| NOTCH1 | 2 | 0 | cycling_tumor | IDH_WT | 0.0131 | 7392 | 0.228 |
| GRIA4 | 2 | 0 | OPC_GABA_AC_like_tumor | IDH_Mutant | 0.00817 | 7989 | 0.166 |
| CD99 | 2 | 0 | cycling_tumor | IDH_Mutant | 0.00363 | 8698 | 0.0922 |
| MIF | 1 | 0 | cycling_tumor | IDH_Mutant | 0.00271 | 8878 | 0.0734 |
| GRIA3 | 2 | 0 | cycling_tumor | IDH_WT | 0.0005 | 9417 | 0.0171 |
| EGFR | 1 | 0 | OPC_GABA_like_tumor | IDH_Mutant | 0.00032 | 9498 | 0.00866 |
| NRXN1 | 2 | 0 | cycling_tumor | IDH_WT | 7.61e-05 | 9574 | 0.000731 |

**Top 25 novel genes** (in neither gene set nor the published 33, housekeeping genes removed; best context per gene)

| gene | celltype_id | IDH_status | p_immune_synapse | rule_pass | deg_log2fc | deg_neglog10_padj | cc_edges | ctx_min_n | why_immune_synapse |
|---|---|---|---|---|---|---|---|---|---|
| NCF1 | TAM1/TAM2/microglia | IDH_Mutant | 0.915 | 0 | -0.659 | 15.7 | 0 | 2106 | emb_03=-1.35 (+1.37); knn_frac_immune=0.1 (+0.68); emb_17=-1.21 (+0.65) |
| ST3GAL1 | TAM1/TAM2/microglia | IDH_Mutant | 0.909 | 0 | 1.27 | 28.7 | 0 | 2106 | emb_03=-1.73 (+1.09); emb_10=-0.741 (+1.02); emb_17=-2.45 (+0.80) |
| PRDM1 | T cell | IDH_Mutant | 0.902 | 0 | -1.09 | 1.61 | 0 | 334 | emb_10=-1.54 (+1.07); emb_03=-2.72 (+1.06); knn_frac_immune=0.2 (+0.89) |
| PTGDR | T cell | IDH_Mutant | 0.9 | 0 | 1.38 | 6.08 | 0 | 334 | emb_10=-0.798 (+1.28); emb_03=-2.1 (+1.11); emb_17=-2.45 (+0.89) |
| LCP2 | TAM1/TAM2/microglia | IDH_WT | 0.896 | 0 | 0.469 | 31.1 | 0 | 2745 | emb_03=-1.34 (+1.17); emb_10=-1.62 (+0.92); emb_17=-0.905 (+0.58) |
| FCER1G | OPC_GABA_like_tumor | IDH_Mutant | 0.886 | 0 | -2.06 | 14.3 | 0 | 1658 | emb_03=-1.37 (+1.36); emb_10=-1.17 (+0.92); emb_17=-1.55 (+0.64) |
| RNASET2 | OPC_GABA_like_tumor | IDH_Mutant | 0.873 | 0 | -1.44 | 14.9 | 0 | 1658 | emb_03=-0.855 (+1.49); knn_frac_immune=0.05 (+0.63); emb_17=-1.32 (+0.61) |
| CD48 | TAM1/TAM2/microglia | IDH_WT | 0.873 | 0 | -2.37 | 42 | 0 | 2745 | emb_03=-3.33 (+1.77); emb_10=-1.14 (+0.97); emb_17=-2.92 (+0.78) |
| FCGR3A | TAM1/TAM2/microglia | IDH_WT | 0.864 | 0 | 0.542 | 52.4 | 0 | 2745 | emb_03=-0.75 (+1.57); emb_10=-1.01 (+0.90); emb_17=-2.79 (+0.66) |
| CD2 | T cell | IDH_WT | 0.862 | 0 | -1.01 | 33.3 | 0 | 1118 | emb_10=-0.973 (+1.50); emb_03=-5.32 (+0.98); knn_frac_immune=0.15 (+0.90) |
| IL10RA | TAM1/TAM2/microglia | IDH_WT | 0.854 | 0 | 0.606 | 34.4 | 0 | 2745 | emb_03=-1.27 (+1.63); emb_17=-2.81 (+0.94); knn_frac_immune=0.15 (+0.88) |
| RAC2 | cycling_tumor | IDH_Mutant | 0.852 | 0 | -7.2 | 1.46 | 0 | 136 | emb_03=-1.93 (+1.45); emb_10=-2.03 (+1.22); knn_frac_immune=0.15 (+0.81) |
| HLA-DQB1 | cycling_tumor | IDH_Mutant | 0.85 | 0 | -6.39 | 8.16 | 0 | 136 | emb_03=-2.52 (+1.90); emb_10=-1.11 (+1.46); knn_frac_immune=0.15 (+0.85) |
| TRADD | T cell | IDH_WT | 0.845 | 0 | -0.84 | 2.06 | 0 | 1118 | emb_03=-1.72 (+1.13); emb_01=0.52 (+0.78); emb_20=1.09 (+0.71) |
| TNFRSF1B | TAM1/TAM2/microglia | IDH_WT | 0.838 | 1 | 0.471 | 29.5 | 2 | 2745 | emb_03=-1.11 (+1.20); emb_10=-1.19 (+0.90); knn_frac_immune=0.1 (+0.81) |
| CXCL16 | TAM1/TAM2/microglia | IDH_WT | 0.838 | 0 | 0.288 | 22.9 | 0 | 2745 | emb_03=-0.501 (+1.30); emb_10=-1.34 (+0.96); emb_17=-1.24 (+0.61) |
| LCP1 | cycling_tumor | IDH_Mutant | 0.836 | 0 | -7.88 | 5.61 | 0 | 136 | emb_03=-1.33 (+1.61); emb_10=-0.866 (+1.41); knn_frac_immune=0.1 (+0.89) |
| NCR3 | T cell | IDH_Mutant | 0.834 | 0 | 2.32 | 11.5 | 0 | 334 | emb_03=-2.62 (+1.49); emb_10=-2.04 (+0.92); knn_frac_immune=0.1 (+0.89) |
| CYTIP | TAM1/TAM2/microglia | IDH_Mutant | 0.833 | 0 | -1.39 | 42.9 | 0 | 2106 | emb_03=-3.88 (+1.73); emb_10=-1.58 (+0.99); knn_frac_immune=0.1 (+0.87) |
| GPRIN3 | TAM1/TAM2/microglia | IDH_WT | 0.831 | 0 | 0.337 | 7.9 | 0 | 2745 | emb_03=-1.27 (+1.58); knn_frac_immune=0.05 (+0.74); emb_10=-1.39 (+0.66) |
| PILRA | TAM1/TAM2/microglia | IDH_Mutant | 0.829 | 1 | -0.465 | 7.19 | 7 | 2106 | emb_03=-1.28 (+1.33); emb_10=-1.84 (+0.88); emb_01=1.49 (+0.63) |
| CD84 | TAM1/TAM2/microglia | IDH_Mutant | 0.819 | 0 | 0.527 | 25.1 | 0 | 2106 | emb_03=-1.3 (+1.27); emb_10=-2.28 (+0.78); knn_frac_immune=0.1 (+0.74) |
| SAMSN1 | T cell | IDH_WT | 0.815 | 0 | -0.688 | 2.83 | 0 | 1118 | emb_03=-1.67 (+1.32); emb_10=-1.92 (+0.84); emb_01=1.27 (+0.58) |
| C2 | TAM1/TAM2/microglia | IDH_Mutant | 0.814 | 0 | 0.435 | 6.31 | 0 | 2106 | emb_03=-0.4 (+1.45); emb_10=-1.31 (+0.94); emb_17=-2.75 (+0.66) |
| HPSE | TAM1/TAM2/microglia | IDH_Mutant | 0.813 | 0 | 0.548 | 5.82 | 0 | 2106 | emb_03=-0.959 (+0.71); emb_17=-0.767 (+0.64); emb_20=1.51 (+0.62) |

**Novel genes that also pass the 3-table rule** (CellChat edge + DEG + n_cells)

| gene | celltype_id | IDH_status | p_immune_synapse | rule_pass | deg_log2fc | deg_neglog10_padj | cc_edges | ctx_min_n | why_immune_synapse |
|---|---|---|---|---|---|---|---|---|---|
| TNFRSF1B | TAM1/TAM2/microglia | IDH_WT | 0.838 | 1 | 0.471 | 29.5 | 2 | 2745 | emb_03=-1.11 (+1.20); emb_10=-1.19 (+0.90); knn_frac_immune=0.1 (+0.81) |
| PILRA | TAM1/TAM2/microglia | IDH_Mutant | 0.829 | 1 | -0.465 | 7.19 | 7 | 2106 | emb_03=-1.28 (+1.33); emb_10=-1.84 (+0.88); emb_01=1.49 (+0.63) |
| HLA-DMA | cycling_tumor | IDH_WT | 0.742 | 1 | -6.04 | 38.4 | 2 | 663 | emb_03=-1.26 (+1.72); emb_10=-0.69 (+1.25); knn_frac_immune=0.1 (+0.65) |
| PTPRC | T cell | IDH_WT | 0.66 | 1 | -0.767 | 29 | 4 | 1118 | emb_03=-2.68 (+1.49); knn_frac_immune=0.05 (+0.80); emb_17=-1.72 (+0.66) |
| HLA-DRB5 | TAM1/TAM2/microglia | IDH_Mutant | 0.652 | 1 | -1.11 | 232 | 4 | 2106 | emb_03=-1.56 (+1.79); emb_10=-0.8 (+1.50); knn_frac_immune=0.1 (+1.08) |
| LAIR1 | TAM1/TAM2/microglia | IDH_Mutant | 0.641 | 1 | -0.275 | 3.68 | 4 | 2106 | emb_03=-1.49 (+1.70); emb_10=-2.33 (+1.12); knn_frac_immune=0.05 (+0.84) |
| FPR1 | TAM1/TAM2/microglia | IDH_Mutant | 0.626 | 1 | 0.696 | 39.5 | 10 | 2106 | emb_03=-1.71 (+1.51); emb_10=-3.48 (+1.22); emb_01=1.01 (+0.55) |
| PTGER4 | TAM1/TAM2/microglia | IDH_WT | 0.58 | 1 | 0.716 | 45.6 | 14 | 2745 | emb_10=-1.8 (+1.24); emb_03=-3.02 (+1.13); emb_17=-3.07 (+0.90) |
| PSAP | cycling_tumor | IDH_Mutant | 0.579 | 1 | -2.44 | 2.31 | 2 | 136 | emb_03=-1.09 (+1.30); emb_20=1.87 (+0.80); knn_frac_immune=0.05 (+0.74) |
| C3AR1 | TAM1/TAM2/microglia | IDH_Mutant | 0.533 | 1 | 0.894 | 107 | 4 | 2106 | emb_03=-1.04 (+1.63); emb_10=-1.35 (+1.35); knn_frac_immune=0.05 (+1.02) |
| CD69 | T cell | IDH_WT | 0.532 | 1 | -0.902 | 16.4 | 4 | 1118 | emb_10=-3.07 (+1.31); emb_03=-3.84 (+0.97); emb_17=-4.03 (+0.80) |
| HLA-DPA1 | T cell | IDH_WT | 0.485 | 1 | 0.639 | 5.05 | 4 | 1118 | emb_03=-2.05 (+1.75); knn_frac_immune=0.1 (+0.82); emb_01=0.712 (+0.75) |
| HBEGF | TAM1/TAM2/microglia | IDH_WT | 0.483 | 1 | 0.91 | 64.8 | 7 | 2745 | emb_03=-3.74 (+1.25); emb_10=-1.86 (+1.13); emb_17=-2.64 (+0.66) |
| ITGAX | TAM1/TAM2/microglia | IDH_WT | 0.455 | 1 | 0.416 | 16 | 4 | 2745 | emb_03=-0.982 (+1.21); emb_01=1.49 (+0.57); emb_17=-1.62 (+0.49) |
| HLA-DPB1 | TAM1/TAM2/microglia | IDH_WT | 0.364 | 1 | -0.376 | 5.58 | 4 | 2745 | emb_03=-1.53 (+1.71); knn_frac_immune=0.15 (+0.85); emb_17=-2.18 (+0.49) |

**Feature importance (gain, top 10)**

| feature | gain | splits | mean_abs_shap |
|---|---|---|---|
| emb_03 | 2.29e+05 | 28.4 | 0.886 |
| emb_10 | 1.79e+05 | 27.2 | 0.456 |
| emb_20 | 7.1e+04 | 20.8 | 0.608 |
| emb_01 | 6.46e+04 | 21.6 | 0.585 |
| emb_27 | 5.29e+04 | 19.2 | 0.207 |
| emb_02 | 5.12e+04 | 26 | 0.563 |
| emb_07 | 5.03e+04 | 12 | 0.248 |
| emb_17 | 4.74e+04 | 25.2 | 0.28 |
| emb_31 | 2.63e+04 | 26 | 0.325 |
| emb_24 | 2.59e+04 | 15.2 | 0.177 |

## Caveats

- GO:0001772 includes all taxa (matching the team's 5,614-annotation download); most are IEA (electronic) annotations mapped by upper-casing symbols. `--immune-human-only` restricts to genes with a human annotation.
- SynGO annotations are mostly from rodent brain synapses; membership says a gene is synaptic in neurons, not in glioma.
- Gene-set membership is a weak label: unlabeled genes are unknown, not negative.
- CellChat features are zero for most genes in the DEG universe (CellChat covers ~114 genes); DEG and n_cells carry most signal.
- SynGO genes are enriched for broadly expressed genes, so the tumor-synaptic model partly learns "highly expressed". Mitochondrial / ribosomal / histone genes are flagged `housekeeping` and left out of the novel tables.
- Immune genes (HLA class II, ITGB2, C3, CCL2) show up as strong Ephys_1-high DEGs inside *tumor* clusters, which suggests myeloid contamination / doublets in Ephys_1 tumor clusters. Immune-synapse nominees in tumor cell types should be checked for this before follow-up.
- High score = "expression pattern resembles known synaptic / immune-synapse genes", not validation.
