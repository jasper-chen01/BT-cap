# LightGBM with knowledge-tool labels: tumor synaptic vs immune synapse

- Universe: `deg`: 27346 gene x cell type x IDH rows, 9580 genes, 75 features.
- Scores are out-of-fold (StratifiedGroupKFold(n_splits=5) grouped by gene, seeds [0, 1, 2, 3, 4]): no gene is scored by a model that trained on it.
- The 33 published glioma genes are **not** labels here. `in_published_33` flags them; the check below shows where each model ranks them.
- LLM run features: none.
- Feature arm: `tables_scetm`.
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
| lightgbm_oof | 0.742 | 0.437 | 0.8 | 0.92 | 0.96 |
| rule_pass | 0.516 | 0.174 | 0.2 | 0.2 | 0.38 |
| deg_abs_log2fc | 0.507 | 0.154 | 0 | 0.04 | 0.02 |
| cc_edges | 0.514 | 0.164 | 0.6 | 0.64 | 0.54 |
| cc_max_prob | 0.514 | 0.165 | 0.9 | 0.76 | 0.64 |

**Gene level (best context per gene)**

| scorer | auroc | auprc | precision_at_10 | precision_at_25 | precision_at_50 |
|---|---|---|---|---|---|
| lightgbm_oof | 0.726 | 0.368 | 0.9 | 0.88 | 0.9 |
| deg_abs_log2fc | 0.539 | 0.146 | 0 | 0.04 | 0.02 |
| cc_edges | 0.512 | 0.14 | 0.6 | 0.48 | 0.54 |

**Published-33 check**

- Published genes already in this label set: APOE, CD44, CNTN1, EGFR, GRIA2, GRIA3, GRIA4, NCAM1, NLGN1, NLGN3, NOTCH1, NRXN1, PTN, PTPRZ1.
- Published genes scored: 30 of 33; not in universe: AXL, ENTPD1, KLRB1.
- Median gene-rank percentile of published genes: 0.82 (1.00 = top). AUROC published vs all other genes: 0.739 (single-column baselines, best context per gene: deg_abs_log2fc 0.906, deg_pct_e2 0.574, cc_edges 0.831, deg_n_celltypes_sig 0.758).
- Held-out test, genes **not** in the label set only (16 published): AUROC 0.634.

| gene | published_tier | in_label_set | celltype_id | IDH_status | best_score | gene_rank | gene_percentile |
|---|---|---|---|---|---|---|---|
| NCAM1 | 2 | 1 | OPC_GABA_like_tumor | IDH_Mutant | 0.931 | 71 | 0.992 |
| PTPRZ1 | 1 | 1 | OPC_GABA_like_tumor | IDH_WT | 0.92 | 80 | 0.991 |
| NRXN1 | 2 | 1 | AC_like_tumor | IDH_Mutant | 0.907 | 93 | 0.989 |
| NLGN1 | 2 | 1 | OPC_GABA_like_tumor | IDH_WT | 0.904 | 94 | 0.989 |
| GRIA2 | 1 | 1 | OPC_GABA_like_tumor | IDH_Mutant | 0.888 | 104 | 0.988 |
| GRIA4 | 2 | 1 | OPC_GABA_like_tumor | IDH_Mutant | 0.872 | 115 | 0.987 |
| NLGN3 | 1 | 1 | cycling_tumor | IDH_WT | 0.861 | 121 | 0.986 |
| EGFR | 1 | 1 | OPC_GABA_like_tumor | IDH_Mutant | 0.861 | 122 | 0.986 |
| CNTN1 | 2 | 1 | cycling_tumor | IDH_WT | 0.847 | 132 | 0.985 |
| PDGFRA | 1 | 0 | OPC_GABA_like_tumor | IDH_WT | 0.771 | 241 | 0.972 |
| DLL3 | 1 | 0 | OPC_GABA_like_tumor | IDH_WT | 0.771 | 242 | 0.972 |
| MIF | 1 | 0 | OPC_GABA_like_tumor | IDH_Mutant | 0.723 | 369 | 0.958 |
| GRIA3 | 2 | 1 | cycling_tumor | IDH_Mutant | 0.576 | 1197 | 0.863 |
| NOTCH1 | 2 | 1 | cycling_tumor | IDH_WT | 0.568 | 1300 | 0.851 |
| HLA-DRA | 2 | 0 | AC_like_tumor | IDH_Mutant | 0.545 | 1534 | 0.824 |
| TYROBP | 2 | 0 | AC_like_tumor | IDH_Mutant | 0.535 | 1641 | 0.812 |
| CD74 | 1 | 0 | cycling_tumor | IDH_Mutant | 0.492 | 2202 | 0.747 |
| PTN | 1 | 1 | OPC_GABA_like_tumor | IDH_WT | 0.491 | 2224 | 0.745 |
| CLEC2D | 1 | 0 | OPC_GABA_like_tumor | IDH_WT | 0.484 | 2329 | 0.733 |
| CD44 | 1 | 1 | OPC_GABA_like_tumor | IDH_Mutant | 0.471 | 2543 | 0.708 |
| TREM2 | 2 | 0 | cycling_tumor | IDH_Mutant | 0.469 | 2566 | 0.706 |
| SPP1 | 1 | 0 | AC_like_tumor | IDH_WT | 0.411 | 3709 | 0.575 |
| HAVCR2 | 1 | 0 | cycling_tumor | IDH_Mutant | 0.399 | 3973 | 0.544 |
| CD99 | 2 | 0 | cycling_tumor | IDH_WT | 0.381 | 4449 | 0.49 |
| APOE | 2 | 1 | AC_like_tumor | IDH_Mutant | 0.367 | 4815 | 0.448 |
| LGALS9 | 2 | 0 | cycling_tumor | IDH_Mutant | 0.339 | 5510 | 0.368 |
| LILRB4 | 2 | 0 | cycling_tumor | IDH_Mutant | 0.327 | 5807 | 0.334 |
| TGFB1 | 2 | 0 | cycling_tumor | IDH_Mutant | 0.321 | 5930 | 0.32 |
| CD4 | 2 | 0 | cycling_tumor | IDH_Mutant | 0.289 | 6708 | 0.23 |
| CXCR4 | 1 | 0 | cycling_tumor | IDH_Mutant | 0.209 | 8127 | 0.0676 |

**Top 25 novel genes** (in neither gene set nor the published 33, housekeeping genes removed; best context per gene)

| gene | celltype_id | IDH_status | p_tumor_synaptic | rule_pass | deg_log2fc | deg_neglog10_padj | cc_edges | ctx_min_n | why_tumor_synaptic |
|---|---|---|---|---|---|---|---|---|---|
| PFDN5 | cycling_tumor | IDH_WT | 0.957 | 0 | -0.674 | 23 | 0 | 663 | knn_frac_synaptic=0.95 (+2.21); deg_pct_e2=0.94 (+0.22); emb_07=1.17 (+0.22) |
| NOP53 | OPC_GABA_like_tumor | IDH_Mutant | 0.922 | 0 | 0.666 | 54.5 | 0 | 1658 | knn_frac_synaptic=0.9 (+2.16); deg_pct_e2=0.695 (+0.22); emb_07=0.793 (+0.15) |
| NPM1 | OPC_GABA_like_tumor | IDH_Mutant | 0.916 | 0 | 0.584 | 95 | 0 | 1658 | knn_frac_synaptic=0.7 (+1.92); deg_pct_e2=0.902 (+0.24); deg_pct_e1=0.829 (+0.21) |
| SCG3 | OPC_GABA_like_tumor | IDH_WT | 0.893 | 0 | 0.291 | 61.6 | 0 | 1935 | knn_frac_synaptic=0.6 (+1.68); emb_15=1.96 (+0.25); emb_07=1.34 (+0.24) |
| NSA2 | cycling_tumor | IDH_WT | 0.892 | 0 | -0.546 | 4.24 | 0 | 663 | knn_frac_synaptic=0.95 (+1.97); deg_pct_e2=0.686 (+0.18); emb_15=0.42 (+0.13) |
| GPR37L1 | OPC_GABA_like_tumor | IDH_Mutant | 0.888 | 1 | 0.829 | 11.3 | 9 | 1658 | knn_frac_synaptic=0.65 (+1.70); emb_15=1.81 (+0.26); emb_07=0.837 (+0.23) |
| SLC25A6 | OPC_GABA_like_tumor | IDH_Mutant | 0.885 | 0 | 0.8 | 88.4 | 0 | 1658 | knn_frac_synaptic=0.85 (+2.12); deg_pct_e2=0.769 (+0.19); deg_pct_e1=0.633 (+0.13) |
| C6orf48 | AC_like_tumor | IDH_Mutant | 0.881 | 0 | 0.408 | 4.78 | 0 | 283 | knn_frac_synaptic=0.8 (+1.81); emb_15=0.743 (+0.13); deg_pct_e2=0.452 (+0.13) |
| SEC11A | OPC_GABA_like_tumor | IDH_WT | 0.878 | 0 | -0.381 | 9.56 | 0 | 1935 | knn_frac_synaptic=0.7 (+1.86); deg_pct_e2=0.611 (+0.15); deg_pct_e1=0.374 (+0.13) |
| LHFPL3 | OPC_GABA_like_tumor | IDH_WT | 0.875 | 0 | 2.67 | 300 | 0 | 1935 | knn_frac_synaptic=0.4 (+0.84); emb_15=1.86 (+0.30); emb_07=0.575 (+0.16) |
| NAP1L1 | OPC_GABA_like_tumor | IDH_Mutant | 0.874 | 0 | 0.544 | 79.8 | 0 | 1658 | knn_frac_synaptic=0.55 (+1.81); deg_pct_e1=0.717 (+0.22); deg_pct_e2=0.847 (+0.21) |
| TOMM7 | cycling_tumor | IDH_WT | 0.871 | 0 | -1.3 | 79.5 | 0 | 663 | knn_frac_synaptic=0.65 (+1.87); deg_pct_e2=0.908 (+0.22); deg_pct_e1=0.846 (+0.18) |
| ZFAS1 | OPC_GABA_like_tumor | IDH_Mutant | 0.866 | 0 | 0.702 | 114 | 0 | 1658 | knn_frac_synaptic=0.7 (+1.64); emb_07=1.06 (+0.23); deg_pct_e2=0.904 (+0.19) |
| FUT9 | cycling_tumor | IDH_Mutant | 0.864 | 0 | 3.69 | 12.3 | 0 | 136 | knn_frac_synaptic=0.45 (+1.03); emb_15=0.172 (+0.17); emb_05=-2.22 (+0.17) |
| ADGRA3 | OPC_GABA_like_tumor | IDH_Mutant | 0.859 | 0 | -0.663 | 6.8 | 0 | 1658 | knn_frac_synaptic=0.55 (+1.84); emb_15=0.342 (+0.21); emb_05=-1.88 (+0.16) |
| LRP1B | OPC_GABA_like_tumor | IDH_Mutant | 0.856 | 0 | -1.15 | 10.9 | 0 | 1658 | knn_frac_synaptic=0.6 (+1.66); emb_15=1.27 (+0.24); emb_07=1.66 (+0.22) |
| PIANP | OPC_GABA_like_tumor | IDH_Mutant | 0.856 | 0 | 0.922 | 13.6 | 0 | 1658 | knn_frac_synaptic=0.5 (+1.08); emb_15=0.817 (+0.24); emb_07=1.31 (+0.22) |
| DPP10 | OPC_GABA_like_tumor | IDH_Mutant | 0.849 | 0 | -1.17 | 8.01 | 0 | 1658 | knn_frac_synaptic=0.6 (+1.70); emb_15=1.08 (+0.22); emb_07=1.06 (+0.19) |
| CHL1 | cycling_tumor | IDH_WT | 0.848 | 0 | -0.259 | 5.39 | 0 | 663 | knn_frac_synaptic=0.6 (+1.68); emb_15=2.55 (+0.28); emb_07=2.07 (+0.26) |
| ZNF462 | OPC_GABA_like_tumor | IDH_Mutant | 0.843 | 0 | 1.28 | 79.3 | 0 | 1658 | knn_frac_synaptic=0.45 (+1.03); emb_17=3.53 (+0.52); emb_07=0.939 (+0.25) |
| CCNI | OPC_GABA_like_tumor | IDH_Mutant | 0.838 | 0 | 0.854 | 170 | 0 | 1658 | knn_frac_synaptic=0.45 (+1.06); deg_pct_e2=0.896 (+0.20); deg_pct_e1=0.73 (+0.19) |
| SNRPD2 | AC_like_tumor | IDH_Mutant | 0.837 | 0 | 0.282 | 3.01 | 0 | 283 | knn_frac_synaptic=0.65 (+1.90); deg_pct_e2=0.498 (+0.15); deg_pct_e1=0.313 (+0.15) |
| LUZP2 | OPC_GABA_like_tumor | IDH_Mutant | 0.835 | 0 | -1.34 | 128 | 0 | 1658 | knn_frac_synaptic=0.45 (+1.01); emb_15=2.48 (+0.30); emb_05=-2.05 (+0.17) |
| MMP16 | OPC_GABA_like_tumor | IDH_WT | 0.83 | 0 | 2.45 | 300 | 0 | 1935 | knn_frac_synaptic=0.45 (+0.99); emb_15=2.2 (+0.25); emb_19=1.34 (+0.16) |
| DDR1 | OPC_GABA_like_tumor | IDH_WT | 0.827 | 0 | 0.403 | 84 | 0 | 1935 | knn_frac_synaptic=0.4 (+0.88); emb_15=1.08 (+0.19); emb_05=-1.52 (+0.16) |

**Novel genes that also pass the 3-table rule** (CellChat edge + DEG + n_cells)

| gene | celltype_id | IDH_status | p_tumor_synaptic | rule_pass | deg_log2fc | deg_neglog10_padj | cc_edges | ctx_min_n | why_tumor_synaptic |
|---|---|---|---|---|---|---|---|---|---|
| GPR37L1 | OPC_GABA_like_tumor | IDH_Mutant | 0.888 | 1 | 0.829 | 11.3 | 9 | 1658 | knn_frac_synaptic=0.65 (+1.70); emb_15=1.81 (+0.26); emb_07=0.837 (+0.23) |
| COL9A1 | cycling_tumor | IDH_WT | 0.821 | 1 | 4.56 | 300 | 5 | 663 | knn_frac_synaptic=0.35 (+0.62); emb_17=4.15 (+0.49); emb_15=1.38 (+0.29) |
| NCL | OPC_GABA_like_tumor | IDH_Mutant | 0.801 | 1 | 0.585 | 75.5 | 18 | 1658 | knn_frac_synaptic=0.25 (+0.39); deg_pct_e1=0.706 (+0.22); deg_pct_e2=0.834 (+0.21) |
| ITGB8 | cycling_tumor | IDH_Mutant | 0.797 | 1 | 5.55 | 22.5 | 1 | 136 | emb_15=1.55 (+0.34); emb_07=1.32 (+0.26); emb_17=3.23 (+0.19) |
| PSAP | OPC_GABA_like_tumor | IDH_Mutant | 0.795 | 1 | -1.36 | 150 | 2 | 1658 | emb_15=1.59 (+0.28); emb_07=1.23 (+0.20); deg_pct_e1=0.698 (+0.19) |
| ROBO1 | OPC_GABA_like_tumor | IDH_Mutant | 0.783 | 1 | 1.24 | 59.1 | 1 | 1658 | knn_frac_synaptic=0.4 (+0.91); emb_07=0.949 (+0.22); deg_pct_e2=0.439 (+0.12) |
| PTGES3 | cycling_tumor | IDH_WT | 0.735 | 1 | 0.601 | 51.2 | 2 | 663 | knn_frac_synaptic=0.3 (+0.45); deg_pct_e2=0.925 (+0.22); deg_pct_e1=0.502 (+0.16) |
| ITGAV | cycling_tumor | IDH_WT | 0.734 | 1 | 0.298 | 54.6 | 13 | 663 | emb_15=2.58 (+0.31); deg_pct_e2=0.501 (+0.16); emb_07=0.801 (+0.15) |
| JAM3 | cycling_tumor | IDH_WT | 0.689 | 1 | 2.63 | 205 | 3 | 663 | knn_frac_synaptic=0.4 (+0.96); emb_07=1.27 (+0.24); emb_15=1.14 (+0.24) |
| SDC3 | cycling_tumor | IDH_WT | 0.673 | 1 | 0.618 | 45.4 | 11 | 663 | knn_frac_synaptic=0.3 (+0.43); emb_15=2.85 (+0.31); emb_04=-1.81 (+0.15) |
| BSG | OPC_GABA_like_tumor | IDH_WT | 0.657 | 1 | 1.01 | 185 | 26 | 1935 | deg_pct_e2=0.777 (+0.21); emb_15=1.15 (+0.18); deg_pct_e1=0.317 (+0.11) |
| SLC1A3 | cycling_tumor | IDH_WT | 0.646 | 1 | -0.341 | 7.48 | 13 | 663 | knn_frac_synaptic=0.25 (+0.38); emb_15=2.18 (+0.25); cc_edges_total=26 (+0.13) |
| KLRC2 | OPC_GABA_like_tumor | IDH_Mutant | 0.552 | 1 | 1.87 | 170 | 6 | 1658 | knn_frac_synaptic=0.25 (+0.46); emb_19=1.5 (+0.19); emb_07=1.04 (+0.16) |
| PDGFA | cycling_tumor | IDH_WT | 0.549 | 1 | 1.03 | 76.4 | 3 | 663 | emb_15=1.77 (+0.30); cc_edges=3 (+0.10); emb_17=1.77 (+0.10) |
| MDK | cycling_tumor | IDH_WT | 0.542 | 1 | -0.411 | 11.4 | 28 | 663 | emb_07=1.9 (+0.20); cc_edges_total=28 (+0.12); emb_09=-1.68 (+0.10) |

**Feature importance (gain, top 10)**

| feature | gain | splits | mean_abs_shap |
|---|---|---|---|
| knn_frac_synaptic | 3.62e+04 | 72.2 | 0.308 |
| emb_15 | 8.25e+03 | 34 | 0.195 |
| emb_07 | 4.96e+03 | 27.8 | 0.105 |
| deg_pct_e2 | 4.21e+03 | 18 | 0.0886 |
| emb_17 | 4.21e+03 | 33.2 | 0.0833 |
| deg_pct_e1 | 3.21e+03 | 12.2 | 0.0568 |
| emb_04 | 2.92e+03 | 26 | 0.0962 |
| emb_09 | 2.55e+03 | 21 | 0.0888 |
| emb_21 | 1.89e+03 | 18.8 | 0.0542 |
| emb_20 | 1.85e+03 | 14 | 0.0538 |

## immune_synapse

GO:0001772 immunological synapse genes, scored on both sides of the synapse. Compartments: T_cell, TAM_microglia, tumor.
45 of 244 set genes are in the universe; 135 positive rows of 27346 (prevalence 0.005).

**Model vs single-column baselines (row level)**

| scorer | auroc | auprc | precision_at_10 | precision_at_25 | precision_at_50 |
|---|---|---|---|---|---|
| lightgbm_oof | 0.725 | 0.021 | 0 | 0.04 | 0.04 |
| rule_pass | 0.555 | 0.0297 | 0.2 | 0.16 | 0.1 |
| deg_abs_log2fc | 0.54 | 0.00966 | 0.1 | 0.04 | 0.02 |
| cc_edges | 0.55 | 0.00914 | 0 | 0 | 0 |
| cc_max_prob | 0.55 | 0.00911 | 0 | 0 | 0 |

**Gene level (best context per gene)**

| scorer | auroc | auprc | precision_at_10 | precision_at_25 | precision_at_50 |
|---|---|---|---|---|---|
| lightgbm_oof | 0.731 | 0.0229 | 0.1 | 0.08 | 0.04 |
| deg_abs_log2fc | 0.533 | 0.0105 | 0.1 | 0.04 | 0.02 |
| cc_edges | 0.528 | 0.0062 | 0 | 0 | 0 |

**Published-33 check**

- Published genes already in this label set: HAVCR2, HLA-DRA.
- Published genes scored: 32 of 33; not in universe: AXL.
- Median gene-rank percentile of published genes: 0.57 (1.00 = top). AUROC published vs all other genes: 0.586 (single-column baselines, best context per gene: deg_abs_log2fc 0.901, deg_pct_e2 0.808, cc_edges 0.982, deg_n_celltypes_sig 0.737).
- Held-out test, genes **not** in the label set only (30 published): AUROC 0.560.

| gene | published_tier | in_label_set | celltype_id | IDH_status | best_score | gene_rank | gene_percentile |
|---|---|---|---|---|---|---|---|
| HAVCR2 | 1 | 1 | cycling_tumor | IDH_Mutant | 0.855 | 17 | 0.998 |
| CD74 | 1 | 0 | AC_like_tumor | IDH_WT | 0.74 | 56 | 0.994 |
| TYROBP | 2 | 0 | AC_like_tumor | IDH_Mutant | 0.623 | 129 | 0.987 |
| HLA-DRA | 2 | 1 | AC_like_tumor | IDH_Mutant | 0.604 | 152 | 0.984 |
| SPP1 | 1 | 0 | AC_like_tumor | IDH_WT | 0.521 | 257 | 0.973 |
| CD4 | 2 | 0 | cycling_tumor | IDH_Mutant | 0.499 | 295 | 0.969 |
| CXCR4 | 1 | 0 | cycling_tumor | IDH_WT | 0.493 | 303 | 0.968 |
| TREM2 | 2 | 0 | cycling_tumor | IDH_Mutant | 0.463 | 354 | 0.963 |
| LGALS9 | 2 | 0 | cycling_tumor | IDH_Mutant | 0.452 | 384 | 0.96 |
| TGFB1 | 2 | 0 | cycling_tumor | IDH_Mutant | 0.42 | 445 | 0.954 |
| LILRB4 | 2 | 0 | cycling_tumor | IDH_Mutant | 0.298 | 861 | 0.91 |
| KLRB1 | 1 | 0 | T cell | IDH_Mutant | 0.243 | 1231 | 0.872 |
| CLEC2D | 1 | 0 | T cell | IDH_Mutant | 0.233 | 1308 | 0.864 |
| APOE | 2 | 0 | TAM1/TAM2/microglia | IDH_WT | 0.136 | 2426 | 0.747 |
| ENTPD1 | 1 | 0 | TAM1/TAM2/microglia | IDH_WT | 0.112 | 2893 | 0.698 |
| PTN | 1 | 0 | T cell | IDH_WT | 0.0781 | 3752 | 0.608 |
| NLGN1 | 2 | 0 | AC_like_tumor | IDH_Mutant | 0.0558 | 4510 | 0.529 |
| CD44 | 1 | 0 | T cell | IDH_WT | 0.042 | 5104 | 0.467 |
| NCAM1 | 2 | 0 | OPC_GABA_AC_like_tumor | IDH_Mutant | 0.038 | 5304 | 0.446 |
| CNTN1 | 2 | 0 | OPC_GABA_like_tumor | IDH_Mutant | 0.0252 | 6073 | 0.366 |
| PDGFRA | 1 | 0 | OPC_GABA_AC_like_tumor | IDH_Mutant | 0.0241 | 6142 | 0.359 |
| GRIA2 | 1 | 0 | OPC_GABA_AC_like_tumor | IDH_Mutant | 0.0237 | 6170 | 0.356 |
| NLGN3 | 1 | 0 | OPC_GABA_like_tumor | IDH_Mutant | 0.0203 | 6448 | 0.327 |
| PTPRZ1 | 1 | 0 | OPC_GABA_AC_like_tumor | IDH_Mutant | 0.0197 | 6511 | 0.32 |
| NOTCH1 | 2 | 0 | OPC_GABA_like_tumor | IDH_WT | 0.0167 | 6779 | 0.292 |
| CD99 | 2 | 0 | OPC_GABA_like_tumor | IDH_Mutant | 0.016 | 6858 | 0.284 |
| DLL3 | 1 | 0 | OPC_GABA_AC_like_tumor | IDH_Mutant | 0.0101 | 7498 | 0.217 |
| MIF | 1 | 0 | TAM1/TAM2/microglia | IDH_Mutant | 0.0074 | 7894 | 0.176 |
| GRIA4 | 2 | 0 | OPC_GABA_like_tumor | IDH_WT | 0.00245 | 8832 | 0.0782 |
| NRXN1 | 2 | 0 | AC_like_tumor | IDH_Mutant | 0.0011 | 9215 | 0.0382 |
| GRIA3 | 2 | 0 | AC_like_tumor | IDH_Mutant | 0.000972 | 9249 | 0.0347 |
| EGFR | 1 | 0 | OPC_GABA_AC_like_tumor | IDH_WT | 0.000255 | 9507 | 0.00772 |

**Top 25 novel genes** (in neither gene set nor the published 33, housekeeping genes removed; best context per gene)

| gene | celltype_id | IDH_status | p_immune_synapse | rule_pass | deg_log2fc | deg_neglog10_padj | cc_edges | ctx_min_n | why_immune_synapse |
|---|---|---|---|---|---|---|---|---|---|
| GRAP2 | T cell | IDH_Mutant | 0.939 | 0 | -2.28 | 2.09 | 0 | 334 | emb_03=-2.71 (+1.54); knn_frac_immune=0.25 (+1.14); emb_10=-1.5 (+1.00) |
| CXCL16 | TAM1/TAM2/microglia | IDH_WT | 0.939 | 0 | 0.288 | 22.9 | 0 | 2745 | emb_03=-0.501 (+1.57); emb_10=-1.34 (+0.90); knn_frac_immune=0.05 (+0.82) |
| CD52 | T cell | IDH_WT | 0.936 | 0 | -0.498 | 10.5 | 0 | 1118 | emb_03=-2.48 (+1.79); knn_frac_immune=0.25 (+1.08); emb_10=-1.68 (+0.90) |
| LAPTM5 | cycling_tumor | IDH_WT | 0.907 | 0 | -6.24 | 38.8 | 0 | 663 | emb_03=-0.89 (+1.93); emb_10=-0.801 (+0.93); knn_frac_immune=0.1 (+0.86) |
| NCF1 | TAM1/TAM2/microglia | IDH_Mutant | 0.901 | 0 | -0.659 | 15.7 | 0 | 2106 | emb_03=-1.35 (+1.56); knn_frac_immune=0.1 (+0.99); emb_10=-1.11 (+0.95) |
| LCP1 | cycling_tumor | IDH_Mutant | 0.898 | 0 | -7.88 | 5.61 | 0 | 136 | emb_03=-1.33 (+1.99); knn_frac_immune=0.1 (+1.26); emb_10=-0.866 (+0.98) |
| LCP2 | TAM1/TAM2/microglia | IDH_WT | 0.883 | 0 | 0.469 | 31.1 | 0 | 2745 | emb_03=-1.34 (+1.34); emb_10=-1.62 (+0.92); knn_frac_immune=0.1 (+0.64) |
| RAC2 | cycling_tumor | IDH_Mutant | 0.882 | 0 | -7.2 | 1.46 | 0 | 136 | emb_03=-1.93 (+1.59); knn_frac_immune=0.15 (+1.08); emb_10=-2.03 (+0.93) |
| CD48 | T cell | IDH_WT | 0.876 | 0 | -0.543 | 3.95 | 0 | 1118 | emb_03=-3.33 (+2.16); knn_frac_immune=0.2 (+1.01); emb_10=-1.14 (+0.95) |
| PTGDR | T cell | IDH_Mutant | 0.875 | 0 | 1.38 | 6.08 | 0 | 334 | emb_03=-2.1 (+1.08); knn_frac_immune=0.1 (+1.07); emb_10=-0.798 (+1.00) |
| RNASET2 | cycling_tumor | IDH_Mutant | 0.867 | 0 | -4.69 | 19 | 0 | 136 | emb_03=-0.855 (+1.95); emb_10=-1.27 (+0.94); knn_frac_immune=0.05 (+0.85) |
| TRADD | T cell | IDH_WT | 0.865 | 0 | -0.84 | 2.06 | 0 | 1118 | emb_03=-1.72 (+1.42); knn_frac_immune=0.05 (+0.91); emb_10=0.193 (+0.71) |
| ST3GAL1 | TAM1/TAM2/microglia | IDH_Mutant | 0.864 | 0 | 1.27 | 28.7 | 0 | 2106 | emb_03=-1.73 (+1.24); emb_10=-0.741 (+0.93); knn_frac_immune=0.05 (+0.60) |
| HLA-DQB1 | cycling_tumor | IDH_WT | 0.844 | 0 | -6.82 | 31.4 | 0 | 663 | emb_03=-2.52 (+2.25); knn_frac_immune=0.15 (+1.22); emb_10=-1.11 (+1.03) |
| TNFAIP3 | T cell | IDH_Mutant | 0.837 | 0 | -1.27 | 15.2 | 0 | 334 | emb_03=-3.96 (+1.47); emb_10=-4.18 (+1.02); knn_frac_immune=0.05 (+0.96) |
| ATP8B4 | TAM1/TAM2/microglia | IDH_WT | 0.835 | 0 | 0.943 | 30.9 | 0 | 2745 | emb_03=-0.527 (+1.92); knn_frac_immune=0.05 (+1.04); emb_10=-2.23 (+1.00) |
| PILRA | TAM1/TAM2/microglia | IDH_Mutant | 0.828 | 1 | -0.465 | 7.19 | 7 | 2106 | emb_03=-1.28 (+1.57); emb_10=-1.84 (+0.92); knn_frac_immune=0.05 (+0.73) |
| EMB | T cell | IDH_WT | 0.827 | 0 | -0.743 | 9.21 | 0 | 1118 | emb_03=-1.95 (+2.01); knn_frac_immune=0.1 (+1.30); emb_10=-1.07 (+1.03) |
| CAPZA1 | cycling_tumor | IDH_WT | 0.827 | 0 | -1.04 | 3.25 | 0 | 663 | emb_03=-0.36 (+1.49); emb_10=0.0488 (+0.68); emb_02=0.762 (+0.64) |
| CSK | TAM1/TAM2/microglia | IDH_Mutant | 0.825 | 0 | 0.375 | 4.13 | 0 | 2106 | emb_03=-1 (+1.47); emb_10=0.401 (+0.70); emb_02=1.13 (+0.62) |
| TNFRSF1B | TAM1/TAM2/microglia | IDH_Mutant | 0.824 | 1 | 0.456 | 14.1 | 2 | 2106 | emb_03=-1.11 (+1.36); knn_frac_immune=0.1 (+1.09); emb_10=-1.19 (+0.90) |
| ADPGK | TAM1/TAM2/microglia | IDH_Mutant | 0.823 | 0 | 0.559 | 13.7 | 0 | 2106 | emb_03=-0.382 (+2.14); emb_10=-0.884 (+0.95); emb_31=0.889 (+0.49) |
| CXCR6 | T cell | IDH_Mutant | 0.819 | 0 | -1.87 | 1.65 | 0 | 334 | emb_03=-2.9 (+1.53); knn_frac_immune=0.15 (+1.10); emb_10=0.499 (+0.64) |
| PRDM1 | T cell | IDH_Mutant | 0.818 | 0 | -1.09 | 1.61 | 0 | 334 | knn_frac_immune=0.2 (+1.26); emb_03=-2.72 (+1.15); emb_10=-1.54 (+0.96) |
| PTPRC | T cell | IDH_Mutant | 0.804 | 1 | -0.631 | 5.43 | 4 | 334 | emb_03=-2.68 (+1.93); knn_frac_immune=0.05 (+1.16); emb_10=-1.11 (+0.97) |

**Novel genes that also pass the 3-table rule** (CellChat edge + DEG + n_cells)

| gene | celltype_id | IDH_status | p_immune_synapse | rule_pass | deg_log2fc | deg_neglog10_padj | cc_edges | ctx_min_n | why_immune_synapse |
|---|---|---|---|---|---|---|---|---|---|
| PILRA | TAM1/TAM2/microglia | IDH_Mutant | 0.828 | 1 | -0.465 | 7.19 | 7 | 2106 | emb_03=-1.28 (+1.57); emb_10=-1.84 (+0.92); knn_frac_immune=0.05 (+0.73) |
| TNFRSF1B | TAM1/TAM2/microglia | IDH_Mutant | 0.824 | 1 | 0.456 | 14.1 | 2 | 2106 | emb_03=-1.11 (+1.36); knn_frac_immune=0.1 (+1.09); emb_10=-1.19 (+0.90) |
| PTPRC | T cell | IDH_Mutant | 0.804 | 1 | -0.631 | 5.43 | 4 | 334 | emb_03=-2.68 (+1.93); knn_frac_immune=0.05 (+1.16); emb_10=-1.11 (+0.97) |
| CD69 | T cell | IDH_Mutant | 0.749 | 1 | -0.527 | 1.47 | 4 | 334 | knn_frac_immune=0.1 (+1.03); emb_10=-3.07 (+0.96); emb_03=-3.84 (+0.88) |
| PSAP | cycling_tumor | IDH_Mutant | 0.707 | 1 | -2.44 | 2.31 | 2 | 136 | emb_03=-1.09 (+1.53); knn_frac_immune=0.05 (+0.87); emb_10=0.0446 (+0.72) |
| LAIR1 | TAM1/TAM2/microglia | IDH_Mutant | 0.705 | 1 | -0.275 | 3.68 | 4 | 2106 | emb_03=-1.49 (+2.20); knn_frac_immune=0.05 (+1.20); emb_10=-2.33 (+1.00) |
| CLEC2B | TAM1/TAM2/microglia | IDH_Mutant | 0.663 | 1 | -0.626 | 48.5 | 2 | 2106 | emb_03=-1.7 (+1.97); knn_frac_immune=0.1 (+0.96); emb_10=0.565 (+0.61) |
| HLA-DMA | cycling_tumor | IDH_Mutant | 0.656 | 1 | -5.39 | 11.3 | 2 | 136 | emb_03=-1.26 (+2.11); knn_frac_immune=0.1 (+1.20); emb_10=-0.69 (+0.88) |
| ITGAX | TAM1/TAM2/microglia | IDH_WT | 0.459 | 1 | 0.416 | 16 | 4 | 2745 | emb_03=-0.982 (+1.47); emb_10=-2.47 (+0.93); emb_02=0.374 (+0.57) |
| HLA-DPB1 | TAM1/TAM2/microglia | IDH_Mutant | 0.428 | 1 | -1.89 | 300 | 4 | 2106 | emb_03=-1.53 (+2.16); knn_frac_immune=0.15 (+1.47); emb_10=-0.442 (+0.58) |
| GLS | cycling_tumor | IDH_WT | 0.403 | 1 | 0.992 | 90.2 | 26 | 663 | emb_03=-0.726 (+1.86); emb_10=-1.35 (+1.01); emb_01=0.816 (+0.48) |
| PDGFB | TAM1/TAM2/microglia | IDH_Mutant | 0.326 | 1 | 0.638 | 16.7 | 3 | 2106 | emb_02=0.587 (+0.99); emb_10=-1.94 (+0.51); emb_22=0.481 (+0.43) |
| HLA-E | T cell | IDH_WT | 0.195 | 1 | -0.504 | 11.8 | 2 | 1118 | emb_03=-1.44 (+2.03); emb_10=-0.578 (+0.62); emb_27=1.38 (+0.37) |
| TNF | TAM1/TAM2/microglia | IDH_WT | 0.176 | 1 | 1.38 | 103 | 3 | 2745 | emb_03=-1.85 (+1.55); emb_10=-4.63 (+0.99); emb_27=2.56 (+0.38) |
| ITGAV | OPC_GABA_like_tumor | IDH_Mutant | 0.168 | 1 | -0.936 | 54.2 | 14 | 1658 | emb_02=0.418 (+1.00); emb_31=1.19 (+0.48); emb_20=1.1 (+0.39) |

**Feature importance (gain, top 10)**

| feature | gain | splits | mean_abs_shap |
|---|---|---|---|
| emb_03 | 2.82e+05 | 28.2 | 1.06 |
| emb_10 | 1.25e+05 | 16.8 | 0.719 |
| emb_07 | 7.67e+04 | 12.2 | 0.403 |
| knn_frac_immune | 7.65e+04 | 20.6 | 0.124 |
| emb_02 | 6.94e+04 | 23 | 0.716 |
| emb_27 | 4.9e+04 | 16.8 | 0.143 |
| emb_22 | 4.2e+04 | 19.8 | 0.492 |
| emb_17 | 2.83e+04 | 18 | 0.259 |
| emb_31 | 2.58e+04 | 26.6 | 0.294 |
| emb_01 | 2.41e+04 | 21 | 0.364 |

## Caveats

- GO:0001772 includes all taxa (matching the team's 5,614-annotation download); most are IEA (electronic) annotations mapped by upper-casing symbols. `--immune-human-only` restricts to genes with a human annotation.
- SynGO annotations are mostly from rodent brain synapses; membership says a gene is synaptic in neurons, not in glioma.
- Gene-set membership is a weak label: unlabeled genes are unknown, not negative.
- CellChat features are zero for most genes in the DEG universe (CellChat covers ~114 genes); DEG and n_cells carry most signal.
- SynGO genes are enriched for broadly expressed genes, so the tumor-synaptic model partly learns "highly expressed". Mitochondrial / ribosomal / histone genes are flagged `housekeeping` and left out of the novel tables.
- Immune genes (HLA class II, ITGB2, C3, CCL2) show up as strong Ephys_1-high DEGs inside *tumor* clusters, which suggests myeloid contamination / doublets in Ephys_1 tumor clusters. Immune-synapse nominees in tumor cell types should be checked for this before follow-up.
- High score = "expression pattern resembles known synaptic / immune-synapse genes", not validation.
