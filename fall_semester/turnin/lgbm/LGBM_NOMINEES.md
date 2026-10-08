# LightGBM gene nominees (gene x cell type x IDH)

- Universe: `cellchat`. 1368 rows, 114 genes, 39 features.
- Labels: literature tiers [1, 2] from `knowledge/literature_positives.yaml`: 200 positive rows (33 genes). Everything else is unlabeled, not negative.
- Scores are out-of-fold: StratifiedGroupKFold(n_splits=5) grouped by gene, averaged over seeds [0, 1, 2, 3, 4].
- LLM run features: none (table-only model).

## Does the model beat single-table baselines?

Row-level prevalence = 0.146 (AUPRC of a random ranking).

| scorer | auroc | auprc | precision_at_10 | precision_at_25 | precision_at_50 |
|---|---|---|---|---|---|
| lightgbm_oof | 0.679 | 0.381 | 0.9 | 0.72 | 0.7 |
| rule_pass | 0.631 | 0.226 | 0 | 0.2 | 0.34 |
| deg_abs_log2fc | 0.612 | 0.188 | 0.2 | 0.12 | 0.1 |
| cc_edges | 0.656 | 0.238 | 0.4 | 0.24 | 0.32 |
| cc_max_prob | 0.649 | 0.243 | 0.4 | 0.56 | 0.32 |

Gene level (best context per gene):

| scorer | auroc | auprc | precision_at_10 | precision_at_25 | precision_at_50 |
|---|---|---|---|---|---|
| lightgbm_oof | 0.679 | 0.541 | 0.7 | 0.56 | 0.46 |
| deg_abs_log2fc | 0.691 | 0.457 | 0.5 | 0.44 | 0.46 |
| cc_edges | 0.69 | 0.562 | 0.7 | 0.52 | 0.44 |

## Top 30 contexts

| rank | gene | celltype_id | IDH_status | score | status | rule_pass | deg_log2fc | deg_neglog10_padj | cc_edges | cc_ephys_bias | ctx_min_n | top_reasons |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | GRIA4 | cycling_tumor | IDH_Mutant | 0.948 | literature tier 2 (label) | 1 | 3.82 | 38.1 | 3 | 1 | 136 | cc_edges_total=15 (+0.96); cc_n_partners=2 (+0.84); deg_pooled_log2fc=1.86 (+0.50) |
| 2 | GRIA4 | AC_like_tumor | IDH_Mutant | 0.925 | literature tier 2 (label) | 1 | 1.73 | 22.5 | 3 | 1 | 283 | cc_edges_total=15 (+0.95); cc_n_partners=2 (+0.84); deg_pooled_log2fc=1.49 (+0.46) |
| 3 | GRIA4 | OPC_GABA_like_tumor | IDH_Mutant | 0.921 | literature tier 2 (label) | 1 | 1.4 | 204 | 6 | 0 | 1658 | cc_edges_total=15 (+0.96); cc_n_partners=2 (+0.89); deg_pooled_log2fc=1.53 (+0.43) |
| 4 | GRIA4 | OPC_GABA_AC_like_tumor | IDH_Mutant | 0.919 | literature tier 2 (label) | 1 | 1.21 | 4.89 | 3 | 1 | 352 | cc_edges_total=15 (+0.96); cc_n_partners=2 (+0.82); deg_pooled_log2fc=1.37 (+0.45) |
| 5 | GRIA4 | cycling_tumor | IDH_WT | 0.891 | literature tier 2 (label) | 1 | 1.3 | 133 | 3 | 1 | 663 | cc_edges_total=15 (+0.97); cc_n_partners=2 (+0.85); deg_pooled_log2fc=1.86 (+0.44) |
| 6 | NLGN1 | cycling_tumor | IDH_WT | 0.877 | literature tier 2 (label) | 1 | 1.87 | 214 | 6 | 1 | 663 | cc_edges_total=18 (+1.00); deg_pooled_log2fc=2.04 (+0.46); cc_n_partners=4 (+0.36) |
| 7 | PTPRZ1 | cycling_tumor | IDH_WT | 0.876 | literature tier 1 (label) | 1 | 2.07 | 201 | 17 | 0.176 | 663 | cc_edges_total=77 (+0.78); deg_pooled_log2fc=2.05 (+0.50); deg_n_celltypes_sig=5 (+0.41) |
| 8 | PTPRZ1 | cycling_tumor | IDH_Mutant | 0.873 | literature tier 1 (label) | 1 | 4 | 54.1 | 17 | 0.176 | 136 | cc_edges_total=77 (+0.78); deg_pooled_log2fc=2.05 (+0.49); deg_n_celltypes_sig=4 (+0.42) |
| 9 | NLGN1 | cycling_tumor | IDH_Mutant | 0.862 | literature tier 2 (label) | 1 | 4.22 | 20.2 | 6 | 1 | 136 | cc_edges_total=18 (+1.00); deg_pooled_log2fc=2.04 (+0.46); cc_n_partners=4 (+0.34) |
| 10 | NCL | cycling_tumor | IDH_WT | 0.851 | novel | 1 | 0.951 | 81.6 | 16 | 0.125 | 663 | cc_edges_total=70 (+0.66); cc_n_cells_e1=6.41e+03 (+0.22); deg_pct_diff=0.398 (+0.19) |
| 11 | PDGFRA | cycling_tumor | IDH_WT | 0.849 | literature tier 1 (label) | 1 | 4.46 | 300 | 2 | 1 | 663 | cc_n_partners=2 (+0.84); deg_pooled_log2fc=3.95 (+0.53); deg_n_celltypes_sig=4 (+0.35) |
| 12 | CNTN1 | cycling_tumor | IDH_WT | 0.845 | literature tier 2 (label) | 1 | 4.19 | 300 | 8 | 1 | 663 | cc_edges_total=16 (+0.99); deg_pooled_log2fc=3.92 (+0.42); cc_n_partners=4 (+0.39) |
| 13 | NLGN1 | AC_like_tumor | IDH_Mutant | 0.845 | literature tier 2 (label) | 1 | 0.76 | 2.42 | 6 | 1 | 283 | cc_edges_total=18 (+1.00); deg_pooled_log2fc=1.94 (+0.47); cc_n_partners=4 (+0.32) |
| 14 | GRIK2 | cycling_tumor | IDH_WT | 0.844 | novel | 1 | 4.53 | 300 | 3 | 1 | 663 | cc_n_partners=2 (+0.85); deg_pooled_log2fc=4.63 (+0.38); cc_edges_e2=3 (+0.26) |
| 15 | NCL | cycling_tumor | IDH_Mutant | 0.843 | novel | 1 | 0.515 | 2.83 | 16 | 0.125 | 136 | cc_edges_total=70 (+0.65); cc_n_cells_e1=6.41e+03 (+0.22); deg_pct_diff=0.401 (+0.20) |
| 16 | BSG | cycling_tumor | IDH_WT | 0.839 | novel | 1 | 1.63 | 201 | 13 | 1 | 663 | cc_edges_total=56 (+0.73); deg_pooled_log2fc=1.47 (+0.40); cc_n_partners=7 (+0.23) |
| 17 | NRXN1 | cycling_tumor | IDH_Mutant | 0.838 | literature tier 2 (label) | 1 | 3.71 | 44.2 | 10 | 1 | 136 | cc_edges_total=50 (+1.21); deg_pooled_log2fc=2.04 (+0.43); cc_n_partners=3 (+0.42) |
| 18 | GRIK2 | OPC_GABA_like_tumor | IDH_WT | 0.835 | novel | 1 | 2.36 | 255 | 3 | 1 | 1935 | cc_n_partners=2 (+0.88); deg_pooled_log2fc=1.95 (+0.38); cc_edges_e2=3 (+0.27) |
| 19 | PDGFRA | OPC_GABA_like_tumor | IDH_WT | 0.834 | literature tier 1 (label) | 1 | 3.47 | 300 | 2 | 1 | 1935 | cc_n_partners=2 (+0.87); deg_pooled_log2fc=2.68 (+0.53); deg_n_celltypes_sig=4 (+0.35) |
| 20 | NCAM1 | OPC_GABA_like_tumor | IDH_WT | 0.83 | literature tier 2 (label) | 1 | 1.91 | 300 | 22 | 0 | 1935 | cc_edges_total=55 (+0.85); deg_pooled_log2fc=1.82 (+0.45); deg_n_celltypes_sig=4 (+0.42) |
| 21 | GRIK2 | cycling_tumor | IDH_Mutant | 0.83 | novel | 1 | 4.4 | 18.6 | 3 | 1 | 136 | cc_n_partners=2 (+0.86); deg_pooled_log2fc=4.63 (+0.38); cc_edges_e2=3 (+0.26) |
| 22 | NCAM1 | cycling_tumor | IDH_Mutant | 0.829 | literature tier 2 (label) | 1 | 4.83 | 49.5 | 11 | 1 | 136 | cc_edges_total=55 (+0.86); deg_pooled_log2fc=2.65 (+0.50); deg_n_celltypes_sig=5 (+0.41) |
| 23 | NCAM1 | cycling_tumor | IDH_WT | 0.827 | literature tier 2 (label) | 1 | 2.45 | 300 | 11 | 1 | 663 | cc_edges_total=55 (+0.86); deg_pooled_log2fc=2.65 (+0.50); deg_n_celltypes_sig=4 (+0.42) |
| 24 | PDGFRA | cycling_tumor | IDH_Mutant | 0.827 | literature tier 1 (label) | 1 | 3.76 | 30.9 | 2 | 1 | 136 | cc_n_partners=2 (+0.85); deg_pooled_log2fc=3.95 (+0.54); deg_n_celltypes_sig=4 (+0.35) |
| 25 | MPZL1 | cycling_tumor | IDH_WT | 0.824 | novel | 1 | 2.13 | 192 | 4 | 1 | 663 | cc_n_partners=2 (+0.85); deg_pooled_log2fc=1.86 (+0.42); deg_n_celltypes_sig=4 (+0.35) |
| 26 | BSG | OPC_GABA_AC_like_tumor | IDH_WT | 0.821 | novel | 1 | 1.32 | 14.5 | 11 | 1 | 171 | cc_edges_total=56 (+0.80); cc_n_cells_e1=6.2e+03 (+0.22); cc_n_partners=7 (+0.21) |
| 27 | GRIA4 | OPC_GABA_like_tumor | IDH_WT | 0.817 | literature tier 2 (label) | 1 | -0.562 | 10.6 | 6 | 0 | 1935 | cc_edges_total=15 (+1.01); cc_n_partners=2 (+0.88); deg_pooled_log2fc=1.53 (+0.38) |
| 28 | PDGFRA | OPC_GABA_like_tumor | IDH_Mutant | 0.813 | literature tier 1 (label) | 1 | 2.06 | 208 | 2 | 1 | 1658 | cc_n_partners=2 (+0.88); deg_pooled_log2fc=2.68 (+0.54); deg_n_celltypes_sig=4 (+0.35) |
| 29 | PDGFRA | AC_like_tumor | IDH_Mutant | 0.811 | literature tier 1 (label) | 1 | 1.43 | 13.2 | 2 | 1 | 283 | cc_n_partners=2 (+0.82); deg_pooled_log2fc=4.14 (+0.54); deg_n_celltypes_sig=4 (+0.35) |
| 30 | NCAM1 | OPC_GABA_AC_like_tumor | IDH_WT | 0.81 | literature tier 2 (label) | 1 | 1.54 | 4.56 | 11 | 1 | 171 | cc_edges_total=55 (+0.86); deg_pooled_log2fc=1.98 (+0.51); deg_n_celltypes_sig=4 (+0.38) |

## Top novel nominees that pass the 3-table rule

Not in the literature list, but CellChat edge + DEG in this cell type/IDH + adequate n_cells.

| rank | gene | celltype_id | IDH_status | score | status | rule_pass | deg_log2fc | deg_neglog10_padj | cc_edges | cc_ephys_bias | ctx_min_n | top_reasons |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | NCL | cycling_tumor | IDH_WT | 0.851 | novel | 1 | 0.951 | 81.6 | 16 | 0.125 | 663 | cc_edges_total=70 (+0.66); cc_n_cells_e1=6.41e+03 (+0.22); deg_pct_diff=0.398 (+0.19) |
| 14 | GRIK2 | cycling_tumor | IDH_WT | 0.844 | novel | 1 | 4.53 | 300 | 3 | 1 | 663 | cc_n_partners=2 (+0.85); deg_pooled_log2fc=4.63 (+0.38); cc_edges_e2=3 (+0.26) |
| 15 | NCL | cycling_tumor | IDH_Mutant | 0.843 | novel | 1 | 0.515 | 2.83 | 16 | 0.125 | 136 | cc_edges_total=70 (+0.65); cc_n_cells_e1=6.41e+03 (+0.22); deg_pct_diff=0.401 (+0.20) |
| 16 | BSG | cycling_tumor | IDH_WT | 0.839 | novel | 1 | 1.63 | 201 | 13 | 1 | 663 | cc_edges_total=56 (+0.73); deg_pooled_log2fc=1.47 (+0.40); cc_n_partners=7 (+0.23) |
| 18 | GRIK2 | OPC_GABA_like_tumor | IDH_WT | 0.835 | novel | 1 | 2.36 | 255 | 3 | 1 | 1935 | cc_n_partners=2 (+0.88); deg_pooled_log2fc=1.95 (+0.38); cc_edges_e2=3 (+0.27) |
| 21 | GRIK2 | cycling_tumor | IDH_Mutant | 0.83 | novel | 1 | 4.4 | 18.6 | 3 | 1 | 136 | cc_n_partners=2 (+0.86); deg_pooled_log2fc=4.63 (+0.38); cc_edges_e2=3 (+0.26) |
| 25 | MPZL1 | cycling_tumor | IDH_WT | 0.824 | novel | 1 | 2.13 | 192 | 4 | 1 | 663 | cc_n_partners=2 (+0.85); deg_pooled_log2fc=1.86 (+0.42); deg_n_celltypes_sig=4 (+0.35) |
| 26 | BSG | OPC_GABA_AC_like_tumor | IDH_WT | 0.821 | novel | 1 | 1.32 | 14.5 | 11 | 1 | 171 | cc_edges_total=56 (+0.80); cc_n_cells_e1=6.2e+03 (+0.22); cc_n_partners=7 (+0.21) |
| 33 | BSG | OPC_GABA_like_tumor | IDH_WT | 0.8 | novel | 1 | 1.01 | 185 | 26 | 0 | 1935 | cc_edges_total=56 (+0.74); cc_n_cells_e1=6.74e+03 (+0.22); cc_n_partners=7 (+0.21) |
| 35 | NCL | OPC_GABA_like_tumor | IDH_WT | 0.795 | novel | 1 | 0.878 | 169 | 18 | 0 | 1935 | cc_edges_total=70 (+0.74); cc_n_cells_e1=6.74e+03 (+0.21); cc_edges_e2=9 (+0.20) |
| 37 | GRIK2 | OPC_GABA_like_tumor | IDH_Mutant | 0.791 | novel | 1 | 0.708 | 46.9 | 3 | 1 | 1658 | cc_n_partners=2 (+0.89); deg_pooled_log2fc=1.95 (+0.39); cc_edges_e2=3 (+0.27) |
| 42 | CDH2 | cycling_tumor | IDH_Mutant | 0.784 | novel | 1 | 2.7 | 16 | 8 | 1 | 136 | cc_n_partners=3 (+0.33); cc_edges_e2=8 (+0.23); cc_n_cells_e1=6.41e+03 (+0.21) |
| 43 | BSG | OPC_GABA_AC_like_tumor | IDH_Mutant | 0.782 | novel | 1 | 0.569 | 1.36 | 11 | 1 | 352 | cc_edges_total=56 (+0.78); cc_n_cells_e1=6.2e+03 (+0.20); deg_pct_diff=0.136 (+0.19) |
| 47 | NCL | OPC_GABA_like_tumor | IDH_Mutant | 0.777 | novel | 1 | 0.585 | 75.5 | 18 | 0 | 1658 | cc_edges_total=70 (+0.74); cc_n_cells_e1=6.74e+03 (+0.21); cc_edges_e2=9 (+0.20) |
| 53 | CDH2 | cycling_tumor | IDH_WT | 0.771 | novel | 1 | 0.828 | 109 | 8 | 1 | 663 | cc_n_partners=3 (+0.32); cc_edges_e2=8 (+0.23); cc_n_cells_e1=6.41e+03 (+0.21) |
| 54 | APP | cycling_tumor | IDH_Mutant | 0.771 | novel | 1 | 3.05 | 30.2 | 13 | 1 | 136 | cc_edges_total=78 (+0.79); cc_n_cells_e1=6.41e+03 (+0.22); deg_pct_diff=0.588 (+0.19) |
| 63 | NCL | AC_like_tumor | IDH_Mutant | 0.753 | novel | 1 | 0.753 | 13.2 | 10 | 0.8 | 283 | cc_edges_total=70 (+0.73); cc_n_cells_e1=7.25e+04 (+0.22); deg_pct_diff=0.245 (+0.19) |
| 73 | APP | OPC_GABA_like_tumor | IDH_Mutant | 0.733 | novel | 1 | -0.317 | 7.99 | 26 | 0 | 1658 | cc_edges_total=78 (+0.78); cc_n_cells_e1=6.74e+03 (+0.24); cc_n_partners=6 (+0.19) |
| 74 | APP | cycling_tumor | IDH_WT | 0.729 | novel | 1 | 0.564 | 78.4 | 13 | 1 | 663 | cc_edges_total=78 (+0.77); cc_n_cells_e1=6.41e+03 (+0.20); deg_pct_diff=0.454 (+0.18) |
| 79 | CADM1 | OPC_GABA_like_tumor | IDH_Mutant | 0.721 | novel | 1 | 0.317 | 6.4 | 16 | 0 | 1658 | cc_n_partners=3 (+0.39); cc_n_cells_e1=6.74e+03 (+0.24); cc_edges_e2=8 (+0.23) |

## Top genes (best context per gene)

| gene_rank | gene | celltype_id | IDH_status | score | status | rule_pass | top_reasons |
|---|---|---|---|---|---|---|---|
| 1 | GRIA4 | cycling_tumor | IDH_Mutant | 0.948 | literature tier 2 (label) | 1 | cc_edges_total=15 (+0.96); cc_n_partners=2 (+0.84); deg_pooled_log2fc=1.86 (+0.50) |
| 2 | NLGN1 | cycling_tumor | IDH_WT | 0.877 | literature tier 2 (label) | 1 | cc_edges_total=18 (+1.00); deg_pooled_log2fc=2.04 (+0.46); cc_n_partners=4 (+0.36) |
| 3 | PTPRZ1 | cycling_tumor | IDH_WT | 0.876 | literature tier 1 (label) | 1 | cc_edges_total=77 (+0.78); deg_pooled_log2fc=2.05 (+0.50); deg_n_celltypes_sig=5 (+0.41) |
| 4 | NCL | cycling_tumor | IDH_WT | 0.851 | novel | 1 | cc_edges_total=70 (+0.66); cc_n_cells_e1=6.41e+03 (+0.22); deg_pct_diff=0.398 (+0.19) |
| 5 | PDGFRA | cycling_tumor | IDH_WT | 0.849 | literature tier 1 (label) | 1 | cc_n_partners=2 (+0.84); deg_pooled_log2fc=3.95 (+0.53); deg_n_celltypes_sig=4 (+0.35) |
| 6 | CNTN1 | cycling_tumor | IDH_WT | 0.845 | literature tier 2 (label) | 1 | cc_edges_total=16 (+0.99); deg_pooled_log2fc=3.92 (+0.42); cc_n_partners=4 (+0.39) |
| 7 | GRIK2 | cycling_tumor | IDH_WT | 0.844 | novel | 1 | cc_n_partners=2 (+0.85); deg_pooled_log2fc=4.63 (+0.38); cc_edges_e2=3 (+0.26) |
| 8 | BSG | cycling_tumor | IDH_WT | 0.839 | novel | 1 | cc_edges_total=56 (+0.73); deg_pooled_log2fc=1.47 (+0.40); cc_n_partners=7 (+0.23) |
| 9 | NRXN1 | cycling_tumor | IDH_Mutant | 0.838 | literature tier 2 (label) | 1 | cc_edges_total=50 (+1.21); deg_pooled_log2fc=2.04 (+0.43); cc_n_partners=3 (+0.42) |
| 10 | NCAM1 | OPC_GABA_like_tumor | IDH_WT | 0.83 | literature tier 2 (label) | 1 | cc_edges_total=55 (+0.85); deg_pooled_log2fc=1.82 (+0.45); deg_n_celltypes_sig=4 (+0.42) |
| 11 | MPZL1 | cycling_tumor | IDH_WT | 0.824 | novel | 1 | cc_n_partners=2 (+0.85); deg_pooled_log2fc=1.86 (+0.42); deg_n_celltypes_sig=4 (+0.35) |
| 12 | CDH2 | cycling_tumor | IDH_Mutant | 0.784 | novel | 1 | cc_n_partners=3 (+0.33); cc_edges_e2=8 (+0.23); cc_n_cells_e1=6.41e+03 (+0.21) |
| 13 | NOTCH1 | cycling_tumor | IDH_WT | 0.778 | literature tier 2 (label) | 1 | cc_edges_total=16 (+1.00); deg_pooled_log2fc=2.05 (+0.37); cc_n_partners=4 (+0.35) |
| 14 | APP | AC_like_tumor | IDH_Mutant | 0.776 | novel | 0 | cc_edges_total=78 (+0.85); cc_n_cells_e1=7.25e+04 (+0.24); cc_n_partners=4 (+0.20) |
| 15 | CD74 | AC_like_tumor | IDH_Mutant | 0.762 | literature gene, other compartment | 0 | cc_edges_total=109 (+0.89); cc_n_cells_e1=7.25e+04 (+0.23); deg_n_celltypes_sig=4 (+0.23) |
| 16 | CD4 | OPC_GABA_AC_like_tumor | IDH_WT | 0.737 | literature gene, other compartment | 0 | cc_edges_total=88 (+1.14); cc_n_cells_e1=6.2e+03 (+0.20); cc_n_celltypes_total=1 (+0.10) |
| 17 | CADM1 | OPC_GABA_like_tumor | IDH_Mutant | 0.721 | novel | 1 | cc_n_partners=3 (+0.39); cc_n_cells_e1=6.74e+03 (+0.24); cc_edges_e2=8 (+0.23) |
| 18 | DLL1 | cycling_tumor | IDH_WT | 0.717 | novel | 1 | cc_n_partners=2 (+0.82); deg_pooled_log2fc=5.21 (+0.33); cc_edges_e2=2 (+0.26) |
| 19 | NRCAM | AC_like_tumor | IDH_Mutant | 0.714 | novel | 1 | cc_n_partners=2 (+0.81); deg_n_celltypes_sig=4 (+0.31); cc_edges_e2=2 (+0.28) |
| 20 | GRIA2 | cycling_tumor | IDH_Mutant | 0.711 | literature tier 1 (label) | 1 | cc_n_partners=2 (+0.86); deg_pooled_log2fc=2.61 (+0.54); deg_n_celltypes_sig=4 (+0.35) |
| 21 | TYROBP | TAM1/TAM2/microglia | IDH_Mutant | 0.703 | literature tier 2 (label) | 1 | cc_edges_total=57 (+0.86); deg_n_celltypes_sig=6 (+0.31); cc_n_partners=7 (+0.25) |
| 22 | SLC1A1 | cycling_tumor | IDH_WT | 0.697 | novel | 1 | cc_edges_total=13 (+0.53); deg_pooled_log2fc=2.26 (+0.37); cc_n_partners=4 (+0.33) |
| 23 | EGFR | cycling_tumor | IDH_Mutant | 0.695 | literature tier 1 (label) | 1 | cc_n_partners=2 (+0.87); cc_edges_total=14 (+0.56); cc_edges_e2=2 (+0.24) |
| 24 | TREM2 | AC_like_tumor | IDH_Mutant | 0.693 | literature gene, other compartment | 0 | cc_edges_total=57 (+1.02); cc_n_cells_e1=7.25e+04 (+0.22); cc_n_celltypes_total=2 (+0.09) |
| 25 | SLIT1 | OPC_GABA_like_tumor | IDH_Mutant | 0.684 | novel | 1 | cc_n_partners=2 (+0.83); deg_pooled_log2fc=2.79 (+0.28); cc_edges_e2=2 (+0.25) |
| 26 | NCAM2 | cycling_tumor | IDH_Mutant | 0.682 | novel | 1 | deg_pooled_log2fc=1.95 (+0.37); cc_n_partners=4 (+0.34); cc_edges_e2=5 (+0.22) |
| 27 | PTN | cycling_tumor | IDH_Mutant | 0.677 | literature gene, other compartment | 1 | cc_edges_total=137 (+0.67); cc_n_cells_e1=6.41e+03 (+0.20); deg_pct_diff=0.837 (+0.19) |
| 28 | LRRTM2 | cycling_tumor | IDH_Mutant | 0.67 | novel | 1 | cc_n_partners=4 (+0.33); cc_edges_e2=6 (+0.24); cc_n_cells_e1=6.41e+03 (+0.21) |
| 29 | SPP1 | TAM1/TAM2/microglia | IDH_Mutant | 0.664 | literature tier 1 (label) | 1 | cc_edges_total=55 (+0.81); deg_n_celltypes_sig=4 (+0.37); cc_n_partners=4 (+0.26) |
| 30 | SLC1A3 | TAM1/TAM2/microglia | IDH_Mutant | 0.649 | novel | 1 | cc_n_partners=4 (+0.34); deg_pooled_log2fc=1.4 (+0.31); cc_n_cells_e1=1.21e+04 (+0.25) |

## Feature importance (seed-averaged gain)

| feature | gain | splits | mean_abs_shap |
|---|---|---|---|
| cc_edges_total | 3.32e+03 | 122 | 0.478 |
| cc_n_partners | 1.62e+03 | 36 | 0.183 |
| cc_n_cells_e1 | 1.05e+03 | 28 | 0.36 |
| deg_n_celltypes_sig | 830 | 46.2 | 0.142 |
| deg_pooled_log2fc | 775 | 36.6 | 0.14 |
| cc_edges_e2 | 673 | 15.4 | 0.0953 |
| cc_n_celltypes_total | 621 | 31.6 | 0.109 |
| deg_pct_diff | 359 | 15.8 | 0.0719 |
| deg_log2fc | 202 | 8.2 | 0.0473 |
| cc_n_cells_e2 | 184 | 7.6 | 0.0646 |
| cc_max_prob | 166 | 11.6 | 0.0209 |
| cc_cross_compartment_edges | 124 | 6.8 | 0.0157 |
| deg_neglog10_padj | 106 | 8.8 | 0.0129 |
| cc_mean_prob | 100 | 7.8 | 0.021 |
| deg_pct_e1 | 93.6 | 7.6 | 0.0216 |

## Caveats

- The positive list is a small, hand-curated draft. Labeled genes are biased toward well-studied biology.
- Tier 2 genes are pathway-level picks that overlap the H1-H3 hypotheses. `--tiers 1` is the strict check; with tier 1 alone (16 genes) the model does not beat single-column baselines.
- CellChat is not IDH-stratified, so `cc_*` features repeat across IDH groups; only DEG and n features differ by IDH.
- `cc_n_cells_*` and `ctx_n_*` are constant per cell type, so they act as a cell-type proxy; most labels are tumor-side, which lifts tumor contexts (e.g. CD74 in AC_like_tumor ranks high despite being a myeloid label).
- Gene-level AUROC is on par with ranking genes by CellChat edge count; the model's added value is mainly picking the right cell type x IDH context for a gene.
- High scores mean "looks like known glioma communication genes in these tables", not validation.
- `top_reasons` are SHAP contributions from the full-data model; scores themselves are out-of-fold.
