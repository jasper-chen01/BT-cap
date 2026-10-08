# LightGBM gene nominees (gene x cell type x IDH)

- Universe: `cellchat`. 1368 rows, 114 genes, 39 features.
- Labels: literature tiers [1] from `knowledge/literature_positives.yaml`: 94 positive rows (16 genes). Everything else is unlabeled, not negative.
- Scores are out-of-fold: StratifiedGroupKFold(n_splits=5) grouped by gene, averaged over seeds [0, 1, 2, 3, 4].
- LLM run features: none (table-only model).

## Does the model beat single-table baselines?

Row-level prevalence = 0.069 (AUPRC of a random ranking).

| scorer | auroc | auprc | precision_at_10 | precision_at_25 | precision_at_50 |
|---|---|---|---|---|---|
| lightgbm_oof | 0.584 | 0.138 | 0.4 | 0.36 | 0.22 |
| rule_pass | 0.671 | 0.136 | 0 | 0.08 | 0.24 |
| deg_abs_log2fc | 0.658 | 0.112 | 0.1 | 0.04 | 0.04 |
| cc_edges | 0.634 | 0.108 | 0 | 0.08 | 0.16 |
| cc_max_prob | 0.651 | 0.143 | 0.4 | 0.32 | 0.2 |

Gene level (best context per gene):

| scorer | auroc | auprc | precision_at_10 | precision_at_25 | precision_at_50 |
|---|---|---|---|---|---|
| lightgbm_oof | 0.536 | 0.199 | 0.2 | 0.12 | 0.16 |
| deg_abs_log2fc | 0.605 | 0.255 | 0.2 | 0.16 | 0.2 |
| cc_edges | 0.626 | 0.276 | 0.3 | 0.24 | 0.2 |

## Top 5 contexts

| rank | gene | celltype_id | IDH_status | score | status | rule_pass | deg_log2fc | deg_neglog10_padj | cc_edges | cc_ephys_bias | ctx_min_n | top_reasons |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | GRIA4 | OPC_GABA_like_tumor | IDH_Mutant | 0.945 | literature tier 2 (unlabeled) | 1 | 1.4 | 204 | 6 | 0 | 1658 | cc_n_partners=2 (+1.23); deg_log2fc=1.4 (+0.63); deg_n_celltypes_sig=4 (+0.62) |
| 2 | GRIK2 | cycling_tumor | IDH_Mutant | 0.926 | novel | 1 | 4.4 | 18.6 | 3 | 1 | 136 | cc_n_partners=2 (+1.22); deg_log2fc=4.4 (+0.55); deg_n_celltypes_sig=3 (+0.50) |
| 3 | PDGFRA | cycling_tumor | IDH_WT | 0.919 | literature tier 1 (label) | 1 | 4.46 | 300 | 2 | 1 | 663 | cc_n_partners=2 (+1.23); deg_log2fc=4.46 (+0.67); deg_n_celltypes_sig=4 (+0.64) |
| 4 | GRIA4 | cycling_tumor | IDH_Mutant | 0.919 | literature tier 2 (unlabeled) | 1 | 3.82 | 38.1 | 3 | 1 | 136 | cc_n_partners=2 (+1.19); deg_log2fc=3.82 (+0.64); deg_n_celltypes_sig=4 (+0.62) |
| 5 | PDGFRA | OPC_GABA_like_tumor | IDH_Mutant | 0.918 | literature tier 1 (label) | 1 | 2.06 | 208 | 2 | 1 | 1658 | cc_n_partners=2 (+1.24); deg_log2fc=2.06 (+0.65); deg_n_celltypes_sig=4 (+0.63) |

## Top novel nominees that pass the 3-table rule

Not in the literature list, but CellChat edge + DEG in this cell type/IDH + adequate n_cells.

| rank | gene | celltype_id | IDH_status | score | status | rule_pass | deg_log2fc | deg_neglog10_padj | cc_edges | cc_ephys_bias | ctx_min_n | top_reasons |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | GRIK2 | cycling_tumor | IDH_Mutant | 0.926 | novel | 1 | 4.4 | 18.6 | 3 | 1 | 136 | cc_n_partners=2 (+1.22); deg_log2fc=4.4 (+0.55); deg_n_celltypes_sig=3 (+0.50) |
| 10 | GRIK2 | cycling_tumor | IDH_WT | 0.905 | novel | 1 | 4.53 | 300 | 3 | 1 | 663 | cc_n_partners=2 (+1.19); deg_log2fc=4.53 (+0.57); cc_cross_compartment_edges=1 (+0.34) |
| 13 | GRIK2 | OPC_GABA_like_tumor | IDH_WT | 0.88 | novel | 1 | 2.36 | 255 | 3 | 1 | 1935 | cc_n_partners=2 (+1.19); deg_log2fc=2.36 (+0.55); cc_cross_compartment_edges=1 (+0.42) |
| 14 | GRIK2 | OPC_GABA_like_tumor | IDH_Mutant | 0.866 | novel | 1 | 0.708 | 46.9 | 3 | 1 | 1658 | cc_n_partners=2 (+1.24); deg_n_celltypes_sig=3 (+0.48); cc_cross_compartment_edges=1 (+0.46) |
| 18 | SLIT1 | OPC_GABA_like_tumor | IDH_Mutant | 0.833 | novel | 1 | 1.49 | 75.2 | 2 | 1 | 1658 | cc_n_partners=2 (+1.20); cc_edges_total=2 (+0.85); deg_n_celltypes_sig=3 (+0.49) |
| 20 | NRCAM | OPC_GABA_like_tumor | IDH_Mutant | 0.813 | novel | 1 | 1.31 | 182 | 4 | 0 | 1658 | cc_n_partners=2 (+1.23); deg_log2fc=1.31 (+0.62); deg_n_celltypes_sig=4 (+0.62) |
| 21 | THY1 | OPC_GABA_like_tumor | IDH_WT | 0.808 | novel | 1 | 2.41 | 300 | 1 | 1 | 1935 | cc_edges_total=2 (+0.76); deg_n_celltypes_sig=4 (+0.58); deg_log2fc=2.41 (+0.47) |
| 22 | NRCAM | AC_like_tumor | IDH_Mutant | 0.808 | novel | 1 | 1.42 | 17.9 | 4 | 0 | 283 | cc_n_partners=2 (+1.20); deg_n_celltypes_sig=4 (+0.62); deg_log2fc=1.42 (+0.61) |
| 23 | THY1 | cycling_tumor | IDH_WT | 0.8 | novel | 1 | 2.09 | 300 | 1 | 1 | 663 | cc_edges_total=2 (+0.75); deg_n_celltypes_sig=4 (+0.57); deg_log2fc=2.09 (+0.46) |
| 24 | KLRC2 | OPC_GABA_like_tumor | IDH_Mutant | 0.799 | novel | 1 | 1.87 | 170 | 6 | 1 | 1658 | deg_log2fc=1.87 (+0.54); deg_n_celltypes_sig=3 (+0.47); cc_cross_compartment_edges=5 (+0.41) |
| 28 | NRCAM | cycling_tumor | IDH_Mutant | 0.758 | novel | 1 | 3.48 | 26.8 | 2 | 1 | 136 | cc_n_partners=2 (+1.19); deg_log2fc=3.48 (+0.64); deg_n_celltypes_sig=4 (+0.62) |
| 29 | THY1 | OPC_GABA_like_tumor | IDH_Mutant | 0.752 | novel | 1 | 1.64 | 132 | 1 | 1 | 1658 | cc_edges_total=2 (+0.76); deg_log2fc=1.64 (+0.45); deg_n_celltypes_sig=3 (+0.44) |
| 30 | SLIT1 | OPC_GABA_like_tumor | IDH_WT | 0.751 | novel | 1 | 2.29 | 69 | 2 | 1 | 1935 | cc_n_partners=2 (+1.16); cc_edges_total=2 (+0.85); deg_log2fc=2.29 (+0.43) |
| 31 | BSG | OPC_GABA_like_tumor | IDH_WT | 0.745 | novel | 1 | 1.01 | 185 | 26 | 0 | 1935 | cc_cross_compartment_edges=10 (+0.62); deg_n_celltypes_sig=3 (+0.54); cc_edges_total=56 (+0.53) |
| 32 | NCL | cycling_tumor | IDH_Mutant | 0.745 | novel | 1 | 0.515 | 2.83 | 16 | 0.125 | 136 | deg_n_celltypes_sig=3 (+0.53); cc_edges_total=70 (+0.41); cc_is_ligand=0 (+0.28) |
| 33 | THY1 | cycling_tumor | IDH_Mutant | 0.741 | novel | 1 | 5.29 | 17.1 | 1 | 1 | 136 | cc_edges_total=2 (+0.75); deg_log2fc=5.29 (+0.46); deg_n_celltypes_sig=3 (+0.44) |
| 34 | NRCAM | OPC_GABA_AC_like_tumor | IDH_Mutant | 0.738 | novel | 1 | 0.638 | 2.04 | 2 | 1 | 352 | cc_n_partners=2 (+1.21); deg_n_celltypes_sig=4 (+0.64); deg_log2fc=0.638 (+0.32) |
| 35 | ROBO1 | cycling_tumor | IDH_Mutant | 0.735 | novel | 1 | 3.24 | 18.5 | 1 | 1 | 136 | cc_edges_total=2 (+0.49); deg_n_celltypes_sig=3 (+0.43); deg_log2fc=3.24 (+0.41) |
| 36 | NCL | OPC_GABA_like_tumor | IDH_Mutant | 0.73 | novel | 1 | 0.585 | 75.5 | 18 | 0 | 1658 | deg_n_celltypes_sig=3 (+0.52); cc_edges_total=70 (+0.42); cc_is_ligand=0 (+0.27) |
| 43 | BSG | cycling_tumor | IDH_WT | 0.703 | novel | 1 | 1.63 | 201 | 13 | 1 | 663 | cc_cross_compartment_edges=5 (+0.55); cc_edges_total=56 (+0.52); deg_n_celltypes_sig=3 (+0.52) |

## Top genes (best context per gene)

| gene_rank | gene | celltype_id | IDH_status | score | status | rule_pass | top_reasons |
|---|---|---|---|---|---|---|---|
| 1 | GRIA4 | OPC_GABA_like_tumor | IDH_Mutant | 0.945 | literature tier 2 (unlabeled) | 1 | cc_n_partners=2 (+1.23); deg_log2fc=1.4 (+0.63); deg_n_celltypes_sig=4 (+0.62) |
| 2 | GRIK2 | cycling_tumor | IDH_Mutant | 0.926 | novel | 1 | cc_n_partners=2 (+1.22); deg_log2fc=4.4 (+0.55); deg_n_celltypes_sig=3 (+0.50) |
| 3 | PDGFRA | cycling_tumor | IDH_WT | 0.919 | literature tier 1 (label) | 1 | cc_n_partners=2 (+1.23); deg_log2fc=4.46 (+0.67); deg_n_celltypes_sig=4 (+0.64) |
| 4 | GRIA2 | cycling_tumor | IDH_Mutant | 0.86 | literature tier 1 (label) | 1 | cc_n_partners=2 (+1.20); deg_log2fc=4.65 (+0.66); deg_n_celltypes_sig=4 (+0.62) |
| 5 | SLIT1 | OPC_GABA_like_tumor | IDH_Mutant | 0.833 | novel | 1 | cc_n_partners=2 (+1.20); cc_edges_total=2 (+0.85); deg_n_celltypes_sig=3 (+0.49) |

## Feature importance (seed-averaged gain)

| feature | gain | splits | mean_abs_shap |
|---|---|---|---|
| cc_edges_total | 4.81e+03 | 125 | 0.406 |
| deg_n_celltypes_sig | 2.15e+03 | 32.8 | 0.306 |
| cc_n_partners | 1.87e+03 | 42.2 | 0.129 |
| deg_log2fc | 1.06e+03 | 19.6 | 0.162 |
| cc_cross_compartment_edges | 694 | 20.2 | 0.166 |
| cc_is_ligand | 692 | 19 | 0.255 |
| cc_n_cells_e1 | 577 | 12.6 | 0.214 |
| deg_pct_diff | 451 | 12 | 0.0707 |
| cc_mean_prob | 412 | 12.8 | 0.0375 |
| cc_n_celltypes_total | 384 | 15.8 | 0.0723 |
| deg_pooled_log2fc | 363 | 13.6 | 0.0568 |
| cc_max_prob | 360 | 9.2 | 0.0706 |
| cc_edges_e2 | 339 | 14.4 | 0.0678 |
| cc_n_pathways_total | 270 | 10 | 0.00889 |
| deg_abs_log2fc | 232 | 7.4 | 0.022 |

## Caveats

- The positive list is a small, hand-curated draft. Labeled genes are biased toward well-studied biology.
- Tier 2 genes are pathway-level picks that overlap the H1-H3 hypotheses. `--tiers 1` is the strict check; with tier 1 alone (16 genes) the model does not beat single-column baselines.
- CellChat is not IDH-stratified, so `cc_*` features repeat across IDH groups; only DEG and n features differ by IDH.
- High scores mean "looks like known glioma communication genes in these tables", not validation.
- `top_reasons` are SHAP contributions from the full-data model; scores themselves are out-of-fold.
