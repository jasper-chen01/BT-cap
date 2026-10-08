# Week 4: baseline comparison and novel nominees

## Run facts

- Universe: `deg`; 27346 gene-context rows, 9580 genes, and 42 numeric features.
- Cross-validation: StratifiedGroupKFold(n_splits=5) grouped by gene; seeds [0, 1, 2, 3, 4].
- SynGO and GO:0001772 provide the training labels. The published 33 are excluded from training-label construction and from the novel-nominee list.

## Does LightGBM beat the baselines?

The comparison uses the same out-of-fold gene-context scores for LightGBM, CellChat maximum probability, and the three-table rule.

| program | program_label | evaluation_level | metric | labeled_genes | positive_rows | prevalence | lightgbm | cellchat_probability | three_table_rule | best_method | best_baseline | delta_vs_best_baseline | lightgbm_beats_both |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tumor_synaptic | Tumor synaptic (SynGO) | gene_context_row | auroc | 1112 | 3249 | 0.149 | 0.675 | 0.514 | 0.516 | lightgbm | 0.516 | 0.16 | True |
| tumor_synaptic | Tumor synaptic (SynGO) | gene_context_row | auprc | 1112 | 3249 | 0.149 | 0.332 | 0.165 | 0.174 | lightgbm | 0.174 | 0.158 | True |
| tumor_synaptic | Tumor synaptic (SynGO) | gene_context_row | precision_at_25 | 1112 | 3249 | 0.149 | 0.56 | 0.76 | 0.2 | cellchat_probability | 0.76 | -0.2 | False |
| immune_synapse | Immune synapse (GO:0001772) | gene_context_row | auroc | 45 | 135 | 0.00494 | 0.666 | 0.55 | 0.555 | lightgbm | 0.555 | 0.111 | True |
| immune_synapse | Immune synapse (GO:0001772) | gene_context_row | auprc | 45 | 135 | 0.00494 | 0.015 | 0.009 | 0.03 | three_table_rule | 0.0297 | -0.015 | False |
| immune_synapse | Immune synapse (GO:0001772) | gene_context_row | precision_at_25 | 45 | 135 | 0.00494 | 0.04 | 0 | 0.16 | three_table_rule | 0.16 | -0.12 | False |

- **Tumor synaptic (SynGO):** LightGBM beats both baselines for auroc, auprc; it does not beat both for precision_at_25.
- **Immune synapse (GO:0001772):** LightGBM beats both baselines for auroc; it does not beat both for auprc, precision_at_25.

## Novel nominees

Strict nominees are the top model-scored genes that are absent from both public label sets (SynGO and GO:0001772), absent from the published 33, not flagged as housekeeping, and supported by the CellChat + DEG + cell-count rule.

| program_label | nominee_rank | gene | model_score | celltype_id | IDH_status | deg_log2fc | cc_edges | ctx_min_n |
|---|---|---|---|---|---|---|---|---|
| Tumor synaptic (SynGO) | 1 | NCL | 0.935 | OPC_GABA_like_tumor | IDH_Mutant | 0.585 | 18 | 1658 |
| Tumor synaptic (SynGO) | 2 | KLRC2 | 0.909 | OPC_GABA_like_tumor | IDH_Mutant | 1.87 | 6 | 1658 |
| Tumor synaptic (SynGO) | 3 | BSG | 0.89 | OPC_GABA_AC_like_tumor | IDH_WT | 1.32 | 11 | 171 |
| Tumor synaptic (SynGO) | 4 | ITGAV | 0.879 | OPC_GABA_like_tumor | IDH_Mutant | -0.936 | 14 | 1658 |
| Tumor synaptic (SynGO) | 5 | COL9A1 | 0.87 | cycling_tumor | IDH_WT | 4.56 | 5 | 663 |
| Tumor synaptic (SynGO) | 6 | JAM3 | 0.87 | cycling_tumor | IDH_WT | 2.63 | 3 | 663 |
| Tumor synaptic (SynGO) | 7 | SLC1A3 | 0.86 | cycling_tumor | IDH_WT | -0.341 | 13 | 663 |
| Tumor synaptic (SynGO) | 8 | MPZL1 | 0.853 | cycling_tumor | IDH_WT | 2.13 | 4 | 663 |
| Tumor synaptic (SynGO) | 9 | ROBO1 | 0.848 | cycling_tumor | IDH_Mutant | 3.24 | 1 | 136 |
| Tumor synaptic (SynGO) | 10 | GPR37L1 | 0.845 | OPC_GABA_like_tumor | IDH_Mutant | 0.829 | 9 | 1658 |
| Tumor synaptic (SynGO) | 11 | SDC3 | 0.843 | cycling_tumor | IDH_WT | 0.618 | 11 | 663 |
| Tumor synaptic (SynGO) | 12 | DLL1 | 0.813 | OPC_GABA_like_tumor | IDH_WT | 3.43 | 2 | 1935 |
| Tumor synaptic (SynGO) | 13 | PTGES3 | 0.812 | cycling_tumor | IDH_WT | 0.601 | 2 | 663 |
| Tumor synaptic (SynGO) | 14 | PILRB | 0.804 | cycling_tumor | IDH_WT | 1.81 | 7 | 663 |
| Tumor synaptic (SynGO) | 15 | JAM2 | 0.798 | cycling_tumor | IDH_Mutant | 0.796 | 6 | 136 |
| Tumor synaptic (SynGO) | 16 | GLS | 0.788 | cycling_tumor | IDH_WT | 0.992 | 26 | 663 |
| Tumor synaptic (SynGO) | 17 | PSAP | 0.783 | OPC_GABA_like_tumor | IDH_Mutant | -1.36 | 2 | 1658 |
| Tumor synaptic (SynGO) | 18 | ITGB8 | 0.773 | cycling_tumor | IDH_Mutant | 5.55 | 1 | 136 |
| Tumor synaptic (SynGO) | 19 | P4HB | 0.769 | cycling_tumor | IDH_WT | 0.468 | 2 | 663 |
| Tumor synaptic (SynGO) | 20 | MDK | 0.769 | cycling_tumor | IDH_WT | -0.411 | 28 | 663 |
| Tumor synaptic (SynGO) | 21 | PCDHGC3 | 0.76 | cycling_tumor | IDH_WT | 3 | 2 | 663 |
| Tumor synaptic (SynGO) | 22 | SLIT1 | 0.748 | OPC_GABA_like_tumor | IDH_Mutant | 1.49 | 2 | 1658 |
| Tumor synaptic (SynGO) | 23 | CLDN11 | 0.702 | cycling_tumor | IDH_WT | 4.01 | 2 | 663 |
| Tumor synaptic (SynGO) | 24 | HLA-DPA1 | 0.678 | AC_like_tumor | IDH_WT | -3.75 | 2 | 108 |
| Tumor synaptic (SynGO) | 25 | PDGFA | 0.654 | cycling_tumor | IDH_WT | 1.03 | 3 | 663 |
| Immune synapse (GO:0001772) | 1 | HLA-DPB1 | 0.96 | cycling_tumor | IDH_Mutant | -4.35 | 2 | 136 |
| Immune synapse (GO:0001772) | 2 | HLA-DPA1 | 0.922 | cycling_tumor | IDH_WT | -6.15 | 2 | 663 |
| Immune synapse (GO:0001772) | 3 | HLA-E | 0.904 | cycling_tumor | IDH_Mutant | -5.3 | 1 | 136 |
| Immune synapse (GO:0001772) | 4 | HLA-DMA | 0.902 | cycling_tumor | IDH_Mutant | -5.39 | 2 | 136 |
| Immune synapse (GO:0001772) | 5 | PSAP | 0.836 | cycling_tumor | IDH_Mutant | -2.44 | 2 | 136 |
| Immune synapse (GO:0001772) | 6 | ITGB2 | 0.834 | TAM1/TAM2/microglia | IDH_WT | 0.28 | 4 | 2745 |
| Immune synapse (GO:0001772) | 7 | C3AR1 | 0.812 | TAM1/TAM2/microglia | IDH_WT | 0.819 | 4 | 2745 |
| Immune synapse (GO:0001772) | 8 | CLEC2B | 0.806 | TAM1/TAM2/microglia | IDH_WT | -0.833 | 2 | 2745 |
| Immune synapse (GO:0001772) | 9 | HLA-DQA1 | 0.793 | TAM1/TAM2/microglia | IDH_Mutant | -1.92 | 4 | 2106 |
| Immune synapse (GO:0001772) | 10 | CD69 | 0.775 | T cell | IDH_WT | -0.902 | 4 | 1118 |
| Immune synapse (GO:0001772) | 11 | TNFRSF1B | 0.75 | TAM1/TAM2/microglia | IDH_Mutant | 0.456 | 2 | 2106 |
| Immune synapse (GO:0001772) | 12 | AREG | 0.737 | T cell | IDH_WT | 0.967 | 7 | 1118 |
| Immune synapse (GO:0001772) | 13 | HLA-DRB5 | 0.732 | TAM1/TAM2/microglia | IDH_WT | -0.953 | 4 | 2745 |
| Immune synapse (GO:0001772) | 14 | HLA-DMB | 0.717 | TAM1/TAM2/microglia | IDH_Mutant | -0.424 | 4 | 2106 |
| Immune synapse (GO:0001772) | 15 | HLA-DQB1 | 0.709 | TAM1/TAM2/microglia | IDH_Mutant | -1.63 | 4 | 2106 |
| Immune synapse (GO:0001772) | 16 | P4HB | 0.705 | TAM1/TAM2/microglia | IDH_WT | -0.563 | 1 | 2745 |
| Immune synapse (GO:0001772) | 17 | TNFRSF1A | 0.675 | TAM1/TAM2/microglia | IDH_WT | 0.256 | 1 | 2745 |
| Immune synapse (GO:0001772) | 18 | BSG | 0.655 | OPC_GABA_AC_like_tumor | IDH_Mutant | 0.569 | 11 | 352 |
| Immune synapse (GO:0001772) | 19 | ITGAX | 0.655 | TAM1/TAM2/microglia | IDH_WT | 0.416 | 4 | 2745 |
| Immune synapse (GO:0001772) | 20 | SLIT1 | 0.648 | OPC_GABA_like_tumor | IDH_Mutant | 1.49 | 2 | 1658 |
| Immune synapse (GO:0001772) | 21 | LAIR1 | 0.648 | TAM1/TAM2/microglia | IDH_Mutant | -0.275 | 4 | 2106 |
| Immune synapse (GO:0001772) | 22 | PTPRC | 0.568 | T cell | IDH_WT | -0.767 | 4 | 1118 |
| Immune synapse (GO:0001772) | 23 | FPR1 | 0.565 | TAM1/TAM2/microglia | IDH_WT | 0.547 | 10 | 2745 |
| Immune synapse (GO:0001772) | 24 | PTGER4 | 0.563 | TAM1/TAM2/microglia | IDH_WT | 0.716 | 14 | 2745 |
| Immune synapse (GO:0001772) | 25 | ITGB8 | 0.553 | OPC_GABA_like_tumor | IDH_Mutant | -0.521 | 2 | 1658 |

`novel_nominees_score_only.csv` is the sensitivity list: it applies the same exclusions but does not require the three-table rule.

## Interpretation guardrails

- AUROC measures broad separation; AUPRC and precision@25 are more informative for the short candidate list.
- Public gene-set membership is a weak label: genes outside the sets are unknown, not confirmed negatives.
- A high model score nominates a gene for validation; it does not establish glioma function or causality.
- Immune-score nominees whose best context is a tumor cluster need a contamination/doublet check before follow-up.
