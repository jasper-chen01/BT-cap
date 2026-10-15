# Week 5: do scETM gene embeddings help the gene-set rankers?

- Universe: `deg`: 27346 gene x cell type x IDH rows, 9580 genes.
- Scores are out-of-fold (StratifiedGroupKFold(n_splits=5) grouped by gene, seeds [0, 1, 2, 3, 4]); checks use each gene's best context.
- Arms: `tables` = week-4 DEG / CellChat / n_cells features; `tables_scetm` = tables + Level 1 (`emb_00..`, the gene's rho vector) + Level 2 (`knn_frac_*`); `scetm_only` = Level 1 + Level 2 with no fold-change or table features.
- Embeddings: `scetm_gene_embeddings_d32.csv` (32-d); 9578 of 9580 universe genes embedded. Level 2 uses k=20 cosine neighbours, rebuilt inside every CV fold from training-fold genes only, and a gene is never its own neighbour.

## tumor_synaptic

| arm | AUROC rows | AUPRC rows | AUROC genes | AUPRC genes | housekeeping in top 100 | of which in label set | label-set genes in top 100 | label-set genes in top 100 (housekeeping removed) | label-set median pct | immune_synapse genes in top 100 | published-33 median pct | published-33 AUROC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tables | 0.675 | 0.332 | 0.668 | 0.289 | 83 | 64 | 76 | 44 | 0.714 | 0 | 0.982 | 0.81 |
| tables_scetm | 0.742 | 0.437 | 0.726 | 0.368 | 83 | 78 | 90 | 53 | 0.777 | 0 | 0.818 | 0.739 |
| scetm_only | 0.738 | 0.433 | 0.721 | 0.363 | 83 | 78 | 87 | 54 | 0.769 | 0 | 0.724 | 0.679 |

**Top 15 genes per arm** (best context per gene)

| rank | tables | tables_scetm | scetm_only |
|---|---|---|---|
| 1 | PTN | RPL9 | RPL9 |
| 2 | PTPRZ1 | PFDN5 | PFDN5 |
| 3 | MT-CO2 | RPL4 | EEF1A1 |
| 4 | MT-CO1 | RPL3 | RPL39 |
| 5 | NDUFA4 | RPL39 | RPL3 |
| 6 | RPL13 | RPL37 | RPL37 |
| 7 | RPS23 | RPS12 | RPL4 |
| 8 | RPS14 | RPS15A | RPL21 |
| 9 | RPL28 | EEF1A1 | RPL13 |
| 10 | RPS18 | RPL21 | RPS6 |
| 11 | EEF1A1 | RPS25 | RPL29 |
| 12 | RPS15 | RPLP2 | RPLP2 |
| 13 | MT-CO3 | RPS9 | RPS15A |
| 14 | NLGN1 | RPL13 | RPS12 |
| 15 | RPS9 | RPL29 | RPL27A |

**Top 15 genes per arm, housekeeping removed** (best context per gene)

| rank | tables | tables_scetm | scetm_only |
|---|---|---|---|
| 1 | PTN | PFDN5 | PFDN5 |
| 2 | PTPRZ1 | RPSA | FAU |
| 3 | NLGN1 | FAU | RPSA |
| 4 | GRIA2 | RACK1 | RACK1 |
| 5 | NCL | NCAM1 | NOP53 |
| 6 | CALM1 | NOP53 | PTPRF |
| 7 | FAU | PTPRZ1 | PTPRD |
| 8 | GNAS | CADM2 | PCDH17 |
| 9 | ZBTB20 | NPM1 | NCAM1 |
| 10 | ACTB | NRXN1 | CADM2 |
| 11 | RACK1 | NLGN1 | NSA2 |
| 12 | GAPDH | PCDH17 | C6orf48 |
| 13 | RPSA | HNRNPA1 | NPM1 |
| 14 | GRIA4 | SCG3 | GPR37L1 |
| 15 | C1orf61 | PCDH9 | SCG3 |

**Feature importance share (gain)**

- `tables`: Level 1 (emb_*) 0.00, Level 2 (knn) 0.00, tables 1.00. Top: deg_pct_e1 0.27, deg_pct_e2 0.24, deg_pooled_log2fc 0.12.
- `tables_scetm`: Level 1 (emb_*) 0.47, Level 2 (knn) 0.42, tables 0.11. Top: knn_frac_synaptic 0.42, emb_15 0.10, emb_07 0.06.
- `scetm_only`: Level 1 (emb_*) 0.54, Level 2 (knn) 0.46, tables 0.00. Top: knn_frac_synaptic 0.46, emb_15 0.11, emb_07 0.07.

- `tables` housekeeping genes in top 100: MT-CO2, MT-CO1, NDUFA4, RPL13, RPS23, RPS14, RPL28, RPS18, EEF1A1, RPS15, MT-CO3, RPS9, RPS19, RPL10, RPL23, RPL30, RPS11, RPLP0, RPL11, H3F3B, RPL39, RPS28, RPL41, RPS12, RPS15A, RPL29, RPS24, RPS2, RPL6, RPL35A, RPL8, RPS16, RPS8, RPL18A, NACA, RPL36, RPS13, RPS25, RPL5, RPL7A, MT-ND3, RPL7, RPLP1, RPL37, RPL3, RPS27A, RPS3A, RPL19, RPL10A, TPT1, RPS4X, RPL37A, RPL32, RPL15, RPS3, RPL24, PTMA, RPS27, RPS5, RPL34, RPL21, RPL18, MT-CYB, RPL9, H3F3A, RPL14, RPL27A, MT-ND2, RPL23A, EIF1, MT-ND1, RPL35, MT-ND4, UBA52, RPL27, FTH1, BTF3, RPL12, MT-ATP6, RPLP2, RPS7, H2AFZ, FTL
- `tables_scetm` housekeeping genes in top 100: RPL9, RPL4, RPL3, RPL39, RPL37, RPS12, RPS15A, EEF1A1, RPL21, RPS25, RPLP2, RPS9, RPL13, RPL29, RPL7A, TPT1, RPL18, RPS6, RPL27A, UBA52, RPS14, RPL19, UQCRB, RPS11, NACA, RPL34, RPL5, EIF3E, RPL37A, RPS16, RPL6, RPS19, RPS23, RPS13, EEF1B2, RPS28, RPS21, RPL23A, RPS3, RPL27, RPL35A, RPL41, RPL15, RPL31, RPL11, RPS27, RPL36, RPL18A, RPL8, RPS7, BTF3, RPS5, RPL22, RPL13A, RPLP0, RPL32, RPS4X, RPL14, RPS18, RPS29, RPL30, RPL24, RPS27A, RPS8, RPL7, RPS2, RPL12, RPS15, RPL10, RPL26, RPL35, RPS3A, RPL10A, RPL23, EIF3H, RPS20, RPL36AL, EIF3L, EEF1D, RPLP1, RPL38, RPS24, EEF2
- `scetm_only` housekeeping genes in top 100: RPL9, EEF1A1, RPL39, RPL3, RPL37, RPL4, RPL21, RPL13, RPS6, RPL29, RPLP2, RPS15A, RPS12, RPL27A, RPS9, RPS14, TPT1, RPS25, UBA52, RPL7A, RPL18, RPS21, RPL5, NACA, RPS19, UQCRB, RPL19, RPL34, RPS23, EIF3E, RPL13A, RPL37A, EEF1B2, RPS7, RPS28, RPL31, RPL8, BTF3, RPL6, RPL22, RPS16, RPL15, RPS27, RPS13, RPS11, RPL41, RPL11, RPL36, RPL18A, RPL27, RPL35A, RPS5, RPL10, RPS18, RPL32, RPS27A, RPL30, RPL23A, RPS3, RPL14, RPS29, RPL26, RPLP0, RPS4X, RPS8, RPL12, RPL35, RPS15, RPS2, RPL10A, RPL7, EIF3H, RPL24, RPS3A, EEF1D, RPS20, EIF3L, RPL36AL, RPL23, RPL38, RPLP1, RPS24, RPL28

## immune_synapse

| arm | AUROC rows | AUPRC rows | AUROC genes | AUPRC genes | housekeeping in top 100 | of which in label set | label-set genes in top 100 | label-set genes in top 100 (housekeeping removed) | label-set median pct | tumor_synaptic genes in top 100 | published-33 median pct | published-33 AUROC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tables | 0.666 | 0.0151 | 0.676 | 0.0179 | 0 | 0 | 2 | 2 | 0.747 | 10 | 0.908 | 0.773 |
| tables_scetm | 0.725 | 0.021 | 0.731 | 0.0229 | 0 | 0 | 3 | 3 | 0.845 | 7 | 0.569 | 0.586 |
| scetm_only | 0.706 | 0.0202 | 0.732 | 0.019 | 0 | 0 | 3 | 3 | 0.829 | 7 | 0.49 | 0.557 |

**Top 15 genes per arm** (best context per gene)

| rank | tables | tables_scetm | scetm_only |
|---|---|---|---|
| 1 | HLA-DPB1 | GRAP2 | NCF1 |
| 2 | SPP1 | CXCL16 | ST3GAL1 |
| 3 | APOE | CD52 | PRDM1 |
| 4 | CD74 | C1QC | PTGDR |
| 5 | HLA-DPA1 | LAPTM5 | LCP2 |
| 6 | ANXA1 | NCF1 | FCER1G |
| 7 | HLA-DMB | LCP1 | RNASET2 |
| 8 | HLA-E | CD28 | CD48 |
| 9 | HLA-DMA | LCP2 | FCGR3A |
| 10 | HLA-DRA | RAC2 | CD2 |
| 11 | HLA-DRB1 | CD48 | IL10RA |
| 12 | EGFR | PTGDR | RAC2 |
| 13 | CLEC2B | RNASET2 | HLA-DQB1 |
| 14 | CLEC2D | TRADD | TRADD |
| 15 | SLC4A4 | MCTP2 | MCTP2 |

**Top 15 genes per arm, housekeeping removed** (best context per gene)

| rank | tables | tables_scetm | scetm_only |
|---|---|---|---|
| 1 | HLA-DPB1 | GRAP2 | NCF1 |
| 2 | SPP1 | CXCL16 | ST3GAL1 |
| 3 | APOE | CD52 | PRDM1 |
| 4 | CD74 | C1QC | PTGDR |
| 5 | HLA-DPA1 | LAPTM5 | LCP2 |
| 6 | ANXA1 | NCF1 | FCER1G |
| 7 | HLA-DMB | LCP1 | RNASET2 |
| 8 | HLA-E | CD28 | CD48 |
| 9 | HLA-DMA | LCP2 | FCGR3A |
| 10 | HLA-DRA | RAC2 | CD2 |
| 11 | HLA-DRB1 | CD48 | IL10RA |
| 12 | EGFR | PTGDR | RAC2 |
| 13 | CLEC2B | RNASET2 | HLA-DQB1 |
| 14 | CLEC2D | TRADD | TRADD |
| 15 | SLC4A4 | MCTP2 | MCTP2 |

**Feature importance share (gain)**

- `tables`: Level 1 (emb_*) 0.00, Level 2 (knn) 0.00, tables 1.00. Top: deg_pooled_log2fc 0.20, deg_pct_e1 0.20, deg_log2fc 0.17.
- `tables_scetm`: Level 1 (emb_*) 0.91, Level 2 (knn) 0.08, tables 0.02. Top: emb_03 0.28, emb_10 0.13, emb_07 0.08.
- `scetm_only`: Level 1 (emb_*) 0.98, Level 2 (knn) 0.02, tables 0.00. Top: emb_03 0.23, emb_10 0.18, emb_20 0.07.


## Notes

- Housekeeping = the week-4 regex (MT-, RPL/RPS, histones, NDUF, COX, ATP5, EEF/EIF, ...). SynGO itself annotates many ribosomal / mitochondrial genes as synaptic, so housekeeping genes in the tumor-synaptic top 100 can be true label hits.
- Level 1 and Level 2 are gene-wide: in `scetm_only` every cell type x IDH row of a gene gets the same score.
- `knn_frac_*` in the per-arm `genesets_scores.csv` is the all-genes version used only for SHAP explanations; the scores themselves use the per-fold version.
- Sanity check during development: with random 32-d vectors in place of rho, `scetm_only` scored AUROC 0.49 (tumor) and 0.43 (immune), i.e. no label leakage through the kNN feature.
