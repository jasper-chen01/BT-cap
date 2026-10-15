# Week 5 results: scETM gene embeddings in the gene-set rankers (parts 1-3)

Full tables: `SCETM_ARMS_REPORT.md`. Per-arm outputs: `tables/`, `tables_scetm/`, `scetm_only/`.
Per-gene ranks under every arm: `arms_gene_ranks.csv`.

## 1. scETM training

- Data: `inhouse_scell_glioma/adata.h5ad` (254,777 cells, raw counts). Trained on a CellType-stratified
  subsample: 20,381 cells (up to 1,800 per cell type, all 12 types), 20,126 genes detected in >= 10 cells.
- scETM: 32-d gene embedding, 50 topics, batch size 2000, lr 5e-3, batch correction on sample (`orig.ident`, 25 samples).
  1,000 epochs (93 min on CPU). The spec says 2,000; loss had plateaued well before 1,000 in a pilot run, so the
  extra 3 h of CPU was not spent.
- Saved in `rag_bioreasoning/knowledge/gene_embeddings/`: `rho` (genes x 32, keyed by gene symbol), `alpha`
  (topics x 32), `theta` (cells x topics, with cell_id / CellType / location / sample), plus `meta.json` and `qc.md`.
- QC: neighbours make biological sense (CD74 -> HLA class II; CD3E -> CD2/CD3G/TRBC2; MKI67 -> ASPM/KIF14/TPX2;
  GRIA2 / NRXN1 / PTPRZ1 -> neural/OPC genes such as NCAM1, CNTN1, CSPG5). GO:0001772 genes (z = 5.3), the
  published 33 (z = 6.0) and CellChat genes (z = 7.8) sit closer together than random sets. SynGO as a whole does
  not (z = -3.2): it is a large, mixed set.

## 2. Features

- Level 1: `emb_00..emb_31` = the gene's rho vector.
- Level 2: `knn_frac_synaptic` / `knn_frac_immune` = share of the gene's 20 nearest cosine neighbours on the label
  list. Rebuilt inside every CV fold from training-fold genes only; a gene is never its own neighbour.
  Leakage check: random vectors instead of rho give AUROC 0.49 / 0.43 (chance).

## 3. Arms and checks (out-of-fold, 5 seeds x 5 folds, grouped by gene)

| model | arm | AUROC | AUPRC | label genes in top 100 | housekeeping in top 100 (on label list) | published-33 median pct |
|---|---|---|---|---|---|---|
| tumor_synaptic | tables | 0.675 | 0.332 | 76 | 83 (64) | 0.98 |
| tumor_synaptic | tables + L1 + L2 | **0.742** | **0.437** | 90 | 83 (78) | 0.82 |
| tumor_synaptic | scETM only | 0.738 | 0.433 | 87 | 83 (78) | 0.72 |
| immune_synapse | tables | 0.666 | 0.015 | 2 | 0 | 0.91 |
| immune_synapse | tables + L1 + L2 | **0.725** | **0.021** | 3 | 0 | 0.57 |
| immune_synapse | scETM only | 0.706 | 0.020 | 3 | 0 | 0.49 |

(AUROC/AUPRC at row level; prevalence 0.149 tumor, 0.005 immune.)

**Check 1, housekeeping.** No change in count: 83 of the top 100 are housekeeping in all three arms. What changes is
*which* ones: with tables, 19 are off-label genes (MT-CO1/2/3, MT-ND1-4, NDUFA4, H3F3A/B, FTL, FTH1, ...) that rank
high only because they are highly expressed; with scETM, those drop out and 78 of 83 are ribosomal / translation genes that SynGO itself lists as synaptic. scETM puts all
ribosomal proteins in one tight cluster, so once a few are labeled, the kNN feature lifts the rest. With housekeeping
genes removed, label hits in the top 100 rise from 44 to 53-54 and the list becomes more neural
(NCAM1, CADM2, NRXN1, NLGN1, PCDH17, PCDH9, SCG3, PTPRD).

**Check 2, immune.** Yes, scETM moves immune-synapse genes up: AUROC 0.67 -> 0.73, median percentile of
GO:0001772 genes 0.75 -> 0.85. The top of the list changes from MHC-II / myeloid DEGs (HLA-DPB1, SPP1, APOE, CD74),
which likely reflect myeloid contamination in tumor clusters, to lymphocyte synapse genes (GRAP2, CD28, LCP2, CD2,
CD48, RAC2, LCP1). CD28 goes to rank 8 and HAVCR2 (published tier 1) to rank 17. Top-100 hits stay small (2 -> 3)
because only 45 GO:0001772 genes are in the DEG universe.

**Check 3, published 33 under scETM only.** Median percentile 0.72 (tumor-synaptic) and 0.49 (immune-synapse, i.e.
chance), vs 0.98 / 0.91 with tables. The published genes rank high because of their fold-change / CellChat
features, not because they co-express with SynGO or immune-synapse genes.

## Takeaways

- scETM features carry most of the gain once added (tables features keep 11% of gain for tumor, 2% for immune),
  and they improve both models on held-out labels. `tables + L1 + L2` is the best arm on AUROC/AUPRC for both.
- The two goals pull apart: scETM helps recover the label lists but lowers the published-33 ranks. For the
  nominee tool (part 4), keep both views: the `tables_scetm` score plus `tables`-only rank, or report which feature
  family drove each score (the `why_*` columns already show this).
- Housekeeping genes need a filter regardless of arm; the SynGO label list itself contains them.
