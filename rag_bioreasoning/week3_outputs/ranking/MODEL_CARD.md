# Week 3 gene-ranking prototype

## Ranked unit

One CellChat ligand/receptor gene in one IDH status × transcriptomic cell-type context.

## Evidence and labels

The 39 features come only from the CellChat, within-cluster DEG, and cell-count tables used by the RAG tools. Positive labels are an internal 33-gene, compartment-aware seed list derived from the project hypotheses; they are **not** an independently curated published gold standard. Metrics therefore measure recovery of those seeds and should not be interpreted as clinical or external biological validation.

## Leakage control

GroupKFold with 5 folds holds out entire genes. The out-of-fold score averages 5 LightGBM fits per candidate using seeds [11, 23, 37, 53, 71]. No model scores a gene it trained on.

## Comparison

| scorer                     |   AUROC |   AUPRC |   Top-25 precision |
|:---------------------------|--------:|--------:|-------------------:|
| LightGBM                   |   0.821 |   0.540 |              1.000 |
| CellChat probability alone |   0.731 |   0.385 |              1.000 |
| 3-table rule alone         |   0.730 |   0.241 |              0.160 |

The baselines are maximum CellChat probability and a three-table rule combining any CellChat edge, DEG presence, and adequate group size.
