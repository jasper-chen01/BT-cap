# Week 5: Q39–Q44 `nominate_lookup` audit

Run against the `tables_scetm` arm (27,346 gene-context rows; five seeds x five folds). Scores are held-out probabilities averaged across seeds. The fold field lists the held-out fold used for each seed; top-feature explanations come from the full explanation model.

| Question | `nominate_lookup` fired | Required terms present | Grounded result |
|---|---:|---:|---|
| Q39 — NRXN1 tumor-synaptic score | yes | yes | IDH-mutant cycling tumor: `p_tumor_synaptic=0.900521`, context rank 9. Main driver: `knn_frac_synaptic`; top three are `knn_frac_synaptic`, `emb_07`, `emb_15`, so the explanation is scETM-driven. |
| Q40 — HLA-DRA in TAM/microglia | yes | yes | IDH-mutant TAM/microglia: `p_immune_synapse=0.583330`, immune context rank 89. There is no tumor-synaptic score because that model is restricted to tumor compartments. The immune explanation is scETM-driven. |
| Q41 — top immune-synapse gene in T cells | yes | yes | `GRAP2`, IDH-mutant T cell, `p_immune_synapse=0.938963`, immune context rank 1. |
| Q42 — not on SynGO but tumor-synaptic | yes | yes | `PFDN5`, IDH-WT cycling tumor, `in_syngo=False`, `p_tumor_synaptic=0.957383`, context rank 2. Its top explanation is mixed scETM + table features. |
| Q43 — GRIA2 rank and driver in OPC-like tumor | yes | yes | IDH-mutant: score 0.888294, rank 84. IDH-WT: score 0.870426, rank 80. Top features are `knn_frac_synaptic`, `emb_07`, and `emb_15`, so both are scETM-driven. |
| Q44 — NCL three-table check | yes | yes | Best returned context is IDH-mutant OPC-like tumor: `passes_three_table=True`, `in_cellchat=True`, 18 CellChat edges, `min_n=1658`, and tumor-synaptic score 0.800899. |

## Routing trace summary

| Question | Tools fired |
|---|---|
| Q39 | `nominate_lookup`, `cellchat_lookup`, `deg_lookup`, `pathway_filter` |
| Q40 | `nominate_lookup`, `cellchat_lookup`, `deg_lookup`, `pathway_filter` |
| Q41 | `nominate_lookup`, `cellchat_lookup`, `pathway_filter` |
| Q42 | `nominate_lookup`, `pathway_filter` |
| Q43 | `nominate_lookup`, `cellchat_lookup`, `deg_lookup` |
| Q44 | `nominate_lookup`, `cellchat_lookup`, `deg_lookup` |

Audit result: **6/6 passed**. Every question fired `nominate_lookup`, and every required-term check passed against the agent context.
