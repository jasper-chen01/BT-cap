# Week 1 TODOs

The assignment is RAG + BioReasoning, not filtering CSVs. Ask a question in English. The starter routes it to tools and retrieves chunks. Improve that loop. Do not open Excel. Do not paste CSVs into Gemini.

## Tools already in the starter

| Tool | Table |
|---|---|
| `cellchat_lookup` | `data/cellchat_ephys_plus_celltype/all_significant_interactions.csv` |
| `count_lookup` | `data/cellchat_ephys_plus_celltype/cellchat_group_counts.csv` |
| `deg_lookup` | `data/within_cluster_ephys_DEGs_by_IDH/` |
| `annotation_lookup` | `data/glioma_compartment_ephys_clustering/glioma_tumor_tcell_tam_ephys_counts.csv` |
| `support_join` | CellChat ⋈ DEG |
| `pathway_filter` | CellChat + theme tags |
| `exclusive_pathway_lookup` | CellChat Ephys contrasts (sender-exclusive pathways) |
| `pathway_contrast_lookup` | CellChat Ephys_1 vs Ephys_2 pathway table |
| `dataset_stats_lookup` | CellChat interaction / flow totals |

Use the count table.

## Checklist

- [x] Install and run `ask --no-llm` from `fall_semester/rag_bioreasoning`
- [x] Q1–Q8 in [questions.md](questions.md): save tool traces + top retrieved titles
- [x] Each question lists tool, file, gene/pair, and whether the required term appeared
- [x] At least one DEG question and one “are there neurons?” question so `deg_lookup` and `annotation_lookup` both fire
- [x] If retrieval misses, fix the agent (not the CSV)
- [x] Half-page note: what works vs what retrieves poorly

Turn-in folder: `fall_semester/turnin/` (`WEEK1_Q1_Q8_SUMMARY.md`, `WEEK1_RETRIEVAL_NOTE.md`, `week1_traces/`).

Empty lookups are allowed. Do not invent a pair or a DEG.
