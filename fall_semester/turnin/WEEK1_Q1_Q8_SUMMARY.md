# Week 1 turn-in — Q1–Q8 tool traces

Agent: `fall_semester/rag_bioreasoning` (`ask --no-llm`).
Raw traces: `fall_semester/turnin/week1_traces/Q{1-8}_trace.txt` and `Q{1-8}.json`.
DEG smoke test: `DEG_sample_NRXN1_*` (`deg_lookup` + `cellchat_lookup`).

## Per-question summary

| Q | Tool(s) | File(s) | Gene / pair / pathway surfaced | Required terms present? |
|---|---|---|---|---|
| 1 | `exclusive_pathway_lookup` | `all_significant_interactions.csv` | MK, Glutamate, CNTN, NOTCH (Ephys_2 tumor senders) | Yes: Glutamate, MK, CNTN, NOTCH |
| 2 | `pathway_contrast_lookup`, `pathway_filter`, `cellchat_lookup` | CellChat | MHC-II / MIF / CD99 (shared both Ephys); EGF (Ephys_2-only T-cell senders) | Yes: MHC-II, MIF, CD99, EGF |
| 3 | `annotation_lookup` | `glioma_tumor_tcell_tam_ephys_counts.csv` | Identities listed; neuron labels: **none** | Yes: annotation / overview identity chunk |
| 4 | `exclusive_pathway_lookup`, `pathway_contrast_lookup` | CellChat | GALECTIN, Glutamate exclusive to myeloid Ephys_2 | Yes: GALECTIN or Glutamate |
| 5 | `cellchat_lookup` (tumor→tumor) | CellChat | **PTN_PTPRZ1** (prob ≈ 0.262) | Yes: PTN_PTPRZ1 |
| 6 | `dataset_stats_lookup`, `count_lookup` | CellChat + `cellchat_group_counts.csv` | **922** interactions; dominant flow **tumor→tumor** (407) | Yes: overview / raw stats |
| 7 | `cellchat_lookup`, `pathway_filter` | CellChat | Glutamate → **GRIA2 / GRIA3 / GRIA4 / GRIK2** | Yes |
| 8 | `cellchat_lookup`, `pathway_filter` | CellChat | NRXN1 → **NLGN1 / NLGN3 / LRRTM2/3** | Yes |

## Top retrieved titles (abbrev.)

1. Pathways exclusive to Ephys_2 tumor senders  
2. MIF_CD74_CXCR4 / CD99_CD99 T-cell interactions; MHC-II / EGF knowledge  
3. Neuron-tumor hypothesis; annotation identities outside tumor/T/TAM; dataset overview  
4. Pathways exclusive to Ephys_2 myeloid senders  
5. Highest-probability tumor-tumor pair: PTN_PTPRZ1  
6. Dataset overview (922 interactions, flows)  
7. Pathway biology: Glutamate; GRIA* interaction rows  
8. Pathway biology: NRXN; NRXN1_NLGN1 rows  

## Agent fixes made so required terms appear

Starter TF-IDF ranked generic “tumor senders” contrasts and missed exclusive pathways. Changes in the agent (not the CSVs):

- Router: `exclusive_pathway_lookup`, `pathway_contrast_lookup`, `dataset_stats_lookup`; better pathway / flip / highest-prob / Glutamate / NRXN routing  
- Corpus: exclusive-pathway docs + top tumor–tumor overview doc  
- Retriever: metadata boosts for exclusive / overview / pathway / neuron questions  
- Windows: UTF-8 stdout so traces with biology text do not crash `cp1252`
