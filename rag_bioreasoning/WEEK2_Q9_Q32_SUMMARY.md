# WEEK2 Q9–Q32: Hugging Face MedGemma Bio-Reasoning Summary

## Run configuration

- Model weights: `google/medgemma-4b-it` from Hugging Face (gated access, read-only token).
- Hosting: Vertex AI endpoint `8471750835909427200` in `us-central1`.
- Serving: Google vLLM 0.19 container on `g2-standard-12` with one NVIDIA L4 GPU.
- Prompt profile: two detailed rows per tool, compact catalogs of all distinct evidence values, and the single highest-ranked retrieval document.
- Dataset: refreshed CellChat interactions, IDH-stratified Ephys DEGs, and cell-count tables used in the earlier analysis.
- Raw run: `analysis_runs/medgemma_hf_q9_q32/evaluation_20260924T172225487430+0000.md`.
- Machine-readable run: `analysis_runs/medgemma_hf_q9_q32/evaluation_20260924T172225487430+0000.json`.

The current Hugging Face Model Garden recipe selected a TGI 2.4 image that failed because it did not support the model's `gemma3` architecture. The same Hugging Face weights were therefore served with the gemma3-compatible vLLM image already proven by the project's healthy MedGemma deployment. A direct endpoint smoke test returned HTTP 200 before the evaluation began.

## Evaluation audit

- Requests completed: **24/24**.
- Required tools present in traces: **24/24**.
- Required evidence terms present in traces: **24/24**.
- Exact required terms repeated in raw model answers: **17/24**.
- Manual answer audit: **18 correct, 3 incomplete, 3 substantively incorrect**.

The table below is the audited deliverable. “Corrected” means the concise answer follows the retrieved tables even when MedGemma's raw interpretation did not.

| Q | Audited table-grounded answer | Raw MedGemma audit |
|---|---|---|
| 9 | **PTN is primarily a shared tumor-sender scaffold, not a clean Ephys discriminator.** Tumor-sender PTN has 68 Ephys_1 versus 59 Ephys_2 events (mean probability 0.1261 vs 0.1171), and `PTN_PTPRZ1` occurs on both sides. PTN can still be Ephys_2-high as a DEG in selected tumor strata. | **Incorrect; corrected here.** It called PTN a discriminator and ignored the aggregate contrast. |
| 10 | MHC-II ligands use **CD4** as the receptor; an example is `HLA-DRA_CD4`. | Correct. |
| 11 | APOE binds the **`TREM2_TYROBP`** receptor complex. | Correct. |
| 12 | **EGF** is private to Ephys_2 T-cell senders: 0 Ephys_1 versus 7 Ephys_2 events, through **`AREG_EGFR`**. PTN is also Ephys_2-only for T-cell senders, with 2 events. | **Incomplete; corrected here.** It identified EGF but omitted `AREG_EGFR`. |
| 13 | **ANNEXIN** is Ephys_1-only among tumor senders (4 vs 0). MHC-I is another Ephys_1-only example (1 vs 0). | Correct. |
| 14 | SPP1 uses **CD44** and **`ITGAV_ITGB1`** in this table. | **Incomplete; corrected here.** It reported only CD44. |
| 15 | CLEC2B, CLEC2C, and CLEC2D bind **KLRB1** on T cells. | Correct. |
| 16 | CellChat p-values vary: **0.00, 0.01, 0.02, 0.03, and 0.04** occur in the export. | Correct; the exact-term checker missed only the `pval` versus “p-values” spelling. |
| 17 | CNTN1 binds **NRCAM** and **NOTCH1**. | **Incomplete; corrected here.** It reported only NRCAM. |
| 18 | The highest-probability myeloid self-loop is **`APOE_TREM2_TYROBP`**, TAM/microglia Ephys_1 → Ephys_1, probability **0.2392**. | Correct. |
| 19 | `NRXN1_NLGN1` and glutamate-to-GRIA interactions support a **synaptic-like tumor/myeloid communication program**. Labeled neurons are not required for that interpretation, and no neuron identity appears in the annotations. | Correct. |
| 20 | An Ephys_1-enriched NRXN claim is refuted by the tumor-sender contrast: **10 Ephys_1 versus 50 Ephys_2 interactions**, with mean probabilities **0.0029 versus 0.0043**. | **Incorrect; corrected here.** It claimed the evidence lacked Ephys information despite the retrieved contrast. |
| 21 | **Yes.** NRXN1 is Ephys_2-high in IDH-mutant cycling tumor: avg_log2FC **3.7084**, adjusted p ≈ **6.98×10^-45**, with 270 Ephys_1 and 136 Ephys_2 cells. | Correct. |
| 22 | **Yes.** HLA-DRA is Ephys_1-high across the retrieved cycling, AC-like, OPC/GABA-like, and mixed OPC/GABA/AC-like tumor strata in both IDH-mutant and IDH-WT data. | Correct. |
| 23 | **Yes.** AREG is a T-cell Ephys_2-high DEG in IDH-mutant cells (log2FC **2.622**) and IDH-WT cells (log2FC **0.9672**). | Correct. |
| 24 | **Yes.** GRIA2 is Ephys_2-high in both IDH-mutant and IDH-WT OPC/GABA-like tumor. | Correct; the exact-term checker missed only the abbreviated IDH wording. |
| 25 | EGFR is Ephys_2-high in IDH-mutant cycling tumor (log2FC **3.583**), but IDH-WT OPC/GABA-like tumor disagrees and is Ephys_1-high (log2FC **−4.8103**). | Correct; the exact-term checker missed only abbreviated labels. |
| 26 | **Yes.** In IDH-mutant TAM1/TAM2/microglia, HLA-DRA is Ephys_1-high (log2FC **−1.304**, adjusted p = **0.0**). | Correct. |
| 27 | **MES-like tumor/TAM1/microglia / Ephys_2** is the six-cell group. | Correct. |
| 28 | T cells: **Ephys_1 = 2,662 cells; Ephys_2 = 1,463 cells**. | Correct. |
| 29 | **Yes, with the assignment's count criterion.** AC-like tumor / Ephys_2 has **n_cells = 446**, above the low-count warning threshold of 50. | **Incorrect/over-cautious; corrected here.** It reported 446 but refused the threshold conclusion. |
| 30 | **Yes.** NRXN1 is a CellChat ligand from cycling-tumor Ephys_2 and an Ephys_2-high DEG in IDH-mutant cycling tumor. | Correct. |
| 31 | **Yes.** AREG is the T-cell EGF ligand in `AREG_EGFR` CellChat interactions and an Ephys_2-high T-cell DEG in both IDH groups. | Correct. |
| 32 | **Yes.** HLA-DRA is an MHC-II CellChat ligand in `HLA-DRA_CD4` and an Ephys_1-high tumor DEG. | Correct. |

## Bottom line

MedGemma successfully used the RAG evidence and answered most questions correctly, but it was not reliable enough to accept without deterministic validation. Its main failure mode was ignoring aggregate contrast or threshold evidence even when that evidence was present. The recommended workflow is therefore **tools and table retrieval first, MedGemma synthesis second, automated term/tool checks plus manual biological audit last**.
