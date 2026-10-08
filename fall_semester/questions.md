# Questions

Required terms are the **minimum** the retriever or tool traces should surface. They are not a full answer key. Empty lookups are allowed. Do not invent a pair or a DEG.

Q1–Q8 were last week. This week: Q9–Q32 with MedGemma and Gemini on the same traces. Q33–Q38 are stretch.

---

## Q1–Q8 — CellChat (already assigned)

1. Which pathways are exclusive to Ephys_2 **tumor senders**?
   - Must retrieve: Glutamate, MK, CNTN, NOTCH
2. Does T-cell ligand identity flip between Ephys_1 and Ephys_2?
   - Must retrieve: MHC-II, MIF, CD99, EGF
3. Are neurons present as a CellChat source or target identity in this table?
   - Must retrieve: a dataset-overview or annotation chunk that lists cell types
4. Name one program that myeloid Ephys_2 senders use and myeloid Ephys_1 senders do not.
   - Must retrieve: GALECTIN or Glutamate
5. What is the highest-probability tumor–tumor pair?
   - Must retrieve: PTN_PTPRZ1
6. How many significant interactions are in the table, and what is the dominant compartment flow?
   - Must retrieve: overview or raw interaction stats
7. Which receptors does the Glutamate pathway hit in this table?
   - Must retrieve: GRIA2, GRIA3, GRIA4, or GRIK2
8. What does NRXN bind in this table?
   - Must retrieve: NLGN1 or NLGN3 or LRRTM

---

## Q9–Q20 — more CellChat

9. Is PTN an Ephys discriminator among tumor senders, or a shared scaffold?
   - Must retrieve: PTN and PTPRZ1
10. Which receptor does MHC-II use here?
    - Must retrieve: CD4
11. What is the APOE receptor complex in this table?
    - Must retrieve: TREM2_TYROBP or APOE_TREM2_TYROBP
12. Which T-cell-sent pathway is private to Ephys_2?
    - Must retrieve: EGF or AREG_EGFR
13. Name a tumor-sender pathway that is exclusive (or nearly exclusive) to Ephys_1.
    - Must retrieve: ANNEXIN or MHC-I or MHC-II
14. Which receptors does SPP1 use here?
    - Must retrieve: CD44 or ITGAV_ITGB1
15. What does CLEC bind on T cells in this table?
    - Must retrieve: KLRB1
16. Do CellChat p-values vary in this export?
    - Must retrieve: a row that shows the pval field
17. CNTN1 binds which partners here?
    - Must retrieve: NRCAM or NOTCH1
18. What is the highest-probability myeloid self-loop?
    - Must retrieve: APOE_TREM2_TYROBP
19. Interpret NRXN1–NLGN1 and glutamate–GRIA in this dataset. Are neurons required for that interpretation?
    - Must retrieve: NRXN and Glutamate
20. If a model claims NRXN is Ephys_1-enriched in tumor, what count evidence would refute it?
    - Must retrieve: NRXN tumor contrast

---

## Q21–Q32 — DEG, counts, and joins (use the other tools)

These should fire `deg_lookup`, `count_lookup`, `annotation_lookup`, or `support_join`. A CellChat-only answer is not enough.

21. Is NRXN1 DEG-supported as Ephys_2-high in IDH-mutant cycling tumor?
    - Must retrieve: NRXN1, cycling, IDH
    - Tool: `deg_lookup`
22. Is HLA-DRA DEG-supported as Ephys_1-high in tumor?
    - Must retrieve: HLA-DRA, Ephys_1
    - Tool: `deg_lookup`
23. Is AREG a T-cell Ephys_2-high DEG?
    - Must retrieve: AREG, T cell
    - Tool: `deg_lookup`
24. Does GRIA2 go the same Ephys direction in IDH-mutant and IDH-WT OPC-like tumor?
    - Must retrieve: GRIA2 and both IDH groups
    - Tool: `deg_lookup`
25. Is EGFR Ephys_2-high in IDH-mutant cycling tumor? Does IDH-WT OPC-like tumor agree?
    - Must retrieve: EGFR, cycling or OPC, IDH
    - Tool: `deg_lookup`
26. In IDH-mutant TAM/microglia, is HLA-DRA Ephys_1-high?
    - Must retrieve: HLA-DRA, TAM or microglia
    - Tool: `deg_lookup`
27. Which labeled group has only 6 cells?
    - Must retrieve: MES and 6
    - Tool: `count_lookup` or `annotation_lookup`
28. How many T cells are labeled Ephys_1 vs Ephys_2?
    - Must retrieve: T cell and a count
    - Tool: `count_lookup`
29. Is AC-like tumor / Ephys_2 large enough to treat as a trusted sender?
    - Must retrieve: AC_like and n_cells
    - Tool: `count_lookup`
30. Is NRXN1 both a CellChat ligand and a DEG in IDH-mutant cycling Ephys_2?
    - Must retrieve: NRXN1 from CellChat **and** from `deg_lookup`
    - Tool: `support_join` or both lookups
31. Is AREG both the T-cell EGF ligand in CellChat and a T-cell DEG?
    - Must retrieve: AREG and EGF or EGFR
    - Tool: `support_join` or both lookups
32. Does HLA-DRA appear as both an MHC-II CellChat ligand and an Ephys_1-high tumor DEG?
    - Must retrieve: HLA-DRA and MHC-II
    - Tool: `support_join` or both lookups

---

## Q33–Q38 — stretch (faithfulness / negative control)

33. What does MIF bind in this table?
    - Must retrieve: CD74
34. What does LGALS9 (GALECTIN) bind?
    - Must retrieve: HAVCR2 or PTPRC or CD44
35. Name a glutamate **ligand** (sender side), not just a receptor.
    - Must retrieve: SLC1A1 or SLC1A3 or GLS
36. Do T cells send NRXN or Glutamate in this table?
    - Must retrieve: T cell. An empty NRXN/Glutamate hit on T-cell senders is correct.
37. Is NCAM used as a tumor–tumor adhesion pair?
    - Must retrieve: NCAM1
38. If a model claims T-cell Ephys_2 is the synaptic (NRXN/GRIA) program, what would refute it?
    - Must retrieve: T cell and AREG or EGF. Empty T-cell NRXN is allowed.
