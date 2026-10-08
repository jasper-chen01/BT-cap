# How to answer Q1–Q8 and the hypotheses (with evidence)

Use this as your write-up skeleton. Every claim below came from `ask --no-llm` tool traces / table lookups in `rag_bioreasoning` — not from grepping CSVs by hand into an LLM. Cite the tool name when you write.

Raw traces: `fall_semester/turnin/week1_traces/`.

---

## Dataset facts (cite once up front)

| Metric | Value | Tool / file |
|---|---|---|
| Significant interactions | **922** | `dataset_stats_lookup` ← `all_significant_interactions.csv` |
| Pathways / unique pairs | 40 / 96 | same |
| Dominant flow | **tumor→tumor (407)** | same |
| Labeled CellChat identities | AC-like, OPC-GABA(-AC), cycling, MES/TAM mix, T cell, TAM1/TAM2/microglia | overview + `annotation_lookup` |
| Neuron identity | **none** | `annotation_lookup` |
| Groups with n_cells &lt; 50 in CellChat counts | MES_like Ephys_2 **n=6** only | `count_lookup` |

---

## Q1–Q8 model answers

### Q1 — Pathways exclusive to Ephys_2 tumor senders

**Answer:** MK, Glutamate, CNTN, and NOTCH (also JAM, COLLAGEN, PDGF at lower counts).

| Pathway | Ephys_1 senders | Ephys_2 senders |
|---|---:|---:|
| MK | 0 | 28 |
| Glutamate | 0 | 26 |
| CNTN | 0 | 16 |
| NOTCH | 0 | 12 |

**Cite:** `exclusive_pathway_lookup` (compartment=tumor, ephys=Ephys_2).

### Q2 — Does T-cell ligand identity flip?

**Answer:** Only partly. Shared synapse pathways stay on both sides; EGF is the clear Ephys_2-only T-cell sender program.

| Pathway | E1 | E2 | Read |
|---|---:|---:|---|
| MHC-II | 8 | 8 | shared |
| MIF | 6 | 6 | shared |
| CD99 | 9 | 9 | shared |
| EGF | 0 | 7 | **flip (E2-only)** |
| CLEC | 6 | 4 | mild E1 bias |

**DEG backup for the flip:** AREG in IDH-mutant T cells is Ephys_2_high (avg_log2FC ≈ +2.62).  
**Cite:** `pathway_contrast_lookup` + `pathway_filter` + `cellchat_lookup(pathway=EGF)` + `deg_lookup(AREG)`.

### Q3 — Are neurons present?

**Answer:** No. Neither as CellChat source/target nor as an annotation identity. “Other” = astrocyte, endothelial, fibroblast, oligodendrocytes, vascular.

**Cite:** `annotation_lookup(ask_neurons=True)` + dataset overview chunk.

### Q4 — Myeloid Ephys_2 program that Ephys_1 lacks

**Answer:** **GALECTIN** or **Glutamate** (both 13 sender events; Ephys_1 = 0).

**Cite:** `exclusive_pathway_lookup` (myeloid, Ephys_2).

### Q5 — Highest-probability tumor–tumor pair

**Answer:** **PTN_PTPRZ1**, probability **0.262**  
(cycling_tumor / Ephys_2 → OPC_GABA_like_tumor / Ephys_2; n=814 → 14114).

**Cite:** `cellchat_lookup` sorted by probability, tumor→tumor filter.

### Q6 — How many interactions; dominant flow?

**Answer:** **922** significant interactions; dominant flow **tumor→tumor (407)**.

**Cite:** `dataset_stats_lookup`.

### Q7 — Glutamate receptors in the table

**Answer:** **GRIA2, GRIA3, GRIA4, GRIK2** (counts 12 / 6 / 15 / 6).

**Cite:** `cellchat_lookup(pathway=Glutamate)`.

### Q8 — What does NRXN bind?

**Answer:** **NLGN1, NLGN3, LRRTM2, LRRTM3** (also CLSTN1, DAG1). Example pair: NRXN1_NLGN1.

**Cite:** `cellchat_lookup(pathway=NRXN)` + pathway biology chunk.

---

## Hypotheses — how to write each paragraph

Template for each hypothesis:

1. **Observation** (only what tools returned)  
2. **Interpretation** (one careful sentence)  
3. **Candidates to keep / drop** (CellChat ∩ DEG ∩ n_cells)

### H1 — Immune synapse

**Observation.** MHC-II, MIF, CD99, SPP1, CLEC appear in T-cell/myeloid CellChat. T-cell sender counts for MHC-II/MIF/CD99 do **not** flip; EGF does (0 vs 7). Myeloid HLA-DRA and CD74 are DEG Ephys_1_high in IDH-mutant TAM. AREG and KLRB1 are Ephys_2_high in IDH-mutant T cells. MIF/CD99 themselves lack T-cell IDH-mut DEGs in the smoke checks.

**Interpretation.** Immune-synapse machinery is real in the table, but Ephys does not cleanly split “synapse on / off” for MHC-II—more like shared synapse + an Ephys_2 EGF/AREG program and an Ephys_1 myeloid APC program.

**Keep for validation:** AREG/EGF (T), HLA-DR/CD74 (myeloid Ephys_1 side), KLRB1 (partial).  
**Do not keep from DEG alone:** MIF/CD99 in T cells (CellChat yes, DEG empty).

### H2 — Neuron–tumor synaptic-like (tumor program)

**Observation.** Tumor senders strongly skew Ephys_2 for NRXN (10→50), Glutamate (0→26), CNTN, NOTCH, MK. Partners = NLGN/LRRTM and GRIA/GRIK. DEG: NRXN1, NLGN1, GRIA4 Ephys_2_high in IDH-mutant cycling (and related) tumor. **No neuron label.**

**Interpretation.** Supported as a **tumor synaptic-like program**, not as CellChat edges to neurons.

**Keep:** NRXN1–NLGN1, GRIA2/GRIA4 (with glutamate pathway). PTN–PTPRZ1 is strongest edge but is growth/OPC axis—cite separately from “synapse.”

### H3 — Myeloid bridge

**Observation.** Myeloid Ephys_1-leaning senders: MHC-II, MIF, PTN, EGF. Myeloid Ephys_2-only: GALECTIN, Glutamate, CD39, TNF, GAS.

**Interpretation.** Supported: TAMs sit between APC/recruitment (E1) and galectin/glutamate remodeling (E2). Myeloid set ≠ T-cell EGF set ≠ tumor NRXN/GRIA set.

**Keep (conditional):** GALECTIN/LGALS9 — strong CellChat exclusive; DEG only clearly E2_high in IDH-WT smoke check, so say so.

---

## Validation shortlist table (use in your report)

| Gene / pair | Role | CellChat | DEG | n_cells | Keep? |
|---|---|---|---|---|---|
| NRXN1 / NLGN1 | H2 tumor | NRXN pairs | IDH-mut cycling E2_high (+3.7 / +4.2) | OPC_GABA_like E2=14114; cycling E2=814 | **Yes** |
| GRIA2 / GRIA4 | H2 glutamate | Glutamate→GRIA* | GRIA2 OPC_GABA IDH-mut E2_high; GRIA4 cycling E2_high | large tumor groups | **Yes** |
| PTN / PTPRZ1 | top tumor–tumor | prob 0.262 | both E2_high IDH-mut cycling | 814→14114 | **Yes** (growth axis) |
| HLA-DRA / CD74 | H1/H3 APC | MHC-II myeloid | TAM IDH-mut E1_high | TAM n large | **Yes** (Ephys_1 side) |
| AREG / EGF | H1 T flip | EGF E1=0,E2=7 | AREG T IDH-mut E2_high +2.6 | T E1=2662, E2=1463 | **Yes** |
| KLRB1 / CLEC | H1 | CLEC present | KLRB1 T E2_high | OK | Partial |
| LGALS9 | H3 | GALECTIN E2-only | IDH-WT E2_high; IDH-mut weak | TAM large | Conditional |
| MIF / CD99 (T) | H1 | present both Ephys | no T IDH-mut DEG | OK | CellChat only — weak for expression claim |

**Hard exclude from stories:** MES_like / Ephys_2 (**n=6**).

---

## One paragraph you can paste as a conclusion

> Using tool-routed RAG over the compartment-separated CellChat and IDH-stratified DEG tables, Ephys_2 tumor senders uniquely carry Glutamate, MK, CNTN, and NOTCH, with NRXN–NLGN/LRRTM and GRIA/GRIK partners and DEG support for NRXN1/NLGN1/GRIA4 in IDH-mutant tumor states—consistent with a tumor synaptic-like program without any labeled neuron identity. T-cell MHC-II/MIF/CD99 are shared across Ephys states; the clearest T-cell flip is EGF/AREG. Myeloid cells bridge programs: Ephys_1-leaning MHC-II/CD74 versus Ephys_2-only GALECTIN and Glutamate. Candidates for experimental validation are therefore split by compartment: tumor NRXN/GRIA (± PTN–PTPRZ1), T-cell AREG/EGF, and myeloid HLA-DR/CD74 versus LGALS9—each retained only where CellChat, DEG, and n_cells agree.

---

## What to turn in alongside this narrative

1. Tool traces (`week1_traces/Q1–Q8_*`)  
2. This evidence write-up (or the canvas `ephys-week1-answers`)  
3. Half-page retrieval note (`WEEK1_RETRIEVAL_NOTE.md`) — already written for the “how the agent behaved” part of the assignment
