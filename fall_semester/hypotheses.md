# Working hypotheses

These are **priors to test** with the RAG / BioReasoning agent, not conclusions.

**Project goal.** Use tool-using RAG and (later) LLM reasoning over `data/` to **nominate a short list of genes and ligand–receptor pairs for experimental validation**. A candidate must be supported by more than one table: it appears in CellChat, it is DEG-supported in the right cell type and IDH group, and the sender/receiver groups have enough cells. Do not propose a gene because a model “thinks” it is interesting.

## H1 — Immune synapse

In T cells and myeloid cells, Ephys_1 vs Ephys_2 may index how active a cell is while it forms or maintains an **immunological synapse**.

Pairs to test in this dataset:

- MHC-II (HLA-DR/DP/DQ/DM) → CD4
- MIF → CD74 with CXCR4 or CD44
- SPP1 (osteopontin) → CD44
- CD99 homophilic adhesion
- CLEC family → KLRB1 (CD161)

**If supported:** immune-synapse genes (HLA-DR, CD74, CD4, SPP1, KLRB1) are the validation set for the TME, not the tumor synaptic set.

**Caution:** Ephys is a cluster label, not a measured synaptic current.

## H2 — Neuron–tumor synaptic-like communication

In glioma-like tumor states (OPC/GABA-like, AC-like, cycling), one Ephys class may index **synaptic-like / neuron-contact machinery**: neurexin–neuroligin, AMPA/kainate glutamate receptors (GRIA), NCAM, contactin.

This is consistent with published glioma–neuron synapses. It is **not** proof that neurons are in this CellChat run.

**If supported:** NRXN, NLGN, GRIA, and related adhesion genes are the tumor validation set — a **program** in tumor cells, not a CellChat edge to a labeled neuron.

**Caution:** There is no `neuron` identity in the sender/receiver labels. Synaptic pairs among tumor cells can be autocrine/paracrine mimicry, or a proxy for tumor cells that contact neurons outside this table.

## H3 — Myeloid bridge

TAM/microglia may sit between the two programs: antigen presentation on one Ephys side, remodeling / glutamatergic / checkpoint pairs on the other.

**If supported:** the myeloid validation set is different from both the T-cell synapse set and the tumor NRXN/GRIA set. Do not assume it.

## How a gene becomes a validation candidate

Ask the agent, then keep a gene only if:

1. It (or its pair) is in CellChat for that compartment and Ephys state.
2. `deg_lookup` supports it in the same cell type, split by IDH when the question is about expression.
3. `count_lookup` shows the group is large enough to trust.
4. The same gene does **not** tell the same story in T cells (negative control).

Week 1 is traces only (`ask --no-llm`). The LLM that writes the nomination list comes later, and it may only cite those traces.
