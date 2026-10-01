# Week 3 stability notes

A question is unstable when either the faithful or required-term hit rate is between 1/5 and 4/5.

## Q16

Status: **unstable**; faithful 4/5; hit 0/5.

Wrong or incomplete runs:

- Run 4: The cellchat_lookup tool returned p-values of 0.0, 0.01, 0.02, 0.03, 0.04. The annotation_lookup tool returned celltype_id and n_cells values. The p-values are 0.0.

## Q21

Status: **unstable**; faithful 5/5; hit 1/5.

All cached runs passed the automatic grounding audit.

## Q25

Status: **stable**; faithful 5/5; hit 0/5.

All cached runs passed the automatic grounding audit.

## Q27

Status: **stable**; faithful 5/5; hit 5/5.

All cached runs passed the automatic grounding audit.

## Q28

Status: **unstable**; faithful 5/5; hit 1/5.

All cached runs passed the automatic grounding audit.

## Q29

Status: **unstable**; faithful 4/5; hit 4/5.

Wrong or incomplete runs:

- Run 1: The `count_lookup` tool found 7 hits for the pair `OPC_GABA_like_tumor / Ephys_2` with `n_cells` of 14114. The `AC_like_tumor / Ephys_2` group has `n_cells` of 446. The `OPC_GABA_AC_like_tumor / Ephys_2` group has `n_cells` of 1012. The `AC_like_tumor / Ephys_2` group has `n

## Q36

Status: **unstable**; faithful 1/5; hit 5/5.

Wrong or incomplete runs:

- Run 1: T cells send NRXN. T cells send Glutamate.
- Run 2: T cells send NRXN in this table. T cells send Glutamate in this table.
- Run 4: T cells send NRXN in this table. T cells send Glutamate in this table.
- Run 5: T cells send NRXN in this table. T cells send Glutamate in this table.

## Q38

Status: **unstable**; faithful 0/5; hit 2/5.

Wrong or incomplete runs:

- Run 1: The evidence does not refute the claim that T-cell Ephys_2 is the synaptic NRXN/GRIA program. There is no evidence of NRXN or GRIA genes in the traces.
- Run 2: The evidence does not refute the claim that T-cell Ephys_2 is the synaptic NRXN/GRIA program. There is no evidence that NRXN or GRIA are expressed in T cells.
- Run 3: The evidence does not refute the claim that T-cell Ephys_2 is the synaptic NRXN/GRIA program. There are no hits for NRXN or GRIA in the cellchat_lookup or deg_lookup tools.
- Run 4: The evidence does not refute the claim that T-cell Ephys_2 is the synaptic NRXN/GRIA program. There is no evidence of NRXN or GRIA genes being significantly differentially expressed between Ephys_2 and Ephys_1 T cells. The only significant interaction found is between AREG and EGFR, which is not a known NRXN/GRIA interaction.
- Run 5: The evidence does not refute the claim that T-cell Ephys_2 is the synaptic NRXN/GRIA program. There is no evidence of NRXN or GRIA genes being significantly differentially expressed between Ephys_2 and Ephys_1 T cells. The only significant interaction found is between AREG and EGFR, which is not a known NRXN/GRIA interaction.
