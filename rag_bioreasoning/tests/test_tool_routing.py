from ephys_rag.ingest import load_pathway_knowledge
from ephys_rag.schema import Interaction
from ephys_rag.tools import ToolRegistry, dispatch_tools
from ephys_rag.tools.registry import infer_query_slots


def _interaction(pathway: str, ephys: str, *, ligand: str, receptor: str) -> Interaction:
    return Interaction(
        source_label=f"T cell / {ephys}",
        target_label="cycling_tumor / Ephys_2",
        ligand=ligand,
        receptor=receptor,
        interaction_name=f"{ligand}_{receptor}",
        pathway=pathway,
        probability=0.1,
        pval=0.0,
        source_cell="T cell",
        source_ephys=ephys,
        target_cell="cycling_tumor",
        target_ephys="Ephys_2",
        source_compartment="tcell",
        target_compartment="tumor",
    )


def test_ephys_side_question_gets_sender_pathway_contrast_evidence():
    interactions = [
        _interaction("MIF", "Ephys_1", ligand="MIF", receptor="CD74"),
        _interaction("MIF", "Ephys_2", ligand="MIF", receptor="CD74"),
        _interaction("EGF", "Ephys_2", ligand="AREG", receptor="EGFR"),
    ]
    registry = ToolRegistry(interactions=interactions, degs_idh=[], degs_pooled=[])

    traces = dispatch_tools(
        "Which T-cell pathways are supported on both Ephys sides, and which pathway is Ephys_2-only?",
        registry,
    )

    contrast = next(trace for trace in traces if trace.tool == "sender_pathway_contrast")
    by_pathway = {row["pathway"]: row for row in contrast.rows}
    assert by_pathway["MIF"]["status"] == "shared"
    assert by_pathway["MIF"]["Ephys_1"] == 1
    assert by_pathway["MIF"]["Ephys_2"] == 1
    assert by_pathway["EGF"]["status"] == "Ephys_2_only"


def test_glutamate_knowledge_names_each_receptor_explicitly():
    summary = load_pathway_knowledge()["Glutamate"]["summary"]

    for receptor in ("GRIA2", "GRIA3", "GRIA4", "GRIK2"):
        assert receptor in summary


def test_mhc_ii_question_routes_to_the_pathway_and_retrieves_cd4():
    registry = ToolRegistry(
        interactions=[
            _interaction("MHC-II", "Ephys_1", ligand="HLA-DRA", receptor="CD4")
        ],
        degs_idh=[],
        degs_pooled=[],
    )

    traces = dispatch_tools("Which receptor does MHC-II use here?", registry)

    cellchat = next(trace for trace in traces if trace.tool == "cellchat_lookup")
    assert cellchat.filters["pathway"] == "MHC-II"
    assert cellchat.rows[0]["pair"] == "HLA-DRA_CD4"


def test_private_tcell_pathway_retrieves_contrast_and_egf_pair():
    registry = ToolRegistry(
        interactions=[
            _interaction("MIF", "Ephys_1", ligand="MIF", receptor="CD74"),
            _interaction("MIF", "Ephys_2", ligand="MIF", receptor="CD74"),
            _interaction("EGF", "Ephys_2", ligand="AREG", receptor="EGFR"),
        ],
        degs_idh=[],
        degs_pooled=[],
    )

    traces = dispatch_tools(
        "Which T-cell-sent pathway is private to Ephys_2?",
        registry,
    )

    contrast = next(trace for trace in traces if trace.tool == "sender_pathway_contrast")
    assert any(row["pathway"] == "EGF" for row in contrast.rows)
    pair_trace = next(
        trace
        for trace in traces
        if trace.tool == "cellchat_lookup" and trace.filters["pathway"] == "EGF"
    )
    assert pair_trace.rows[0]["pair"] == "AREG_EGFR"


def test_cellchat_trace_rows_expose_exported_pvalues():
    registry = ToolRegistry(
        interactions=[_interaction("EGF", "Ephys_2", ligand="AREG", receptor="EGFR")],
        degs_idh=[],
        degs_pooled=[],
    )

    trace = registry.cellchat_lookup(pathway="EGF")

    assert trace.rows[0]["pval"] == 0.0
    assert "unique_pvals=0.0" in trace.note


def test_glutamate_text_does_not_accidentally_infer_tam():
    slots = infer_query_slots(
        "Name a glutamate ligand on the sender side, not just a receptor."
    )

    assert slots["celltype"] is None
    assert slots["compartment"] is None


def test_two_idh_groups_and_two_celltypes_keep_the_deg_lookup_broad():
    registry = ToolRegistry(interactions=[], degs_idh=[], degs_pooled=[])

    traces = dispatch_tools(
        "Is EGFR Ephys_2-high in IDH-mutant cycling tumor? "
        "Does IDH-WT OPC-like tumor agree?",
        registry,
    )

    deg = next(trace for trace in traces if trace.tool == "deg_lookup")
    assert deg.filters["idh"] is None
    assert deg.filters["celltype"] is None


def test_ephys_count_comparison_returns_both_sides():
    registry = ToolRegistry(interactions=[], degs_idh=[], degs_pooled=[])

    traces = dispatch_tools(
        "How many T cells are labeled Ephys_1 vs Ephys_2?",
        registry,
    )

    count = next(trace for trace in traces if trace.tool == "count_lookup")
    assert count.filters["ephys"] is None
    assert {row["group"] for row in count.rows} == {
        "T cell / Ephys_1",
        "T cell / Ephys_2",
    }


def test_six_cell_group_question_uses_the_low_count_lookup():
    registry = ToolRegistry(interactions=[], degs_idh=[], degs_pooled=[])

    traces = dispatch_tools("Which labeled group has only 6 cells?", registry)

    count = next(trace for trace in traces if trace.tool == "count_lookup")
    assert count.filters["low_n"] is True
    assert any(row["n_cells"] == 6 for row in count.rows)


def test_tcell_sender_negative_control_checks_nrxn_and_glutamate_as_sources():
    registry = ToolRegistry(
        interactions=[
            _interaction("Glutamate", "Ephys_2", ligand="SLC1A1_GLS", receptor="GRIA2")
        ],
        degs_idh=[],
        degs_pooled=[],
    )

    traces = dispatch_tools("Do T cells send NRXN or Glutamate in this table?", registry)

    lookups = [trace for trace in traces if trace.tool == "cellchat_lookup"]
    assert any(
        trace.filters["gene"] == "NRXN" and trace.filters["role"] == "source"
        for trace in lookups
    )
    assert any(
        trace.filters["pathway"] == "Glutamate" and trace.filters["role"] == "source"
        for trace in lookups
    )


def test_tcell_synaptic_refutation_also_retrieves_areg_egf_evidence():
    registry = ToolRegistry(
        interactions=[_interaction("EGF", "Ephys_2", ligand="AREG", receptor="EGFR")],
        degs_idh=[],
        degs_pooled=[],
    )

    traces = dispatch_tools(
        "If a model claims T-cell Ephys_2 is the synaptic NRXN/GRIA program, "
        "what would refute it?",
        registry,
    )

    lookups = [trace for trace in traces if trace.tool == "cellchat_lookup"]
    assert any(
        trace.filters["gene"] == "AREG" and trace.filters["role"] == "source"
        for trace in lookups
    )
    assert any(row["pathway"] == "EGF" for trace in lookups for row in trace.rows)
