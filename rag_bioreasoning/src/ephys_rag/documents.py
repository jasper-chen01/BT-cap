from __future__ import annotations

from ephys_rag.config import DATA_DIR
from ephys_rag.contrasts import pathway_contrasts, summarize_dataset
from ephys_rag.ingest import (
    load_annotation_counts,
    load_cellchat_counts,
    load_deg_idh_summary,
    load_hypotheses,
    load_pathway_knowledge,
)
from ephys_rag.schema import Interaction, RAGDocument


def _join(values: list[str]) -> str:
    return ", ".join(values) if values else "none"


def interaction_documents(interactions: list[Interaction]) -> list[RAGDocument]:
    docs: list[RAGDocument] = []
    for idx, row in enumerate(interactions):
        docs.append(
            RAGDocument(
                doc_id=f"interaction-{idx}",
                title=f"{row.interaction_name} | {row.source_label} → {row.target_label}",
                text=row.as_sentence(),
                kind="interaction",
                metadata={
                    "pathway": row.pathway,
                    "ligand": row.ligand,
                    "receptor": row.receptor,
                    "interaction_name": row.interaction_name,
                    "source_cell": row.source_cell,
                    "target_cell": row.target_cell,
                    "source_ephys": row.source_ephys,
                    "target_ephys": row.target_ephys,
                    "source_compartment": row.source_compartment,
                    "target_compartment": row.target_compartment,
                    "themes": row.themes,
                    "probability": row.probability,
                    "source_n_cells": row.source_n_cells,
                    "target_n_cells": row.target_n_cells,
                },
            )
        )
    return docs


def knowledge_documents() -> list[RAGDocument]:
    docs: list[RAGDocument] = []
    for name, info in load_pathway_knowledge().items():
        themes = info.get("themes", [])
        text = f"Pathway {name}. Themes: {_join(themes)}. {info.get('summary', '').strip()}"
        docs.append(
            RAGDocument(
                doc_id=f"pathway-{name}",
                title=f"Pathway biology: {name}",
                text=" ".join(text.split()),
                kind="pathway_knowledge",
                metadata={"pathway": name, "themes": themes},
            )
        )
    for hyp in load_hypotheses():
        text = (
            f"Working hypothesis {hyp['id']}: {hyp['title']}. "
            f"Applies to compartments {_join(hyp.get('compartments', []))}. "
            f"{hyp.get('summary', '').strip()} "
            f"Caution: {hyp.get('caution', '').strip()}"
        )
        docs.append(
            RAGDocument(
                doc_id=f"hypothesis-{hyp['id']}",
                title=f"Hypothesis: {hyp['title']}",
                text=" ".join(text.split()),
                kind="hypothesis",
                metadata={"hypothesis_id": hyp["id"], "compartments": hyp.get("compartments", [])},
            )
        )
    return docs


def count_documents() -> list[RAGDocument]:
    docs: list[RAGDocument] = []
    for row in load_cellchat_counts():
        docs.append(
            RAGDocument(
                doc_id=f"count-cellchat-{row.group}",
                title=f"n_cells {row.group}",
                text=(
                    f"CellChat group {row.group} has n_cells={row.n_cells} "
                    f"(compartment {row.compartment}). "
                    "Groups with n_cells < 50 should not drive a biological story."
                ),
                kind="count",
                metadata={
                    "cell": row.cell,
                    "ephys": row.ephys,
                    "compartment": row.compartment,
                    "n_cells": row.n_cells,
                },
            )
        )
    others = [row for row in load_annotation_counts() if row.compartment == "other"]
    identities = sorted({row.cell for row in others})
    docs.append(
        RAGDocument(
            doc_id="annotation-other-identities",
            title="Annotation identities outside tumor / T cell / TAM",
            text=(
                "The annotation count table labels compartment 'other' as: "
                f"{_join(identities)}. There is no neuron identity. "
                "Do not treat oligodendrocytes or astrocytes as neurons."
            ),
            kind="annotation",
            metadata={"compartment": "other", "themes": ["neuron_tumor"]},
        )
    )
    return docs


def deg_summary_documents() -> list[RAGDocument]:
    docs: list[RAGDocument] = []
    frame = load_deg_idh_summary()
    for rec in frame.to_dict(orient="records"):
        cell = rec["celltype_id"]
        idh = rec["IDH_status"]
        docs.append(
            RAGDocument(
                doc_id=f"deg-summary-{idh}-{cell}",
                title=f"DEG summary {idh} {cell}",
                text=(
                    f"IDH-stratified Ephys_2 vs Ephys_1 DEG summary for {cell} "
                    f"({idh}): n_Ephys_1={rec['n_Ephys_1']}, n_Ephys_2={rec['n_Ephys_2']}, "
                    f"n_sig_total={rec['n_sig_total']}, "
                    f"n_sig_Ephys2_high={rec['n_sig_Ephys2_high']}, "
                    f"n_sig_Ephys1_high={rec['n_sig_Ephys1_high']}. "
                    f"Top Ephys_2_high genes: {rec['top_Ephys2_high']}. "
                    f"Top Ephys_1_high genes: {rec['top_Ephys1_high']}. "
                    "Pooled DEGs are a different table; always show both when claiming NRXN/GRIA."
                ),
                kind="deg_summary",
                metadata={
                    "celltype_id": cell,
                    "idh": idh,
                    "compartment": rec.get("cluster_class", ""),
                },
            )
        )
    return docs


def contrast_documents(interactions: list[Interaction]) -> list[RAGDocument]:
    docs: list[RAGDocument] = []
    summary = summarize_dataset(interactions)
    cell_types = sorted({row.source_cell for row in interactions} | {row.target_cell for row in interactions})
    docs.append(
        RAGDocument(
            doc_id="dataset-overview",
            title="Dataset overview",
            text=(
                f"DATA_DIR={DATA_DIR}. "
                f"The CellChat table has {summary['n_interactions']} significant "
                f"ligand-receptor interactions across {summary['n_pathways']} pathways "
                f"and {summary['n_pairs']} unique pairs. "
                f"Labeled identities: {_join(cell_types)}. "
                f"Source Ephys counts: {summary['source_ephys']}. "
                f"Compartment flows: {summary['flows']}. "
                f"{summary['n_pairs_with_group_n_lt_50']} unique pairs touch a group with n_cells < 50. "
                "There is no neuron identity in the source or target labels."
            ),
            kind="overview",
            metadata={"themes": ["neuron_tumor", "immune_synapse"]},
        )
    )
    for item in pathway_contrasts(interactions):
        docs.append(
            RAGDocument(
                doc_id=f"contrast-{item.compartment}-{item.pathway}",
                title=f"{item.pathway} contrast in {item.compartment} senders",
                text=(
                    f"Ephys contrast for pathway {item.pathway} with {item.compartment} "
                    f"cells as senders. Ephys_1 count {item.ephys_1} "
                    f"(mean probability {item.mean_prob_e1:.4f}). "
                    f"Ephys_2 count {item.ephys_2} "
                    f"(mean probability {item.mean_prob_e2:.4f}). "
                    f"Delta Ephys_2 minus Ephys_1 = {item.delta_e2_minus_e1}."
                ),
                kind="contrast",
                metadata={
                    "pathway": item.pathway,
                    "source_compartment": item.compartment,
                    "delta": item.delta_e2_minus_e1,
                },
            )
        )
    return docs


def build_corpus(interactions: list[Interaction]) -> list[RAGDocument]:
    return (
        knowledge_documents()
        + contrast_documents(interactions)
        + count_documents()
        + deg_summary_documents()
        + interaction_documents(interactions)
    )

