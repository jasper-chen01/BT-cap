from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ephys_rag.contrasts import exclusive_pathways, pathway_contrasts
from ephys_rag.config import (
    ANNOTATION_CELLS_CSV,
    ANNOTATION_COUNTS_CSV,
    CELLCHAT_COUNTS_CSV,
    CELLCHAT_CSV,
    DEG_IDH_CSV,
    LOW_N_CELLS,
)
from ephys_rag.ingest import (
    load_annotation_counts,
    load_cellchat_counts,
    load_degs_idh,
    load_degs_pooled,
    load_interactions,
    load_pathway_knowledge,
)
from ephys_rag.schema import DegRecord, Interaction, ToolResult, normalize_idh

_GENE_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,}(?:-[A-Z0-9]+)?)\b")
_PAIR_RE = re.compile(r"\b([A-Z][A-Z0-9-]+)_([A-Z][A-Z0-9-]+)\b")
def _match_cell(value: str, query: str | None) -> bool:
    if not query:
        return True
    return query.lower() in value.lower()


def _match_ephys(value: str, query: str | None) -> bool:
    if not query:
        return True
    return value.lower() == query.lower()


def _match_compartment(value: str, query: str | None) -> bool:
    if not query:
        return True
    return value.lower() == query.lower()


@dataclass
class ToolRegistry:
    """Look up rows. Do not paste whole CSVs into a prompt."""

    interactions: list[Interaction] = field(default_factory=load_interactions)
    degs_idh: list[DegRecord] = field(default_factory=load_degs_idh)
    degs_pooled: list[DegRecord] = field(default_factory=load_degs_pooled)

    def cellchat_lookup(
        self,
        *,
        gene: str | None = None,
        pathway: str | None = None,
        celltype: str | None = None,
        ephys: str | None = None,
        compartment: str | None = None,
        role: str = "either",
        limit: int = 15,
    ) -> ToolResult:
        hits: list[Interaction] = []
        for row in self.interactions:
            if gene:
                g = gene.upper()
                if row.ligand.upper() != g and row.receptor.upper() != g and g not in row.interaction_name.upper():
                    continue
            if pathway and pathway.lower() not in row.pathway.lower():
                continue
            if role == "source":
                if not _match_cell(row.source_cell, celltype):
                    continue
                if not _match_ephys(row.source_ephys, ephys):
                    continue
                if not _match_compartment(row.source_compartment, compartment):
                    continue
            elif role == "target":
                if not _match_cell(row.target_cell, celltype):
                    continue
                if not _match_ephys(row.target_ephys, ephys):
                    continue
                if not _match_compartment(row.target_compartment, compartment):
                    continue
            else:
                cell_ok = _match_cell(row.source_cell, celltype) or _match_cell(row.target_cell, celltype)
                ephys_ok = _match_ephys(row.source_ephys, ephys) or _match_ephys(row.target_ephys, ephys)
                comp_ok = _match_compartment(row.source_compartment, compartment) or _match_compartment(
                    row.target_compartment, compartment
                )
                if celltype and not cell_ok:
                    continue
                if ephys and not ephys_ok:
                    continue
                if compartment and not comp_ok:
                    continue
            hits.append(row)
        hits.sort(key=lambda item: item.probability, reverse=True)
        return ToolResult(
            tool="cellchat_lookup",
            source_file=str(CELLCHAT_CSV.name),
            filters={
                "gene": gene,
                "pathway": pathway,
                "celltype": celltype,
                "ephys": ephys,
                "compartment": compartment,
                "role": role,
            },
            n_hits=len(hits),
            rows=[
                {
                    "pair": row.interaction_name,
                    "pathway": row.pathway,
                    "source": row.source_label,
                    "target": row.target_label,
                    "prob": round(row.probability, 4),
                    "pval": row.pval,
                    "source_n": row.source_n_cells,
                    "target_n": row.target_n_cells,
                }
                for row in hits[:limit]
            ],
            note=(
                "unique_pvals="
                + ",".join(str(value) for value in sorted({row.pval for row in hits}))
                if hits
                else "unique_pvals=none"
            ),
        )

    def count_lookup(
        self,
        *,
        celltype: str | None = None,
        ephys: str | None = None,
        low_n: bool = False,
    ) -> ToolResult:
        rows = load_cellchat_counts()
        hits = [
            row
            for row in rows
            if _match_cell(row.cell, celltype) and _match_ephys(row.ephys, ephys) and (not low_n or row.n_cells < LOW_N_CELLS)
        ]
        return ToolResult(
            tool="count_lookup",
            source_file=str(CELLCHAT_COUNTS_CSV.name),
            filters={"celltype": celltype, "ephys": ephys, "low_n": low_n or None},
            n_hits=len(hits),
            rows=[{"group": row.group, "n_cells": row.n_cells, "compartment": row.compartment} for row in hits],
            note="Flag any CellChat pair whose sender or receiver has n_cells < 50.",
        )

    def deg_lookup(
        self,
        *,
        gene: str,
        celltype: str | None = None,
        idh: str | None = None,
        pooled: bool = False,
        limit: int = 20,
    ) -> ToolResult:
        table = self.degs_pooled if pooled else self.degs_idh
        source = DEG_IDH_CSV.name if not pooled else "combined_Ephys2_vs_Ephys1_significant_DEGs_by_cluster.csv"
        idh_norm = normalize_idh(idh)
        gene_u = gene.upper()
        hits = []
        for rec in table:
            if rec.gene.upper() != gene_u:
                continue
            if celltype and not _match_cell(rec.celltype_id, celltype):
                continue
            if idh_norm and rec.idh_status != idh_norm:
                continue
            hits.append(rec)
        hits.sort(key=lambda item: abs(item.avg_log2fc), reverse=True)
        return ToolResult(
            tool="deg_lookup",
            source_file=source,
            filters={"gene": gene, "celltype": celltype, "idh": idh_norm, "pooled": pooled},
            n_hits=len(hits),
            rows=[
                {
                    "gene": rec.gene,
                    "direction": rec.direction,
                    "avg_log2FC": round(rec.avg_log2fc, 4),
                    "p_val_adj": rec.p_val_adj,
                    "celltype_id": rec.celltype_id,
                    "IDH_status": rec.idh_status,
                    "n_Ephys_1": rec.n_ephys_1,
                    "n_Ephys_2": rec.n_ephys_2,
                }
                for rec in hits[:limit]
            ],
        )

    def annotation_lookup(
        self,
        *,
        compartment: str | None = None,
        celltype: str | None = None,
        ask_neurons: bool = False,
    ) -> ToolResult:
        rows = load_annotation_counts()
        hits = [
            row
            for row in rows
            if (not compartment or row.compartment.lower() == compartment.lower())
            and _match_cell(row.cell, celltype)
        ]
        identities = sorted({row.cell for row in rows})
        neuron_like = [name for name in identities if "neuron" in name.lower()]
        note = (
            f"Identities present: {', '.join(identities)}. "
            f"Neuron labels: {neuron_like or 'none'}. "
            f"Do not scan {ANNOTATION_CELLS_CSV.name} (~255k rows); use this count table."
        )
        if ask_neurons:
            hits = [row for row in rows if row.compartment == "other"]
        return ToolResult(
            tool="annotation_lookup",
            source_file=str(ANNOTATION_COUNTS_CSV.name),
            filters={"compartment": compartment, "celltype": celltype, "ask_neurons": ask_neurons or None},
            n_hits=len(hits),
            rows=[
                {
                    "compartment": row.compartment,
                    "celltype_id": row.cell,
                    "ephys": row.ephys,
                    "n_cells": row.n_cells,
                }
                for row in hits
            ],
            note=note,
        )

    def support_join(
        self,
        *,
        ligand: str,
        receptor: str,
        sender: str | None = None,
        receiver: str | None = None,
        idh: str | None = None,
    ) -> ToolResult:
        """Baseline A: ligand DEG in sender cluster AND receptor DEG in receiver cluster."""
        pairs = [
            row
            for row in self.interactions
            if row.ligand.upper() == ligand.upper() and row.receptor.upper() == receptor.upper()
        ]
        if sender:
            pairs = [row for row in pairs if _match_cell(row.source_cell, sender)]
        if receiver:
            pairs = [row for row in pairs if _match_cell(row.target_cell, receiver)]

        out = []
        for row in pairs[:20]:
            lig = self.deg_lookup(gene=row.ligand, celltype=row.source_cell, idh=idh)
            rec = self.deg_lookup(gene=row.receptor, celltype=row.target_cell, idh=idh)
            lig_hit = lig.rows[0] if lig.rows else None
            rec_hit = rec.rows[0] if rec.rows else None
            supported = bool(lig_hit and rec_hit)
            out.append(
                {
                    "pair": row.interaction_name,
                    "source": row.source_label,
                    "target": row.target_label,
                    "ligand_deg": lig_hit["direction"] if lig_hit else "missing",
                    "receptor_deg": rec_hit["direction"] if rec_hit else "missing",
                    "expression_supported": supported,
                    "idh": normalize_idh(idh) or "all_IDH_rows",
                    "source_n": row.source_n_cells,
                    "target_n": row.target_n_cells,
                }
            )
        return ToolResult(
            tool="support_join",
            source_file=f"{CELLCHAT_CSV.name} ⋈ {DEG_IDH_CSV.name}",
            filters={"ligand": ligand, "receptor": receptor, "sender": sender, "receiver": receiver, "idh": idh},
            n_hits=len(out),
            rows=out,
            note="expression-supported means ligand is a significant DEG in the sender cluster and receptor in the receiver cluster (same IDH filter).",
        )

    def pathway_filter(
        self,
        *,
        theme: str | None = None,
        pathway: str | None = None,
        compartment: str | None = None,
        ephys: str | None = None,
        limit: int = 15,
    ) -> ToolResult:
        knowledge = load_pathway_knowledge()
        wanted = set()
        if theme:
            for name, info in knowledge.items():
                if theme.lower() in [t.lower() for t in info.get("themes", [])]:
                    wanted.add(name)
        hits = []
        for row in self.interactions:
            if pathway and pathway.lower() not in row.pathway.lower():
                continue
            if theme and row.pathway not in wanted and theme.lower() not in [t.lower() for t in row.themes]:
                continue
            if compartment and row.source_compartment != compartment and row.target_compartment != compartment:
                continue
            if ephys and row.source_ephys != ephys and row.target_ephys != ephys:
                continue
            hits.append(row)
        hits.sort(key=lambda item: item.probability, reverse=True)
        return ToolResult(
            tool="pathway_filter",
            source_file=str(CELLCHAT_CSV.name),
            filters={"theme": theme, "pathway": pathway, "compartment": compartment, "ephys": ephys},
            n_hits=len(hits),
            rows=[
                {
                    "pair": row.interaction_name,
                    "pathway": row.pathway,
                    "themes": ",".join(row.themes),
                    "source": row.source_label,
                    "target": row.target_label,
                    "prob": round(row.probability, 4),
                }
                for row in hits[:limit]
            ],
        )

    def sender_pathway_contrast(
        self,
        *,
        compartment: str,
        exclusive_ephys: str | None = None,
    ) -> ToolResult:
        contrasts = pathway_contrasts(self.interactions)
        if exclusive_ephys:
            hits = exclusive_pathways(
                contrasts,
                compartment=compartment,
                ephys=exclusive_ephys,
            )
        else:
            hits = [row for row in contrasts if row.compartment == compartment]
        return ToolResult(
            tool="sender_pathway_contrast",
            source_file=str(CELLCHAT_CSV.name),
            filters={
                "source_compartment": compartment,
                "exclusive_ephys": exclusive_ephys,
            },
            n_hits=len(hits),
            rows=[
                {
                    "pathway": row.pathway,
                    "Ephys_1": row.ephys_1,
                    "Ephys_2": row.ephys_2,
                    "delta_E2_minus_E1": row.delta_e2_minus_e1,
                    "status": (
                        "Ephys_2_only"
                        if row.ephys_1 == 0
                        else "Ephys_1_only"
                        if row.ephys_2 == 0
                        else "shared"
                    ),
                }
                for row in hits
            ],
            note="Counts use the pathway's source compartment and source Ephys label.",
        )

    def call(self, name: str, **kwargs) -> ToolResult:
        fn = getattr(self, name, None)
        if fn is None:
            raise KeyError(f"Unknown tool: {name}")
        return fn(**kwargs)


def infer_query_slots(question: str) -> dict:
    text = question.strip()
    lowered = text.lower()
    slots: dict = {"genes": [], "pair": None, "idh": None, "celltype": None, "ephys": None, "compartment": None}

    pair = _PAIR_RE.search(text.replace("→", "_").replace("->", "_"))
    if pair:
        slots["pair"] = (pair.group(1), pair.group(2))

    stop = {
        "EPHYS",
        "CELL",
        "TUMOR",
        "CELLCHAT",
        "DEG",
        "DEGS",
        "IDH",
        "WT",
        "MUTANT",
        "MHC",
        "MHC-II",
        "AMPA",
        "GABA",
        "OPC",
        "TAM",
        "MES",
        "RAG",
        "LLM",
        "CSV",
    }
    genes = []
    for match in _GENE_RE.findall(text):
        if match in stop or match.startswith("EPHYS"):
            continue
        if len(match) < 3:
            continue
        genes.append(match)
    slots["genes"] = genes[:4]
    has_idh_mutant = bool(re.search(r"idh[-_\s]?mutant", lowered))
    has_idh_wt = bool(re.search(r"idh[-_\s]?(?:wt|wild[-_\s]?type)", lowered))
    slots["idh"] = (
        None
        if has_idh_mutant and has_idh_wt
        else normalize_idh(text) if "idh" in lowered else None
    )

    has_ephys_1 = bool(re.search(r"ephys[_\s-]?1", lowered))
    has_ephys_2 = bool(re.search(r"ephys[_\s-]?2", lowered))
    if has_ephys_1 and has_ephys_2:
        slots["ephys"] = None
    elif has_ephys_2:
        slots["ephys"] = "Ephys_2"
    elif has_ephys_1:
        slots["ephys"] = "Ephys_1"

    if any(term in lowered for term in ("t cell", "t-cell", "tcell")):
        slots["celltype"] = "T cell"
        slots["compartment"] = "tcell"
    elif "cycling" in lowered and "opc" in lowered:
        slots["compartment"] = "tumor"
    elif "cycling" in lowered:
        slots["celltype"] = "cycling_tumor"
        slots["compartment"] = "tumor"
    elif "mes" in lowered:
        slots["celltype"] = "MES_like"
        slots["compartment"] = "tumor"
    elif "opc" in lowered:
        slots["celltype"] = "OPC_GABA"
        slots["compartment"] = "tumor"
    elif "myeloid" in lowered or "microglia" in lowered or re.search(r"\btam\b", lowered):
        slots["celltype"] = "TAM"
        slots["compartment"] = "myeloid"
    elif any(term in lowered for term in ("tumor", "glioma")):
        slots["compartment"] = "tumor"
    return slots


def dispatch_tools(question: str, registry: Optional[ToolRegistry] = None) -> list[ToolResult]:
    """Route a natural-language question to one or more table tools."""
    tools = registry or ToolRegistry()
    slots = infer_query_slots(question)
    lowered = question.lower()
    traces: list[ToolResult] = []
    source_role = "source" if any(
        term in lowered for term in (" send", "sent", "sender", "ligand")
    ) else "either"
    wants_sender_contrast = (
        "pathway" in lowered
        and "ephys" in lowered
        and any(term in lowered for term in ("side", "only", "exclusive", "sender", "private"))
        and bool(slots["compartment"])
    )
    nrxn_and_glutamate = "nrxn" in lowered and "glutamate" in lowered
    asks_low_n = bool(re.search(r"\bonly\s+\d+\s+cells?\b", lowered))
    tcell_synaptic_refutation = (
        slots["compartment"] == "tcell"
        and "nrxn" in lowered
        and "refut" in lowered
    )

    if "neuron" in lowered:
        traces.append(tools.annotation_lookup(ask_neurons=True))
    if asks_low_n or any(term in lowered for term in ("n_cells", "how many", "n=6", "n = 6", "trust", "mes")):
        traces.append(
            tools.count_lookup(
                celltype=slots.get("celltype"),
                ephys=slots.get("ephys"),
                low_n=asks_low_n,
            )
        )
    if "mhc-ii" in lowered or "mhc ii" in lowered:
        traces.append(
            tools.cellchat_lookup(
                pathway="MHC-II",
                celltype=slots["celltype"],
                ephys=slots["ephys"],
                compartment=slots["compartment"],
                role=source_role,
            )
        )
    if nrxn_and_glutamate:
        traces.append(
            tools.cellchat_lookup(
                gene="NRXN",
                celltype=slots["celltype"],
                ephys=slots["ephys"],
                compartment=slots["compartment"],
                role=source_role,
            )
        )
        traces.append(
            tools.cellchat_lookup(
                pathway="Glutamate",
                celltype=slots["celltype"],
                ephys=slots["ephys"],
                compartment=slots["compartment"],
                role=source_role,
            )
        )
    elif slots["pair"]:
        traces.append(tools.support_join(ligand=slots["pair"][0], receptor=slots["pair"][1], idh=slots["idh"]))
        traces.append(tools.cellchat_lookup(gene=slots["pair"][0], celltype=slots["celltype"], ephys=slots["ephys"], role=source_role))
    elif slots["genes"]:
        gene = slots["genes"][0]
        traces.append(
            tools.cellchat_lookup(
                gene=gene,
                celltype=slots["celltype"],
                ephys=slots["ephys"],
                compartment=slots["compartment"],
                role=source_role,
            )
        )
        traces.append(tools.deg_lookup(gene=gene, celltype=slots["celltype"], idh=slots["idh"]))
        if "pooled" in lowered:
            traces.append(tools.deg_lookup(gene=gene, celltype=slots["celltype"], pooled=True))
    if wants_sender_contrast:
        exclusive_ephys = (
            slots["ephys"]
            if any(term in lowered for term in ("exclusive", "private", "only")) and "both" not in lowered
            else None
        )
        traces.append(
            tools.sender_pathway_contrast(
                compartment=slots["compartment"],
                exclusive_ephys=exclusive_ephys,
            )
        )
        if slots["compartment"] == "tcell" and exclusive_ephys == "Ephys_2":
            traces.append(
                tools.cellchat_lookup(
                    pathway="EGF",
                    celltype="T cell",
                    ephys="Ephys_2",
                    compartment="tcell",
                    role="source",
                )
            )
    if "glutamate" in lowered and any(term in lowered for term in ("ligand", "sender")):
        traces.append(
            tools.pathway_filter(
                pathway="Glutamate",
                compartment=slots["compartment"],
                ephys=slots["ephys"],
            )
        )
    if tcell_synaptic_refutation:
        traces.append(
            tools.cellchat_lookup(
                gene="AREG",
                celltype="T cell",
                ephys="Ephys_2",
                compartment="tcell",
                role="source",
            )
        )
        traces.append(tools.deg_lookup(gene="AREG", celltype="T cell", idh=None))
    if not wants_sender_contrast and any(
        term in lowered for term in ("synapse", "synaptic", "immune", "glutamate", "nrxn")
    ):
        theme = "immune_synapse" if "immune" in lowered else "neuron_tumor"
        traces.append(
            tools.pathway_filter(
                theme=theme,
                compartment=slots["compartment"],
                ephys=slots["ephys"],
            )
        )
    if not traces:
        traces.append(
            tools.cellchat_lookup(
                celltype=slots["celltype"],
                ephys=slots["ephys"],
                compartment=slots["compartment"],
            )
        )
        traces.append(tools.annotation_lookup(compartment=slots["compartment"]))
    return traces
