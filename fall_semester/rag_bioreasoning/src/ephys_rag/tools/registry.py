from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ephys_rag.config import (
    ANNOTATION_CELLS_CSV,
    ANNOTATION_COUNTS_CSV,
    CELLCHAT_COUNTS_CSV,
    CELLCHAT_CSV,
    DEG_IDH_CSV,
    LOW_N_CELLS,
)
from ephys_rag.contrasts import exclusive_pathways, pathway_contrasts, summarize_dataset
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
        source_compartment: str | None = None,
        target_compartment: str | None = None,
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
            if source_compartment and row.source_compartment != source_compartment:
                continue
            if target_compartment and row.target_compartment != target_compartment:
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
                "source_compartment": source_compartment,
                "target_compartment": target_compartment,
                "role": role,
            },
            n_hits=len(hits),
            rows=[
                {
                    "pair": row.interaction_name,
                    "pathway": row.pathway,
                    "receptor": row.receptor,
                    "ligand": row.ligand,
                    "source": row.source_label,
                    "target": row.target_label,
                    "prob": round(row.probability, 4),
                    "pval": row.pval,
                    "source_n": row.source_n_cells,
                    "target_n": row.target_n_cells,
                }
                for row in hits[:limit]
            ],
        )

    def exclusive_pathway_lookup(
        self,
        *,
        compartment: str = "tumor",
        ephys: str = "Ephys_2",
        limit: int = 20,
    ) -> ToolResult:
        contrasts = pathway_contrasts(self.interactions)
        hits = exclusive_pathways(contrasts, compartment=compartment, ephys=ephys)
        hits = sorted(hits, key=lambda item: item.ephys_2 if ephys == "Ephys_2" else item.ephys_1, reverse=True)
        return ToolResult(
            tool="exclusive_pathway_lookup",
            source_file=str(CELLCHAT_CSV.name),
            filters={"compartment": compartment, "ephys": ephys},
            n_hits=len(hits),
            rows=[
                {
                    "pathway": item.pathway,
                    "compartment": item.compartment,
                    "Ephys_1": item.ephys_1,
                    "Ephys_2": item.ephys_2,
                    "delta": item.delta_e2_minus_e1,
                    "exclusive_to": ephys,
                }
                for item in hits[:limit]
            ],
            note=f"Pathways with sender events only in {ephys} for {compartment} (0 on the other side).",
        )

    def pathway_contrast_lookup(
        self,
        *,
        compartment: str | None = None,
        limit: int = 25,
    ) -> ToolResult:
        rows = pathway_contrasts(self.interactions)
        if compartment:
            rows = [item for item in rows if item.compartment == compartment]
        return ToolResult(
            tool="pathway_contrast_lookup",
            source_file=str(CELLCHAT_CSV.name),
            filters={"compartment": compartment},
            n_hits=len(rows),
            rows=[
                {
                    "pathway": item.pathway,
                    "compartment": item.compartment,
                    "Ephys_1": item.ephys_1,
                    "Ephys_2": item.ephys_2,
                    "delta": item.delta_e2_minus_e1,
                    "mean_prob_e1": round(item.mean_prob_e1, 4),
                    "mean_prob_e2": round(item.mean_prob_e2, 4),
                }
                for item in rows[:limit]
            ],
            note="Sender-side pathway counts by Ephys class. Use to see ligand-program flips.",
        )

    def dataset_stats_lookup(self) -> ToolResult:
        summary = summarize_dataset(self.interactions)
        flows = summary["flows"]
        dominant = max(flows, key=flows.get) if flows else None
        return ToolResult(
            tool="dataset_stats_lookup",
            source_file=str(CELLCHAT_CSV.name),
            filters={},
            n_hits=1,
            rows=[
                {
                    "n_interactions": summary["n_interactions"],
                    "n_pathways": summary["n_pathways"],
                    "n_pairs": summary["n_pairs"],
                    "flows": flows,
                    "dominant_flow": dominant,
                    "dominant_flow_n": flows.get(dominant, 0) if dominant else 0,
                    "labeled_identities": summary["labeled_identities"],
                }
            ],
            note="Raw CellChat interaction stats. Dominant flow is the most common source->target compartment.",
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
            rec_gene = rec.gene.upper()
            if rec_gene != gene_u and not rec_gene.startswith(gene_u):
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

    def call(self, name: str, **kwargs) -> ToolResult:
        fn = getattr(self, name, None)
        if fn is None:
            raise KeyError(f"Unknown tool: {name}")
        return fn(**kwargs)


def infer_query_slots(question: str) -> dict:
    text = question.strip()
    lowered = text.lower()
    slots: dict = {
        "genes": [],
        "pair": None,
        "idh": None,
        "celltype": None,
        "ephys": None,
        "compartment": None,
        "pathway": None,
    }

    pair = _PAIR_RE.search(text.replace("→", "_").replace("->", "_").replace("–", "-"))
    if pair:
        slots["pair"] = (pair.group(1), pair.group(2))

    knowledge = load_pathway_knowledge()
    for name in sorted(knowledge, key=len, reverse=True):
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(name.lower())}(?![A-Za-z0-9])", lowered):
            slots["pathway"] = name
            break

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
        "AMPA",
        "GABA",
        "OPC",
        "TAM",
        "MES",
        "RAG",
        "LLM",
        "CSV",
        "WHICH",
        "WHAT",
        "NAME",
        "DOES",
        "ARE",
        "HOW",
        "MANY",
        "THIS",
        "TABLE",
        "BIND",
        "HIT",
        "FLIP",
        "BETWEEN",
        "PRESENT",
        "SOURCE",
        "TARGET",
        "IDENTITY",
        "PROGRAM",
        "SENDERS",
        "SENDER",
        "PATHWAYS",
        "PATHWAY",
        "RECEPTORS",
        "RECEPTOR",
        "HIGHEST",
        "PROBABILITY",
        "DOMINANT",
        "COMPARTMENT",
        "FLOW",
        "SIGNIFICANT",
        "INTERACTIONS",
        "INTERACTION",
        "LIGAND",
        "EXCLUSIVE",
        "NEURONS",
        "NEURON",
        "MYELOID",
        "GLUTAMATE",
    }
    genes = []
    for match in _GENE_RE.findall(text):
        if match in stop or match.startswith("EPHYS"):
            continue
        if slots["pathway"] and match.upper() == slots["pathway"].upper():
            continue
        if len(match) < 3:
            continue
        genes.append(match)
    slots["genes"] = genes[:4]
    slots["idh"] = normalize_idh(text) if ("idh" in lowered) else None

    if re.search(r"ephys[_\s-]?2", lowered):
        slots["ephys"] = "Ephys_2"
    elif re.search(r"ephys[_\s-]?1", lowered):
        slots["ephys"] = "Ephys_1"

    if any(term in lowered for term in ("t cell", "t-cell", "tcell")):
        slots["celltype"] = "T cell"
        slots["compartment"] = "tcell"
    elif "cycling" in lowered:
        slots["celltype"] = "cycling_tumor"
        slots["compartment"] = "tumor"
    elif "mes" in lowered:
        slots["celltype"] = "MES_like"
        slots["compartment"] = "tumor"
    elif "opc" in lowered:
        slots["celltype"] = "OPC_GABA"
        slots["compartment"] = "tumor"
    elif any(term in lowered for term in ("myeloid", "tam", "microglia")):
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

    wants_deg = any(
        term in lowered
        for term in ("deg", "deg-supported", "expression", "ephys_2-high", "ephys_1-high", "ephys-2-high", "ephys-1-high")
    )
    wants_counts = any(
        term in lowered
        for term in (
            "how many",
            "significant interaction",
            "dominant",
            "compartment flow",
            "n_cells",
            "n=6",
            "n = 6",
            "only 6",
            "trust",
            "trusted",
            "large enough",
        )
    )
    both_idh = "idh-mutant" in lowered and ("idh-wt" in lowered or "idh_wt" in lowered or "idh-wild" in lowered)

    if "neuron" in lowered:
        traces.append(tools.annotation_lookup(ask_neurons=True))

    if wants_counts:
        if any(term in lowered for term in ("how many", "significant interaction", "dominant", "compartment flow")):
            traces.append(tools.dataset_stats_lookup())
        cell = slots.get("celltype")
        if "mes" in lowered:
            cell = "MES"
        if "ac-like" in lowered or "ac_like" in lowered:
            cell = "AC_like"
        traces.append(tools.count_lookup(celltype=cell, ephys=slots.get("ephys")))
        if "mes" in lowered or "only 6" in lowered:
            traces.append(tools.annotation_lookup(celltype="MES"))

    if any(term in lowered for term in ("exclusive", "do not", "absent from", "private to")) and slots.get(
        "compartment"
    ):
        ephys = slots.get("ephys") or ("Ephys_1" if "ephys_1" in lowered.replace("-", "_") or "ephys-1" in lowered else "Ephys_2")
        if "ephys_1" in lowered.replace("-", "_") or "ephys-1" in lowered:
            ephys = "Ephys_1"
        traces.append(tools.exclusive_pathway_lookup(compartment=slots["compartment"], ephys=ephys))

    if any(
        term in lowered
        for term in (
            "flip",
            "ligand identity",
            "program that",
            "discriminator",
            "shared scaffold",
            "contrast",
            "refute",
            "enriched",
        )
    ):
        compartment = slots.get("compartment") or "tumor"
        traces.append(tools.pathway_contrast_lookup(compartment=compartment))
        if compartment == "tcell":
            traces.append(tools.pathway_filter(theme="immune_synapse", compartment="tcell"))
            traces.append(tools.cellchat_lookup(pathway="EGF", compartment="tcell"))
        if compartment == "myeloid" and slots.get("ephys"):
            if not any(t.tool == "exclusive_pathway_lookup" for t in traces):
                traces.append(
                    tools.exclusive_pathway_lookup(compartment="myeloid", ephys=slots["ephys"])
                )

    if any(term in lowered for term in ("highest", "highest-probability", "top probability", "self-loop", "self loop")):
        src = slots.get("compartment") or "tumor"
        tgt = src
        if "tumor" in lowered and "myeloid" not in lowered:
            src = tgt = "tumor"
        if "myeloid" in lowered or "self-loop" in lowered or "self loop" in lowered:
            src = tgt = "myeloid"
        traces.append(
            tools.cellchat_lookup(
                source_compartment=src,
                target_compartment=tgt,
                limit=10,
            )
        )

    if "pval" in lowered or "p-value" in lowered or "p value" in lowered:
        traces.append(tools.cellchat_lookup(limit=5))

    if slots["pathway"]:
        traces.append(
            tools.cellchat_lookup(
                pathway=slots["pathway"],
                celltype=slots["celltype"],
                ephys=slots["ephys"],
                compartment=slots["compartment"],
                role="source" if "sender" in lowered else "either",
            )
        )
        traces.append(
            tools.pathway_filter(
                pathway=slots["pathway"],
                compartment=slots["compartment"],
                ephys=slots["ephys"],
            )
        )

    if slots["pair"]:
        traces.append(tools.support_join(ligand=slots["pair"][0], receptor=slots["pair"][1], idh=slots["idh"]))
        traces.append(tools.cellchat_lookup(gene=slots["pair"][0], celltype=slots["celltype"], ephys=slots["ephys"]))
    elif slots["genes"]:
        gene = slots["genes"][0]
        traces.append(
            tools.cellchat_lookup(
                gene=gene,
                celltype=slots["celltype"],
                ephys=slots["ephys"],
                compartment=slots["compartment"],
            )
        )
        if wants_deg or both_idh or "both" in lowered or "support" in lowered:
            if both_idh:
                traces.append(tools.deg_lookup(gene=gene, celltype=slots["celltype"], idh="IDH_Mutant"))
                traces.append(tools.deg_lookup(gene=gene, celltype=slots["celltype"], idh="IDH_WT"))
            else:
                traces.append(tools.deg_lookup(gene=gene, celltype=slots["celltype"], idh=slots["idh"]))
            if "pooled" in lowered:
                traces.append(tools.deg_lookup(gene=gene, celltype=slots["celltype"], pooled=True))
        elif any(term in lowered for term in ("bind", "receptor", "partners", "use here")):
            pass
        else:
            # Default: also surface DEG when a gene is named (Week 2 joins)
            traces.append(tools.deg_lookup(gene=gene, celltype=slots["celltype"], idh=slots["idh"]))

    if wants_deg and slots["genes"] and not any(t.tool == "deg_lookup" for t in traces):
        gene = slots["genes"][0]
        if both_idh:
            traces.append(tools.deg_lookup(gene=gene, celltype=slots["celltype"], idh="IDH_Mutant"))
            traces.append(tools.deg_lookup(gene=gene, celltype=slots["celltype"], idh="IDH_WT"))
        else:
            traces.append(tools.deg_lookup(gene=gene, celltype=slots["celltype"], idh=slots["idh"]))

    if any(term in lowered for term in ("synapse", "synaptic", "immune")) and not any(
        t.tool == "pathway_filter" for t in traces
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
                role="source" if "sender" in lowered else "either",
            )
        )
        traces.append(tools.annotation_lookup(compartment=slots["compartment"]))
    return traces
