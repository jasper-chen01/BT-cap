from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def parse_cell_ephys(label: str) -> tuple[str, str]:
    if " / " not in label:
        return label.strip(), "unknown"
    cell, ephys = label.rsplit(" / ", 1)
    return cell.strip(), ephys.strip()


def compartment_of(cell_type: str) -> str:
    name = cell_type.lower()
    if "t cell" in name:
        return "tcell"
    if "tam" in name or "microglia" in name:
        return "myeloid"
    if "tumor" in name:
        return "tumor"
    return "other"


def normalize_idh(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip().lower().replace("-", "_").replace(" ", "_")
    if "mut" in text:
        return "IDH_Mutant"
    if "wt" in text or "wild" in text:
        return "IDH_WT"
    return value.strip()


@dataclass
class Interaction:
    source_label: str
    target_label: str
    ligand: str
    receptor: str
    interaction_name: str
    pathway: str
    probability: float
    pval: float
    source_cell: str
    source_ephys: str
    target_cell: str
    target_ephys: str
    source_compartment: str
    target_compartment: str
    source_n_cells: int | None = None
    target_n_cells: int | None = None
    themes: list[str] = field(default_factory=list)
    same_ephys: bool = False

    @property
    def flow(self) -> str:
        return f"{self.source_compartment}->{self.target_compartment}"

    @property
    def ephys_axis(self) -> str:
        return f"{self.source_ephys}->{self.target_ephys}"

    def as_sentence(self) -> str:
        theme_txt = ", ".join(self.themes) if self.themes else "unassigned"
        source_n = f", n_cells={self.source_n_cells}" if self.source_n_cells is not None else ""
        target_n = f", n_cells={self.target_n_cells}" if self.target_n_cells is not None else ""
        return (
            f"{self.source_cell} ({self.source_compartment}, {self.source_ephys}{source_n}) "
            f"sends {self.ligand} to {self.receptor} on {self.target_cell} "
            f"({self.target_compartment}, {self.target_ephys}{target_n}). "
            f"Pair {self.interaction_name} in pathway {self.pathway} "
            f"with CellChat probability {self.probability:.4f} (p={self.pval}). "
            f"Biological themes: {theme_txt}."
        ).replace("\u2192", "->")


@dataclass
class CountRow:
    group: str
    n_cells: int
    cell: str
    ephys: str
    compartment: str
    source_file: str


@dataclass
class DegRecord:
    gene: str
    direction: str
    avg_log2fc: float
    p_val_adj: float
    n_ephys_1: int
    n_ephys_2: int
    celltype_id: str
    idh_status: str
    cluster_class: str = ""
    source_file: str = ""


@dataclass
class RAGDocument:
    doc_id: str
    title: str
    text: str
    kind: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    tool: str
    source_file: str
    filters: dict[str, Any]
    n_hits: int
    rows: list[dict[str, Any]]
    note: str = ""

    def as_text(self, *, limit: int = 12) -> str:
        filt = ", ".join(f"{k}={v}" for k, v in self.filters.items() if v not in (None, "", []))
        lines = [
            f"tool={self.tool} file={self.source_file} filters=[{filt or 'none'}] n_hits={self.n_hits}"
        ]
        if self.note:
            lines.append(self.note)
        if not self.rows:
            lines.append("empty result (this is allowed; do not invent a row)")
            return "\n".join(lines)
        for row in self.rows[:limit]:
            parts = [f"{k}={v}" for k, v in row.items()]
            lines.append("  " + "; ".join(parts))
        if self.n_hits > limit:
            lines.append(f"  ... {self.n_hits - limit} more rows omitted")
        return "\n".join(lines)
