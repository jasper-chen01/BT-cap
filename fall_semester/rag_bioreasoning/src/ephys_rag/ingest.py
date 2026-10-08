from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

from ephys_rag.config import (
    ANNOTATION_COUNTS_CSV,
    CELLCHAT_COUNTS_CSV,
    CELLCHAT_CSV,
    DATA_DIR,
    DEG_IDH_CSV,
    DEG_IDH_SUMMARY_CSV,
    DEG_POOLED_CSV,
    DEG_POOLED_SUMMARY_CSV,
    HYPOTHESES_YAML,
    PATHWAY_BIOLOGY_YAML,
)
from ephys_rag.schema import CountRow, DegRecord, Interaction, compartment_of, parse_cell_ephys


def load_pathway_knowledge(path: Path = PATHWAY_BIOLOGY_YAML) -> dict:
    with path.open() as handle:
        payload = yaml.safe_load(handle) or {}
    return payload.get("pathways", {})


def load_hypotheses(path: Path = HYPOTHESES_YAML) -> list[dict]:
    with path.open() as handle:
        payload = yaml.safe_load(handle) or {}
    return payload.get("hypotheses", [])


@lru_cache(maxsize=1)
def load_cellchat_counts(path: str | None = None) -> list[CountRow]:
    csv_path = Path(path) if path else CELLCHAT_COUNTS_CSV
    frame = pd.read_csv(csv_path)
    rows: list[CountRow] = []
    for rec in frame.to_dict(orient="records"):
        label = str(rec["ephys_plus_celltype"])
        cell, ephys = parse_cell_ephys(label)
        rows.append(
            CountRow(
                group=label,
                n_cells=int(rec["n_cells"]),
                cell=cell,
                ephys=ephys,
                compartment=compartment_of(cell),
                source_file=str(csv_path.relative_to(DATA_DIR) if csv_path.is_relative_to(DATA_DIR) else csv_path),
            )
        )
    return rows


@lru_cache(maxsize=1)
def load_annotation_counts(path: str | None = None) -> list[CountRow]:
    csv_path = Path(path) if path else ANNOTATION_COUNTS_CSV
    frame = pd.read_csv(csv_path)
    rows: list[CountRow] = []
    for rec in frame.to_dict(orient="records"):
        cell = str(rec["celltype_id"])
        ephys = str(rec["ephys_cluster_id"])
        compartment = str(rec["compartment"])
        label = f"{cell} / {ephys}"
        rows.append(
            CountRow(
                group=label,
                n_cells=int(rec["n_cells"]),
                cell=cell,
                ephys=ephys,
                compartment=compartment,
                source_file=str(csv_path.relative_to(DATA_DIR) if csv_path.is_relative_to(DATA_DIR) else csv_path),
            )
        )
    return rows


def _count_lookup_map() -> dict[str, int]:
    return {row.group: row.n_cells for row in load_cellchat_counts()}


@lru_cache(maxsize=1)
def load_interactions(path: str | None = None) -> list[Interaction]:
    csv_path = Path(path) if path else CELLCHAT_CSV
    knowledge = load_pathway_knowledge()
    frame = pd.read_csv(csv_path)
    required = {
        "source",
        "target",
        "ligand",
        "receptor",
        "interaction_name",
        "pathway",
        "prob",
        "pval",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Interaction table missing columns: {sorted(missing)}")
    counts = _count_lookup_map()

    rows: list[Interaction] = []
    for rec in frame.to_dict(orient="records"):
        source_cell, source_ephys = parse_cell_ephys(str(rec["source"]))
        target_cell, target_ephys = parse_cell_ephys(str(rec["target"]))
        pathway = str(rec["pathway"])
        themes = list(knowledge.get(pathway, {}).get("themes", []))
        rows.append(
            Interaction(
                source_label=str(rec["source"]),
                target_label=str(rec["target"]),
                ligand=str(rec["ligand"]),
                receptor=str(rec["receptor"]),
                interaction_name=str(rec["interaction_name"]),
                pathway=pathway,
                probability=float(rec["prob"]),
                pval=float(rec["pval"]),
                source_cell=source_cell,
                source_ephys=source_ephys,
                target_cell=target_cell,
                target_ephys=target_ephys,
                source_compartment=compartment_of(source_cell),
                target_compartment=compartment_of(target_cell),
                source_n_cells=counts.get(str(rec["source"])),
                target_n_cells=counts.get(str(rec["target"])),
                themes=themes,
                same_ephys=source_ephys == target_ephys,
            )
        )
    return rows


def _deg_records(frame: pd.DataFrame, source_file: str, default_idh: str = "pooled") -> list[DegRecord]:
    records: list[DegRecord] = []
    for rec in frame.to_dict(orient="records"):
        records.append(
            DegRecord(
                gene=str(rec["gene"]),
                direction=str(rec.get("direction", "")),
                avg_log2fc=float(rec.get("avg_log2FC", rec.get("avg_log2fc", 0.0))),
                p_val_adj=float(rec.get("p_val_adj", rec.get("p_val", 1.0))),
                n_ephys_1=int(rec.get("n_Ephys_1", 0) or 0),
                n_ephys_2=int(rec.get("n_Ephys_2", 0) or 0),
                celltype_id=str(rec.get("celltype_id", "")),
                idh_status=str(rec.get("IDH_status", default_idh)),
                cluster_class=str(rec.get("cluster_class", rec.get("compartment", ""))),
                source_file=source_file,
            )
        )
    return records


@lru_cache(maxsize=1)
def load_degs_idh(path: str | None = None) -> list[DegRecord]:
    csv_path = Path(path) if path else DEG_IDH_CSV
    rel = str(csv_path.relative_to(DATA_DIR) if csv_path.is_relative_to(DATA_DIR) else csv_path)
    return _deg_records(pd.read_csv(csv_path), rel)


@lru_cache(maxsize=1)
def load_degs_pooled(path: str | None = None) -> list[DegRecord]:
    csv_path = Path(path) if path else DEG_POOLED_CSV
    rel = str(csv_path.relative_to(DATA_DIR) if csv_path.is_relative_to(DATA_DIR) else csv_path)
    return _deg_records(pd.read_csv(csv_path), rel, default_idh="pooled")


@lru_cache(maxsize=1)
def load_deg_idh_summary(path: str | None = None) -> pd.DataFrame:
    csv_path = Path(path) if path else DEG_IDH_SUMMARY_CSV
    return pd.read_csv(csv_path)


@lru_cache(maxsize=1)
def load_deg_pooled_summary(path: str | None = None) -> pd.DataFrame:
    csv_path = Path(path) if path else DEG_POOLED_SUMMARY_CSV
    return pd.read_csv(csv_path)


def file_inventory() -> list[dict]:
    """Week-1 data dictionary helper: files that exist under DATA_DIR."""
    inventory = []
    for path in sorted(DATA_DIR.rglob("*.csv")):
        rel = path.relative_to(DATA_DIR)
        try:
            n_rows = sum(1 for _ in path.open()) - 1
        except OSError:
            n_rows = None
        inventory.append(
            {
                "file": str(rel),
                "n_rows": n_rows,
                "bytes": path.stat().st_size,
            }
        )
    return inventory
