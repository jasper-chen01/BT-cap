from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parents[1]


def resolve_data_dir() -> Path:
    env = os.getenv("DATA_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    sibling = ROOT.parent / "data"
    if sibling.exists():
        return sibling
    local = ROOT / "data"
    if local.exists():
        return local
    raise FileNotFoundError(
        "Could not find the fall_semester data drop. "
        "Set DATA_DIR to the folder that contains cellchat_ephys_plus_celltype/."
    )


DATA_DIR = resolve_data_dir()
KNOWLEDGE_DIR = ROOT / "knowledge"
PATHWAY_BIOLOGY_YAML = KNOWLEDGE_DIR / "pathway_biology.yaml"
HYPOTHESES_YAML = KNOWLEDGE_DIR / "hypotheses.yaml"

CELLCHAT_CSV = DATA_DIR / "cellchat_ephys_plus_celltype" / "all_significant_interactions.csv"
CELLCHAT_COUNTS_CSV = DATA_DIR / "cellchat_ephys_plus_celltype" / "cellchat_group_counts.csv"
ANNOTATION_COUNTS_CSV = (
    DATA_DIR / "glioma_compartment_ephys_clustering" / "glioma_tumor_tcell_tam_ephys_counts.csv"
)
ANNOTATION_CELLS_CSV = (
    DATA_DIR / "glioma_compartment_ephys_clustering" / "glioma_final_cell_id_celltype_ephys_annotation.csv"
)
DEG_IDH_CSV = (
    DATA_DIR
    / "within_cluster_ephys_DEGs_by_IDH"
    / "combined_Ephys2_vs_Ephys1_significant_DEGs_by_IDH_and_cluster.csv"
)
DEG_POOLED_CSV = (
    DATA_DIR / "within_cluster_ephys_DEGs" / "combined_Ephys2_vs_Ephys1_significant_DEGs_by_cluster.csv"
)
DEG_IDH_SUMMARY_CSV = DATA_DIR / "within_cluster_ephys_DEGs_by_IDH" / "DEG_summary_by_IDH_and_cluster.csv"
DEG_POOLED_SUMMARY_CSV = DATA_DIR / "within_cluster_ephys_DEGs" / "DEG_summary_by_cluster.csv"

DEFAULT_TOP_K = 10
LOW_N_CELLS = 50

