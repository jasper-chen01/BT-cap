"""Copy existing Weeks 1–5 artifacts into the portal runtime layout."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil


TABLE_FILES = (
    "cellchat_ephys_plus_celltype/all_significant_interactions.csv",
    "cellchat_ephys_plus_celltype/cellchat_group_counts.csv",
    "glioma_compartment_ephys_clustering/glioma_final_cell_id_celltype_ephys_annotation.csv",
    "glioma_compartment_ephys_clustering/glioma_tumor_tcell_tam_ephys_counts.csv",
    "within_cluster_ephys_DEGs/combined_Ephys2_vs_Ephys1_significant_DEGs_by_cluster.csv",
    "within_cluster_ephys_DEGs/DEG_summary_by_cluster.csv",
    "within_cluster_ephys_DEGs_by_IDH/combined_Ephys2_vs_Ephys1_significant_DEGs_by_IDH_and_cluster.csv",
    "within_cluster_ephys_DEGs_by_IDH/DEG_summary_by_IDH_and_cluster.csv",
)


def prepare_reasoning_data(source_root: Path, destination: Path) -> dict[str, Path]:
    """Validate all required inputs, then copy them and return their runtime paths."""
    sources = {name: source_root / "fall_semester/data" / name for name in TABLE_FILES}
    ranker = source_root / "fall_semester/turnin/lgbm_scetm"
    sources["ranker/arms_gene_ranks.csv"] = ranker / "arms_gene_ranks.csv"
    for arm in ("tables", "tables_scetm", "scetm_only"):
        sources[f"ranker/{arm}/genesets_scores.csv"] = ranker / arm / "genesets_scores.csv"
    sources["gene_embeddings/scetm_gene_embeddings_d32.csv"] = (
        source_root / "fall_semester/rag_bioreasoning/knowledge/gene_embeddings/scetm_gene_embeddings_d32.csv"
    )
    missing = [path for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required reasoning inputs:\n" + "\n".join(map(str, missing)))
    prepared = {}
    for name, source in sources.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        prepared[name] = target
    (destination / "runs/portal").mkdir(parents=True, exist_ok=True)
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("."))
    parser.add_argument("--destination", type=Path, default=Path("data/reasoning"))
    args = parser.parse_args()
    try:
        prepared = prepare_reasoning_data(args.source, args.destination)
    except FileNotFoundError as error:
        parser.exit(1, f"{error}\n")
    for path in prepared.values():
        print(path)
    print(args.destination / "runs/portal")


if __name__ == "__main__":
    main()
