from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TABLES = (
    "cellchat_ephys_plus_celltype/all_significant_interactions.csv",
    "cellchat_ephys_plus_celltype/cellchat_group_counts.csv",
    "glioma_compartment_ephys_clustering/glioma_final_cell_id_celltype_ephys_annotation.csv",
    "glioma_compartment_ephys_clustering/glioma_tumor_tcell_tam_ephys_counts.csv",
    "within_cluster_ephys_DEGs/combined_Ephys2_vs_Ephys1_significant_DEGs_by_cluster.csv",
    "within_cluster_ephys_DEGs/DEG_summary_by_cluster.csv",
    "within_cluster_ephys_DEGs_by_IDH/combined_Ephys2_vs_Ephys1_significant_DEGs_by_IDH_and_cluster.csv",
    "within_cluster_ephys_DEGs_by_IDH/DEG_summary_by_IDH_and_cluster.csv",
)
FILES = {name: f"fall_semester/data/{name}" for name in TABLES}
FILES.update({
    "ranker/arms_gene_ranks.csv": "fall_semester/turnin/lgbm_scetm/arms_gene_ranks.csv",
    **{f"ranker/{arm}/genesets_scores.csv": f"fall_semester/turnin/lgbm_scetm/{arm}/genesets_scores.csv"
       for arm in ("tables", "tables_scetm", "scetm_only")},
    "gene_embeddings/scetm_gene_embeddings_d32.csv":
        "fall_semester/rag_bioreasoning/knowledge/gene_embeddings/scetm_gene_embeddings_d32.csv",
})


def load_module(relative_path):
    path = ROOT / relative_path
    assert path.is_file(), f"Missing implementation: {relative_path}"
    spec = importlib.util.spec_from_file_location("runtime_test_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_reasoning_data_copies_required_files(tmp_path):
    source = tmp_path / "source"
    for output, relative in FILES.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture:{output}")
    destination = tmp_path / "reasoning"
    prepared = load_module("scripts/prepare_reasoning_data.py").prepare_reasoning_data(source, destination)
    assert set(prepared.values()) == {destination / name for name in FILES}
    for name in FILES:
        assert (destination / name).read_text() == f"fixture:{name}"
    assert (destination / "runs/portal").is_dir()


def test_prepare_reasoning_data_lists_all_missing_sources(tmp_path):
    source = tmp_path / "source"
    existing = source / next(iter(FILES.values()))
    existing.parent.mkdir(parents=True)
    existing.write_text("existing")
    destination = tmp_path / "reasoning"
    prepare = load_module("scripts/prepare_reasoning_data.py").prepare_reasoning_data
    with pytest.raises(FileNotFoundError) as error:
        prepare(source, destination)
    for relative in list(FILES.values())[1:]:
        assert str(source / relative) in str(error.value)
    assert not destination.exists()


@pytest.mark.parametrize("preferred", ["preferred", "", "  "])
def test_config_prefers_ephys_rag_data_dir(tmp_path, monkeypatch, preferred):
    selected = tmp_path / (preferred if preferred.strip() else "legacy")
    monkeypatch.setenv("EPHYS_RAG_DATA_DIR", str(selected) if preferred.strip() else preferred)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "legacy"))
    for name in ("RANKER_DIR", "ARMS_GENE_RANKS_CSV", "SCETM_GENE_EMBEDDINGS_CSV",
                 "NOMINATE_SCORES_CSV", "EPHYS_RAG_RUNS_DIR", "TABLES_SCORES_CSV",
                 "TABLES_SCETM_SCORES_CSV", "SCETM_ONLY_SCORES_CSV"):
        monkeypatch.delenv(name, raising=False)
    rag = load_module("fall_semester/rag_bioreasoning/src/ephys_rag/config.py")
    assert rag.DATA_DIR == selected
    ranker = selected / "ranker" if preferred.strip() else ROOT / "fall_semester/turnin/lgbm_scetm"
    assert rag.RANKER_DIR == ranker
    assert rag.ARMS_GENE_RANKS_CSV == ranker / "arms_gene_ranks.csv"
    assert rag.ARM_SCORE_FILES == {
        arm: ranker / arm / "genesets_scores.csv"
        for arm in ("tables", "tables_scetm", "scetm_only")
    }
    assert rag.NOMINATE_SCORES_CSV == rag.ARM_SCORE_FILES["tables_scetm"]
    embeddings = selected if preferred.strip() else ROOT / "fall_semester/rag_bioreasoning/knowledge"
    assert rag.SCETM_GENE_EMBEDDINGS_CSV == embeddings / "gene_embeddings/scetm_gene_embeddings_d32.csv"
    backend = load_module("backend/config.py")
    assert backend.settings.EPHYS_RAG_DATA_DIR == selected
    assert backend.settings.EPHYS_RAG_RUNS_DIR == selected / "runs/portal"


@pytest.mark.parametrize("ranker_override", [False, True])
def test_legacy_data_dir_preserves_tracked_nomination_scores(monkeypatch, tmp_path, ranker_override):
    monkeypatch.setenv("EPHYS_RAG_DATA_DIR", "")
    monkeypatch.setenv("DATA_DIR", str(ROOT / "fall_semester/data"))
    monkeypatch.setenv("TABLES_SCETM_SCORES_CSV", "")
    monkeypatch.setenv("RANKER_DIR", "")
    expected = ROOT / "fall_semester/turnin/lgbm_scetm/tables_scetm/genesets_scores.csv"
    if ranker_override:
        expected = tmp_path / "custom-ranker/tables_scetm/genesets_scores.csv"
        expected.parent.mkdir(parents=True)
        expected.write_text("gene,score\nNRXN1,0.8\n")
        monkeypatch.setenv("RANKER_DIR", str(tmp_path / "custom-ranker"))
    rag = load_module("fall_semester/rag_bioreasoning/src/ephys_rag/config.py")
    assert rag.NOMINATE_SCORES_CSV == expected
    assert "NRXN1" in rag.NOMINATE_SCORES_CSV.read_text()
