from __future__ import annotations

from ephys_rag.ingest import load_pathway_knowledge


def test_pathway_knowledge_is_read_as_utf8_on_windows(tmp_path):
    """Catches dependence on the host code page for biological Unicode text."""
    path = tmp_path / "pathways.yaml"
    path.write_text(
        "pathways:\n  NOTCH:\n    summary: 'DLL1 → NOTCH1'\n",
        encoding="utf-8",
    )

    assert load_pathway_knowledge(path)["NOTCH"]["summary"] == "DLL1 → NOTCH1"
