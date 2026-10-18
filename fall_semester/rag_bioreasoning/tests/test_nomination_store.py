"""Shared loader contracts; fixtures use the prepared CSV column schema."""
import pandas as pd
import pytest

from ephys_rag.nomination import NominationStore


@pytest.fixture
def store():
    rows = pd.DataFrame([
        dict(gene="ALPHA", celltype_id="cycling_tumor", IDH_status="IDH_Mutant",
             p_tumor_synaptic=.8, p_immune_synapse=.2, in_tumor_synaptic=1,
             in_immune_synapse=0, in_published_33=1, novel=0, housekeeping=0,
             rule_pass=1, ctx_min_n=80, why_tumor_synaptic="emb_00=1 (+.4)",
             folds_tumor_synaptic="seed0:fold2", tumor_rank_in_context=7),
        dict(gene="BETA", celltype_id="cycling_tumor", IDH_status="IDH_Mutant",
             p_tumor_synaptic=.8, p_immune_synapse=.3, in_tumor_synaptic=0,
             in_immune_synapse=1, in_published_33=0, novel=1, housekeeping=0,
             rule_pass=1, ctx_min_n=60),
        dict(gene="BETA", celltype_id="T cell", IDH_status="IDH_WT",
             p_tumor_synaptic=None, p_immune_synapse=.9, in_tumor_synaptic=0,
             in_immune_synapse=1, in_published_33=0, novel=1, housekeeping=0,
             rule_pass=0, ctx_min_n=20),
        dict(gene="RPL5", celltype_id="cycling_tumor", IDH_status="IDH_Mutant",
             p_tumor_synaptic=.95, p_immune_synapse=.4, in_tumor_synaptic=0,
             in_immune_synapse=0, in_published_33=0, novel=1, housekeeping=1,
             rule_pass=0, ctx_min_n=90),
    ])
    ranks = pd.DataFrame([
        {"gene": gene, **{f"p_{model}__{arm}": score
         for arm in ("tables", "tables_scetm", "scetm_only")
         for model in ("tumor_synaptic", "immune_synapse")},
         **{f"gene_rank_{model}__{arm}": rank
         for arm in ("tables", "tables_scetm", "scetm_only")
         for model in ("tumor_synaptic", "immune_synapse")}}
        for gene, score, rank in [("BETA", .8, 22), ("ALPHA", .8, 23), ("RPL5", .95, 1)]
    ])
    embeddings = pd.DataFrame({"gene": ["ALPHA", "BETA", "RPL5", "UNSCORED"],
                               "n_cells_detected": [1, 999, 3, 1],
                               "emb_00": [1., 2., 0., -1.], "emb_01": [0., 0., 1., 0.]})
    return NominationStore({arm: rows.copy() for arm in
                            ("tables", "tables_scetm", "scetm_only")}, ranks, embeddings)


def test_gene_rows_preserve_missing_model_scores(store):
    """An unscored tumor/T-cell combination must never become a zero row."""
    rows = store.gene_rows("beta")
    assert len(rows) == 9
    assert {r["arm"] for r in rows} == {"tables", "tables_scetm", "scetm_only"}
    assert not any(r["model"] == "tumor_synaptic" and r["celltype_id"] == "T cell" for r in rows)
    assert next(r for r in rows if r["celltype_id"] == "T cell")["p_tumor_synaptic"] is None


def test_gene_rows_keep_saved_rank_explanation_and_fold(store):
    """Aliases must not overwrite saved context ranks or explanation/fold strings."""
    row = next(r for r in store.gene_rows("alpha") if r["model"] == "tumor_synaptic")
    assert row["score"] == .8
    assert row["gene_rank"] == 23
    assert row["rank_in_context"] == 7
    assert row["why"] == "emb_00=1 (+.4)"
    assert row["scoring_folds"] == "seed0:fold2"
    assert row["in_label_set"] is True


def ranked(store, **changes):
    options = dict(model="tumor_synaptic", arm="tables_scetm", exclude_housekeeping=False,
                   novel_only=False, rule_pass=None, min_n=0, limit=20)
    options.update(changes)
    return store.ranked_rows(**options)


def test_ranking_preserves_saved_gene_ranks_and_stable_ties(store):
    """Sorting must honor the saved rank for tied probabilities and remain gene-level."""
    rows = ranked(store)
    assert [r["gene"] for r in rows] == ["RPL5", "BETA", "ALPHA"]
    assert [r["gene_rank"] for r in rows] == [1, 22, 23]
    assert ranked(store, limit=1)[0]["score"] == .95


@pytest.mark.parametrize("options,expected", [
    ({"exclude_housekeeping": True}, ["BETA", "ALPHA"]),
    ({"novel_only": True, "exclude_housekeeping": True}, ["BETA"]),
    ({"rule_pass": True, "min_n": 70}, ["ALPHA"]),
    ({"rule_pass": False}, ["RPL5"]),
    ({"min_n": 100}, []),
])
def test_ranking_filters_require_one_scored_supporting_context(store, options, expected):
    """A non-scored or separately matching context cannot satisfy ranking filters."""
    assert [r["gene"] for r in ranked(store, **options)] == expected


def test_ranking_supporting_context_satisfies_all_filters(store):
    row = ranked(store, model="immune_synapse", novel_only=True, exclude_housekeeping=True,
                 rule_pass=True, min_n=50)[0]
    assert row["celltype_id"] == "cycling_tumor"
    assert row["min_n"] == 60
    assert row["passes_three_table"] is True


def test_neighbors_use_only_embedding_columns_and_program_labels(store):
    """Detection counts must not affect cosine similarity; query must be excluded."""
    rows = store.neighbors("alpha", model="immune_synapse", k=3)
    assert [r["gene"] for r in rows] == ["BETA", "RPL5", "UNSCORED"]
    assert [r["similarity"] for r in rows] == pytest.approx([1., 0., -1.])
    assert [r["in_label_set"] for r in rows] == [True, False, False]
    assert len(store.neighbors("ALPHA", model="tumor_synaptic", k=1)) == 1


@pytest.mark.parametrize("model,symbol", [
    ("tumor_synaptic", "ABCC8"),
    ("immune_synapse", "STX11"),
])
def test_neighbors_label_embedding_genes_absent_from_score_tables(store, model, symbol):
    """Canonical positives retain neighbour labels outside the ranking universe."""
    embeddings = pd.DataFrame({"gene": ["QUERY", symbol, "UNLABELED"],
                               "emb_00": [1., 1., 0.], "emb_01": [0., 0., 1.]})
    loaded = NominationStore(store.arm_scores, store.gene_ranks, embeddings)
    assert symbol not in set(store.gene_ranks["gene"])
    assert loaded.neighbors("QUERY", model=model, k=1) == [
        {"gene": symbol, "similarity": 1., "in_label_set": True}]


def test_unknown_gene_raises_lookup_error(store):
    with pytest.raises(KeyError):
        store.gene_rows("MISSING")
    with pytest.raises(KeyError):
        store.neighbors("MISSING", model="tumor_synaptic")


def test_nominate_keeps_missing_gene_score_and_filters(store):
    """Explicit gene lookups retain missing scores; top lists omit unscored rows."""
    rows = store.nominate_rows(gene="beta", celltype="T cell", idh="WT", order_by="tumor",
                               exclude_syngo=False, limit=10)
    assert len(rows) == 1
    assert rows[0]["p_tumor_synaptic"] is None
    rows = store.nominate_rows(gene=None, celltype=None, idh=None, order_by="tumor",
                               exclude_syngo=True, limit=10)
    assert [r["gene"] for r in rows] == ["RPL5", "BETA"]


def test_from_config_loads_prepared_files_once(tmp_path, monkeypatch, store):
    """Configured files must back lookups even after the on-disk files disappear."""
    from ephys_rag import config
    paths = {}
    for arm, frame in store.arm_scores.items():
        paths[arm] = tmp_path / f"{arm}.csv"
        frame.to_csv(paths[arm], index=False)
    ranks_path, emb_path = tmp_path / "ranks.csv", tmp_path / "emb.csv"
    store.gene_ranks.to_csv(ranks_path, index=False)
    pd.DataFrame({"gene": ["ALPHA", "BETA"], "emb_00": [1., 2.]}).to_csv(emb_path, index=False)
    monkeypatch.setattr(config, "ARM_SCORE_FILES", paths)
    monkeypatch.setattr(config, "ARMS_GENE_RANKS_CSV", ranks_path)
    monkeypatch.setattr(config, "SCETM_GENE_EMBEDDINGS_CSV", emb_path)
    loaded = NominationStore.from_config()
    for path in [*paths.values(), ranks_path, emb_path]:
        path.unlink()
    assert loaded.gene_rows("alpha")[0]["score"] == .8
    assert loaded.neighbors("alpha", model="tumor_synaptic", k=1)[0]["gene"] == "BETA"
