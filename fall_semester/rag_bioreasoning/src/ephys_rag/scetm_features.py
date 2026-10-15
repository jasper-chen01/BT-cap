"""scETM gene-embedding features for the gene-set rankers.

Level 1: emb_00..emb_{d-1}, the gene's rho vector. Learned without labels, same for every
         cell type x IDH row of a gene.
Level 2: knn_frac_<set>, the fraction of the gene's k nearest neighbours (cosine on rho) that
         are in the label set. This uses labels, so it is rebuilt inside every CV fold with
         only that fold's training genes as candidate neighbours, and a gene is never its own
         neighbour.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ephys_rag.config import KNOWLEDGE_DIR

DEFAULT_EMBEDDINGS = KNOWLEDGE_DIR / "gene_embeddings" / "scetm_gene_embeddings_d32.csv"
KNN_FEATURE = {"tumor_synaptic": "knn_frac_synaptic", "immune_synapse": "knn_frac_immune"}


def load_embeddings(path: Path = DEFAULT_EMBEDDINGS) -> pd.DataFrame:
    """Gene-indexed rho matrix (columns emb_00..)."""
    table = pd.read_csv(path)
    cols = [c for c in table.columns if c.startswith("emb_")]
    if "gene" not in table or not cols:
        raise ValueError(f"{path}: expected a 'gene' column and emb_* columns.")
    return table.drop_duplicates("gene").set_index("gene")[cols].astype(float)


class NeighbourIndex:
    """Cosine neighbours among a fixed gene list, precomputed once and filtered per fold."""

    def __init__(self, emb: pd.DataFrame, genes: Iterable[str], k: int = 20, candidates: int = 300):
        self.genes = np.array([g for g in dict.fromkeys(genes) if g in emb.index])
        self.k = k
        unit = emb.loc[self.genes].to_numpy(dtype=np.float32)
        self.unit = unit / np.clip(np.linalg.norm(unit, axis=1, keepdims=True), 1e-12, None)
        n = len(self.genes)
        m = min(candidates, n - 1)
        self.order = np.empty((n, m), dtype=np.int32)
        for start in range(0, n, 1024):
            stop = min(start + 1024, n)
            sims = self.unit[start:stop] @ self.unit.T
            sims[np.arange(stop - start), np.arange(start, stop)] = -np.inf
            top = np.argpartition(-sims, m - 1, axis=1)[:, :m]
            ranks = np.take_along_axis(sims, top, axis=1)
            self.order[start:stop] = np.take_along_axis(top, np.argsort(-ranks, axis=1), axis=1)

    def fraction(self, labelled: np.ndarray, pool: np.ndarray) -> np.ndarray:
        """Per gene: share of its k nearest pool genes (self excluded) that are labelled."""
        in_pool = pool[self.order]
        taken = in_pool & (np.cumsum(in_pool, axis=1) <= self.k)
        n_taken = taken.sum(axis=1)
        frac = (labelled[self.order] & taken).sum(axis=1) / np.maximum(n_taken, 1)
        available = int(pool.sum()) - pool.astype(int)
        for i in np.flatnonzero(n_taken < np.minimum(self.k, available)):
            sims = self.unit @ self.unit[i]
            sims[i] = -np.inf
            sims[~pool] = -np.inf
            nearest = np.argsort(-sims)[: self.k]
            nearest = nearest[np.isfinite(sims[nearest])]
            frac[i] = labelled[nearest].mean() if len(nearest) else np.nan
        return frac.astype(float)

    def feature(self, row_genes: np.ndarray, label_genes: set[str], pool_genes: set[str]) -> np.ndarray:
        """knn fraction mapped onto rows; NaN for genes without an embedding."""
        labelled = np.isin(self.genes, list(label_genes))
        pool = np.isin(self.genes, list(pool_genes))
        lookup = pd.Series(self.fraction(labelled, pool), index=self.genes)
        return pd.Series(row_genes).map(lookup).to_numpy(dtype=float)
