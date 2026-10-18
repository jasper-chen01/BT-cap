"""Read-only access to prepared held-out scores and scETM neighbours.

Context scores and gene-level scores/ranks are different quantities. Context
rows retain the original wide CSV fields; aliases identify the selected program.
Ranked rows use the saved gene-level score/rank and a context satisfying every
requested evidence filter. No model is fit or score imputed here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ephys_rag import config
from ephys_rag.schema import normalize_idh

MODELS = ("tumor_synaptic", "immune_synapse")
ARMS = ("tables", "tables_scetm", "scetm_only")


def _model(value: str) -> str:
    value = {"tumor": "tumor_synaptic", "immune": "immune_synapse"}.get(value, value)
    if value not in MODELS:
        raise ValueError(f"Unknown model: {value}; expected one of {MODELS}")
    return value


def _flag(value: object) -> bool:
    return False if value is None or pd.isna(value) else bool(int(value))


def _records(frame: pd.DataFrame) -> list[dict]:
    # JSON cannot represent NaN. None keeps absence distinct from probability 0.
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")


class NominationStore:
    def __init__(self, arm_scores: dict[str, pd.DataFrame],
                 gene_ranks: pd.DataFrame | None = None,
                 embeddings: pd.DataFrame | None = None):
        self.arm_scores = {}
        for arm, original in arm_scores.items():
            if arm not in ARMS:
                raise ValueError(f"Unknown arm: {arm}")
            frame = original.copy()
            frame["gene"] = frame["gene"].str.upper()
            for model in MODELS:
                column = f"{model.split('_')[0]}_rank_in_context"
                ranks = frame.groupby(["celltype_id", "IDH_status"])[f"p_{model}"].rank(
                    method="min", ascending=False, na_option="keep")
                # Preserve saved values, filling only absent context ranks.
                frame[column] = frame[column].fillna(ranks) if column in frame else ranks
            self.arm_scores[arm] = frame
        self.gene_ranks = gene_ranks.copy() if gene_ranks is not None else pd.DataFrame()
        if not self.gene_ranks.empty:
            self.gene_ranks["gene"] = self.gene_ranks["gene"].str.upper()
        self._rank_records = {r["gene"]: r for r in _records(self.gene_ranks)}
        self._labels: dict[str, dict[str, bool]] = {}
        for frame in [self.gene_ranks, *self.arm_scores.values()]:
            for rec in _records(frame):
                labels = self._labels.setdefault(rec["gene"], {})
                for name in ("in_tumor_synaptic", "in_immune_synapse", "in_published_33", "housekeeping"):
                    labels[name] = labels.get(name, False) or _flag(rec.get(name))
        self._embedding_genes = np.array([], dtype=str)
        self._vectors = np.empty((0, 0))
        if embeddings is not None:
            self._embedding_genes = embeddings["gene"].str.upper().to_numpy()
            columns = [name for name in embeddings if name.startswith("emb_")]
            if not columns:
                raise ValueError("No emb_ columns in scETM embeddings")
            vectors = embeddings[columns].to_numpy(dtype=float)
            if not np.isfinite(vectors).all():
                raise ValueError("Non-finite scETM embedding")
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            self._vectors = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0)

    @classmethod
    def from_config(cls) -> "NominationStore":
        paths = [*config.ARM_SCORE_FILES.values(), config.ARMS_GENE_RANKS_CSV,
                 config.SCETM_GENE_EMBEDDINGS_CSV]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing nomination runtime data; run prepare_reasoning_data.py: "
                                    + ", ".join(missing))
        return cls({arm: pd.read_csv(path) for arm, path in config.ARM_SCORE_FILES.items()},
                   pd.read_csv(config.ARMS_GENE_RANKS_CSV),
                   pd.read_csv(config.SCETM_GENE_EMBEDDINGS_CSV))

    def _row(self, rec: dict, *, model: str, arm: str) -> dict:
        row = dict(rec)
        saved = self._rank_records.get(rec["gene"], {})
        labels = self._labels.get(rec["gene"], {})
        row.update(model=model, arm=arm, score=rec[f"p_{model}"],
                   gene_rank=saved.get(f"gene_rank_{model}__{arm}"),
                   gene_score=saved.get(f"p_{model}__{arm}"),
                   rank_in_context=rec.get(f"{model.split('_')[0]}_rank_in_context"),
                   why=rec.get(f"why_{model}"), scoring_folds=rec.get(f"folds_{model}"),
                   min_n=rec.get("ctx_min_n"), passes_three_table=_flag(rec.get("rule_pass")),
                   in_label_set=labels.get(f"in_{model}", False),
                   in_syngo=labels.get("in_tumor_synaptic", False),
                   in_go_0001772=labels.get("in_immune_synapse", False))
        row.update(labels)
        return row

    def gene_rows(self, symbol: str) -> list[dict]:
        symbol = symbol.strip().upper()
        if symbol not in self._labels:
            raise KeyError(symbol)
        rows = []
        for arm, frame in self.arm_scores.items():
            for rec in _records(frame[frame["gene"] == symbol]):
                for model in MODELS:
                    if rec[f"p_{model}"] is not None:
                        rows.append(self._row(rec, model=model, arm=arm))
        return rows

    def ranked_rows(self, *, model: str, arm: str, exclude_housekeeping: bool = True,
                    novel_only: bool = False, rule_pass: bool | None = None,
                    min_n: int = 0, limit: int = 15) -> list[dict]:
        model = _model(model)
        if arm not in self.arm_scores:
            raise ValueError(f"Unknown arm: {arm}")
        if min_n < 0 or limit < 0:
            raise ValueError("min_n and limit must be nonnegative")
        frame = self.arm_scores[arm].dropna(subset=[f"p_{model}"])
        if exclude_housekeeping:
            frame = frame[~frame["gene"].map(lambda gene: self._labels[gene].get("housekeeping", False))]
        if novel_only:
            frame = frame[frame["novel"].map(_flag)]
        if rule_pass is not None:
            frame = frame[frame["rule_pass"].map(_flag) == rule_pass]
        frame = frame[frame["ctx_min_n"] >= min_n]
        frame = frame.sort_values([f"p_{model}", "gene", "celltype_id", "IDH_status"],
                                  ascending=[False, True, True, True], kind="stable")
        rows = []
        for rec in _records(frame.drop_duplicates("gene")):
            saved = self._rank_records.get(rec["gene"], {})
            score = saved.get(f"p_{model}__{arm}")
            if score is None:
                continue
            row = self._row(rec, model=model, arm=arm)
            row.update(context_score=row["score"], score=score)
            rows.append(row)
        rows.sort(key=lambda row: (-row["score"],
                  row["gene_rank"] if row["gene_rank"] is not None else float("inf"), row["gene"]))
        return rows[:limit]

    def nominate_rows(self, *, gene: str | None = None, celltype: str | None = None,
                      idh: str | None = None, order_by: str = "tumor",
                      exclude_syngo: bool = False, limit: int | None = 15) -> list[dict]:
        if order_by not in {"tumor", "immune"}:
            raise ValueError("order_by must be 'tumor' or 'immune'")
        model = _model(order_by)
        frame = self.arm_scores["tables_scetm"]
        if gene:
            frame = frame[frame["gene"] == gene.strip().upper()]
        if celltype:
            frame = frame[frame["celltype_id"].str.contains(celltype, case=False, regex=False)]
        idh_norm = normalize_idh(idh)
        if idh_norm:
            frame = frame[frame["IDH_status"] == idh_norm]
        if exclude_syngo:
            frame = frame[~frame["in_tumor_synaptic"].map(_flag)]
        if gene is None:
            frame = frame.dropna(subset=[f"p_{model}"])
        # Keep the Week 5 quicksort tie ordering for existing tool traces.
        frame = frame.sort_values(f"p_{model}", ascending=False, na_position="last")
        return _records(frame if limit is None else frame.head(limit))

    def neighbors(self, symbol: str, *, model: str, k: int = 20) -> list[dict]:
        model = _model(model)
        if not 1 <= k <= 100:
            raise ValueError("k must be between 1 and 100")
        symbol = symbol.strip().upper()
        indices = np.flatnonzero(self._embedding_genes == symbol)
        if not len(indices):
            raise KeyError(symbol)
        similarities = self._vectors @ self._vectors[indices[0]]
        candidates = [i for i in range(len(similarities)) if self._embedding_genes[i] != symbol]
        candidates.sort(key=lambda i: (-similarities[i], self._embedding_genes[i]))
        return [{"gene": str(self._embedding_genes[i]), "similarity": float(similarities[i]),
                 "in_label_set": self._labels.get(str(self._embedding_genes[i]), {}).get(f"in_{model}", False)}
                for i in candidates[:k]]
