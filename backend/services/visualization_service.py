"""
Service for visualization workflows (normalization, clustering, UMAP, DE).
"""
from __future__ import annotations

import io
import logging
import urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc

from backend.config import settings
from backend.services.supptable_service import SupptableService


class VisualizationService:
    """Run Scanpy-based visualization and analysis pipelines."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def process_file(
        self,
        file_path: str,
        supptable_url: Optional[str] = None,
        supptable_doc_id: Optional[str] = None,
        de_top_n: int = 15,
        cluster_resolution: float = 1.0,
    ) -> Dict:
        adata = sc.read_h5ad(file_path)

        if adata.var_names.isna().any() or "_index" in adata.var.columns:
            adata.var_names = adata.var.get("_index", adata.var_names)

        if "gene_symbol" not in adata.var.columns:
            adata.var["gene_symbol"] = adata.var_names

        if adata.raw is not None:
            adata = adata.raw.to_adata()

        self._preprocess(adata)

        try:
            sc.tl.leiden(adata, resolution=cluster_resolution, key_added="leiden")
        except ImportError as exc:
            raise ImportError(
                "Leiden clustering requires the 'leidenalg' package. "
                "Install it with `pip install leidenalg python-igraph`."
            ) from exc

        supptable_summary = None
        supptable_meta = {}
        resolved_url = self._resolve_supptable_url(supptable_url, supptable_doc_id)
        if resolved_url:
            try:
                df = self._load_supptable(resolved_url)
                supptable_summary = self._apply_supptable(df, adata)
                supptable_meta["supptable_url"] = resolved_url
            except Exception as exc:
                self.logger.warning("Failed to load supptable: %s", exc)

        points = self._build_umap_points(adata)
        cluster_labels = sorted(adata.obs["leiden"].astype(str).unique().tolist())
        cluster_counts = adata.obs["leiden"].astype(str).value_counts().to_dict()

        de_by_cluster = self._rank_genes(adata, groupby="leiden", top_n=de_top_n)
        de_by_cell_type = None
        if "cell_type" in adata.obs:
            de_by_cell_type = self._rank_genes(
                adata,
                groupby="cell_type",
                top_n=de_top_n,
            )

        return {
            "total_cells": adata.n_obs,
            "umap_points": points,
            "cluster_labels": cluster_labels,
            "cluster_counts": cluster_counts,
            "cell_types": supptable_summary,
            "de_by_cluster": de_by_cluster,
            "de_by_cell_type": de_by_cell_type,
            "metadata": {
                "de_top_n": de_top_n,
                "cluster_resolution": cluster_resolution,
                **supptable_meta,
            },
        }

    def _preprocess(self, adata) -> None:
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
        sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, svd_solver="arpack")
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=min(40, adata.obsm["X_pca"].shape[1]))
        sc.tl.umap(adata)

    def _resolve_supptable_url(
        self,
        supptable_url: Optional[str],
        supptable_doc_id: Optional[str],
    ) -> Optional[str]:
        if supptable_url:
            return supptable_url
        if settings.SUPPTABLE_URL:
            return settings.SUPPTABLE_URL
        doc_id = supptable_doc_id or settings.SUPPTABLE_DOC_ID
        if not doc_id:
            return None
        try:
            service = SupptableService()
            return service.get_supptable_url(doc_id)
        except Exception as exc:
            self.logger.warning("Firestore supptable lookup failed: %s", exc)
            return None

    def _load_supptable(self, url: str) -> pd.DataFrame:
        with urllib.request.urlopen(url) as response:
            content = response.read()
        return pd.read_excel(io.BytesIO(content))

    def _apply_supptable(self, df: pd.DataFrame, adata) -> Optional[List[Dict]]:
        cell_id_col, cell_type_col, score_col = self._detect_columns(df)
        if not cell_id_col or not cell_type_col:
            self.logger.warning("Supptable missing required columns: %s", df.columns.tolist())
            return None

        working = df[[cell_id_col, cell_type_col]].copy()
        working = working.dropna(subset=[cell_id_col])
        working[cell_id_col] = working[cell_id_col].astype(str)

        if score_col:
            working[score_col] = pd.to_numeric(working[score_col], errors="coerce")

        working = working.set_index(cell_id_col)

        cell_type_series = working[cell_type_col].reindex(adata.obs_names)
        adata.obs["cell_type"] = cell_type_series

        if score_col:
            score_series = working[score_col].reindex(adata.obs_names)
            adata.obs["cell_type_score"] = score_series

        valid_types = adata.obs["cell_type"].dropna().astype(str)
        if valid_types.empty:
            return None

        summary = []
        counts = valid_types.value_counts()
        for name, count in counts.items():
            avg_score = None
            if "cell_type_score" in adata.obs:
                scores = adata.obs.loc[adata.obs["cell_type"] == name, "cell_type_score"]
                if scores.notna().any():
                    avg_score = float(scores.mean())
            summary.append(
                {
                    "name": str(name),
                    "count": int(count),
                    "avg_score": avg_score,
                }
            )
        return summary

    def _detect_columns(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        normalized = {self._normalize(col): col for col in df.columns}

        def find(candidates: List[str]) -> Optional[str]:
            for candidate in candidates:
                key = self._normalize(candidate)
                if key in normalized:
                    return normalized[key]
            return None

        cell_id_col = find(["cell_id", "cellid", "cell", "barcode", "cellbarcode"])
        cell_type_col = find(["cell_type", "celltype", "type", "celltypeannotation"])
        score_col = find(["score", "confidence", "probability", "cell_score"])

        return cell_id_col, cell_type_col, score_col

    def _normalize(self, value: str) -> str:
        return (
            str(value)
            .lower()
            .strip()
            .replace(" ", "")
            .replace("_", "")
            .replace("-", "")
        )

    def _build_umap_points(self, adata) -> List[Dict]:
        coords = adata.obsm["X_umap"]
        points = []
        cell_types = adata.obs.get("cell_type") if "cell_type" in adata.obs else None
        scores = adata.obs.get("cell_type_score") if "cell_type_score" in adata.obs else None
        clusters = adata.obs["leiden"].astype(str)

        for idx, cell_id in enumerate(adata.obs_names):
            cell_type = None
            score = None
            if cell_types is not None:
                value = cell_types.iloc[idx]
                if pd.notna(value):
                    cell_type = str(value)
            if scores is not None:
                value = scores.iloc[idx]
                score = self._safe_float(value)
            points.append(
                {
                    "cell_id": str(cell_id),
                    "x": float(coords[idx, 0]),
                    "y": float(coords[idx, 1]),
                    "cluster": str(clusters.iloc[idx]),
                    "cell_type": cell_type,
                    "score": score,
                }
            )
        return points

    def _rank_genes(self, adata, groupby: str, top_n: int) -> Optional[List[Dict]]:
        if groupby not in adata.obs:
            return None
        series = adata.obs[groupby]
        if series.dropna().nunique() < 2:
            return None

        sc.tl.rank_genes_groups(adata, groupby=groupby, method="wilcoxon", use_raw=False)
        result = adata.uns.get("rank_genes_groups", {})
        names = result.get("names")
        if names is None:
            return None

        groups = list(names.dtype.names) if hasattr(names, "dtype") and names.dtype.names else []
        if not groups:
            return None

        output = []
        scores = result.get("scores")
        logfold = result.get("logfoldchanges")
        pvals_adj = result.get("pvals_adj")

        for group in groups:
            genes = [str(gene) for gene in np.asarray(names[group])[:top_n].tolist()]
            group_scores = None
            group_logfold = None
            group_pvals = None

            if scores is not None:
                group_scores = [
                    self._safe_float(x) for x in np.asarray(scores[group])[:top_n].tolist()
                ]
            if logfold is not None:
                group_logfold = [
                    self._safe_float(x) for x in np.asarray(logfold[group])[:top_n].tolist()
                ]
            if pvals_adj is not None:
                group_pvals = [
                    self._safe_float(x) for x in np.asarray(pvals_adj[group])[:top_n].tolist()
                ]

            output.append(
                {
                    "group": str(group),
                    "genes": genes,
                    "scores": group_scores,
                    "logfoldchanges": group_logfold,
                    "pvals_adj": group_pvals,
                }
            )

        return output

    def _safe_float(self, value) -> Optional[float]:
        if value is None or pd.isna(value):
            return None
        try:
            casted = float(value)
        except (TypeError, ValueError):
            return None
        return casted if np.isfinite(casted) else None

