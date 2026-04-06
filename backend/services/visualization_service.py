"""
Service for visualization workflows (normalization, clustering, UMAP, DE).
"""
from __future__ import annotations

import io
import json
import logging
import math
import os
import re
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats

from backend.config import settings
from backend.services.supptable_service import SupptableService


def _sanitize_for_json(obj):
    """Replace inf/nan and NumPy scalars so json.dumps / FastAPI responses stay valid."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


class VisualizationService:
    """Run Scanpy-based visualization and analysis pipelines."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._annotation_cache: Optional[Dict] = None

    def process_file(
        self,
        file_path: str,
        supptable_url: Optional[str] = None,
        supptable_doc_id: Optional[str] = None,
        supptable_path: Optional[str] = None,
        embedding_matches_path: Optional[str] = None,
        source_filename: Optional[str] = None,
        de_top_n: int = 15,
        cluster_resolution: float = 1.0,
        use_hvg: bool = True,
        apply_filtering: bool = True,
        selected_program: Optional[str] = None,
        program_name: Optional[str] = None,
    ) -> Dict:
        # Keep these args for compatibility with the route for now,
        # but use only one resolved selected program for metadata/UI convenience.
        requested_program = (selected_program or program_name or "").strip() or None

        adata = sc.read_h5ad(file_path)
        self.logger.info("VAR columns: %s", adata.var.columns.tolist())
        self.logger.info("var_names sample: %s", list(adata.var_names[:10]))

        # If raw exists, use it as the working matrix so score_genes runs on genes, not HVG-subset only.
        if adata.raw is not None:
            adata = adata.raw.to_adata()

        # Standardize gene symbols
        candidate_cols = [
            "gene_symbol",
            "gene_symbols",
            "symbol",
            "symbols",
            "gene_name",
            "gene_names",
            "features",
            "feature_name",
            "var_names",
        ]

        def looks_like_symbol(series: pd.Series) -> float:
            vals = series.dropna().astype(str).head(200).tolist()
            if not vals:
                return 0.0
            ok = 0
            for v in vals:
                v = v.strip()
                if not v:
                    continue
                if v.upper().startswith("ENSG"):
                    continue
                if re.search(r"[A-Za-z]", v):
                    ok += 1
            return ok / max(len(vals), 1)

        best_col = None
        best_score = 0.0
        for col in candidate_cols:
            if col in adata.var.columns:
                score = looks_like_symbol(adata.var[col])
                if score > best_score:
                    best_score = score
                    best_col = col

        ensg_ratio = np.mean(
            [str(x).upper().startswith("ENSG") for x in adata.var_names[:500]]
        )
        if best_col and (ensg_ratio > 0.5 or best_score > 0.5):
            adata.var["gene_symbol"] = adata.var[best_col].astype(str)
        else:
            adata.var["gene_symbol"] = adata.var_names.astype(str)

        adata.var["gene_symbol"] = (
            adata.var["gene_symbol"].astype(str).str.strip().str.upper()
        )
        adata.var_names = pd.Index(adata.var["gene_symbol"].astype(str).values)
        adata.var_names_make_unique()
        has_umap = "X_umap" in adata.obsm
        has_leiden = "leiden" in adata.obs

        if not has_umap or not has_leiden:
            self._preprocess(
                adata,
                use_hvg=use_hvg,
                apply_filtering=apply_filtering,
                compute_umap=not has_umap,
            )

        if not has_leiden:
            try:
                sc.tl.leiden(adata, resolution=cluster_resolution, key_added="leiden")
            except ImportError as exc:
                raise ImportError(
                    "Leiden clustering requires the 'leidenalg' package. "
                    "Install it with `pip install leidenalg python-igraph`."
                ) from exc

        # Load supptable once, apply annotations once, build programs once.
        supptable_summary = None
        supptable_meta = {}
        df = None
        resolved_source = None

        default_supptable = settings.DATA_DIR / "SuppTable1.xlsx"
        if supptable_path:
            resolved_source = supptable_path
        elif default_supptable.is_file():
            resolved_source = str(default_supptable)
        else:
            resolved_source = self._resolve_supptable_url(
                supptable_url, supptable_doc_id
            )

        if resolved_source:
            try:
                df = self._load_supptable(resolved_source)
                supptable_summary = self._apply_supptable(df, adata)
                supptable_meta["supptable_source"] = resolved_source
            except Exception as exc:
                self.logger.warning("Failed to load supptable: %s", exc)
                df = None

        embedding_meta = {}
        resolved_matches_path = self._resolve_embedding_matches_path(
            embedding_matches_path, source_filename
        )
        if resolved_matches_path:
            try:
                applied = self._apply_embedding_matches(resolved_matches_path, adata)
                if applied:
                    embedding_meta["embedding_matches_path"] = resolved_matches_path
            except Exception as exc:
                self.logger.warning("Failed to apply embedding matches: %s", exc)

        # Use embedding predictions as cell_type when supptable (or file) did not provide labels,
        # so UMAP coloring and DE-by-cell-type work without a separate annotate + supptable step.
        self._coalesce_cell_type_obs(adata)

        programs: Dict[str, List[str]] = {}
        program_columns: Dict[str, str] = {}
        program_details: Dict[str, Dict] = {}

        if df is not None:
            programs = self._build_programs_from_supptable(df)
        else:
            self.logger.warning("No supptable loaded; cannot build gene programs.")

        var_set = set(map(str, adata.var_names))
        candidate_programs = sorted(programs.keys())

        for program in candidate_programs:
            gene_list = [str(g).strip().upper() for g in programs[program] if g]
            present = [g for g in gene_list if g in var_set]
            missing = [g for g in gene_list if g not in var_set]

            self.logger.warning(
                "PROGRAM_SCORE: program=%s total_genes=%d present=%d missing=%d present_sample=%s missing_sample=%s",
                program,
                len(gene_list),
                len(present),
                len(missing),
                present[:10],
                missing[:10],
            )

            base_detail = {
                "total_genes": len(gene_list),
                "present_genes": len(present),
                "missing_genes": len(missing),
                "present_sample": present[:10],
                "missing_sample": missing[:10],
                "scored": False,
                "score_stats": None,
                "top_clusters": None,
            }

            if len(present) < 3:
                self.logger.warning(
                    "Skipping program '%s' because fewer than 3 genes are present.",
                    program,
                )
                program_details[program] = base_detail
                continue

            safe_program = re.sub(r"[^A-Za-z0-9_]+", "_", str(program)).strip("_")
            score_name = f"program__{safe_program}"

            sc.tl.score_genes(
                adata,
                gene_list=present,
                score_name=score_name,
                use_raw=False,
            )

            program_columns[program] = score_name

            s = adata.obs[score_name]
            score_stats = {
                "min": self._safe_float(s.min()),
                "max": self._safe_float(s.max()),
                "mean": self._safe_float(s.mean()),
                "median": self._safe_float(s.median()),
                "p01": self._safe_float(s.quantile(0.01)),
                "p05": self._safe_float(s.quantile(0.05)),
                "p95": self._safe_float(s.quantile(0.95)),
                "p99": self._safe_float(s.quantile(0.99)),
            }

            self.logger.warning(
                "PROGRAM_SCORE_STATS: program=%s min=%.4f max=%.4f mean=%.4f median=%.4f p01=%.4f p05=%.4f p95=%.4f p99=%.4f",
                program,
                score_stats["min"],
                score_stats["max"],
                score_stats["mean"],
                score_stats["median"],
                score_stats["p01"],
                score_stats["p05"],
                score_stats["p95"],
                score_stats["p99"],
            )

            cluster_means = (
                adata.obs.groupby("leiden", observed=False)[score_name]
                .mean()
                .sort_values(ascending=False)
            )
            top_clusters = cluster_means.head(10).round(4).to_dict()

            self.logger.warning(
                "PROGRAM_CLUSTER_MEANS: program=%s top_clusters=%s",
                program,
                top_clusters,
            )

            program_details[program] = {
                **base_detail,
                "scored": True,
                "score_stats": score_stats,
                "top_clusters": top_clusters,
            }

        available_programs = sorted(program_columns.keys())

        resolved_selected_program = None
        if requested_program and requested_program in program_columns:
            resolved_selected_program = requested_program
        elif available_programs:
            resolved_selected_program = available_programs[0]

        points = self._build_umap_points(adata, program_columns=program_columns)
        cluster_labels = sorted(adata.obs["leiden"].astype(str).unique().tolist())
        cluster_counts = adata.obs["leiden"].astype(str).value_counts().to_dict()

        annotation_data = self._load_ligand_receptor_drug_annotations()
        de_by_cluster = self._rank_genes(
            adata,
            groupby="leiden",
            top_n=de_top_n,
            annotation_data=annotation_data,
        )

        de_by_cell_type = None
        if "cell_type" in adata.obs:
            de_by_cell_type = self._rank_genes(
                adata,
                groupby="cell_type",
                top_n=de_top_n,
                annotation_data=annotation_data,
            )

        target_sets = self._load_target_gene_sets()
        overlap_report = self._compute_overlap_report(
            de_by_cluster, de_by_cell_type, target_sets
        )

        analysis_summary_path = self._write_analysis_summary(
            {
                "total_cells": adata.n_obs,
                "cluster_counts": cluster_counts,
                "cell_types": supptable_summary,
                "program_details": program_details,
                "metadata": {
                    "de_top_n": de_top_n,
                    "cluster_resolution": cluster_resolution,
                    "use_hvg": use_hvg,
                    "apply_filtering": apply_filtering,
                    "de_overlaps": overlap_report,
                    "available_programs": available_programs,
                    "program_columns": program_columns,
                    "selected_program": resolved_selected_program,
                    "has_program_scores": bool(program_columns),
                    **supptable_meta,
                    **embedding_meta,
                },
            }
        )

        payload = {
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
                "used_existing_umap": has_umap,
                "used_existing_leiden": has_leiden,
                "use_hvg": use_hvg,
                "apply_filtering": apply_filtering,
                "de_overlaps": overlap_report,
                "analysis_summary_path": analysis_summary_path,
                "available_programs": available_programs,
                "program_columns": program_columns,
                "selected_program": resolved_selected_program,
                "has_program_scores": bool(program_columns),
                "program_details": program_details,
                **supptable_meta,
                **embedding_meta,
            },
        }
        return _sanitize_for_json(payload)

    def _write_analysis_summary(self, summary: Dict) -> str:
        analysis_root = settings.DATA_DIR / "analysis_runs"
        analysis_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        summary_path = analysis_root / f"visualization_summary_{timestamp}.json"
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(_sanitize_for_json(summary), handle, indent=2)
        return str(summary_path)

    def _preprocess(
        self, adata, use_hvg: bool, apply_filtering: bool, compute_umap: bool = True
    ) -> None:
        if apply_filtering:
            sc.pp.filter_cells(adata, min_genes=200)
            sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        if use_hvg:
            sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
        sc.pp.scale(adata, max_value=10)
        sc.tl.pca(adata, svd_solver="arpack")
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=min(40, adata.obsm["X_pca"].shape[1]))
        if compute_umap:
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

    def _load_supptable(self, source: str) -> pd.DataFrame:
        if os.path.isfile(source):
            ext = os.path.splitext(source)[1].lower()
            if ext in [".csv", ".tsv", ".txt"]:
                sep = "\t" if ext == ".tsv" else ","
                return pd.read_csv(source, sep=sep)
            return pd.read_excel(source)
        with urllib.request.urlopen(source) as response:
            content = response.read()
        return pd.read_excel(io.BytesIO(content))
    
    
    def _load_marker_weight_xlsx(self) -> Optional[pd.DataFrame]:
        xlsx_path = settings.DATA_DIR / "SuppTable1.xlsx"
        if not xlsx_path.exists():
            self.logger.warning("SuppTable1.xlsx not found at %s", xlsx_path)
            return None
        wb = pd.read_excel(xlsx_path, sheet_name=None)
        if "LLM markers" in wb:
            return wb["LLM markers"]
        return next(iter(wb.values()), None)

    def _build_programs_from_supptable(
        self,
        df: pd.DataFrame,
        min_genes: int = 3,
        max_genes: int = 100,
    ) -> Dict[str, List[str]]:
        cols = {self._normalize(c): c for c in df.columns}

        cell_col = cols.get("celltype") or cols.get("cell_type")
        gene_col = cols.get("genemarker") or cols.get("markergene") or cols.get("gene")
        w_col = cols.get("weights") or cols.get("weight")

        if not cell_col or not gene_col:
            self.logger.warning(
                "Supptable missing program columns. Found columns: %s",
                df.columns.tolist(),
            )
            return {}

        keep_cols = [cell_col, gene_col] + ([w_col] if w_col else [])
        working = df[keep_cols].copy()
        working = working.dropna(subset=[cell_col, gene_col])

        working[cell_col] = working[cell_col].astype(str).str.strip()
        working[gene_col] = (
            working[gene_col].astype(str).str.strip().str.upper()
        )

        if w_col:
            working[w_col] = pd.to_numeric(working[w_col], errors="coerce").fillna(0.0)

        programs: Dict[str, List[str]] = {}

        for cell_type, grp in working.groupby(cell_col):
            if w_col:
                genes = grp.sort_values(w_col, ascending=False)[gene_col].tolist()
            else:
                genes = grp[gene_col].tolist()

            seen = set()
            uniq = []
            for g in genes:
                if g and g not in seen:
                    seen.add(g)
                    uniq.append(g)

            uniq = uniq[:max_genes]
            if len(uniq) >= min_genes:
                programs[str(cell_type)] = uniq

        self.logger.info("Built %d programs from supptable.", len(programs))
        return programs

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
                    avg_score = self._safe_float(scores.mean())
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
        cell_type_col = find(
            ["cell_type", "celltype", "type", "celltypeannotation", "matched_cell_type"]
        )
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

    def _resolve_embedding_matches_path(
        self,
        embedding_matches_path: Optional[str],
        source_filename: Optional[str],
    ) -> Optional[str]:
        if embedding_matches_path:
            candidate = Path(embedding_matches_path)
            if candidate.is_file():
                return str(candidate)
            self.logger.warning("Embedding matches file not found: %s", embedding_matches_path)
            return None

        if source_filename:
            stem = Path(source_filename).stem
            candidate = (
                Path(settings.DATA_DIR)
                / "embedding_runs"
                / f"{stem}_embs"
                / "embedding_matches_with_celltypes.csv"
            )
            if candidate.is_file():
                return str(candidate)
        return None

    def _apply_embedding_matches(self, matches_path: str, adata) -> bool:
        df = pd.read_csv(matches_path)
        if df.empty:
            return False

        normalized = {self._normalize(col): col for col in df.columns}

        def find(candidates: List[str]) -> Optional[str]:
            for candidate in candidates:
                key = self._normalize(candidate)
                if key in normalized:
                    return normalized[key]
            return None

        cell_id_col = find(["cell_id", "cell", "new_cell_id"])
        cell_type_col = find(["matched_cell_type", "predicted_cell_type", "cell_type"])
        score_col = find(["max_cosine", "similarity", "score"])

        if not cell_id_col or not cell_type_col:
            self.logger.warning(
                "Embedding matches missing required columns: %s", df.columns.tolist()
            )
            return False

        working = df[[cell_id_col, cell_type_col]].copy()
        if score_col:
            working[score_col] = pd.to_numeric(working[score_col], errors="coerce")

        working = working.dropna(subset=[cell_id_col])
        working[cell_id_col] = working[cell_id_col].astype(str)
        working = working.set_index(cell_id_col)

        predicted_series = working[cell_type_col].reindex(adata.obs_names)
        adata.obs["predicted_cell_type"] = predicted_series

        if score_col:
            score_series = working[score_col].reindex(adata.obs_names)
            adata.obs["predicted_cell_type_score"] = score_series

        return True

    def _coalesce_cell_type_obs(self, adata) -> None:
        """Fill `cell_type` from `predicted_cell_type` where missing."""
        if "predicted_cell_type" not in adata.obs.columns:
            return
        pred = pd.Series(adata.obs["predicted_cell_type"], dtype="string")
        pred = pred.mask(pred.str.strip() == "", pd.NA)
        if "cell_type" not in adata.obs.columns:
            adata.obs["cell_type"] = pred
            return
        ct = pd.Series(adata.obs["cell_type"], dtype="string")
        ct = ct.mask(ct.str.strip() == "", pd.NA)
        adata.obs["cell_type"] = ct.where(ct.notna(), pred)

    def _build_umap_points(
        self,
        adata,
        program_columns: Optional[Dict[str, str]] = None,
    ) -> List[Dict]:
        coords = adata.obsm["X_umap"]
        points = []

        cell_types = adata.obs.get("cell_type") if "cell_type" in adata.obs else None
        predicted = (
            adata.obs.get("predicted_cell_type")
            if "predicted_cell_type" in adata.obs
            else None
        )
        predicted_scores = (
            adata.obs.get("predicted_cell_type_score")
            if "predicted_cell_type_score" in adata.obs
            else None
        )
        clusters = adata.obs["leiden"].astype(str)

        for idx, cell_id in enumerate(adata.obs_names):
            cell_type = None
            predicted_cell_type = None
            predicted_score = None

            if cell_types is not None:
                value = cell_types.iloc[idx]
                if pd.notna(value):
                    cell_type = str(value)

            if predicted is not None:
                value = predicted.iloc[idx]
                if pd.notna(value):
                    predicted_cell_type = str(value)

            if predicted_scores is not None:
                predicted_score = self._safe_float(predicted_scores.iloc[idx])

            x_coord = self._safe_float(coords[idx, 0])
            y_coord = self._safe_float(coords[idx, 1])

            point = {
                "cell_id": str(cell_id),
                "x": x_coord if x_coord is not None else 0.0,
                "y": y_coord if y_coord is not None else 0.0,
                "cluster": str(clusters.iloc[idx]),
                "cell_type": cell_type,
                "predicted_cell_type": predicted_cell_type,
                "predicted_score": predicted_score,
                "program_scores": {},
            }

            if program_columns:
                for program_name, col_name in program_columns.items():
                    if col_name in adata.obs.columns:
                        point["program_scores"][program_name] = self._safe_float(
                            adata.obs.iloc[idx][col_name]
                        )

            points.append(point)

        return points

    def _expression_log2fc_for_genes(
        self, adata, groupby: str, group_id: str, genes: List[str]
    ) -> List[Optional[float]]:
        """
        log2 fold change from pseudocount-protected means. The Scanpy workflow stores
        log1p-normalized counts in ``X``; averaging in log space then taking log2 of
        a ratio often yields NaN (e.g. negative means on corrected data). Here we use
        mean(expm1(X)) per group — on log1p data this approximates mean raw scale, so
        log2((mean_g + eps) / (mean_r + eps)) is finite and comparable to standard DE FC.
        """
        obs_groups = adata.obs[groupby].astype(str)
        mask = (obs_groups == str(group_id)).to_numpy()
        ref = ~mask
        if not mask.any() or not ref.any():
            return [None] * len(genes)
        eps = 1e-9
        out: List[Optional[float]] = []
        for gene in genes:
            idx_arr = adata.var_names.get_indexer([str(gene)])
            if idx_arr[0] < 0:
                out.append(None)
                continue
            col = adata.X[:, int(idx_arr[0])]
            if hasattr(col, "toarray"):
                v = col.toarray().ravel()
            else:
                v = np.asarray(col).ravel()
            v = v.astype(np.float64, copy=False)
            vmin = float(np.min(v))
            vmax = float(np.max(v))
            # After ``log1p``, values are small and non-negative; expm1 undoes log1p for a
            # pseudo-count mean. Skip expm1 for likely raw/linear or strongly corrected X.
            if vmin >= -0.05 and vmax <= 40.0:
                v_in = np.expm1(np.maximum(v[mask], 0.0))
                v_out = np.expm1(np.maximum(v[ref], 0.0))
            else:
                v_in = np.maximum(v[mask], 0.0)
                v_out = np.maximum(v[ref], 0.0)
            mg = float(np.mean(v_in))
            mr = float(np.mean(v_out))
            if not np.isfinite(mg) or not np.isfinite(mr):
                out.append(None)
                continue
            num = max(mg, 0.0) + eps
            den = max(mr, 0.0) + eps
            ratio = num / den
            if ratio <= 0:
                out.append(None)
                continue
            lfc = float(np.log2(ratio))
            out.append(lfc if np.isfinite(lfc) else None)
        return out

    def _mannwhitney_p_for_gene(
        self, adata, groupby: str, group_id: str, gene: str
    ) -> Optional[float]:
        idx_arr = adata.var_names.get_indexer([str(gene)])
        if idx_arr[0] < 0:
            return None
        col = adata.X[:, int(idx_arr[0])]
        if hasattr(col, "toarray"):
            v = col.toarray().ravel()
        else:
            v = np.asarray(col).ravel()
        obs_groups = adata.obs[groupby].astype(str)
        mask = obs_groups == str(group_id)
        g_in = v[mask.to_numpy()]
        g_out = v[(~mask).to_numpy()]
        if len(g_in) < 3 or len(g_out) < 3:
            return None
        try:
            _, p = stats.mannwhitneyu(g_in, g_out, alternative="two-sided")
            return float(p) if np.isfinite(p) else None
        except ValueError:
            return None

    def _backfill_de_statistics(
        self,
        adata,
        groupby: str,
        group_id: str,
        genes: List[str],
        group_logfold: Optional[List[Optional[float]]],
        group_pvals_adj: Optional[List[Optional[float]]],
        group_pvals_unc: Optional[List[Optional[float]]],
    ) -> Tuple[List[Optional[float]], List[Optional[float]]]:
        """
        Wilcoxon logFC from Scanpy is often NaN when group means are zero; recompute from
        observed means. Fill missing adjusted p-values from uncorrected Scanpy p-values,
        then Mann--Whitney, and Benjamini--Hochberg on values that were not already
        Scanpy FDR-adjusted.
        """
        n = len(genes)
        lfc_in = list(group_logfold or [])[:n]
        if len(lfc_in) < n:
            lfc_in.extend([None] * (n - len(lfc_in)))
        computed_lfc = self._expression_log2fc_for_genes(adata, groupby, group_id, genes)
        merged_lfc: List[Optional[float]] = []
        for i in range(n):
            v = lfc_in[i] if i < len(lfc_in) else None
            if v is not None and np.isfinite(v):
                merged_lfc.append(v)
            else:
                merged_lfc.append(computed_lfc[i] if i < len(computed_lfc) else None)

        def _get(lst: Optional[List[Optional[float]]], i: int) -> Optional[float]:
            if lst is None or i >= len(lst):
                return None
            return lst[i]

        p_out: List[Optional[float]] = [None] * n
        scanpy_adj_indices = set()
        for i in range(n):
            pa = _get(group_pvals_adj, i)
            if pa is not None and np.isfinite(pa):
                p_out[i] = pa
                scanpy_adj_indices.add(i)
                continue
            pu = _get(group_pvals_unc, i)
            if pu is not None and np.isfinite(pu):
                p_out[i] = pu

        for i in range(n):
            if p_out[i] is None or not np.isfinite(p_out[i]):
                p_out[i] = self._mannwhitney_p_for_gene(
                    adata, groupby, group_id, genes[i]
                )

        to_adjust = [
            i
            for i in range(n)
            if i not in scanpy_adj_indices
            and p_out[i] is not None
            and np.isfinite(p_out[i])
        ]
        if to_adjust:
            batch = np.array([float(p_out[i]) for i in to_adjust], dtype=float)
            batch = np.clip(batch, 0.0, 1.0)
            try:
                adj_batch = stats.false_discovery_control(batch)
            except Exception:
                adj_batch = np.minimum(batch * len(batch), 1.0)
            for j, i in enumerate(to_adjust):
                p_out[i] = float(adj_batch[j])

        return merged_lfc, p_out

    def _rank_genes(
        self,
        adata,
        groupby: str,
        top_n: int,
        annotation_data: Optional[Dict] = None,
    ) -> Optional[List[Dict]]:
        if groupby not in adata.obs:
            return None
        series = adata.obs[groupby]
        if series.dropna().nunique() < 2:
            return None

        if not annotation_data:
            annotation_data = self._load_ligand_receptor_drug_annotations()

        sc.tl.rank_genes_groups(adata, groupby=groupby, method="wilcoxon", use_raw=False)
        result = adata.uns.get("rank_genes_groups", {})
        names = result.get("names")
        if names is None:
            return None

        groups = list(names.dtype.names) if hasattr(names, "dtype") and names.dtype.names else []
        if not groups:
            return None

        gene_symbol_map = None
        for col in ["gene_symbol", "gene_name", "symbol", "gene", "Gene"]:
            if col in adata.var.columns:
                gene_symbol_map = dict(
                    zip(
                        adata.var_names.astype(str),
                        adata.var[col].astype(str),
                    )
                )
                break
        if gene_symbol_map is None:
            gene_symbol_map = self._infer_gene_symbol_map(adata)

        output = []
        scores = result.get("scores")
        logfold = result.get("logfoldchanges")
        pvals_adj = result.get("pvals_adj")
        pvals_unc = result.get("pvals")
        ligand_genes = annotation_data.get("ligands", set())
        receptor_genes = annotation_data.get("receptors", set())
        drug_targets = annotation_data.get("drug_targets", {})

        for group in groups:
            raw_genes = np.asarray(names[group])[:top_n].tolist()
            genes = [str(gene) for gene in raw_genes]
            if gene_symbol_map:
                genes = [gene_symbol_map.get(gene, gene) for gene in genes]
            group_scores = None
            group_logfold = None
            group_pvals_adj_list = None
            group_pvals_unc_list = None

            if scores is not None:
                group_scores = [
                    self._safe_float(x) for x in np.asarray(scores[group])[:top_n].tolist()
                ]
            if logfold is not None:
                group_logfold = [
                    self._safe_float(x) for x in np.asarray(logfold[group])[:top_n].tolist()
                ]
            if pvals_adj is not None:
                group_pvals_adj_list = [
                    self._safe_float(x) for x in np.asarray(pvals_adj[group])[:top_n].tolist()
                ]
            if pvals_unc is not None:
                group_pvals_unc_list = [
                    self._safe_float(x) for x in np.asarray(pvals_unc[group])[:top_n].tolist()
                ]

            merged_logfc, merged_pvals = self._backfill_de_statistics(
                adata,
                groupby,
                str(group),
                genes,
                group_logfold,
                group_pvals_adj_list,
                group_pvals_unc_list,
            )

            output.append(
                {
                    "group": str(group),
                    "genes": genes,
                    "scores": group_scores,
                    "logfoldchanges": merged_logfc,
                    "pvals_adj": merged_pvals,
                    "gene_annotations": self._build_gene_annotations(
                        genes,
                        ligand_genes=ligand_genes,
                        receptor_genes=receptor_genes,
                        drug_targets=drug_targets,
                    ),
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

    def _safe_bool(self, value) -> Optional[bool]:
        if value is None or pd.isna(value):
            return None
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "t", "1", "yes", "y"}:
            return True
        if text in {"false", "f", "0", "no", "n"}:
            return False
        return None

    def _safe_str(self, value) -> Optional[str]:
        if value is None or pd.isna(value):
            return None
        text = str(value).strip()
        return text if text else None

    def _normalize_gene(self, value: str) -> str:
        return str(value).strip().upper()

    def _load_ligand_receptor_drug_annotations(self) -> Dict:
        if self._annotation_cache is not None:
            return self._annotation_cache

        annotations_dir = settings.ANNOTATIONS_DIR
        ligands_path = annotations_dir / "ligands.txt"
        receptors_path = annotations_dir / "receptors.txt"
        drugs_path = annotations_dir / "drug.tsv"

        ligand_genes = self._load_gene_list(ligands_path)
        receptor_genes = self._load_gene_list(receptors_path)
        drug_targets = self._load_drug_targets(drugs_path)

        self._annotation_cache = {
            "ligands": ligand_genes,
            "receptors": receptor_genes,
            "drug_targets": drug_targets,
        }
        return self._annotation_cache

    def _find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        normalized = {self._normalize(col): col for col in df.columns}
        for candidate in candidates:
            key = self._normalize(candidate)
            if key in normalized:
                return normalized[key]
        return None

    def _load_gene_list(self, path: Path) -> set:
        if not path.is_file():
            self.logger.warning("Annotation file not found: %s", path)
            return set()
        df = pd.read_csv(path, sep="\t")
        gene_col = self._find_column(df, ["hgnc symbol", "gene_name", "gene", "symbol"])
        if not gene_col:
            self.logger.warning("Gene column not found in %s", path)
            return set()
        genes = df[gene_col].dropna().astype(str)
        return {self._normalize_gene(gene) for gene in genes if gene.strip()}

    def _load_drug_targets(self, path: Path) -> Dict[str, List[Dict]]:
        if not path.is_file():
            self.logger.warning("Drug targets file not found: %s", path)
            return {}
        df = pd.read_csv(path, sep="\t")
        gene_col = self._find_column(df, ["gene_name", "gene", "hgnc symbol", "symbol"])
        if not gene_col:
            self.logger.warning("Gene column not found in %s", path)
            return {}

        output: Dict[str, List[Dict]] = {}
        for record in df.to_dict(orient="records"):
            gene_value = record.get(gene_col)
            if gene_value is None or pd.isna(gene_value):
                continue
            gene_key = self._normalize_gene(gene_value)
            entry = {
                "drug_name": self._safe_str(record.get("drug_name")),
                "drug_claim_name": self._safe_str(record.get("drug_claim_name")),
                "drug_concept_id": self._safe_str(record.get("drug_concept_id")),
                "interaction_source_db_name": self._safe_str(
                    record.get("interaction_source_db_name")
                ),
                "interaction_type": self._safe_str(record.get("interaction_type")),
                "interaction_score": self._safe_float(record.get("interaction_score")),
                "approved": self._safe_bool(record.get("approved")),
                "immunotherapy": self._safe_bool(record.get("immunotherapy")),
                "anti_neoplastic": self._safe_bool(record.get("anti_neoplastic")),
            }
            output.setdefault(gene_key, []).append(entry)
        return output

    def _build_gene_annotations(
        self,
        genes: List[str],
        ligand_genes: set,
        receptor_genes: set,
        drug_targets: Dict[str, List[Dict]],
    ) -> List[Dict]:
        annotations = []
        for gene in genes:
            gene_key = self._normalize_gene(gene)
            targets = drug_targets.get(gene_key)
            annotations.append(
                {
                    "gene": gene,
                    "is_ligand": gene_key in ligand_genes,
                    "is_receptor": gene_key in receptor_genes,
                    "drug_targets": targets if targets else None,
                }
            )
        return annotations
    
    def _load_target_gene_sets(self) -> Dict[str, set]:
        targets: Dict[str, set] = {}
        drug_path = settings.ANNOTATIONS_DIR / "drug.tsv"
        ligands_path = settings.ANNOTATIONS_DIR / "ligands.txt"
        receptors_path = settings.ANNOTATIONS_DIR / "receptors.txt"

        if drug_path.exists():
            df = pd.read_csv(drug_path, sep="\t")
            if "gene_name" in df.columns:
                genes = df["gene_name"].dropna().astype(str).tolist()
            else:
                genes = df.iloc[:, 0].dropna().astype(str).tolist()
            targets["drug_targets"] = {self._normalize_gene(g) for g in genes}

        if ligands_path.exists():
            with open(ligands_path, "r", encoding="utf-8", errors="ignore") as f:
                genes = [self._extract_target_gene(line) for line in f if line.strip()]
            targets["ligands"] = {self._normalize_gene(g) for g in genes if g}

        if receptors_path.exists():
            with open(receptors_path, "r", encoding="utf-8", errors="ignore") as f:
                genes = [self._extract_target_gene(line) for line in f if line.strip()]
            targets["receptors"] = {self._normalize_gene(g) for g in genes if g}

        return targets

    def _extract_target_gene(self, raw: str) -> str:
        cleaned = str(raw).strip()
        if not cleaned:
            return ""
        # Target lists append descriptors like "LIGAND" or "ECM/RECEPTOR"
        return cleaned.split()[0]

    def _infer_gene_symbol_map(self, adata) -> Optional[Dict[str, str]]:
        for col in adata.var.columns:
            series = adata.var[col]
            if not pd.api.types.is_string_dtype(series):
                continue
            sample = series.dropna().astype(str).head(200).tolist()
            if not sample:
                continue
            with_letters = sum(bool(re.search(r"[A-Za-z]", value)) for value in sample)
            numeric_like = sum(bool(re.fullmatch(r"\d+", value)) for value in sample)
            if with_letters / len(sample) >= 0.6 and numeric_like / len(sample) <= 0.2:
                return dict(zip(adata.var_names.astype(str), series.astype(str)))
        return None

    def _compute_overlap_report(
        self,
        de_by_cluster: Optional[List[Dict]],
        de_by_cell_type: Optional[List[Dict]],
        target_sets: Dict[str, set],
    ) -> Optional[Dict]:
        if not target_sets:
            return None
        report = {
            "targets_loaded": {name: len(values) for name, values in target_sets.items()},
            "by_cluster": self._compute_overlaps_for_de(de_by_cluster, target_sets),
            "by_cell_type": self._compute_overlaps_for_de(de_by_cell_type, target_sets),
            "debug_samples": self._build_overlap_debug_samples(
                de_by_cluster, de_by_cell_type, target_sets
            ),
        }
        if not report["by_cluster"] and not report["by_cell_type"]:
            return report
        return report

    def _compute_overlaps_for_de(
        self, de_groups: Optional[List[Dict]], target_sets: Dict[str, set]
    ) -> Optional[Dict]:
        if not de_groups:
            return None
        output: Dict[str, Dict[str, List[str]]] = {}
        for group in de_groups:
            group_name = str(group.get("group", ""))
            genes = [self._normalize_gene(g) for g in group.get("genes", []) if g]
            if not genes:
                continue
            group_set = set(genes)
            overlaps: Dict[str, List[str]] = {}
            for target_name, target_set in target_sets.items():
                overlap = sorted(group_set.intersection(target_set))
                if overlap:
                    overlaps[target_name] = overlap
            if overlaps:
                output[group_name] = overlaps
        return output or None

    def _build_overlap_debug_samples(
        self,
        de_by_cluster: Optional[List[Dict]],
        de_by_cell_type: Optional[List[Dict]],
        target_sets: Dict[str, set],
    ) -> Dict:
        def sample_genes(groups: Optional[List[Dict]]) -> Optional[List[str]]:
            if not groups:
                return None
            for group in groups:
                genes = [self._normalize_gene(g) for g in group.get("genes", []) if g]
                if genes:
                    return genes[:25]
            return None

        debug = {
            "de_cluster_sample": sample_genes(de_by_cluster),
            "de_cell_type_sample": sample_genes(de_by_cell_type),
            "target_samples": {
                name: sorted(list(values))[:25] for name, values in target_sets.items()
            },
        }
        return debug
    