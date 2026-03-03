"""
Service for visualization workflows (normalization, clustering, UMAP, DE).
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import urllib.request
from datetime import datetime
from pathlib import Path
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
    ) -> Dict:
        adata = sc.read_h5ad(file_path)

        if adata.var_names.isna().any() or "_index" in adata.var.columns:
            adata.var_names = adata.var.get("_index", adata.var_names)

        if "gene_symbol" not in adata.var.columns:
            adata.var["gene_symbol"] = adata.var_names

        if adata.raw is not None:
            adata = adata.raw.to_adata()

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

        supptable_summary = None
        supptable_meta = {}
        resolved_source = supptable_path or self._resolve_supptable_url(
            supptable_url, supptable_doc_id
        )
        if resolved_source:
            try:
                df = self._load_supptable(resolved_source)
                supptable_summary = self._apply_supptable(df, adata)
                supptable_meta["supptable_source"] = resolved_source
            except Exception as exc:
                self.logger.warning("Failed to load supptable: %s", exc)

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

        points = self._build_umap_points(adata)
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
                "metadata": {
                    "de_top_n": de_top_n,
                    "cluster_resolution": cluster_resolution,
                    "use_hvg": use_hvg,
                    "apply_filtering": apply_filtering,
                    "de_overlaps": overlap_report,
                    **supptable_meta,
                },
            }
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
                "used_existing_umap": has_umap,
                "used_existing_leiden": has_leiden,
                "use_hvg": use_hvg,
                "apply_filtering": apply_filtering,
                "de_overlaps": overlap_report,
                "analysis_summary_path": analysis_summary_path,
                **supptable_meta,
                **embedding_meta,
            },
        }

    def _write_analysis_summary(self, summary: Dict) -> str:
        analysis_root = settings.DATA_DIR / "analysis_runs"
        analysis_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        summary_path = analysis_root / f"visualization_summary_{timestamp}.json"
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
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

    def _build_umap_points(self, adata) -> List[Dict]:
        coords = adata.obsm["X_umap"]
        points = []
        cell_types = adata.obs.get("cell_type") if "cell_type" in adata.obs else None
        scores = adata.obs.get("cell_type_score") if "cell_type_score" in adata.obs else None
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
            score = None
            predicted_cell_type = None
            predicted_score = None
            if cell_types is not None:
                value = cell_types.iloc[idx]
                if pd.notna(value):
                    cell_type = str(value)
            if scores is not None:
                value = scores.iloc[idx]
                score = self._safe_float(value)
            if predicted is not None:
                value = predicted.iloc[idx]
                if pd.notna(value):
                    predicted_cell_type = str(value)
            if predicted_scores is not None:
                value = predicted_scores.iloc[idx]
                predicted_score = self._safe_float(value)
            points.append(
                {
                    "cell_id": str(cell_id),
                    "x": float(coords[idx, 0]),
                    "y": float(coords[idx, 1]),
                    "cluster": str(clusters.iloc[idx]),
                    "cell_type": cell_type,
                    "score": score,
                    "predicted_cell_type": predicted_cell_type,
                    "predicted_score": predicted_score,
                }
            )
        return points

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
        ligand_genes = set()
        receptor_genes = set()
        drug_targets = {}
        if annotation_data:
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
                    "gene_annotations": self._build_gene_annotations(
                        genes,
                        ligand_genes=ligand_genes,
                        receptor_genes=receptor_genes,
                        drug_targets=drug_targets,
                    )
                    if annotation_data
                    else None,
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
                "drug_name": record.get("drug_name"),
                "drug_claim_name": record.get("drug_claim_name"),
                "drug_concept_id": record.get("drug_concept_id"),
                "interaction_source_db_name": record.get("interaction_source_db_name"),
                "interaction_type": record.get("interaction_type"),
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

