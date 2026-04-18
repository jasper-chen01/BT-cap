"""
Background jobs: run unmodified preps/generate_preds.py, then preps/patchseq_predict.py when
configured, merge prediction CSVs into AnnData, then reuse VisualizationService for UMAP (Scanpy).
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import scanpy as sc

from backend.config import settings
from backend.services.visualization_service import VisualizationService

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pick_scores_csv(preds_dir: Path, reference_substring: Optional[str]) -> Path:
    files = sorted(preds_dir.glob("preds_by_*_num_classes_*_scores.csv"))
    if not files:
        raise FileNotFoundError(f"No PREPS *_scores.csv files in {preds_dir}")
    if reference_substring:
        ref = reference_substring.strip()
        for f in files:
            if ref in f.name:
                return f
        raise FileNotFoundError(
            f"No scores file matching substring {ref!r} under {preds_dir}"
        )
    default = settings.PREPS_DEFAULT_REFERENCE_SUBSTRING.strip()
    for f in files:
        if default and default in f.name:
            return f
    return files[0]


def _ref_prefix_from_scores_filename(path: Path) -> Optional[str]:
    m = re.match(r"preds_by_(.+)_num_classes_\d+_scores\.csv$", path.name)
    return m.group(1) if m else None


def merge_preps_scores_into_adata(
    adata_path: Path,
    preds_dir: Path,
    reference_substring: Optional[str],
    merged_out: Path,
) -> Dict[str, Any]:
    """
    Read PREPS annotate.py output (*_scores.csv), align by cell id (`individual`),
    set predicted_cell_type / predicted_cell_type_score, write h5ad for visualization.
    """
    scores_path = _pick_scores_csv(preds_dir, reference_substring)
    ref_from_file = _ref_prefix_from_scores_filename(scores_path)

    df = pd.read_csv(scores_path, index_col=0)
    if df.index.name != "individual":
        df.index.name = "individual"
    df.index = df.index.astype(str)

    ann_cols = [c for c in df.columns if str(c).endswith("_ann")]
    if not ann_cols:
        raise ValueError(f"No *_ann column in {scores_path}")

    ann_col = None
    score_col = None
    if ref_from_file:
        want_ann = f"{ref_from_file}_ann"
        want_score = f"{ref_from_file}_score"
        if want_ann in df.columns and want_score in df.columns:
            ann_col, score_col = want_ann, want_score
    if ann_col is None:
        ann_col = ann_cols[-1]
        prefix = str(ann_col)[:-4]
        score_col = f"{prefix}_score"
        if score_col not in df.columns:
            raise ValueError(f"Missing score column for {ann_col!r} in {scores_path}")

    adata = sc.read_h5ad(str(adata_path))
    adata.obs_names = adata.obs_names.astype(str)

    ann_series = df[ann_col].reindex(adata.obs_names)
    score_series = pd.to_numeric(df[score_col], errors="coerce").reindex(adata.obs_names)
    adata.obs["predicted_cell_type"] = ann_series
    adata.obs["predicted_cell_type_score"] = score_series

    n_match = int(ann_series.notna().sum())
    merged_out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(str(merged_out), compression=None)

    return {
        "scores_csv": str(scores_path),
        "ann_column": ann_col,
        "score_column": score_col,
        "cells_with_prediction": str(n_match),
        "merged_h5ad": str(merged_out),
    }


def merge_patchseq_ephys_xlsx_into_adata(
    merged_h5ad: Path,
    patchseq_dir: Path,
) -> Dict[str, Any]:
    """
    Read patchseq_predict.py output .xlsx files and append numeric columns to adata.obs
    so VisualizationService can color UMAP by predicted ephys (same pathway as program scores).
    """
    if not patchseq_dir.is_dir():
        return {"ephys_merged_for_viz": "skipped_no_dir"}

    xlsx_files = sorted(patchseq_dir.glob("*.xlsx"))
    if not xlsx_files:
        return {"ephys_merged_for_viz": "skipped_no_xlsx"}

    adata = sc.read_h5ad(str(merged_h5ad))
    adata.obs_names = adata.obs_names.astype(str)

    labels: Dict[str, str] = {}
    used_obs_keys: set[str] = set()

    for path in xlsx_files:
        try:
            df = pd.read_excel(path, index_col=0, engine="openpyxl")
        except Exception as exc:
            logger.warning("Skipping ephys xlsx %s: %s", path, exc)
            continue
        if df.shape[1] < 1:
            continue
        df.index = df.index.astype(str)
        col_name = str(df.columns[0])
        series = pd.to_numeric(df.iloc[:, 0], errors="coerce").reindex(adata.obs_names)

        base_slug = re.sub(r"[^A-Za-z0-9_]+", "_", col_name).strip("_")[:100]
        if not base_slug:
            base_slug = re.sub(r"[^A-Za-z0-9_]+", "_", path.stem).strip("_")[:100] or "feature"
        obs_key = f"ephys__{base_slug}"
        n = 1
        while obs_key in adata.obs.columns or obs_key in used_obs_keys:
            obs_key = f"ephys__{base_slug}_{n}"
            n += 1
        used_obs_keys.add(obs_key)

        display = f"Ephys · {col_name}"
        if display in labels.values():
            display = f"Ephys · {col_name} ({path.stem[:32]})"

        adata.obs[obs_key] = series
        labels[obs_key] = display

    if not labels:
        return {"ephys_merged_for_viz": "skipped_no_columns"}

    adata.uns["ephys_column_labels"] = labels
    adata.write_h5ad(str(merged_h5ad), compression=None)
    return {
        "ephys_merged_for_viz": "ok",
        "ephys_obs_columns": list(labels.keys()),
        "ephys_feature_count": str(len(labels)),
    }


@dataclass
class PrepsJob:
    job_id: str
    status: str
    created_at: str
    started_at: Optional[str]
    finished_at: Optional[str]
    test_name: str
    command: List[str]
    log_path: str
    work_dir: str
    pid: Optional[int]
    exit_code: Optional[int]
    error_message: Optional[str]
    merge_info: Optional[Dict[str, Any]]
    visualization: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


class PrepsJobService:
    _jobs: Dict[str, PrepsJob] = {}
    _lock = threading.Lock()

    def preps_available(self) -> bool:
        if not settings.PREPS_PYTHON or not Path(settings.PREPS_PYTHON).is_file():
            return False
        gen = settings.PREPS_DIR / "generate_preds.py"
        if not gen.is_file():
            return False
        if not settings.PREPS_DICT_DIR.is_dir():
            return False
        if not settings.PREPS_MODELS_ROOT.is_dir():
            return False
        return True

    def start_job(
        self,
        job_id: str,
        input_h5ad_path: str,
        species: str = "human",
        gpu: str = "0",
        reference_substring: Optional[str] = None,
        de_top_n: int = 15,
        cluster_resolution: float = 1.0,
    ) -> PrepsJob:
        if not self.preps_available():
            raise RuntimeError(
                "PREPS is not configured. Set PREPS_PYTHON to your conda preps env "
                "python executable, and ensure PREPS_DIR, PREPS_DICT_DIR, and "
                "PREPS_MODELS_ROOT exist (see preps/HOWTO_PREPS.md)."
            )

        input_h5ad_path = str(Path(input_h5ad_path).resolve())
        if not Path(input_h5ad_path).is_file():
            raise FileNotFoundError(input_h5ad_path)

        preps_dir = settings.PREPS_DIR.resolve()
        test_name = f"portal_{job_id.replace('-', '')[:12]}"

        work_root = settings.DATA_DIR / "preps_jobs" / job_id
        work_root.mkdir(parents=True, exist_ok=True)
        log_path = str(work_root / "preps.log")

        cmd = [
            settings.PREPS_PYTHON,
            str(preps_dir / "generate_preds.py"),
            input_h5ad_path,
            test_name,
            "-s",
            species,
            "--copy",
            "-g",
            str(gpu),
            "--models-root",
            str(settings.PREPS_MODELS_ROOT.resolve()),
            "--dict-dir",
            str(settings.PREPS_DICT_DIR.resolve()),
        ]

        job = PrepsJob(
            job_id=str(job_id),
            status="queued",
            created_at=_utc_now(),
            started_at=None,
            finished_at=None,
            test_name=test_name,
            command=cmd,
            log_path=log_path,
            work_dir=str(work_root),
            pid=None,
            exit_code=None,
            error_message=None,
            merge_info=None,
            visualization=None,
        )

        ctx = {
            "reference_substring": reference_substring,
            "de_top_n": de_top_n,
            "cluster_resolution": cluster_resolution,
        }

        with self._lock:
            self._jobs[job_id] = job

        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, ctx),
            daemon=True,
        )
        thread.start()
        return job

    def get_job(self, job_id: str) -> Optional[PrepsJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def read_log(self, job_id: str, tail: int = 200) -> str:
        job = self.get_job(job_id)
        if not job:
            raise KeyError(job_id)
        p = Path(job.log_path)
        if not p.is_file():
            return ""
        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        if tail <= 0:
            return "\n".join(lines)
        return "\n".join(lines[-tail:])

    def _run_patchseq_predict(
        self,
        test_name: str,
        preps_dir: Path,
        logf,
    ) -> tuple[Optional[str], Dict[str, Any]]:
        """
        Run preps/patchseq_predict.py after generate_preds. Writes to the same log file.

        Returns (fatal_error_message_or_none, ephys_info_dict). Skipping is non-fatal
        (missing script or combined_patchseq_all_preds); failure is fatal if the script runs
        and exits non-zero.
        """
        ephys: Dict[str, Any] = {"ephys_patchseq_ran": False}
        if not settings.PREPS_RUN_PATCHSEQ_PREDICT:
            ephys["ephys_patchseq_note"] = "skipped (PREPS_RUN_PATCHSEQ_PREDICT disabled)"
            return None, ephys

        script = preps_dir / "patchseq_predict.py"
        ref_dir = preps_dir / "combined_patchseq_all_preds"
        if not script.is_file():
            ephys["ephys_patchseq_note"] = f"skipped (missing {script.name})"
            return None, ephys
        if not ref_dir.is_dir():
            ephys["ephys_patchseq_note"] = (
                f"skipped (missing {ref_dir.name}/ — see preps/HOWTO_PREPS.md)"
            )
            return None, ephys

        cmd = [settings.PREPS_PYTHON, str(script), test_name]
        logf.write("\n--- patchseq_predict.py (electrophysiology predictions) ---\n")
        logf.write(f"Command: {' '.join(cmd)}\n")
        logf.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(preps_dir),
            stdout=logf,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        if proc.returncode != 0:
            return (
                f"patchseq_predict.py exited with code {proc.returncode}",
                ephys,
            )

        out_dir = preps_dir / f"{test_name}_patchseq"
        xlsx = sorted(out_dir.glob("*.xlsx")) if out_dir.is_dir() else []
        ephys.update(
            {
                "ephys_patchseq_ran": True,
                "ephys_patchseq_output_dir": str(out_dir),
                "ephys_patchseq_xlsx_files": [p.name for p in xlsx],
            }
        )
        return None, ephys

    def _run_job(self, job_id: str, ctx: Dict[str, Any]) -> None:
        job = self.get_job(job_id)
        if not job:
            return

        job.status = "running"
        job.started_at = _utc_now()
        preps_dir = settings.PREPS_DIR.resolve()
        adata_after = preps_dir / job.test_name / "adata.h5ad"
        preds_dir = preps_dir / f"{job.test_name}_preds"
        merged_path = Path(job.work_dir) / "merged_for_viz.h5ad"
        ephys_info: Dict[str, Any] = {}

        try:
            with open(job.log_path, "a", encoding="utf-8") as logf:
                logf.write(f"Command: {' '.join(job.command)}\n")
                logf.flush()
                proc = subprocess.Popen(
                    job.command,
                    cwd=str(preps_dir),
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
                job.pid = proc.pid
                exit_code = proc.wait()
                job.exit_code = exit_code
                if exit_code != 0:
                    job.status = "failed"
                    job.error_message = f"generate_preds.py exited with code {exit_code}"
                    return

                ephys_err, ephys_info = self._run_patchseq_predict(
                    job.test_name, preps_dir, logf
                )
                if ephys_err:
                    job.status = "failed"
                    job.error_message = ephys_err
                    return

            if not adata_after.is_file():
                job.status = "failed"
                job.error_message = f"Expected adata not found: {adata_after}"
                return
            if not preds_dir.is_dir():
                job.status = "failed"
                job.error_message = f"Expected preds dir not found: {preds_dir}"
                return

            merge_info = merge_preps_scores_into_adata(
                adata_after,
                preds_dir,
                ctx.get("reference_substring"),
                merged_path,
            )
            merge_info = {**merge_info, **ephys_info}
            if ephys_info.get("ephys_patchseq_ran"):
                patchseq_out = preps_dir / f"{job.test_name}_patchseq"
                ephys_merge = merge_patchseq_ephys_xlsx_into_adata(merged_path, patchseq_out)
                merge_info = {**merge_info, **ephys_merge}
            job.merge_info = merge_info

            viz = VisualizationService().process_file(
                str(merged_path),
                de_top_n=int(ctx.get("de_top_n") or 15),
                cluster_resolution=float(ctx.get("cluster_resolution") or 1.0),
                source_filename=f"{job.test_name}_preps_merged.h5ad",
            )
            if isinstance(viz.get("metadata"), dict):
                viz["metadata"] = {
                    **viz["metadata"],
                    "preps_job_id": job_id,
                    "preps_test_name": job.test_name,
                    **merge_info,
                }
            job.visualization = viz
            job.status = "succeeded"
        except Exception as exc:
            logger.exception("PREPS job failed")
            job.status = "failed"
            job.error_message = repr(exc)
        finally:
            job.finished_at = _utc_now()


preps_job_service = PrepsJobService()
