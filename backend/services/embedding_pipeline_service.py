"""
Background job runner for embedding + matching pipeline.
"""
from __future__ import annotations

import glob
import os
import sys
import threading
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, List
from uuid import uuid4

import pandas as pd

from backend.config import settings, BASE_DIR


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EmbeddingPipelineJob:
    job_id: str
    status: str
    created_at: str
    started_at: Optional[str]
    finished_at: Optional[str]
    command: List[str]
    out_dir: str
    log_path: str
    pid: Optional[int]
    exit_code: Optional[int]
    error_message: Optional[str]
    embeddings_csv: Optional[str]
    matches_csv: Optional[str]
    annotated_matches_csv: Optional[str]

    def to_dict(self) -> Dict:
        return asdict(self)


class EmbeddingPipelineJobService:
    """
    In-memory job registry for embedding + matching pipeline.
    """

    _jobs: Dict[str, EmbeddingPipelineJob] = {}
    _lock = threading.Lock()

    def start_job(
        self,
        h5ad_path: str,
        dict_dir: Optional[str] = None,
        models_root: Optional[str] = None,
        out_dir: Optional[str] = None,
        gene_id_type: str = "symbol",
        gpu: str = "0",
        max_ncells: int = 1_000_000,
        forward_batch_size: int = 100,
        finetune_subdir: Optional[str] = None,
        trained_embeddings_path: Optional[str] = None,
        celltypes_path: Optional[str] = None,
        id_col: str = "individual",
    ) -> EmbeddingPipelineJob:
        h5ad_path = os.path.abspath(h5ad_path)
        dict_dir = os.path.abspath(dict_dir or (BASE_DIR / "backend" / "dict"))
        models_root = os.path.abspath(models_root or (BASE_DIR / "backend"))
        trained_embeddings_path = os.path.abspath(
            trained_embeddings_path or settings.TRAINED_EMBEDDINGS_PATH
        )
        celltypes_path = os.path.abspath(celltypes_path or settings.CELLTYPES_PATH)

        if not os.path.isfile(h5ad_path):
            raise FileNotFoundError(f"h5ad not found: {h5ad_path}")
        if not os.path.isdir(dict_dir):
            raise FileNotFoundError(f"dict dir not found: {dict_dir}")
        if not os.path.isdir(models_root):
            raise FileNotFoundError(f"models root not found: {models_root}")
        if not os.path.isfile(trained_embeddings_path):
            raise FileNotFoundError(
                f"trained embeddings not found: {trained_embeddings_path}"
            )
        if not os.path.isfile(celltypes_path):
            raise FileNotFoundError(f"celltypes file not found: {celltypes_path}")

        stem = Path(h5ad_path).stem
        if out_dir:
            out_dir = os.path.abspath(out_dir)
        else:
            out_dir = os.path.join(settings.DATA_DIR, "embedding_runs", f"{stem}_embs")
        os.makedirs(out_dir, exist_ok=True)

        logs_dir = os.path.join(settings.DATA_DIR, "embedding_jobs")
        os.makedirs(logs_dir, exist_ok=True)

        job_id = str(uuid4())
        log_path = os.path.join(logs_dir, f"{job_id}.log")

        cmd = [
            sys.executable,
            os.path.join(BASE_DIR, "backend", "run_embeddings.py"),
            "--h5ad",
            h5ad_path,
            "--dict-dir",
            dict_dir,
            "--models-root",
            models_root,
            "--out-dir",
            out_dir,
            "--gene-id-type",
            gene_id_type,
            "--gpu",
            str(gpu),
            "--max-ncells",
            str(max_ncells),
            "--forward-batch-size",
            str(forward_batch_size),
        ]
        if finetune_subdir:
            cmd.extend(["--finetune-subdir", finetune_subdir])

        job = EmbeddingPipelineJob(
            job_id=job_id,
            status="queued",
            created_at=_utc_now(),
            started_at=None,
            finished_at=None,
            command=cmd,
            out_dir=out_dir,
            log_path=log_path,
            pid=None,
            exit_code=None,
            error_message=None,
            embeddings_csv=None,
            matches_csv=None,
            annotated_matches_csv=None,
        )

        with self._lock:
            self._jobs[job_id] = job

        thread = threading.Thread(
            target=self._run_job,
            args=(
                job_id,
                trained_embeddings_path,
                celltypes_path,
                id_col,
            ),
            daemon=True,
        )
        thread.start()

        return job

    def get_job(self, job_id: str) -> Optional[EmbeddingPipelineJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> List[EmbeddingPipelineJob]:
        with self._lock:
            return list(self._jobs.values())

    def read_log(self, job_id: str, tail: int = 200) -> str:
        job = self.get_job(job_id)
        if not job:
            raise KeyError(f"Unknown job: {job_id}")
        if not os.path.isfile(job.log_path):
            return ""
        with open(job.log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        if tail <= 0:
            return "".join(lines)
        return "".join(lines[-tail:])

    def _run_job(
        self,
        job_id: str,
        trained_embeddings_path: str,
        celltypes_path: str,
        id_col: str,
    ) -> None:
        job = self.get_job(job_id)
        if not job:
            return
        job.status = "running"
        job.started_at = _utc_now()

        try:
            with open(job.log_path, "a", encoding="utf-8") as log_file:
                log_file.write("Command: " + " ".join(job.command) + "\n")
                log_file.flush()
                proc = subprocess.Popen(
                    job.command,
                    cwd=str(BASE_DIR),
                    stdout=log_file,
                    stderr=log_file,
                )
                job.pid = proc.pid
                exit_code = proc.wait()
                job.exit_code = exit_code
                if exit_code != 0:
                    job.status = "failed"
                    job.error_message = f"Embedding process exited with code {exit_code}"
                    return

            embeddings_csv = self._find_embeddings_csv(job.out_dir)
            job.embeddings_csv = embeddings_csv
            matches_csv = os.path.join(job.out_dir, "embedding_matches.csv")
            match_cmd = [
                sys.executable,
                os.path.join(BASE_DIR, "backend", "match_embeddings.py"),
                "--input",
                embeddings_csv,
                "--trained",
                trained_embeddings_path,
                "--id-col",
                id_col,
                "--out",
                matches_csv,
            ]

            with open(job.log_path, "a", encoding="utf-8") as log_file:
                log_file.write("Command: " + " ".join(match_cmd) + "\n")
                log_file.flush()
                proc = subprocess.Popen(
                    match_cmd,
                    cwd=str(BASE_DIR),
                    stdout=log_file,
                    stderr=log_file,
                )
                exit_code = proc.wait()
                job.exit_code = exit_code
                if exit_code != 0:
                    job.status = "failed"
                    job.error_message = f"Match process exited with code {exit_code}"
                    return

            job.matches_csv = matches_csv
            annotated_csv = os.path.join(
                job.out_dir, "embedding_matches_with_celltypes.csv"
            )
            self._annotate_matches(matches_csv, celltypes_path, annotated_csv)
            job.annotated_matches_csv = annotated_csv
            job.status = "succeeded"
        except Exception as exc:
            job.status = "failed"
            job.error_message = f"Job runner failed: {exc!r}"
        finally:
            job.finished_at = _utc_now()

    def _find_embeddings_csv(self, out_dir: str) -> str:
        patterns = [
            os.path.join(out_dir, "embs_by_*_emb_layer_-1.csv"),
            os.path.join(out_dir, "embs_by_*_emb_layer_-1*.csv"),
        ]
        matches = []
        for pattern in patterns:
            matches.extend(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No embedding CSV found in {out_dir}")
        matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return matches[0]

    def _annotate_matches(
        self, matches_csv: str, celltypes_csv: str, out_csv: str
    ) -> None:
        matches_df = pd.read_csv(matches_csv)
        if matches_df.empty:
            matches_df.to_csv(out_csv, index=False)
            return

        celltypes_df = pd.read_csv(celltypes_csv)
        if celltypes_df.empty:
            matches_df.to_csv(out_csv, index=False)
            return

        first_col = celltypes_df.columns[0]
        if first_col.startswith("Unnamed") or first_col == "":
            celltypes_df = celltypes_df.rename(columns={first_col: "cell_id"})
        else:
            celltypes_df = celltypes_df.rename(columns={first_col: "cell_id"})

        celltype_col = "seuratObj.CellType"
        if celltype_col not in celltypes_df.columns:
            for col in celltypes_df.columns:
                if col != "cell_id":
                    celltype_col = col
                    break

        mapping = dict(
            zip(
                celltypes_df["cell_id"].astype(str),
                celltypes_df[celltype_col].astype(str),
            )
        )

        matches_df = matches_df.rename(
            columns={
                "new_cell_id": "cell_id",
                "old_cell_id": "matched_trained_cell_id",
            }
        )
        matches_df["matched_cell_type"] = matches_df["matched_trained_cell_id"].astype(
            str
        ).map(mapping)
        matches_df.to_csv(out_csv, index=False)

