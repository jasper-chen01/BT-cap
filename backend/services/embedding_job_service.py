"""
Background job runner for embedding extraction.
"""
from __future__ import annotations

import os
import sys
import threading
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, List
from uuid import uuid4

from backend.config import settings, BASE_DIR


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EmbeddingJob:
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

    def to_dict(self) -> Dict:
        return asdict(self)


class EmbeddingJobService:
    """
    In-memory job registry for embedding extraction.
    """

    _jobs: Dict[str, EmbeddingJob] = {}
    _lock = threading.Lock()

    def start_job(
        self,
        h5ad_path: str,
        dict_dir: str,
        models_root: str,
        out_dir: Optional[str] = None,
        gene_id_type: str = "ensembl",
        gpu: str = "0",
        max_ncells: int = 1_000_000,
        forward_batch_size: int = 100,
        finetune_subdir: Optional[str] = None,
    ) -> EmbeddingJob:
        h5ad_path = os.path.abspath(h5ad_path)
        dict_dir = os.path.abspath(dict_dir)
        models_root = os.path.abspath(models_root)

        if not os.path.isfile(h5ad_path):
            raise FileNotFoundError(f"h5ad not found: {h5ad_path}")
        if not os.path.isdir(dict_dir):
            raise FileNotFoundError(f"dict dir not found: {dict_dir}")
        if not os.path.isdir(models_root):
            raise FileNotFoundError(f"models root not found: {models_root}")

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

        job = EmbeddingJob(
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
        )

        with self._lock:
            self._jobs[job_id] = job

        thread = threading.Thread(target=self._run_job, args=(job_id,), daemon=True)
        thread.start()

        return job

    def get_job(self, job_id: str) -> Optional[EmbeddingJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> List[EmbeddingJob]:
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

    def _run_job(self, job_id: str) -> None:
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
                if exit_code == 0:
                    job.status = "succeeded"
                else:
                    job.status = "failed"
                    job.error_message = f"Process exited with code {exit_code}"
        except Exception as exc:
            job.status = "failed"
            job.error_message = f"Job runner failed: {exc!r}"
        finally:
            job.finished_at = _utc_now()


