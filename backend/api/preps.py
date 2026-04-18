"""
PREPS pipeline integration: runs preps/generate_preds.py in a subprocess (conda preps env),
then preps/patchseq_predict.py for electrophysiology predictions when enabled, then serves
Scanpy UMAP visualization JSON for the electrophysiology portal tab.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from backend.config import settings
from backend.services.preps_job_service import preps_job_service

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/preps/status")
async def preps_status() -> Dict[str, Any]:
    return {
        "preps_available": preps_job_service.preps_available(),
        "preps_dir": str(settings.PREPS_DIR),
    }


@router.get("/preps/config")
async def preps_config() -> Dict[str, Any]:
    return {
        "preps_available": preps_job_service.preps_available(),
        "preps_dir": str(settings.PREPS_DIR),
        "preps_models_root": str(settings.PREPS_MODELS_ROOT),
        "preps_dict_dir": str(settings.PREPS_DICT_DIR),
        "preps_python_configured": bool(settings.PREPS_PYTHON),
        "default_reference_substring": settings.PREPS_DEFAULT_REFERENCE_SUBSTRING,
        "run_patchseq_predict": settings.PREPS_RUN_PATCHSEQ_PREDICT,
    }


@router.post("/preps/jobs")
async def start_preps_job(
    file: UploadFile = File(..., description="Input single-cell .h5ad for PREPS"),
    species: str = Form("human"),
    gpu: str = Form("0"),
    reference_substring: Optional[str] = Form(None),
    de_top_n: Optional[int] = Form(15),
    cluster_resolution: Optional[float] = Form(1.0),
):
    if not file.filename or not str(file.filename).lower().endswith(".h5ad"):
        raise HTTPException(status_code=400, detail="Upload a .h5ad file")

    if species not in ("human", "mouse"):
        raise HTTPException(status_code=400, detail="species must be human or mouse")

    if not preps_job_service.preps_available():
        raise HTTPException(
            status_code=503,
            detail=(
                "PREPS is not configured. Set PREPS_PYTHON in .env to your conda preps "
                "interpreter and ensure dict/ and fine-tuned_models/ paths are valid "
                "(see preps/HOWTO_PREPS.md)."
            ),
        )

    jobs_root = settings.DATA_DIR / "preps_jobs"
    jobs_root.mkdir(parents=True, exist_ok=True)
    job_id = str(uuid4())
    work_dir = jobs_root / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    upload_path = work_dir / "upload.h5ad"

    try:
        with open(upload_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)

        job = preps_job_service.start_job(
            job_id,
            str(upload_path),
            species=species,
            gpu=str(gpu),
            reference_substring=reference_substring,
            de_top_n=int(de_top_n or 15),
            cluster_resolution=float(cluster_resolution or 1.0),
        )
        return {"job_id": job.job_id, "test_name": job.test_name, "status": job.status}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to start PREPS job")
        raise HTTPException(status_code=500, detail=repr(exc))
    finally:
        await file.close()


@router.get("/preps/jobs/{job_id}")
async def get_preps_job(job_id: str, include_log_tail: int = 80) -> Dict[str, Any]:
    job = preps_job_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job_id")

    payload = job.to_dict()
    if include_log_tail and include_log_tail > 0:
        try:
            payload["log_tail"] = preps_job_service.read_log(job_id, tail=include_log_tail)
        except Exception:
            payload["log_tail"] = ""
    return payload


@router.get("/preps/jobs/{job_id}/log")
async def get_preps_job_log(job_id: str, tail: int = 400) -> Dict[str, str]:
    try:
        return {"log": preps_job_service.read_log(job_id, tail=tail)}
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown job_id")
