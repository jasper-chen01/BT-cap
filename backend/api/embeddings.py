"""
Embedding extraction job endpoints.
"""
from fastapi import APIRouter, HTTPException

from backend.models.schemas import (
    EmbeddingJobRequest,
    EmbeddingJobResponse,
    EmbeddingJobLogResponse,
    EmbeddingPipelineJobRequest,
    EmbeddingPipelineJobResponse,
    EmbeddingPipelineJobLogResponse,
)
from backend.services.embedding_job_service import EmbeddingJobService
from backend.services.embedding_pipeline_service import EmbeddingPipelineJobService

router = APIRouter()


@router.post("/embeddings/jobs", response_model=EmbeddingJobResponse)
async def start_embedding_job(payload: EmbeddingJobRequest):
    """
    Start a background job to run backend/run_embeddings.py.
    """
    service = EmbeddingJobService()
    try:
        job = service.start_job(
            h5ad_path=payload.h5ad_path,
            dict_dir=payload.dict_dir,
            models_root=payload.models_root,
            out_dir=payload.out_dir,
            gene_id_type=payload.gene_id_type,
            gpu=payload.gpu,
            max_ncells=payload.max_ncells,
            forward_batch_size=payload.forward_batch_size,
            finetune_subdir=payload.finetune_subdir,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to start job: {exc!r}") from exc

    return EmbeddingJobResponse(**job.to_dict())


@router.get("/embeddings/jobs/{job_id}", response_model=EmbeddingJobResponse)
async def get_embedding_job(job_id: str):
    service = EmbeddingJobService()
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return EmbeddingJobResponse(**job.to_dict())


@router.get("/embeddings/jobs/{job_id}/log", response_model=EmbeddingJobLogResponse)
async def get_embedding_job_log(job_id: str, tail: int = 200):
    service = EmbeddingJobService()
    try:
        text = service.read_log(job_id, tail=tail)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")
    return EmbeddingJobLogResponse(job_id=job_id, tail=tail, log=text)


@router.post("/embeddings/pipeline/jobs", response_model=EmbeddingPipelineJobResponse)
async def start_embedding_pipeline_job(payload: EmbeddingPipelineJobRequest):
    """
    Start a background job to run backend/run_embeddings.py -> backend/match_embeddings.py
    and map matched cell types.
    """
    service = EmbeddingPipelineJobService()
    try:
        job = service.start_job(
            h5ad_path=payload.h5ad_path,
            dict_dir=payload.dict_dir,
            models_root=payload.models_root,
            out_dir=payload.out_dir,
            gene_id_type=payload.gene_id_type,
            gpu=payload.gpu,
            max_ncells=payload.max_ncells,
            forward_batch_size=payload.forward_batch_size,
            finetune_subdir=payload.finetune_subdir,
            trained_embeddings_path=payload.trained_embeddings_path,
            celltypes_path=payload.celltypes_path,
            id_col=payload.id_col,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to start job: {exc!r}") from exc

    return EmbeddingPipelineJobResponse(**job.to_dict())


@router.get(
    "/embeddings/pipeline/jobs/{job_id}",
    response_model=EmbeddingPipelineJobResponse,
)
async def get_embedding_pipeline_job(job_id: str):
    service = EmbeddingPipelineJobService()
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return EmbeddingPipelineJobResponse(**job.to_dict())


@router.get(
    "/embeddings/pipeline/jobs/{job_id}/log",
    response_model=EmbeddingPipelineJobLogResponse,
)
async def get_embedding_pipeline_job_log(job_id: str, tail: int = 200):
    service = EmbeddingPipelineJobService()
    try:
        text = service.read_log(job_id, tail=tail)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found")
    return EmbeddingPipelineJobLogResponse(job_id=job_id, tail=tail, log=text)


