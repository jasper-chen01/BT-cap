"""
Annotation endpoints
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional
import os
import logging
from pathlib import Path
from uuid import uuid4

from backend.models.schemas import AnnotationResponse, CellAnnotation
from backend.services.annotation_service import AnnotationService
from backend.services.embedding_pipeline_service import EmbeddingPipelineJobService
from backend.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/annotate", response_model=AnnotationResponse)
async def annotate_cells(
    file: UploadFile = File(..., description="Single-cell data file (h5ad format)"),
    top_k: Optional[int] = Form(10),
    similarity_threshold: Optional[float] = Form(0.7)
):
    """
    Annotate single-cell data by matching against reference embeddings
    
    Args:
        file: h5ad file containing single-cell data
        top_k: Number of nearest neighbors to consider
        similarity_threshold: Minimum similarity score for annotation
    
    Returns:
        AnnotationResponse with predicted annotations for each cell
    """
    if not file.filename.endswith('.h5ad'):
        raise HTTPException(
            status_code=400,
            detail="File must be in h5ad format"
        )
    
    uploads_dir = settings.DATA_DIR / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename).name
    suffix = Path(safe_name).suffix or ".h5ad"
    upload_path = uploads_dir / f"{Path(safe_name).stem}_{uuid4().hex}{suffix}"

    try:
        content = await file.read()
        with open(upload_path, "wb") as handle:
            handle.write(content)

        # Always run standard annotation
        annotation_service = AnnotationService()
        result = await annotation_service.annotate_file(
            str(upload_path),
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )

        # Start embedding pipeline job in the background
        pipeline_job = None
        try:
            pipeline_service = EmbeddingPipelineJobService()
            pipeline_job = pipeline_service.start_job(h5ad_path=str(upload_path))
        except Exception as exc:
            logger.warning("Failed to start embedding pipeline: %s", exc)

        if result.metadata is None:
            result.metadata = {}
        if pipeline_job is not None:
            result.metadata["embedding_pipeline_job"] = pipeline_job.to_dict()
        else:
            result.metadata["embedding_pipeline_job"] = {"status": "not_started"}

        return result

    except Exception as e:
        logger.exception("Annotation failed for uploaded file")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {e!r}"
        )
    finally:
        await file.close()


@router.get("/annotate/status")
async def annotation_status():
    """Get status of annotation system"""
    try:
        annotation_service = AnnotationService()
        status = annotation_service.get_status()
        return JSONResponse(content=status)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting status: {str(e)}"
        )
