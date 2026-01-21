"""
Annotation endpoints
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional
import tempfile
import os
import logging

from backend.models.schemas import AnnotationResponse, CellAnnotation
from backend.services.annotation_service import AnnotationService

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
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.h5ad') as tmp_file:
        try:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
            
            # Annotate using the service
            annotation_service = AnnotationService()
            result = await annotation_service.annotate_file(
                tmp_file_path,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
            
            return result
            
        except Exception as e:
            logger.exception("Annotation failed for uploaded file")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing file: {e!r}"
            )
        finally:
            await file.close()
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                except PermissionError:
                    # Windows can keep the file locked briefly; ignore cleanup failure.
                    pass


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
