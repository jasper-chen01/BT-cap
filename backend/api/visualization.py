"""
Visualization endpoints.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
import tempfile
import os
import logging
from typing import Optional

from backend.models.schemas import VisualizationResponse
from backend.services.visualization_service import VisualizationService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/visualize", response_model=VisualizationResponse)
async def visualize_cells(
    file: UploadFile = File(..., description="Single-cell data file (h5ad format)"),
    supptable_url: Optional[str] = Form(None),
    supptable_doc_id: Optional[str] = Form(None),
    supptable_path: Optional[str] = Form(None),
    embedding_matches_path: Optional[str] = Form(None),
    de_top_n: Optional[int] = Form(15),
    cluster_resolution: Optional[float] = Form(1.0),
    use_hvg: Optional[bool] = Form(True),
    apply_filtering: Optional[bool] = Form(True),
):
    if not file.filename.endswith(".h5ad"):
        raise HTTPException(status_code=400, detail="File must be in h5ad format")

    tmp_file_path = None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5ad") as tmp_file:
        tmp_file_path = tmp_file.name
        try:
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                tmp_file.write(chunk)

            service = VisualizationService()
            result = service.process_file(
                tmp_file_path,
                supptable_url=supptable_url,
                supptable_doc_id=supptable_doc_id,
                supptable_path=supptable_path,
                embedding_matches_path=embedding_matches_path,
                source_filename=file.filename,
                de_top_n=de_top_n or 15,
                cluster_resolution=cluster_resolution or 1.0,
                use_hvg=bool(use_hvg),
                apply_filtering=bool(apply_filtering),
            )
            return result

        except Exception as exc:
            logger.exception("Visualization failed for uploaded file")
            raise HTTPException(status_code=500, detail=f"Error processing file: {exc!r}")

        finally:
            await file.close()
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                except PermissionError:
                    pass

    



