"""
Health check endpoints
"""
from fastapi import APIRouter
from backend.models.schemas import HealthResponse
from backend.services.embedding_service import EmbeddingService

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and data availability"""
    try:
        embedding_service = EmbeddingService()
        reference_loaded = embedding_service.is_reference_loaded()
        embeddings_indexed = embedding_service.is_index_loaded()
        
        return HealthResponse(
            status="healthy",
            message="API is running",
            reference_data_loaded=reference_loaded,
            embeddings_indexed=embeddings_indexed
        )
    except Exception as e:
        return HealthResponse(
            status="error",
            message=f"Error: {str(e)}",
            reference_data_loaded=False,
            embeddings_indexed=False
        )
