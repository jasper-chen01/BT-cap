"""
Pydantic schemas for API requests and responses
"""
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np


class AnnotationRequest(BaseModel):
    """Request model for annotation"""
    top_k: Optional[int] = 10
    similarity_threshold: Optional[float] = 0.7


class CellAnnotation(BaseModel):
    """Single cell annotation result"""
    cell_id: str
    predicted_annotation: str
    confidence_score: float
    top_matches: List[Dict[str, float]]  # List of {annotation: similarity_score}


class AnnotationResponse(BaseModel):
    """Response model for annotation"""
    total_cells: int
    annotations: List[CellAnnotation]
    metadata: Optional[Dict] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str
    reference_data_loaded: bool
    embeddings_indexed: bool


class ChatMessage(BaseModel):
    """Single chat message"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None
    file_uploaded: Optional[str] = None
    annotation_results: Optional[Dict] = None
    
    class Config:
        json_encoders = {
            # Handle numpy types if needed
        }


class ChatSession(BaseModel):
    """Chat session"""
    session_id: str
    messages: List[ChatMessage]
    uploaded_files: List[Dict] = []


class ChatResponse(BaseModel):
    """Chat agent response"""
    message: ChatMessage
    session_id: str
    suggestions: Optional[List[str]] = None
    requires_action: Optional[str] = None  # e.g., "annotate", "upload_file"
