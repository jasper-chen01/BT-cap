"""
Pydantic schemas for API requests and responses
"""
from pydantic import BaseModel, validator
from typing import List, Dict, Optional
import numpy as np


class AnnotationRequest(BaseModel):
    """Request model for annotation"""
    top_k: Optional[int] = 10
    similarity_threshold: Optional[float] = 0.7


class TopMatch(BaseModel):
    annotation: str
    similarity: float


class CellAnnotation(BaseModel):
    """Single cell annotation result"""
    cell_id: str
    predicted_annotation: str
    confidence_score: float
    top_matches: List[TopMatch]


class AnnotationResponse(BaseModel):
    """Response model for annotation"""
    total_cells: int
    annotations: List[CellAnnotation]
    metadata: Optional[Dict] = None


class UmapPoint(BaseModel):
    cell_id: str
    x: float
    y: float
    cluster: str
    cell_type: Optional[str] = None
    score: Optional[float] = None
    predicted_cell_type: Optional[str] = None
    predicted_score: Optional[float] = None

    class Config:
        extra = "allow"


class AnnotationRequest(BaseModel):
    """Request model for annotation"""
    top_k: Optional[int] = 10
    similarity_threshold: Optional[float] = 0.7


class TopMatch(BaseModel):
    cell_id: str
    x: float
    y: float
    cluster: str
    cell_type: Optional[str] = None
    score: Optional[float] = None


class CellTypeSummary(BaseModel):
    name: str
    count: int
    avg_score: Optional[float] = None


class DifferentialExpressionGroup(BaseModel):
    group: str
    genes: List[str]
    scores: Optional[List[Optional[float]]] = None
    logfoldchanges: Optional[List[Optional[float]]] = None
    pvals_adj: Optional[List[Optional[float]]] = None
    gene_annotations: Optional[List[Dict]] = None


class VisualizationResponse(BaseModel):
    total_cells: int
    umap_points: List[UmapPoint]
    cluster_labels: List[str]
    cluster_counts: Dict[str, int]
    cell_types: Optional[List[CellTypeSummary]] = None
    de_by_cluster: Optional[List[DifferentialExpressionGroup]] = None
    de_by_cell_type: Optional[List[DifferentialExpressionGroup]] = None
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
    analysis_context: Optional[Dict] = None

class ChatResponse(BaseModel):
    """Chat agent response"""
    message: ChatMessage
    session_id: str
    suggestions: Optional[List[str]] = None
    requires_action: Optional[str] = None  # e.g., "annotate", "upload_file"


class SignUpRequest(BaseModel):
    """Request model for user signup"""
    name: str
    email: str
    password: str

    @validator("password")
    def password_min_length(cls, value: str) -> str:
        # Enforce a minimum length; max length is handled by the hash scheme.
        if value is None:
            return value
        if len(value) < 4:
            raise ValueError("Password must be at least 4 characters long.")
        return value


class SignInRequest(BaseModel):
    """Request model for user signin"""
    email: str
    password: str

    @validator("password")
    def password_min_length(cls, value: str) -> str:
        # Enforce a minimum length; max length is handled by the hash scheme.
        if value is None:
            return value
        if len(value) < 4:
            raise ValueError("Password must be at least 4 characters long.")
        return value


class AuthResponse(BaseModel):
    """Response model for auth"""
    user_id: str
    name: str
    email: str
class EmbeddingJobRequest(BaseModel):
    """Request model to start embedding extraction"""
    h5ad_path: str
    dict_dir: str
    models_root: str
    out_dir: Optional[str] = None
    gene_id_type: str = "ensembl"
    gpu: str = "0"
    max_ncells: int = 1_000_000
    forward_batch_size: int = 100
    finetune_subdir: Optional[str] = None


class EmbeddingJobResponse(BaseModel):
    """Status of an embedding extraction job"""
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


class EmbeddingJobLogResponse(BaseModel):
    """Log output for an embedding extraction job"""
    job_id: str
    tail: int
    log: str


class EmbeddingPipelineJobRequest(BaseModel):
    """Request model to start embedding + matching pipeline"""
    h5ad_path: str
    dict_dir: Optional[str] = None
    models_root: Optional[str] = None
    out_dir: Optional[str] = None
    gene_id_type: str = "symbol"
    gpu: str = "0"
    max_ncells: int = 1_000_000
    forward_batch_size: int = 100
    finetune_subdir: Optional[str] = None
    trained_embeddings_path: Optional[str] = None
    celltypes_path: Optional[str] = None
    id_col: str = "individual"


class EmbeddingPipelineJobResponse(BaseModel):
    """Status of an embedding pipeline job"""
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


class EmbeddingPipelineJobLogResponse(BaseModel):
    """Log output for an embedding pipeline job"""
    job_id: str
    tail: int
    log: str