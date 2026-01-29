"""
Configuration settings
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Load environment variables from .env at repo root (if present)
load_dotenv(BASE_DIR / ".env")

class Settings:
    """Application settings"""
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))

    # Gemini settings (optional)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    VERTEX_PROJECT_ID: str = os.getenv("VERTEX_PROJECT_ID", "")
    VERTEX_LOCATION: str = os.getenv("VERTEX_LOCATION", "us-central1")
    GOOGLE_APPLICATION_CREDENTIALS: str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    
    # Data paths
    DATA_DIR: Path = BASE_DIR / "data"
    REFERENCE_EMBEDDINGS_DIR: Path = DATA_DIR / "embeddings"
    ANNOTATIONS_DIR: Path = DATA_DIR / "annotations"
    REFERENCE_DATA_PATH: Path = DATA_DIR / "adata.h5ad"
    REFERENCE_EMBEDDING_COORDS_PATH: Path = REFERENCE_EMBEDDINGS_DIR / "embedding_coordinates.csv"
    FAISS_INDEX_PATH: Path = DATA_DIR / "reference_embeddings.faiss"
    ANNOTATIONS_PATH: Path = DATA_DIR / "reference_annotations.pkl"
    
    # Embedding settings
    EMBEDDING_LAYER: int = -1  # Use layer -1 from transformer
    
    # Similarity search settings
    TOP_K: int = 10  # Number of nearest neighbors to retrieve
    SIMILARITY_THRESHOLD: float = 0.7  # Minimum similarity score
    
    # Create directories if they don't exist
    def __init__(self):
        self.DATA_DIR.mkdir(exist_ok=True)
        self.REFERENCE_EMBEDDINGS_DIR.mkdir(exist_ok=True)
        self.ANNOTATIONS_DIR.mkdir(exist_ok=True)

settings = Settings()
