"""
Configuration settings
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

class Settings:
    """Application settings"""
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    
    # Data paths
    DATA_DIR: Path = BASE_DIR / "data"
    REFERENCE_EMBEDDINGS_DIR: Path = DATA_DIR / "embeddings"
    ANNOTATIONS_DIR: Path = DATA_DIR / "annotations"
    REFERENCE_DATA_PATH: Path = DATA_DIR / "glioma_original.h5ad"
    REFERENCE_EMBEDDING_COORDS_PATH: Path = DATA_DIR / "embedding_coordinates.csv"
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
