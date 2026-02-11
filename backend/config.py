"""
Configuration settings
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory
BASE_DIR = Path(__file__).parent.parent
DEFAULT_CREDENTIALS_DIR = BASE_DIR / "backend" / "credentials"

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

    # Firestore settings
    # Default to empty so the service account project_id is used when available.
    FIRESTORE_PROJECT_ID: str = os.getenv("FIRESTORE_PROJECT_ID", "")
    FIRESTORE_COLLECTION: str = os.getenv("FIRESTORE_COLLECTION", "users")
    # Optional Firestore database id (non-default)
    FIRESTORE_DATABASE_ID: str = os.getenv("FIRESTORE_DATABASE_ID", "")
    FIRESTORE_SUPPTABLE_COLLECTION: str = os.getenv(
        "FIRESTORE_SUPPTABLE_COLLECTION",
        "supptables"
    )
    SUPPTABLE_DOC_ID: str = os.getenv("SUPPTABLE_DOC_ID", "supptable1")
    SUPPTABLE_URL: str = os.getenv("SUPPTABLE_URL", "")
    
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

        if not self.GOOGLE_APPLICATION_CREDENTIALS and DEFAULT_CREDENTIALS_DIR.exists():
            json_files = sorted(DEFAULT_CREDENTIALS_DIR.glob("*.json"))
            if json_files:
                self.GOOGLE_APPLICATION_CREDENTIALS = str(json_files[0])

settings = Settings()
