"""
Service for managing reference embeddings and FAISS index
"""
import numpy as np
import faiss
import pickle
import os
from pathlib import Path
from typing import Optional, Tuple
import scanpy as sc
import pandas as pd

from backend.config import settings


class EmbeddingService:
    """Service for managing and querying reference embeddings"""
    
    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.reference_annotations: Optional[dict] = None
        self.reference_cell_ids: Optional[np.ndarray] = None
        self._load_index()
        self._load_annotations()
    
    def _load_index(self):
        """Load FAISS index if it exists"""
        if os.path.exists(settings.FAISS_INDEX_PATH):
            try:
                self.index = faiss.read_index(str(settings.FAISS_INDEX_PATH))
                # Load corresponding cell IDs
                cell_ids_path = settings.DATA_DIR / "reference_cell_ids.pkl"
                if cell_ids_path.exists():
                    with open(cell_ids_path, 'rb') as f:
                        self.reference_cell_ids = pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load FAISS index: {e}")
                self.index = None
    
    def _load_annotations(self):
        """Load reference annotations"""
        if os.path.exists(settings.ANNOTATIONS_PATH):
            try:
                with open(settings.ANNOTATIONS_PATH, 'rb') as f:
                    self.reference_annotations = pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load annotations: {e}")
                self.reference_annotations = None
    
    def is_reference_loaded(self) -> bool:
        """Check if reference data is available"""
        return os.path.exists(settings.REFERENCE_DATA_PATH)
    
    def is_index_loaded(self) -> bool:
        """Check if FAISS index is loaded"""
        return self.index is not None
    
    def build_index_from_embeddings(
        self,
        embeddings: np.ndarray,
        cell_ids: np.ndarray,
        annotations: dict
    ):
        """
        Build FAISS index from embeddings
        
        Args:
            embeddings: numpy array of shape (n_cells, embedding_dim)
            cell_ids: array of cell IDs corresponding to embeddings
            annotations: dictionary mapping cell_id to annotation
        """
        # Normalize embeddings for cosine similarity
        embeddings = np.asarray(embeddings, dtype="float32", order="C")
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index (inner product for cosine similarity)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product = cosine for normalized vectors
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        # Store cell IDs and annotations
        self.reference_cell_ids = cell_ids
        self.reference_annotations = annotations
        
        # Save index
        faiss.write_index(self.index, str(settings.FAISS_INDEX_PATH))
        
        # Save cell IDs
        cell_ids_path = settings.DATA_DIR / "reference_cell_ids.pkl"
        with open(cell_ids_path, 'wb') as f:
            pickle.dump(cell_ids, f)
        
        # Save annotations
        with open(settings.ANNOTATIONS_PATH, 'wb') as f:
            pickle.dump(annotations, f)
        
        print(f"Built FAISS index with {self.index.ntotal} embeddings")
    
    def search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors in reference embeddings
        
        Args:
            query_embeddings: numpy array of shape (n_queries, embedding_dim)
            top_k: number of nearest neighbors to return
        
        Returns:
            distances: array of shape (n_queries, top_k) with similarity scores
            indices: array of shape (n_queries, top_k) with indices into reference
        """
        if self.index is None:
            raise ValueError("FAISS index not loaded. Please build index first.")
        
        # Normalize query embeddings
        query_embeddings = np.asarray(query_embeddings, dtype="float32", order="C")
        faiss.normalize_L2(query_embeddings)
        
        # Search
        distances, indices = self.index.search(query_embeddings, top_k)
        
        return distances, indices
    
    def get_annotation_for_cell(self, cell_id: str) -> Optional[str]:
        """Get annotation for a reference cell ID"""
        if self.reference_annotations is None:
            return None
        return self.reference_annotations.get(cell_id)
    
    def get_cell_id_at_index(self, index: int) -> Optional[str]:
        """Get cell ID at a given index in the reference"""
        if self.reference_cell_ids is None:
            return None
        if index < len(self.reference_cell_ids):
            return self.reference_cell_ids[index]
        return None
