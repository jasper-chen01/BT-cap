"""
Service for annotating new single-cell data
"""
import numpy as np
import scanpy as sc
import pandas as pd
from typing import Dict, List
from collections import Counter

from backend.models.schemas import AnnotationResponse, CellAnnotation
from backend.services.embedding_service import EmbeddingService
from backend.services.data_processor import DataProcessor


class AnnotationService:
    """Service for annotating single-cell data"""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.data_processor = DataProcessor()
    
    async def annotate_file(
        self,
        file_path: str,
        top_k: int = 10,
        similarity_threshold: float = 0.7
    ) -> AnnotationResponse:
        """
        Annotate cells in a new h5ad file
        
        Args:
            file_path: path to h5ad file
            top_k: number of nearest neighbors to consider
            similarity_threshold: minimum similarity score
        
        Returns:
            AnnotationResponse with annotations for each cell
        """
        # Load and preprocess new data
        adata = self.data_processor.load_and_preprocess(file_path)
        
        # Extract embeddings (assuming they're in adata.obsm or need to be computed)
        query_embeddings = self.data_processor.extract_embeddings(adata)
        
        if query_embeddings is None:
            raise ValueError("Could not extract embeddings from data. Ensure embeddings are available.")
        
        # Search for nearest neighbors
        distances, indices = self.embedding_service.search(
            query_embeddings,
            top_k=top_k
        )
        
        # Annotate each cell
        cell_annotations = []
        cell_ids = adata.obs_names.values
        
        for i, cell_id in enumerate(cell_ids):
            # Get top matches
            top_matches = []
            match_annotations = []
            
            for j in range(top_k):
                ref_idx = indices[i, j]
                similarity = float(distances[i, j])
                
                if similarity >= similarity_threshold:
                    ref_cell_id = self.embedding_service.get_cell_id_at_index(ref_idx)
                    annotation = self.embedding_service.get_annotation_for_cell(ref_cell_id)
                    
                    if annotation:
                        top_matches.append({
                            "annotation": annotation,
                            "similarity": similarity
                        })
                        match_annotations.append(annotation)
            
            # Predict annotation (majority vote from top matches)
            if match_annotations:
                # Weight by similarity
                weighted_annotations = []
                for match in top_matches:
                    weighted_annotations.extend([match["annotation"]] * int(match["similarity"] * 100))
                
                predicted_annotation = Counter(weighted_annotations).most_common(1)[0][0]
                confidence = match_annotations.count(predicted_annotation) / len(match_annotations)
            else:
                predicted_annotation = "Unknown"
                confidence = 0.0
            
            cell_annotations.append(CellAnnotation(
                cell_id=cell_id,
                predicted_annotation=predicted_annotation,
                confidence_score=confidence,
                top_matches=top_matches
            ))
        
        return AnnotationResponse(
            total_cells=len(cell_annotations),
            annotations=cell_annotations,
            metadata={
                "top_k": top_k,
                "similarity_threshold": similarity_threshold,
                "n_cells_annotated": len(cell_annotations)
            }
        )
    
    def get_status(self) -> Dict:
        """Get status of annotation system"""
        return {
            "reference_loaded": self.embedding_service.is_reference_loaded(),
            "index_loaded": self.embedding_service.is_index_loaded(),
            "index_size": self.embedding_service.index.ntotal if self.embedding_service.index else 0
        }
