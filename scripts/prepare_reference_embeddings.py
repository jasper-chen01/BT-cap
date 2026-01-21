"""
Script to prepare reference embeddings and build FAISS index

This script:
1. Loads the reference glioma data and transformer embeddings
2. Loads attention-based annotations
3. Builds a FAISS index for fast similarity search
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import scanpy as sc
import pickle
import glob
import os
from backend.config import settings
from backend.services.embedding_service import EmbeddingService
from backend.services.data_processor import DataProcessor


def load_transformer_embeddings(embeddings_dir: Path, layer: int = -1) -> np.ndarray:
    """
    Load transformer embeddings from directory
    
    Args:
        embeddings_dir: directory containing embedding files
        layer: which layer to use (default: -1 for last layer)
    
    Returns:
        numpy array of embeddings
    """
    # Look for embedding files (adjust pattern based on actual file structure)
    embedding_files = glob.glob(str(embeddings_dir / "*.npy"))
    embedding_files.extend(glob.glob(str(embeddings_dir / "*.csv")))
    embedding_files.extend(glob.glob(str(embeddings_dir / "*.pkl")))
    
    if not embedding_files:
        raise FileNotFoundError(f"No embedding files found in {embeddings_dir}")
    
    print(f"Found {len(embedding_files)} embedding files")
    
    # Try to load embeddings (adjust based on actual format)
    embeddings_list = []
    for file_path in sorted(embedding_files):
        try:
            if file_path.endswith('.npy'):
                emb = np.load(file_path)
            elif file_path.endswith('.csv'):
                emb = pd.read_csv(file_path, index_col=0).values
            elif file_path.endswith('.pkl'):
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    if isinstance(data, dict):
                        # If it's a dict with layer keys
                        if layer in data:
                            emb = data[layer]
                        else:
                            emb = list(data.values())[layer]
                    else:
                        emb = data
            
            # Handle multi-layer embeddings
            if len(emb.shape) == 3:  # (n_cells, n_layers, embedding_dim)
                emb = emb[:, layer, :]
            
            embeddings_list.append(emb)
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
            continue
    
    if not embeddings_list:
        raise ValueError("Could not load any embeddings")
    
    # Concatenate if multiple files
    if len(embeddings_list) > 1:
        embeddings = np.concatenate(embeddings_list, axis=0)
    else:
        embeddings = embeddings_list[0]
    
    print(f"Loaded embeddings shape: {embeddings.shape}")
    return embeddings


def load_annotations(annotations_path: Path) -> dict:
    """
    Load attention-based annotations
    
    Args:
        annotations_path: path to annotations file
    
    Returns:
        dictionary mapping cell_id to annotation
    """
    # Try different formats
    if annotations_path.suffix == '.csv':
        df = pd.read_csv(annotations_path, index_col=0)
        # Assume first column or column named 'annotation' contains annotations
        if 'annotation' in df.columns:
            return df['annotation'].to_dict()
        else:
            return df.iloc[:, 0].to_dict()
    elif annotations_path.suffix == '.pkl':
        with open(annotations_path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                return data
            elif isinstance(data, pd.DataFrame):
                if 'annotation' in data.columns:
                    return data['annotation'].to_dict()
                else:
                    return data.iloc[:, 0].to_dict()
    elif annotations_path.suffix == '.tsv':
        df = pd.read_csv(annotations_path, sep='\t', index_col=0)
        if 'annotation' in df.columns:
            return df['annotation'].to_dict()
        else:
            return df.iloc[:, 0].to_dict()
    
    raise ValueError(f"Could not load annotations from {annotations_path}")


def main():
    """Main function to prepare reference embeddings"""
    print("=" * 60)
    print("Preparing Reference Embeddings for BAT Portal")
    print("=" * 60)
    
    # Check if reference data exists
    if not settings.REFERENCE_DATA_PATH.exists():
        print(f"Error: Reference data not found at {settings.REFERENCE_DATA_PATH}")
        print("Please download glioma_original.h5ad and place it in data/")
        return
    
    # Load reference data
    print("\n1. Loading reference data...")
    data_processor = DataProcessor()
    adata, emb_df = data_processor.load_reference_data()
    print(f"   Loaded {adata.n_obs} cells, {adata.n_vars} genes")
    
    # Load transformer embeddings
    print("\n2. Loading transformer embeddings...")
    try:
        # Try to load from embeddings directory
        embeddings = load_transformer_embeddings(
            settings.REFERENCE_EMBEDDINGS_DIR,
            layer=settings.EMBEDDING_LAYER
        )
        
        # Ensure embeddings match cell order
        if embeddings.shape[0] != adata.n_obs:
            print(f"   Warning: Embedding shape {embeddings.shape[0]} doesn't match cells {adata.n_obs}")
            # Try to align by cell IDs if possible
            # This would require knowing the cell ID order in embeddings
            print("   Assuming embeddings are in same order as adata.obs_names")
    except Exception as e:
        print(f"   Warning: Could not load transformer embeddings: {e}")
        print("   Using UMAP coordinates as embeddings...")
        embeddings = adata.obsm.get('X_umap', emb_df.values)
    
    # Load annotations
    print("\n3. Loading annotations...")
    annotation_files = list(settings.ANNOTATIONS_DIR.glob("*"))
    if not annotation_files:
        print("   Warning: No annotation files found")
        print("   Creating placeholder annotations...")
        annotations = {cell_id: "Unknown" for cell_id in adata.obs_names}
    else:
        try:
            annotations = load_annotations(annotation_files[0])
            print(f"   Loaded annotations for {len(annotations)} cells")
        except Exception as e:
            print(f"   Error loading annotations: {e}")
            print("   Creating placeholder annotations...")
            annotations = {cell_id: "Unknown" for cell_id in adata.obs_names}
    
    # Ensure all cells have annotations
    cell_ids = adata.obs_names.values
    for cell_id in cell_ids:
        if cell_id not in annotations:
            annotations[cell_id] = "Unknown"
    
    # Build FAISS index
    print("\n4. Building FAISS index...")
    embedding_service = EmbeddingService()
    embedding_service.build_index_from_embeddings(
        embeddings=embeddings,
        cell_ids=cell_ids,
        annotations=annotations
    )
    
    print("\n" + "=" * 60)
    print("Reference embeddings prepared successfully!")
    print(f"Index size: {embedding_service.index.ntotal}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Annotations: {len(set(annotations.values()))} unique types")
    print("=" * 60)


if __name__ == "__main__":
    main()
