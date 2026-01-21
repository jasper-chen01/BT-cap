"""
Data processing utilities for single-cell data
"""
import numpy as np
import scanpy as sc
import pandas as pd
from typing import Optional
import anndata as ad

from backend.config import settings


class DataProcessor:
    """Service for processing single-cell data"""
    
    def load_and_preprocess(self, file_path: str) -> ad.AnnData:
        """
        Load and preprocess single-cell data
        
        Args:
            file_path: path to h5ad file
        
        Returns:
            Preprocessed AnnData object
        """
        # Load data
        adata = sc.read_h5ad(file_path)
        
        # Set gene names if needed
        if adata.var_names.isna().any() or '_index' in adata.var.columns:
            adata.var_names = adata.var.get("_index", adata.var_names)
        
        if "gene_symbol" not in adata.var.columns:
            adata.var["gene_symbol"] = adata.var_names
        
        # Preprocess if raw data exists
        if adata.raw is not None:
            adata = adata.raw.to_adata()
        
        # Basic filtering
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        
        # Normalization
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
        return adata
    
    def extract_embeddings(self, adata: ad.AnnData) -> Optional[np.ndarray]:
        """
        Extract embeddings from AnnData object
        
        Looks for embeddings in:
        1. adata.obsm (e.g., 'X_umap', 'X_pca', or custom embedding keys)
        2. Computed from raw data if needed
        
        Args:
            adata: AnnData object
        
        Returns:
            numpy array of embeddings or None if not found
        """
        # Check common embedding keys
        embedding_keys = ['X_umap', 'X_pca', 'X_embedding', 'embeddings', 'X_transformer']
        
        for key in embedding_keys:
            if key in adata.obsm:
                embeddings = adata.obsm[key]
                # If 2D UMAP, might need to use PCA or other higher-dim embeddings
                if embeddings.shape[1] < 10 and 'X_pca' in adata.obsm:
                    # Use PCA if available and UMAP is too low-dimensional
                    continue
                return embeddings
        
        # If no embeddings found, compute PCA as fallback
        if 'X_pca' not in adata.obsm:
            sc.tl.pca(adata, n_comps=50)
        
        if 'X_pca' in adata.obsm:
            return adata.obsm['X_pca']
        
        return None
    
    def load_reference_data(self):
        """
        Load reference glioma data and embedding coordinates
        
        Returns:
            tuple of (AnnData, embedding DataFrame)
        """
        # Load reference data
        adata = sc.read_h5ad(settings.REFERENCE_DATA_PATH)
        adata.var_names = adata.var.get("_index", adata.var_names)
        adata.var["gene_symbol"] = adata.var_names
        
        # Load embedding coordinates
        emb_df = pd.read_csv(settings.REFERENCE_EMBEDDING_COORDS_PATH, index_col=0)
        emb_df = emb_df.loc[adata.obs_names]
        adata.obsm["X_umap"] = emb_df.values
        
        # Preprocess
        if adata.raw is not None:
            adata = adata.raw.to_adata()
        
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
        return adata, emb_df
