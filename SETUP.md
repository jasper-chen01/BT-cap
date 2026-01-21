# Setup Guide for BAT Portal

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Access to Box links for downloading reference data

## Step-by-Step Setup

### 1. Clone or Download the Project

Navigate to the project directory:
```bash
cd capstone_BAT_portal_012125
```

### 2. Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Reference Data

Create the data directory structure:
```bash
mkdir -p data/embeddings
mkdir -p data/annotations
```

Download the following files from Box and place them in the appropriate locations:

#### Required Files:

1. **Reference glioma data** (`glioma_original.h5ad`)
   - Source: https://bcm.box.com/s/fcslrsl6bcfr6hgux7mdw1sbyazmjkzb
   - Destination: `data/glioma_original.h5ad`

2. **Embedding coordinates** (`embedding_coordinates.csv`)
   - Source: Same Box link as above (should be included)
   - Destination: `data/embedding_coordinates.csv`

3. **Transformer embeddings**
   - Source: https://bcm.box.com/s/x5vvac8ug0exxpqe7wlfng3zzk4zkhmg
   - Note: Use layer -1 from the transformer model
   - Destination: `data/embeddings/` (can be .npy, .csv, or .pkl format)

4. **Attention-based annotations**
   - Source: https://bcm.box.com/s/q6zu7g0dbow081wwm8kz3rmgzdyicko6
   - Destination: `data/annotations/` (can be .csv, .tsv, or .pkl format)

### 5. Prepare Reference Embeddings

Run the preparation script to build the FAISS index:

```bash
python scripts/prepare_reference_embeddings.py
```

This script will:
- Load the reference glioma data
- Load transformer embeddings (layer -1)
- Load attention-based annotations
- Build a FAISS index for fast similarity search
- Save the index and metadata to `data/`

Expected output:
```
============================================================
Preparing Reference Embeddings for BAT Portal
============================================================

1. Loading reference data...
   Loaded X cells, Y genes

2. Loading transformer embeddings...
   Loaded embeddings shape: (X, embedding_dim)

3. Loading annotations...
   Loaded annotations for X cells

4. Building FAISS index...
Built FAISS index with X embeddings

============================================================
Reference embeddings prepared successfully!
Index size: X
Embedding dimension: Y
Annotations: Z unique types
============================================================
```

### 6. Verify Setup

Check that the following files were created:
- `data/reference_embeddings.faiss` (FAISS index)
- `data/reference_cell_ids.pkl` (Cell ID mapping)
- `data/reference_annotations.pkl` (Annotation mapping)

### 7. Start the Servers

#### Option A: Using the startup script
```bash
./start_server.sh
```

#### Option B: Manual start

**Terminal 1 - Backend:**
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - Frontend:**
```bash
cd frontend && python3 -m http.server 8080
```

### 8. Access the Portal

- **Web Interface**: http://localhost:8080
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

## Troubleshooting

### Issue: "Reference data not found"
**Solution**: Ensure `glioma_original.h5ad` is in the `data/` directory

### Issue: "No embedding files found"
**Solution**: 
- Check that transformer embeddings are in `data/embeddings/`
- Verify file format (.npy, .csv, or .pkl)
- Check that the script can read the embedding format

### Issue: "Could not load annotations"
**Solution**:
- Ensure annotation file is in `data/annotations/`
- Check file format (CSV should have 'annotation' column or first column)
- Verify cell IDs match between data and annotations

### Issue: Port already in use
**Solution**: 
- Change ports in `backend/config.py` or use environment variables
- For frontend, use: `python3 -m http.server 8081`

### Issue: FAISS installation fails
**Solution**:
- Try: `pip install faiss-cpu --no-cache-dir`
- For GPU support: `pip install faiss-gpu` (requires CUDA)

## Data Format Requirements

### Input h5ad Files
- Must contain single-cell expression data
- Should have embeddings in `adata.obsm` (e.g., 'X_umap', 'X_pca', 'X_embedding')
- If no embeddings found, PCA will be computed automatically

### Embedding Files
Supported formats:
- `.npy`: NumPy array of shape (n_cells, embedding_dim) or (n_cells, n_layers, embedding_dim)
- `.csv`: CSV file with embeddings (first column as index)
- `.pkl`: Pickle file containing numpy array or dict with layer keys

### Annotation Files
Supported formats:
- `.csv`: CSV with cell IDs as index and 'annotation' column
- `.tsv`: Tab-separated values with same structure
- `.pkl`: Pickle file with dict mapping cell_id -> annotation

## Next Steps

Once setup is complete:
1. Test the API using `scripts/example_usage.py`
2. Upload a test h5ad file through the web interface
3. Review annotation results and adjust parameters if needed

## Support

For issues or questions, check:
- API documentation at http://localhost:8000/docs
- Example usage in `scripts/example_usage.py`
- README.md for general information
