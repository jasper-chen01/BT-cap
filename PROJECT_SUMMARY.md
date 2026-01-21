# Brain Tumor Annotation Portal - Project Summary

## Overview

The Brain Tumor Annotation Portal (BAT Portal) is a web-based platform for annotating glioma single-cell RNA-seq data using vector embeddings and attention-based cell annotations. It provides both a user-friendly web interface and a RESTful API for programmatic access.

## Architecture

### Backend (FastAPI)
- **API Layer** (`backend/api/`):
  - `health.py`: Health check endpoints
  - `annotation.py`: Main annotation endpoint for processing uploaded data

- **Services Layer** (`backend/services/`):
  - `embedding_service.py`: Manages FAISS index for fast similarity search
  - `annotation_service.py`: Core annotation logic using nearest neighbor matching
  - `data_processor.py`: Handles loading and preprocessing of single-cell data

- **Models** (`backend/models/`):
  - `schemas.py`: Pydantic models for API requests/responses

- **Configuration** (`backend/config.py`):
  - Centralized settings and paths

### Frontend
- **Web Interface** (`frontend/index.html`):
  - Modern, responsive UI for data upload
  - Real-time annotation visualization
  - Results table with confidence scores
  - Download functionality for results

### Scripts
- **prepare_reference_embeddings.py**: Builds FAISS index from reference data
- **example_usage.py**: Example Python client for API usage

## Key Features

1. **Fast Similarity Search**
   - Uses FAISS (Facebook AI Similarity Search) for efficient vector matching
   - Supports cosine similarity via normalized inner product
   - Configurable top-k nearest neighbors

2. **Embedding-based Annotation**
   - Matches new single-cell data against reference embeddings
   - Uses attention-based annotations from in-house glioma datasets
   - Majority vote prediction with confidence scoring

3. **Flexible Data Processing**
   - Supports multiple embedding formats (.npy, .csv, .pkl)
   - Automatic PCA fallback if embeddings not found
   - Handles various annotation file formats

4. **User-Friendly Interface**
   - Drag-and-drop file upload
   - Real-time processing status
   - Interactive results visualization
   - CSV export functionality

5. **RESTful API**
   - OpenAPI/Swagger documentation
   - Programmatic access for integration
   - Health check endpoints

## Workflow

1. **Setup Phase**:
   - Download reference data (glioma data, embeddings, annotations)
   - Run `prepare_reference_embeddings.py` to build FAISS index
   - Start backend and frontend servers

2. **Annotation Phase**:
   - User uploads new single-cell data (h5ad format)
   - System extracts embeddings from uploaded data
   - FAISS searches for nearest neighbors in reference space
   - Predictions made via majority vote of top matches
   - Results returned with confidence scores

## Technology Stack

- **Backend**: FastAPI, Uvicorn
- **Data Processing**: Scanpy, AnnData, Pandas, NumPy
- **Vector Search**: FAISS
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **ML/Science**: scikit-learn, SciPy

## File Structure

```
capstone_BAT_portal_012125/
├── backend/
│   ├── api/              # API endpoints
│   ├── services/         # Business logic
│   ├── models/           # Data models
│   ├── utils/            # Utilities
│   ├── config.py         # Configuration
│   └── main.py           # FastAPI app
├── frontend/
│   └── index.html        # Web interface
├── scripts/
│   ├── prepare_reference_embeddings.py
│   └── example_usage.py
├── data/                 # Data storage (created at runtime)
├── requirements.txt      # Python dependencies
├── README.md             # Main documentation
├── SETUP.md              # Setup guide
└── start_server.sh       # Startup script
```

## API Endpoints

- `GET /api/health` - System health and data availability
- `POST /api/annotate` - Annotate uploaded single-cell data
- `GET /api/annotate/status` - Annotation system status

## Configuration

Key settings in `backend/config.py`:
- `EMBEDDING_LAYER`: Which transformer layer to use (default: -1)
- `TOP_K`: Default number of nearest neighbors (default: 10)
- `SIMILARITY_THRESHOLD`: Minimum similarity for annotation (default: 0.7)

## Data Requirements

### Reference Data
- Glioma single-cell data (h5ad format)
- Transformer model embeddings (layer -1)
- Attention-based cell annotations

### Query Data
- Single-cell data in h5ad format
- Should contain embeddings in `adata.obsm` or will compute PCA

## Future Enhancements

Potential improvements:
1. **AI Agent Integration**: Add chat-based interface similar to Owkin
2. **Batch Processing**: Support for multiple files
3. **Visualization**: UMAP/t-SNE plots of annotated cells
4. **Authentication**: User accounts and data privacy
5. **Database**: Store annotation history and results
6. **Advanced Matching**: Weighted similarity, ensemble methods
7. **Export Formats**: Support for more output formats (JSON, Excel)

## References

- Inspired by Cellarium Cell Annotation Service: https://cellarium.ai/tool/cellarium-cell-annotation-service-cas/
- Inspired by Owkin AI Agent: https://k.owkin.com/chat/ee64bf17-e1ab-458e-b0e7-fe9f9b5d6514
