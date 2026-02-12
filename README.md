# Brain Tumor Annotation Portal (BAT Portal)

A web-based platform for annotating glioma single-cell RNA-seq data using
embedding similarity, plus a lightweight visualization pipeline for QC and UMAP.

## Features

- **AI chat upload** with background embedding pipeline
- **Annotation API** using FAISS similarity search
- **Visualization API** using Scanpy (UMAP, Leiden, DE genes)
- **REST API** for integration and scripted runs

## Repository layout

```
├── backend/           FastAPI backend
│   ├── api/           API endpoints
│   ├── services/      Pipeline + annotation + visualization logic
│   ├── run_embeddings.py
│   └── match_embeddings.py
├── data/              Reference data + generated outputs
├── frontend/          Static HTML frontend
├── frontend-react/    Optional Vite app
├── scripts/           Offline utilities
└── requirements.txt
```

## Data paths used by the backend

These are the canonical paths used in code (all inside this repo):

- `backend/dict/` (Geneformer dictionaries)
- `backend/dirks_primary_gbm_combined_2000perCellType/` (fine-tuned model)
- `data/embeddings/embs_by_dirks_primary_gbm_combined_2000perCellType_num_classes_13_emb_layer_-1.csv`
- `data/annotations/celltypes.csv`
- `data/reference_embeddings.faiss` (built index)

## Quick start

**Python requirement:** use Python 3.10 or 3.11.

### Windows (PowerShell)

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\prepare_reference_embeddings.py
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

In a new terminal:

```powershell
cd frontend
python -m http.server 8080
```

Open:
- http://localhost:8080
- http://localhost:8000/docs

### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_reference_embeddings.py
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

In a new terminal:

```bash
cd frontend
python3 -m http.server 8080
```

### Optional React UI

```bash
cd frontend-react
npm install
npm run dev
```

## System flowcharts

```mermaid
flowchart LR
  subgraph PIPELINE[Upload -> embedding pipeline (chat upload)]
    A[Upload .h5ad in chat UI] --> B[Backend saves temp file]
    B --> C[Start background job]
    C --> D[run_embeddings.py]
    D --> E[embs_by_*_emb_layer_-1.csv]
    E --> F[match_embeddings.py]
    F --> G[embedding_matches.csv]
    G --> H[Map celltypes.csv]
    H --> I[embedding_matches_with_celltypes.csv]
    I --> J[Saved under data/embedding_runs/<h5ad_stem>_embs/]
  end

  subgraph ANNOTATE[Annotation endpoint (no files written)]
    K[Upload .h5ad to /api/annotate] --> L[Temp file saved]
    L --> M[Compute/load embeddings]
    M --> N[FAISS search]
    N --> O[Return JSON annotations]
    O --> P[Temp file deleted]
  end

  subgraph VIS[Visualization endpoint (no files written)]
    Q[Upload .h5ad to /api/visualize] --> R[Temp file saved]
    R --> S[Scanpy preprocess + UMAP + Leiden]
    S --> T[Optional supptable cell types]
    T --> U[Return JSON UMAP + clusters + DE genes]
    U --> V[Temp file deleted]
  end
```

## Outputs written to disk

Only the chat upload pipeline writes files:

- `data/embedding_runs/<h5ad_stem>_embs/embs_by_*_emb_layer_-1.csv`
- `data/embedding_runs/<h5ad_stem>_embs/embedding_matches.csv`
- `data/embedding_runs/<h5ad_stem>_embs/embedding_matches_with_celltypes.csv`

`/api/annotate` and `/api/visualize` return JSON only.

## API endpoints

Core:
- `GET /api/health`
- `POST /api/annotate`
- `POST /api/visualize`

Chat:
- `POST /api/chat/session`
- `POST /api/chat/{session_id}/message`
- `POST /api/chat/{session_id}/annotate`

Embedding pipeline:
- `POST /api/embeddings/pipeline/jobs`
- `GET /api/embeddings/pipeline/jobs/{job_id}`
- `GET /api/embeddings/pipeline/jobs/{job_id}/log`

Legacy embeddings job (run_embeddings only):
- `POST /api/embeddings/jobs`
- `GET /api/embeddings/jobs/{job_id}`
- `GET /api/embeddings/jobs/{job_id}/log`

## Troubleshooting

### Index missing
Run:
```bash
python scripts/prepare_reference_embeddings.py
```

### Port already in use
Set `PORT` or use a different HTTP port for the static frontend.

### Embedding extraction fails
Ensure the `.h5ad` has valid gene identifiers and counts. For gene symbols,
use the pipeline with `gene_id_type=symbol` (default).
