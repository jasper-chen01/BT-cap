# Brain Tumor Annotation Portal (BAT Portal)

A web-based platform for annotating glioma single-cell RNA-seq data using vector embeddings and attention-based cell annotations.

## Features

- **AI Chat Agent**: Conversational interface for natural language interaction (similar to Owkin)
- **Embedding-based Annotation**: Match new single-cell data against in-house annotated embeddings
- **Fast Similarity Search**: Uses FAISS for efficient vector similarity matching
- **Web Portal Interface**: Upload data, visualize annotations, and download results
- **RESTful API**: Programmatic access for integration with other tools

## Project Structure

```
├── backend/           # FastAPI backend server
│   ├── api/          # API endpoints
│   ├── services/     # Business logic (embedding matching, annotation)
│   ├── models/       # Data models
│   └── utils/        # Utility functions
├── frontend/         # Web interface
├── data/             # Data storage (reference embeddings, annotations)
├── scripts/          # Data processing scripts
└── requirements.txt  # Python dependencies
```

## Quick Start

**Python requirement:** Use Python 3.10 or 3.11. Python 3.12+ (including 3.13) is not supported by some dependencies in `requirements.txt` (notably `pandas`).

### Option 1: Using the startup script (macOS/Linux)
```bash
./start_server.sh
```

This will:
- Create a virtual environment (if needed)
- Install dependencies
- Start both backend and frontend servers

### Option 2: Manual Setup (macOS/Linux)

1. **Create virtual environment** (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download reference data**:
   - Download `glioma_original.h5ad` from Box and place in `data/`
   - Download `embedding_coordinates.csv` from Box and place in `data/`
   - Download transformer embeddings from Box and place in `data/embeddings/`
   - Download attention-based annotations from Box and place in `data/annotations/`

4. **Initialize reference embeddings**:
```bash
python scripts/prepare_reference_embeddings.py
```

5. **Start the backend server**:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

6. **Start the frontend** (in a new terminal):
```bash
cd frontend && python3 -m http.server 8080
```

7. **Access the portal**:
   - Web interface: http://localhost:8080
   - API documentation: http://localhost:8000/docs
   - API endpoint: http://localhost:8000/api

### Option 3: Manual Setup (Windows PowerShell)
1. **Create a Python 3.11 virtual environment**:
```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

2. **Install dependencies**:
```powershell
pip install -r requirements.txt
```

3. **Download reference data** (same as above)

4. **Initialize reference embeddings**:
```powershell
python scripts\prepare_reference_embeddings.py
```

5. **Start the backend server**:
```powershell
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

6. **Start the frontend** (in a new PowerShell window):
```powershell
cd frontend
python -m http.server 8080
```

7. **Access the portal**:
   - Web interface: http://localhost:8080
   - API documentation: http://localhost:8000/docs
   - API endpoint: http://localhost:8000/api

### Optional: React Frontend (Vite)
If you prefer a React-based UI, a Vite app is available in `frontend-react/`.

```bash
cd frontend-react
npm install
npm run dev
```

Then open: http://localhost:5173

## Usage

### AI Chat Agent (Recommended)
1. Navigate to `http://localhost:8080/chat.html`
2. Start chatting with the AI agent
3. Upload files by dragging and dropping or using the upload button
4. Ask questions like "annotate my data" or "analyze with top k 15"
5. Get conversational responses with annotation results

### Web Portal
1. Navigate to `http://localhost:8080`
2. Upload your single-cell data (h5ad format)
3. View annotations and download results

### API

#### Python Example
```python
import requests

# Upload and annotate data
with open('new_data.h5ad', 'rb') as f:
    files = {'file': f}
    data = {
        'top_k': 10,
        'similarity_threshold': 0.7
    }
    response = requests.post('http://localhost:8000/api/annotate', files=files, data=data)
    annotations = response.json()
    print(f"Annotated {annotations['total_cells']} cells")
```

#### cURL Example
```bash
curl -X POST "http://localhost:8000/api/annotate" \
  -F "file=@new_data.h5ad" \
  -F "top_k=10" \
  -F "similarity_threshold=0.7"
```

See `scripts/example_usage.py` for more examples.

## Data Sources

- **In-house glioma data**: https://bcm.box.com/s/fcslrsl6bcfr6hgux7mdw1sbyazmjkzb
- **Transformer embeddings**: https://bcm.box.com/s/x5vvac8ug0exxpqe7wlfng3zzk4zkhmg (use layer -1)
- **Attention-based annotations**: https://bcm.box.com/s/q6zu7g0dbow081wwm8kz3rmgzdyicko6

## How It Works

1. **Reference Data Preparation**: The system loads your in-house glioma data with transformer embeddings and attention-based annotations
2. **FAISS Index**: Creates a fast similarity search index using FAISS for efficient embedding matching
3. **Query Processing**: When new data is uploaded:
   - Extracts embeddings from the new data
   - Searches for nearest neighbors in the reference embedding space
   - Predicts annotations based on majority vote of top matches
   - Returns confidence scores based on similarity

## API Endpoints

### Core Endpoints
- `GET /api/health` - Check API health and data availability
- `POST /api/annotate` - Annotate single-cell data file
- `GET /api/annotate/status` - Get annotation system status

### Chat Agent Endpoints
- `POST /api/chat/session` - Create a new chat session
- `GET /api/chat/session/{session_id}` - Get chat session
- `POST /api/chat/{session_id}/message` - Send message to chat agent
- `POST /api/chat/{session_id}/annotate` - Annotate file via chat
- `DELETE /api/chat/session/{session_id}` - Delete chat session

See http://localhost:8000/docs for interactive API documentation.

## Troubleshooting

### Reference embeddings not found
If you see a warning about missing reference embeddings:
1. Ensure all data files are downloaded and placed in the correct directories
2. Run `python scripts/prepare_reference_embeddings.py` to build the index

### Port already in use
If port 8000 or 8080 is already in use:
- Backend: Change `PORT` in `backend/config.py` or set `PORT` environment variable
- Frontend: Use a different port: `python3 -m http.server 8081`

### Embedding extraction fails
If embeddings cannot be extracted from uploaded data:
- Ensure your h5ad file contains embeddings in `adata.obsm` (e.g., 'X_umap', 'X_pca', 'X_embedding')
- The system will attempt to compute PCA as a fallback if no embeddings are found
