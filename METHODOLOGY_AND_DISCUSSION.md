# BAT Portal Methodology (Chat Excluded)

This document describes the technical methodology implemented in the BAT Portal codebase. It focuses on the annotation, visualization, and embedding pipelines and excludes chatbot-related functionality.

## System Architecture and Code Structure

The system is a FastAPI backend with a static HTML frontend (and an optional React UI). Methodologically, all computation occurs in the backend; the frontend only uploads files and renders responses.

Key modules:

- `backend/main.py`: FastAPI application, CORS, and router registration.
- `backend/config.py`: centralized parameters and filesystem paths.
- `backend/api/`: REST endpoints for annotation, visualization, and embedding jobs.
- `backend/services/`: core logic for preprocessing, indexing, annotation, visualization, and job execution.
- `backend/models/schemas.py`: Pydantic request/response schema definitions.
- `scripts/prepare_reference_embeddings.py`: offline reference index builder.
- `backend/run_embeddings.py`: Geneformer embedding extraction.
- `backend/match_embeddings.py`: cosine similarity matching between embedding sets.

## Data Inputs and File Conventions

Reference inputs (expected inside `data/`):

- `adata.h5ad`: reference scRNA-seq dataset.
- `embeddings/`: transformer embeddings (supported formats: `.npy`, `.csv`, `.pkl`).
- `annotations/`: reference labels (supported formats: `.csv`, `.tsv`, `.pkl`).

Generated reference outputs:

- `reference_embeddings.faiss`
- `reference_cell_ids.pkl`
- `reference_annotations.pkl`

Visualization annotation resources:

- `annotations/ligands.txt`
- `annotations/receptors.txt`
- `annotations/drug.tsv`

User uploads (annotation/visualization):

- `.h5ad` single-cell datasets.

Embedding pipeline outputs (per input file):

- `data/embedding_runs/<stem>_embs/embs_by_*_emb_layer_-1.csv`
- `data/embedding_runs/<stem>_embs/embedding_matches.csv`
- `data/embedding_runs/<stem>_embs/embedding_matches_with_celltypes.csv`

## Configuration Parameters

Core defaults in `backend/config.py`:

- `EMBEDDING_LAYER = -1` (last transformer layer).
- `TOP_K = 10` and `SIMILARITY_THRESHOLD = 0.7` for annotation.
- `TRAINED_EMBEDDINGS_PATH` and `CELLTYPES_PATH` for embedding matching.

Visualization endpoint parameters:

- `de_top_n` (default 15).
- `cluster_resolution` (default 1.0).
- `use_hvg` (default true).
- `apply_filtering` (default true).

## Methodology Pipeline A: Reference Index Preparation

Script: `scripts/prepare_reference_embeddings.py`

Purpose: build a FAISS index from reference embeddings and reference annotations.

Procedure:

1. Load reference AnnData (`data/adata.h5ad`).
2. Load UMAP coordinates (`embedding_coordinates.csv`) and align to `adata.obs_names`.
3. Load transformer embeddings from `data/embeddings/`:
   - `.npy`, `.csv`, and `.pkl` supported.
   - For multi-layer embeddings, select `EMBEDDING_LAYER`.
4. Load annotations from `data/annotations/`:
   - Accept `.csv`, `.tsv`, or `.pkl`.
   - If unavailable, create `"Unknown"` labels for all cells.
5. Build FAISS index:
   - L2-normalize embeddings for cosine similarity.
   - Construct `IndexFlatIP` (inner product equals cosine on normalized vectors).
6. Persist:
   - FAISS index.
   - Cell IDs (`reference_cell_ids.pkl`).
   - Annotation mapping (`reference_annotations.pkl`).

Outputs provide a fixed reference space for all downstream annotation.

## Methodology Pipeline B: Annotation (Online)

Endpoint: `POST /api/annotate`

Purpose: annotate new datasets by similarity search in the reference embedding space.

Procedure:

1. Receive `.h5ad` upload and store as a temporary file.
2. Load and preprocess via `DataProcessor.load_and_preprocess`:
   - If embeddings exist in `adata.obsm`, skip preprocessing.
   - Otherwise, filter cells/genes, normalize counts, and log1p.
3. Extract embeddings via `DataProcessor.extract_embeddings`:
   - Check `adata.obsm` keys in priority order:
     - `X_umap`, `X_pca`, `X_embedding`, `embeddings`, `X_transformer`.
   - If UMAP is low-dimensional (<10), prefer PCA.
   - If no matching dimension exists, compute PCA with `n_comps = target_dim or 50`.
4. FAISS search:
   - Normalize query embeddings.
   - Retrieve `top_k` nearest neighbors from the reference index.
5. Label assignment (per cell):
   - Collect neighbor annotations with similarity >= threshold.
   - Perform a similarity-weighted majority vote:
     - Each label is repeated `int(similarity * 100)` times.
   - Confidence = fraction of neighbors that match the predicted label.
6. Return JSON response and delete temporary file.

Output:

- Per-cell predictions with confidence and top neighbor annotations.

## Methodology Pipeline C: Visualization and Analysis (Online)

Endpoint: `POST /api/visualize`

Purpose: generate UMAP, clustering, and differential expression summaries.

Procedure:

1. Receive `.h5ad` upload and store as a temporary file.
2. Load AnnData and standardize metadata:
   - Ensure `gene_symbol` exists.
   - Convert `.raw` to `.X` if present.
3. If UMAP or Leiden missing:
   - Optional filtering: `min_genes=200`, `min_cells=3`.
   - Normalize total counts and log1p.
   - Optional HVG selection (n=2000).
   - Scale, PCA, nearest-neighbors graph, and compute UMAP.
   - Leiden clustering (requires `leidenalg`).
4. Optional annotation overlay (supptable):
   - Accepts local file, URL, or Firestore document reference.
   - Detects `cell_id`, `cell_type`, and optional score columns.
   - Adds `cell_type` and `cell_type_score` to `adata.obs`.
5. Optional embedding-match overlay:
   - If provided, load `embedding_matches_with_celltypes.csv`.
   - Map to `predicted_cell_type` and optional scores.
6. Differential expression:
   - `sc.tl.rank_genes_groups` (Wilcoxon).
   - Summarize per Leiden cluster.
   - If `cell_type` exists, summarize per cell type.
7. Functional gene annotation:
   - Load ligand, receptor, and drug target sets from `data/annotations/`.
   - Annotate DE genes with ligand/receptor status and drug target metadata.
8. Overlap report:
   - Intersect DE gene lists with target sets for summary reporting.
9. Write analysis summary JSON to `data/analysis_runs/`.

Outputs:

- UMAP coordinates, cluster labels, cluster counts.
- Differential expression tables with annotations.
- Optional cell type overlays and summary statistics.
- Analysis summary file path in metadata.

## Methodology Pipeline D: Embedding Extraction (Background Job)

Endpoint: `POST /api/embeddings/jobs`

Purpose: extract Geneformer embeddings from a new dataset.

Procedure:

1. Spawn `backend/run_embeddings.py` as a background subprocess.
2. Prepare input:
   - Ensure `group`, `isTumor`, `n_counts`, `filter_pass`, and `individual` exist in `.obs`.
   - Map gene IDs to Ensembl if input uses gene symbols.
3. Tokenize with Geneformer dictionaries:
   - `token_dictionary_gc30M.pkl`
   - `gene_median_dictionary_gc30M.pkl`
   - `ensembl_mapping_dict_gc30M.pkl`
4. Load fine-tuned Geneformer model and extract embeddings:
   - Cell-level embeddings (`emb_mode="cell"`).
   - Layer `-1`.
5. Write embeddings to `data/embedding_runs/<stem>_embs/`.

Job tracking:

- Status and logs stored in `data/embedding_jobs/`.
- Queryable via `/api/embeddings/jobs/{job_id}` and `/api/embeddings/jobs/{job_id}/log`.

## Methodology Pipeline E: Embedding + Matching (Background Job)

Endpoint: `POST /api/embeddings/pipeline/jobs`

Purpose: generate embeddings and map each new cell to the most similar reference embedding.

Procedure:

1. Run the embedding extraction job (Pipeline D).
2. Match new embeddings to trained reference embeddings:
   - Normalize rows to unit length.
   - Compute cosine similarity by matrix multiplication.
   - Select the best match per cell.
3. Attach cell types:
   - Map matched reference IDs to labels using `celltypes.csv`.
4. Save outputs:
   - `embedding_matches.csv`
   - `embedding_matches_with_celltypes.csv`

Outputs are used by visualization to overlay predicted cell types.

## API Inputs/Outputs (Method-Facing)

Annotation:

- Input: `.h5ad`, `top_k`, `similarity_threshold`.
- Output: list of per-cell predicted labels, confidence, and top matches.

Visualization:

- Input: `.h5ad`, optional supptable reference, optional embedding matches path.
- Output: UMAP points, cluster labels, DE gene lists, and metadata.

Embedding jobs:

- Input: `h5ad_path`, `dict_dir`, `models_root`, and job settings.
- Output: job status, logs, and file paths.

## Reproducibility Notes

- Reference index is fixed once built and stored in `data/`.
- Outputs are deterministic for the same inputs and parameter settings.
- All jobs persist logs and outputs under `data/` subfolders for traceability.

## LaTeX Flow Charts (TikZ)

Below are LaTeX flow charts for each methodology pipeline. Each block is a standalone
`tikzpicture` and can be placed inside a LaTeX document that includes:

```latex
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric}
```

### A) Reference Index Preparation

```latex
\begin{tikzpicture}[
  node distance=10mm and 12mm,
  box/.style={rectangle, rounded corners, draw=black, align=center, minimum width=32mm, minimum height=7mm},
  arrow/.style={-Latex, thick}
]
\node[box] (start) {Start};
\node[box, below=of start] (loadref) {Load reference\\ `adata.h5ad`};
\node[box, below=of loadref] (loadumap) {Load UMAP coords\\ `embedding_coordinates.csv`};
\node[box, below=of loadumap] (loademb) {Load transformer\\ embeddings (.npy/.csv/.pkl)};
\node[box, below=of loademb] (loadann) {Load annotations\\ from `data/annotations/`};
\node[box, below=of loadann] (build) {Normalize embeddings\\ + build FAISS index};
\node[box, below=of build] (save) {Save index + cell IDs\\ + annotations};
\node[box, below=of save] (end) {End};
\draw[arrow] (start) -- (loadref);
\draw[arrow] (loadref) -- (loadumap);
\draw[arrow] (loadumap) -- (loademb);
\draw[arrow] (loademb) -- (loadann);
\draw[arrow] (loadann) -- (build);
\draw[arrow] (build) -- (save);
\draw[arrow] (save) -- (end);
\end{tikzpicture}
```

### B) Annotation Endpoint

```latex
\begin{tikzpicture}[
  node distance=10mm and 12mm,
  box/.style={rectangle, rounded corners, draw=black, align=center, minimum width=34mm, minimum height=7mm},
  diamond/.style={diamond, aspect=2, draw=black, align=center, inner sep=1pt},
  arrow/.style={-Latex, thick}
]
\node[box] (start) {Upload `.h5ad`};
\node[box, below=of start] (load) {Load AnnData\\ + standardize metadata};
\node[diamond, below=of load] (hasemb) {Embeddings\\ in `obsm`?};
\node[box, below left=of hasemb] (skip) {Skip preprocessing};
\node[box, below right=of hasemb] (prep) {Filter + normalize\\ + log1p};
\node[box, below=of skip, xshift=17mm] (extract) {Extract embeddings\\ or PCA fallback};
\node[box, below=of extract] (search) {FAISS search\\ (top-k)};
\node[box, below=of search] (vote) {Similarity-weighted\\ majority vote};
\node[box, below=of vote] (return) {Return JSON\\ + delete temp file};
\draw[arrow] (start) -- (load);
\draw[arrow] (load) -- (hasemb);
\draw[arrow] (hasemb) -- node[left]{Yes} (skip);
\draw[arrow] (hasemb) -- node[right]{No} (prep);
\draw[arrow] (skip) -- (extract);
\draw[arrow] (prep) -- (extract);
\draw[arrow] (extract) -- (search);
\draw[arrow] (search) -- (vote);
\draw[arrow] (vote) -- (return);
\end{tikzpicture}
```

### C) Visualization Endpoint

```latex
\begin{tikzpicture}[
  node distance=9mm and 12mm,
  box/.style={rectangle, rounded corners, draw=black, align=center, minimum width=36mm, minimum height=7mm},
  diamond/.style={diamond, aspect=2, draw=black, align=center, inner sep=1pt},
  arrow/.style={-Latex, thick}
]
\node[box] (start) {Upload `.h5ad`};
\node[box, below=of start] (load) {Load AnnData\\ + standardize metadata};
\node[diamond, below=of load] (hasumap) {UMAP and\\ Leiden exist?};
\node[box, below left=of hasumap] (skip) {Skip recompute};
\node[box, below right=of hasumap] (compute) {Filter + normalize\\ HVG + PCA\\ neighbors + UMAP\\ Leiden clustering};
\node[box, below=of skip, xshift=18mm] (suppt) {Optional supptable\\ (cell types + scores)};
\node[box, below=of suppt] (match) {Optional embedding matches\\ (predicted types)};
\node[box, below=of match] (de) {DE genes\\ (Wilcoxon)};
\node[box, below=of de] (annot) {Annotate genes\\ (ligand/receptor/drug)};
\node[box, below=of annot] (overlap) {Overlap report\\ vs target sets};
\node[box, below=of overlap] (save) {Write analysis summary\\ JSON to `analysis_runs/`};
\node[box, below=of save] (return) {Return UMAP + clusters\\ + DE results};
\draw[arrow] (start) -- (load);
\draw[arrow] (load) -- (hasumap);
\draw[arrow] (hasumap) -- node[left]{Yes} (skip);
\draw[arrow] (hasumap) -- node[right]{No} (compute);
\draw[arrow] (skip) -- (suppt);
\draw[arrow] (compute) -- (suppt);
\draw[arrow] (suppt) -- (match);
\draw[arrow] (match) -- (de);
\draw[arrow] (de) -- (annot);
\draw[arrow] (annot) -- (overlap);
\draw[arrow] (overlap) -- (save);
\draw[arrow] (save) -- (return);
\end{tikzpicture}
```

### D) Embedding Extraction Job

```latex
\begin{tikzpicture}[
  node distance=9mm and 12mm,
  box/.style={rectangle, rounded corners, draw=black, align=center, minimum width=34mm, minimum height=7mm},
  arrow/.style={-Latex, thick}
]
\node[box] (start) {Start job};
\node[box, below=of start] (prepare) {Prepare AnnData\\ + gene IDs};
\node[box, below=of prepare] (tokenize) {Tokenize with\\ Geneformer dicts};
\node[box, below=of tokenize] (model) {Load fine-tuned\\ Geneformer model};
\node[box, below=of model] (extract) {Extract embeddings\\ (layer -1)};
\node[box, below=of extract] (save) {Write embeddings CSV\\ to `embedding_runs/`};
\node[box, below=of save] (log) {Persist logs\\ `embedding_jobs/`};
\node[box, below=of log] (end) {Job done};
\draw[arrow] (start) -- (prepare);
\draw[arrow] (prepare) -- (tokenize);
\draw[arrow] (tokenize) -- (model);
\draw[arrow] (model) -- (extract);
\draw[arrow] (extract) -- (save);
\draw[arrow] (save) -- (log);
\draw[arrow] (log) -- (end);
\end{tikzpicture}
```

### E) Embedding + Matching Pipeline Job

```latex
\begin{tikzpicture}[
  node distance=9mm and 12mm,
  box/.style={rectangle, rounded corners, draw=black, align=center, minimum width=36mm, minimum height=7mm},
  arrow/.style={-Latex, thick}
]
\node[box] (start) {Start pipeline job};
\node[box, below=of start] (emb) {Run embedding extraction\\ (Pipeline D)};
\node[box, below=of emb] (norm) {Normalize embeddings\\ (unit length)};
\node[box, below=of norm] (cosine) {Cosine similarity\\ vs trained embeddings};
\node[box, below=of cosine] (best) {Select best match\\ per cell};
\node[box, below=of best] (map) {Map cell IDs\\ to cell types};
\node[box, below=of map] (save) {Save `embedding_matches.csv`\\ + annotated matches};
\node[box, below=of save] (end) {Pipeline done};
\draw[arrow] (start) -- (emb);
\draw[arrow] (emb) -- (norm);
\draw[arrow] (norm) -- (cosine);
\draw[arrow] (cosine) -- (best);
\draw[arrow] (best) -- (map);
\draw[arrow] (map) -- (save);
\draw[arrow] (save) -- (end);
\end{tikzpicture}
```


