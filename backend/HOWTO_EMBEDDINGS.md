## Quick Start: Embeddings (Layer -1)

This guide explains how to generate Geneformer embedding layer `-1` from a
`.h5ad` file using the fine-tuned model and dictionaries bundled in this repo.
It is meant for offline preprocessing (not the live web app runtime).

### What to share (from this repo)
- `backend/run_embeddings.py`
- `backend/match_embeddings.py` (optional)
- `backend/dict/` (the `.pkl` dictionaries)
- `backend/dirks_primary_gbm_combined_2000perCellType/finetune/...` (model)

### Create a clean environment
Use Python 3.10 or 3.11 (Geneformer deps are not stable on 3.12+).

```bash
# conda (macOS/Linux)
conda create -y -n geneformer -c conda-forge python=3.10
conda activate geneformer
python --version
```

```powershell
# PowerShell (Windows)
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python --version
```

### Install dependencies
This script needs `geneformer`, `scanpy`, `datasets`, and friends. If you already
set up the repo, you can also just `pip install -r requirements.txt` and then
install Geneformer.

```bash
# GPU PyTorch (CUDA 12.1 example)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Core deps
pip install gprofiler-official scanpy loompy datasets "transformers==4.46.0"

# Install Geneformer (local checkout)
pip install ./Geneformer
```

### Run embeddings (layer -1)
From the repo root:

```bash
python backend/run_embeddings.py \
  --h5ad /path/to/input.h5ad \
  --dict-dir backend/dict \
  --models-root backend
```

Outputs are written to `<input_stem>_embs/` (created next to where you run it).

### Common options
```bash
--gene-id-type symbol      # if your h5ad var_names are gene symbols
--gpu 0                    # CUDA_VISIBLE_DEVICES value
--max-ncells 1000000       # cap processed cells
--forward-batch-size 100   # inference batch size
--finetune-subdir <name>   # override the default finetune subdir
```

### Where the new files go (for this project)
If you are producing **reference** embeddings that the portal should use:
- Copy the generated CSV(s) (e.g. `embs_by_*_emb_layer_-1.csv`) into
  `data/embeddings/`.
- Ensure the reference `.h5ad` used for those embeddings is located at
  `data/adata.h5ad`.
- Then run `python scripts/prepare_reference_embeddings.py` to rebuild the
  FAISS index used by the backend.

If you are just generating embeddings for **one-off analysis**, you can leave
the output in `<input_stem>_embs/` and use the CSV directly in notebooks.

## Match input to trained embeddings (optional)
Use `backend/match_embeddings.py` to map each input cell to the most similar
trained cell by cosine similarity.

```bash
python backend/match_embeddings.py \
  --input /path/to/embs_by_dirks_primary_gbm_combined_2000perCellType_num_classes_13_emb_layer_-1.csv \
  --trained /path/to/embs_by_dirks_primary_gbm_combined_2000perCellType_num_classes_13_emb_layer_-1.csv \
  --id-col individual \
  --out /path/to/embedding_matches.csv
```

Output columns: `new_cell_id`, `old_cell_id`, `max_cosine`.

### Notes
- `.h5ad` must contain raw counts and gene identifiers.
- If `adata.var['ensembl_id']` is missing, the script will use `adata.var_names`
  (set `--gene-id-type symbol` if those are gene symbols).
