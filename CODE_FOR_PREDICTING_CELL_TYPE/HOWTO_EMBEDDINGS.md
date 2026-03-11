## Quick Start: Embeddings (Layer -1)

This guide lets someone else generate embedding layer `-1` from their own `.h5ad`
using your fine-tuned models and dictionaries.

### What to share
- `fine_tuned_models` folders (your `ref_name/finetune/...` directories)
- `dict/` with real dictionary `.pkl` files
- `run_embeddings.py`

### Create a clean environment
```bash
conda create -y -n geneformer -c conda-forge python=3.10
conda activate geneformer
python --version
#export PATH="/mnt/data/home/u237333/miniconda3/envs/geneformer/bin:$PATH"
#which python
#python --version

```

### Install dependencies
```bash
# GPU PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Core deps
pip install gprofiler-official scanpy loompy datasets "transformers==4.46.0"

# Install Geneformer (local checkout)
pip install ./Geneformer
```

### Run embeddings (layer -1)
```bash
python run_embeddings.py \
  --h5ad /path/to/input.h5ad \
  --dict-dir /path/to/dict \
  --models-root /path/to/models/root
```

Outputs are written to `<input_stem>_embs/`.

### Common options
```bash
--no-plot                # skip UMAP plot
--gpu 0                  # select GPU
--max-ncells 1000000      # cap processed cells
--forward-batch-size 100  # inference batch size
```

## Match input to trained embeddings
Use `match_embeddings.py` to map each input cell to the most similar trained cell
by cosine similarity.

```bash
python match_embeddings.py \
  --input /path/to/embs_by_dirks_primary_gbm_combined_2000perCellType_num_classes_13_emb_layer_-1.csv \
  --trained /path/to/embs_by_dirks_primary_gbm_combined_2000perCellType_num_classes_13_emb_layer_-1.csv \
  --id-col individual \
  --out /path/to/embedding_matches.csv

  python match_embeddings.py \
  --input /mnt/data/serinharmanci/BT_CAP_code/embs_by_dirks_primary_gbm_combined_2000perCellType_num_classes_13_emb_layer_-1.csv \
  --trained /mnt/data/serinharmanci/BT_CAP_code/embs_by_dirks_primary_gbm_combined_2000perCellType_num_classes_13_emb_layer_-1_OLD.csv \
  --id-col individual \
  --out /mnt/data/serinharmanci/BT_CAP_code/embedding_matches.csv

```

Output columns: `new_cell_id`, `old_cell_id`, `max_cosine`.

### Notes
- `.h5ad` must have raw counts and gene identifiers.
- If `ensembl_id` is missing, the script will use `adata.var_names`.
