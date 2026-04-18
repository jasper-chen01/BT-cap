## PREPS: how to run on a new `.h5ad`

This guide runs PREPS from `ephys_prediction/preps/` to build `<test_name>_preds/` and then electrophysiology prediction with `patchseq_predict.py`.

### What this folder expects

| Item | Location / notes |
|------|------------------|
| Scripts | `generate_preds.py`, `preps_tokenize.py`, `annotate.py`, `patchseq_predict.py` |
| Geneformer dicts | `preps/dict/` (see below) |
| Fine-tuned CellClassifier models | Your choice; pass `--models-root` (see below) |
| Patch-seq resources | `combined_patchseq_all_preds/` (for `patchseq_predict.py`) |

**Do not** name a script `tokenize.py` in `preps/`: it shadows Python’s standard library `tokenize` and breaks imports. The tokenizer script is **`preps_tokenize.py`**.

### Geneformer dictionary files (`preps/dict/`)

Place these in **`ephys_prediction/preps/dict/`** (or pass `--dict-dir`):

- `token_dictionary_gc30M.pkl`
- `gene_median_dictionary_gc30M.pkl`
- `ensembl_mapping_dict_gc30M.pkl`

If the `.pkl` files sit directly in `preps/` (not `preps/dict/`), `annotate.py` can still find `token_dictionary_gc30M.pkl` via a fallback; tokenization should use **`--dict-dir`** explicitly so all three files are found.

### Fine-tuned models (`--models-root`)

`annotate.py` loads reference models from:

```text
<models_root>/<ref_name>/finetune/240605_geneformer_CellClassifier_0_L2048_B12_LR5e-05_LSlinear_WU500_E10_Oadamw_F0/
```

and:

```text
<models_root>/<ref_name>/finetune/target_names.xlsx
```

Point `--models-root` at the folder that **contains** the `aldinger_2000perCellType/`, `allen_2000perCellType/`, … directories (not the parent of `preps`).

### Sequence length (2048)

Fine-tuned checkpoints are **L2048** (max length **2048**). **`preps_tokenize.py`** must tokenize with **`model_input_size=2048`** and the V2 tokenizer settings (already set in the script). If you change tokenizer settings, rebuild `<test_name>/<test_name>.dataset/`.

---

## Environment

Use **Python 3.10** if possible.

```bash
conda create -y -n preps python=3.10
conda activate preps
```

### Dependencies

```bash
pip install torch torchvision torchaudio
conda install -n preps -c conda-forge llvmlite numba
pip install numpy pandas scipy scikit-learn joblib tqdm openpyxl scanpy loompy datasets "transformers==4.46.0"
gprofiler-official
git clone https://huggingface.co/ctheodoris/Geneformer
pip install Geneformer
```

### NumPy 2.x vs PyTorch

If you see errors about **NumPy 1.x vs 2.x** or `_ARRAY_API not found`, pin NumPy:

```bash
pip install "numpy<2"
```

### Verify

```bash
python -c "import torch, scanpy, pandas, datasets, transformers, geneformer; print('ok')"
```

---


From **`ephys_prediction/preps/`**:

```bash
cd /path/to/capstone_project_012026/ephys_prediction/preps

python generate_preds.py \
  /path/to/capstone_project_012026/inhouse_glioma_data/python/adata_subsampled.h5ad \
  glioma_sub \
  -s human \
  --copy \
  -g 0 \
  --models-root /path/to/capstone_project_012026/inhouse_glioma_data/fine-tuned_models/\
  --dict-dir /absolute/path/to/capstone_project_012026/ephys_prediction/preps/dict
  
  python patchseq_predict.py glioma_sub

```



| Argument | Meaning |
|----------|---------|
| First arg | Path to input `.h5ad` |
| Second arg | `<test_name>` (creates `./<test_name>/adata.h5ad` and outputs) |
| `-s human` \| `-s mouse` | Species for `preps_tokenize.py` |
| `--copy` | Copy `.h5ad` into `preps/` (default is symlink) |
| `-g 0` | GPU index for `annotate.py` (`CUDA_VISIBLE_DEVICES`; ignored on CPU/MPS-only Macs) |
| `--models-root` | Folder containing `*_perCellType/` reference trees |
| `--dict-dir` | Folder with the three `*_gc30M.pkl` files (passed to **both** tokenize and annotate) |

Defaults in code may point at your machine’s `fine-tuned_models` and `preps/dict`; override with the flags above when sharing the repo.

This runs:

1. Copy/symlink → `./<test_name>/adata.h5ad`
2. `preps_tokenize.py` → `./<test_name>/<test_name>.dataset/`
3. `annotate.py` → `./<test_name>_preds/`

Optional:

```bash
--python-bin /absolute/path/to/python
```

---

## Manual steps

```bash
cd .../ephys_prediction/preps
mkdir -p glioma_sub
cp /absolute/path/to/input.h5ad glioma_sub/adata.h5ad

python preps_tokenize.py glioma_sub -s human --dict-dir ./dict

python annotate.py glioma_sub -g 0 \
  --models-root /path/to/fine-tuned_models \
  --dict-dir ./dict
```

---

## Device: CUDA vs Apple Silicon (MPS) vs CPU

- **Linux + NVIDIA GPU:** CUDA works; `-g` selects the GPU index.
- **Mac:** PyTorch is usually **CPU/MPS only**. `annotate.py` picks **MPS** if available, else **CPU**. `.to("cuda")` is not used on Mac; do not expect NVIDIA CUDA on an iMac.

---

## Electrophysiology: `patchseq_predict.py`

After `<test_name>_preds/` exists:

```bash
python patchseq_predict.py glioma_sub
```

Outputs typically under `./<test_name>_patchseq/` (see script comments). `patchseq_predict.py` uses `combined_patchseq_all_preds/` and prediction artifacts; it does not read the `.h5ad` directly.

---

## Expected outputs

```text
./<test_name>/adata.h5ad
./<test_name>/<test_name>.dataset/
./<test_name>_preds/
./<test_name>_patchseq/    # after patchseq_predict.py
```

---

## Troubleshooting

| Issue | What to do |
|-------|------------|
| `tokenize` / `AttributeError: ... tokenize ... Name` | Use **`preps_tokenize.py`**, not a file named `tokenize.py`. |
| `Torch not compiled with CUDA enabled` | Expected on Mac; use updated `annotate.py` (MPS/CPU). |
| `KeyError: 'token_dictionary'` in collator | Ensure `--dict-dir` has `token_dictionary_gc30M.pkl`; pass `--dict-dir` to `annotate.py`. |
| BERT error **2048 vs 4096** (sequence length) | Re-tokenize with current **`preps_tokenize.py`** (`model_input_size=2048`), delete old `<test_name>.dataset`, run again. |
| `FileNotFoundError: ... target_names.xlsx` | Fix `--models-root` so each `<ref>/finetune/target_names.xlsx` exists. |
| sklearn `InconsistentVersionWarning` (pickle) | Match scikit-learn version to the one used when models were saved, or re-export models. |

---

## Notes

- `preps_tokenize.py` expects raw counts in the `.h5ad` (and gene IDs compatible with g:Profiler conversion / Ensembl as in the original PREPS workflow).
- `annotate.py` iterates over a fixed list of reference names (`aldinger_2000perCellType`, …); each must exist under `--models-root` or annotation will skip or fail per reference.
