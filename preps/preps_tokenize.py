#!/usr/bin/env python3
"""
Tokenize scRNA-seq data for Geneformer. Renamed from tokenize.py to avoid
shadowing Python's standard library module `tokenize` (breaks pandas, etc.).

Must use model_input_size=2048 to match fine-tuned checkpoints (L2048_*).
"""
import argparse
from pathlib import Path

_PREPS_DIR = Path(__file__).resolve().parent
_DEFAULT_DICT_DIR = _PREPS_DIR / "dict"

parser = argparse.ArgumentParser(description="scRNA-seq data tokenization.")
parser.add_argument(
    "test_name",
    help="Directory name of the dataset to be tokenized (e.g., mouse, glioma).",
)
parser.add_argument(
    "-s",
    "--species",
    choices=["human", "mouse"],
    default="human",
    help="Species (default human).",
)
parser.add_argument(
    "--dict-dir",
    type=Path,
    default=_DEFAULT_DICT_DIR,
    help=f"Folder with token_dictionary_gc30M.pkl, gene_median_dictionary_gc30M.pkl, "
    f"ensembl_mapping_dict_gc30M.pkl (default: {_DEFAULT_DICT_DIR})",
)


def main() -> None:
    args = parser.parse_args()
    test_name = args.test_name
    species = args.species
    dict_dir = args.dict_dir.expanduser().resolve()

    import os
    import random

    import numpy as np
    import pandas as pd
    import scanpy as sc
    import torch
    from gprofiler import GProfiler
    from geneformer import TranscriptomeTokenizer
    from scipy.io import mmread
    import loompy

    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    token_ref_directory = f"{test_name}/"
    if os.path.isfile(token_ref_directory + "adata.h5ad"):
        print(f"Loading {token_ref_directory}adata.h5ad")
        adata = sc.read_h5ad(token_ref_directory + "adata.h5ad")
        print(f"{token_ref_directory}adata.h5ad loaded")

    else:
        # Load genes, barcodes, matrix, and metadata if h5ad is unavailable
        with open(token_ref_directory + "genes.tsv") as f:
            genes = f.read().rstrip().split("\n")

        with open(token_ref_directory + "barcodes.tsv") as f:
            barcodes = f.read().rstrip().split("\n")

        print(f"Loading {token_ref_directory}matrix.mtx")
        mat = mmread(token_ref_directory + "matrix.mtx")
        df = pd.DataFrame.sparse.from_spmatrix(mat, index=genes, columns=barcodes).fillna(0)
        adata = sc.AnnData(df.T)
        print(f"{token_ref_directory}matrix.mtx loaded")
        del mat, df, genes, barcodes

        index_col = "CellID"
        df_ref_meta = pd.read_csv(token_ref_directory + "meta.tsv", sep="\t", index_col=index_col)
        df_ref_meta = df_ref_meta.loc[adata.obs_names, :]
        adata.obs = df_ref_meta.copy()

        adata.write(token_ref_directory + "adata.h5ad")
        print(f"{token_ref_directory}adata.h5ad saved")

    adata.obs["group"] = "_"  # If adata is for GPT model finetuning, designate one column of adata.obs as "group" that contains group information
    adata.obs["isTumor"] = 0  # A trick related to finetuning: only those cells labeled with "isTumor = 0" are to be used for model finetuning
    adata.obs = adata.obs[["group", "isTumor"]].copy()  # All other columns of adata.obs are excluded as they may disturb tokenization

    gp = GProfiler(return_dataframe=True)
    if species == "human":
        df_genes_converted = gp.convert(
            organism="hsapiens", query=adata.var_names.tolist(), target_namespace="ENSG"
        )
        df_genes_converted = df_genes_converted[~df_genes_converted["incoming"].duplicated()]
        df_genes_converted = df_genes_converted[
            ~df_genes_converted["converted"].isin([None, np.nan, "None", "N/A"])
        ]
        df_genes_converted[["incoming", "converted", "name", "description"]].to_excel(
            token_ref_directory + f"{test_name}_convertedGenes.xlsx", index=False
        )

    else:
        df_genes_converted = gp.orth(
            organism="mmusculus", query=adata.var_names.tolist(), target="hsapiens"
        )
        df_genes_converted = df_genes_converted[~df_genes_converted["incoming"].duplicated()]
        df_genes_converted = df_genes_converted[
            ~df_genes_converted["ortholog_ensg"].isin([None, np.nan, "None", "N/A"])
        ]
        df_genes_converted[
            ["incoming", "converted", "ortholog_ensg", "name", "description"]
        ].to_excel(token_ref_directory + f"{test_name}_convertedGenes.xlsx", index=False)

    # Filter out those genes with no ENSG IDs
    adata = adata[:, df_genes_converted["incoming"].tolist()].copy()

    # Add metadata required by tokenizer, don't change the feature names
    if species == "human":
        adata.var["ensembl_id"] = df_genes_converted["converted"].tolist()
    else:
        adata.var["ensembl_id"] = df_genes_converted["ortholog_ensg"].tolist()

    adata.obs["n_counts"] = np.sum(adata.X.toarray(), axis=1)  # total read counts in each cell
    adata.obs["filter_pass"] = 1
    adata.obs["individual"] = adata.obs.index.tolist()  # cell IDs

    # Save as [name].loom in [ref_directory]
    data = adata.X.toarray().T
    df_row_metadata = adata.var.copy()
    df_col_metadata = adata.obs.copy()
    loompy.create(
        f"{token_ref_directory}{test_name}.loom",
        data,
        df_row_metadata.to_dict("list"),
        df_col_metadata.to_dict("list"),
    )
    del adata, data, df_row_metadata, df_col_metadata

    # Tokenize [name].loom — 2048 tokens to match Geneformer CellClassifier finetune (L2048_*)
    token_dictionary_file = dict_dir / "token_dictionary_gc30M.pkl"
    gene_median_file = dict_dir / "gene_median_dictionary_gc30M.pkl"
    gene_mapping_file = dict_dir / "ensembl_mapping_dict_gc30M.pkl"
    for p in (token_dictionary_file, gene_median_file, gene_mapping_file):
        if not p.is_file():
            raise FileNotFoundError(
                f"Missing Geneformer dict file: {p}\n"
                "Place *.pkl under preps/dict/ or pass --dict-dir."
            )

    tk = TranscriptomeTokenizer(
        {
            "individual": "individual",
            "isTumor": "isTumor",
            "group": "group",
            "n_counts": "n_counts",
        },
        nproc=1,
        model_version="V2",
        model_input_size=2048,
        special_token=False,
        token_dictionary_file=str(token_dictionary_file),
        gene_median_file=str(gene_median_file),
        gene_mapping_file=str(gene_mapping_file),
    )
    tk.tokenize_data(token_ref_directory, token_ref_directory, test_name)
    os.remove(f"{token_ref_directory}{test_name}.loom")


if __name__ == "__main__":
    main()
