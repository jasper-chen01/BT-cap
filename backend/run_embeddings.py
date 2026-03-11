import argparse
import os
import sys

# Avoid shadowing stdlib modules (e.g., tokenize.py) in preps/.
script_dir = os.path.dirname(os.path.abspath(__file__))
preps_dir = os.path.join(script_dir, "preps")
sys.path = [p for p in sys.path if p != preps_dir]

import numpy as np
import scanpy as sc
from datasets import load_from_disk
from gprofiler import GProfiler
from geneformer import EmbExtractor, TranscriptomeTokenizer
from tqdm import tqdm


REF_NUM_TUPS = [
    ("dirks_primary_gbm_combined_2000perCellType", 13),
]

DEFAULT_FINETUNE_SUBDIR = (
    "240605_geneformer_CellClassifier_0_L2048_B12_LR5e-05_LSlinear_WU500_E10_Oadamw_F0"
)


def prepare_h5ad(in_h5ad: str, out_h5ad: str, gene_id_type: str):
    adata = sc.read_h5ad(in_h5ad)

    if "group" not in adata.obs:
        adata.obs["group"] = "_"
    if "isTumor" not in adata.obs:
        adata.obs["isTumor"] = 0
    if "n_counts" not in adata.obs:
        # Avoid densifying large sparse matrices.
        if hasattr(adata.X, "sum"):
            adata.obs["n_counts"] = np.asarray(adata.X.sum(axis=1)).ravel()
        else:
            adata.obs["n_counts"] = np.sum(adata.X, axis=1)
    if "filter_pass" not in adata.obs:
        adata.obs["filter_pass"] = 1
    if "individual" not in adata.obs:
        adata.obs["individual"] = adata.obs.index.tolist()

    adata.obs = adata.obs[
        ["group", "isTumor", "n_counts", "filter_pass", "individual"]
    ].copy()

    # anndata reserves "_index" as a column name when writing.
    if "_index" in adata.var.columns:
        adata.var = adata.var.drop(columns=["_index"])
    if adata.raw is not None and "_index" in adata.raw.var.columns:
        # Raw.var is read-only; drop raw entirely to avoid write_h5ad errors.
        adata.raw = None

    if "ensembl_id" not in adata.var:
        if gene_id_type == "symbol":
            gp = GProfiler(return_dataframe=True)
            df_map = gp.convert(
                organism="hsapiens",
                query=adata.var_names.tolist(),
                target_namespace="ENSG",
            )
            df_map = df_map[~df_map["incoming"].duplicated()]
            ensembl_map = dict(zip(df_map["incoming"], df_map["converted"]))
            adata.var["ensembl_id"] = [
                ensembl_map.get(g, None) for g in adata.var_names.tolist()
            ]
        else:
            adata.var["ensembl_id"] = adata.var_names.tolist()

    adata.write(out_h5ad)


def tokenize_h5ad(h5ad_path: str, output_directory: str, dict_dir: str):
    os.makedirs(output_directory, exist_ok=True)
    token_dictionary_file = os.path.join(dict_dir, "token_dictionary_gc30M.pkl")
    gene_median_file = os.path.join(dict_dir, "gene_median_dictionary_gc30M.pkl")
    gene_mapping_file = os.path.join(dict_dir, "ensembl_mapping_dict_gc30M.pkl")

    for p in [token_dictionary_file, gene_median_file, gene_mapping_file]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing dictionary file: {p}")

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
        token_dictionary_file=token_dictionary_file,
        gene_median_file=gene_median_file,
        gene_mapping_file=gene_mapping_file,
    )
    h5ad_dir = os.path.dirname(h5ad_path)
    h5ad_name = os.path.basename(h5ad_path)
    tk.tokenize_data(
        data_directory=h5ad_dir,
        output_directory=output_directory,
        output_prefix="tokenized_copy",
        file_format="h5ad",
        input_identifier=os.path.splitext(h5ad_name)[0],
    )

    dataset_path = os.path.join(output_directory, "tokenized_copy.dataset")
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Tokenized dataset not found: {dataset_path}")

    try:
        dataset = load_from_disk(dataset_path)
    except Exception as exc:
        raise RuntimeError(
            "Failed to read tokenized dataset. This usually means tokenization "
            "produced no cells. Check that gene IDs in the h5ad match the "
            "token dictionary (Ensembl IDs expected)."
        ) from exc

    if len(dataset) == 0:
        raise RuntimeError(
            "Tokenized dataset is empty. This usually means none of the genes "
            "in the h5ad match the token dictionary. Ensure adata.var['ensembl_id'] "
            "contains Ensembl IDs that match the provided dictionaries."
        )

    return dataset_path


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize h5ad and extract emb layer -1."
    )
    parser.add_argument("--h5ad", required=True, help="Input .h5ad file")
    parser.add_argument(
        "--gene-id-type",
        choices=["ensembl", "symbol"],
        default="ensembl",
        help="Gene ID type in h5ad var names (default ensembl)",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory (default: <h5ad_stem>_embs)",
    )
    parser.add_argument(
        "--dict-dir",
        required=True,
        help="Directory containing geneformer *.pkl dictionaries",
    )
    parser.add_argument(
        "--models-root",
        default=".",
        help="Root directory containing model folders",
    )
    parser.add_argument(
        "--finetune-subdir",
        default=DEFAULT_FINETUNE_SUBDIR,
        help="Model subdir under <ref>/finetune/",
    )
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--max-ncells", type=int, default=1000 * 1000)
    parser.add_argument("--forward-batch-size", type=int, default=100)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["NCCL_DEBUG"] = "INFO"

    h5ad_path = os.path.abspath(args.h5ad)
    if not os.path.isfile(h5ad_path):
        raise FileNotFoundError(h5ad_path)

    h5ad_stem = os.path.splitext(os.path.basename(h5ad_path))[0]
    out_dir = args.out_dir or f"{h5ad_stem}_embs"
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    prepared_h5ad = os.path.join(out_dir, f"{h5ad_stem}.prepared.h5ad")
    prepare_h5ad(h5ad_path, prepared_h5ad, args.gene_id_type)

    tokenized_dataset = tokenize_h5ad(prepared_h5ad, out_dir, args.dict_dir)

    token_dictionary_file = os.path.join(args.dict_dir, "token_dictionary_gc30M.pkl")

    model_dirs = []
    for ref_name, num_classes in REF_NUM_TUPS:
        model_dir = os.path.join(
            args.models_root, ref_name, "finetune", args.finetune_subdir
        )
        if os.path.isdir(model_dir):
            model_dirs.append((ref_name, num_classes, model_dir))

    if not model_dirs:
        raise RuntimeError("No model directories found under ref_num_tups")

    for ref_name, num_classes, model_dir in tqdm(model_dirs):
        output_prefix = f"embs_by_{ref_name}_num_classes_{num_classes}_emb_layer_-1"

        embex = EmbExtractor(
            model_type="CellClassifier",
            num_classes=num_classes,
            emb_mode="cell",
            filter_data=None,
            max_ncells=args.max_ncells,
            emb_layer=-1,
            emb_label=["individual", "group"],
            labels_to_plot=["group"],
            forward_batch_size=args.forward_batch_size,
            nproc=1,
            token_dictionary_file=token_dictionary_file,
        )

        embs = embex.extract_embs(
            model_directory=model_dir,
            input_data_file=tokenized_dataset,
            output_directory=out_dir,
            output_prefix=output_prefix,
        )



if __name__ == "__main__":
    main()
