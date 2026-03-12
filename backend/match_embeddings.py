import argparse
import os

import numpy as np
import pandas as pd


def _read_csv_safe(path: str, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kwargs)
    except pd.errors.ParserError:
        # Fallback for malformed rows or mixed delimiters.
        return pd.read_csv(path, engine="python", on_bad_lines="warn", **kwargs)


def load_embeddings(path: str, id_col: str):
    df = _read_csv_safe(path)
    # Prefer explicit cell id column if present
    if id_col in df.columns:
        df = df.set_index(id_col)
    else:
        # Assume first column is cell id if it is non-numeric or named like index
        first_col = df.columns[0]
        if first_col.lower() in ["cell", "cell_id", "cells", "index"] or not pd.api.types.is_numeric_dtype(
            df[first_col]
        ):
            df = df.set_index(first_col)
    # Keep only numeric embedding columns
    df = df.select_dtypes(include=[np.number])
    return df


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def main():
    parser = argparse.ArgumentParser(
        description="Match new embeddings to old by max cosine similarity."
    )
    parser.add_argument("--input", required=True, help="Input embeddings CSV")
    parser.add_argument("--trained", required=True, help="Trained embeddings CSV")
    parser.add_argument(
        "--id-col",
        default="individual",
        help="Column name that contains cell IDs (default individual)",
    )
    parser.add_argument(
        "--out",
        default="embedding_matches.csv",
        help="Output CSV path (default embedding_matches.csv)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Rows per chunk for new embeddings (default 1000)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input) or not os.path.isfile(args.trained):
        raise FileNotFoundError("Input CSV file not found.")

    old_df = load_embeddings(args.trained, args.id_col)
    old_ids = old_df.index.astype(str).tolist()
    old_mat = old_df.to_numpy(dtype=np.float32)
    old_mat = normalize_rows(old_mat)

    results = []
    # Read new embeddings in chunks to limit memory
    for chunk in _read_csv_safe(args.input, chunksize=args.chunk_size):
        first_col = chunk.columns[0]
        if args.id_col in chunk.columns:
            chunk = chunk.set_index(args.id_col)
        elif first_col.lower() in ["cell", "cell_id", "cells", "index"] or not pd.api.types.is_numeric_dtype(
            chunk[first_col]
        ):
            chunk = chunk.set_index(first_col)
        chunk = chunk.select_dtypes(include=[np.number])
        new_ids = chunk.index.astype(str).tolist()
        new_mat = chunk.to_numpy(dtype=np.float32)
        new_mat = normalize_rows(new_mat)

        # cosine similarity = dot of normalized vectors
        sims = np.matmul(new_mat, old_mat.T)
        best_idx = np.argmax(sims, axis=1)
        best_score = sims[np.arange(sims.shape[0]), best_idx]

        for nid, oi, sc in zip(new_ids, best_idx, best_score):
            results.append(
                {"new_cell_id": nid, "old_cell_id": old_ids[int(oi)], "max_cosine": float(sc)}
            )

    pd.DataFrame(results).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
