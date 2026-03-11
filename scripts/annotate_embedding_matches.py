import argparse
import os

import pandas as pd


def annotate_matches(matches_csv: str, celltypes_csv: str, out_csv: str) -> None:
    matches_df = pd.read_csv(matches_csv)
    if matches_df.empty:
        matches_df.to_csv(out_csv, index=False)
        return

    celltypes_df = pd.read_csv(celltypes_csv)
    if celltypes_df.empty:
        matches_df.to_csv(out_csv, index=False)
        return

    first_col = celltypes_df.columns[0]
    if first_col.startswith("Unnamed") or first_col == "":
        celltypes_df = celltypes_df.rename(columns={first_col: "cell_id"})
    else:
        celltypes_df = celltypes_df.rename(columns={first_col: "cell_id"})

    celltype_col = "seuratObj.CellType"
    if celltype_col not in celltypes_df.columns:
        for col in celltypes_df.columns:
            if col != "cell_id":
                celltype_col = col
                break

    mapping = dict(
        zip(
            celltypes_df["cell_id"].astype(str),
            celltypes_df[celltype_col].astype(str),
        )
    )

    matches_df = matches_df.rename(
        columns={
            "new_cell_id": "cell_id",
            "old_cell_id": "matched_trained_cell_id",
        }
    )
    matches_df["matched_cell_type"] = matches_df["matched_trained_cell_id"].astype(
        str
    ).map(mapping)
    matches_df.to_csv(out_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge embedding matches with cell type annotations."
    )
    parser.add_argument(
        "--matches",
        required=True,
        help="CSV with embedding matches (new_cell_id, old_cell_id, max_cosine).",
    )
    parser.add_argument(
        "--celltypes",
        required=True,
        help="CSV with cell type annotations.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output CSV with matched cell types.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.matches):
        raise FileNotFoundError(f"Matches CSV not found: {args.matches}")
    if not os.path.isfile(args.celltypes):
        raise FileNotFoundError(f"Cell types CSV not found: {args.celltypes}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    annotate_matches(args.matches, args.celltypes, args.out)


if __name__ == "__main__":
    main()

