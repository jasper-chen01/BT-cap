#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_MODELS_ROOT = Path(
    "/Users/akdes/Library/CloudStorage/Box-Box/baylor/capstone_project_012026/inhouse_glioma_data/fine-tuned_models"
)
# Same folder as this script: ephys_prediction/preps/dict/
DEFAULT_DICT_DIR = Path(__file__).resolve().parent / "dict"


def run(cmd: list[str], cwd: Path) -> None:
    print(f"\nRunning: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate <test_name>_preds from a new input h5ad."
    )
    parser.add_argument(
        "input_h5ad",
        help="Path to the new input .h5ad file.",
    )
    parser.add_argument(
        "test_name",
        help="Dataset name used by PREPS, e.g. glioma2.",
    )
    parser.add_argument(
        "-s",
        "--species",
        choices=["human", "mouse"],
        default="human",
        help="Species passed to preps_tokenize.py (default: human).",
    )
    parser.add_argument(
        "-g",
        "--gpu-name",
        default="0",
        help="GPU id passed to annotate.py (default: 0).",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy the input file into PREPS instead of symlinking it.",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable to use for preps_tokenize.py and annotate.py (default: current interpreter).",
    )
    parser.add_argument(
        "--models-root",
        type=Path,
        default=DEFAULT_MODELS_ROOT,
        help=(
            "Directory passed to annotate.py containing <ref_name>/finetune/... "
            f"(default: {DEFAULT_MODELS_ROOT})"
        ),
    )
    parser.add_argument(
        "--dict-dir",
        type=Path,
        default=DEFAULT_DICT_DIR,
        help=(
            "Geneformer dict dir (passed to preps_tokenize.py and annotate.py; needs gc30M pkls). "
            f"(default: {DEFAULT_DICT_DIR})"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preps_dir = Path(__file__).resolve().parent
    input_h5ad = Path(args.input_h5ad).expanduser().resolve()

    if not input_h5ad.is_file():
        raise FileNotFoundError(f"Input h5ad not found: {input_h5ad}")

    test_dir = preps_dir / args.test_name
    test_dir.mkdir(parents=True, exist_ok=True)
    target_h5ad = test_dir / "adata.h5ad"

    if target_h5ad.exists() or target_h5ad.is_symlink():
        target_h5ad.unlink()

    if args.copy:
        shutil.copy2(input_h5ad, target_h5ad)
        print(f"Copied input to {target_h5ad}")
    else:
        target_h5ad.symlink_to(input_h5ad)
        print(f"Symlinked input to {target_h5ad}")

    dict_dir = args.dict_dir.expanduser().resolve()
    run(
        [
            args.python_bin,
            "preps_tokenize.py",
            args.test_name,
            "-s",
            args.species,
            "--dict-dir",
            str(dict_dir),
        ],
        cwd=preps_dir,
    )
    models_root = args.models_root.expanduser().resolve()
    run(
        [
            args.python_bin,
            "annotate.py",
            args.test_name,
            "-g",
            args.gpu_name,
            "--models-root",
            str(models_root),
            "--dict-dir",
            str(dict_dir),
        ],
        cwd=preps_dir,
    )

    print("\nDone.")
    print(f"Predictions folder: {preps_dir / f'{args.test_name}_preds'}")


if __name__ == "__main__":
    main()
