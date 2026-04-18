import argparse
from pathlib import Path

_PREPS_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS_ROOT = Path(
    "/Users/akdes/Library/CloudStorage/Box-Box/baylor/capstone_project_012026/inhouse_glioma_data/fine-tuned_models"
)
# Geneformer pickles live next to these scripts: preps/dict/*.pkl
DEFAULT_DICT_DIR = _PREPS_DIR / "dict"
TOKEN_DICT_FILENAME = "token_dictionary_gc30M.pkl"
GENE_MEDIAN_FILENAME = "gene_median_dictionary_gc30M.pkl"

parser = argparse.ArgumentParser(
    description="Annotate one dataset using fine-tuned models. Output cell embeddings (_preds.csv) and cell-type scores (_scores.csv)."
)
parser.add_argument("test_name", help="Input the test dataset to be annotated (e.g., mouse, glioma).")
parser.add_argument(
    "-g",
    "--gpu_name",
    choices=list(map(str, range(1000))),
    default="0",
    help="Input the idle GPU on which to run the code (e.g., 0, 1, 2).",
)
parser.add_argument(
    "--models-root",
    type=Path,
    default=DEFAULT_MODELS_ROOT,
    help=(
        "Directory containing reference folders (each <ref_name>/finetune/...). "
        f"Default: {DEFAULT_MODELS_ROOT}"
    ),
)
parser.add_argument(
    "--dict-dir",
    type=Path,
    default=DEFAULT_DICT_DIR,
    help=(
        f"Directory with Geneformer pickles (expects {TOKEN_DICT_FILENAME}). "
        f"Default: <preps>/dict next to annotate.py ({DEFAULT_DICT_DIR})"
    ),
)
args = parser.parse_args()
test_name = args.test_name
gpu_name = args.gpu_name
models_root = args.models_root.expanduser().resolve()
dict_dir = args.dict_dir.expanduser().resolve()
# Allow *.pkl directly in preps/ if not under preps/dict/
if not (dict_dir / TOKEN_DICT_FILENAME).is_file() and (_PREPS_DIR / TOKEN_DICT_FILENAME).is_file():
    dict_dir = _PREPS_DIR
    print(f"annotate.py: using dict pickles from {_PREPS_DIR} (not in dict/ subfolder)")

import os
import pickle
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_name
os.environ['NCCL_DEBUG'] = 'INFO'

from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertForSequenceClassification
from transformers import Trainer
from transformers.training_args import TrainingArguments

from geneformer import DataCollatorForCellClassification
from scipy.special import softmax
import shutil
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def pick_device() -> torch.device:
    """CUDA if available, else Apple Silicon MPS, else CPU (Mac has no CUDA)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = pick_device()
print(f"annotate.py: using device = {device}")

token_dictionary_path = dict_dir / TOKEN_DICT_FILENAME
if not token_dictionary_path.is_file():
    raise FileNotFoundError(
        f"Missing token dictionary: {token_dictionary_path}\n"
        f"Set --dict-dir to the folder containing {TOKEN_DICT_FILENAME} "
        f"(e.g. {_PREPS_DIR / 'dict'})."
    )
with open(token_dictionary_path, "rb") as _tf:
    token_dictionary = pickle.load(_tf)
print(f"annotate.py: loaded token_dictionary from {token_dictionary_path}")

gene_median_path = dict_dir / GENE_MEDIAN_FILENAME
gene_median_dictionary = None
if gene_median_path.is_file():
    with open(gene_median_path, "rb") as _gf:
        gene_median_dictionary = pickle.load(_gf)
    print(f"annotate.py: loaded gene_median_dictionary from {gene_median_path}")
else:
    print(
        f"annotate.py: optional {GENE_MEDIAN_FILENAME} not found in {dict_dir} "
        "(collator may still work if only token_dictionary is required)."
    )


def make_data_collator():
    """Build DataCollatorForCellClassification for newer Geneformer APIs."""
    kwargs = {"token_dictionary": token_dictionary}
    if gene_median_dictionary is not None:
        kwargs["gene_median_dictionary"] = gene_median_dictionary
    try:
        return DataCollatorForCellClassification(**kwargs)
    except TypeError:
        return DataCollatorForCellClassification(token_dictionary=token_dictionary)


data_collator = make_data_collator()

ann_output_directory = f"{test_name}_preds/"
os.makedirs(ann_output_directory, exist_ok=True)
token_dst = os.path.join(ann_output_directory, "tokenized_copy.dataset")
if os.path.isdir(token_dst):
    shutil.rmtree(token_dst)
shutil.copytree(f"{test_name}/{test_name}.dataset", token_dst)

ref_num_tups = [('aldinger_2000perCellType', 21), 
                ('allen_2000perCellType', 20), 
                ('bhaduri_3000perCellType', 10), 
                ('bhaduri_d2_4000perCellType', 10), 
                ('codex_1000perCellType', 16), 
                ('devbrain_3000perCellType', 10), 
                ('dirks_primary_gbm_combined_2000perCellType', 13), 
                ('primary_gbm_2000perCellType', 8), 
                ('recurrent_gbm_1000perCellType', 14), 
                ('TissueImmune_2000perCellType', 45)]

FINETUNE_SUBDIR = "finetune/240605_geneformer_CellClassifier_0_L2048_B12_LR5e-05_LSlinear_WU500_E10_Oadamw_F0"

for ref_name, num_classes in tqdm(ref_num_tups):
    output_prefix = f'preds_by_{ref_name}_num_classes_{num_classes}'
    model_directory = str(models_root / ref_name / FINETUNE_SUBDIR)
    target_names_xlsx = str(models_root / ref_name / "finetune" / "target_names.xlsx")


    # load data, labels, model
    tokenized_dataset = load_from_disk(ann_output_directory + 'tokenized_copy.dataset')
    # tokenized_dataset = tokenized_dataset.filter(lambda x: x['isTumor'] == 1, num_proc=1)
    labels = [0] * tokenized_dataset.num_rows
    tokenized_dataset = tokenized_dataset.add_column('label', labels)

    if not os.path.isfile(target_names_xlsx):
        raise FileNotFoundError(
            f"Missing target names file: {target_names_xlsx}\n"
            f"Expected under --models-root: {models_root}"
        )
    if not os.path.isdir(model_directory):
        raise FileNotFoundError(
            f"Missing model directory: {model_directory}\n"
            f"Expected under --models-root: {models_root}"
        )

    df_target_names = pd.read_excel(target_names_xlsx, header=None)


    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        # calculate accuracy and macro f1 using sklearn's function
        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average='macro')
        return {
        'accuracy': acc,
        'macro_f1': macro_f1
        }


    # set model parameters
    # max input size
    max_input_size = 2 ** 11  # 2048
    # number gpus
    num_gpus = 1
    # batch size for training and eval
    geneformer_batch_size = 12

    model = BertForSequenceClassification.from_pretrained(
        model_directory,
        num_labels=num_classes,
        output_attentions=False,
        output_hidden_states=False,
    ).to(device)

    # predict — Trainer must match device (CUDA / MPS / CPU)
    training_args: dict = {
        "do_train": False,
        "do_eval": False,
        "evaluation_strategy": "epoch",
        "group_by_length": True,
        "length_column_name": "length",
        "disable_tqdm": False,
        "per_device_train_batch_size": geneformer_batch_size,
        "per_device_eval_batch_size": geneformer_batch_size,
        "output_dir": ann_output_directory,
    }
    if device.type == "cpu":
        training_args["use_cpu"] = True
    elif device.type == "mps":
        # Apple Silicon GPU (not all transformers versions accept this kwarg)
        training_args["use_mps_device"] = True

    try:
        training_args_init = TrainingArguments(**training_args)
    except TypeError:
        training_args.pop("use_mps_device", None)
        training_args_init = TrainingArguments(**training_args)
    trainer = Trainer(
        model=model,
        args=training_args_init,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
    )
    predictions = trainer.predict(tokenized_dataset)

    df_preds = pd.DataFrame(predictions.predictions, index=tokenized_dataset['individual'])
    df_preds.index.name = 'individual'
    df_preds.to_csv(ann_output_directory + f'{output_prefix}_preds.csv', sep=',')

    y_predict_prob = softmax(predictions.predictions, axis=1)
    y_predict_int = np.argmax(y_predict_prob, axis=1)
    y_predict_class = [df_target_names.iat[i, 0] for i in y_predict_int]
    y_predict_score = np.amax(y_predict_prob, axis=1)

    # save results
    col_ann = f'{ref_name}_ann'
    col_score = f'{ref_name}_score'

    df_predict_prob = pd.DataFrame(y_predict_prob, index=tokenized_dataset['individual'])
    df_predict_prob.index.name = 'individual'
    df_predict_prob.columns = df_target_names.iloc[:, 0].tolist()
    df_predict_prob[col_ann] = y_predict_class
    df_predict_prob[col_score] = y_predict_score
    df_predict_prob.to_csv(ann_output_directory + f'{output_prefix}_scores.csv', sep=',')
