"""Train scETM on in-house glioma scRNA-seq and export gene / topic / cell embeddings.

Adapted from BioreasoningChallengeB/extract_gene_embeddings.py, but trained on our glioma
cells instead of the challenge Perturb-seq reference. scETM factorizes expression into
topics; each gene gets a learned vector (rho) in the same space as the topics, so genes in
the same expression programs end up close together. No labels are used.

Needs an environment with scETM, scanpy, anndata, torch (not the ephys-rag venv), e.g.:
    py -3.13 scripts/extract_scetm_gene_embeddings.py --max-cells-per-group 1800 --n-epochs 1000

Input (default): inhouse_scell_glioma/adata.h5ad, raw counts, gene symbols in var_names,
obs with cell_id, CellType, location, orig.ident (sample). The full file has ~255k cells, so
by default a CellType-stratified subsample is read from disk without loading everything.

Outputs (knowledge/gene_embeddings/), {stem} = scetm_gene_embeddings_d{dim}:
    {stem}.csv            rho:   gene, n_cells_detected, emb_00..emb_{dim-1}   (genes x dim)
    {stem}.alpha.csv      alpha: topic, emb_00..                              (topics x dim)
    {stem}.theta.csv.gz   theta: cell_id, CellType, location, sample, topic_00..  (cells x topics)
    {stem}.meta.json      input, parameters, timings, coverage, QC numbers
    {stem}.qc.md          coverage, gene-set coherence, nearest neighbours, topic summary
Checkpoints go to artifacts/scetm/ (git-ignored).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import date
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]  # rag_bioreasoning/
DEFAULT_H5AD = ROOT.parents[2] / "pertubation_modelling" / "inhouse_scell_glioma" / "adata.h5ad"
OUT_DIR = ROOT / "knowledge" / "gene_embeddings"
CKPT_DIR = ROOT / "artifacts" / "scetm"
GENE_SET_DIR = ROOT / "knowledge" / "gene_sets"
LITERATURE_YAML = ROOT / "knowledge" / "literature_positives.yaml"
DEG_IDH_CSV = (
    ROOT.parent
    / "data"
    / "within_cluster_ephys_DEGs_by_IDH"
    / "combined_Ephys2_vs_Ephys1_significant_DEGs_by_IDH_and_cluster.csv"
)
CELLCHAT_CSV = ROOT.parent / "data" / "cellchat_ephys_plus_celltype" / "all_significant_interactions.csv"
QC_GENES = ["GRIA2", "NLGN3", "NRXN1", "PTPRZ1", "PDGFRA", "EGFR", "CD74", "HLA-DRA", "SPP1", "CD3E", "GFAP", "MKI67"]
META_COLS = ["cell_id", "CellType", "location", "tumorType", "orig.ident"]


def select_cells(obs: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    rng = np.random.default_rng(args.seed)
    idx = np.arange(len(obs))
    if args.stratify_col and args.stratify_col in obs and args.max_cells_per_group:
        picked = []
        for _, group in obs.groupby(args.stratify_col, observed=True).indices.items():
            n = min(len(group), args.max_cells_per_group)
            picked.append(rng.choice(group, n, replace=False))
        idx = np.concatenate(picked)
    if args.max_cells and len(idx) > args.max_cells:
        idx = rng.choice(idx, args.max_cells, replace=False)
    return np.sort(idx)


def load_counts(args: argparse.Namespace) -> ad.AnnData:
    backed = ad.read_h5ad(args.h5ad, backed="r")
    idx = select_cells(backed.obs, args)
    adata = backed[idx].to_memory()
    backed.file.close()
    X = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    X = X.astype(np.float32).tocsr()
    sample = X[: min(500, X.shape[0])].toarray()
    if not np.allclose(sample, np.round(sample)):
        raise ValueError("scETM expects raw counts; adata.X is not integer-valued.")
    detected = np.asarray((X > 0).sum(axis=0)).ravel()
    keep = detected >= args.min_cells
    adata = ad.AnnData(X=X[:, keep], obs=adata.obs.copy(), var=adata.var.loc[keep].copy())
    adata.var["n_cells_detected"] = detected[keep]
    adata = adata[np.asarray(adata.X.sum(axis=1)).ravel() > 0].copy()
    adata.var_names_make_unique()
    return adata


def _batch_codes(adata: ad.AnnData, args: argparse.Namespace) -> tuple[str | None, int]:
    if not args.batch_col or args.batch_col not in adata.obs:
        return None, 1
    values = adata.obs[args.batch_col].astype(str)
    if args.batch_suffix_sep:
        values = values.str.rsplit(args.batch_suffix_sep, n=1).str[-1]
    codes = pd.Categorical(values).codes
    n_batches = int(codes.max() + 1)
    if not 1 < n_batches <= args.max_batches:
        return None, 1
    adata.obs["batch_indices"] = codes
    return "batch_indices", n_batches


def train(adata: ad.AnnData, args: argparse.Namespace) -> tuple[dict, dict]:
    import torch
    from scETM import UnsupervisedTrainer, scETM

    torch.set_num_threads(max(os.cpu_count() or 1, 1))
    batch_col, n_batches = _batch_codes(adata, args)
    model = scETM(
        adata.n_vars,
        n_batches=n_batches,
        n_topics=args.n_topics,
        trainable_gene_emb_dim=args.emb_dim,
    )
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    trainer = UnsupervisedTrainer(
        model,
        adata,
        ckpt_dir=str(CKPT_DIR),
        test_ratio=args.test_ratio,
        init_lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    start = time.time()
    trainer.train(
        n_epochs=args.n_epochs,
        eval_every=max(args.n_epochs // 4, 1),
        n_samplers=1,
        eval=False,
        batch_col=batch_col or "batch_indices",
        save_model_ckpt=True,
    )
    elapsed = time.time() - start
    model.eval()
    test_nll = None
    if trainer.test_adata is not trainer.adata:
        test_nll = model.get_cell_embeddings_and_nll(
            trainer.test_adata, args.batch_size, batch_col=batch_col or "batch_indices", emb_names=[]
        )
    embs, _ = model.get_all_embeddings_and_nll(
        adata, args.batch_size, emb_names=["theta"], batch_col=batch_col or "batch_indices", inplace=False
    )
    info = {
        "n_batches": n_batches,
        "batch_col": args.batch_col if n_batches > 1 else None,
        "train_seconds": round(elapsed, 1),
        "test_nll": None if test_nll is None else float(test_nll),
        "checkpoint_dir": str(trainer.ckpt_dir),
    }
    return embs, info


def _unit(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)


def coherence(unit: np.ndarray, index: dict[str, int], genes: set[str], rng: np.random.Generator, n_perm: int = 200) -> dict:
    """Mean pairwise cosine within a gene set vs random sets of the same size."""
    idx = [index[g] for g in genes if g in index]
    if len(idx) < 3:
        return {"n_found": len(idx)}
    sub = unit[idx]
    k = len(idx)
    observed = float((sub @ sub.T)[np.triu_indices(k, 1)].mean())
    null = []
    for _ in range(n_perm):
        r = unit[rng.choice(unit.shape[0], k, replace=False)]
        null.append(float((r @ r.T)[np.triu_indices(k, 1)].mean()))
    null = np.array(null)
    return {
        "n_found": k,
        "mean_cosine": observed,
        "random_mean": float(null.mean()),
        "random_sd": float(null.std()),
        "z": float((observed - null.mean()) / max(null.std(), 1e-12)),
        "empirical_p": float((np.sum(null >= observed) + 1) / (n_perm + 1)),
    }


def _gene_lists() -> dict[str, set[str]]:
    sets: dict[str, set[str]] = {}
    for name, fname in [("SynGO (tumor_synaptic)", "tumor_synaptic_syngo.tsv"), ("GO:0001772 (immune_synapse)", "immune_synapse_go0001772.tsv")]:
        path = GENE_SET_DIR / fname
        if path.exists():
            sets[name] = set(pd.read_csv(path, sep="\t")["symbol"].astype(str))
    if LITERATURE_YAML.exists():
        import yaml

        payload = yaml.safe_load(LITERATURE_YAML.read_text(encoding="utf-8")) or {}
        sets["published 33"] = {p["gene"] for p in payload.get("positives", [])}
    if DEG_IDH_CSV.exists():
        sets["DEG universe"] = set(pd.read_csv(DEG_IDH_CSV, usecols=["gene"])["gene"].astype(str))
    if CELLCHAT_CSV.exists():
        cc = pd.read_csv(CELLCHAT_CSV, usecols=["ligand", "receptor"])
        genes = set()
        for value in pd.concat([cc["ligand"], cc["receptor"]]):
            for token in str(value).split("_"):
                genes.add(token.split("-", 1)[1] if token.startswith(("Glu-", "PGE2-")) else token)
        sets["CellChat L/R genes"] = genes
    return sets


def qc_report(genes: list[str], rho: np.ndarray, alpha: np.ndarray, theta: pd.DataFrame, seed: int) -> tuple[dict, str]:
    rng = np.random.default_rng(seed)
    unit = _unit(rho)
    index = {g: i for i, g in enumerate(genes)}
    lists = _gene_lists()
    coverage = {
        name: {"n": len(s), "n_embedded": len(s & set(index)), "fraction": round(len(s & set(index)) / max(len(s), 1), 3)}
        for name, s in lists.items()
    }
    coherent = {name: coherence(unit, index, s, rng) for name, s in lists.items() if name != "DEG universe"}

    lines = ["## Coverage", "", "| gene list | genes | embedded | fraction |", "|---|---|---|---|"]
    lines += [f"| {n} | {c['n']} | {c['n_embedded']} | {c['fraction']:.3f} |" for n, c in coverage.items()]
    lines += [
        "",
        "## Gene-set coherence",
        "",
        "Mean pairwise cosine similarity inside each list vs 200 random gene sets of the same size.",
        "z > 2 means the list's genes sit closer together in embedding space than chance.",
        "",
        "| gene list | n | mean cosine | random mean | z | empirical p |",
        "|---|---|---|---|---|---|",
    ]
    for n, c in coherent.items():
        if "mean_cosine" in c:
            lines.append(f"| {n} | {c['n_found']} | {c['mean_cosine']:.3f} | {c['random_mean']:.3f} | {c['z']:.1f} | {c['empirical_p']:.3f} |")
    lines += ["", "## Nearest neighbours (cosine on rho)", "", "| gene | top 8 neighbours |", "|---|---|"]
    for g in QC_GENES:
        if g not in index:
            lines.append(f"| {g} | not embedded |")
            continue
        sims = unit @ unit[index[g]]
        order = [i for i in np.argsort(-sims) if i != index[g]][:8]
        lines.append(f"| {g} | " + ", ".join(f"{genes[i]} ({sims[i]:.2f})" for i in order) + " |")

    beta = alpha @ rho.T  # topics x genes
    topic_cols = [c for c in theta.columns if c.startswith("topic_")]
    lines += [
        "",
        "## Topics",
        "",
        "Top genes per topic (beta = alpha @ rho^T) and the cell type where the topic is most active (mean theta).",
        "",
        "| topic | top 10 genes | most active in | mean theta there |",
        "|---|---|---|---|",
    ]
    by_type = theta.groupby("CellType", observed=True)[topic_cols].mean() if "CellType" in theta else None
    topics = []
    for t in range(beta.shape[0]):
        top = [genes[i] for i in np.argsort(-beta[t])[:10]]
        where, level = ("", float("nan"))
        if by_type is not None:
            col = topic_cols[t]
            where, level = by_type[col].idxmax(), float(by_type[col].max())
        topics.append({"topic": t, "top_genes": top, "most_active_in": where, "mean_theta": level})
        lines.append(f"| {t:02d} | {', '.join(top)} | {where} | {level:.3f} |")
    return {"coverage": coverage, "coherence": coherent, "topics": topics}, "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--emb-dim", type=int, default=32)
    parser.add_argument("--n-topics", type=int, default=50)
    parser.add_argument("--n-epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--min-cells", type=int, default=10, help="Keep genes detected in at least this many cells")
    parser.add_argument("--stratify-col", default="CellType", help="obs column for stratified subsampling")
    parser.add_argument("--max-cells-per-group", type=int, default=1800, help="Cap per stratum; 0 = no stratified cap")
    parser.add_argument("--max-cells", type=int, default=None, help="Overall cap after stratification")
    parser.add_argument("--batch-col", default="orig.ident", help="obs column for scETM batch correction (sample)")
    parser.add_argument(
        "--batch-suffix-sep",
        default="",
        help="Use the text after the last separator as batch (e.g. '_' when the column is barcode_sample).",
    )
    parser.add_argument("--max-batches", type=int, default=200)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    adata = load_counts(args)
    print(f"cells={adata.n_obs} genes={adata.n_vars} (detected in >= {args.min_cells} cells)")
    embs, info = train(adata, args)
    rho, alpha, theta_arr = embs["rho"], embs["alpha"], embs["theta"]

    genes = list(adata.var_names)
    emb_cols = [f"emb_{i:02d}" for i in range(rho.shape[1])]
    table = pd.DataFrame(rho, columns=emb_cols)
    table.insert(0, "gene", genes)
    table.insert(1, "n_cells_detected", adata.var["n_cells_detected"].to_numpy())
    alpha_df = pd.DataFrame(alpha, columns=emb_cols)
    alpha_df.insert(0, "topic", [f"topic_{i:02d}" for i in range(alpha.shape[0])])
    theta = pd.DataFrame(theta_arr, columns=[f"topic_{i:02d}" for i in range(theta_arr.shape[1])])
    for col in reversed(META_COLS):
        if col in adata.obs:
            theta.insert(0, col, adata.obs[col].astype(str).to_numpy())

    qc, qc_md = qc_report(genes, rho, alpha, theta, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"scetm_gene_embeddings_d{args.emb_dim}"
    table.to_csv(args.out_dir / f"{stem}.csv", index=False, float_format="%.6g")
    alpha_df.to_csv(args.out_dir / f"{stem}.alpha.csv", index=False, float_format="%.6g")
    theta.to_csv(args.out_dir / f"{stem}.theta.csv.gz", index=False, float_format="%.4g", compression="gzip")
    cell_types = adata.obs["CellType"].astype(str).value_counts().to_dict() if "CellType" in adata.obs else {}
    meta = {
        "created": date.today().isoformat(),
        "input_h5ad": str(args.h5ad),
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "cells_per_type": cell_types,
        "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        **info,
        **qc,
    }
    (args.out_dir / f"{stem}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    header = [
        f"# scETM gene embeddings ({args.emb_dim}-d) QC",
        "",
        f"- Input: `{args.h5ad.name}`: {adata.n_obs} cells (stratified by `{args.stratify_col}`, cap "
        f"{args.max_cells_per_group}/group), {adata.n_vars} genes detected in >= {args.min_cells} cells.",
        f"- scETM: {args.n_topics} topics, {args.n_epochs} epochs, batch size {args.batch_size}, lr {args.lr}, "
        f"batch correction on `{info['batch_col']}` ({info['n_batches']} batches). Trained in {info['train_seconds']} s on CPU.",
        f"- Held-out test NLL: {info['test_nll']}.",
        "",
    ]
    (args.out_dir / f"{stem}.qc.md").write_text("\n".join(header) + qc_md + "\n", encoding="utf-8")
    print(f"wrote {args.out_dir / (stem + '.csv')}  rho={rho.shape} alpha={alpha.shape} theta={theta_arr.shape}")
    print(qc_md)


if __name__ == "__main__":
    main()
