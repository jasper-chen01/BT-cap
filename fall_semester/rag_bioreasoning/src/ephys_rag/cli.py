from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ephys_rag.chain import RAGEngine
from ephys_rag.config import DATA_DIR
from ephys_rag.contrasts import pathway_contrasts, summarize_dataset
from ephys_rag.ingest import file_inventory, load_interactions
from ephys_rag.tools import ToolRegistry


def _configure_stdout() -> None:
    """Avoid Windows cp1252 crashes on biology unicode in traces."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _cmd_stats(_: argparse.Namespace) -> None:
    interactions = load_interactions()
    summary = summarize_dataset(interactions)
    payload = {
        "DATA_DIR": str(DATA_DIR),
        "files": file_inventory(),
        "cellchat": summary,
    }
    print(json.dumps(payload, indent=2))


def _cmd_contrast(args: argparse.Namespace) -> None:
    interactions = load_interactions()
    rows = pathway_contrasts(interactions)
    if args.compartment:
        rows = [row for row in rows if row.compartment == args.compartment]
    print(
        f"{'pathway':<16} {'comp':<8} {'Ephys_1':>8} {'Ephys_2':>8} {'delta':>7} {'mean_p1':>8} {'mean_p2':>8}"
    )
    for row in rows:
        if args.min_delta is not None and abs(row.delta_e2_minus_e1) < args.min_delta:
            continue
        print(
            f"{row.pathway:<16} {row.compartment:<8} {row.ephys_1:8d} {row.ephys_2:8d} "
            f"{row.delta_e2_minus_e1:7d} {row.mean_prob_e1:8.3f} {row.mean_prob_e2:8.3f}"
        )


def _cmd_tools(args: argparse.Namespace) -> None:
    registry = ToolRegistry()
    results = []
    if args.gene:
        results.append(
            registry.cellchat_lookup(
                gene=args.gene,
                celltype=args.celltype,
                ephys=args.ephys,
                compartment=args.compartment,
            )
        )
        results.append(registry.deg_lookup(gene=args.gene, celltype=args.celltype, idh=args.idh))
    if args.ligand and args.receptor:
        results.append(
            registry.support_join(
                ligand=args.ligand,
                receptor=args.receptor,
                sender=args.celltype,
                idh=args.idh,
            )
        )
    if args.counts or not results:
        results.append(registry.count_lookup(celltype=args.celltype, ephys=args.ephys))
        results.append(registry.annotation_lookup(compartment=args.compartment, celltype=args.celltype))
    if args.theme:
        results.append(
            registry.pathway_filter(theme=args.theme, compartment=args.compartment, ephys=args.ephys)
        )
    for item in results:
        print(item.as_text())
        print()


def _cmd_ask(args: argparse.Namespace) -> None:
    engine = RAGEngine.from_disk()
    result = engine.ask(
        args.question,
        top_k=args.top_k,
        use_llm=None if args.llm is None else args.llm,
        provider=args.provider,
        dual=args.dual,
    )
    print(result["answer"])
    print("\n--- tool traces ---")
    for trace in result["traces"]:
        print(f"{trace.tool}: {trace.n_hits} hits from {trace.source_file}")
    print("\n--- retrieved ---")
    for hit in result["hits"]:
        print(f"{hit.score:.3f}  [{hit.document.kind}] {hit.document.title}")


def _invention_flags(text: str, context: str) -> list[str]:
    from ephys_rag.llm import _invention_flags as _flags

    return _flags(text, context)


WEEK2_QUESTIONS = [
    (9, "Is PTN an Ephys discriminator among tumor senders, or a shared scaffold?", ["PTN", "PTPRZ1"]),
    (10, "Which receptor does MHC-II use here?", ["CD4"]),
    (11, "What is the APOE receptor complex in this table?", ["TREM2", "TYROBP"]),
    (12, "Which T-cell-sent pathway is private to Ephys_2?", ["EGF", "AREG"]),
    (13, "Name a tumor-sender pathway that is exclusive (or nearly exclusive) to Ephys_1.", ["ANNEXIN", "MHC-I", "MHC-II"]),
    (14, "Which receptors does SPP1 use here?", ["CD44", "ITGAV"]),
    (15, "What does CLEC bind on T cells in this table?", ["KLRB1"]),
    (16, "Do CellChat p-values vary in this export?", ["pval"]),
    (17, "CNTN1 binds which partners here?", ["NRCAM", "NOTCH1"]),
    (18, "What is the highest-probability myeloid self-loop?", ["APOE_TREM2_TYROBP"]),
    (19, "Interpret NRXN1–NLGN1 and glutamate–GRIA in this dataset. Are neurons required for that interpretation?", ["NRXN", "Glutamate"]),
    (20, "If a model claims NRXN is Ephys_1-enriched in tumor, what count evidence would refute it?", ["NRXN"]),
    (21, "Is NRXN1 DEG-supported as Ephys_2-high in IDH-mutant cycling tumor?", ["NRXN1"]),
    (22, "Is HLA-DRA DEG-supported as Ephys_1-high in tumor?", ["HLA-DRA"]),
    (23, "Is AREG a T-cell Ephys_2-high DEG?", ["AREG"]),
    (24, "Does GRIA2 go the same Ephys direction in IDH-mutant and IDH-WT OPC-like tumor?", ["GRIA2"]),
    (25, "Is EGFR Ephys_2-high in IDH-mutant cycling tumor? Does IDH-WT OPC-like tumor agree?", ["EGFR"]),
    (26, "In IDH-mutant TAM/microglia, is HLA-DRA Ephys_1-high?", ["HLA-DRA"]),
    (27, "Which labeled group has only 6 cells?", ["MES", "6"]),
    (28, "How many T cells are labeled Ephys_1 vs Ephys_2?", ["T cell"]),
    (29, "Is AC-like tumor / Ephys_2 large enough to treat as a trusted sender?", ["AC_like"]),
    (30, "Is NRXN1 both a CellChat ligand and a DEG in IDH-mutant cycling Ephys_2?", ["NRXN1"]),
    (31, "Is AREG both the T-cell EGF ligand in CellChat and a T-cell DEG?", ["AREG"]),
    (32, "Does HLA-DRA appear as both an MHC-II CellChat ligand and an Ephys_1-high tumor DEG?", ["HLA-DRA", "MHC-II"]),
]


def _cmd_week2(args: argparse.Namespace) -> None:
    """Run Q9–Q32 with MedGemma + Gemini on the same traces; write turn-in JSON/CSV."""
    engine = RAGEngine.from_disk()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    questions = WEEK2_QUESTIONS
    if args.only:
        wanted = {int(x) for x in args.only.split(",")}
        questions = [q for q in questions if q[0] in wanted]

    for qid, question, required in questions:
        print(f"\n===== Q{qid} =====\n{question}")
        result = engine.ask(question, top_k=args.top_k, use_llm=True, dual=True)
        context = result["context"]
        tools = [t.tool for t in result["traces"]]
        blob = context
        term_hits = {term: term.lower() in blob.lower() for term in required}
        required_ok = any(term_hits.values()) if qid in {12, 13, 14, 17} else all(term_hits.values())
        if qid == 11:
            required_ok = ("TREM2" in blob.upper() and "TYROBP" in blob.upper()) or (
                "APOE_TREM2_TYROBP" in blob.upper()
            )
        if qid == 16:
            required_ok = "pval" in blob.lower()
        if qid == 18:
            required_ok = "APOE_TREM2_TYROBP" in blob.upper()
        if qid == 27:
            required_ok = "MES" in blob.upper() and "6" in blob

        med = result["answers"].get("medgemma")
        gem = result["answers"].get("gemini")
        med_text = (med.text if med and not med.error else "") or ""
        gem_text = (gem.text if gem and not gem.error else "") or ""
        med_err = med.error if med else "missing"
        gem_err = gem.error if gem else "missing"
        invent_med = _invention_flags(med_text, context)
        invent_gem = _invention_flags(gem_text, context)

        payload = {
            "question_id": qid,
            "question": question,
            "required_terms": required,
            "term_found": term_hits,
            "required_ok": required_ok,
            "tools": [
                {
                    "tool": t.tool,
                    "file": t.source_file,
                    "n_hits": t.n_hits,
                    "filters": t.filters,
                    "preview": [
                        r.get("pair") or r.get("pathway") or r.get("gene") or r.get("group")
                        for r in t.rows[:8]
                    ],
                }
                for t in result["traces"]
            ],
            "retrieved_titles": [f"{h.score:.3f} [{h.document.kind}] {h.document.title}" for h in result["hits"]],
            "medgemma": med_text,
            "medgemma_error": med_err,
            "gemini": gem_text,
            "gemini_error": gem_err,
            "invented_medgemma": invent_med,
            "invented_gemini": invent_gem,
            "context": context,
        }
        (out_dir / f"Q{qid}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        (out_dir / f"Q{qid}_trace.txt").write_text(context, encoding="utf-8")
        (out_dir / f"Q{qid}_answers.txt").write_text(
            f"QUESTION: {question}\n\n=== MedGemma ===\n{med_text or med_err}\n\n=== Gemini ===\n{gem_text or gem_err}\n",
            encoding="utf-8",
        )
        row = {
            "Question": f"Q{qid}",
            "Tool that fired": ", ".join(tools),
            "Required term in traces?": "yes" if required_ok else f"no ({term_hits})",
            "MedGemma answer": (med_text or f"[error] {med_err}")[:1200],
            "Gemini answer": (gem_text or f"[error] {gem_err}")[:1200],
            "Either invented a pair/DEG?": (
                f"medgemma={invent_med or 'no'}; gemini={invent_gem or 'no'}"
            ),
        }
        rows.append(row)
        print(f"  tools={tools} required_ok={required_ok}")
        print(f"  medgemma_err={med_err} gemini_err={gem_err}")

    summary_path = out_dir / "WEEK2_Q9_Q32_SUMMARY.json"
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # Markdown table for turn-in
    md_lines = [
        "# Week 2 — Q9–Q32 MedGemma vs Gemini",
        "",
        "| Question | Tool that fired | Required term in traces? | MedGemma answer | Gemini answer | Either model invented a pair/DEG? |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        def cell(s: str) -> str:
            return s.replace("|", "/").replace("\n", "<br>")[:800]

        md_lines.append(
            "| "
            + " | ".join(
                cell(row[k])
                for k in [
                    "Question",
                    "Tool that fired",
                    "Required term in traces?",
                    "MedGemma answer",
                    "Gemini answer",
                    "Either invented a pair/DEG?",
                ]
            )
            + " |"
        )
    (out_dir / "WEEK2_Q9_Q32_SUMMARY.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nWrote {summary_path}")


def _cmd_rank(args: argparse.Namespace) -> None:
    from ephys_rag.lgbm_rank import rank, write_outputs

    result = rank(
        universe=args.universe,
        tiers=[int(t) for t in args.tiers.split(",")],
        llm_runs=Path(args.llm_runs) if args.llm_runs else None,
        llm_keys=[k.strip() for k in args.llm_keys.split(",") if k.strip()],
        seeds=list(range(args.seeds)),
        n_splits=args.folds,
    )
    paths = write_outputs(result, Path(args.out_dir), top=args.top)
    m = result.metrics
    lgbm = m["row_level"]["lightgbm_oof"]
    print(
        f"rows={m['n_rows']} genes={m['n_genes']} positives={m['n_positive_rows']} rows / "
        f"{m['n_positive_genes']} genes, llm_runs={m['n_llm_runs']}"
    )
    print(f"OOF AUROC={lgbm['auroc']:.3f} AUPRC={lgbm['auprc']:.3f} (prevalence {m['prevalence_rows']:.3f})")
    cols = ["rank", "gene", "celltype_id", "IDH_status", "score", "status", "rule_pass"]
    print(result.nominees[cols].head(args.top).to_string(index=False))
    for path in paths:
        print(f"wrote {path}")


def _cmd_fetch_genesets(_: argparse.Namespace) -> None:
    from ephys_rag.gene_sets import GENE_SET_DIR, fetch_all

    sources = fetch_all()
    print(json.dumps(sources, indent=2))
    print(f"wrote gene sets to {GENE_SET_DIR}")


def _cmd_rank_genesets(args: argparse.Namespace) -> None:
    from ephys_rag.lgbm_genesets import rank_gene_sets, write_gene_set_outputs

    result = rank_gene_sets(
        universe=args.universe,
        sets=[s.strip() for s in args.sets.split(",") if s.strip()],
        immune_human_only=args.immune_human_only,
        llm_runs=Path(args.llm_runs) if args.llm_runs else None,
        llm_keys=[k.strip() for k in args.llm_keys.split(",") if k.strip()],
        seeds=list(range(args.seeds)),
        n_splits=args.folds,
    )
    paths = write_gene_set_outputs(result, Path(args.out_dir), top=args.top)
    m = result.metrics
    print(f"rows={m['n_rows']} genes={m['n_genes']} features={m['n_features']}")
    for name, s in m["sets"].items():
        row = s["row_level"]
        check = s["published_check"]
        print(
            f"{name}: positives={s['n_positive_rows']} rows / {s['n_positive_genes']} genes | "
            f"LGBM AUROC={row['lightgbm_oof']['auroc']:.3f} AUPRC={row['lightgbm_oof']['auprc']:.3f} "
            f"(prevalence {s['prevalence_rows']:.3f}) | published-33 median pct="
            f"{check.get('median_percentile_published', float('nan')):.2f}"
        )
    for path in paths:
        print(f"wrote {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bioreasoning RAG over fall_semester/data")
    sub = parser.add_subparsers(dest="command", required=True)

    stats = sub.add_parser("stats", help="Data dictionary + CellChat summary JSON")
    stats.set_defaults(func=_cmd_stats)

    contrast = sub.add_parser("contrast", help="Ephys_1 vs Ephys_2 pathway table")
    contrast.add_argument("--compartment", choices=["tumor", "tcell", "myeloid"])
    contrast.add_argument("--min-delta", type=int, default=None)
    contrast.set_defaults(func=_cmd_contrast)

    tools = sub.add_parser("tools", help="Run Challenge B-style table lookups")
    tools.add_argument("--gene")
    tools.add_argument("--ligand")
    tools.add_argument("--receptor")
    tools.add_argument("--celltype")
    tools.add_argument("--idh")
    tools.add_argument("--ephys")
    tools.add_argument("--compartment", choices=["tumor", "tcell", "myeloid", "other"])
    tools.add_argument("--theme")
    tools.add_argument("--counts", action="store_true")
    tools.set_defaults(func=_cmd_tools)

    ask = sub.add_parser("ask", help="Tools + TF-IDF retrieve (LLM optional)")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=10)
    ask.add_argument("--llm", action=argparse.BooleanOptionalAction, default=None)
    ask.add_argument("--provider", choices=["medgemma", "gemini"], default=None)
    ask.add_argument("--dual", action="store_true", help="Run MedGemma and Gemini on the same traces")
    ask.set_defaults(func=_cmd_ask)

    week2 = sub.add_parser("week2", help="Batch Q9–Q32 with MedGemma + Gemini")
    week2.add_argument("--top-k", type=int, default=10)
    week2.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[3] / "turnin" / "week2_traces"),
    )
    week2.add_argument("--only", help="Comma-separated question ids, e.g. 9,10,21")
    week2.set_defaults(func=_cmd_week2)

    rank_p = sub.add_parser("rank", help="LightGBM PU ranker: nominate gene x cell type x IDH")
    rank_p.add_argument("--universe", choices=["cellchat", "deg"], default="cellchat")
    rank_p.add_argument("--tiers", default="1,2", help="Literature tiers used as labels, e.g. 1 or 1,2")
    rank_p.add_argument("--llm-runs", help="Folder of repeated LLM runs (JSON/JSONL answers)")
    rank_p.add_argument("--llm-keys", default="answer,response,text,ollama,medgemma")
    rank_p.add_argument("--seeds", type=int, default=5)
    rank_p.add_argument("--folds", type=int, default=5)
    rank_p.add_argument("--top", type=int, default=30)
    rank_p.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[3] / "turnin" / "lgbm"),
    )
    rank_p.set_defaults(func=_cmd_rank)

    fetch_p = sub.add_parser("fetch-genesets", help="Download SynGO + GO:0001772 gene sets")
    fetch_p.set_defaults(func=_cmd_fetch_genesets)

    gs = sub.add_parser("rank-genesets", help="LightGBM with SynGO / GO:0001772 labels (two scores)")
    gs.add_argument("--universe", choices=["cellchat", "deg"], default="deg")
    gs.add_argument("--sets", default="tumor_synaptic,immune_synapse")
    gs.add_argument("--immune-human-only", action="store_true", help="GO:0001772 genes with a human annotation only")
    gs.add_argument("--llm-runs", help="Folder of repeated LLM runs (JSON/JSONL answers)")
    gs.add_argument("--llm-keys", default="answer,response,text,ollama,medgemma")
    gs.add_argument("--seeds", type=int, default=5)
    gs.add_argument("--folds", type=int, default=5)
    gs.add_argument("--top", type=int, default=25)
    gs.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[3] / "turnin" / "lgbm_genesets"),
    )
    gs.set_defaults(func=_cmd_rank_genesets)
    return parser


def main() -> None:
    _configure_stdout()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
