from __future__ import annotations

import argparse
import json

from ephys_rag.chain import RAGEngine
from ephys_rag.config import DATA_DIR
from ephys_rag.contrasts import pathway_contrasts, summarize_dataset
from ephys_rag.evaluation import (
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_DIR,
    load_question_manifest,
    run_evaluation,
    write_evaluation_reports,
)
from ephys_rag.ingest import file_inventory, load_interactions
from ephys_rag.providers.factory import (
    SUPPORTED_PROVIDERS,
    ProviderSettings,
    provider_status,
)
from ephys_rag.tools import ToolRegistry


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
        provider=args.provider,
        use_llm=args.llm,
    )
    print(
        f"provider={result['provider']} model={result['model']} "
        f"elapsed_ms={result['elapsed_ms']:.3f}"
    )
    print(result["answer"])
    print("\n--- tool traces ---")
    for trace in result["traces"]:
        print(f"{trace.tool}: {trace.n_hits} hits from {trace.source_file}")
    print("\n--- retrieved ---")
    for hit in result["hits"]:
        print(f"{hit.score:.3f}  [{hit.document.kind}] {hit.document.title}")


def _cmd_providers(_: argparse.Namespace) -> None:
    print(json.dumps(provider_status(ProviderSettings.from_env()), indent=2))


def _cmd_evaluate(args: argparse.Namespace) -> None:
    engine = RAGEngine.from_disk()
    questions = load_question_manifest(args.manifest)
    providers = args.provider or ["medgemma"]
    summary = run_evaluation(
        engine,
        questions,
        providers,
        top_k=args.top_k,
        trace_limit=args.trace_limit,
        include_trace_catalog=args.compact_traces,
    )
    json_path, markdown_path = write_evaluation_reports(summary, args.output_dir)
    print(f"json_report={json_path}")
    print(f"markdown_report={markdown_path}")
    if not summary.success:
        raise SystemExit(1)


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

    ask = sub.add_parser("ask", help="Tools + TF-IDF retrieve + grounded answer")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=10)
    ask.add_argument(
        "--provider",
        choices=list(SUPPORTED_PROVIDERS),
        default=None,
        help="Answer provider. Default comes from LLM_PROVIDER or auto selection.",
    )
    ask.add_argument("--llm", action=argparse.BooleanOptionalAction, default=None)
    ask.set_defaults(func=_cmd_ask)

    providers = sub.add_parser(
        "providers",
        help="Show provider readiness without printing configuration values",
    )
    providers.set_defaults(func=_cmd_providers)

    evaluate = sub.add_parser(
        "evaluate",
        help="Run versioned biological questions through one or more providers",
    )
    evaluate.add_argument(
        "--provider",
        action="append",
        choices=[name for name in SUPPORTED_PROVIDERS if name != "auto"],
        help="Repeat to compare multiple providers. Default: medgemma.",
    )
    evaluate.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    evaluate.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    evaluate.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of retrieval hits to include; use 0 for tool-trace-only prompts.",
    )
    evaluate.add_argument(
        "--trace-limit",
        type=int,
        default=12,
        help="Maximum detailed rows included from each tool trace.",
    )
    evaluate.add_argument(
        "--compact-traces",
        action="store_true",
        help="Add compact value catalogs so small prompts retain non-top-row evidence.",
    )
    evaluate.set_defaults(func=_cmd_evaluate)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
