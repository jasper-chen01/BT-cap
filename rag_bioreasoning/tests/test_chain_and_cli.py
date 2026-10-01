from ephys_rag.chain import RAGEngine, format_traces
from ephys_rag.cli import build_parser
from ephys_rag.providers.base import GenerationResult
from ephys_rag.retrieve import HybridRetriever
from ephys_rag.schema import Interaction, RAGDocument, ToolResult
from ephys_rag.tools import ToolRegistry


class RecordingProvider:
    """Local provider double for observing the real RAG engine boundary."""

    name = "recording"
    model = "fake-model"

    def __init__(self):
        self.question = ""
        self.context = ""

    def generate(self, question, context):
        self.question = question
        self.context = context
        return GenerationResult("answer", self.name, self.model, 12.5)


def make_engine():
    interaction = Interaction(
        source_label="OPC_GABA_like_tumor / Ephys_2",
        target_label="OPC_GABA_like_tumor / Ephys_2",
        ligand="NRXN1",
        receptor="NLGN1",
        interaction_name="NRXN1_NLGN1",
        pathway="NRXN",
        probability=0.0202,
        pval=0.0,
        source_cell="OPC_GABA_like_tumor",
        source_ephys="Ephys_2",
        target_cell="OPC_GABA_like_tumor",
        target_ephys="Ephys_2",
        source_compartment="tumor",
        target_compartment="tumor",
        source_n_cells=14114,
        target_n_cells=14114,
        themes=["neuron_tumor"],
        same_ephys=True,
    )
    documents = [
        RAGDocument(
            doc_id="nrxn",
            title="NRXN1_NLGN1 tumor interaction",
            text="NRXN1 binds NLGN1 in Ephys_2 tumor cells.",
            kind="interaction",
        )
    ]
    return RAGEngine(
        interactions=[interaction],
        documents=documents,
        retriever=HybridRetriever(documents),
        tools=ToolRegistry(
            interactions=[interaction],
            degs_idh=[],
            degs_pooled=[],
        ),
    )


def test_engine_passes_tool_traces_and_hits_to_provider():
    """Losing either tools or retrieved evidence before generation must fail."""
    engine = make_engine()
    provider = RecordingProvider()

    result = engine.ask("What does NRXN1 bind?", provider=provider)

    assert provider.question == "What does NRXN1 bind?"
    assert "=== tool traces ===" in provider.context
    assert "NRXN1_NLGN1" in provider.context
    assert "NRXN1 binds NLGN1" in provider.context
    assert result["answer"] == "answer"
    assert result["provider"] == "recording"
    assert result["model"] == "fake-model"
    assert result["elapsed_ms"] == 12.5
    assert result["used_llm"] is True
    assert result["context"] == provider.context


def test_explicit_none_preserves_extractive_mode():
    """Offline mode must not be mislabeled as an LLM answer."""
    result = make_engine().ask("What does NRXN bind?", provider="none")

    assert result["provider"] == "none"
    assert result["model"] == "extractive"
    assert result["used_llm"] is False
    assert "NRXN1_NLGN1" in result["answer"]


def test_legacy_no_llm_flag_forces_extractive_mode():
    """Existing callers that disable generation must remain offline."""
    result = make_engine().ask("What does NRXN bind?", use_llm=False)

    assert result["provider"] == "none"
    assert result["used_llm"] is False


def test_ask_parser_accepts_provider():
    """The CLI must let users prove which model produced an answer."""
    args = build_parser().parse_args(
        ["ask", "--provider", "medgemma", "What does NRXN bind?"]
    )

    assert args.provider == "medgemma"


def test_ask_parser_accepts_free_local_providers():
    """The Week 3 CLI must expose both the real local and mock providers."""
    parser = build_parser()

    assert parser.parse_args(["ask", "--provider", "ollama", "question"]).provider == "ollama"
    assert parser.parse_args(["ask", "--provider", "mock", "question"]).provider == "mock"


def test_providers_command_exists():
    """Users need a non-secret readiness check before cloud calls."""
    args = build_parser().parse_args(["providers"])

    assert args.command == "providers"


def test_compact_trace_catalog_preserves_pairs_beyond_the_detailed_row_limit():
    """A small prompt must not hide a relevant receptor found below the first row."""
    trace = ToolResult(
        tool="cellchat_lookup",
        source_file="all_significant_interactions.csv",
        filters={"gene": "SPP1"},
        n_hits=2,
        rows=[
            {"pair": "SPP1_CD44", "prob": 0.2},
            {"pair": "SPP1_ITGAV_ITGB1", "prob": 0.1},
        ],
    )

    rendered = format_traces([trace], limit=1, include_catalog=True)

    assert "pair_values=SPP1_CD44, SPP1_ITGAV_ITGB1" in rendered
    assert rendered.count("prob=") == 1


def test_evaluate_parser_accepts_compact_context_controls():
    """The paid MedGemma run must be able to fit its verified 1,024-token input."""
    args = build_parser().parse_args(
        [
            "evaluate",
            "--provider",
            "medgemma",
            "--top-k",
            "0",
            "--trace-limit",
            "2",
            "--compact-traces",
        ]
    )

    assert args.top_k == 0
    assert args.trace_limit == 2
    assert args.compact_traces is True


def test_week3_cli_exposes_five_run_and_replay_table_commands():
    """The assignment must be runnable and replayable from stable CLI commands."""
    parser = build_parser()

    run_args = parser.parse_args(["week3-run"])
    replay_args = parser.parse_args(["week3-replay", "--table"])

    assert run_args.provider == "ollama"
    assert run_args.repeats == 5
    assert replay_args.table is True


def test_rank_genes_cli_exposes_reproducible_defaults():
    """Week 3 ranking must be reproducible from a stable command."""
    args = build_parser().parse_args(["rank-genes"])

    assert args.n_splits == 5
    assert args.top_k == 25
    assert args.seeds == "11,23,37,53,71"
