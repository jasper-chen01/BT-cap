from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ephys_rag.config import DEFAULT_TOP_K
from ephys_rag.documents import build_corpus
from ephys_rag.ingest import load_interactions
from ephys_rag.llm import generate_answer
from ephys_rag.providers.base import LLMProvider
from ephys_rag.retrieve import HybridRetriever, RetrievalHit, format_context
from ephys_rag.schema import Interaction, RAGDocument, ToolResult
from ephys_rag.tools import ToolRegistry, dispatch_tools


_TRACE_CATALOG_KEYS = (
    "pair",
    "pathway",
    "gene",
    "direction",
    "IDH_status",
    "celltype_id",
    "group",
    "n_cells",
    "pval",
)


def _format_trace_catalog(trace: ToolResult, *, value_limit: int = 12) -> str:
    lines: list[str] = []
    for key in _TRACE_CATALOG_KEYS:
        values = list(
            dict.fromkeys(str(row[key]) for row in trace.rows if key in row)
        )
        if not values:
            continue
        rendered = ", ".join(values[:value_limit])
        suffix = " [more omitted]" if len(values) > value_limit else ""
        lines.append(f"{key}_values={rendered}{suffix}")
    return "\n".join(lines)


def format_traces(
    traces: list[ToolResult],
    *,
    limit: int = 12,
    include_catalog: bool = False,
) -> str:
    if not traces:
        return ""
    blocks = ["=== tool traces ==="]
    for idx, trace in enumerate(traces, start=1):
        block = f"[{idx}] {trace.as_text(limit=limit)}"
        if include_catalog:
            catalog = _format_trace_catalog(trace)
            if catalog:
                block = f"{block}\n{catalog}"
        blocks.append(block)
    return "\n\n".join(blocks)


@dataclass
class RAGEngine:
    interactions: list[Interaction]
    documents: list[RAGDocument]
    retriever: HybridRetriever
    tools: ToolRegistry

    @classmethod
    def from_disk(cls) -> "RAGEngine":
        interactions = load_interactions()
        documents = build_corpus(interactions)
        tools = ToolRegistry(interactions=interactions)
        return cls(
            interactions=interactions,
            documents=documents,
            retriever=HybridRetriever(documents),
            tools=tools,
        )

    def retrieve(self, question: str, *, top_k: int = DEFAULT_TOP_K) -> list[RetrievalHit]:
        return self.retriever.search(question, top_k=top_k)

    def ask(
        self,
        question: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        provider: str | LLMProvider | None = None,
        use_llm: Optional[bool] = None,
        trace_limit: int = 12,
        include_trace_catalog: bool = False,
    ) -> dict:
        traces = dispatch_tools(question, self.tools)
        hits = self.retrieve(question, top_k=top_k)
        context = "\n\n".join(
            part
            for part in (
                format_traces(
                    traces,
                    limit=trace_limit,
                    include_catalog=include_trace_catalog,
                ),
                format_context(hits),
            )
            if part
        )
        selected_provider: str | LLMProvider | None = provider
        if use_llm is False:
            selected_provider = "none"
        elif use_llm is True and selected_provider is None:
            selected_provider = "auto"
        generation = generate_answer(
            question,
            context,
            provider=selected_provider,
        )
        return {
            "question": question,
            "answer": generation.text,
            "hits": hits,
            "traces": traces,
            "provider": generation.provider,
            "model": generation.model,
            "elapsed_ms": generation.elapsed_ms,
            "provider_metadata": generation.metadata,
            "used_llm": generation.provider != "none",
            "context": context,
        }
