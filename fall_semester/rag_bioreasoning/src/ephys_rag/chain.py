from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ephys_rag.config import DEFAULT_TOP_K
from ephys_rag.documents import build_corpus
from ephys_rag.ingest import load_interactions
from ephys_rag.llm import (
    extractive_answer,
    generate_answer,
    generate_answers_dual,
    llm_available,
)
from ephys_rag.retrieve import HybridRetriever, RetrievalHit, format_context
from ephys_rag.schema import Interaction, RAGDocument, ToolResult
from ephys_rag.tools import ToolRegistry, dispatch_tools


def format_traces(traces: list[ToolResult]) -> str:
    if not traces:
        return ""
    blocks = ["=== tool traces ==="]
    for idx, trace in enumerate(traces, start=1):
        blocks.append(f"[{idx}] {trace.as_text()}")
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
        use_llm: Optional[bool] = None,
        provider: Optional[str] = None,
        dual: bool = False,
    ) -> dict:
        traces = dispatch_tools(question, self.tools)
        hits = self.retrieve(question, top_k=top_k)
        context = "\n\n".join(part for part in (format_traces(traces), format_context(hits)) if part)
        should_generate = llm_available() if use_llm is None else use_llm

        answers = {}
        answer = extractive_answer(question, context)
        if should_generate:
            if dual:
                answers = generate_answers_dual(question, context)
                med = answers.get("medgemma")
                gem = answers.get("gemini")
                parts = []
                if med is not None:
                    parts.append(
                        f"=== MedGemma ===\n{med.text if not med.error else f'[error] {med.error}'}"
                    )
                if gem is not None:
                    parts.append(
                        f"=== Gemini ===\n{gem.text if not gem.error else f'[error] {gem.error}'}"
                    )
                answer = "\n\n".join(parts)
            else:
                answer = generate_answer(question, context, provider=provider)

        return {
            "question": question,
            "answer": answer,
            "answers": answers,
            "hits": hits,
            "traces": traces,
            "context": context,
            "used_llm": should_generate,
            "provider": provider,
            "dual": dual,
        }
