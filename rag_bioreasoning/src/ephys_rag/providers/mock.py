from __future__ import annotations

from ephys_rag.providers.base import GenerationResult


class MockTraceProvider:
    """Deterministic free provider that echoes the bounded trace facts."""

    name = "mock"
    model = "trace-rules-v1"

    def generate(self, question: str, context: str) -> GenerationResult:
        lines = [line.strip() for line in context.splitlines()]
        facts = [
            line
            for line in lines
            if "Tool:" in line
            or line.startswith(("Source:", "- ", "pair_values=", "gene_values="))
        ]
        if facts:
            text = "Trace-grounded mock answer:\n" + "\n".join(facts)
        else:
            text = "Trace-grounded mock answer: the supplied lookup was empty."
        return GenerationResult(text=text, provider=self.name, model=self.model)
